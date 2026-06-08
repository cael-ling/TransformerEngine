#!/bin/bash
# ============================================================================
# One-shot image builder: bake the per-token TE (editable registration) +
# container hygiene + tokenizer deps into a NEW .sqsh, so downstream sbatch
# jobs run with SKIP_BUILD=1 and never recompile TE.
#
# What gets baked into the new image:
#   - flash-attn removed (TE c10 ABI mismatch -> skip FA backend)
#   - huggingface_hub pinned <1.0 (transformers requirement)
#   - tokenizer deps for the Moonlight-16B custom tokenizer (tiktoken, blobfile)
#   - the editable TE *registration* (.pth / dist-info) pointing at the mounted
#     workspace, plus a warm workspace build/ (compiled .so).
#
# Why editable (not a frozen `pip install .`): you keep iterating on TE. With the
# editable registration baked, Python edits (e.g. recipe knobs) take effect with
# ZERO rebake, and only a C++ change needs a one-off `pip install -e .` in an
# interactive container (the .so lands in the persistent workspace build/).
#
# Result: future runs use the new image with SKIP_BUILD=1 and start instantly.
#
# Usage:
#     sbatch sbatch_build_te_image.sh
#     sbatch --export=ALL,OUT_IMAGE=/lustre/.../image/my_te.sqsh sbatch_build_te_image.sh
#
# Prereq: the workspace must be mounted at the SAME path at run time as here
#         (/home/caell/github/caell/TransformerEngine), because the baked
#         editable registration hardcodes that path. Our --container-mounts
#         guarantees this.
# ============================================================================
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -q short
#SBATCH -A coreai_devtech_all
#SBATCH --gres=gpu:4
#SBATCH --time=1:00:00
#SBATCH -J build-te-pertoken-image
# NOTE: #SBATCH --output/--error are resolved by Slurm on the HOST filesystem
# (the batch body runs on the host, outside the container), so these MUST be the
# real lustre path, NOT the in-container /home/caell mount target.
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/caell/github/caell/TransformerEngine/_scratch_mcore/slurm_logs/build-te-image-%j.out
#SBATCH --error=/lustre/fsw/portfolios/coreai/users/caell/github/caell/TransformerEngine/_scratch_mcore/slurm_logs/build-te-image-%j.err

set -euo pipefail

# --- Host-side config -------------------------------------------------------
BASE_IMAGE="${BASE_IMAGE:-/lustre/fsw/portfolios/coreai/users/caell/image/arch100a_pt_2508_te_main_9ad2e7b.sqsh}"
OUT_IMAGE="${OUT_IMAGE:-/lustre/fsw/portfolios/coreai/users/caell/image/arch100a_pt_2508_te_pertoken.sqsh}"
HOST_MOUNT="/lustre/fsw/portfolios/coreai/users/caell:/home/caell"

mkdir -p /lustre/fsw/portfolios/coreai/users/caell/github/caell/TransformerEngine/_scratch_mcore/slurm_logs
mkdir -p "$(dirname "$OUT_IMAGE")"

if [[ -e "$OUT_IMAGE" ]]; then
    echo "[build] WARNING: $OUT_IMAGE exists; pyxis --container-save will overwrite it."
fi

echo "[build] job=$SLURM_JOB_ID node=$(hostname)"
echo "[build] base  = $BASE_IMAGE"
echo "[build] out   = $OUT_IMAGE"

# --- Build inside the base container, then save the rootfs to OUT_IMAGE ------
# --container-save writes the container ROOTFS (writes from this step) to a new
# squashfs. Bind mounts (the workspace) are NOT included -- only the deltas we
# make to the image's own filesystem (pip installs, editable registration).
srun --container-image="$BASE_IMAGE" \
     --container-writable \
     --container-save="$OUT_IMAGE" \
     --container-mounts="$HOST_MOUNT" \
     --container-remap-root \
     --container-workdir="/home/caell" \
     --export=ALL \
     bash <<'EOF'
set -euo pipefail
TE_DIR="/home/caell/github/caell/TransformerEngine"

echo "==> [1/5] flash-attn removal (c10 ABI mismatch)"
pip uninstall -y flash-attn flash_attn flash_attn_2_cuda >/dev/null 2>&1 || true

echo "==> [2/5] pin huggingface_hub <1.0 (+ tokenizer deps)"
pip install -q "huggingface_hub>=0.34,<1.0"
# Moonlight-16B custom tokenizer (trust_remote_code) needs tiktoken + blobfile.
pip install -q tiktoken blobfile sentencepiece

echo "==> [3/5] remove any pre-baked TE (avoid a shadowing main install)"
pip uninstall -y transformer_engine transformer_engine_torch transformer-engine >/dev/null 2>&1 || true
# Image-baked installs are not pip-tracked; rm them so they cannot shadow the
# workspace editable at runtime (see te-nvfp4-build-overrides double-install).
for d in /usr/local/lib/python3.12/dist-packages /root/.local/lib/python3.12/site-packages; do
    rm -rf "$d"/transformer_engine "$d"/transformer_engine_torch* \
           "$d"/transformer_engine-*.dist-info "$d"/transformer_engine_torch-*.dist-info \
           "$d"/__editable__.transformer_engine* "$d"/__editable___transformer_engine* 2>/dev/null || true
done

echo "==> [4/5] editable install TE from workspace (builds into persistent build/)"
cd "$TE_DIR"
NVTE_CUDA_ARCHS=100a NVTE_BUILD_THREADS_PER_JOB=8 NVTE_FRAMEWORK=pytorch \
    pip install -e . --no-build-isolation 2>&1 | tee "build_image_${SLURM_JOB_ID}.log"

echo "==> [5/5] sanity: import TE + per-token symbol present"
python - <<'PY'
import transformer_engine            # dlopen libtransformer_engine.so first
import transformer_engine_torch as tex
assert hasattr(tex, "nvfp4_per_token_quantize"), "per-token TE symbol missing!"
import transformer_engine as te
print("[build] TE", getattr(te, "__version__", "?"), "per-token OK")
PY

# Prove the editable registration that will be baked actually resolves to the
# workspace (so it works in a fresh container with the workspace mounted).
echo "[build] editable finders in site-packages:"
ls -1 /usr/local/lib/python3.12/dist-packages/__editable__*transformer_engine* 2>/dev/null \
    || ls -1 /usr/local/lib/python3.12/dist-packages/transformer_engine*.dist-info 2>/dev/null \
    || echo "  (none found -- check pip install output above)"

echo "==> in-container build complete; pyxis will now save the rootfs."
EOF

echo "[build] saved new image -> $OUT_IMAGE"
echo "[build] use it with: sbatch --export=ALL,IMAGE=$OUT_IMAGE,SKIP_BUILD=1 sbatch_moe_nvfp4_singlegpu.sh <mode>"
