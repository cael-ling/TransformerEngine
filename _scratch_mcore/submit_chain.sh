#!/bin/bash
# ============================================================================
# Submit a CHAIN of dependent sbatch jobs that auto-resume one after another.
#
# Each link runs under the 2h wall, then (via EXIT_DURATION_MIN in the run
# script) saves a checkpoint and exits cleanly. The next link is submitted with
# a Slurm dependency so it only starts after the previous one finishes, and
# because the checkpoint path is a STABLE per-variant dir (work/<variant>/
# checkpoints, no timestamp), it resumes from that checkpoint automatically.
#
# IMPORTANT: TRAIN_ITERS is the TOTAL target (global iteration count), NOT
# per-link. mcore counts iterations globally, resumes from the saved iteration,
# and stops at --train-iters. So pass the same TRAIN_ITERS to every link (just
# set it once in --export below); the chain simply accumulates wall-clock until
# that total is reached.
#
# Usage (run on the LOGIN node, not inside a container):
#   CHAIN=3 bash _scratch_mcore/submit_chain.sh \
#       --export=ALL,IMAGE=/lustre/.../arch100a_pt_2508_te_pertoken.sqsh,SKIP_BUILD=1,TRAIN_ITERS=60000 \
#       _scratch_mcore/sbatch_moe_nvfp4_singlegpu.sh pertoken
#
# Everything after the (optional) CHAIN/DEP_TYPE env vars is forwarded verbatim
# to `sbatch`. So any spec the wrapper accepts works, e.g. a single variant
# (pertoken) or a concurrent list ("bf16,pertensor+rht+sr,pertoken").
#
# Knobs:
#   CHAIN     number of links to submit (default 2)
#   DEP_TYPE  dependency type (default afterany -> next runs even if prev
#             TIMEOUTs/fails; use afterok to stop the chain on a hard failure)
# ============================================================================
set -euo pipefail

CHAIN="${CHAIN:-2}"
DEP_TYPE="${DEP_TYPE:-afterany}"

if [[ $# -lt 1 ]]; then
    echo "usage: CHAIN=<n> bash submit_chain.sh <sbatch args...> <script> [spec]" >&2
    echo "   e.g. CHAIN=3 bash submit_chain.sh --export=ALL,IMAGE=...,SKIP_BUILD=1 \\" >&2
    echo "            _scratch_mcore/sbatch_moe_nvfp4_singlegpu.sh pertoken" >&2
    exit 1
fi

echo "[chain] submitting $CHAIN links (dependency=$DEP_TYPE) for: $*"

prev=""
for (( i=1; i<=CHAIN; i++ )); do
    if [[ -z "$prev" ]]; then
        out="$(sbatch "$@")"
    else
        out="$(sbatch "--dependency=${DEP_TYPE}:${prev}" "$@")"
    fi
    # sbatch prints "Submitted batch job <ID>"
    jid="$(awk '{print $NF}' <<<"$out")"
    if ! [[ "$jid" =~ ^[0-9]+$ ]]; then
        echo "[chain] ERROR: could not parse job id from: $out" >&2
        exit 1
    fi
    if [[ -z "$prev" ]]; then
        echo "[chain] link $i/$CHAIN -> job $jid (no dependency, runs first)"
    else
        echo "[chain] link $i/$CHAIN -> job $jid (starts ${DEP_TYPE} job $prev)"
    fi
    prev="$jid"
done

echo "[chain] done. Inspect with: squeue -u \$USER ; scontrol show job <id> | grep -i depend"
