#!/usr/bin/env python3
"""Empirical probe: confirm the per-token recipe actually emits a 2D-quantized
WEIGHT (Route A) when NVTE_NVFP4_PER_TOKEN_WEIGHT_2D=1.

Run on an SM100 (Blackwell) GPU node inside the per-token/w2d container:
    python _scratch_mcore/probe_pertoken_weight_2d.py
"""
import os

os.environ.setdefault("NVTE_NVFP4_PER_TOKEN", "1")
os.environ.setdefault("NVTE_NVFP4_PER_TOKEN_WEIGHT_2D", "1")

import torch
import transformer_engine.pytorch as te  # noqa: F401  (dlopen libtransformer_engine first)
from transformer_engine.pytorch import NVFP4Quantizer
import transformer_engine_torch as tex
from transformer_engine.common.recipe import NVFP4BlockScaling


def main() -> int:
    r = NVFP4BlockScaling()
    print(f"recipe: per_token={r.nvfp4_per_token()}  per_token_weight_2d={r.per_token_weight_2d}")

    if not torch.cuda.is_available():
        print("SKIP: no CUDA device (run on a GPU node).")
        return 0

    # The weight quantizer the recipe state builds for the 'weight' slot.
    q = NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=True,
        per_token=True,
        per_token_weight_2d=True,
    )
    w = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda")
    dst = q.make_empty(w.shape, dtype=w.dtype, device=w.device, requires_grad=False)
    t = q.update_quantized(w, dst)

    amax_row = t._amax_rowwise.reshape(-1)
    amax_col = t._amax_columnwise.reshape(-1)
    row_const = bool(torch.all(amax_row == amax_row[0]))
    col_const = bool(torch.all(amax_col == amax_col[0]))

    print(f"_per_token (drives GEMM dispatch) = {t._per_token}")
    print(f"rowwise outer-amax constant (2D scalar broadcast) = {row_const}  (numel={amax_row.numel()})")
    print(f"colwise outer-amax constant (2D scalar broadcast) = {col_const}  (numel={amax_col.numel()})")

    # Control: a genuine per-token 1D weight has a NON-constant per-row amax.
    q1d = NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=True,
        per_token=True,
        per_token_weight_2d=False,
    )
    dst1 = q1d.make_empty(w.shape, dtype=w.dtype, device=w.device, requires_grad=False)
    t1 = q1d.update_quantized(w, dst1)
    amax1 = t1._amax_rowwise.reshape(-1)
    one_d_const = bool(torch.all(amax1 == amax1[0]))
    print(f"[control 1D] rowwise outer-amax constant = {one_d_const}  (expected False)")

    ok = (
        r.per_token_weight_2d
        and t._per_token
        and row_const
        and col_const
        and not one_d_const
    )
    print("=" * 60)
    print("RESULT:", "PASS — weight is 2D-quantized, dressed as per-token." if ok
          else "FAIL — weight did NOT take the 2D path.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
