#!/usr/bin/env python3
"""Empirical probe: confirm `pertoken+rht+sr+2d` constructs WITHOUT hitting the
per_token_weight_2d <-> rht/sr mutex, by driving the REAL recipe->quantizer
path (the same RecipeState.make_quantizers() training uses).

Expectation: no exception, and per slot:
    input  : with_rht=True , sr=False, weight_2d=False
    weight : with_rht=False, sr=False, weight_2d=True
    output : with_rht=False, sr=False, weight_2d=False
    grad_* : with_rht=True , sr=True , weight_2d=False

Run on an SM100 GPU node inside the per-token/w2d container:
    python _scratch_mcore/probe_pertoken_rht_sr_2d.py
"""
import os

# The full pertoken+rht+sr+2d variant.
os.environ.setdefault("NVTE_NVFP4_PER_TOKEN", "1")
os.environ.setdefault("NVTE_NVFP4_PER_TOKEN_RHT", "1")
os.environ.setdefault("NVTE_NVFP4_PER_TOKEN_SR", "1")
os.environ.setdefault("NVTE_NVFP4_PER_TOKEN_WEIGHT_2D", "1")

import torch
import transformer_engine.pytorch as te  # noqa: F401  (dlopen libtransformer_engine first)
from transformer_engine.common.recipe import NVFP4BlockScaling
from transformer_engine.pytorch.quantization import RecipeState


def _row(name, q):
    return (
        f"  {name:<12} per_token={getattr(q, 'per_token', '?')!s:<5} "
        f"weight_2d={getattr(q, 'per_token_weight_2d', '?')!s:<5} "
        f"with_rht={getattr(q, 'with_rht', '?')!s:<5} "
        f"sr={getattr(q, 'stochastic_rounding', '?')!s:<5}"
    )


def main() -> int:
    r = NVFP4BlockScaling()
    print(
        f"recipe: per_token={r.nvfp4_per_token()} per_token_rht={r.per_token_rht} "
        f"per_token_sr={r.per_token_sr} per_token_weight_2d={r.per_token_weight_2d}"
    )
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device (run on a GPU node).")
        return 0

    # This is exactly what a Linear/GroupedLinear does to build its quantizers.
    # If the mutex would fire, make_quantizers() raises here.
    try:
        fwd = RecipeState.create(r, mode="forward", num_quantizers=3).make_quantizers()
        bwd = RecipeState.create(r, mode="backward", num_quantizers=2).make_quantizers()
    except Exception as e:  # noqa: BLE001
        print("FAIL: quantizer construction raised:")
        print(f"  {type(e).__name__}: {e}")
        return 1

    print("forward quantizers (positional: input, weight, output):")
    for nm, q in zip(("input", "weight", "output"), fwd):
        print(_row(nm, q))
    print("backward quantizers (positional: grad_output, grad_input):")
    for nm, q in zip(("grad_output", "grad_input"), bwd):
        print(_row(nm, q))

    w_q = fwd[1]
    in_q = fwd[0]
    g_q = bwd[0]
    ok = (
        w_q.per_token_weight_2d and not w_q.with_rht and not w_q.stochastic_rounding
        and in_q.with_rht and not in_q.per_token_weight_2d
        and g_q.stochastic_rounding and not g_q.per_token_weight_2d
    )
    print("=" * 64)
    print(
        "RESULT:",
        "PASS - pertoken+rht+sr+2d builds cleanly; weight=2D, act/grad=rht/sr."
        if ok else "FAIL - slot flags not as expected (see rows above).",
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
