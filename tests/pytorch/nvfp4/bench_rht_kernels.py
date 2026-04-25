#!/usr/bin/env python3
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Benchmark for split_quantize kernel paths (RHT and non-RHT).
#
# All splits are 128-aligned (matching real Fp8Padding behavior).
#
# ============== Non-RHT kernel paths ==============
#
#   noRHT both_dir:
#       nvte_group_amax + cudaMemcpyAsync(colwise_amax←rowwise_amax)
#     + group_quantize_transpose_nvfp4_kernel (RETURN_TRANSPOSE=true)
#
#   noRHT rowwise_only:
#       nvte_group_amax
#     + group_quantize_transpose_nvfp4_kernel (RETURN_TRANSPOSE=false)
#
#   noRHT colwise_only:
#       nvte_group_amax
#     + N x nvte_quantize_v2  (per-tensor fallback)
#
# ============== RHT kernel paths (aligned) ==============
#
#   RHT both_dir:
#       nvte_group_hadamard_transform_amax
#     + nvte_group_hadamard_transform_cast_fusion
#
#   RHT rowwise_only:
#       nvte_group_hadamard_transform_amax
#     + nvte_group_hadamard_transform_cast_fusion  (colwise no-op)
#
#   RHT colwise_only:
#       nvte_group_hadamard_transform_amax
#     + nvte_group_hadamard_transform_cast_fusion  (rowwise no-op)
#
# For per-kernel breakdown:
#   nsys profile -o rht_bench python bench_rht_kernels.py --iters 10
#   nsys stats rht_bench.nsys-rep

import argparse
import torch
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer

recipe_available, reason = te.is_nvfp4_available(return_reason=True)
if not recipe_available:
    raise RuntimeError(f"NVFP4 not available: {reason}")


def make_quantizers(num_chunks, rowwise, columnwise, with_rht):
    return [
        NVFP4Quantizer(
            fp4_dtype=tex.DType.kFloat4E2M1,
            rowwise=rowwise,
            columnwise=columnwise,
            with_amax_reduction=False,
            amax_reduction_group=None,
            with_rht=with_rht,
            with_post_rht_amax=with_rht,
            with_random_sign_mask=True,
        )
        for _ in range(num_chunks)
    ]


def bench_one(x, split_sections, quantizers, warmup, iters):
    """Return median latency in microseconds."""
    for _ in range(warmup):
        tex.split_quantize(x, split_sections, quantizers)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        tex.split_quantize(x, split_sections, quantizers)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # ms -> us

    times.sort()
    return times[len(times) // 2]


def run_benchmark(M, N, num_chunks, warmup, iters):
    device = "cuda"
    x = torch.randn((M, N), dtype=torch.bfloat16, device=device)

    avg = M // num_chunks
    assert avg % 128 == 0, (
        f"M/num_chunks={avg} is not 128-aligned. "
        f"Choose M={M} and num_chunks={num_chunks} accordingly."
    )
    splits = [avg] * num_chunks
    assert sum(splits) == M

    print(f"\n{'='*80}")
    print(f"  M={M:>6}, N={N:>6}, chunks={num_chunks}, splits={splits}")
    print(f"{'='*80}")

    modes = [
        ("both_dir", True, True),
        ("rowwise ", True, False),
        ("colwise ", False, True),
    ]

    results = {}

    header = f"  {'Path':<58} {'Median (us)':>12}"
    print(header)
    print(f"  {'-'*58} {'-'*12}")

    # Non-RHT paths
    print(f"  {'--- Non-RHT ---'}")
    for mode_name, rw, cw in modes:
        kernel_desc = {
            ("both_dir", False): "group_quant_transpose (fused row+col)",
            ("rowwise ", False): "group_quant (RETURN_TRANSPOSE=false)",
            ("colwise ", False): "per-tensor nvte_quantize_v2 x N",
        }[(mode_name, False)]

        q = make_quantizers(num_chunks, rowwise=rw, columnwise=cw, with_rht=False)
        t = bench_one(x, splits, q, warmup, iters)
        label = f"noRHT {mode_name} [{kernel_desc}]"
        print(f"  {label:<58} {t:>10.1f} us")
        results[f"norht_{mode_name.strip()}"] = t

    # RHT paths
    print(f"  {'--- RHT (aligned) ---'}")
    for mode_name, rw, cw in modes:
        kernel_desc = "hadamard_cast_fusion"
        q = make_quantizers(num_chunks, rowwise=rw, columnwise=cw, with_rht=True)
        t = bench_one(x, splits, q, warmup, iters)
        label = f"RHT   {mode_name} [{kernel_desc}]"
        print(f"  {label:<58} {t:>10.1f} us")
        results[f"rht_{mode_name.strip()}"] = t

    # Analysis
    print()
    print(f"  {'--- Analysis ---'}")

    # RHT vs non-RHT comparison
    for mode in ["both_dir", "rowwise", "colwise"]:
        t_rht = results[f"rht_{mode}"]
        t_norht = results[f"norht_{mode}"]
        ratio = t_rht / t_norht if t_norht > 0 else float('inf')
        print(f"  {mode:<10}  RHT={t_rht:>8.1f} us  noRHT={t_norht:>8.1f} us  ratio={ratio:.2f}x")

    # Colwise incremental cost (both - rowwise)
    col_inc_rht = results["rht_both_dir"] - results["rht_rowwise"]
    col_inc_norht = results["norht_both_dir"] - results["norht_rowwise"]
    print()
    print(f"  colwise incremental cost (both_dir - rowwise):")
    print(f"    RHT:   {col_inc_rht:>8.1f} us  (Hadamard + colwise quant fused)")
    print(f"    noRHT: {col_inc_norht:>8.1f} us  (transpose + colwise quant fused)")

    # Non-RHT: fused vs per-tensor colwise
    t_norht_col_fused = results["norht_both_dir"] - results["norht_rowwise"]
    t_norht_col_pertensor = results["norht_colwise"] - results["norht_rowwise"]
    print()
    print(f"  noRHT colwise: fused (in both_dir) ≈ {t_norht_col_fused:>8.1f} us")
    print(f"  noRHT colwise: per-tensor fallback  = {results['norht_colwise']:>8.1f} us (includes amax)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark split_quantize kernel paths")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--num-chunks", type=int, default=4)
    args = parser.parse_args()

    sizes = [
        (512, 1024),
        (1024, 1024),
        (8192, 1024),
        (8192, 8192),
        (16384, 8192),
        (16384, 16384),
    ]

    print("split_quantize Kernel Path Benchmark")
    print("=" * 80)
    print(f"All splits are 128-aligned (matching real Fp8Padding behavior).")
    print(f"For per-kernel breakdown: nsys profile -o rht_bench python {__file__}")

    for M, N in sizes:
        if M < args.num_chunks * 128:
            print(f"\nSkipping M={M}, N={N} (too small)")
            continue
        run_benchmark(M, N, args.num_chunks, args.warmup, args.iters)


if __name__ == "__main__":
    main()
