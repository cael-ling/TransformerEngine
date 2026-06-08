#!/usr/bin/env python3
"""Summary table: every variant's relative loss gap vs a baseline (default bf16).

Auto-discovers all work/<variant>/ dirs that have a TB log, computes
    rel_gap = (loss[variant] - loss[bf16]) / loss[bf16]
on the steps the two share, and prints a markdown table + writes CSV.

Usage:
    python3 _scratch_mcore/loss_gap_table.py
    python3 _scratch_mcore/loss_gap_table.py --baseline bf16 --tag "lm loss" --csv summary.csv
"""
import argparse
import os
import statistics

from loss_gap import load_series  # same dir

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def short_label(variant):
    """work dir name -> compact label, e.g. nvfp4-pertoken-rht-sr -> pertoken+rht+sr."""
    s = variant
    if s.startswith("nvfp4-"):
        s = s[len("nvfp4-"):]
    return (
        s.replace("-rht", "+rht")
        .replace("-sr", "+sr")
        .replace("-1d", "+1d")
        .replace("-2d", "+2d")
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="bf16")
    ap.add_argument("--tag", default="lm loss")
    ap.add_argument("--work", default=os.path.join(SCRIPT_DIR, "work"))
    ap.add_argument("--csv", default=os.path.join(SCRIPT_DIR, "work", "loss_gap_summary.csv"))
    ap.add_argument("--max-step", type=int, default=None,
                    help="only compare steps <= this (fair common window across variants)")
    ap.add_argument("--min-steps", type=int, default=0,
                    help="skip variants with fewer than this many overlapping steps")
    args = ap.parse_args()

    base = load_series(args.baseline, args.work, args.tag)

    variants = sorted(
        d
        for d in os.listdir(args.work)
        if d != args.baseline and os.path.isdir(os.path.join(args.work, d, "tb"))
    )

    rows = []
    for v in variants:
        try:
            s = load_series(v, args.work, args.tag)
        except SystemExit:
            continue
        common = sorted(set(base) & set(s))
        if args.max_step is not None:
            common = [k for k in common if k <= args.max_step]
        if not common or len(common) < args.min_steps:
            continue
        gaps = [(s[k] - base[k]) / base[k] for k in common if base[k] != 0]
        agaps = [abs(g) for g in gaps]
        tail = gaps[-min(len(gaps), 50):]
        atail = [abs(g) for g in tail]
        rows.append(
            {
                "variant": v,
                "steps": len(common),
                "last_step": common[-1],
                "mean_signed": statistics.mean(gaps) * 100,   # bias/direction (can cancel)
                "mean_abs": statistics.mean(agaps) * 100,      # MAE magnitude (no cancel)
                "median_abs": statistics.median(agaps) * 100,
                "rms": (statistics.mean([g * g for g in gaps]) ** 0.5) * 100,
                "mean_abs_tail50": statistics.mean(atail) * 100,
                "last_abs": abs(gaps[-1]) * 100,
                "max_abs": max(agaps) * 100,
            }
        )

    rows.sort(key=lambda r: r["mean_abs"])  # sort by MAE (full-overlap |gap|): more samples, less noise than tail50

    hdr = ["variant", "steps", "last_step", "mean|gap|% (MAE)", "median|gap|%", "rms%",
           "mean|gap|(tail50)%", "last|gap|%", "max|gap|%", "mean_signed%"]
    print(f"baseline = {args.baseline}   tag = '{args.tag}'   (|.| = absolute, no sign cancellation)\n")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join("---" for _ in hdr) + "|")
    for r in rows:
        print(
            f"| {short_label(r['variant'])} | {r['steps']} | {r['last_step']} | "
            f"{r['mean_abs']:.3f} | {r['median_abs']:.3f} | {r['rms']:.3f} | "
            f"{r['mean_abs_tail50']:.3f} | {r['last_abs']:.3f} | {r['max_abs']:.3f} | "
            f"{r['mean_signed']:+.3f} |"
        )

    with open(args.csv, "w") as fh:
        fh.write(",".join(["variant", "overlap_steps", "last_step", "mean_abs_pct_MAE",
                            "median_abs_pct", "rms_pct", "mean_abs_tail50_pct", "last_abs_pct",
                            "max_abs_pct", "mean_signed_pct"]) + "\n")
        for r in rows:
            fh.write(
                f"{r['variant']},{r['steps']},{r['last_step']},{r['mean_abs']:.4f},"
                f"{r['median_abs']:.4f},{r['rms']:.4f},{r['mean_abs_tail50']:.4f},"
                f"{r['last_abs']:.4f},{r['max_abs']:.4f},{r['mean_signed']:.4f}\n"
            )
    print(f"\n[loss_gap_table] wrote {len(rows)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
