#!/usr/bin/env python3
"""Per-iteration relative loss gap between two run variants (from TB logs).

Reads the 'lm loss' scalar from each variant's TensorBoard dir
(work/<variant>/tb/, which spans all chain links continuously), aligns the two
series by global step, and reports

    rel_gap(step) = (loss[variant] - loss[baseline]) / loss[baseline]

Usage:
    python3 _scratch_mcore/loss_gap.py nvfp4-pertoken            # vs bf16 (default baseline)
    python3 _scratch_mcore/loss_gap.py nvfp4-pertoken bf16       # explicit baseline
    python3 _scratch_mcore/loss_gap.py nvfp4-pertoken --csv gap.csv
Options:
    --tag "lm loss"     scalar tag to compare (default "lm loss")
    --work <dir>        work root (default: alongside this script)
    --csv  <path>       also write a per-step CSV
"""
import argparse
import glob
import os
import statistics
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_series(variant, work_root, tag):
    """step -> value, merged across all event files (later writes win per step)."""
    tb_dir = os.path.join(work_root, variant, "tb")
    files = sorted(glob.glob(os.path.join(tb_dir, "events.*")), key=os.path.getmtime)
    if not files:
        sys.exit(f"[loss_gap] no TB event files under {tb_dir}")
    series = {}
    for f in files:
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        if tag not in ea.Tags().get("scalars", []):
            continue
        for ev in ea.Scalars(tag):
            series[ev.step] = ev.value  # later file (newer mtime) overrides on resume overlap
    if not series:
        sys.exit(f"[loss_gap] tag '{tag}' not found in {tb_dir}")
    return series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", help="run variant dir name, e.g. nvfp4-pertoken")
    ap.add_argument("baseline", nargs="?", default="bf16", help="baseline variant (default bf16)")
    ap.add_argument("--tag", default="lm loss")
    ap.add_argument("--work", default=os.path.join(SCRIPT_DIR, "work"))
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    a = load_series(args.baseline, args.work, args.tag)  # baseline
    b = load_series(args.variant, args.work, args.tag)  # variant under test
    common = sorted(set(a) & set(b))
    if not common:
        sys.exit("[loss_gap] no overlapping steps between the two variants")

    rows = []
    for s in common:
        la, lb = a[s], b[s]
        gap = (lb - la) / la if la != 0 else float("nan")
        rows.append((s, la, lb, gap))

    gaps = [r[3] for r in rows]
    print(f"baseline = {args.baseline:24s} steps {min(a)}..{max(a)} ({len(a)} pts)")
    print(f"variant  = {args.variant:24s} steps {min(b)}..{max(b)} ({len(b)} pts)")
    print(f"overlap  = {len(common)} steps  ({common[0]}..{common[-1]})\n")
    print(f"{'step':>7} {'bf16':>10} {'variant':>10} {'rel_gap%':>10}")
    # print a sampled view: first 3, last 3, and every ~10th in between
    show = set(common[:3] + common[-3:] + common[:: max(1, len(common) // 20)])
    for s, la, lb, g in rows:
        if s in show:
            print(f"{s:>7} {la:>10.4f} {lb:>10.4f} {g * 100:>9.2f}%")

    print("\n=== relative loss gap (variant vs baseline) summary ===")
    print(f"  mean   : {statistics.mean(gaps) * 100:+.3f}%")
    print(f"  median : {statistics.median(gaps) * 100:+.3f}%")
    print(f"  min    : {min(gaps) * 100:+.3f}%   (step {rows[gaps.index(min(gaps))][0]})")
    print(f"  max    : {max(gaps) * 100:+.3f}%   (step {rows[gaps.index(max(gaps))][0]})")
    print(f"  last   : {gaps[-1] * 100:+.3f}%   (step {common[-1]})")
    tail = gaps[-min(len(gaps), 50):]
    print(f"  mean(last {len(tail)}): {statistics.mean(tail) * 100:+.3f}%")

    if args.csv:
        with open(args.csv, "w") as fh:
            fh.write("step,baseline_loss,variant_loss,rel_gap\n")
            for s, la, lb, g in rows:
                fh.write(f"{s},{la:.6f},{lb:.6f},{g:.6f}\n")
        print(f"\n[loss_gap] wrote {len(rows)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
