#!/usr/bin/env python3
# MIT License
#
# Paired per-patient comparison of two runs from their metrics3d.py outputs.
# For each organ and metric: mean difference B - A, a bootstrap 95% CI over
# patients, and how many patients got better / worse. With 8 validation
# patients a mean alone hides whether one patient drives it; this does not.
#
#   python scripts/paired_stats.py --a results/X/eval/metrics --b results/Y/eval/metrics \
#       --label_a baseline --label_b 3D
import argparse
from pathlib import Path

import numpy as np

ORGANS = ["esophagus", "heart", "trachea", "aorta"]
HIGHER_IS_BETTER = {"dice": True, "iou": True, "nsd": True, "hd95": False, "assd": False}
UNITS = {"dice": "", "iou": "", "nsd": "", "hd95": " mm", "assd": " mm"}


def paired(a: Path, b: Path, metric: str, n_boot: int, seed: int) -> list[dict]:
    na, nb = np.load(a / f"{metric}.npz"), np.load(b / f"{metric}.npz")
    ids = sorted(set(na.files) & set(nb.files))
    assert ids, f"no common patients in {a} and {b}"
    rng = np.random.default_rng(seed)
    rows = []
    for k, organ in enumerate(ORGANS, start=1):
        d = np.array([nb[i][k] - na[i][k] for i in ids], dtype=float)
        d = d[~np.isnan(d)]
        sign = 1 if HIGHER_IS_BETTER[metric] else -1
        tol = 1e-3 if HIGHER_IS_BETTER[metric] else 0.05  # "unchanged" below this
        boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"organ": organ, "n": len(d), "mean": d.mean(), "lo": lo, "hi": hi,
                     "better": int((sign * d > tol).sum()), "worse": int((sign * d < -tol).sum())})
    return rows


def verdict(r: dict, metric: str) -> str:
    """How well the per-patient differences support "B is better than A"."""
    higher = HIGHER_IS_BETTER[metric]
    changed = r["better"] + r["worse"]
    good_side = r["lo"] > 0 if higher else r["hi"] < 0   # whole CI on the improving side
    bad_side = r["hi"] < 0 if higher else r["lo"] > 0     # whole CI on the worsening side
    if changed == 0:
        return "no change"
    if changed <= 2:
        # A bootstrap over 8 patients where 1-2 moved says nothing general
        direction = "better" if r["better"] > r["worse"] else "worse" if r["worse"] > r["better"] else "mixed"
        return f"{changed} patient(s) only ({direction})"
    if good_side:
        return "supported" if r["worse"] == 0 else "supported, not uniform"
    if bad_side:
        return "worse (supported)"
    improving = r["mean"] > 0 if higher else r["mean"] < 0
    return "inconclusive (CI crosses 0)" if improving else "no improvement"


def main(args: argparse.Namespace) -> None:
    print(f"Paired: **{args.label_b}** minus **{args.label_a}**, per patient; "
          f"95% bootstrap CI ({args.n_boot} resamples of patients).\n")
    print("| Metric | Organ | Mean Δ | 95% CI | Patients better / worse | Verdict |")
    print("|---|---|---|---|---|---|")
    for metric in args.metrics:
        u, nd = UNITS[metric], (1 if UNITS[metric] else 3)
        for r in paired(args.a, args.b, metric, args.n_boot, args.seed):
            print(f"| {metric.upper()}{' (mm)' if u else ''} | {r['organ']} | {r['mean']:+.{nd}f} | "
                  f"[{r['lo']:+.{nd}f}, {r['hi']:+.{nd}f}] | {r['better']} / {r['worse']} of {r['n']} | "
                  f"{verdict(r, metric)} |")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Paired per-patient comparison of two metrics3d.py outputs")
    p.add_argument("--a", type=Path, required=True, help="Reference run metrics folder")
    p.add_argument("--b", type=Path, required=True, help="Compared run metrics folder")
    p.add_argument("--label_a", default="A")
    p.add_argument("--label_b", default="B")
    p.add_argument("--metrics", nargs="+", default=["dice", "hd95", "assd"])
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
