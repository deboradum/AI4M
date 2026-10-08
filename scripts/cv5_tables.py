#!/usr/bin/env python3
# MIT License
#
# Markdown tables for the 5-fold CV (scripts/run_cv5.sh), in the format of
# scripts/presentation_tables.py: per metric and organ, the mean over folds ± the
# std across the 5 fold means, raw and with the final post-processing; then a
# paired comparison of the first two models over all patients (bootstrap 95% CI).
# Best value per column in bold.
#
#   python scripts/cv5_tables.py --model "2.5D U-Net (E_F05)=E_F05_unet25d_adamw_cosine_150ep" \
#       --model "2D U-Net (E_F06)=E_F06_unet2d_adamw_cosine_150ep"
import argparse
from pathlib import Path

import numpy as np

ORGANS = ["esophagus", "heart", "trachea", "aorta"]
METRICS = [("dice", "Dice", "↑", 3), ("iou", "IoU", "↑", 3), ("hd95", "HD95 (mm)", "↓", 1),
           ("assd", "ASSD (mm)", "↓", 2), ("nsd", "NSD (2 mm)", "↑", 3)]
SUBS = {"raw": "eval/metrics", "pp": "eval/pp_final/metrics"}


def load(root: Path, base: str, sub: str, n_folds: int) -> list[dict[str, dict[str, np.ndarray]]]:
    """One dict per fold: metric -> patient -> (4 organs)"""
    folds = []
    for k in range(n_folds):
        folder = root / f"{base}_f{k}" / sub
        folds.append({m: {p: z[p][1:] for p in z.files} for m, *_ in METRICS
                      for z in [np.load(folder / f"{m}.npz")]})
    return folds


def fold_table(folds: list[dict]) -> dict[str, np.ndarray]:
    """metric -> (folds, 4 organs + mean)"""
    out = {}
    for m, *_ in METRICS:
        per_fold = np.stack([np.nanmean(np.stack(list(f[m].values())), 0) for f in folds])
        out[m] = np.concatenate([per_fold, per_fold.mean(1, keepdims=True)], 1)
    return out


def table(models: dict[str, dict[str, np.ndarray]], title: str) -> None:
    print(f"### {title}\n")
    print("| Metric | Model | " + " | ".join(ORGANS) + " | mean |")
    print("|---|---|" + "---|" * (len(ORGANS) + 1))
    for m, name, arrow, dec in METRICS:
        means = np.stack([t[m].mean(0) for t in models.values()])
        best = means.argmax(0) if arrow == "↑" else means.argmin(0)
        for i, (label, t) in enumerate(models.items()):
            cells = []
            for j, (mu, sd) in enumerate(zip(t[m].mean(0), t[m].std(0, ddof=1))):
                cell = f"{mu:.{dec}f} ± {sd:.{dec}f}"
                cells.append(f"**{cell}**" if len(models) > 1 and best[j] == i else cell)
            print(f"| {name} {arrow} | {label} | " + " | ".join(cells) + " |")
    print()


def paired(a: list[dict], b: list[dict], la: str, lb: str, n_boot: int, seed: int) -> None:
    pa = {p: v for f in a for p, v in f["dice"].items()}
    ids = sorted(pa)
    print(f"### Paired over {len(ids)} patients, post-processed: {lb} − {la}\n")
    print("| Metric | " + " | ".join(ORGANS) + f" | mean | 95% CI (mean) | {lb} better |")
    print("|---|" + "---|" * (len(ORGANS) + 3))
    rng = np.random.default_rng(seed)
    for m, name, arrow, dec in METRICS:
        xa = np.stack([{p: v for f in a for p, v in f[m].items()}[i] for i in ids])
        xb = np.stack([{p: v for f in b for p, v in f[m].items()}[i] for i in ids])
        d = xb - xa
        dm = np.nanmean(d, 1)
        boot = [dm[rng.integers(0, len(dm), len(dm))].mean() for _ in range(n_boot)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        better = (dm > 0).sum() if arrow == "↑" else (dm < 0).sum()
        print(f"| {name} {arrow} | " + " | ".join(f"{v:+.{dec}f}" for v in np.nanmean(d, 0))
              + f" | {dm.mean():+.{dec}f} | [{lo:+.{dec}f}, {hi:+.{dec}f}] | {better}/{len(ids)} |")
    print()


def main(args: argparse.Namespace) -> None:
    models = dict(spec.split("=", 1) for spec in args.model)
    data = {tag: {label: load(args.root, base, sub, args.n_folds) for label, base in models.items()}
            for tag, sub in SUBS.items()}
    n = sum(len(f["dice"]) for f in next(iter(data["raw"].values())))
    for tag, desc in [("raw", "raw"), ("pp", "with the final post-processing (--lcc 2 3 --min_ml 4:1)")]:
        table({label: fold_table(folds) for label, folds in data[tag].items()},
              f"5-fold CV, {n} patients, final-epoch weights, {desc}: mean over folds ± std across folds")
    if len(models) >= 2:
        (la, a), (lb, b) = list(data["pp"].items())[:2]
        paired(a, b, la, lb, args.n_boot, args.seed)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Markdown tables for the 5-fold CV")
    p.add_argument("--model", action="append", required=True, metavar="LABEL=BASE",
                   help="results/cv5/<BASE>_f<k>; repeat per model, the first two are compared paired")
    p.add_argument("--root", type=Path, default=Path("results/cv5"))
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
