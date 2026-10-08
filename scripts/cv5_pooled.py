#!/usr/bin/env python3
# MIT License
#
# Pooled per-patient 5-fold CV tables from results/cv5/<run>_f<k>/eval[/pp_final]/metrics (metrics3d.py
# outputs of scripts/run_cv5.sh). Every patient is held out exactly once, so the folds
# are pooled into one per-patient table (up to 40 patients) instead of averaging folds.
# Models with missing folds are reported on the folds they have, and every paired
# comparison uses only the patients both models have.
#
# Complements scripts/cv5_tables.py (mean ± std across folds, needs all 5 folds, two models):
# this one also works while some folds are still training, compares every pair of models
# and reports what the post-processing changes.
#
#   python scripts/cv5_pooled.py > results/cv5/cv5_pooled.md
import argparse
from pathlib import Path

import numpy as np

ORGANS = ["esophagus", "heart", "trachea", "aorta"]
METRICS = [("dice", "Dice", True, 3), ("hd95", "HD95 (mm)", False, 1), ("nsd", "NSD (2 mm)", True, 3),
           ("iou", "IoU", True, 3), ("assd", "ASSD (mm)", False, 2)]
MODELS = [("E_F11_enet_ce_dice", "E_F11 ENet"), ("E_F06_unet2d_adamw_cosine_150ep", "E_F06 2D"), ("E_F05_unet25d_adamw_cosine_150ep", "E_F05 2.5D"),
          ("E_F07_unet3d_adamw_cosine", "E_F07 3D")]


def load(root: Path, run: str, sub: str, metric: str) -> dict[str, np.ndarray]:
    """patient -> 4 organ values, pooled over the folds that have finished."""
    out = {}
    for k in range(5):
        f = root / f"{run}_f{k}" / sub / f"{metric}.npz"
        if f.exists():
            npz = np.load(f)
            out.update({p: npz[p][1:].astype(float) for p in npz.files})
    return out


def boot_ci(x: np.ndarray, rng: np.random.Generator, n: int = 10000) -> tuple[float, float]:
    b = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return tuple(np.percentile(b, [2.5, 97.5]))


def folds_done(root: Path, run: str) -> list[int]:
    return [k for k in range(5) if (root / f"{run}_f{k}" / "eval/pp_final/metrics/nsd.npz").exists()]


def main(a: argparse.Namespace) -> None:
    rng = np.random.default_rng(0)
    subs = [("eval/pp_final/metrics", "with the final post-processing"), ("eval/metrics", "raw (no post-processing)")]
    print("# 5-fold cross-validation, final-epoch weights\n")
    for run, name in MODELS:
        print(f"- {name}: folds {folds_done(a.root, run) or 'none'} finished")
    print("\nPooled per patient (each patient held out once); mean over patients, 95% bootstrap CI of the mean.")
    for sub, title in subs:
        print(f"\n## Per model, {title}\n")
        print("| Metric | Model | n | " + " | ".join(ORGANS) + " | mean [95% CI] |")
        print("|---|---|---|" + "---|" * (len(ORGANS) + 1))
        for m, label, _, nd in METRICS:
            for run, name in MODELS:
                d = load(a.root, run, sub, m)
                if not d:
                    continue
                arr = np.stack(list(d.values()))
                per_patient = np.nanmean(arr, 1)
                lo, hi = boot_ci(per_patient, rng)
                cells = [f"{v:.{nd}f}" for v in np.nanmean(arr, 0)]
                print(f"| {label} | {name} | {len(d)} | " + " | ".join(cells)
                      + f" | {per_patient.mean():.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}] |")

    pairs = [(MODELS[0], MODELS[1]), (MODELS[0], MODELS[2]), (MODELS[1], MODELS[2])]
    for sub, title in subs[:1]:
        print(f"\n## Paired differences, {title}\n")
        print("B − A per patient on the patients both have; 95% bootstrap CI; ✓ = CI excludes 0 in B's favour,"
              " ✗ = in A's favour.\n")
        print("| B − A | Metric | n | " + " | ".join(ORGANS) + " | mean | B better (patients) |")
        print("|---|---|---|" + "---|" * (len(ORGANS) + 2))
        for (ra, na_), (rb, nb_) in pairs:
            for m, label, higher, nd in METRICS[:3]:
                da, db = load(a.root, ra, sub, m), load(a.root, rb, sub, m)
                ids = sorted(set(da) & set(db))
                if not ids:
                    continue
                diff = np.stack([db[i] - da[i] for i in ids])
                cells = []
                for col in list(diff.T) + [np.nanmean(diff, 1)]:
                    col = col[~np.isnan(col)]
                    lo, hi = boot_ci(col, rng)
                    sign = 1 if higher else -1
                    mark = " ✓" if sign * lo > 0 and sign * hi > 0 else " ✗" if sign * lo < 0 and sign * hi < 0 else ""
                    cells.append(f"{col.mean():+.{nd}f}{mark}")
                better = int(((1 if higher else -1) * np.nanmean(diff, 1) > 0).sum())
                print(f"| {nb_} − {na_} | {label} | {len(ids)} | " + " | ".join(cells) + f" | {better}/{len(ids)} |")

    print("\n## Effect of the post-processing (post-processed − raw, per patient)\n")
    print("| Model | Metric | n | " + " | ".join(ORGANS) + " | patients changed |")
    print("|---|---|---|" + "---|" * (len(ORGANS) + 1))
    for run, name in MODELS:
        for m, label, higher, nd in METRICS[:2]:
            raw, pp = load(a.root, run, subs[1][0], m), load(a.root, run, subs[0][0], m)
            ids = sorted(set(raw) & set(pp))
            if not ids:
                continue
            diff = np.stack([pp[i] - raw[i] for i in ids])
            changed = int((np.abs(np.nan_to_num(diff)) > 1e-9).any(1).sum())
            print(f"| {name} | {label} | {len(ids)} | " + " | ".join(f"{v:+.{nd}f}" for v in np.nanmean(diff, 0))
                  + f" | {changed}/{len(ids)} |")


def variants(a: argparse.Namespace) -> None:
    """Post-processing variants (scripts/cv5_pp_sweep.sh, eval/pp_<name>) against raw and the
    final post-processing, pooled per patient; Δ = variant − final, paired, 95% bootstrap CI."""
    rng = np.random.default_rng(0)
    subs = [("raw", "eval/metrics"), ("final", "eval/pp_final/metrics")] + \
           [(v, f"eval/pp_{v}/metrics") for v in a.variants]
    print("\n## Post-processing variants (pooled per patient)\n")
    print("final = --lcc 2 3 --min_ml 4:1. Δ vs final: paired per patient, ✓/✗ = 95% CI excludes 0"
          " in the variant's favour / against it.\n")
    print("| Model | Variant | n | Dice eso | Dice trachea | Dice mean | HD95 eso | HD95 trachea | HD95 mean"
          " | Δ Dice mean | Δ HD95 mean (mm) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for run, name in MODELS:
        ref = {m: load(a.root, run, "eval/pp_final/metrics", m) for m in ("dice", "hd95")}
        for v, sub in subs:
            cur = {m: load(a.root, run, sub, m) for m in ("dice", "hd95")}
            ids = sorted(set(cur["dice"]) & set(ref["dice"]))
            if not ids:
                continue
            d = np.stack([cur["dice"][i] for i in ids]); h = np.stack([cur["hd95"][i] for i in ids])
            cells = [f"{d[:, 0].mean():.3f}", f"{d[:, 2].mean():.3f}", f"{np.nanmean(d, 1).mean():.3f}",
                     f"{h[:, 0].mean():.1f}", f"{h[:, 2].mean():.1f}", f"{np.nanmean(h, 1).mean():.1f}"]
            for m, nd, higher in (("dice", 3, True), ("hd95", 1, False)):
                diff = np.array([np.nanmean(cur[m][i]) - np.nanmean(ref[m][i]) for i in ids])
                if v == "final" or not np.any(diff):
                    cells.append("–")
                    continue
                lo, hi = boot_ci(diff, rng)
                sign = 1 if higher else -1
                mark = " ✓" if sign * lo > 0 and sign * hi > 0 else " ✗" if sign * lo < 0 and sign * hi < 0 else ""
                cells.append(f"{diff.mean():+.{nd}f}{mark}")
            print(f"| {name} | {v} | {len(ids)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("results/cv5"))
    p.add_argument("--variants", nargs="*", default=[],
                   help="Also compare post-processing variants eval/pp_<name> (scripts/cv5_pp_sweep.sh)")
    args = p.parse_args()
    main(args)
    if args.variants:
        variants(args)
