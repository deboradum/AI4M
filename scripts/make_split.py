#!/usr/bin/env python3

# MIT License
#
# Build a stratified train/validation split of the SegTHOR training set and write it
# to a JSON file that slice_segthor.py consumes with --split_file, so every run and
# every teammate uses the same validation patients.
#
# Strata, measured from the NIfTI files (no hand-typed patient lists):
#   contrast   iodine contrast agent: median HU inside the aorta label above 100
#   spacing    in-plane pixel spacing away from the cohort's usual 0.977 mm
#   volume     an organ volume far from the cohort median (a hard or odd case)
# The validation set holds the cohort's share of contrast scans and at least one of
# each rare stratum, but never most of them: hard cases must be scored, not hidden,
# and not removed from training either.
#
#   python scripts/make_split.py --source_dir data/segthor_train_full \
#       --dest configs/splits/segthor_full_32_8.json --n_val 8

import json
import random
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib

CONTRAST_HU: float = 100.0          # median aorta HU above this: contrast-enhanced scan
USUAL_SPACING: tuple[float, float] = (0.95, 1.0)   # in-plane spacing outside this: outlier
VOLUME_FACTOR: float = 1.8          # organ volume above this multiple of the median: outlier
ORGANS: dict[int, str] = {1: "esophagus", 2: "heart", 3: "trachea", 4: "aorta"}


def measure(src: Path) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(src.glob("train/Patient_*")):
        pid = p.name
        ct_nib = nib.load(str(p / f"{pid}.nii.gz"))
        gt = np.asarray(nib.load(str(p / "GT.nii.gz")).dataobj)
        ct = np.asarray(ct_nib.dataobj)
        dx, dy, dz = (float(z) for z in ct_nib.header.get_zooms()[:3])
        voxel_ml = dx * dy * dz / 1000.0
        rows.append({"id": pid,
                     "spacing_mm": [round(dx, 3), round(dy, 3), round(dz, 3)],
                     "slices": int(gt.shape[2]),
                     "aorta_median_hu": float(np.median(ct[gt == 4])) if (gt == 4).any() else float("nan"),
                     **{f"{name}_ml": round(float((gt == k).sum()) * voxel_ml, 1) for k, name in ORGANS.items()}})
        print(f"  {pid}: spacing {rows[-1]['spacing_mm']}, aorta HU {rows[-1]['aorta_median_hu']:.0f}")
    return rows


def add_strata(rows: list[dict]) -> list[dict]:
    medians = {name: float(np.median([r[f"{name}_ml"] for r in rows])) for name in ORGANS.values()}
    for r in rows:
        r["contrast"] = bool(r["aorta_median_hu"] > CONTRAST_HU)
        r["spacing_outlier"] = not (USUAL_SPACING[0] <= r["spacing_mm"][0] <= USUAL_SPACING[1])
        r["volume_outlier"] = any(r[f"{name}_ml"] > VOLUME_FACTOR * medians[name] for name in ORGANS.values())
    return rows


def choose_validation(rows: list[dict], n_val: int, seed: int = 0, max_tries: int = 10000) -> tuple[list[str], int]:
    # Deterministic: the first seed (from `seed` upwards) whose random draw satisfies
    # the constraints wins, so the choice is reproducible and auditable.
    ids = [r["id"] for r in rows]
    n = len(ids)
    by = {k: {r["id"] for r in rows if r[k]} for k in ("contrast", "spacing_outlier", "volume_outlier")}
    share = n_val / n
    target_contrast = round(len(by["contrast"]) * share)

    def ok(val: set[str]) -> bool:
        return (len(val & by["contrast"]) == target_contrast
                and (not by["spacing_outlier"] or 1 <= len(val & by["spacing_outlier"]) <= max(1, len(by["spacing_outlier"]) // 2))
                and (not by["volume_outlier"] or 1 <= len(val & by["volume_outlier"]) <= max(1, len(by["volume_outlier"]) // 2)))

    for s in range(seed, seed + max_tries):
        rng = random.Random(s)
        val = set(rng.sample(ids, n_val))
        if ok(val):
            return sorted(val), s
    raise RuntimeError("no draw satisfied the stratification constraints")


def main(args: argparse.Namespace) -> None:
    print(f"Measuring {args.source_dir}")
    rows = add_strata(measure(args.source_dir))
    validation, seed = choose_validation(rows, args.n_val, args.seed)
    training = [r["id"] for r in rows if r["id"] not in validation]

    def count(ids: list[str], key: str) -> int:
        return sum(1 for r in rows if r["id"] in ids and r[key])

    summary = {k: {"train": count(training, k), "validation": count(validation, k)}
               for k in ("contrast", "spacing_outlier", "volume_outlier")}
    out = {"source_dir": str(args.source_dir), "n_train": len(training), "n_validation": len(validation),
           "seed": seed, "criteria": {"contrast": f"aorta median HU > {CONTRAST_HU}",
                                      "spacing_outlier": f"in-plane spacing outside {USUAL_SPACING} mm",
                                      "volume_outlier": f"any organ volume > {VOLUME_FACTOR} x cohort median"},
           "strata_counts": summary, "training": training, "validation": validation, "patients": rows}
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    args.dest.write_text(json.dumps(out, indent=2) + "\n")

    print(f"\nvalidation ({len(validation)}): {validation}")
    for k, v in summary.items():
        print(f"  {k:16s} train {v['train']:2d}  validation {v['validation']:2d}")
    print(f"seed {seed}; wrote {args.dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stratified train/validation split for SegTHOR")
    parser.add_argument("--source_dir", type=Path, required=True, help="Folder holding train/Patient_XX/")
    parser.add_argument("--dest", type=Path, required=True, help="Output JSON, e.g. configs/splits/segthor_full_32_8.json")
    parser.add_argument("--n_val", type=int, default=8, help="Number of validation patients")
    parser.add_argument("--seed", type=int, default=0, help="First random seed tried")
    main(parser.parse_args())
