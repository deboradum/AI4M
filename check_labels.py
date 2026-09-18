#!/usr/bin/env python3

# MIT License
#
# Report which label values a ground-truth file actually holds, and answer the
# question "is the aorta label there?" in one glance.
#
# Motivation: data/segthor_part1.zip ships GT.nii.gz files holding only
# {0,1,2,3}, with class 1 = esophagus UNION aorta, so the aorta has no class of
# its own even though it is labelled. Nothing in the repository noticed, because
# main.py encodes the configured 5 classes into .pngs with img / 63 (an empty
# class 4 becomes an all-zero channel, not an error) and the model-selection Dice
# averages that empty class in at exactly 1.0. inspectDataset.py cannot replace
# this script either: it needs pandas/seaborn, absent from ai4mi/, and reports
# class volumes only, with no per-file verdict and no folded-class detection.
#
# Modes, combinable:
#   --nifti DIR     per-file class voxel counts, per-class PRESENT/ABSENT verdict
#   --png DIR       same for the 2D .png slices produced by slice_segthor.py
#   --components    per-file, per-class connected components: the folded-class detector
#   --diff A B      the KxK confusion between two label maps
#   --dice_npy PATH training's dice_val.npy, with the phantom-class-4 inflation
#   --json PATH     every aggregated result, machine-readable
#
# Exit codes: 0 all requested classes present; 1 a class in 1..K-1 is absent from
# every scanned file; 2 malformed input (a label value >= K, or nothing to scan).

import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import label as nd_label
from skimage.io import imread

from stitch import label_scale

CLASS_NAMES: list[str] = ["background", "esophagus", "heart", "trachea", "aorta"]


def load_labels(path: Path) -> np.ndarray:
    arr = np.asarray(nib.load(str(path)).dataobj)
    assert arr.ndim == 3, (path, arr.shape)
    assert np.issubdtype(arr.dtype, np.integer), (path, arr.dtype)
    return arr.astype(np.int32)


def counts(arr: np.ndarray, K: int) -> np.ndarray:
    return np.bincount(arr.ravel(), minlength=K)[:K]


def sibling_ct(path: Path) -> Path | None:
    # A GT sits next to its CT, named after the patient directory: <id>/GT.nii.gz
    # and <id>/<id>.nii.gz.
    candidate = path.parent / f"{path.parent.name}.nii.gz"
    return candidate if candidate.exists() else None


def scan_nifti(directory: Path, K: int, class_names: list[str],
               components: bool) -> tuple[dict, bool]:
    paths = sorted(directory.rglob("GT*.nii.gz"))
    assert paths, f"No GT*.nii.gz under {directory}"

    per_file: list[dict] = []
    stacked = np.zeros(K, dtype=np.int64)
    malformed = False
    print(f"Scanned {len(paths)} label files under {directory}")
    for path in paths:
        arr = load_labels(path)
        c = counts(arr, K)
        stacked += c
        print(f"{path}  " + " ".join(f"{class_names[k]}={int(c[k])}" for k in range(K)))
        for v in np.unique(arr):
            if int(v) >= K:
                print(f"ERROR {path}: label value {int(v)} >= K={K}")
                malformed = True
        entry = {"path": str(path), "counts": {class_names[k]: int(c[k]) for k in range(K)}}

        if components:
            entry["components"] = component_report(path, arr, K, class_names)
        per_file.append(entry)

    print("")
    for k in range(K):
        present = sum(1 for e in per_file if e["counts"][class_names[k]] > 0)
        if present:
            smallest = min(e["counts"][class_names[k]] for e in per_file
                           if e["counts"][class_names[k]] > 0)
            print(f"{class_names[k]} PRESENT in {present}/{len(per_file)} files (min {smallest} voxels)")
        else:
            print(f"{class_names[k]} ABSENT in {len(per_file)}/{len(per_file)} files")

    return {"files": per_file, "presence": {class_names[k]: int(stacked[k] > 0) for k in range(K)},
            "malformed": malformed}, malformed


def component_report(path: Path, arr: np.ndarray, K: int, class_names: list[str]) -> dict:
    ct = sibling_ct(path)
    hu = None
    if ct is not None:
        hu = np.asarray(nib.load(str(ct)).dataobj)
        assert hu.shape == arr.shape, (path, hu.shape, arr.shape)

    report: dict[str, list[dict]] = {}
    for k in range(1, K):
        mask = arr == k
        if not mask.any():
            print(f"  {class_names[k]}: absent")
            report[class_names[k]] = []
            continue
        labels, n = nd_label(mask)
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        order = [int(lab) for lab in np.argsort(sizes)[::-1] if sizes[lab] > 0][:2]
        found: list[dict] = []
        parts: list[str] = []
        for rank, lab in enumerate(order, start=1):
            comp = labels == lab
            n_vox = int(comp.sum())
            n_slices = int(comp.any(axis=(0, 1)).sum())
            mean_hu = f"{float(hu[comp].mean()):.1f}" if hu is not None else "-"
            found.append({"voxels": n_vox, "slices": n_slices, "meanHU": mean_hu})
            parts.append(f"{rank}: {n_vox} vx, {n_slices} slices, meanHU {mean_hu}")
        print(f"  {class_names[k]}: {n} component(s) | " + " | ".join(parts))
        report[class_names[k]] = found

    return report


def scan_png(directory: Path, K: int, class_names: list[str]) -> dict:
    paths = sorted(directory.rglob("gt/*.png"))
    assert paths, f"No gt/*.png under {directory}"

    scale = label_scale(K)
    per_class = np.zeros(K, dtype=np.int64)
    raw_values: set[int] = set()
    malformed = False
    print(f"Scanned {len(paths)} label slices under {directory} (label_scale={scale})")
    for path in paths:
        raw = imread(path)
        assert raw.ndim == 2 and raw.dtype == np.uint8, (path, raw.shape, raw.dtype)
        raw_values |= {int(v) for v in np.unique(raw)}
        labels = np.rint(raw / scale).astype(np.int32)
        for v in np.unique(labels):
            if int(v) >= K:
                print(f"ERROR {path}: label value {int(v)} >= K={K}")
                malformed = True
        present = np.unique(labels)
        present = present[(present >= 0) & (present < K)]
        per_class[present] += 1

    print(f"raw .png values: {sorted(raw_values)}")
    for k in range(K):
        print(f"{class_names[k]}: {int(per_class[k])}/{len(paths)} slices")

    return {"files": len(paths), "raw_values": sorted(raw_values),
            "slices_with_class": {class_names[k]: int(per_class[k]) for k in range(K)},
            "malformed": malformed}, malformed


def diff(a_path: Path, b_path: Path, K: int, class_names: list[str]) -> dict:
    a = load_labels(a_path)
    b = load_labels(b_path)
    assert a.shape == b.shape, (a_path, a.shape, b_path, b.shape)
    print(f"{a_path}: {a.shape}")
    print(f"{b_path}: {b.shape}")

    conf = np.zeros((K, K), dtype=np.int64)
    for va in range(K):
        for vb in range(K):
            conf[va, vb] = int(((a == va) & (b == vb)).sum())

    print(f"\nconfusion: rows {a_path.name}, cols {b_path.name}")
    row_column_label = "A\\B"
    print(f"{row_column_label:>8s}" + "".join(f"{class_names[k]:>12s}" for k in range(K)))
    for va in range(K):
        print(f"{class_names[va]:>8s}" + "".join(f"{conf[va, vb]:>12d}" for vb in range(K)))

    print("")
    for vb in range(K):
        sources = [f"A={va} {int(conf[va, vb])}" for va in range(K) if conf[va, vb] > 0]
        print(f"B={vb}: " + (" ".join(sources) if sources else "empty"))

    identical = bool(np.array_equal(a, b))
    print(f"\nidentical: {identical}")

    return {"a": str(a_path), "b": str(b_path), "shape": list(a.shape),
            "confusion": conf.tolist(), "identical": identical}


def dice_report(path: Path, K: int, class_names: list[str]) -> dict:
    dice = np.load(path)
    assert dice.ndim == 3, (path, dice.shape)
    present_path = path.parent / "present_val.npy"
    e = dice.shape[0] - 1
    print(f"{path}: {dice.shape}, last epoch {e}")

    present = None
    if present_path.exists():
        present = np.load(present_path)
        assert present.shape == dice.shape, (present_path, present.shape, dice.shape)

    rows = []
    for k in range(K):
        mean_dice = float(dice[e, :, k].mean())
        n_gt = int(present[e, :, k].sum()) if present is not None else None
        rows.append({"class": class_names[k], "mean_dice": mean_dice, "slices_with_gt": n_gt})
        print(f"{class_names[k]:12s} mean_dice={mean_dice:.4f}"
              + (f"  slices with GT={n_gt}" if n_gt is not None else ""))

    main_mean = float(dice[e, :, 1:].mean())
    print(f"main.py:321 mean (classes 1:): {main_mean:.4f}")

    result = {"path": str(path), "epoch": e, "per_class": rows, "main_mean": main_mean}
    if present is None:
        print("phantom check unavailable (no present_val.npy next to it)")
        return result

    with_gt = [k for k in range(1, K) if present[e, :, k].any()]
    gt_mean = float(np.mean([dice[e, :, k].mean() for k in with_gt]))
    print(f"mean over classes with GT ({[class_names[k] for k in with_gt]}): {gt_mean:.4f}")
    print(f"difference (phantom class inflation): {main_mean - gt_mean:+.4f}")

    result |= {"classes_with_gt": [class_names[k] for k in with_gt],
               "mean_over_classes_with_gt": gt_mean,
               "difference": main_mean - gt_mean}
    return result


def main(args: argparse.Namespace) -> None:
    K: int = args.K
    class_names: list[str] = args.class_names or [f"class{k}" for k in range(K)]
    assert len(class_names) == K, (class_names, K)
    assert K >= 2, K

    if not any([args.nifti, args.png, args.diff, args.dice_npy]):
        raise SystemExit("Nothing to do: pass --nifti, --png, --diff or --dice_npy")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        aggregated: dict = {"K": K, "class_names": class_names}
        malformed = False
        scanned = False
        absent_anywhere: list[str] = []

        if args.nifti is not None:
            assert args.nifti.is_dir(), f"No such directory: {args.nifti}"
            aggregated["nifti"], bad = scan_nifti(args.nifti, K, class_names, args.components)
            malformed |= bad
            scanned = True
            absent_anywhere += [class_names[k] for k in range(1, K)
                                if aggregated["nifti"]["presence"][class_names[k]] == 0]

        if args.png is not None:
            assert args.png.is_dir(), f"No such directory: {args.png}"
            aggregated["png"], bad = scan_png(args.png, K, class_names)
            malformed |= bad
            scanned = True
            absent_anywhere += [class_names[k] for k in range(1, K)
                                if aggregated["png"]["slices_with_class"][class_names[k]] == 0]

        if args.diff is not None:
            a_path, b_path = (Path(p) for p in args.diff)
            assert a_path.exists() and b_path.exists(), (a_path, b_path)
            aggregated["diff"] = diff(a_path, b_path, K, class_names)

        if args.dice_npy is not None:
            assert args.dice_npy.exists(), f"No such file: {args.dice_npy}"
            aggregated["dice"] = dice_report(args.dice_npy, K, class_names)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(aggregated, f, indent=2)
        print(f"\nWrote {args.json}")

    if malformed:
        raise SystemExit(2)
    if scanned and absent_anywhere:
        print(f"\nAbsent from every scanned file: {sorted(set(absent_anywhere))}")
        raise SystemExit(1)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Report the label values a ground-truth file or slice holds '
                    '(the aorta checker for the folded segthor_part1 labels)',
        epilog="Exit codes: 0 all classes present; 1 a class in 1..K-1 is absent from every "
               "scanned file; 2 malformed input (a label value >= K).")
    parser.add_argument('--nifti', type=Path, default=None,
                        help="Folder scanned recursively for GT*.nii.gz label volumes")
    parser.add_argument('--png', type=Path, default=None,
                        help="Sliced dataset folder scanned recursively for gt/*.png label slices")
    parser.add_argument('--K', type=int, default=5,
                        help="Number of classes, background included")
    parser.add_argument('--class_names', type=str, nargs='+', default=None,
                        help="K names, background first (default: background esophagus heart trachea aorta)")
    parser.add_argument('--components', action='store_true',
                        help="Per file and class, report the two largest connected components "
                             "with voxel/slice counts and mean HU (the folded-class detector)")
    parser.add_argument('--diff', type=str, nargs=2, default=None, metavar=('A', 'B'),
                        help="Two label volumes: print the KxK confusion (rows A, cols B) and identity")
    parser.add_argument('--dice_npy', type=Path, default=None,
                        help="A training dice_val.npy (and its sibling present_val.npy): last epoch "
                             "per-class Dice, the selection mean, and the phantom-class inflation")
    parser.add_argument('--json', type=Path, default=None,
                        help="Also dump every aggregated result to this JSON file")

    args = parser.parse_args()
    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
