#!/usr/bin/env python3

# MIT License
#
# 3D evaluation of stitched NIfTI predictions against the ground truth, per
# patient and per class, following the Metrics Reloaded recommendations for
# semantic segmentation (Maier-Hein et al., 2024): an overlap metric (Dice),
# and boundary metrics computed in millimetres (95th percentile Hausdorff
# distance, average symmetric surface distance, normalised surface Dice).
#
# Conventions for degenerate cases (Metrics Reloaded, "empty reference/prediction"):
# * class absent from both GT and prediction: every metric is NaN (undefined),
#   never 1.0. Averages over patients ignore NaN.
# * class present in only one of the two: Dice/IoU are 0, NSD is 0, and the
#   distance metrics are NaN since no surface exists to measure to.
#
# The metrics.py file next to this one holds the 2D per-slice metrics used
# during training; this script is the one that produces the numbers reported.

import csv
import argparse
import warnings
from pathlib import Path
from functools import partial
from multiprocessing import Pool

import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion, distance_transform_edt

from utils import tqdm_

METRICS: list[str] = ["dice", "iou", "hd95", "assd", "nsd"]


def surface(mask: np.ndarray) -> np.ndarray:
    # Voxels of the mask that touch the outside (6-connectivity). Voxels on the
    # array border count as surface, hence border_value=0.
    eroded = binary_erosion(mask, iterations=1, border_value=0)
    return mask & ~eroded


def surface_distances(pred: np.ndarray, gt: np.ndarray,
                      spacing: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    # Distances (in mm) from every surface voxel of pred to the closest surface
    # voxel of gt, and vice versa.
    surf_pred = surface(pred)
    surf_gt = surface(gt)

    dt_to_gt = distance_transform_edt(~surf_gt, sampling=spacing)
    dt_to_pred = distance_transform_edt(~surf_pred, sampling=spacing)

    return dt_to_gt[surf_pred], dt_to_pred[surf_gt]


def class_metrics(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float],
                  tolerance: float) -> dict[str, float]:
    assert pred.shape == gt.shape, (pred.shape, gt.shape)
    assert pred.dtype == bool and gt.dtype == bool

    n_pred: int = int(pred.sum())
    n_gt: int = int(gt.sum())

    if n_pred == 0 and n_gt == 0:
        return {m: np.nan for m in METRICS}

    inter: int = int((pred & gt).sum())
    dice: float = 2 * inter / (n_pred + n_gt)
    iou: float = inter / (n_pred + n_gt - inter)

    if n_pred == 0 or n_gt == 0:
        return {"dice": dice, "iou": iou, "hd95": np.nan, "assd": np.nan, "nsd": 0.0}

    d_pred_to_gt, d_gt_to_pred = surface_distances(pred, gt, spacing)

    hd95: float = max(np.percentile(d_pred_to_gt, 95), np.percentile(d_gt_to_pred, 95))
    assd: float = (d_pred_to_gt.sum() + d_gt_to_pred.sum()) / (len(d_pred_to_gt) + len(d_gt_to_pred))
    nsd: float = ((d_pred_to_gt <= tolerance).sum() + (d_gt_to_pred <= tolerance).sum()) \
        / (len(d_pred_to_gt) + len(d_gt_to_pred))

    return {"dice": dice, "iou": iou, "hd95": float(hd95), "assd": float(assd), "nsd": float(nsd)}


def load_labels(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    nib_obj = nib.load(str(path))
    arr = np.asarray(nib_obj.dataobj)
    assert arr.ndim == 3, (path, arr.shape)
    assert np.issubdtype(arr.dtype, np.integer), (path, arr.dtype)
    zooms = tuple(float(z) for z in nib_obj.header.get_zooms()[:3])
    return arr.astype(np.uint8), zooms


def evaluate_patient(id_: str, pred_folder: Path, gt_pattern: str, scan_pattern: str | None,
                     K: int, tolerance: float) -> dict[str, np.ndarray]:
    pred, pred_spacing = load_labels(pred_folder / f"{id_}.nii.gz")
    gt, gt_spacing = load_labels(Path(gt_pattern.format(id_=id_)))
    assert pred.shape == gt.shape, f"{id_}: prediction {pred.shape} vs GT {gt.shape}"
    assert set(np.unique(pred)) <= set(range(K)), (id_, np.unique(pred))
    assert set(np.unique(gt)) <= set(range(K)), (id_, np.unique(gt))

    # Spacing: the GT files of SegTHOR carry an identity affine (1x1x1 mm) which
    # is wrong. Take it from the CT scan when given, else from the prediction
    # (which our stitch.py copies from the CT).
    spacing: tuple[float, float, float]
    if scan_pattern is not None:
        spacing = tuple(float(z) for z in nib.load(scan_pattern.format(id_=id_)).header.get_zooms()[:3])
    else:
        spacing = pred_spacing
    if not np.allclose(spacing, gt_spacing):
        warnings.warn(f"{id_}: using spacing {spacing} ; GT header says {gt_spacing} (ignored)")

    results: dict[str, np.ndarray] = {m: np.full(K, np.nan, dtype=np.float64) for m in METRICS}
    for k in range(K):
        per_class = class_metrics(pred == k, gt == k, spacing, tolerance)
        for m in METRICS:
            results[m][k] = per_class[m]

    return results


def print_table(ids: list[str], per_patient: dict[str, dict[str, np.ndarray]],
                K: int, class_names: list[str]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN columns
        for m in METRICS:
            stacked = np.stack([per_patient[id_][m] for id_ in ids])  # (N, K)
            print(f"\n{m.upper()}" + ("" if m in ["dice", "iou", "nsd"] else " (mm)"))
            header = f"{'patient':12s}" + "".join(f"{name:>12s}" for name in class_names[1:])
            print(header)
            for id_, row in zip(ids, stacked):
                print(f"{id_:12s}" + "".join(f"{v:12.3f}" for v in row[1:]))
            mean = np.nanmean(stacked, axis=0)
            std = np.nanstd(stacked, axis=0)
            print(f"{'mean':12s}" + "".join(f"{v:12.3f}" for v in mean[1:]))
            print(f"{'std':12s}" + "".join(f"{v:12.3f}" for v in std[1:]))
            fg = mean[1:]
            print(f"{'fg mean':12s}{np.nanmean(fg):12.3f}   (classes present only)")


def save_results(dest: Path, ids: list[str], per_patient: dict[str, dict[str, np.ndarray]],
                 class_names: list[str]) -> None:
    dest.mkdir(parents=True, exist_ok=True)

    # Submission format: one .npz per metric mapping patient id -> array of shape (K,)
    for m in METRICS:
        np.savez(dest / f"{m}.npz", **{id_: per_patient[id_][m] for id_ in ids})

    # Long-format csv, convenient for pandas or the experiment log
    with open(dest / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient", "class", "class_name"] + METRICS)
        for id_ in ids:
            for k, name in enumerate(class_names):
                writer.writerow([id_, k, name] + [f"{per_patient[id_][m][k]:.6f}" for m in METRICS])

    print(f"\nSaved {', '.join(m + '.npz' for m in METRICS)} and metrics.csv to {dest}")


def main(args: argparse.Namespace) -> None:
    K: int = args.num_classes
    class_names: list[str] = args.class_names or [f"class{k}" for k in range(K)]
    assert len(class_names) == K, (class_names, K)

    ids: list[str] = sorted(p.name.removesuffix(".nii.gz") for p in args.pred_folder.glob("*.nii.gz"))
    assert len(ids) > 0, f"No .nii.gz found in {args.pred_folder}"
    print(f"Evaluating {len(ids)} patients from {args.pred_folder}: {ids}")

    pfun = partial(evaluate_patient,
                   pred_folder=args.pred_folder,
                   gt_pattern=args.gt_pattern,
                   scan_pattern=args.scan_pattern,
                   K=K,
                   tolerance=args.tolerance)

    results: list[dict[str, np.ndarray]]
    match args.process:
        case 1:
            results = [pfun(id_) for id_ in tqdm_(ids)]
        case -1:
            results = Pool().map(pfun, ids)
        case _ as p:
            results = Pool(p).map(pfun, ids)

    per_patient: dict[str, dict[str, np.ndarray]] = dict(zip(ids, results))

    print_table(ids, per_patient, K, class_names)
    if args.dest:
        save_results(args.dest, ids, per_patient, class_names)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='3D metrics of stitched predictions against the ground truth')
    parser.add_argument('--pred_folder', type=Path, required=True,
                        help="Folder with the predicted Patient_XX.nii.gz volumes (output of stitch.py)")
    parser.add_argument('--gt_pattern', type=str, required=True,
                        help="Pattern of the GT volume with {id_} placeholder, "
                             "e.g. 'data/segthor_part1/train/{id_}/GT.nii.gz'")
    parser.add_argument('--scan_pattern', type=str, default=None,
                        help="Optional pattern of the CT scan, used only to read the voxel spacing. "
                             "Defaults to the spacing stored in the prediction header.")
    parser.add_argument('--dest', type=Path, default=None,
                        help="Folder where to save the .npz (one per metric) and metrics.csv")
    parser.add_argument('--num_classes', '-K', type=int, default=5)
    parser.add_argument('--class_names', type=str, nargs='+', default=None,
                        help="K names, background first, e.g. background esophagus heart trachea aorta")
    parser.add_argument('--tolerance', type=float, default=2.0,
                        help="Tolerance in mm for the normalised surface Dice (NSD)")
    parser.add_argument('--process', '-p', type=int, default=1,
                        help="Number of processes (1: sequential, -1: all cores)")

    args = parser.parse_args()
    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
