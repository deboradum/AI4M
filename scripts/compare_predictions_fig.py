#!/usr/bin/env python3
# Side-by-side prediction figures for the E_F05 / E_F06 / E_F07 comparison (8 Oct 2026).
# Rows = ground truth (val only) + one row per model (post-processed); columns = views.
#
#   python scripts/compare_predictions_fig.py --split val --patient Patient_15 --dest figs/val_P15.png
#   python scripts/compare_predictions_fig.py --split test --patient Patient_41 --dest figs/test_P41.png
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

SNAP = Path("results/snapshots")
MODELS = [("E_F06  2D", "E_F06_final"), ("E_F05  2.5D", "E_F05_final"), ("E_F07  3D", "E_F07_final")]
ORGANS = ["esophagus", "heart", "trachea", "aorta"]
COLORS = ["#eb6834", "#d6455e", "#2a78d6", "#3aa66b"]
CMAP = ListedColormap([(0, 0, 0, 0)] + [matplotlib.colors.to_rgba(c, 0.55) for c in COLORS])


def load(p: Path) -> np.ndarray:
    return np.asarray(nib.load(str(p)).dataobj)


def views(vol: np.ndarray, ref: np.ndarray) -> list[np.ndarray]:
    """Axial slice through the esophagus' middle + coronal and sagittal through the heart's centre."""
    z_eso = np.where((ref == 1).any((0, 1)))[0]
    z = int(np.median(z_eso)) if len(z_eso) else vol.shape[2] // 2
    cx, cy, _ = [int(c) for c in np.argwhere(ref == 2).mean(0)] if (ref == 2).any() else [s // 2 for s in vol.shape]
    return [np.rot90(vol[:, :, z]), np.rot90(vol[:, cy, :]), np.rot90(vol[cx, :, :])]


def main(a: argparse.Namespace) -> None:
    ct_pat = ("data/segthor_train_full/train/{p}/{p}.nii.gz" if a.split == "val"
              else "data/segthor_test/test/{p}.nii.gz")
    ct_img = nib.load(ct_pat.format(p=a.patient))
    ct = np.asarray(ct_img.dataobj).astype(np.float32)
    rows = []
    if a.split == "val":
        rows.append(("Ground truth", load(Path(f"data/segthor_train_full/train/{a.patient}/GT.nii.gz"))))
    for name, snap in MODELS:
        rows.append((name, load(SNAP / snap / a.split / "pp_final" / "nii" / f"{a.patient}.nii.gz")))
    ref = rows[0][1]
    sx, sy, sz = ct_img.header.get_zooms()[:3]
    aspects = [sx / sy, sz / sx, sz / sy]
    ct_v = views(np.clip(ct, -400, 600), ref)
    fig, axes = plt.subplots(len(rows), 3, figsize=(10.5, 3.2 * len(rows)), facecolor="#fcfcfb")
    for i, (name, lab) in enumerate(rows):
        for j, (c, l) in enumerate(zip(ct_v, views(lab, ref))):
            ax = axes[i, j]
            ax.imshow(c, cmap="gray", aspect=aspects[j], interpolation="lanczos")
            ax.imshow(l, cmap=CMAP, vmin=0, vmax=4, aspect=aspects[j], interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if j == 0:
                ax.set_ylabel(name, fontsize=12, color="#0b0b0b")
            if i == 0:
                ax.set_title(["Axial", "Coronal", "Sagittal"][j], fontsize=12, color="#52514e")
    handles = [matplotlib.patches.Patch(color=c, label=o) for o, c in zip(ORGANS, COLORS)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=11)
    fig.suptitle(f"{a.patient} ({a.split}), predictions after post-processing", fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    a.dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.dest, dpi=110)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--patient", required=True)
    p.add_argument("--dest", type=Path, required=True)
    main(p.parse_args())
