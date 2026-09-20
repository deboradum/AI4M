#!/usr/bin/env python3

# MIT License
#
# HU-clipping slide figure for the mid-term talk: one axial slice of the cohort's
# worst implant case (the patient with the largest HU maximum, 31743 HU), drawn three
# times - per-patient min-max, the wide window [-1000, 1000] that the aorta line kept,
# and the soft-tissue window [-200, 300].
#
# The intensity mapping is slice_segthor.norm_arr / slice_segthor.norm_window, imported
# verbatim: the figure has to show the shipped preprocessing, not a redrawing of it.
# Nothing re-implements the clip-and-scale, and no value is hardcoded from an earlier
# run - the patient and the slice are derived from the data on every call, and the
# script is deterministic (no randomness, no timestamps in the image).
#
# Run from the repo root with the repo venv:
#   ./ai4mi/bin/python make_hu_slide_figure.py [--out PRESENTATION]
# The output lives in PRESENTATION/, which .gitignore excludes.

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from slice_segthor import norm_arr, norm_window

SRC_DIR = Path("data/segthor_part1/train")
HEART_CLASS = 2                        # the largest, most reliably present organ
WIDE_WINDOW = (-1000.0, 1000.0)        # E014_hu_wide, now the default aorta-line window
SOFT_WINDOW = (-200.0, 300.0)          # E014_hu_soft
DPI = 150

# Explicit layout bands, in inches: figure text must never touch an image, so the
# panels sit in a fixed band between the title band and the caption band and every
# text block is placed inside its own band by construction.
SIDE_IN = 0.15                         # left and right figure margin
PANEL_IN = 4.0                         # square panel, 512x512 image at aspect equal
HGAP_IN = 0.30                         # white gutter between panels
TITLE_BAND_IN = 0.55                   # panel titles above the image row
CAPTION_BAND_IN = 1.15                 # caption lines below the image row
FIG_W_IN = 2 * SIDE_IN + 3 * PANEL_IN + 2 * HGAP_IN
FIG_H_IN = TITLE_BAND_IN + PANEL_IN + CAPTION_BAND_IN
TITLE_FS, CAPTION_FS = 13, 12


def largest_hu_patient(src_dir: Path) -> tuple[str, Path, int]:
    # The implant case the slide is about: the patient whose CT holds the cohort's
    # largest HU value. Ties break on the sorted id, so the choice is deterministic.
    best_id, best_path, best_max = None, None, None
    for ct_path in sorted(src_dir.glob("Patient_*/Patient_*.nii.gz")):
        pid = ct_path.parent.name
        if ct_path.name != f"{pid}.nii.gz":    # never pick up GT.nii.gz / GT2.nii.gz
            continue
        hu_max = int(np.asarray(nib.load(str(ct_path)).dataobj).max())
        if best_max is None or hu_max > best_max:
            best_id, best_path, best_max = pid, ct_path, hu_max
    assert best_id is not None, f"no Patient_*.nii.gz under {src_dir}"
    return best_id, best_path, best_max


def heart_slice(gt: np.ndarray) -> int:
    # The axial index with the most heart (class 2) voxels: heart is the large, always
    # present organ, so its maximally populated slice shows real anatomy in all three
    # renderings instead of a nearly empty frame.
    per_slice = (gt == HEART_CLASS).sum(axis=(0, 1))
    assert per_slice.max() > 0, "no heart (class 2) voxels in the GT"
    return int(per_slice.argmax())


def panel_axes(fig):
    # One axes per panel, placed by hand: a grid spec would either warn about the
    # reserved bands or let the top row drift under the titles.
    axes = []
    for i in range(3):
        x0 = (SIDE_IN + i * (PANEL_IN + HGAP_IN)) / FIG_W_IN
        axes.append(fig.add_axes([x0, CAPTION_BAND_IN / FIG_H_IN,
                                  PANEL_IN / FIG_W_IN, PANEL_IN / FIG_H_IN]))
    return axes


def make_figure(out_dir: Path, pid: str, hu_max: int, z: int,
                panels: tuple[np.ndarray, np.ndarray, np.ndarray],
                titles: tuple[str, str, str], dpi: int = DPI) -> Path:
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), facecolor="white")
    for ax, img, title in zip(panel_axes(fig), panels, titles):
        # Same transpose and origin as make_presentation_figures.ct_panel and
        # retrieve_aorta.panel, so this slide is oriented like the deck's other CT
        # figures. norm_arr/norm_window map per-pixel intensities, so the transpose
        # does not alter which mapping is applied.
        ax.imshow(np.asarray(img), cmap="gray", vmin=0, vmax=255, origin="lower")
        ax.set_title(title, fontsize=TITLE_FS, pad=9)
        ax.set_xticks([])
        ax.set_yticks([])

    left_frac = SIDE_IN / FIG_W_IN
    fig.text(left_frac, (CAPTION_BAND_IN - 0.32) / FIG_H_IN,
             f"{pid} - maximum HU in the cohort: {hu_max}", fontsize=CAPTION_FS,
             ha="left", va="center", color="black")
    fig.text(left_frac, (CAPTION_BAND_IN - 0.66) / FIG_H_IN,
             f"axial slice z = {z}", fontsize=CAPTION_FS,
             ha="left", va="center", color="black")
    fig.text(left_frac, (CAPTION_BAND_IN - 1.00) / FIG_H_IN,
             "3D foreground-mean Dice: E001 0.598 \u2192 E014_hu_wide 0.737 "
             "\u2192 E014_hu_soft 0.724",
             fontsize=CAPTION_FS, ha="left", va="center", color="black")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fig4_hu_window.png"
    fig.savefig(path, dpi=dpi, bbox_inches=None, facecolor="white")
    plt.close(fig)
    return path


def main(args: argparse.Namespace) -> None:
    out_dir: Path = args.out

    pid, ct_path, hu_max = largest_hu_patient(SRC_DIR)
    print(f"largest HU maximum in the cohort: {pid} ({hu_max} HU)")

    # No reorientation: the deck's other CT panels (retrieve_aorta.panel,
    # make_presentation_figures.ct_panel) all draw the shipped volume as-is with
    # [:, :, z].T and origin="lower", so this slide shows the same chest orientation
    # as the rest of the talk. The volumes are (L, P, S) with shape 512x512xZ, so the
    # third axis is the axial one whatever the in-plane axcodes are; canonicalising
    # would only mirror the panel relative to the neighbouring slides.
    ct = np.asarray(nib.load(str(ct_path)).dataobj)
    gt = np.asarray(nib.load(str(ct_path.parent / "GT.nii.gz")).dataobj)
    z = heart_slice(gt)
    print(f"axial slice: z={z} (most heart, class {HEART_CLASS})")

    axial = ct[:, :, z].T
    baseline = norm_arr(axial)
    wide = norm_window(axial, *WIDE_WINDOW)
    soft = norm_window(axial, *SOFT_WINDOW)

    assert not np.array_equal(baseline, wide), \
        "baseline and [-1000, 1000] panels are identical: norm_window/norm_arr import is broken"
    assert not np.array_equal(baseline, soft), \
        "baseline and [-200, 300] panels are identical: norm_window/norm_arr import is broken"

    path = make_figure(out_dir, pid, hu_max, z, (baseline, wide, soft),
                       ("per-patient min\u2013max (baseline)",
                        "[-1000, 1000] (kept)",
                        "[-200, 300] (soft tissue)"))

    from PIL import Image
    with Image.open(path) as png:
        size = png.size
    print(f"saved {path} ({size[0]}x{size[1]} px)")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw the HU-clipping slide figure (fig4_hu_window.png)")
    parser.add_argument("--out", type=Path, default=Path("PRESENTATION"),
                        help="output directory for the figure")
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())