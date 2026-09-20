#!/usr/bin/env python3

# MIT License

# Augmentation slide figure for the mid-term talk: one vertical strip of a training
# slice drawn as the network receives it, then again under three independent
# augmentation draws, then once with the elastic term enabled.
#
# Everything is produced by the shipped code: the slice is loaded with main.img_transform
# / main.gt_transform and warped with dataset.augment_sample under dataset.AugParams,
# exactly the functions the DataLoader calls. Nothing here re-implements a transform,
# and the GT contours are drawn from the warped one-hot to show that the label follows
# the image.
#
# The point of the strip: each draw is a different transform, because __getitem__ samples
# per slice. Three draws of one slice therefore disagree with each other, which is what
# z-consistency costs.
#
# The bottom panel is the elastic term at its configured strength (alpha 10 px, sigma 4),
# with the [2/w, 2/h] grid scaling fix in place. It is in the code but disabled in every
# kept run, so it is drawn here to be explicit about what "off" means.
#
# Orientation: panels are shown exactly as the network receives them (the slice PNG
# straight from disk, no transpose), so the strip can be checked against the tensor.
#
# Run from the repo root with the repo venv:
#   ./ai4mi/bin/python make_augmentation_slide_figure.py [--out PRESENTATION]
# The output lives in PRESENTATION/, which .gitignore excludes.

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from main import img_transform, gt_transform
from dataset import AugParams, augment_sample
from visualize_labels import class_colour, CLASS_NAMES

DATASET = Path("data/SEGTHOR_aorta_huwide/train")
K = 5                                  # background, esophagus, heart, trachea, aorta
SEEDS = (11, 12, 13)                   # one seed per draw; fixed, so the strip reproduces
ELASTIC_SEED = 21
DPI = 150

PANEL_IN = 3.0                         # square panel
SIDE_IN = 0.20
GAP_IN = 0.32                          # holds each panel's title without touching the next
HEAD_BAND_IN = 0.80                    # organ legend, two rows, above the first panel
CAPTION_BAND_IN = 2.95                 # ten short caption lines, one line per row
TITLE_FS, CAPTION_FS, LEGEND_FS = 11, 11, 10
CONTOUR_LW = 1.1
CAPTION_LINES: tuple[str, ...] = (
    "shown as the network receives it",
    "",
    "per draw: rotation +/-15 deg,",
    "scale 0.9-1.1, intensity +/-0.1",
    "(uniform, resampled for every slice)",
    "",
    "no horizontal flip: the thorax",
    "is not left-right symmetric",
    "",
    "bottom panel: elastic alpha=10 px,",
    "sigma=4 - in the code, never on",
    "in a kept run",
)


def organ_slice() -> tuple[Path, Path, str, int]:
    # Deterministic pick: the slice in the TRAIN split that shows all four organs at
    # once, with the most foreground voxels. Training split on purpose - validation is
    # never augmented, so a validation slice would misrepresent what the strip shows.
    # Patient_15 cannot be used: no slice of it carries all four organs at once.
    gt_paths = sorted((DATASET / "gt").glob("Patient_*.png"))
    assert gt_paths, f"no GT slices under {DATASET}"
    best: tuple[int, Path, dict[int, int]] | None = None
    for path in gt_paths:
        raw = np.array(Image.open(path))
        counts = {k: int((raw == k * 63).sum()) for k in range(1, K)}
        if all(n > 0 for n in counts.values()):
            total = sum(counts.values())
            if best is None or total > best[0]:
                best = (total, path, counts)
    assert best is not None, f"no slice with all four organs under {DATASET}"
    total, gt_path, counts = best
    img_path = DATASET / "img" / gt_path.name
    assert img_path.exists(), img_path
    patient, z = gt_path.stem.rsplit("_", maxsplit=1)
    return img_path, gt_path, patient, int(z)


def draw(ax, img: np.ndarray, gt: np.ndarray, title: str) -> None:
    # img (H, W) float [0, 1], gt (K, H, W) one-hot; contours follow the warped label.
    ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
    for k in range(1, K):
        mask = gt[k]
        if mask.any():
            ax.contour(mask.astype(float), levels=[0.5], colors=[class_colour(k, K)],
                       linewidths=CONTOUR_LW)
    ax.set_title(title, fontsize=TITLE_FS, pad=7)
    ax.set_xticks([])
    ax.set_yticks([])


def augment(img: torch.Tensor, gt: torch.Tensor, params: AugParams, seed: int):
    torch.manual_seed(seed)            # deterministic: same seed, same draw
    return augment_sample(img.clone(), gt.clone(), params, K)


def make_figure(out_dir: Path, patient: str, z: int,
                panels: list[tuple[np.ndarray, np.ndarray, str]]) -> Path:
    n = len(panels)
    fig_w = SIDE_IN * 2 + PANEL_IN
    fig_h = HEAD_BAND_IN + n * PANEL_IN + (n - 1) * GAP_IN + CAPTION_BAND_IN
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    for i, (img, gt, title) in enumerate(panels):
        y0 = CAPTION_BAND_IN + (n - 1 - i) * (PANEL_IN + GAP_IN)
        ax = fig.add_axes([SIDE_IN / fig_w, y0 / fig_h, PANEL_IN / fig_w, PANEL_IN / fig_h])
        draw(ax, img, gt, title)

    left = SIDE_IN / fig_w
    lines = (f"{patient}, axial z = {z} (train slice)",) + CAPTION_LINES
    line_h = (CAPTION_BAND_IN - 0.35) / len(lines)
    for j, line in enumerate(lines):
        if not line:
            continue
        fig.text(left, (CAPTION_BAND_IN - 0.20 - line_h * (j + 0.5)) / fig_h, line,
                 fontsize=CAPTION_FS, ha="left", va="center", color="black")

    # Legend in its own band above the panels: it must never reach into the caption.
    # Two columns: four entries on one row are wider than this narrow figure.
    handles = [plt.Line2D([], [], color=class_colour(k, K), lw=2, label=CLASS_NAMES[k])
               for k in range(1, K)]
    fig.legend(handles=handles, loc="upper left", fontsize=LEGEND_FS, frameon=False, ncol=2,
               bbox_to_anchor=(left, 0.999))

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fig5_augmentation.png"
    fig.savefig(path, dpi=DPI, bbox_inches=None, facecolor="white")
    plt.close(fig)
    return path


def main(args: argparse.Namespace) -> None:
    img_path, gt_path, patient, z = organ_slice()
    print(f"slice: {img_path.name} (all four organs, most foreground voxels)")

    prepared = AugParams(elastic_alpha=0.0)      # the kept recipe: elastic disabled
    elastic = AugParams()                        # alpha 10 px, sigma 4: the code default

    # Loaded by the shipped transforms, so the strip starts from the real network input.
    img0 = img_transform(Image.open(img_path))
    gt0 = gt_transform(K, Image.open(gt_path)).float()
    assert img0.shape[-2:] == gt0.shape[-2:], (img0.shape, gt0.shape)

    # Every organ drawn in the legend must actually be on this slice, otherwise the
    # strip would advertise contours that are not in the picture.
    present = {CLASS_NAMES[k]: int(gt0[k].sum()) for k in range(1, K)}
    print("organs on the slice: " + ", ".join(f"{n} {v}" for n, v in present.items()))
    assert sum(v > 0 for v in present.values()) == K - 1, present

    panels: list[tuple[np.ndarray, np.ndarray, str]] = [
        (img0[0].numpy(), gt0.numpy(), f"original (z = {z})"),
    ]
    for i, seed in enumerate(SEEDS, start=1):
        img_a, gt_a = augment(img0, gt0, prepared, seed)
        panels.append((img_a[0].numpy(), gt_a.numpy(), f"draw {i} (seed {seed})"))

    img_e, gt_e = augment(img0, gt0, elastic, ELASTIC_SEED)
    panels.append((img_e[0].numpy(), gt_e.numpy(), f"elastic on (seed {ELASTIC_SEED})"))

    # A broken import or a no-op transform would silently render the same thing twice.
    for i in range(1, len(SEEDS) + 1):
        base, other = panels[0][0], panels[i][0]
        assert not np.allclose(base, other), \
            f"original and draw {i} are identical: augment_sample did not run"

    path = make_figure(args.out, patient, z, panels)
    with Image.open(path) as png:
        w, h = png.size
    print(f"saved {path} ({w}x{h} px)")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw the augmentation slide figure (fig5_augmentation.png)")
    parser.add_argument("--out", type=Path, default=Path("PRESENTATION"),
                        help="output directory for the figure")
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
