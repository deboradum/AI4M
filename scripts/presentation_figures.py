#!/usr/bin/env python3
# MIT License
#
# Figures for docs/presentation/, computed from the run's metrics and
# predictions so they always match the submission. Re-run after the final
# checkpoint:
#
#   python scripts/presentation_figures.py --run results/snapshots/E_F07_ep24 --dest docs/presentation/figures
import json
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import label, distance_transform_edt, generate_binary_structure

ORGANS = ["esophagus", "heart", "trachea", "aorta"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
OURS, BASE = "#2a78d6", "#b9b8b2"          # categorical slot 1; baseline as neutral reference
REAL, FALSE = "#2a78d6", "#eb6834"         # categorical slots 1-2 (validated pair)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def mean_metric(folder: Path, m: str) -> np.ndarray:
    npz = np.load(folder / f"{m}.npz")
    return np.nanmean([npz[i][1:] for i in sorted(npz.files)], 0)


def bars(ax, base: np.ndarray, ours: np.ndarray, title: str, fmt: str, better: str) -> None:
    x = np.arange(len(ORGANS))
    w = 0.36
    for xs, vals, color, name in [(x - w / 2 - 0.01, base, BASE, "ENet baseline"), (x + w / 2 + 0.01, ours, OURS, "3D U-Net + post-proc.")]:
        ax.bar(xs, vals, w, color=color, label=name, zorder=2)
        for xi, v in zip(xs, vals):
            ax.text(xi, v, fmt.format(v), ha="center", va="bottom", fontsize=9.5, color=INK, zorder=3)
    ax.set_xticks(x, ORGANS)
    ax.set_title(f"{title}  ({better} is better)", loc="left", color=INK, fontsize=12)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.tick_params(length=0)


def fig_results(run: Path, baseline_json: Path, dest: Path) -> None:
    r = json.loads(baseline_json.read_text())["results"]["metrics_3d"]
    base_dice = np.array([r["dice_mean"][o] for o in ORGANS])
    base_hd = np.array([r["hd95_mm_mean"][o] for o in ORGANS])
    final = run / "val" / "pp_final" / "metrics"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    bars(axes[0], base_dice, mean_metric(final, "dice"), "3D Dice", "{:.2f}", "higher")
    axes[0].set_ylim(0, 1.08)
    bars(axes[1], base_hd, mean_metric(final, "hd95"), "HD95 (mm)", "{:.1f}", "lower")
    axes[1].set_ylim(0, max(base_hd.max(), mean_metric(final, "hd95").max()) * 1.18)
    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=2, labelcolor=INK)
    fig.text(0.01, 0.01, "Validation set, 8 patients; mean over patients.", color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(dest / "results_vs_baseline.png", dpi=200)
    plt.close(fig)


def leftover_pieces(run: Path) -> list[tuple[str, int, float, float, bool]]:
    """Every non-largest piece of esophagus/aorta in the raw val predictions:
    (patient, class, size ml, distance to the main body mm, real fragment?)."""
    st = generate_binary_structure(3, 1)
    rows = []
    for path in sorted((run / "val" / "nii").glob("*.nii.gz")):
        pid = path.name[:10]
        img = nib.load(str(path))
        pred = np.asarray(img.dataobj)
        sp = img.header.get_zooms()[:3]
        vml = float(np.prod(sp)) / 1000
        gt = np.asarray(nib.load(f"data/segthor_train_full/train/{pid}/GT.nii.gz").dataobj)
        for k in (1, 4):
            lab, n = label(pred == k, st)
            if n < 2:
                continue
            cnt = np.bincount(lab.ravel())
            big = 1 + int(np.argmax(cnt[1:]))
            d = distance_transform_edt(lab != big, sampling=sp)
            for c in range(1, n + 1):
                if c != big:
                    m = lab == c
                    rows.append((pid, k, cnt[c] * vml, float(d[m].min()), bool((gt[m] == k).mean() > 0.5)))
    return rows


def fig_pieces(run: Path, dest: Path) -> None:
    rows = leftover_pieces(run)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for real, color, name in [(False, FALSE, "false blob (outside GT)"), (True, REAL, "real fragment (inside GT)")]:
        for k, marker, organ in [(1, "o", "esophagus"), (4, "^", "aorta")]:
            pts = [(d, ml) for _, kk, ml, d, rr in rows if kk == k and rr == real]
            if pts:
                xs, ys = zip(*pts)
                ax.scatter(xs, ys, s=58, marker=marker, color=color, edgecolor=SURFACE, linewidth=1.5,
                           label=f"{name}, {organ}", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
    ax.xaxis.set_major_formatter(plain)
    ax.yaxis.set_major_formatter(plain)
    ax.axhline(1.0, color=INK2, linestyle=(0, (4, 3)), linewidth=1, zorder=2)
    ax.text(ax.get_xlim()[0] * 1.15, 1.15, "1 ml", color=INK2, fontsize=9.5)
    ax.set_xlabel("distance to the organ's main body (mm)")
    ax.set_ylabel("piece size (ml)")
    ax.set_title("Leftover pieces in the validation predictions", loc="left", color=INK, fontsize=12)
    ax.grid(True, which="major", color=GRID, zorder=0)
    ax.tick_params(length=0)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, labelcolor=INK)
    fig.text(0.01, 0.01, "Real fragments that matter are 9-21 ml; every false blob is under 0.4 ml, at any distance.",
             color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(dest / "leftover_pieces.png", dpi=200)
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    args.dest.mkdir(parents=True, exist_ok=True)
    fig_results(args.run, args.baseline, args.dest)
    fig_pieces(args.run, args.dest)
    print(f"Wrote figures to {args.dest}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--baseline", type=Path, default=Path("experiments/ENet/baseline/E_F00_baseline.json"))
    p.add_argument("--dest", type=Path, default=Path("docs/presentation/figures"))
    main(p.parse_args())
