#!/usr/bin/env python3
# MIT License
#
# Portrait slide panel: one patient's CT, coronal over axial, with two models' predictions
# overlaid. Same canvas, font and colour code as the vertical cards of scripts/render_enet_tables.py:
# the U-Net in organ colours (fill + outline), ENet as a red outline, ground truth as a thin white
# line; per-organ Dice for this patient underneath (ENet in red -> U-Net in bold).
#
#   python scripts/render_overlap_slices.py --patient Patient_14
import sys
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from render_enet_tables import (CARD_SIZE, COLORS, INK, INK2, MUTED, NAMES, RED, SURFACE, TRACK,  # noqa: E402
                                save, style)

WINDOW = (-350, 450)  # display window only (mediastinum + airway); the models saw -1000..1000
RUNS = {"enet": "E_F11_enet_ce_dice", "unet": "E_F06_unet2d_adamw_cosine_150ep"}


def fold_of(root: Path, run: str, pid: str) -> int:
    return next(k for k in range(5) if (root / f"{run}_f{k}/eval/pp_final/nii/{pid}.nii.gz").exists())


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return 2 * (a & b).sum() / s if s else np.nan


def outline(ax, mask2d: np.ndarray, colour: str, lw: float, halo: str | None = None, extent=None) -> None:
    if not mask2d.any():
        return
    cs = ax.contour(mask2d.astype(float), levels=[0.5], colors=[colour], linewidths=lw, extent=extent,
                    origin="lower")
    if halo:
        cs.set_path_effects([pe.Stroke(linewidth=lw + 1.0, foreground=halo), pe.Normal()])


def panel(ax, ct2d, gt2d, unet2d, enet2d, aspect: float, extent) -> None:
    """ct2d etc. are (rows=vertical, cols=horizontal) slices, displayed with origin lower."""
    ax.imshow(ct2d, cmap="gray", vmin=WINDOW[0], vmax=WINDOW[1], origin="lower", aspect=aspect,
              extent=extent, interpolation="bilinear")
    for k, col in zip((1, 2, 3, 4), COLORS):  # U-Net: fill first, then its outline
        m = unet2d == k
        if m.any():
            ax.contourf(m.astype(float), levels=[0.5, 1.5], colors=[col], alpha=0.38, extent=extent, origin="lower")
            outline(ax, m, col, 1.1, extent=extent)
    for k in (1, 2, 3, 4):
        outline(ax, gt2d == k, "#ffffff", 0.8, extent=extent)
    for k in (1, 2, 3, 4):  # ENet on top, dark halo so it reads on the red heart fill too
        outline(ax, enet2d == k, RED, 1.2, halo="#0b0b0b", extent=extent)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def main(a: argparse.Namespace) -> None:
    style()
    pid, root = a.patient, a.root
    ct_img = nib.load(a.data / pid / f"{pid}.nii.gz")
    sx, sy, sz = ct_img.header.get_zooms()[:3]
    ct = np.asarray(ct_img.dataobj, dtype=np.float32)
    gt = np.asarray(nib.load(a.data / pid / "GT.nii.gz").dataobj).astype(np.uint8)
    pred = {}
    for key, run in RUNS.items():
        k = fold_of(root, run, pid)
        pred[key] = np.asarray(nib.load(root / f"{run}_f{k}/eval/pp_final/nii/{pid}.nii.gz").dataobj).astype(np.uint8)
    fold = fold_of(root, RUNS["enet"], pid)

    # crop: box around every organ in GT and both predictions, plus a margin
    any_ = (gt > 0) | (pred["unet"] > 0) | (pred["enet"] > 0)
    idx = np.argwhere(any_)
    (x0, y0, z0), (x1, y1, z1) = idx.min(0), idx.max(0) + 1
    mx, mz = int(25 / sx), int(15 / sz)
    x0, x1 = max(x0 - mx, 0), min(x1 + mx, ct.shape[0])
    y0, y1 = max(y0 - mx, 0), min(y1 + mx, ct.shape[1])
    z0, z1 = max(z0 - mz, 0), min(z1 + mz, ct.shape[2])
    # coronal plane through the GT trachea
    tr = gt == 3
    yc = a.coronal if a.coronal is not None else int(round(np.argwhere(tr)[:, 1].mean()))
    # axial: where the U-Net finds the most real trachea that ENet misses
    gain = ((pred["unet"] == 3) & tr & (pred["enet"] != 3)).sum((0, 1))
    za = a.axial if a.axial is not None else int(np.argmax(gain))

    def cor(v):
        return v[x0:x1, yc, z0:z1].T  # rows = z (head up with origin lower), cols = x

    def axi(v):
        return v[x0:x1, y0:y1, za].T[::-1]  # rows = y flipped (LPS: anterior at the top), cols = x

    fig = plt.figure(figsize=CARD_SIZE)
    x_l, x_r = 0.10, 0.92
    fig.text(x_l, 0.950, f"2D U-Net vs ENet baseline · {pid.replace('_', ' ')}", fontsize=12, color=INK2)
    fig.text(x_l, 0.925, f"held out in CV fold {fold} · post-processed", fontsize=9, color=MUTED)

    w_mm, hc_mm, ha_mm = (x1 - x0) * sx, (z1 - z0) * sz, (y1 - y0) * sy
    ext_c, ext_a = [0, w_mm, 0, hc_mm], [0, w_mm, 0, ha_mm]
    ax_c = fig.add_axes([x_l, 0.505, x_r - x_l, 0.405])
    panel(ax_c, cor(ct), cor(gt), cor(pred["unet"]), cor(pred["enet"]), "equal", ext_c)
    ax_c.text(0.02, 0.98, "coronal", transform=ax_c.transAxes, ha="left", va="top", fontsize=9, color="#ffffff",
              path_effects=[pe.withStroke(linewidth=2, foreground="#0b0b0b")])
    ax_c.axhline((za - z0 + 0.5) * sz, color="#ffffff", lw=0.6, ls=(0, (2, 3)), alpha=0.7)
    ax_a = fig.add_axes([x_l, 0.255, x_r - x_l, 0.235])
    panel(ax_a, axi(ct), axi(gt), axi(pred["unet"]), axi(pred["enet"]), "equal", ext_a)
    ax_a.text(0.02, 0.97, "axial · dashed line above", transform=ax_a.transAxes, ha="left", va="top", fontsize=9,
              color="#ffffff", path_effects=[pe.withStroke(linewidth=2, foreground="#0b0b0b")])

    if not pred["enet"][..., za].any():
        ax_a.text(0.02, 0.03, "ENet: nothing on this slice", transform=ax_a.transAxes, ha="left", va="bottom",
                  fontsize=9, color="#ff6b6b", fontweight="bold")
    # legend, drawn like the cards' key: organ swatches = the U-Net, red line = ENet, thin line = ground truth
    ly = 0.228
    for n, c in enumerate(COLORS):
        fig.add_artist(Line2D([x_l + n * 0.018, x_l + 0.014 + n * 0.018], [ly + 0.006, ly + 0.006], color=c, lw=6,
                              solid_capstyle="butt"))
    fig.text(x_l + 0.085, ly, "2D U-Net", fontsize=8.5, color=INK2)
    fig.add_artist(Line2D([x_l + 0.30, x_l + 0.345], [ly + 0.006, ly + 0.006], color=RED, lw=1.8,
                          path_effects=[pe.Stroke(linewidth=2.8, foreground="#0b0b0b"), pe.Normal()]))
    fig.text(x_l + 0.36, ly, "ENet", fontsize=8.5, color=INK2)
    fig.add_artist(Line2D([x_l + 0.50, x_l + 0.545], [ly + 0.006, ly + 0.006], color=MUTED, lw=1.0))
    fig.text(x_l + 0.56, ly, "ground truth (white)", fontsize=8.5, color=INK2)
    fig.add_artist(Line2D([x_l, x_r], [0.205, 0.205], color=TRACK, lw=1))

    # per-organ Dice on this patient: ENet (red) -> U-Net (bold), organ colour as a small swatch
    fig.text(x_l, 0.172, "Dice on this patient", fontsize=11, color=INK2)
    for n, i in enumerate((2, 0, 3, 1)):  # trachea, esophagus, aorta, heart: biggest story first
        cx, cy = x_l + (n % 2) * 0.43, 0.125 - (n // 2) * 0.06
        fig.add_artist(Line2D([cx, cx + 0.025], [cy + 0.008, cy + 0.008], color=COLORS[i], lw=6,
                              solid_capstyle="butt"))
        fig.text(cx + 0.04, cy, NAMES[i], fontsize=10, color=INK)
        k = i + 1
        de, du = dice(pred["enet"] == k, gt == k), dice(pred["unet"] == k, gt == k)
        fig.text(cx + 0.04, cy - 0.027, f"{de:.3f}", fontsize=10, color=RED)
        fig.text(cx + 0.14, cy - 0.027, f"→  {du:.3f}", fontsize=10, color=INK, fontweight="bold")
    a.dest.mkdir(parents=True, exist_ok=True)
    save(fig, a.dest, f"unet2d_vs_enet_overlay_{pid}", tight=False)
    print(f"{pid}: coronal y={yc}, axial z={za}; wrote {a.dest}/unet2d_vs_enet_overlay_{pid}.png/.pdf")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--patient", default="Patient_14")
    p.add_argument("--root", type=Path, default=Path("results/cv5"))
    p.add_argument("--data", type=Path, default=Path("data/segthor_train_full/train"))
    p.add_argument("--dest", type=Path, default=Path("results/cv5/renders_pp/tables/UNET2d"))
    p.add_argument("--coronal", type=int, default=None, help="y index of the coronal plane (default: GT trachea)")
    p.add_argument("--axial", type=int, default=None, help="z index of the axial slice (default: most trachea the U-Net finds and ENet misses)")
    main(p.parse_args())
