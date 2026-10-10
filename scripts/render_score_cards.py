#!/usr/bin/env python3
# MIT License
#
# One 16:9 slide card per model: a rotating 3D render of the model's post-processed prediction
# for one held-out patient on the left, the model's pooled 5-fold CV scores on the right
# (same pooling as scripts/cv5_pooled.py: per organ = mean over patients; mean = mean over
# patients of the per-patient organ mean). All models share one bounding box and rotation.
# Writes <dest>/<slug>.mp4 / .gif / .png per model. 8 Oct 2026.
#
#   python scripts/render_score_cards.py --patient Patient_38 --dest results/cv5/score_cards
import sys
import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import nibabel as nib
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LightSource, to_rgba
from matplotlib.patches import Ellipse, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).parent))
from render_3d import ORGANS, meshes  # noqa: E402

# (run, slug, name, description)
MODELS = [("E_F11_enet_ce_dice", "enet", "ENet", "Baseline · 2D, one slice · 0.28 M parameters"),
          ("E_F06_unet2d_adamw_cosine_150ep", "unet2d", "2D U-Net", "One slice · 7.8 M parameters"),
          ("E_F05_unet25d_adamw_cosine_150ep", "unet25d", "2.5D U-Net", "Three adjacent slices · 7.8 M parameters"),
          ("E_F07_unet3d_adamw_cosine", "unet3d", "3D U-Net", "Volumetric patches · 16.5 M parameters")]
THEMES = {
    "dark": {"bg": "#0f1115", "panel": "#181b21", "ink": "#f4f4f2", "ink2": "#a9aab0", "rule": "#2a2e36",
             "accent": "#eb6834"},
    "light": {"bg": "#ffffff", "panel": "#f4f3ef", "ink": "#0b0b0b", "ink2": "#52514e", "rule": "#e1dfd9",
              "accent": "#d4541f"},
}
FONT = "Lato"
# Per-model accent = the model's colour in plot_boundary_curves.py (baseline grey lightened on dark)
ACCENT = {"dark": {"enet": "#c9c7c0", "unet2d": "#3987e5", "unet25d": "#1baf7a", "unet3d": "#eb6834"},
          "light": {"enet": "#7a7873", "unet2d": "#2a78d6", "unet25d": "#13915f", "unet3d": "#d4541f"}}


def pooled_scores(root: Path, run: str) -> tuple[dict, list[int]]:
    """metric -> (per-organ means [4], overall mean), as cv5_pooled.py; plus the finished folds."""
    folds = [k for k in range(5) if (root / f"{run}_f{k}" / "eval/pp_final/metrics/nsd.npz").exists()]
    out = {}
    for m in ("dice", "iou", "hd95", "assd", "nsd"):
        d = {}
        for k in folds:
            npz = np.load(root / f"{run}_f{k}" / "eval/pp_final/metrics" / f"{m}.npz")
            d.update({p: npz[p][1:].astype(float) for p in npz.files})
        arr = np.stack(list(d.values()))
        out[m] = (np.nanmean(arr, 0), float(np.nanmean(arr, 1).mean()), len(d))
    return out, folds


def draw_card(job):
    az, mesh, bounds, model, scores, n_pat, patient, a = job
    th = THEMES[a.theme]
    _, slug, name, desc = model
    accent = ACCENT[a.theme][slug]
    W, H = 16, 9
    fig = plt.figure(figsize=(W, H), dpi=a.dpi)
    fig.patch.set_facecolor(th["bg"])

    # --- left: rotating prediction ---------------------------------------------------------
    ax = fig.add_axes([0.0, 0.06, 0.52, 0.9], projection="3d")
    ax.set_facecolor(th["bg"])
    tris = np.concatenate([v[f] for v, f in mesh.values()])
    cols = np.concatenate([np.tile(to_rgba(ORGANS[k][1]), (len(f), 1)) for k, (_, f) in mesh.items()])
    ax.add_collection3d(Poly3DCollection(tris, facecolors=cols, edgecolor="none", shade=True,
                                         lightsource=LightSource(azdeg=225, altdeg=45)))
    lo, hi = bounds
    for setlim, l, h in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), lo, hi):
        setlim(l, h)
    ext = hi - lo
    ax.set_box_aspect((max(ext[:2]), max(ext[:2]), ext[2]), zoom=1.25)
    ax.set_axis_off()
    ax.view_init(elev=a.elev, azim=az)
    fig.text(0.26, 0.045, f"{patient} · held-out prediction, post-processed", ha="center", fontsize=12,
             color=th["ink2"], family=FONT)

    # --- right: scores ----------------------------------------------------------------------
    x0 = 0.555
    fig.add_artist(FancyBboxPatch((x0 - 0.015, 0.06), 0.445, 0.88, boxstyle="round,pad=0,rounding_size=0.012",
                                  transform=fig.transFigure, facecolor=th["panel"], edgecolor="none", zorder=0))
    fig.text(x0 + 0.01, 0.855, name, fontsize=50, color=th["ink"], family=FONT, weight="black", va="center")
    fig.text(x0 + 0.012, 0.78, desc, fontsize=15, color=th["ink2"], family=FONT, va="center")

    dice_org, dice_mean, _ = scores["dice"]
    fig.text(x0 + 0.01, 0.655, f"{dice_mean:.3f}", fontsize=74, color=accent, family=FONT, weight="black",
             va="center")
    fig.text(x0 + 0.215, 0.675, "mean Dice", fontsize=19, color=th["ink"], family=FONT, weight="bold", va="center")
    fig.text(x0 + 0.215, 0.633, "higher is better", fontsize=12.5, color=th["ink2"], family=FONT, va="center")

    # per-organ table
    cx = [x0 + 0.012, x0 + 0.205, x0 + 0.295, x0 + 0.385]  # organ, Dice, HD95, NSD (right-aligned numbers)
    y = 0.52
    for x, lab, ha in zip(cx, ["Organ", "Dice", "HD95 (mm)", "NSD"], ["left", "right", "right", "right"]):
        fig.text(x, y, lab, fontsize=13, color=th["ink2"], family=FONT, weight="bold", ha=ha, va="center")
    fig.add_artist(plt.Line2D([x0 + 0.01, x0 + 0.405], [y - 0.028, y - 0.028], transform=fig.transFigure,
                              color=th["rule"], lw=1.2))
    for i, k in enumerate(sorted(ORGANS)):
        yy = y - 0.075 - i * 0.062
        organ, color = ORGANS[k]
        fig.add_artist(Ellipse((x0 + 0.02, yy), 0.015 * H / W, 0.015, transform=fig.transFigure, color=color))
        fig.text(x0 + 0.036, yy, organ.capitalize(), fontsize=17, color=th["ink"], family=FONT, va="center")
        vals = [f"{scores['dice'][0][i]:.3f}", f"{scores['hd95'][0][i]:.1f}", f"{scores['nsd'][0][i]:.3f}"]
        for x, v in zip(cx[1:], vals):
            fig.text(x, yy, v, fontsize=17, color=th["ink"], family=FONT, ha="right", va="center")

    # mean strip
    ys = 0.165
    fig.add_artist(plt.Line2D([x0 + 0.01, x0 + 0.405], [ys + 0.06, ys + 0.06], transform=fig.transFigure,
                              color=th["rule"], lw=1.2))
    strip = [("IoU", f"{scores['iou'][1]:.3f}"), ("HD95", f"{scores['hd95'][1]:.1f} mm"),
             ("ASSD", f"{scores['assd'][1]:.2f} mm"), ("NSD 2 mm", f"{scores['nsd'][1]:.3f}")]
    for j, (lab, v) in enumerate(strip):
        xx = x0 + 0.012 + j * 0.1
        fig.text(xx, ys + 0.018, v, fontsize=19, color=th["ink"], family=FONT, weight="bold", va="center")
        fig.text(xx, ys - 0.025, f"mean {lab}", fontsize=12, color=th["ink2"], family=FONT, va="center")
    fig.text(x0 + 0.012, 0.09, f"5-fold cross-validation · {n_pat} patients · final-epoch weights · post-processed",
             fontsize=11.5, color=th["ink2"], family=FONT, va="center")

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    h, w = frame.shape[:2]
    return frame[:h - h % 2, :w - w % 2].copy()


def draw_bare(job):
    """Just the rotating prediction on white, square, no text (for building slides by hand)."""
    az, mesh, bounds, a = job
    fig = plt.figure(figsize=(9, 9), dpi=a.dpi)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    ax.set_facecolor("white")
    tris = np.concatenate([v[f] for v, f in mesh.values()])
    cols = np.concatenate([np.tile(to_rgba(ORGANS[k][1]), (len(f), 1)) for k, (_, f) in mesh.items()])
    ax.add_collection3d(Poly3DCollection(tris, facecolors=cols, edgecolor="none", shade=True,
                                         lightsource=LightSource(azdeg=225, altdeg=45)))
    lo, hi = bounds
    for setlim, l, h in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), lo, hi):
        setlim(l, h)
    ext = hi - lo
    ax.set_box_aspect((max(ext[:2]), max(ext[:2]), ext[2]), zoom=1.1)
    ax.set_axis_off()
    ax.view_init(elev=a.elev, azim=az)
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    h, w = frame.shape[:2]
    return frame[:h - h % 2, :w - w % 2].copy()


def main(a: argparse.Namespace) -> None:
    models = [m for m in MODELS if a.only is None or m[1] in a.only]
    scan = Path(f"data/segthor_train_full/train/{a.patient}/{a.patient}.nii.gz")
    spacing = tuple(float(z) for z in nib.load(str(scan)).header.get_zooms()[:3])

    def pred_path(run: str) -> Path:
        hits = [p for k in range(5) for p in [a.root / f"{run}_f{k}/eval/pp_final/nii/{a.patient}.nii.gz"] if p.exists()]
        assert len(hits) == 1, (run, a.patient, hits)
        return hits[0]

    # Shared box over every model's prediction, so the cards line up when shown one after another
    all_meshes = {m[1]: meshes(np.asarray(nib.load(str(pred_path(m[0]))).dataobj).astype(np.uint8), spacing, a.step)
                  for m in MODELS}
    allv = np.concatenate([v for ms in all_meshes.values() for v, _ in ms.values()])
    lo, hi = allv.min(0) - 5, allv.max(0) + 5
    side = (hi[:2] - lo[:2]).max()
    mid = (hi[:2] + lo[:2]) / 2
    lo[:2], hi[:2] = mid - side / 2, mid + side / 2
    bounds = np.stack([lo, hi])

    a.dest.mkdir(parents=True, exist_ok=True)
    azs = np.linspace(0, 360, a.frames, endpoint=False)
    with Pool(a.workers) as pool:
        for m in models:
            if a.bare:
                frames = pool.map(draw_bare, [(az, all_meshes[m[1]], bounds, a) for az in azs])
                dest = a.dest / f"{a.patient}_{m[1]}"
                imageio.mimsave(dest.with_suffix(".mp4"), frames, fps=a.fps, macro_block_size=1, quality=9)
                imageio.mimsave(dest.with_suffix(".gif"), frames[::2], duration=2 / a.fps, loop=0)
                imageio.imwrite(dest.with_suffix(".png"), frames[a.frames // 8])
                print(f"Wrote {dest}.mp4/.gif/.png")
                continue
            scores, folds = pooled_scores(a.root, m[0])
            n_pat = scores["dice"][2]
            if len(folds) < 5:
                print(f"WARNING {m[2]}: only folds {folds} finished ({n_pat} patients)")
            frames = pool.map(draw_card, [(az, all_meshes[m[1]], bounds, m, scores, n_pat, a.patient, a) for az in azs])
            dest = a.dest / f"{m[1]}_{a.theme}"
            imageio.mimsave(dest.with_suffix(".mp4"), frames, fps=a.fps, macro_block_size=1, quality=9)
            imageio.mimsave(dest.with_suffix(".gif"), frames[::2], duration=2 / a.fps, loop=0)
            imageio.imwrite(dest.with_suffix(".png"), frames[a.frames // 8])
            print(f"Wrote {dest}.mp4/.gif/.png · {n_pat} patients · Dice "
                  + " ".join(f"{v:.3f}" for v in scores["dice"][0]) + f" mean {scores['dice'][1]:.3f}"
                  + f" · HD95 {scores['hd95'][1]:.1f} · NSD {scores['nsd'][1]:.3f} · IoU {scores['iou'][1]:.3f}"
                  + f" · ASSD {scores['assd'][1]:.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--patient", default="Patient_38")
    p.add_argument("--root", type=Path, default=Path("results/cv5"))
    p.add_argument("--dest", type=Path, default=Path("results/cv5/score_cards"))
    p.add_argument("--only", nargs="+", default=None, choices=[m[1] for m in MODELS])
    p.add_argument("--bare", action="store_true", help="Only the rotating render on white, no scores or text")
    p.add_argument("--theme", choices=list(THEMES), default="dark")
    p.add_argument("--step", type=int, default=2, help="marching-cubes step, 2 as in renders_pp")
    p.add_argument("--frames", type=int, default=96)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--elev", type=float, default=10)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--workers", type=int, default=48)
    main(p.parse_args())
