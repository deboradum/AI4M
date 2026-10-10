#!/usr/bin/env python3
# MIT License
#
# Rotating 3D "Dice map" of one organ for several models on the same patient: voxels both
# the model and the ground truth call organ are a translucent shell in the organ's colour,
# voxels the model adds are black, voxels it misses are magenta; Dice under each model's name. With --tolerance T (mm), errors within T of the other mask's border
# (border jitter, what NSD forgives) are drawn faint and only the errors beyond T solid, so
# a one-voxel disagreement along the whole surface does not paint the organ black.
# Reuses render_3d.py's mesh/axes conventions. 8 Oct 2026.
#
#   python scripts/render_dice_errors.py --patient Patient_38 --organ esophagus \
#       --preds a.nii.gz b.nii.gz --titles "ENet" "2D U-Net" --dest results/cv5/renders_dice/x
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
from scipy.ndimage import distance_transform_edt
from skimage.measure import marching_cubes

sys.path.insert(0, str(Path(__file__).parent))
from render_3d import ORGANS, save_gif  # noqa: E402

FP_COLOR, FN_COLOR = "#111111", "#ff00d4"
TP_ALPHA = 0.35
NEAR_ALPHA = 0.12
NAME2ID = {n: k for k, (n, _) in ORGANS.items()}


def surface(mask: np.ndarray, spacing: tuple, offset: np.ndarray, step: int):
    if mask.sum() < 3:
        return None
    verts, faces, _, _ = marching_cubes(np.pad(mask, 1).astype(np.float32), 0.5, spacing=spacing, step_size=step)
    return (verts - np.array(spacing) + offset)[faces]


def panel(gt: np.ndarray, pred: np.ndarray, spacing: tuple, offset: np.ndarray, color: str, step: int,
          tol: float) -> dict:
    tp, fp, fn = gt & pred, pred & ~gt, gt & ~pred
    if tol > 0:  # distance of each wrong voxel from the mask it should (FN) / should not (FP) extend
        fp_far = fp & (distance_transform_edt(~gt, sampling=spacing) > tol)
        fn_far = fn & (distance_transform_edt(~pred, sampling=spacing) > tol)
    else:
        fp_far, fn_far = fp, fn
    fp_near, fn_near = fp & ~fp_far, fn & ~fn_far
    tris, cols = [], []
    for mask, c, a, s in ((tp, color, TP_ALPHA, step), (fn_near, FN_COLOR, NEAR_ALPHA, 1),
                          (fp_near, FP_COLOR, NEAR_ALPHA, 1), (fn_far, FN_COLOR, 1.0, 1), (fp_far, FP_COLOR, 1.0, 1)):
        t = surface(mask, spacing, offset, s)
        if t is not None:
            tris.append(t)
            cols.append(np.tile(to_rgba(c, a), (len(t), 1)))
    ml = np.prod(spacing) / 1000
    n_tp, n_fp, n_fn = int(tp.sum()), int(fp.sum()), int(fn.sum())
    return {"tris": np.concatenate(tris), "cols": np.concatenate(cols),
            "tp": n_tp * ml, "fp": n_fp * ml, "fn": n_fn * ml,
            "fn_far": int(fn_far.sum()) * ml, "fp_far": int(fp_far.sum()) * ml,
            "dice": 2 * n_tp / max(2 * n_tp + n_fp + n_fn, 1)}


def render_frame(job):
    az, panels, titles, bounds, color, a = job
    n = len(panels)
    fig = plt.figure(figsize=(max(4.2 * n, 6.0), 6.4), dpi=a.dpi)
    fig.patch.set_facecolor("white")
    lo, hi = bounds
    ext = hi - lo
    for i, (p, title) in enumerate(zip(panels, titles)):
        ax = fig.add_axes([i / n, 0.1, 1 / n, 0.74], projection="3d")
        ax.add_collection3d(Poly3DCollection(p["tris"], facecolors=p["cols"], edgecolor="none", shade=True,
                                             lightsource=LightSource(azdeg=225, altdeg=45)))
        for setlim, l, h in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), lo, hi):
            setlim(l, h)
        ax.set_box_aspect((max(ext[:2]), max(ext[:2]), ext[2]), zoom=1.2)
        ax.set_axis_off()
        ax.view_init(elev=a.elev, azim=az)
        fig.text((i + 0.5) / n, 0.92, title, ha="center", fontsize=15)
        fig.text((i + 0.5) / n, 0.865, f"Dice {p['dice']:.2f}", ha="center", fontsize=13, color="#52514e")
    handles = [plt.Line2D([], [], color=to_rgba(color, 0.6), lw=8), plt.Line2D([], [], color=FN_COLOR, lw=8),
               plt.Line2D([], [], color=FP_COLOR, lw=8)]
    fig.legend(handles, ["correct", "missed", "wrongly added"], loc="lower center", ncol=3, frameon=False,
               fontsize=12, bbox_to_anchor=(0.5, 0.0))
    if a.tolerance > 0:
        fig.text(0.5, 0.065, f"faint = off by less than {a.tolerance:g} mm", ha="center", fontsize=10, color="#52514e")
    fig.suptitle(a.organ.capitalize(), fontsize=13, y=0.995, color="#52514e")
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    h, w = frame.shape[:2]
    return frame[:h - h % 2, :w - w % 2].copy()


def write(frames: list, dest: Path, a: argparse.Namespace) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(dest.with_suffix(".mp4"), frames, fps=a.fps, macro_block_size=1)
    save_gif(dest.with_suffix(".gif"), frames[::2], a.fps / 2)
    imageio.imwrite(dest.with_suffix(".png"), frames[a.frames // 8])


def main(a: argparse.Namespace) -> None:
    k = NAME2ID[a.organ]
    color = ORGANS[k][1]
    spacing = tuple(float(z) for z in nib.load(str(a.scan)).header.get_zooms()[:3])
    gt = np.asarray(nib.load(str(a.gt)).dataobj) == k
    preds = [np.asarray(nib.load(str(p)).dataobj) == k for p in a.preds]
    for p in preds:
        assert p.shape == gt.shape, (p.shape, gt.shape)

    # Crop to the organ in any of the volumes (meshes stay in mm of the full scan)
    union = gt.copy()
    for p in preds:
        union |= p
    idx = np.argwhere(union)
    c0, c1 = np.maximum(idx.min(0) - 2, 0), idx.max(0) + 3
    sl = tuple(slice(s, e) for s, e in zip(c0, c1))
    offset = c0 * np.array(spacing)
    panels = [panel(gt[sl], p[sl], spacing, offset, color, a.step, a.tolerance) for p in preds]

    allv = np.concatenate([p["tris"].reshape(-1, 3) for p in panels])
    lo, hi = allv.min(0) - 3, allv.max(0) + 3
    side = (hi[:2] - lo[:2]).max()
    mid = (hi[:2] + lo[:2]) / 2
    lo[:2], hi[:2] = mid - side / 2, mid + side / 2
    bounds = np.stack([lo, hi])

    azs = np.linspace(0, 360, a.frames, endpoint=False)
    # All models side by side, then (--single) each model alone on the same box and view
    groups = [(panels, a.titles, a.dest)]
    if a.single:
        groups += [([p], [t], a.dest.parent / f"{a.dest.name}_{slug}") for p, t, slug in zip(panels, a.titles, a.slugs)]
    with Pool(a.workers) as pool:
        for ps, ts, dest in groups:
            write(pool.map(render_frame, [(az, ps, ts, bounds, color, a) for az in azs]), dest, a)
            print(f"Wrote {dest}.mp4/.gif/.png")
    print("Dice " + ", ".join(f"{t} {p['dice']:.3f}" for t, p in zip(a.titles, panels)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Rotating 3D Dice map: TP / FN / FP surfaces of one organ per model")
    p.add_argument("--patient", required=True)
    p.add_argument("--organ", required=True, choices=list(NAME2ID))
    p.add_argument("--preds", type=Path, nargs="+", required=True)
    p.add_argument("--titles", nargs="+", required=True)
    p.add_argument("--gt", type=Path, default=None, help="default: data/segthor_train_full/train/<patient>/GT.nii.gz")
    p.add_argument("--scan", type=Path, default=None, help="CT, for the voxel spacing (default: next to the GT)")
    p.add_argument("--dest", type=Path, required=True, help="Output path without extension")
    p.add_argument("--step", type=int, default=1, help="marching-cubes step for the TP shell")
    p.add_argument("--frames", type=int, default=72)
    p.add_argument("--fps", type=int, default=18)
    p.add_argument("--elev", type=float, default=12)
    p.add_argument("--dpi", type=int, default=90)
    p.add_argument("--single", action="store_true", help="Also write one file per model")
    p.add_argument("--slugs", nargs="+", default=None, help="File suffix per model for --single (default: titles)")
    p.add_argument("--tolerance", type=float, default=0.0,
                   help="mm; errors within it are drawn faint (0 = every error solid, the plain Dice map)")
    p.add_argument("--workers", type=int, default=24)
    a = p.parse_args()
    root = Path("data/segthor_train_full/train") / a.patient
    a.gt = a.gt or root / "GT.nii.gz"
    a.scan = a.scan or root / f"{a.patient}.nii.gz"
    assert len(a.titles) == len(a.preds)
    a.slugs = a.slugs or [t.lower().replace(" ", "_").replace(".", "") for t in a.titles]
    main(a)
