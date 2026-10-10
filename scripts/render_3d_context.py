#!/usr/bin/env python3
# MIT License
#
# Rotating 3D render of the input context of each model, same meshes, look and timing as
# scripts/render_3d.py: one patient's organs in three panels (2D, 2.5D, 3D U-Net). The part of
# the thorax one forward pass sees is drawn in organ colours, the rest as a faint grey ghost:
# 2D one 2.5 mm slice, 2.5D three slices (7.5 mm), 3D one 80-slice patch (200 mm). In-plane all
# three see the full 256 x 256 slice, so the context differs only along z. The organs are the
# reference labels, used only as anatomy: the figure is about input extent, not about a prediction.
#
#   python scripts/render_3d_context.py --patient Patient_14
import sys
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from skimage.measure import marching_cubes

sys.path.insert(0, str(Path(__file__).parent))
from render_3d import ORGANS, save_gif  # noqa: E402

Z_RES = 2.5                       # the models' slice spacing (data resampled to 1.0 x 1.0 x 2.5 mm)
PANELS = [("2D U-Net", "1 slice · 2.5 mm", 1), ("2.5D U-Net", "3 slices · 7.5 mm", 3),
          ("3D U-Net", "80-slice patch · 200 mm", 80)]
GHOST = (0.55, 0.55, 0.55, 0.10)  # organs outside the input
BOX = "#52514e"


def mesh(mask: np.ndarray, spacing: tuple, step: int) -> tuple | None:
    if mask.sum() < 10:
        return None
    verts, faces, _, _ = marching_cubes(np.pad(mask, 1).astype(np.float32), 0.5, spacing=spacing, step_size=step)
    return verts - np.array(spacing), faces


def panel_meshes(gt: np.ndarray, spacing: tuple, z_mm: np.ndarray, lo: float, hi: float, step: int):
    """[(verts, faces, rgba)] for every organ: inside the slab in colour, outside as a ghost."""
    inside = (z_mm >= lo) & (z_mm < hi)
    out = []
    for k, (_, col) in ORGANS.items():
        m = gt == k
        m_in, m_out = m.copy(), m.copy()
        m_in[..., ~inside] = False
        m_out[..., inside] = False
        thin = inside.sum() <= 4  # a few slices: full-resolution mesh, or the slab vanishes
        for part, rgba, st in ((m_in, to_rgba(col), 1 if thin else step), (m_out, GHOST, step)):
            r = mesh(part, spacing, st)
            if r is not None:
                out.append((*r, rgba))
    return out


def draw(ax, parts, bounds: np.ndarray, lo: float, hi: float, title: str, sub: str) -> None:
    tris = np.concatenate([v[f] for v, f, _ in parts])
    cols = np.concatenate([np.tile(c, (len(f), 1)) for _, f, c in parts])
    ax.add_collection3d(Poly3DCollection(tris, facecolors=cols, edgecolor="none", shade=True,
                                         lightsource=LightSource(azdeg=225, altdeg=45)))
    (x0, y0, z0), (x1, y1, z1) = bounds
    # the input slab / patch as a thin wire box, full in-plane extent
    rect = lambda z: [((x0, y0, z), (x1, y0, z)), ((x1, y0, z), (x1, y1, z)), ((x1, y1, z), (x0, y1, z)),
                      ((x0, y1, z), (x0, y0, z))]
    segs = rect(lo) + rect(hi)
    if hi - lo > 20:
        segs += [((x, y, lo), (x, y, hi)) for x in (x0, x1) for y in (y0, y1)]
    ax.add_collection3d(Line3DCollection(segs, colors=BOX, linewidths=0.6, alpha=0.6))
    for setlim, a, b in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), bounds[0], bounds[1]):
        setlim(a, b)
    ext = bounds[1] - bounds[0]
    ax.set_box_aspect((max(ext[:2]), max(ext[:2]), ext[2]), zoom=1.15)
    ax.set_axis_off()
    ax.set_title(title, fontsize=12)
    ax.text2D(0.5, 0.06, sub, transform=ax.transAxes, ha="center", fontsize=10.5, color="#52514e")


def main(a: argparse.Namespace) -> None:
    pid = a.patient
    ct = nib.load(str(a.data / pid / f"{pid}.nii.gz"))
    spacing = tuple(float(s) for s in ct.header.get_zooms()[:3])
    gt = np.asarray(nib.load(str(a.data / pid / "GT.nii.gz")).dataobj).astype(np.uint8)
    z_mm = np.arange(gt.shape[2]) * spacing[2] + spacing[2] / 2   # voxel centres, same frame as the meshes

    # centre of the input: middle of the esophagus (where it runs between heart, aorta and spine)
    eso_z = np.flatnonzero((gt == 1).any((0, 1)))
    c = a.center_mm if a.center_mm is not None else float(z_mm[int(np.median(eso_z))])

    # bounds from the organs, padded; in-plane box squared as in render_3d
    idx = np.argwhere(gt > 0)
    lo_b = idx.min(0) * np.array(spacing) - 5
    hi_b = (idx.max(0) + 1) * np.array(spacing) + 5
    side = (hi_b[:2] - lo_b[:2]).max()
    mid = (hi_b[:2] + lo_b[:2]) / 2
    lo_b[:2], hi_b[:2] = mid - side / 2, mid + side / 2

    panels = []
    for title, sub, n in PANELS:
        half = n * Z_RES / 2
        lo, hi = c - half, c + half
        if n > 3:  # the 3D patch slides to stay inside the scan, as a training patch would
            lo = min(max(lo, 0.0), z_mm[-1] + spacing[2] / 2 - 2 * half)
            hi = lo + 2 * half
        panels.append((panel_meshes(gt, spacing, z_mm, lo, hi, a.step), lo, hi, title, sub))
        lo_b[2], hi_b[2] = min(lo_b[2], lo - 2), max(hi_b[2], hi + 2)
    bounds = np.stack([lo_b, hi_b])

    a.dest.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for az in np.linspace(0, 360, a.frames, endpoint=False):
        fig = plt.figure(figsize=(4.5 * len(panels), 7), dpi=a.dpi)
        fig.subplots_adjust(left=0, right=1, bottom=0.06, top=0.88, wspace=0)
        for i, (parts, lo, hi, title, sub) in enumerate(panels):
            ax = fig.add_subplot(1, len(panels), i + 1, projection="3d")
            draw(ax, parts, bounds, lo, hi, title, sub)
            ax.view_init(elev=a.elev, azim=az)
        handles = [plt.Line2D([], [], color=col, lw=6) for _, col in ORGANS.values()]
        handles.append(plt.Line2D([], [], color=(0.55, 0.55, 0.55, 0.35), lw=6))
        fig.legend(handles, [n for n, _ in ORGANS.values()] + ["outside the input"], loc="lower center",
                   ncol=5, frameon=False)
        fig.suptitle(f"What one forward pass sees · {pid.replace('_', ' ')}", fontsize=13)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        h, w = frame.shape[:2]
        frames.append(frame[:h - h % 2, :w - w % 2].copy())
        plt.close(fig)
        if a.frames == 1:
            break

    if a.frames == 1:
        imageio.imwrite(a.dest.with_suffix(".png"), frames[0])
        print(f"Wrote {a.dest.with_suffix('.png')}")
        return
    imageio.imwrite(a.dest.with_suffix(".png"), frames[0])
    imageio.mimsave(a.dest.with_suffix(".mp4"), frames, fps=a.fps, macro_block_size=1)
    save_gif(a.dest.with_suffix(".gif"), frames[::2], a.fps / 2)
    print(f"Wrote {a.dest}.mp4/.gif/.png ({len(frames)} frames), centre {c:.0f} mm")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--patient", default="Patient_14")
    p.add_argument("--data", type=Path, default=Path("data/segthor_train_full/train"))
    p.add_argument("--dest", type=Path, default=None, help="output path without extension")
    p.add_argument("--center_mm", type=float, default=None, help="z centre of the input (default: mid-esophagus)")
    p.add_argument("--step", type=int, default=2)
    p.add_argument("--frames", type=int, default=72)
    p.add_argument("--fps", type=int, default=18)
    p.add_argument("--elev", type=float, default=10)
    p.add_argument("--dpi", type=int, default=90)
    a = p.parse_args()
    a.dest = a.dest or Path(f"results/cv5/renders_pp/tables/UNET3d/{a.patient}_context_2d_25d_3d")
    main(a)
