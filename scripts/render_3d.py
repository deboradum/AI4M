#!/usr/bin/env python3
# MIT License
#
# Qualitative 3D comparison: a rotating side-by-side render of the ground truth
# and a prediction for one patient (the prediction alone when no GT is given,
# as for the test set), written as .mp4 and .gif. Surfaces are
# marching-cubes meshes in millimetres (spacing from the CT header, since the
# SegTHOR GT header carries a wrong identity affine).
#
#   python scripts/render_3d.py --pred results/X/eval/nii/Patient_37.nii.gz \
#       --gt data/segthor_train_full/train/Patient_37/GT.nii.gz \
#       --scan data/segthor_train_full/train/Patient_37/Patient_37.nii.gz \
#       --dest results/X/figures/Patient_37_3d --title "E_F07 + LCC"
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LightSource, to_rgba
from skimage.measure import marching_cubes

ORGANS = {1: ("esophagus", "#2ca02c"), 2: ("heart", "#d62728"),
          3: ("trachea", "#1f77b4"), 4: ("aorta", "#ff7f0e")}


def meshes(labels: np.ndarray, spacing: tuple[float, float, float], step: int) -> dict[int, tuple]:
    out = {}
    for k in ORGANS:
        mask = labels == k
        if mask.sum() < 10:
            continue
        verts, faces, _, _ = marching_cubes(np.pad(mask, 1).astype(np.float32), 0.5,
                                            spacing=spacing, step_size=step)
        out[k] = (verts - np.array(spacing), faces)  # undo the padding offset
    return out


def draw(ax, mesh: dict[int, tuple], title: str, bounds: np.ndarray) -> None:
    # One collection for all organs: matplotlib depth-sorts the triangles inside
    # a collection, but orders separate collections only by their centre, so
    # one collection per organ made whole organs pop in front/behind each
    # other as the view rotated.
    tris = np.concatenate([verts[faces] for verts, faces in mesh.values()])
    colors = np.concatenate([np.tile(to_rgba(ORGANS[k][1]), (len(faces), 1))
                             for k, (_, faces) in mesh.items()])
    poly = Poly3DCollection(tris, facecolors=colors, edgecolor="none", shade=True,
                            lightsource=LightSource(azdeg=225, altdeg=45))
    ax.add_collection3d(poly)
    lo, hi = bounds
    for setlim, a, b in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), lo, hi):
        setlim(a, b)
    # True millimetre proportions; the in-plane box is squared so it does not
    # wobble as the view rotates.
    ext = hi - lo
    ax.set_box_aspect((max(ext[:2]), max(ext[:2]), ext[2]), zoom=1.15)
    ax.set_axis_off()
    ax.set_title(title, fontsize=12)


def save_gif(path: Path, frames: list[np.ndarray], fps: float) -> None:
    """GIF that plays smoothly in PowerPoint and browsers: whole-millisecond delays (imageio >= 2.28
    reads `duration` in ms, so passing seconds wrote 0 ms frames and PowerPoint fell back to its own
    timing), one palette shared by every frame (no flicker), infinite loop, no frame optimisation."""
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    # global palette from a strip of evenly spaced frames, then every frame mapped to it
    sample = imgs[:: max(1, len(imgs) // 8)]
    strip = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
    for i, im in enumerate(sample):
        strip.paste(im, (0, i * im.height))
    pal = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in imgs]
    q[0].save(path, save_all=True, append_images=q[1:], duration=int(round(1000 / fps)), loop=0,
              optimize=False, disposal=1)


def main(args: argparse.Namespace) -> None:
    spacing = tuple(float(z) for z in nib.load(str(args.scan)).header.get_zooms()[:3])
    pred = np.asarray(nib.load(str(args.pred)).dataobj).astype(np.uint8)
    panels = [(meshes(pred, spacing, args.step), args.title)]
    if args.gt is not None:  # the test set has no GT: prediction alone
        gt = np.asarray(nib.load(str(args.gt)).dataobj).astype(np.uint8)
        assert gt.shape == pred.shape, (gt.shape, pred.shape)
        panels.insert(0, (meshes(gt, spacing, args.step), "Ground truth"))

    allv = np.concatenate([v for m, _ in panels for v, _ in m.values()])
    lo, hi = allv.min(0) - 5, allv.max(0) + 5
    side = (hi[:2] - lo[:2]).max()  # square in-plane box (see draw)
    mid = (hi[:2] + lo[:2]) / 2
    lo[:2], hi[:2] = mid - side / 2, mid + side / 2
    bounds = np.stack([lo, hi])

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for az in np.linspace(0, 360, args.frames, endpoint=False):
        fig = plt.figure(figsize=(4.5 * len(panels), 7), dpi=args.dpi)
        fig.subplots_adjust(left=0, right=1, bottom=0.06, top=0.9, wspace=0)
        for i, (mesh, title) in enumerate(panels):
            ax = fig.add_subplot(1, len(panels), i + 1, projection="3d")
            draw(ax, mesh, title, bounds)
            ax.view_init(elev=args.elev, azim=az)
        handles = [plt.Line2D([], [], color=c, lw=6) for _, c in ORGANS.values()]
        fig.legend(handles, [n for n, _ in ORGANS.values()], loc="lower center", ncol=2 * len(panels), frameon=False)
        fig.suptitle(args.patient or args.pred.name.split(".")[0], fontsize=13)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        h, w = frame.shape[:2]
        frames.append(frame[:h - h % 2, :w - w % 2].copy())  # H.264 needs even sizes
        plt.close(fig)

    imageio.mimsave(args.dest.with_suffix(".mp4"), frames, fps=args.fps, macro_block_size=1)
    save_gif(args.dest.with_suffix(".gif"), frames[::2], args.fps / 2)
    print(f"Wrote {args.dest.with_suffix('.mp4')} and {args.dest.with_suffix('.gif')} ({len(frames)} frames)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Rotating 3D render, ground truth vs prediction")
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--gt", type=Path, default=None, help="Omit (e.g. test set) to render the prediction alone")
    p.add_argument("--scan", type=Path, required=True, help="CT, for the voxel spacing")
    p.add_argument("--dest", type=Path, required=True, help="Output path without extension")
    p.add_argument("--title", default="Prediction")
    p.add_argument("--patient", default=None)
    p.add_argument("--step", type=int, default=2, help="marching-cubes step (mesh decimation)")
    p.add_argument("--frames", type=int, default=72)
    p.add_argument("--fps", type=int, default=18)
    p.add_argument("--elev", type=float, default=10)
    p.add_argument("--dpi", type=int, default=90)
    main(p.parse_args())
