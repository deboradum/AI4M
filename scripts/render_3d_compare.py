#!/usr/bin/env python3
# Rotating 3D render of several predictions side by side (optionally after the ground truth),
# reusing render_3d.py's meshes/draw so the look matches the earlier renders. 8 Oct 2026.
#
#   python scripts/render_3d_compare.py --scan <ct> --gt <GT> --preds a.nii.gz b.nii.gz \
#       --titles "E_F06 2D" "E_F05 2.5D" --dest out/Patient_15 --patient Patient_15
import sys
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from render_3d import ORGANS, meshes, draw, save_gif  # noqa: E402


def main(args: argparse.Namespace) -> None:
    spacing = tuple(float(z) for z in nib.load(str(args.scan)).header.get_zooms()[:3])
    vols = ([("Ground truth", args.gt)] if args.gt else []) + list(zip(args.titles, args.preds))
    panels = [(meshes(np.asarray(nib.load(str(p)).dataobj).astype(np.uint8), spacing, args.step), t) for t, p in vols]

    allv = np.concatenate([v for m, _ in panels for v, _ in m.values()])
    lo, hi = allv.min(0) - 5, allv.max(0) + 5
    side = (hi[:2] - lo[:2]).max()
    mid = (hi[:2] + lo[:2]) / 2
    lo[:2], hi[:2] = mid - side / 2, mid + side / 2
    bounds = np.stack([lo, hi])

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for az in np.linspace(0, 360, args.frames, endpoint=False):
        fig = plt.figure(figsize=(3.6 * len(panels), 6), dpi=args.dpi)
        fig.subplots_adjust(left=0, right=1, bottom=0.07, top=0.88, wspace=0)
        for i, (mesh, title) in enumerate(panels):
            ax = fig.add_subplot(1, len(panels), i + 1, projection="3d")
            draw(ax, mesh, title, bounds)
            ax.view_init(elev=args.elev, azim=az)
        handles = [plt.Line2D([], [], color=c, lw=6) for _, c in ORGANS.values()]
        fig.legend(handles, [n for n, _ in ORGANS.values()], loc="lower center", ncol=4, frameon=False)
        fig.suptitle(args.patient, fontsize=13)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        h, w = frame.shape[:2]
        frames.append(frame[:h - h % 2, :w - w % 2].copy())
        plt.close(fig)
    save_gif(args.dest.with_suffix(".gif"), frames, args.fps)
    imageio.mimsave(args.dest.with_suffix(".mp4"), frames, fps=args.fps, macro_block_size=1)
    print(f"Wrote {args.dest}.gif/.mp4 ({len(frames)} frames)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scan", type=Path, required=True)
    p.add_argument("--gt", type=Path, default=None)
    p.add_argument("--preds", type=Path, nargs="+", required=True)
    p.add_argument("--titles", nargs="+", required=True)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--patient", default="")
    p.add_argument("--step", type=int, default=2)
    p.add_argument("--frames", type=int, default=36)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--elev", type=float, default=10)
    p.add_argument("--dpi", type=int, default=70)
    main(p.parse_args())
