#!/usr/bin/env python3
# MIT License
#
# Rotating 3D render of where two models agree: ground truth on the left, on the right one merged
# volume where voxels both models give the same organ keep that organ's colour, voxels only model A
# labels are dark grey and voxels only model B labels are purple (A wins where they name different
# organs); disagreement within --tolerance mm of the agreed region (border jitter) is not drawn. Same meshes, look, frame size and timing as scripts/render_3d_compare.py. 9 Oct 2026.
#
#   python scripts/render_3d_overlap.py --patient Patient_14 \
#       --a results/cv5/E_F06_unet2d_adamw_cosine_150ep_f1/eval/nii/Patient_14.nii.gz \
#       --b results/cv5/E_F11_enet_ce_dice_f1/eval/nii/Patient_14.nii.gz \
#       --name_a "2D U-Net" --name_b ENet --dest results/cv5/renders_pp/tables/UNET2d/Patient_14_f1_unet2d_vs_enet
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
import render_3d  # noqa: E402
from render_3d import meshes, draw, save_gif  # noqa: E402
from scipy.ndimage import distance_transform_edt  # noqa: E402

ONLY_A, ONLY_B = 5, 6
ORGAN_KEYS = (1, 2, 3, 4)


def merged(a: np.ndarray, b: np.ndarray, spacing: tuple, tol: float) -> np.ndarray:
    """Agreement in organ colours; disagreement only where it lies more than tol mm from any agreed
    voxel, so the two models' slightly different borders do not wrap every organ in a grey shell."""
    agree = (a == b) & (a > 0)
    out = np.where(agree, a, 0).astype(np.uint8)
    far = distance_transform_edt(~agree, sampling=spacing) > tol if tol > 0 else np.ones_like(agree)
    out[(a > 0) & (a != b) & far] = ONLY_A
    out[(b > 0) & (a == 0) & far] = ONLY_B
    return out


def main(args: argparse.Namespace) -> None:
    # draw()/meshes() colour by render_3d.ORGANS: add the two disagreement classes
    render_3d.ORGANS[ONLY_A] = (f"only {args.name_a}", "#3a3936")
    render_3d.ORGANS[ONLY_B] = (f"only {args.name_b}", "#9467bd")
    spacing = tuple(float(z) for z in nib.load(str(args.scan)).header.get_zooms()[:3])
    gt = np.asarray(nib.load(str(args.gt)).dataobj).astype(np.uint8)
    a = np.asarray(nib.load(str(args.a)).dataobj).astype(np.uint8)
    b = np.asarray(nib.load(str(args.b)).dataobj).astype(np.uint8)
    m = merged(a, b, spacing, args.tolerance)
    for k, (name, _) in render_3d.ORGANS.items():
        print(f"  {name}: {(m == k).sum():,} voxels")
    panels = [(meshes(gt, spacing, args.step), "Ground truth"),
              (meshes(m, spacing, args.step), f"{args.name_a} vs {args.name_b}")]

    allv = np.concatenate([v for mesh, _ in panels for v, _ in mesh.values()])
    lo, hi = allv.min(0) - 5, allv.max(0) + 5
    side = (hi[:2] - lo[:2]).max()
    mid = (hi[:2] + lo[:2]) / 2
    lo[:2], hi[:2] = mid - side / 2, mid + side / 2
    bounds = np.stack([lo, hi])

    keys = list(ORGAN_KEYS) + [ONLY_A, ONLY_B]
    handles = [plt.Line2D([], [], color=render_3d.ORGANS[k][1], lw=6) for k in keys]
    labels = [render_3d.ORGANS[k][0] for k in ORGAN_KEYS] + [render_3d.ORGANS[ONLY_A][0], render_3d.ORGANS[ONLY_B][0]]
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for az in np.linspace(0, 360, args.frames, endpoint=False):
        fig = plt.figure(figsize=(3.6 * len(panels), 6), dpi=args.dpi)
        fig.subplots_adjust(left=0, right=1, bottom=0.11, top=0.88, wspace=0)
        for i, (mesh, title) in enumerate(panels):
            ax = fig.add_subplot(1, len(panels), i + 1, projection="3d")
            draw(ax, mesh, title, bounds)
            ax.view_init(elev=args.elev, azim=az)
        # organs (= both agree) on the first row, the two disagreement colours on the second
        fig.legend(handles[:4], labels[:4], loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.045))
        fig.legend(handles[4:], [f"{l} (> {args.tolerance:g} mm)" for l in labels[4:]], loc="lower center", ncol=2,
                   frameon=False, bbox_to_anchor=(0.5, 0.0))
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
    p.add_argument("--scan", type=Path, default=None, help="CT for the voxel spacing (default: from --patient)")
    p.add_argument("--gt", type=Path, default=None)
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)
    p.add_argument("--name_a", default="2D U-Net")
    p.add_argument("--name_b", default="ENet")
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--patient", default="")
    p.add_argument("--tolerance", type=float, default=3.0,
                   help="mm; disagreement closer than this to the agreed region is not drawn (border jitter)")
    p.add_argument("--step", type=int, default=2)
    p.add_argument("--frames", type=int, default=36)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--elev", type=float, default=10)
    p.add_argument("--dpi", type=int, default=70)
    args = p.parse_args()
    pid = args.patient.split()[0] if args.patient else None
    data = Path("data/segthor_train_full/train")
    args.scan = args.scan or data / pid / f"{pid}.nii.gz"
    args.gt = args.gt or data / pid / "GT.nii.gz"
    main(args)
