#!/usr/bin/env python3
# Wide, short rotating 3D render for a slide band: each volume lies on its side (head left,
# feet right) and spins about the body axis; panels sit side by side. Reuses render_3d.py's
# meshes/colours/GIF writer so the look matches the other renders. 10 Oct 2026.
#
#   python scripts/render_3d_band.py --gt GT.nii.gz --preds a.nii.gz b.nii.gz \
#       --titles "Ground truth" "3D U-Net" "3D U-Net + Skeleton Recall" --classes 1 3 4 \
#       --dest out/Patient_37_band
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, str(Path(__file__).parent))
from render_3d import ORGANS, meshes  # noqa: E402


def save_gif_white(path: Path, frames: list[np.ndarray], fps: float) -> None:
    """render_3d.save_gif, but with the palette entry nearest to white set to pure white: the
    median-cut palette rounds the background to about 251, a visible grey box on a white slide."""
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    sample = imgs[:: max(1, len(imgs) // 8)]
    strip = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
    for i, im in enumerate(sample):
        strip.paste(im, (0, i * im.height))
    pal = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    entries = np.array(pal.getpalette()[:765]).reshape(-1, 3)
    white = int(np.argmin(((255 - entries) ** 2).sum(1)))
    entries[white] = 255
    pal.putpalette(entries.flatten().tolist())
    q = []
    for f, im in zip(frames, imgs):
        # Pillow's palette lookup is approximate and can still send white to a near-white entry
        idx = np.asarray(im.quantize(palette=pal, dither=Image.Dither.NONE)).copy()
        idx[(f == 255).all(2)] = white
        out = Image.fromarray(idx, "P")
        out.putpalette(entries.flatten().tolist())
        q.append(out)
    q[0].save(path, save_all=True, append_images=q[1:], duration=int(round(1000 / fps)), loop=0,
              optimize=False, disposal=1)


def main(args: argparse.Namespace) -> None:
    spacing = tuple(float(z) for z in nib.load(str(args.vols[0])).header.get_zooms()[:3])
    panels = []
    for p in args.vols:
        lab = np.asarray(nib.load(str(p)).dataobj).astype(np.uint8)
        lab[~np.isin(lab, args.classes)] = 0
        panels.append(meshes(lab, spacing, args.step))

    # Shared frame: body axis (z, superior = high z) becomes horizontal, head on the left;
    # the in-plane axes rotate about the centre of everything drawn.
    allv = np.concatenate([v for m in panels for v, _ in m.values()])
    centre = (allv.min(0) + allv.max(0)) / 2
    radius = np.linalg.norm(allv[:, :2] - centre[:2], axis=1).max() + 3
    zlo, zhi = allv[:, 2].min() - 3, allv[:, 2].max() + 3

    def place(verts: np.ndarray, angle: float) -> np.ndarray:
        x, y = verts[:, 0] - centre[0], verts[:, 1] - centre[1]
        c, s = np.cos(angle), np.sin(angle)
        return np.stack([-(verts[:, 2] - centre[2]), c * x - s * y, -(s * x + c * y)], 1)

    # Render each panel on its own, then crop the white margin shared by every panel and
    # frame (same crop everywhere keeps the scale identical) and stitch with a title strip.
    n = len(panels)
    width_mm, height_mm = zhi - zlo, 2 * radius
    fig_w = args.width / args.dpi / n * 1.6  # oversize; the crop removes the margin
    fig_h = fig_w * height_mm / width_mm
    raw = [[] for _ in range(n)]
    for angle in np.linspace(0, 2 * np.pi, args.frames, endpoint=False):
        for i, mesh in enumerate(panels):
            fig = plt.figure(figsize=(fig_w, fig_h), dpi=args.dpi)
            ax = fig.add_axes((0, 0, 1, 1), projection="3d")
            tris = np.concatenate([place(v, angle)[f] for v, f in mesh.values()])
            colors = np.concatenate([np.tile(to_rgba(ORGANS[k][1]), (len(f), 1)) for k, (_, f) in mesh.items()])
            ax.add_collection3d(Poly3DCollection(tris, facecolors=colors, edgecolor="none", shade=True,
                                                 lightsource=LightSource(azdeg=225, altdeg=45)))
            hx = width_mm / 2
            ax.set_xlim(-hx, hx), ax.set_ylim(-radius, radius), ax.set_zlim(-radius, radius)
            ax.set_box_aspect((width_mm, height_mm, height_mm), zoom=1.4)
            ax.set_proj_type("ortho")
            ax.view_init(elev=0, azim=-90)
            ax.set_axis_off()
            fig.canvas.draw()
            raw[i].append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
            plt.close(fig)
    ink = np.any([np.any(f < 245, axis=2) for fr in raw for f in fr], axis=0)
    rows, cols = np.nonzero(ink.any(1))[0], np.nonzero(ink.any(0))[0]
    pad = 6
    r0, r1 = max(rows[0] - pad, 0), rows[-1] + pad + 1
    c0, c1 = max(cols[0] - pad, 0), cols[-1] + pad + 1

    from PIL import Image, ImageDraw, ImageFont
    from matplotlib import font_manager
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), args.fontsize)
    pw, ph, gap, th = c1 - c0, r1 - r0, args.gap, int(args.fontsize * 1.7)
    W, H = n * pw + (n - 1) * gap, th + ph
    frames = []
    for j in range(args.frames):
        canvas = Image.new("RGB", (W, H), "white")
        draw = ImageDraw.Draw(canvas)
        for i, title in enumerate(args.titles):
            x = i * (pw + gap)
            canvas.paste(Image.fromarray(raw[i][j][r0:r1, c0:c1]), (x, th))
            draw.text((x + pw / 2, th / 2), title, fill="black", font=font, anchor="mm")
            if i:  # thin divider, so stray pieces read as belonging to their own panel
                draw.line([(x - gap / 2, th * 0.4), (x - gap / 2, H - th * 0.4)], fill="#cccccc", width=2)
        if args.width and W != args.width:
            canvas = canvas.resize((args.width, round(H * args.width / W)), Image.LANCZOS)
        f = np.asarray(canvas)
        frames.append(f[:f.shape[0] - f.shape[0] % 2, :f.shape[1] - f.shape[1] % 2].copy())
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    save_gif_white(args.dest.with_suffix(".gif"), frames, args.fps)
    imageio.mimsave(args.dest.with_suffix(".mp4"), frames, fps=args.fps, macro_block_size=1)
    imageio.imwrite(args.dest.with_suffix(".png"), frames[0])
    print(f"Wrote {args.dest}.gif/.mp4/.png ({len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--preds", type=Path, nargs="+", required=True)
    p.add_argument("--titles", nargs="+", required=True)
    p.add_argument("--classes", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--step", type=int, default=2)
    p.add_argument("--frames", type=int, default=36)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--width", type=int, default=1800, help="output width in pixels")
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--fontsize", type=int, default=30)
    p.add_argument("--gap", type=int, default=40, help="pixels between panels")
    a = p.parse_args()
    a.vols = [a.gt] + a.preds
    assert len(a.titles) == len(a.vols), "one title per volume (ground truth first)"
    main(a)
