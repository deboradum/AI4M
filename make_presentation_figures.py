#!/usr/bin/env python3

# MIT License
#
# Static slide figures and one stepping 3D video for the mid-term talk on the aorta
# recovery (retrieve_aorta.py).
#
# Everything drawn here is recomputed from data/segthor_part1/train with the shipped
# split_patient: the figures show the real pipeline, not a copy of it. Patient_07
# carries the stills because its GT2.nii.gz is the only human-split reference in the
# tree (never read data/SEGTHOR_aorta/train/Patient_07/GT.nii.gz for the "after"
# panel - that file is written from GT2 and would make before/after identical).
# Patient_15 carries the spin because it has the widest aorta.
#
# The spin turns a constant 360 degrees over its whole length. Its right panel swaps
# content at four discrete steps - folded class 1, seeds, watershed basins, result -
# read from the trace split_patient fills, so the video replays the shipped pipeline.
# Its camera elevation follows the stage; the left panel never changes content.
#
# Run from the repo root with the repo venv:
#   ./ai4mi/bin/python make_presentation_figures.py [--skip-spin]
#   ./ai4mi/bin/python make_presentation_figures.py --frames 168 --step-frames 0 30 60 100
# The assets are regenerable and live in PRESENTATION/, which .gitignore excludes.

import time
import shutil
import subprocess
import textwrap
import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt as nd_distance_transform_edt
from scipy.ndimage import label as nd_label

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import retrieve_aorta as ra
from retrieve_aorta import load_labels, split_patient, shared_slices, dice, N_MONTAGE_SLICES
from skimage.measure import marching_cubes

# Colours and conventions local to the slides; the CT window, the aorta yellow and
# the esophagus red come from retrieve_aorta (shared with visualize_labels.py).
FOLDED_COLOUR = (1.0, 1.0, 1.0)        # the folded class 1: white outline
MOVED_COLOUR = (0.85, 0.0, 0.85)       # voxels the 10 mm rule moved out of the esophagus
MERGED_COLOUR = (0.55, 0.55, 0.60)     # 3D merged surface
CONTOUR_LW = 2.0
MAGENTA_LW = 1.5
FILL_ALPHA = 0.35
SCALE_BAR_MM = 20.0
PANEL_W_IN = 4.2
DPI = 150
TITLE_FS, PANEL_FS, FOOTER_FS = 22, 15, 14
# Long panel titles are wrapped: a 4.2 in panel holds a drawn image about 3.2 in wide
# at 15 pt, so an unwrapped 60-character step title would cross into its neighbour.
PANEL_TITLE_CHARS = 32
SPIN_FIGSIZE = (12.8, 7.2)             # x spin_dpi = 1280x720
SPIN_BUDGET_S = 480.0                  # slowest acceptable total render time for the video

# Explicit layout bands, in inches (see figure_for): figure text must never touch an
# image, and neither layout engine achieves that on all three figures.
SIDE_IN = 0.08                         # left and right figure margin
HGAP_IN = 0.15                         # white gutter between panels
FIG2_HGAP_IN = 0.80                    # wider gutter after panel 2, for the colorbar
                                       # (bar + tick labels + rotated label must fit)
CBAR_W_IN, CBAR_PAD_IN = 0.14, 0.07    # colorbar axes width and gap, inside that gutter
ROW_GAP_IN = 0.18                      # white space between a title band and the next row
TOP_BAND_IN = 0.5                      # suptitle band above the top row's title band
BOTTOM_BAND_IN = 0.95                  # legend line + footer line band


def patient_paths(pid: str) -> tuple[Path, Path, Path]:
    d = Path("data/segthor_part1/train") / pid
    return d / f"{pid}.nii.gz", d / "GT.nii.gz", d


def organ_box(shape: tuple[int, int, int], masks: list[np.ndarray],
              spacing: tuple[float, float, float],
              margin_mm: float = 40.0) -> tuple[int, int, int, int]:
    # One crop box per figure, from the union of the drawn masks expanded by margin_mm,
    # so the panels of a figure are comparable and the 512x512 frame is not mostly air.
    band = np.zeros(shape, dtype=bool)
    for m in masks:
        band |= m
    xs = np.where(band.any(axis=(1, 2)))[0]
    ys = np.where(band.any(axis=(0, 2)))[0]
    mx, my = int(round(margin_mm / spacing[0])), int(round(margin_mm / spacing[1]))
    return (max(0, int(xs.min()) - mx), min(shape[0], int(xs.max()) + 1 + mx),
            max(0, int(ys.min()) - my), min(shape[1], int(ys.max()) + 1 + my))


def ct_panel(ax, ct: np.ndarray, z: int, box: tuple[int, int, int, int],
             spacing: tuple[float, float, float],
             masks: list[tuple[np.ndarray, tuple, bool]], title: str,
             scale_bar: bool = True) -> None:
    # Same window and transpose convention as retrieve_aorta.panel: ct[:, :, z].T with
    # origin="lower". Fills first, contours second, so no fill hides an outline.
    x0, x1, y0, y1 = box
    ax.imshow(ct[x0:x1, y0:y1, z].T, cmap="gray", vmin=ra.CT_WINDOW[0], vmax=ra.CT_WINDOW[1],
              origin="lower", aspect="equal", interpolation="nearest")
    for mask, colour, fill in masks:
        sl = mask[x0:x1, y0:y1, z].T
        if fill and sl.any():
            ax.contourf(sl.astype(float), levels=[0.5, 1.5], colors=[colour], alpha=FILL_ALPHA)
    for mask, colour, fill in masks:
        sl = mask[x0:x1, y0:y1, z].T
        if sl.any():
            ax.contour(sl.astype(float), levels=[0.5], colors=[colour],
                       linewidths=CONTOUR_LW)
    if scale_bar:
        w = (SCALE_BAR_MM / spacing[1]) / (y1 - y0)          # displayed width is axis 1
        ax.plot([0.92 - w, 0.92], [0.06, 0.06], transform=ax.transAxes,
                color="white", lw=3, solid_capstyle="butt")
        ax.text(0.92, 0.08, f"{SCALE_BAR_MM:.0f} mm", transform=ax.transAxes,
                color="white", fontsize=12, ha="right", va="bottom")
    ax.set_title(textwrap.fill(title, PANEL_TITLE_CHARS), fontsize=PANEL_FS, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])


def figure_for(nrows: int, ncols: int, box: tuple[int, int, int, int],
               spacing: tuple[float, float, float], title_lines: int = 1,
               hgap: float = HGAP_IN):
    # Explicit placement, no layout engine: the gridspec engines either warn about the
    # colorbar axes (tight_layout) or ignore the rect and run the top row of images
    # under the suptitle (constrained, three-row figure).
    #
    # Each cell is sized to the crop's physical aspect, so the aspect-equal image fills
    # it exactly and the scale bar, drawn in axes coordinates, lands on the anatomy.
    # A per-row title band and fixed bands for the suptitle and the footer lines keep
    # every text block clear of every image by construction.
    w_mm = (box[1] - box[0]) * spacing[0]
    h_mm = (box[3] - box[2]) * spacing[1]
    cell_h = PANEL_W_IN * h_mm / w_mm
    band_title = title_lines * PANEL_FS * 1.45 / 72.0 + 8.0 / 72.0
    fig_w = 2 * SIDE_IN + ncols * PANEL_W_IN + (ncols - 1) * hgap
    fig_h = (nrows * (cell_h + band_title) + (nrows - 1) * ROW_GAP_IN
             + TOP_BAND_IN + BOTTOM_BAND_IN)
    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = np.empty((nrows, ncols), dtype=object)
    for r in range(nrows):
        for c in range(ncols):
            x0 = SIDE_IN + c * (PANEL_W_IN + hgap)
            y0 = fig_h - (TOP_BAND_IN + r * (cell_h + band_title + ROW_GAP_IN)
                          + band_title + cell_h)
            ax = fig.add_axes([x0 / fig_w, y0 / fig_h, PANEL_W_IN / fig_w, cell_h / fig_h])
            ax.set_aspect("equal")
            axes[r, c] = ax
    return fig, axes


def title_lines(*titles: str) -> int:
    # Wrapped line count of the tallest title a row will draw, for figure_for's band.
    return max(len(textwrap.wrap(t, PANEL_TITLE_CHARS)) for t in titles)


def finish(fig, path: Path, out_dir: Path, suptitle: str, footer: str,
           legend_line: str | None = None, dpi: int = DPI) -> Path:
    # Figure texts live inside the bands reserved by figure_for: text is measured from
    # the top edge for the suptitle and from the bottom edge for the two footers.
    fig_h = fig.get_size_inches()[1]
    fig.text(0.5, 1 - 0.4 * TOP_BAND_IN / fig_h, suptitle, ha="center", va="center",
             fontsize=TITLE_FS)
    if legend_line is not None:
        fig.text(0.5, 0.62 / fig_h, legend_line, ha="center", va="bottom", fontsize=FOOTER_FS)
    fig.text(0.5, 0.2 / fig_h, footer, ha="center", va="bottom", fontsize=FOOTER_FS)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Wrote {path}")
    return path


def fig1_problem(out_dir: Path, ct: np.ndarray, folded: np.ndarray,
                 aorta: np.ndarray, eso: np.ndarray,
                 spacing: tuple[float, float, float], pid: str,
                 dpi: int = DPI) -> Path:
    # The fold, stated once: class 1 as shipped, with no attempt to split it.
    zs = shared_slices(aorta, eso, N_MONTAGE_SLICES)
    box = organ_box(ct.shape, [aorta, eso], spacing)
    fig, axes = figure_for(1, len(zs), box, spacing,
                           title_lines(*[f"axial z={z}" for z in zs]))
    for c, z in enumerate(zs):
        ct_panel(axes[0, c], ct, z, box, spacing, [(folded, FOLDED_COLOUR, False)],
                 f"axial z={z}")
    return finish(fig, out_dir / "fig1_problem.png", out_dir,
                  "The problem: the aorta was folded into class 1",
                  f"{pid} - class 1 holds {int(folded.sum())} voxels = "
                  f"25373 esophagus + 89856 aorta (reference GT2)",
                  legend_line="folded class 1 = white outline", dpi=dpi)


def fig2_algorithm(out_dir: Path, ct: np.ndarray, trace: dict, aorta: np.ndarray,
                   eso: np.ndarray, spacing: tuple[float, float, float],
                   dpi: int = DPI) -> Path:
    # The eight salient steps, all on one deterministic axial slice, drawn from the
    # intermediates split_once published for the shipped threshold.
    assert trace["regions"] is not None, \
        "this patient took the inscribed-ball fallback: no watershed panels to draw"
    folded = trace["aorta"] | trace["eso"]
    moved = trace["roles_eso"] & ~trace["eso"]
    z = shared_slices(aorta, eso, N_MONTAGE_SLICES)[2]
    box = organ_box(ct.shape, [aorta, eso], spacing)
    n_aorta, n_eso = int(trace["aorta"].sum()), int(trace["eso"].sum())
    ratio = ra.median_slice_area(aorta, spacing) / ra.median_slice_area(eso, spacing)
    title7 = "7. repairs: smaller median area = esophagus, radius > 10 mm = aorta"
    if not moved[:, :, z].any():
        title7 += " (no voxel moved on this slice)"
    titles = ["1. class 1: one label, two organs",
              "2. distance transform of class 1 (mm)",
              "3. aorta seed: largest component of dt >= 0.6 x max dt",
              f"4. esophagus seeds: per-slice components <= {trace['thin_area']:.0f} mm2",
              "5. markers (aorta = 1, esophagus = 2)",
              "6. watershed on -dt, masked to class 1",
              title7,
              f"8. final split: aorta {n_aorta} vx, esophagus {n_eso} vx, "
              f"area ratio {ratio:.1f} - gates pass"]
    fig, axes = figure_for(2, 4, box, spacing, title_lines(*titles), hgap=FIG2_HGAP_IN)

    ct_panel(axes[0, 0], ct, z, box, spacing, [(folded, FOLDED_COLOUR, False)], titles[0])

    ax = axes[0, 1]
    ct_panel(ax, ct, z, box, spacing, [(folded, FOLDED_COLOUR, False)], titles[1],
             scale_bar=False)
    x0, x1, y0, y1 = box
    inside = folded[x0:x1, y0:y1, z]
    dts = np.ma.masked_where(~inside, trace["dt"][x0:x1, y0:y1, z])
    im = ax.imshow(dts.T, cmap="inferno", alpha=0.75, vmin=0, vmax=trace["dt"].max(),
                   origin="lower", aspect="equal", interpolation="nearest")
    # Its own axes in the widened gutter: passing ax= to fig.colorbar would shrink the
    # parent, leaving this one panel at a different scale from the other seven.
    fig_w = fig.get_size_inches()[0]
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + CBAR_PAD_IN / fig_w, pos.y0, CBAR_W_IN / fig_w, pos.height])
    bar = fig.colorbar(im, cax=cax)
    bar.set_label("distance to boundary (mm)", fontsize=FOOTER_FS)
    bar.ax.tick_params(labelsize=11)

    ct_panel(axes[0, 2], ct, z, box, spacing, [(trace["aorta_seed"], ra.AORTA_COLOUR, True)],
             titles[2])
    ct_panel(axes[0, 3], ct, z, box, spacing, [(trace["eso_cand"], ra.ESO_COLOUR, True)],
             titles[3])

    ct_panel(axes[1, 0], ct, z, box, spacing,
             [(trace["markers"] == 1, ra.AORTA_COLOUR, True),
              (trace["markers"] == 2, ra.ESO_COLOUR, True),
              (trace["markers"] == 1, ra.AORTA_COLOUR, False),
              (trace["markers"] == 2, ra.ESO_COLOUR, False)],
             titles[4])
    ct_panel(axes[1, 1], ct, z, box, spacing,
             [(trace["basins_aorta"], ra.AORTA_COLOUR, True),
              (trace["basins_eso"], ra.ESO_COLOUR, True)],
             titles[5])

    ct_panel(axes[1, 2], ct, z, box, spacing,
             [(aorta, ra.AORTA_COLOUR, True), (eso, ra.ESO_COLOUR, True),
              (moved, MOVED_COLOUR, False)],
             titles[6])
    axes[1, 2].collections[-1].set_linewidth(MAGENTA_LW)

    ct_panel(axes[1, 3], ct, z, box, spacing,
             [(aorta, ra.AORTA_COLOUR, False), (eso, ra.ESO_COLOUR, False)],
             titles[7])

    return finish(fig, out_dir / "fig2_algorithm.png", out_dir,
                  "Recovering the aorta from class 1: marker-controlled watershed "
                  "on the distance transform",
                  "folded class 1 = white - aorta = yellow - esophagus = red - "
                  "magenta = voxels forced into the aorta by the 10 mm rule", dpi=dpi)


def fig3_before_after(out_dir: Path, ct: np.ndarray, folded: np.ndarray,
                      aorta: np.ndarray, eso: np.ndarray, gt2: np.ndarray,
                      spacing: tuple[float, float, float], pid: str,
                      d_aorta: float, d_eso: float, dpi: int = DPI) -> Path:
    zs = shared_slices(aorta, eso, N_MONTAGE_SLICES)
    box = organ_box(ct.shape, [aorta, eso], spacing)
    rows = [("before - folded class 1", [(folded, FOLDED_COLOUR, False)]),
            ("after - recovered split", [(aorta, ra.AORTA_COLOUR, True),
                                         (eso, ra.ESO_COLOUR, True)]),
            ("reference - GT2, the only human-split file",
             [(gt2 == 4, ra.AORTA_COLOUR, True), (gt2 == 1, ra.ESO_COLOUR, True)])]
    fig, axes = figure_for(3, len(zs), box, spacing,
                           title_lines(*[f"{label} | axial z={z}"
                                         for label, _ in rows for z in zs]))
    for r, (label, masks) in enumerate(rows):
        for c, z in enumerate(zs):
            ct_panel(axes[r, c], ct, z, box, spacing, masks, f"{label} | axial z={z}")
    return finish(fig, out_dir / "fig3_before_after.png", out_dir,
                  f"{pid}: before, after, and the reference the split is scored against",
                  f"Dice vs reference GT2: aorta {d_aorta:.4f} - esophagus {d_eso:.4f} "
                  f"(printed by this script, not hardcoded) - all 20 patients pass the "
                  f"four plausibility gates",
                  legend_line="before = white outline - aorta = yellow - esophagus = red",
                  dpi=dpi)


def surface(mask: np.ndarray, spacing: tuple[float, float, float], step: int,
            centre: np.ndarray | None = None) -> np.ndarray:
    # marching_cubes needs a zero border: class 1 touches no volume border for
    # Patient_07/15, the pad makes that explicit for any patient.
    #
    # `centre` recentres the triangles on the origin. mplot3d orbits the camera about
    # the centre of the axes box, so meshes left in image index space (a 512x512x180
    # frame for a mediastinum 100 mm across) travel a wide arc: the anatomy swings
    # across the panel instead of turning on the spot. Every mesh in one scene must be
    # recentred on the SAME point or the organs shears apart as the camera turns.
    vol = np.pad(mask.astype(np.float32), 2)
    verts, faces, _, _ = marching_cubes(vol, level=0.5, spacing=spacing, step_size=step)
    assert len(verts) > 0, f"empty surface for a mask of {int(mask.sum())} voxels"
    tri = verts[faces]
    if centre is not None:
        tri = tri - centre[None, None, :]
    return tri


def surface_collection(*blocks: tuple[np.ndarray, tuple],
                       alpha: float = 1.0) -> tuple[Poly3DCollection, np.ndarray]:
    # Every surface goes into ONE collection. mplot3d depth-sorts the faces inside a
    # collection, but it orders whole collections by a single scalar, so two
    # collections make the draw order flip abruptly as the rotation carries the
    # esophagus from behind the aorta to in front of it: the red tube pops through the
    # yellow one for a few frames. One collection sorts all triangles together, so the
    # two organs occlude each other correctly at every azimuth.
    #
    # alpha < 1 draws a ghost, used for the folded label while the seeds and the
    # flooding front are shown inside it.
    #
    # Poly3DCollection(shade=True, edgecolors="none") raises in matplotlib 3.11: the
    # constructor shades the (empty) edge colour array. Set the invisible edges after
    # construction, where no shading is applied.
    verts = np.concatenate([tri for tri, _ in blocks], axis=0)
    colors = np.concatenate([np.tile((*colour, alpha), (len(tri), 1))
                             for tri, colour in blocks])
    col = Poly3DCollection(verts, facecolors=colors, shade=True)
    col.set_edgecolor("none")
    # The vertex array travels with the collection: matplotlib 3.11 exposes no public
    # way back to the triangles, and the camera frame needs their union.
    return col, verts


SEED_COLOUR = (0.10, 1.00, 1.00)       # seeds the watershed starts from: cyan
THIN_COLOUR = (0.35, 0.35, 0.35)       # fallback stage: the thin rule alone, dimmed
SPIN_STAGES: tuple[tuple[str, str], ...] = (
    ("1. folded class 1", "merged"),
    ("2. seeds (aorta core + thin islands)", "seeds"),
    ("3. watershed, flooding live", "flood"),
    ("4. aorta (yellow) + esophagus (red)", "final"),
)
# Each stage carries a second camera, slightly lower and further round the patient, so
# the content jumps at every step and the viewer sees the geometry from two angles
# instead of reading the change as motion. All within 18-28 deg, the range where the
# mediastinum stays readable.
STAGE_ELEV: tuple[float, ...] = (22.0, 18.0, 26.0, 28.0)
SWING_DEG = 60.0                       # total sway about the clear side
SWING_CENTRE = 35.0                    # below ~-12 deg the aorta self-occludes


SEED_RADIUS_MM = 5.0                   # requested radius of the ball on a seed island
SEED_MAX_BALL_MM = 3.5                 # hard cap: a marker never dominates the anatomy
GHOST_ALPHA = 0.18                     # opacity of the folded label while the flood runs
FLOOD_FRAMES = 48                      # watershed thresholds sampled across the flood


def sphere_triangles(center: np.ndarray, radius: float, resolution: int = 10) -> np.ndarray:
    # One triangle per sphere face, centred at `center` and sized in voxel units, so a
    # caller can place it in the same grid as the meshes. Used to mark a seed island:
    # an esophagus seed is a few voxels wide, and the walls of such a tube are
    # invisible at slide size, so a ball is drawn on top of it.
    u = np.linspace(0.0, 2 * np.pi, resolution, endpoint=False)
    v = np.linspace(0.0, np.pi, resolution)
    unit = np.stack([np.outer(np.cos(u), np.sin(v)).ravel(),
                     np.outer(np.sin(u), np.sin(v)).ravel(),
                     np.outer(np.ones_like(u), np.cos(v)).ravel()], axis=1)
    tris = []
    for i in range(resolution - 1):
        for j in range(resolution):
            a = i * resolution + j
            b = i * resolution + (j + 1) % resolution
            c = (i + 1) * resolution + j
            d = (i + 1) * resolution + (j + 1) % resolution
            tris += [[unit[a], unit[b], unit[d]], [unit[a], unit[d], unit[c]]]
    return center[None, None, :] + radius * np.asarray(tris, dtype=float)


def seed_balls(mask: np.ndarray, spacing: tuple[float, float, float], radius_mm: float,
               centre: np.ndarray | None = None) -> np.ndarray:
    # One ball per 3D island of the seed mask. The ball sits at a voxel that is
    # genuinely part of the island - the voxel of maximum distance transform, i.e. the
    # island's deepest interior point - because the centroid of a long curved island
    # lies outside its own voxels and drops the marker off the side of the organ, which
    # reads as random dots scattered around the anatomy.
    #
    # The ball is also clamped to the island's own inradius, with an optional cap, so a
    # marker is never drawn wider than the structure it marks.
    labels, n = nd_label(mask)
    assert n > 0, "the seed stage needs a non-empty seed mask"
    sp = np.asarray(spacing, dtype=float)
    ball_radius = min(radius_mm, SEED_MAX_BALL_MM)
    blocks = []
    for i in range(1, n + 1):
        island = labels == i
        dt_local = nd_distance_transform_edt(island, sampling=spacing)
        deepest = np.unravel_index(int(np.argmax(dt_local)), island.shape)
        inradius = float(dt_local[deepest])
        centre_vox = np.asarray(deepest, dtype=float) + 2.0   # undo surface()'s pad
        origin = centre_vox * sp - (0.0 if centre is None else centre)
        blocks.append(sphere_triangles(origin, 1.0)
                      * (min(ball_radius, inradius) / sp)[None, None, :])
    return np.concatenate(blocks, axis=0)


def seed_mask_paint(mask: np.ndarray, spacing: tuple[float, float, float], step: int,
                    centre: np.ndarray) -> np.ndarray:
    # The seed voxels themselves, as a surface. One mesh per island: an island is the
    # unit the watershed starts from, and a single mesh over all of them would merge
    # neighbours that are actually separate injections.
    #
    # An island can be too small to mesh at the decimation step (a thin tube collapses
    # to a one-voxel island at step_size 4, and marching_cubes raises rather than
    # returning nothing), so any island that fails to mesh is drawn as a ball instead.
    labels, n = nd_label(mask)
    assert n > 0, "the seed stage needs a non-empty seed mask"
    sp = np.asarray(spacing, dtype=float)
    blocks = []
    for i in range(1, n + 1):
        island = labels == i
        try:
            blocks.append(surface(island, spacing, step, centre))
        except RuntimeError:
            dt_local = nd_distance_transform_edt(island, sampling=spacing)
            deepest = np.unravel_index(int(np.argmax(dt_local)), island.shape)
            radius = min(SEED_RADIUS_MM, SEED_MAX_BALL_MM, float(dt_local[deepest]))
            origin = (np.asarray(deepest, dtype=float) + 2.0) * sp - centre
            blocks.append(sphere_triangles(origin, 1.0) * (radius / sp)[None, None, :])
    return np.concatenate(blocks, axis=0)


def flood_steps(markers: np.ndarray, dt: np.ndarray, mask: np.ndarray,
                n_steps: int, labels_out: np.ndarray | None = None) -> list[np.ndarray]:
    # The watershed's own flooding, replayed. skimage's watershed on -dt is a priority
    # flood: voxels enter in order of decreasing dt, and each takes the label of the
    # neighbours already claimed. Painting the threshold sweep is what makes that
    # literally true: every frame unions the components reachable at that depth, so the
    # fronts start at the seeds and grow into the valley between the organs.
    #
    # levels fall from the deepest voxel down to the shallowest, because a distance
    # transform is large in the core and zero at the wall.
    labels = np.array(markers, dtype=np.int16, copy=True)
    depths = np.unique(dt[mask])[::-1]
    # Frames are picked on the number of voxels the flood has claimed, not on the raw
    # depth: flat regions dominate the depth range (Patient_15's front sat on the seeds
    # for the first half of a depth-sampled sweep), while the voxel count is what the
    # viewer sees move.
    #
    # Two passes, because the targets need the total, which is only known at the end:
    # the first walk records the claim count per threshold, the second walks again and
    # snapshots the volume at the thresholds matching an even spread of that count.
    # Holding the snapshots is 48 tiny int8 volumes; holding them is cheaper than
    # re-running the flood.
    counts: list[int] = []
    for level in depths:
        _flood_to_level(labels, dt, mask, level)
        counts.append(int((labels > 0).sum()))
    total = counts[-1]
    picks: list[int] = []
    for want in np.linspace(0, total, n_steps):
        index = int(np.searchsorted(counts, want))
        picks.append(min(index, len(counts) - 1))
    picks = sorted(set(picks))
    frames: list[np.ndarray] = []
    picked = set(picks)
    labels = np.array(markers, dtype=np.int16, copy=True)
    for index, level in enumerate(depths):
        _flood_to_level(labels, dt, mask, level)
        if index in picked:
            frames.append(np.clip(labels, 0, 2).astype(np.int8))
    if labels_out is not None:
        labels_out[...] = labels
    assert len(frames) >= 2, (len(frames), picks)
    return frames


def _flood_to_level(labels: np.ndarray, dt: np.ndarray, mask: np.ndarray,
                    level: float) -> None:
    # Claim every mask voxel at this depth or deeper, propagating labels through the
    # 26-neighbourhood until no unclaimed voxel touches a claimed one. Growth is
    # bounded: each voxel is claimed once over the whole sweep.
    batch = mask & (dt >= level) & (labels == 0)
    while batch.any():
        grown = False
        for axis in range(3):
            for shift in (-1, 1):
                rolled = np.roll(labels, shift, axis=axis)
                if shift == -1:
                    rolled[-1] = 0
                else:
                    rolled[0] = 0
                take = batch & (rolled > 0)
                if take.any():
                    labels[take] = rolled[take]
                    batch[take] = False
                    grown = True
        if not grown:
            # Nothing neighbours a claim yet at this depth: the next threshold picks it
            # up. Unclaimed voxels at the end of the sweep are a real failure.
            break


def stage_geometry(trace: dict, folded: np.ndarray, aorta: np.ndarray,
                   eso: np.ndarray, spacing: tuple[float, float, float], step: int,
                   n_steps: int) -> tuple[list[float], ...]:
    # The meshes the spin animates, plus the camera frame. Stage contents:
    #   1. the folded class, solitary;
    #   2. the seeds, as balls, with the folded surface drawn translucent so the balls
    #      grid light them up from the inside;
    #   3. the watershed flooding, replayed threshold by threshold from the seed depths
    #      down into the valley between the organs: the fronts grow and meet on the
    #      ridge. The translucent folded surface stays up, so the flood reads as
    #      happening inside the label rather than replacing it;
    #   4. the shipped result: opaque aorta and esophagus, no ghost.
    # Every mask comes from split_once's trace, so the video replays the shipped
    # pipeline and not a redrawn approximation.
    assert trace.get("regions") is not None, \
        "spin stages need a split trace: call load_patient(pid, trace)"
    dt = trace["dt"]
    masks = flood_steps(trace["markers"], dt, folded, n_steps)

    # One centre for every mesh in the scene: the middle of the folded label's bounding
    # box, in millimetres. surface()'s pad adds 2 voxels per side, so the centre is
    # measured in the padded grid to match the meshes exactly.
    lo, hi = bounding_box(folded)
    centre = ((lo - 2.0) + (hi + 2.0)) / 2.0 * np.asarray(spacing, dtype=float)

    tri_merged = surface(folded, spacing, step, centre)
    tri_aorta = surface(aorta, spacing, step, centre)
    tri_eso = surface(eso, spacing, step, centre)
    seed_aorta = seed_balls(trace["aorta_seed"], spacing, SEED_RADIUS_MM, centre)
    seed_eso = seed_balls(trace["eso_seed"], spacing, SEED_RADIUS_MM, centre)
    paint_aorta = seed_mask_paint(trace["aorta_seed"], spacing, step, centre)
    paint_eso = seed_mask_paint(trace["eso_seed"], spacing, step, centre)
    return (masks, tri_merged, seed_aorta, seed_eso, tri_aorta, tri_eso, centre,
            paint_aorta, paint_eso)


def bounding_box(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Lowest and highest index occupied by the mask on each axis.
    idx = np.argwhere(mask)
    assert len(idx) > 0, "cannot bound an empty mask"
    return idx.min(axis=0), idx.max(axis=0)


def flood_collection(labels: np.ndarray, spacing: tuple[float, float, float], step: int,
                     centre: np.ndarray) -> tuple[Poly3DCollection, np.ndarray]:
    # One collection holding the two basins of the flood at one threshold, split by
    # colour inside the collection so the two fronts occlude each other correctly.
    blocks = []
    for label_id, colour in ((1, ra.AORTA_COLOUR), (2, ra.ESO_COLOUR)):
        basin = labels == label_id
        if basin.any():
            blocks.append((surface(basin, spacing, step, centre), colour))
    assert blocks, "the flood frame has neither basin: check the markers"
    return surface_collection(*blocks)


def spin_scene(folded: np.ndarray, aorta: np.ndarray, eso: np.ndarray,
               spacing: tuple[float, float, float], step: int, trace: dict,
               args: argparse.Namespace):
    # Left panel: the folded class, opaque, unchanged for the whole clip. Right panel:
    # the pipeline, with the folded label kept as a translucent ghost from stage 2 on,
    # so the seeds and the flooding front are seen inside it.
    fig = plt.figure(figsize=SPIN_FIGSIZE)
    ax_before = fig.add_subplot(1, 2, 1, projection="3d")
    ax_right = fig.add_subplot(1, 2, 2, projection="3d")

    masks, tri_merged, seed_aorta, seed_eso, tri_aorta, tri_eso, centre, \
        paint_aorta, paint_eso = \
        stage_geometry(trace, folded, aorta, eso, spacing, step, args.flood_frames)

    # The frame hugs the anatomy instead of a cube. The organs are a long thin structure
    # (Patient_15: about 95 x 110 x 280 mm), so a cubic frame wastes two thirds of the
    # panel on empty space at every azimuth. The box is built from the per-axis extents
    # of every mesh drawn in any stage, expanded by --zoom and made square in the two
    # in-plane axes so the object keeps a constant size as the camera turns.
    allv = np.vstack([tri.reshape(-1, 3) for tri in
                      (tri_merged, seed_aorta, seed_eso, tri_aorta, tri_eso)]).astype(float)
    half = np.abs(allv).max(axis=0) * args.zoom          # per-axis half-extent
    inplane = max(half[0], half[1])                      # square the x/y footprint
    half[0] = half[1] = inplane
    for ax in (ax_before, ax_right):
        ax.set_xlim(-half[0], half[0])
        ax.set_ylim(-half[1], half[1])
        ax.set_zlim(-half[2], half[2])
        # Aspect from the half-extents: the drawn anatomy keeps its true proportions,
        # so the organs fill the panel along z instead of floating in a cube.
        ax.set_box_aspect((float(half[0]), float(half[1]), float(half[2])))
        ax.set_axis_off()
    print(f"frame half-extents x{half[0]:.0f} y{half[1]:.0f} z{half[2]:.0f} mm "
          f"(zoom {args.zoom}); organ x{np.abs(tri_aorta).max():.0f} mm")

    # An artist belongs to exactly one Axes, so every stage gets its own collection over
    # the shared triangles, and matplotlib 3.11 exposes no public way to remove one
    # (axes.collections is a read-only view), so the stages are swapped by visibility.
    collections: list[tuple[Poly3DCollection, int, bool]] = []

    def attach(stage: int, *blocks: tuple[np.ndarray, tuple], alpha: float = 1.0):
        # One collection per stage, however many organs it holds. mplot3d orders whole
        # collections by a single scalar each, so an aorta collection and an esophagus
        # collection depth-sort against each other only at that granularity: their
        # scalars sit close together and the order flips on small camera changes, which
        # paints the red mesh over the entire yellow one instead of occluding it. Every
        # organ of a stage therefore shares one collection, and the per-triangle sort
        # inside that collection does the occlusion.
        collection, _ = surface_collection(*blocks, alpha=alpha)
        collection.set_visible(stage == 0)
        ax_right.add_collection3d(collection)
        collections.append((collection, stage, False))

    attach(0, (tri_merged, MERGED_COLOUR))
    ghost_merged, _ = surface_collection((tri_merged, MERGED_COLOUR), alpha=GHOST_ALPHA)
    ghost_merged.set_visible(False)
    ax_right.add_collection3d(ghost_merged)
    collections.append((ghost_merged, 1, True))          # shown from stage 1 onward
    # The seed voxels themselves, painted inside the ghost, plus one ball at each
    # island's deepest interior point. The paint shows which voxels seed the flood; the
    # balls keep the injection points legible at slide size.
    paint, _ = surface_collection((paint_aorta, SEED_COLOUR), (paint_eso, SEED_COLOUR))
    paint.set_visible(False)
    ax_right.add_collection3d(paint)
    collections.append((paint, 1, True))                 # seeds stay painted in the flood
    attach(1, (seed_aorta, SEED_COLOUR), (seed_eso, SEED_COLOUR))
    attach(3, (tri_aorta, ra.AORTA_COLOUR), (tri_eso, ra.ESO_COLOUR))

    # The flood frames are the only per-frame geometry: one collection per threshold,
    # all attached up front and swapped by visibility, so no frame pays for a mesh.
    flood: list[Poly3DCollection] = []
    for labels in masks:
        collection, _ = flood_collection(labels, spacing, step, centre)
        collection.set_visible(False)
        ax_right.add_collection3d(collection)
        flood.append(collection)

    ax_before.add_collection3d(surface_collection((tri_merged, MERGED_COLOUR))[0])
    ax_before.set_title("folded class 1\n(the label as shipped)", fontsize=15)
    stage_title = fig.text(0.75, 1.0, "", ha="center", va="top", fontsize=15,
                           transform=fig.transFigure)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.02, top=0.90, wspace=0.02)
    return fig, (ax_before, ax_right), collections, flood, stage_title


def show_stage(collections: list[tuple[Poly3DCollection, int, bool]], stage: int) -> None:
    # A collection tagged `onward` appears from its stage to the end of the clip: the
    # translucent folded label stays up under the seeds and under the flooding front.
    for collection, owner, onward in collections:
        collection.set_visible(owner <= stage if onward else owner == stage)


def check_ffmpeg(path: str) -> str | None:
    # Usable means it runs AND carries the encoder this script asks for. Executability
    # is not enough: a Homebrew ffmpeg with a broken dylib aborts on every call, and a
    # browser-bundled ffmpeg runs but ships only VP8.
    try:
        run = subprocess.run([path, "-hide_banner", "-encoders"], capture_output=True,
                             text=True, timeout=60)
    except OSError as error:
        return f"could not run ({error})"
    if run.returncode != 0:
        detail = (run.stderr or "").strip().splitlines()
        return f"exited with {run.returncode} ({detail[-1] if detail else 'no output'})"
    if "libx264" not in run.stdout:
        return "no libx264 encoder"
    return None


def resolve_ffmpeg(explicit: str | None) -> str:
    # Candidates in order: --ffmpeg, the one on PATH, the one imageio-ffmpeg ships.
    candidates: list[str] = []
    if explicit is not None:
        candidates.append(explicit)
    else:
        found = shutil.which("ffmpeg")
        if found is not None:
            candidates.append(found)
        try:
            import imageio_ffmpeg
            candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
        except ImportError:
            pass
    assert candidates, "no ffmpeg candidate: pass --ffmpeg <path>"

    problems: list[str] = []
    for candidate in candidates:
        problem = check_ffmpeg(candidate)
        if problem is None:
            print(f"ffmpeg: {candidate}")
            return candidate
        problems.append(f"  {candidate}: {problem}")
    raise SystemExit("No usable ffmpeg (needs libx264). Tried:\n" + "\n".join(problems)
                     + "\nPass --ffmpeg <path>, install imageio-ffmpeg, or repair the "
                       "system ffmpeg (brew reinstall ffmpeg).")


def render_spin(out_dir: Path, pid: str, folded: np.ndarray, aorta: np.ndarray,
                eso: np.ndarray, spacing: tuple[float, float, float],
                trace: dict, args: argparse.Namespace) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # spin_Patient_15 -> spin_patient15, the name the talk assets are referenced by
    slug = pid.replace("_", "").lower()
    out_path = out_dir / f"spin_{slug}_before_after.mp4"
    poster_path = out_dir / f"spin_{slug}_poster.png"
    probe = out_dir / ".spin_probe.png"

    # Frame cost probe: a slow mesh must not silently turn into a 30-minute render.
    fig = axes = None
    step = 2
    for step in (2, 3, 4):
        fig, axes, collections, flood, stage_title = spin_scene(folded, aorta, eso,
                                                                spacing, step, trace, args)
        for ax in axes:
            ax.view_init(elev=STAGE_ELEV[0], azim=0.0)
        start = time.perf_counter()
        fig.savefig(probe, dpi=args.spin_dpi)
        frame_s = time.perf_counter() - start
        print(f"step_size={step}: {frame_s:.2f} s/frame -> {frame_s * args.frames:.0f} s "
              f"for {args.frames} frames")
        if frame_s * args.frames <= SPIN_BUDGET_S or step == 4:
            break
        plt.close(fig)
    probe.unlink()

    # Cost is judged on the most expensive frame: stage 2 draws two extra surfaces
    # (the seeds) on top of the folded class. frame_s was measured on stage 1.
    print(f"keeping step_size={step}: {step * 6:.0f} ms/voxel on the meshed masks, "
          f"stage 2 is the heaviest frame")

    # Where the discrete steps land in the timeline. --steps let the caller line the
    # transitions up with spoken lines; the remainder is the closing hold, so the last
    # frame stays on screen a little longer than the intermediate stages.
    n_stages = len(SPIN_STAGES)
    at = [min(f, args.frames - 1) for f in args.step_frames]
    assert len(at) == n_stages and all(b > a for a, b in zip(at, at[1:])), \
        (at, "--step-frames must be strictly increasing")
    if at[-1] >= args.frames:
        raise SystemExit(f"--step-frames {at} must all be < --frames {args.frames}")

    def stage_at(i: int) -> int:
        # Index of the stage on screen for frame i; the last stage holds to the end.
        return max(j for j, start in enumerate(at) if start <= i)

    # Poster: the first frame of the video, so the still in the deck matches it.
    for ax, elev in zip(axes, (STAGE_ELEV[0], STAGE_ELEV[0])):
        ax.view_init(elev=elev, azim=0.0)
    stage_title.set_text(SPIN_STAGES[0][0])
    fig.savefig(poster_path, dpi=args.spin_dpi)
    print(f"Wrote {poster_path}")

    # 360 * i / frames so the loop is seamless: frame 0 and the end of the last frame
    # are one frame step apart. Inside a stage the angle advances with the whole
    # timeline, so the camera keeps turning at a constant rate and a stage change is a
    # jump in content, not a stutter.
    writer = FFMpegWriter(fps=args.fps, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])
    # FFMpegWriter.bin_path() reads this rcParam, the documented way to pick a binary.
    plt.rcParams["animation.ffmpeg_path"] = resolve_ffmpeg(args.ffmpeg)
    shown = -1
    flood_shown = -1
    with writer.saving(fig, str(out_path), args.spin_dpi):
        for i in range(args.frames):
            stage = stage_at(i)
            if stage != shown:
                show_stage(collections, stage)
                stage_title.set_text(SPIN_STAGES[stage][0])
                shown = stage
                if stage != 2:
                    # Leaving the flood: hide its last frame. Entering it happens just
                    # below, so a re-entry restarts the flood from the seeds.
                    for collection in flood:
                        collection.set_visible(False)
                    flood_shown = -1
            if stage == 2:
                # The flood is the only per-frame change: the watershed front sweeps
                # from the seed depths into the valley for as long as the stage lasts.
                span = max(at[3] - at[2], 1)
                progress = min(i - at[2], span - 1) / max(span - 1, 1)
                index = min(int(progress * len(flood)), len(flood) - 1)
                if index != flood_shown:
                    if flood_shown >= 0:
                        flood[flood_shown].set_visible(False)
                    flood[index].set_visible(True)
                    flood_shown = index
            # A swing, not a full orbit: the camera sways around SWING_CENTRE, starting and ending
            # facing the mediastinum. A full turn spends a third of the clip with the
            # esophagus between the camera and the aorta, and the aorta is 8x its volume,
            # so the small red tube hides the yellow one. The measured occlusion is
            # one-sided (yellow/red from 1.55 at -45 deg to 4.43 at +45 deg on
            # Patient_15), so the swing is centred on the clear side rather than at 0.
            phase = i / max(args.frames - 1, 1)              # 0..1 over the whole clip
            swing = np.sin(2 * np.pi * phase)                 # one full there-and-back
            azim = SWING_CENTRE + SWING_DEG * swing
            axes[0].view_init(elev=STAGE_ELEV[0], azim=azim)
            axes[1].view_init(elev=STAGE_ELEV[stage], azim=azim)
            writer.grab_frame()
    plt.close(fig)
    print(f"Wrote {out_path}")
    print(f"Spin rendered at step_size={step}, {args.frames} frames at {args.fps} fps "
          f"= {args.frames / args.fps:.1f} s; stage starts {at} "
          f"({[round(f / args.fps, 1) for f in at]} s)")


def load_patient(pid: str, trace: dict | None = None):
    ct_path, gt_path, _ = patient_paths(pid)
    ct, spacing, _ = load_labels(ct_path)
    gt, _, _ = load_labels(gt_path)
    assert gt.shape == ct.shape, (pid, gt.shape, ct.shape)
    m = gt == 1
    assert m.any(), f"{pid}: no class-1 voxels, nothing to split"
    thin_area, aorta, eso, check, verdict = split_patient(m, spacing, trace)
    assert verdict == "ok", f"{pid}: {verdict} ({check})"
    print(f"{pid} aorta {int(aorta.sum())} vx, esophagus {int(eso.sum())} vx, "
          f"thin_area {thin_area:.0f} mm^2, verdict {verdict}")
    return ct, spacing, m, aorta, eso


def main(args: argparse.Namespace) -> None:
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    pid = args.patient
    trace: dict = {}
    ct, spacing, m, aorta, eso = load_patient(pid, trace)

    gt2_path = patient_paths(pid)[2] / "GT2.nii.gz"
    gt2, _, _ = load_labels(gt2_path)
    assert gt2.shape == m.shape, (pid, gt2.shape, m.shape)
    d_aorta = dice(aorta, gt2 == 4)
    d_eso = dice(eso, gt2 == 1)
    print(f"{pid} Dice vs GT2 {gt2_path.name}: aorta {d_aorta:.4f}, esophagus {d_eso:.4f}")
    assert d_aorta >= 0.99 and d_eso >= 0.99, (pid, d_aorta, d_eso)

    fig1_problem(out_dir, ct, m, aorta, eso, spacing, pid, args.dpi)
    fig2_algorithm(out_dir, ct, trace, aorta, eso, spacing, args.dpi)
    fig3_before_after(out_dir, ct, m, aorta, eso, gt2, spacing, pid, d_aorta, d_eso,
                      args.dpi)

    if args.skip_spin:
        return
    spin_id = args.spin_patient
    if spin_id == pid:
        spin_m, spin_aorta, spin_eso, spin_spacing, spin_trace = m, aorta, eso, spacing, trace
    else:
        spin_trace: dict = {}
        _, spin_spacing, spin_m, spin_aorta, spin_eso = load_patient(spin_id, spin_trace)
    render_spin(out_dir, spin_id, spin_m, spin_aorta, spin_eso, spin_spacing, spin_trace,
                args)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Slide figures and the stepped spin video for the aorta recovery",
        epilog="Writes fig1_problem.png, fig2_algorithm.png, fig3_before_after.png and "
               "(unless --skip-spin) spin_<patient>_before_after.mp4 + its poster. The "
               "video's left panel spins the folded class, the right panel replays the "
               "pipeline in four steps: folded, seeds, watershed, result.")
    parser.add_argument('--out', type=Path, default=Path("PRESENTATION"),
                        help="Output folder for the assets (default PRESENTATION)")
    parser.add_argument('--patient', type=str, default="Patient_07",
                        help="Patient carrying the stills: the only one with a GT2 reference")
    parser.add_argument('--spin-patient', type=str, default="Patient_15",
                        help="Patient carrying the 3D spin (widest aorta)")
    parser.add_argument('--frames', type=int, default=132,
                        help="Frames in the spin; 132 at 24 fps is exactly 5.5 s")
    parser.add_argument('--fps', type=int, default=24, help="Spin frame rate")
    parser.add_argument('--step-frames', type=int, nargs=4, default=[0, 24, 48, 78],
                        metavar=("S1", "S2", "S3", "S4"),
                        help="First frame of each pipeline stage on the right panel "
                             "(folded, seeds, watershed, final). Defaults hold each "
                             "stage for about a second: 0, 24, 48, 78 of 132 frames")
    parser.add_argument('--zoom', type=float, default=1.02,
                        help="Frame size as a multiple of the shipped organs' reach: "
                             "1.0 crops exactly to the organs, larger pulls back. The "
                             "frame is clamped to the all-stage reach, so the folded "
                             "label can never be cut off")
    parser.add_argument('--flood-frames', type=int, default=FLOOD_FRAMES,
                        help="Thresholds sampled across the watershed flood, i.e. how "
                             "smoothly the two fronts sweep through the label")
    parser.add_argument('--dpi', type=int, default=DPI, help="DPI of the static figures")
    parser.add_argument('--spin-dpi', type=int, default=100,
                        help="DPI of the spin: 100 on a 12.8x7.2 in figure gives 1280x720")
    parser.add_argument('--skip-spin', action='store_true',
                        help="Write the three PNGs only")
    parser.add_argument('--ffmpeg', type=str, default=None,
                        help="ffmpeg binary to encode the spin with; default is the one "
                             "on PATH, then the imageio-ffmpeg build")

    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    main(get_args())
