#!/usr/bin/env python3
# MIT License
#
# Portrait headline card for "why 3D": axial CT slices where the ground truth has the esophagus or
# trachea but the model predicts none of it, for 2D (E_F06), 2.5D (E_F05) and 3D (E_F07). Same
# canvas, font and colours as card_vertical in render_enet_tables.py. Per-patient counts come from
# results/cv5/tube_continuity_stats.json (analysis/tube_continuity_stats.py, pp_final predictions);
# paired 3D - 2D totals with a 95% patient bootstrap CI.
#
#   python scripts/render_tube_card.py
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from render_enet_tables import CARD_SIZE, COLORS, INK, INK2, MUTED, TRACK, save, style  # noqa: E402

MODELS = ["2D", "2.5D", "3D"]
TUBES = [("esophagus", COLORS[0]), ("trachea", COLORS[2])]
OTHER = "#b9b8b2"  # 2D / 2.5D bars: neutral, so the organ colour marks the 3D model


def boot_ci(d: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    s = d[rng.integers(0, len(d), (n, len(d)))].sum(1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def main(a: argparse.Namespace) -> None:
    style()
    rows = json.loads(a.stats.read_text())
    n = len(rows)
    tot = {(m, o): sum(r[f"{m}_{o}_missed"] for r in rows) for m in MODELS for o, _ in TUBES}
    gts = {o: sum(r[f"{o}_gt_slices"] for r in rows) for o, _ in TUBES}
    both2d = sum(tot["2D", o] for o, _ in TUBES)
    both3d = sum(tot["3D", o] for o, _ in TUBES)

    fig = plt.figure(figsize=CARD_SIZE)
    x0, x1 = 0.10, 0.92
    fig.text(x0, 0.950, "Why 3D · where the tubes vanish", fontsize=12, color=INK2)
    fig.text(x0, 0.815, f"−{100 * (1 - both3d / both2d):.0f}%", fontsize=58, fontweight="bold", color=INK)
    fig.text(x0, 0.775, f"missed slices, 2D → 3D U-Net: {both2d} → {both3d}", fontsize=11, color=INK2)
    fig.text(x0, 0.748, "slice has the organ, model predicts none of it", fontsize=10, color=INK2)
    fig.text(x0, 0.715, f"esophagus + trachea · 5-fold CV · {n} patients · post-processed", fontsize=9, color=MUTED)
    fig.add_artist(plt.Line2D([x0, x1], [0.695, 0.695], color=TRACK, lw=1))

    fig.text(x0, 0.660, "Missed slices, % of ground-truth slices", fontsize=11, color=INK2)
    ax = fig.add_axes([x0, 0.335, x1 - x0, 0.305])
    xmax = 1.05 * max(tot[m, o] / gts[o] for m in MODELS for o, _ in TUBES)
    y = 0.0
    for o, col in TUBES:
        ax.text(0, y, o, ha="left", va="center", fontsize=11, color=INK)
        y -= 0.85
        for m in MODELS:
            v = tot[m, o] / gts[o]
            ax.text(0, y, m, ha="left", va="center", fontsize=9.5, color=INK2)
            ax.barh(y, 0.70, left=0.16, height=0.42, color=TRACK, zorder=1)
            ax.barh(y, 0.70 * v / xmax, left=0.16, height=0.42, color=col if m == "3D" else OTHER, zorder=2)
            ax.text(1.0, y, f"{100 * v:.1f}%", ha="right", va="center", fontsize=10.5,
                    fontweight="bold" if m == "3D" else "normal", color=INK)
            y -= 0.65
        y -= 0.35
    ax.set_xlim(0, 1.0)
    ax.set_ylim(y + 0.4, 0.5)
    ax.axis("off")
    fig.add_artist(plt.Line2D([x0, x1], [0.315, 0.315], color=TRACK, lw=1))

    fig.text(x0, 0.280, "3D vs 2D, paired per patient", fontsize=11, color=INK2)
    for k, (o, col) in enumerate(TUBES):
        d = np.array([r[f"3D_{o}_missed"] - r[f"2D_{o}_missed"] for r in rows])
        lo, hi = boot_ci(d)
        cx = x0 + k * 0.43
        fig.text(cx, 0.205, f"{tot['2D', o]} → {tot['3D', o]}", fontsize=17, fontweight="bold", color=INK)
        fig.add_artist(plt.Line2D([cx, cx + 0.025], [0.183, 0.183], color=col, lw=6, solid_capstyle="butt"))
        fig.text(cx + 0.035, 0.175, f"{o} slices", fontsize=10, color=INK2)
        fig.text(cx, 0.140, f"fewer in {np.sum(d < 0)}, tied in {np.sum(d == 0)}", fontsize=9.5, color=INK2)
        fig.text(cx, 0.115, f"of {n} patients", fontsize=9.5, color=INK2)
        fig.text(cx, 0.085, f"Δ {d.sum():+d}, 95% CI {lo:+.0f} to {hi:+.0f}".replace("-", "−"), fontsize=9, color=MUTED)
    a.dest.mkdir(parents=True, exist_ok=True)
    save(fig, a.dest, "tube_continuity_vertical", tight=False)
    print(f"wrote {a.dest}/tube_continuity_vertical.png/.pdf")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stats", type=Path, default=Path("results/cv5/tube_continuity_stats.json"))
    p.add_argument("--dest", type=Path, default=Path("results/cv5/renders_pp/tables/UNET3d"))
    main(p.parse_args())
