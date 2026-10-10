#!/usr/bin/env python3
# MIT License
#
# Three alternative figures for the ENet baseline (E_F11) 5-fold CV scores, instead of the plain
# metric x organ table: (A) a dot table, (B) per-patient strips, (C) a headline card. Organ colours
# match render_3d.py; every organ is also named in text. Summary numbers are read from
# results/cv5/cv5_pooled.md (the doc's table), per-patient values from the pp_final npz files.
#
#   python scripts/render_enet_tables.py --dest results/cv5/renders_pp/tables/enet
import re
import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, str(Path(__file__).parent))
from render_3d import ORGANS  # noqa: E402
from cv5_pooled import load  # noqa: E402

RUN, LABEL = "E_F11_enet_ce_dice", "E_F11 ENet (baseline)"
NAMES = [ORGANS[k][0] for k in (1, 2, 3, 4)]
COLORS = [ORGANS[k][1] for k in (1, 2, 3, 4)]
# metric key, row name in cv5_pooled.md, header, higher is better, decimals, axis max
METRICS = [("dice", "Dice", "Dice", True, 3, 1.0), ("iou", "IoU", "IoU", True, 3, 1.0),
           ("nsd", "NSD (2 mm)", "NSD 2 mm", True, 3, 1.0),
           ("hd95", "HD95 (mm)", "HD95 mm", False, 1, 25.0), ("assd", "ASSD (mm)", "ASSD mm", False, 2, 5.0)]
INK, INK2, MUTED, TRACK, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#ffffff"


def style() -> None:
    fam = "Lato" if any("Lato" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
    plt.rcParams.update({"font.family": fam, "font.size": 10, "text.color": INK, "axes.edgecolor": TRACK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def summary(md: Path, label: str = LABEL) -> dict[str, tuple[list[float], float, float, float]]:
    """metric -> (4 organ means, mean, CI low, CI high) for one model (default ENet), post-processed."""
    text = md.read_text().split("## Per model, raw")[0]
    out = {}
    for key, row, *_ in METRICS:
        line = next(l for l in text.splitlines() if l.startswith(f"| {row} | {label} |"))
        cells = [c.strip() for c in line.strip("|").split("|")]
        m, lo, hi = map(float, re.findall(r"[-\d.]+", cells[-1]))
        out[key] = ([float(c) for c in cells[3:7]], m, lo, hi)
    return out


def save(fig, dest: Path, name: str, tight: bool = True) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(dest / f"{name}.{ext}", dpi=220, **({"bbox_inches": "tight", "pad_inches": 0.25} if tight else {}))
    plt.close(fig)


def dot_table(s: dict, dest: Path) -> None:
    """(A) The table's layout, but every number sits on its own scale: a dot on a 0-1 track for the
    overlap metrics, on a mm track for the distances; the mean row carries the 95% CI."""
    rows = NAMES + ["mean"]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(11.5, 3.6), sharey=True,
                             gridspec_kw={"wspace": 0.18})
    for ax, (key, _, head, higher, nd, xmax) in zip(axes, METRICS):
        organs, mean, lo, hi = s[key]
        for i, (name, v) in enumerate(zip(rows, organs + [mean])):
            y = len(rows) - 1 - i
            ax.plot([0, xmax], [y, y], color=TRACK, lw=3, solid_capstyle="round", zorder=1)
            if name == "mean":
                ax.plot([lo, hi], [y, y], color=INK, lw=1.6, solid_capstyle="round", zorder=2)
                ax.scatter([v], [y], s=70, color=INK, zorder=3, edgecolor=SURFACE, linewidth=1.5)
            else:
                ax.scatter([v], [y], s=70, color=COLORS[i], zorder=3, edgecolor=SURFACE, linewidth=1.5)
            ax.text(v, y + 0.32, f"{v:.{nd}f}", ha="center", va="bottom", fontsize=9,
                    color=INK if name == "mean" else INK2, fontweight="bold" if name == "mean" else "normal")
        ax.set_xlim(-0.04 * xmax, xmax * 1.04)
        ax.set_ylim(-0.6, len(rows) - 0.3)
        ax.set_title(f"{head}  {'↑' if higher else '↓'}", fontsize=11, loc="left", color=INK, pad=10)
        ax.set_xticks([0, xmax / 2, xmax])
        ax.set_xticklabels([f"{t:g}" for t in (0, xmax / 2, xmax)], fontsize=8, color=MUTED)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels(rows[::-1], fontsize=10.5)
    for t, c in zip(axes[0].get_yticklabels()[::-1], COLORS + [INK]):
        t.set_color(c if c != INK else INK)
        t.set_fontweight("bold" if c == INK else "normal")
    fig.suptitle("ENet baseline (E_F11), 5-fold cross-validation, 40 patients", x=0.125, ha="left",
                 fontsize=13, y=1.10)
    fig.text(0.125, 1.0, "Per-organ mean over held-out patients, post-processed. Mean row: dot = mean over "
             "organs and patients, line = 95% bootstrap CI.", fontsize=9, color=INK2)
    save(fig, dest, "A_dot_table")


def patient_strips(dest: Path, root: Path) -> None:
    """(B) What the averages hide: every held-out patient as a dot, organ mean as a bar."""
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (key, head, higher, nd, xmax) in zip(axes, [("dice", "Dice  ↑", True, 3, 1.0),
                                                       ("hd95", "HD95 (mm)  ↓", False, 1, 125.0)]):
        d = load(root, RUN, "eval/pp_final/metrics", key)
        ids = sorted(d)
        vals = np.stack([d[p] for p in ids])
        for i, (name, col) in enumerate(zip(NAMES, COLORS)):
            y = 3 - i
            v = vals[:, i]
            jit = rng.uniform(-0.18, 0.18, len(v))
            ax.scatter(v, y + jit, s=22, color=col, alpha=0.6, edgecolor=SURFACE, linewidth=0.6, zorder=2)
            mu = np.nanmean(v)
            ax.plot([mu, mu], [y - 0.32, y + 0.32], color=INK, lw=2.2, solid_capstyle="round", zorder=3)
            ax.text(mu, y + 0.36, f"{mu:.{nd}f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
            worst = int(np.nanargmin(v) if higher else np.nanargmax(v))
            if name == "trachea":
                up = 1 if key == "hd95" else -1
                ax.annotate(ids[worst].replace("_", " "), (v[worst], y + jit[worst]), xytext=(0, 14 * up),
                            textcoords="offset points", ha="right" if up > 0 else "center",
                            va="bottom" if up > 0 else "top", fontsize=8, color=INK2,
                            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
        ax.set_xlim(-0.03 * xmax, xmax)
        ax.set_title(head, loc="left", fontsize=11, pad=10)
        ax.grid(axis="x", color=TRACK, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, labelsize=8.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_yticks(range(4))
    axes[0].set_yticklabels(NAMES[::-1], fontsize=10.5)
    for t, c in zip(axes[0].get_yticklabels(), COLORS[::-1]):
        t.set_color(c)
    fig.suptitle("ENet baseline (E_F11): every held-out patient", x=0.125, ha="left", fontsize=13, y=1.05)
    fig.text(0.125, 0.975, "5-fold CV, 40 patients, post-processed. Dot = one patient; black bar = organ mean.",
             fontsize=9, color=INK2)
    save(fig, dest, "B_patient_strips")


def headline(s: dict, dest: Path) -> None:
    """(C) Slide card: the one number, then Dice per organ, then the other metrics in one line."""
    organs, mean, lo, hi = s["dice"]
    fig = plt.figure(figsize=(11.5, 3.8))
    fig.text(0.04, 0.80, "ENet baseline · mean Dice", fontsize=12, color=INK2)
    fig.text(0.04, 0.42, f"{mean:.3f}", fontsize=58, fontweight="bold", color=INK)
    fig.text(0.04, 0.30, f"95% CI {lo:.3f} – {hi:.3f}", fontsize=11, color=INK2)
    fig.text(0.04, 0.22, "5-fold CV · 40 patients · post-processed", fontsize=9, color=MUTED)
    ax = fig.add_axes([0.40, 0.30, 0.55, 0.52])
    order = np.argsort(organs)[::-1]
    for j, i in enumerate(order):
        y = 3 - j
        ax.barh(y, 1.0, height=0.34, color=TRACK, zorder=1)
        ax.barh(y, organs[i], height=0.34, color=COLORS[i], zorder=2)
        ax.text(-0.02, y, NAMES[i], ha="right", va="center", fontsize=11, color=INK)
        ax.text(organs[i] + 0.012, y, f"{organs[i]:.3f}", ha="left", va="center", fontsize=10,
                fontweight="bold", color=INK)
    ax.set_xlim(0, 1.08)
    ax.set_ylim(-0.5, 3.5)
    ax.axis("off")
    ax.set_title("Dice per organ", loc="left", fontsize=11, color=INK2, pad=6)
    parts = [f"{h}  {s[k][1]:.{nd}f}" for k, _, h, _, nd, _ in METRICS[1:]]
    fig.text(0.40, 0.14, "     ·     ".join(parts), fontsize=10.5, color=INK2)
    save(fig, dest, "C_headline")


UNET2D, RED = "E_F06 2D", "#d62728"
CARD_SIZE = (939 / 220, 1553 / 220)  # every vertical card: 939 x 1553 px at 220 dpi (the first ENet card), no tight crop


def paired(md: Path, b: str, a: str = LABEL) -> dict[str, tuple[str, str]]:
    """metric -> (mean difference as printed, 'k/n' patients better) from cv5_pooled.md, B - A."""
    out = {}
    for line in md.read_text().splitlines():
        if line.startswith(f"| {b} − {a} |"):
            c = [x.strip() for x in line.strip("|").split("|")]
            key = {"Dice": "dice", "HD95 (mm)": "hd95", "NSD (2 mm)": "nsd"}[c[1]]
            out[key] = (c[7], c[8])
    return out


def card_vertical(md: Path, dest: Path, name: str, label: str = LABEL, title: str = "ENet baseline",
                  vs_enet: bool = False) -> None:
    """Portrait headline card, one fixed layout for every model (organ colours as in the renders).
    With vs_enet the ENet baseline is drawn in red on every number: a red tick on each organ bar,
    the ENet value in red under each metric; without it those slots stay empty, so cards line up."""
    u = summary(md, label)
    e, d = (summary(md), paired(md, label)) if vs_enet else (None, None)
    organs, mean, lo, hi = u["dice"]
    fig = plt.figure(figsize=CARD_SIZE)
    x0, x1 = 0.10, 0.92
    fig.text(x0, 0.950, f"{title} vs ENet baseline · mean Dice" if vs_enet else f"{title} · mean Dice",
             fontsize=12, color=INK2)
    fig.text(x0, 0.815, f"{mean:.3f}", fontsize=58, fontweight="bold", color=INK)
    fig.text(x0, 0.775, f"95% CI {lo:.3f} – {hi:.3f}", fontsize=11, color=INK2)
    if vs_enet:
        dm, better = d["dice"]
        fig.text(x0, 0.740, f"ENet {e['dice'][1]:.3f}", fontsize=11, color=RED, fontweight="bold")
        fig.text(x0, 0.710, f"{dm.replace(' ✓', '')} Dice · better in {better.replace('/', ' of ')} patients",
                 fontsize=10, color=INK2)
    fig.text(x0, 0.680 if vs_enet else 0.742, "5-fold CV · 40 patients · post-processed", fontsize=9, color=MUTED)
    fig.add_artist(plt.Line2D([x0, x1], [0.660, 0.660], color=TRACK, lw=1))

    fig.text(x0, 0.625, "Dice per organ", fontsize=11, color=INK2)
    if vs_enet:  # key: organ-coloured bar = the model, red tick = ENet
        for n, c in enumerate(COLORS):
            fig.add_artist(plt.Line2D([0.57 + n * 0.018, 0.584 + n * 0.018], [0.631, 0.631], color=c, lw=6,
                                      solid_capstyle="butt"))
        fig.text(0.655, 0.625, title, fontsize=8.5, color=INK2)
        fig.add_artist(plt.Line2D([0.81, 0.81], [0.621, 0.641], color=RED, lw=2))
        fig.text(0.825, 0.625, "ENet", fontsize=8.5, color=INK2)
    ax = fig.add_axes([x0, 0.335, x1 - x0, 0.275])
    order = np.argsort(organs)[::-1]
    for j, i in enumerate(order):
        y = 3 - j
        ax.text(0, y + 0.30, NAMES[i], ha="left", va="bottom", fontsize=11, color=INK)
        ax.text(1.0, y + 0.30, f"{organs[i]:.3f}", ha="right", va="bottom", fontsize=11, fontweight="bold", color=INK)
        ax.barh(y, 1.0, height=0.24, color=TRACK, zorder=1)
        ax.barh(y, organs[i], height=0.24, color=COLORS[i], zorder=2)
        if vs_enet:
            b = e["dice"][0][i]
            ax.text(0.84, y + 0.30, f"+{organs[i] - b:.3f}", ha="right", va="bottom", fontsize=9.5, color=MUTED)
            # ENet tick pokes out above and below the bar, white outline so it reads on the red heart bar too
            ax.plot([b, b], [y - 0.24, y + 0.24], color=RED, lw=2.4, zorder=4, solid_capstyle="butt")
            ax.plot([b, b], [y - 0.26, y + 0.26], color=SURFACE, lw=5.2, zorder=3, solid_capstyle="butt")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.4, 3.75)
    ax.axis("off")
    fig.add_artist(plt.Line2D([x0, x1], [0.315, 0.315], color=TRACK, lw=1))

    for n, (k, _, h, higher, nd, _) in enumerate(METRICS[1:]):
        cx, cy = x0 + (n % 2) * 0.43, 0.245 - (n // 2) * 0.12
        unit = " mm" if h.startswith(("HD95", "ASSD")) else ""
        fig.text(cx, cy, f"{u[k][1]:.{nd}f}{unit}", fontsize=17, fontweight="bold", color=INK)
        fig.text(cx, cy - 0.030, f"{h.replace(' mm', '') if unit else h}  {'↑' if higher else '↓'}",
                 fontsize=10, color=INK2)
        if vs_enet:
            fig.text(cx, cy - 0.057, f"ENet {e[k][1]:.{nd}f}{unit}", fontsize=10, color=RED)
    dest.mkdir(parents=True, exist_ok=True)
    save(fig, dest, name, tight=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("results/cv5"))
    p.add_argument("--dest", type=Path, default=Path("results/cv5/renders_pp/tables/enet"))
    a = p.parse_args()
    a.dest.mkdir(parents=True, exist_ok=True)
    style()
    s = summary(a.root / "cv5_pooled.md")
    dot_table(s, a.dest)
    patient_strips(a.dest, a.root)
    headline(s, a.dest)
    md = a.root / "cv5_pooled.md"
    card_vertical(md, a.dest, "C_headline_vertical")
    card_vertical(md, a.dest.parent / "UNET2d", "unet2d_vs_enet_vertical", UNET2D, "2D U-Net", vs_enet=True)
    print(f"Wrote A_dot_table, B_patient_strips, C_headline, C_headline_vertical (.png/.pdf) to {a.dest}; unet2d_vs_enet_vertical to {a.dest.parent / 'UNET2d'}")
