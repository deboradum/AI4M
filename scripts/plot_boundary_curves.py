#!/usr/bin/env python3
# MIT License
#
# Boundary quality figure from scripts/boundary_curves.py's histograms:
#   left  - share of the organ surface farther than tau from the true border (log scale), per model
#   right - paired NSD(tau) gain of the 3D U-Net over the 2D and 2.5D U-Nets, 95% bootstrap band
# Per patient = mean over the 4 organs; curves = mean over patients. 8 Oct 2026.
#
#   python scripts/plot_boundary_curves.py --hist results/cv5/boundary_hist.npz --dest results/cv5/figures/boundary
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reference categorical palette (dataviz skill), slots 1-3 validated all-pairs; baseline in muted grey
COLORS = {"ENet (baseline)": "#a3a19b", "2D U-Net": "#2a78d6", "2.5D U-Net": "#1baf7a", "3D U-Net": "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ORGANS = ["esophagus", "heart", "trachea", "aorta"]


def nsd_curves(hist: np.ndarray, edges: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """patient x model x organ x tau: share of both surfaces within tau (metrics3d NSD)."""
    c = hist.cumsum(-1)
    idx = np.array([np.argmin(np.abs(edges[1:] - t)) for t in taus])
    within = c[..., 0, :][..., idx] + c[..., 1, :][..., idx]
    total = (c[..., 0, -1] + c[..., 1, -1])[..., None]
    return within / np.maximum(total, 1)


def boot_band(x: np.ndarray, rng: np.random.Generator, n: int = 10000) -> tuple[np.ndarray, np.ndarray]:
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return np.percentile(b, 2.5, 0), np.percentile(b, 97.5, 0)


def style(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=10)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main(a: argparse.Namespace) -> None:
    z = np.load(a.hist)
    models = list(z["models"])
    taus = np.round(np.arange(0.5, 15.01, 0.1), 1)
    nsd = nsd_curves(z["hist"], z["edges"], taus)
    organs = [ORGANS.index(o) for o in a.organs]
    per_patient = nsd[:, :, organs].mean(2)  # patient x model x tau
    n = per_patient.shape[0]
    rng = np.random.default_rng(0)
    i3 = models.index("3D U-Net")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.9), dpi=a.dpi)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.14, top=0.8, wspace=0.28)

    # Left: share of surface beyond tau
    beyond = 100 * (1 - per_patient.mean(0))  # model x tau
    for m, name in enumerate(models):
        ax1.plot(taus, beyond[m], color=COLORS[name], lw=2.6 if m == i3 else 2, label=name, zorder=3 if m == i3 else 2)
    ax1.set_yscale("log")
    ax1.set_ylim(0.7, 60)
    ax1.set_yticks([1, 2, 5, 10, 20, 50])
    ax1.set_yticklabels(["1%", "2%", "5%", "10%", "20%", "50%"])
    ax1.set_xlim(taus[0], taus[-1])
    ax1.set_xlabel("Distance from the true border, τ (mm)", color=INK2, fontsize=11)
    ax1.set_title("Share of the organ surface farther than τ", loc="left", color=INK, fontsize=12)
    style(ax1)
    t10 = int(np.argmin(np.abs(taus - 10)))
    # Direct labels at 10 mm, pushed apart (in log space) where they would overlap
    ys = beyond[:, t10].copy()
    pos = np.log10(ys)
    for j in np.argsort(-ys)[1:]:
        above = [pos[i] for i in range(len(ys)) if ys[i] > ys[j] or (ys[i] == ys[j] and i < j)]
        pos[j] = min(pos[j], min(above) - 0.085)
    for m, name in enumerate(models):
        ax1.annotate(f"{ys[m]:.1f}%", (taus[t10] + 0.25, 10 ** pos[m]), va="center", fontsize=9.5, color=INK2)
        ax1.plot(taus[t10], beyond[m, t10], "o", ms=5.5, color=COLORS[name], mec="white", mew=1.5, zorder=4)
    ax1.axvline(2, color=INK2, lw=1, ls=(0, (3, 3)))
    ax1.text(2.15, 48, "NSD tolerance\n(2 mm)", fontsize=9, color=INK2, va="top")

    # Right: paired gain of 3D
    for name in ("2D U-Net", "2.5D U-Net"):
        m = models.index(name)
        d = 100 * (per_patient[:, i3] - per_patient[:, m])  # percentage points
        lo, hi = boot_band(d, rng)
        ax2.fill_between(taus, lo, hi, color=COLORS[name], alpha=0.18, lw=0)
        ax2.plot(taus, d.mean(0), color=COLORS[name], lw=2, label=f"3D − {name.replace(' U-Net', '')}")
        sig = lo > 0
        ax2.plot(taus[sig], np.full(sig.sum(), -2.6 if name == "2D U-Net" else -3.1), "s", ms=2.6,
                 color=COLORS[name], lw=0)
    ax2.axhline(0, color=INK2, lw=1)
    ax2.set_xlim(taus[0], taus[-1])
    ax2.set_ylim(-3.5, 4)
    ax2.set_xlabel("Tolerance τ (mm)", color=INK2, fontsize=11)
    ax2.set_ylabel("NSD(τ) gain of 3D (percentage points)", color=INK2, fontsize=10.5)
    ax2.set_title("3D U-Net gain over the slice models, per patient", loc="left", color=INK, fontsize=12)
    ax2.text(taus[-1], -2.25, "ticks = 95% CI above zero", ha="right", fontsize=9, color=INK2)
    style(ax2)
    ax2.legend(frameon=False, fontsize=10, loc="upper right", labelcolor=INK2)

    handles = [plt.Line2D([], [], color=COLORS[n_], lw=2.6) for n_ in models]
    fig.legend(handles, models, loc="upper left", bbox_to_anchor=(0.07, 0.905), ncol=4, frameon=False,
               fontsize=10.5, labelcolor=INK2)
    org = "all four organs" if len(organs) == 4 else ", ".join(a.organs)
    others = min(beyond[m, t10] for m in range(len(models)) if m != i3)
    headline = (f"The 3D U-Net leaves the fewest far-off borders: {beyond[i3, t10]:.1f}% of its surface is more than "
                f"10 mm off, vs {others:.1f}% for the best slice model") if len(organs) == 4 else \
        f"{org.capitalize()}: distance of the predicted border from the true one, by model"
    fig.suptitle(headline,
                 x=0.07, ha="left", y=0.985, fontsize=13, color=INK)
    fig.text(0.98, 0.015, f"5-fold CV, {n} patients every model has, post-processed; {org}; mean over organs, then "
             "patients. Band = 95% bootstrap CI of the paired difference.", ha="right", fontsize=8.5, color=INK2)
    a.dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.dest.with_suffix(".png"), facecolor="white")
    fig.savefig(a.dest.with_suffix(".pdf"), facecolor="white")
    print(f"Wrote {a.dest}.png/.pdf")


def simple(a: argparse.Namespace) -> None:
    """One panel for slides: share of the border more than tau off, per model, labelled at the line ends."""
    z = np.load(a.hist)
    models = list(z["models"])
    taus = np.round(np.arange(1.0, 15.01, 0.1), 1)
    organs = [ORGANS.index(o) for o in a.organs]
    beyond = 100 * (1 - nsd_curves(z["hist"], z["edges"], taus)[:, :, organs].mean(2).mean(0))  # model x tau
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=a.dpi)
    fig.subplots_adjust(left=0.1, right=0.8, bottom=0.13, top=0.85)
    ys = beyond[:, -1]
    pos = np.log10(ys)
    for j in np.argsort(-ys)[1:]:  # push end labels apart
        pos[j] = min(pos[j], min(pos[i] for i in range(len(ys)) if ys[i] > ys[j]) - 0.06)
    for m, name in enumerate(models):
        main_ = name == "3D U-Net"
        ax.plot(taus, beyond[m], color=COLORS[name], lw=3 if main_ else 2, zorder=3 if main_ else 2)
        ax.text(taus[-1] + 0.3, 10 ** pos[m], name, color=INK if main_ else INK2, va="center", fontsize=11,
                fontweight="bold" if main_ else "normal")
    ax.set_yscale("log")
    ax.set_ylim(0.5, 50)
    ax.set_yticks([0.5, 1, 2, 5, 10, 20, 50])
    ax.set_yticklabels(["0.5%", "1%", "2%", "5%", "10%", "20%", "50%"])
    ax.set_xlim(taus[0], taus[-1])
    ax.set_xlabel("Distance from the true border (mm)", color=INK2, fontsize=11)
    ax.set_ylabel("Share of the border farther than that", color=INK2, fontsize=11)
    style(ax)
    org = "all organs" if len(organs) == 4 else ", ".join(a.organs)
    fig.suptitle("The 3D U-Net has the fewest far-off borders" if len(organs) == 4 else org.capitalize(),
                 x=0.1, ha="left", y=0.96, fontsize=14, color=INK)
    fig.text(0.1, 0.885, f"Lower is better · {org}, 5-fold CV, {z['hist'].shape[0]} patients", fontsize=10,
             color=INK2)
    a.dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.dest.with_suffix(".png"), facecolor="white")
    fig.savefig(a.dest.with_suffix(".pdf"), facecolor="white")
    print(f"Wrote {a.dest}.png/.pdf")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hist", type=Path, default=Path("results/cv5/boundary_hist.npz"))
    p.add_argument("--dest", type=Path, default=Path("results/cv5/figures/boundary"))
    p.add_argument("--organs", nargs="+", default=ORGANS, choices=ORGANS)
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--simple", action="store_true", help="One panel for slides instead of the two-panel figure")
    args = p.parse_args()
    simple(args) if args.simple else main(args)
