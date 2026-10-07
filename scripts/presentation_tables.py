#!/usr/bin/env python3
# MIT License
#
# Markdown tables for docs/presentation/, computed from metrics3d.py outputs so
# the numbers in the slides are exactly the numbers of the submission. Re-run
# after the final checkpoint and paste the output over the old tables.
#
#   python scripts/presentation_tables.py --run results/snapshots/E_F07_ep24
import json
import argparse
from pathlib import Path

import numpy as np

ORGANS = ["esophagus", "heart", "trachea", "aorta"]
METRICS = [("dice", "Dice", "↑", 3), ("iou", "IoU", "↑", 3), ("hd95", "HD95 (mm)", "↓", 1),
           ("assd", "ASSD (mm)", "↓", 2), ("nsd", "NSD (2 mm)", "↑", 3)]


def load(folder: Path) -> dict[str, np.ndarray]:
    """metric -> (patients, 4 organs)"""
    out = {}
    for m, *_ in METRICS:
        npz = np.load(folder / f"{m}.npz")
        out[m] = np.stack([npz[i][1:] for i in sorted(npz.files)])
    return out


def baseline(path: Path) -> dict[str, np.ndarray]:
    r = json.loads(path.read_text())["results"]["metrics_3d"]
    keys = {"dice": "dice_mean", "iou": "iou_mean", "hd95": "hd95_mm_mean", "assd": "assd_mm_mean", "nsd": "nsd_mean"}
    return {m: np.array([r[keys[m]][o] for o in ORGANS]) for m in keys}


def main(args: argparse.Namespace) -> None:
    base = baseline(args.baseline)
    raw = load(args.run / "val" / "metrics")
    final = load(args.run / "val" / "pp_final" / "metrics")

    print("### Validation, 8 patients: baseline vs 3D U-Net vs 3D U-Net + post-processing\n")
    print("| Metric | Model | " + " | ".join(ORGANS) + " | mean |")
    print("|---|---|" + "---|" * (len(ORGANS) + 1))
    for m, name, arrow, nd in METRICS:
        for label, vals in [("ENet baseline (E_F00)", base[m]), ("3D U-Net (E_F07)", np.nanmean(raw[m], 0)),
                            ("3D U-Net + post-proc.", np.nanmean(final[m], 0))]:
            bold = label.startswith("3D U-Net +")
            cells = [f"{v:.{nd}f}" for v in vals] + [f"{np.mean(vals):.{nd}f}"]
            if bold:
                cells = [f"**{c}**" for c in cells]
            print(f"| {name} {arrow} | {label} | " + " | ".join(cells) + " |")

    print("\n### Per patient, 3D U-Net + post-processing (Dice / HD95 mm)\n")
    npz = np.load(args.run / "val" / "pp_final" / "metrics" / "dice.npz")
    ids = sorted(npz.files)
    print("| Patient | " + " | ".join(ORGANS) + " |")
    print("|---|" + "---|" * len(ORGANS))
    for j, pid in enumerate(ids):
        print(f"| {pid} | " + " | ".join(f"{final['dice'][j, k]:.3f} / {final['hd95'][j, k]:.1f}"
                                         for k in range(len(ORGANS))) + " |")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True, help="Folder with val/metrics and val/pp_final/metrics")
    p.add_argument("--baseline", type=Path, default=Path("experiments/ENet/baseline/E_F00_baseline.json"))
    main(p.parse_args())
