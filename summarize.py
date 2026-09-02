#!/usr/bin/env python3

# MIT License
#
# Collect, for one or several runs, the configuration (config_dump.yaml), the
# cost (stats.json from main.py, timing.txt from infer.py) and the 3D metrics
# (metrics3d/metrics.csv from metrics3d.py) into one row per run, printed as
# a Markdown table ready for EXPERIMENTS.md and optionally saved as csv.
#
#   python summarize.py results/E001 results/E002 --volumes volumes --csv results/summary.csv

import csv
import json
import argparse
from pathlib import Path

import yaml
import numpy as np

CLASSES: list[str] = ["esophagus", "heart", "trachea", "aorta"]
COLUMNS: list[str] = (["run", "commit", "model", "mode", "epochs", "lr", "B", "seed",
                       "train_min", "s_per_epoch", "peak_gpu_mb", "params_k", "infer_s_per_patient"]
                      + [f"dice_{c}" for c in CLASSES] + [f"hd95_{c}" for c in CLASSES])


def read_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def read_timing(path: Path) -> dict[str, str]:
    with open(path) as f:
        return dict(line.split(maxsplit=1) for line in f if line.strip())


def read_metrics(path: Path) -> dict[str, dict[str, float]]:
    # metrics.csv is long format: patient, class, class_name, dice, iou, hd95, assd, nsd
    per_class: dict[str, dict[str, list[float]]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            d = per_class.setdefault(row["class_name"], {})
            for m in ["dice", "hd95"]:
                d.setdefault(m, []).append(float(row[m]))
    return {name: {m: float(np.nanmean(v)) if not np.all(np.isnan(v)) else float('nan')
                   for m, v in d.items()}
            for name, d in per_class.items()}


def summarize_run(run: Path, volumes_root: Path | None) -> dict[str, object]:
    row: dict[str, object] = {c: None for c in COLUMNS}
    row["run"] = run.name

    if (run / "config_dump.yaml").exists():
        cfg = read_yaml(run / "config_dump.yaml")
        row.update({"model": cfg.get("net_name"), "mode": cfg.get("mode"), "epochs": cfg.get("epochs"),
                    "lr": cfg.get("lr"), "B": cfg.get("B"), "seed": cfg.get("seed")})

    if (run / "stats.json").exists():
        with open(run / "stats.json") as f:
            st = json.load(f)
        row.update({"commit": st.get("git_commit"), "train_min": st.get("total_min"),
                    "s_per_epoch": st.get("train_s_per_epoch"), "peak_gpu_mb": st.get("peak_gpu_mem_mb"),
                    "params_k": None if st.get("n_params") is None else round(st["n_params"] / 1000, 1)})
        if st.get("epochs_done") != row["epochs"]:
            row["epochs"] = f"{st.get('epochs_done')}/{row['epochs']}"

    timing = None
    for candidate in [run / "timing.txt"] + ([volumes_root / run.name / "timing.txt"] if volumes_root else []):
        if candidate.exists():
            timing = read_timing(candidate)
            break
    if timing:
        row["infer_s_per_patient"] = float(timing["per_patient_s"])

    if (run / "metrics3d" / "metrics.csv").exists():
        metrics = read_metrics(run / "metrics3d" / "metrics.csv")
        for c in CLASSES:
            if c in metrics:
                row[f"dice_{c}"] = metrics[c]["dice"]
                row[f"hd95_{c}"] = metrics[c]["hd95"]

    return row


def fmt(v: object) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        if np.isnan(v):
            return "n/a"
        if abs(v) < 0.01:  # learning rates
            return f"{v:.2g}"
        return f"{v:.3f}" if v < 10 else f"{v:.1f}"
    return str(v)


def print_markdown(rows: list[dict[str, object]]) -> None:
    print("| " + " | ".join(COLUMNS) + " |")
    print("|" + "|".join("---" for _ in COLUMNS) + "|")
    for row in rows:
        print("| " + " | ".join(fmt(row[c]) for c in COLUMNS) + " |")


def main(args: argparse.Namespace) -> None:
    rows = [summarize_run(run, args.volumes) for run in args.runs]
    print_markdown(rows)

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {args.csv}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Summarize runs (config, cost, 3D metrics) into one table')
    parser.add_argument('runs', type=Path, nargs='+', help="Run folders, e.g. results/E001 results/E002")
    parser.add_argument('--volumes', type=Path, default=Path("volumes"),
                        help="Root of the infer.py outputs, to find <volumes>/<run>/timing.txt")
    parser.add_argument('--csv', type=Path, default=None, help="Optional csv output")

    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
