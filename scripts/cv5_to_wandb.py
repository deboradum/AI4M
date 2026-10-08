#!/usr/bin/env python3
# MIT License
#
# Write a CV fold's final scores (lastweights.pt on the fold's val patients, raw and with
# the final post-processing) into the summary of that fold's W&B training run, as
# cv/<raw|pp>/<metric>_<organ> and cv/<raw|pp>/<metric>_mean. Called by scripts/run_cv5.sh;
# safe to rerun (it overwrites the same keys).
#
#   python scripts/cv5_to_wandb.py --dest results/cv5/E_F07_unet3d_adamw_cosine_f1
import os
import argparse
from pathlib import Path

import numpy as np

ORGANS = ["esophagus", "heart", "trachea", "aorta"]
METRICS = ["dice", "iou", "hd95", "assd", "nsd"]
# E_F07 fold 0 is the original full run (same split), see run_cv5.sh
REUSED_RUN = Path("results/full/E_F07_unet3d_adamw_cosine")


def scores(folder: Path) -> dict[str, float]:
    out = {}
    for m in METRICS:
        npz = np.load(folder / f"{m}.npz")
        per_organ = np.nanmean(np.stack([npz[p][1:] for p in sorted(npz.files)]), 0)
        out.update({f"{m}_{o}": float(v) for o, v in zip(ORGANS, per_organ)})
        out[f"{m}_mean"] = float(per_organ.mean())
    out["n_patients"] = len(npz.files)
    return out


def main(args: argparse.Namespace) -> None:
    import wandb
    run_dir = REUSED_RUN if (args.dest / "REUSED").exists() else args.dest
    latest = run_dir / "wandb" / "latest-run"
    if not latest.exists():
        print(f"no W&B run under {run_dir}, skipping")
        return
    run_id = os.path.realpath(latest).rsplit("-", 1)[-1]
    summary = {f"cv/{tag}/{k}": v for tag, sub in [("raw", "eval/metrics"), ("pp", "eval/pp_final/metrics")]
               for k, v in scores(args.dest / sub).items()}
    run = wandb.Api().run(f"{args.entity}/{args.project}/{run_id}")
    run.summary.update(summary)
    print(f"{run.name} ({run_id}): cv/pp/dice_mean {summary['cv/pp/dice_mean']:.4f}, "
          f"cv/pp/hd95_mean {summary['cv/pp/hd95_mean']:.1f} mm")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dest", type=Path, required=True, help="results/cv5/<exp>")
    p.add_argument("--entity", default="ai4miFrancesco")
    p.add_argument("--project", default="AI4M")
    main(p.parse_args())
