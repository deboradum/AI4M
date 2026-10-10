#!/usr/bin/env python3
# MIT License
#
# Boundary quality across tolerances: for every patient, model and organ, histograms of the
# surface distances metrics3d.py uses (prediction surface -> GT surface and back, in mm),
# so NSD can be read at any tolerance tau instead of only at 2 mm:
#   NSD(tau) = (#pred-surface voxels within tau + #GT-surface voxels within tau) / (#both)
# Only patients every model has (pooled 5-fold CV, post-processed predictions). 8 Oct 2026.
#
#   python scripts/boundary_curves.py --dest results/cv5/boundary_hist.npz
import sys
import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics3d import load_labels, surface_distances  # noqa: E402

MODELS = [("E_F11_enet_ce_dice", "ENet (baseline)"), ("E_F06_unet2d_adamw_cosine_150ep", "2D U-Net"),
          ("E_F05_unet25d_adamw_cosine_150ep", "2.5D U-Net"), ("E_F07_unet3d_adamw_cosine", "3D U-Net")]
# Upper edges sit just above multiples of 0.1 mm so that "<= tau" (as metrics3d.py) is a cumulative sum
EDGES = np.concatenate([[0], np.arange(0.1, 20, 0.1) + 1e-6, [20, 30, 50, 100, 1e9]])


def preds(root: Path, run: str) -> dict[str, Path]:
    out = {}
    for k in range(5):
        for f in (root / f"{run}_f{k}" / "eval/pp_final/nii").glob("Patient_*.nii.gz"):
            out[f.name.split(".")[0]] = f
    return out


def one(job):
    pid, paths = job
    gt, _ = load_labels(Path(f"data/segthor_train_full/train/{pid}/GT.nii.gz"))
    _, spacing = load_labels(Path(f"data/segthor_train_full/train/{pid}/{pid}.nii.gz"))
    out = np.zeros((len(paths), 4, 2, len(EDGES) - 1), dtype=np.int64)  # model, organ, direction, bin
    for m, p in enumerate(paths):
        pred, _ = load_labels(p)
        for k in range(1, 5):
            g, q = gt == k, pred == k
            if not g.any() or not q.any():
                continue
            idx = np.argwhere(g | q)
            sl = tuple(slice(max(a - 3, 0), b + 4) for a, b in zip(idx.min(0), idx.max(0)))
            d_pg, d_gp = surface_distances(q[sl], g[sl], spacing)
            out[m, k - 1, 0] = np.histogram(d_pg, EDGES)[0]
            out[m, k - 1, 1] = np.histogram(d_gp, EDGES)[0]
    return pid, out


def main(a: argparse.Namespace) -> None:
    per_model = [preds(a.root, run) for run, _ in MODELS]
    ids = sorted(set.intersection(*[set(d) for d in per_model]))
    jobs = [(pid, [d[pid] for d in per_model]) for pid in ids]
    with Pool(a.workers) as pool:
        res = dict(pool.map(one, jobs))
    hist = np.stack([res[pid] for pid in ids])  # patient, model, organ, direction, bin
    np.savez_compressed(a.dest, hist=hist, edges=EDGES, patients=np.array(ids),
                        models=np.array([n for _, n in MODELS]))
    print(f"Wrote {a.dest}: {len(ids)} patients x {len(MODELS)} models")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("results/cv5"))
    p.add_argument("--dest", type=Path, default=Path("results/cv5/boundary_hist.npz"))
    p.add_argument("--workers", type=int, default=32)
    main(p.parse_args())
