#!/usr/bin/env python3
# Per-patient continuity of the thin tubes (esophagus=1, trachea=3) for the CV
# predictions of 2D (E_F06), 2.5D (E_F05) and 3D (E_F07), final post-processing.
# "missed slices": axial slices where GT has the organ and the prediction has none.
# "pieces": 3D connected components of the prediction.
# Run: .venv/bin/python analysis/tube_continuity_stats.py
import glob, json
from pathlib import Path
import numpy as np, nibabel as nib
from scipy.ndimage import label

R = Path(__file__).resolve().parents[1]
CV = R / "results/cv5"
GTDIR = R / "data/segthor_train_full/train"
MODELS = {"2D": "E_F06_unet2d_adamw_cosine_150ep", "2.5D": "E_F05_unet25d_adamw_cosine_150ep",
          "3D": "E_F07_unet3d_adamw_cosine"}
ORG = {"esophagus": 1, "trachea": 3}

def pred_path(run, pid):
    for f in range(5):
        p = CV / f"{run}_f{f}/eval/pp_final/nii/{pid}.nii.gz"
        if p.exists(): return p
    raise FileNotFoundError(pid)

def dice(a, b):
    s = a.sum() + b.sum()
    return 2 * (a & b).sum() / s if s else 1.0

rows = []
pids = sorted(p.name for p in GTDIR.iterdir())
for pid in pids:
    gt = np.asarray(nib.load(GTDIR / pid / "GT.nii.gz").dataobj).astype(np.uint8)
    row = {"pid": pid}
    for m, run in MODELS.items():
        pr = np.asarray(nib.load(pred_path(run, pid)).dataobj).astype(np.uint8)
        for o, c in ORG.items():
            g, p = gt == c, pr == c
            gz, pz = g.any((0, 1)), p.any((0, 1))
            row[f"{m}_{o}_dice"] = float(dice(g, p))
            row[f"{m}_{o}_missed"] = int((gz & ~pz).sum())
            row[f"{m}_{o}_pieces"] = int(label(p)[1])
            row[f"{o}_gt_slices"] = int(gz.sum())
    rows.append(row); print(pid, {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items() if "esophagus" in k})
out = CV / "tube_continuity_stats.json"
json.dump(rows, open(out, "w"), indent=1)
for o in ORG:
    print(f"\n== {o}")
    for m in MODELS:
        d = np.array([r[f"{m}_{o}_dice"] for r in rows]); ms = np.array([r[f"{m}_{o}_missed"] for r in rows])
        pc = np.array([r[f"{m}_{o}_pieces"] for r in rows])
        print(f"{m:5s} dice {d.mean():.3f}  missed slices total {ms.sum()} (patients with any {np.sum(ms>0)})  pieces median {np.median(pc)} mean {pc.mean():.2f}  >1 piece: {np.sum(pc>1)}")
print("saved", out)
