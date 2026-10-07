"""Measure plan 1.7 (largest 3D connected component per class) on existing
prediction volumes, through the exact same metric code as metrics3d.py."""
import sys
import numpy as np
import nibabel as nib
from scipy.ndimage import label as nd
from metrics3d import class_metrics

K = 5
CLASSES = ['background', 'esophagus', 'heart', 'trachea', 'aorta']
VAL = ['Patient_01', 'Patient_11', 'Patient_15', 'Patient_17', 'Patient_19']
GT = 'data/SEGTHOR_aorta/train/{}/GT.nii.gz'
CT = 'data/SEGTHOR_aorta/train/{}/{}.nii.gz'


def keep_largest_per_class(vol):
    out = np.zeros_like(vol)
    removed = {}
    for k in range(1, K):
        mask = vol == k
        if not mask.any():
            continue
        lab, n = nd(mask)
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        big = sizes.argmax()
        out[lab == big] = k
        removed[CLASSES[k]] = int(mask.sum() - sizes[big])
    return out, removed


def score(run, volumes_root='volumes'):
    rows = {c: {} for c in CLASSES[1:]}
    for pid in VAL:
        gt = np.asarray(nib.load(GT.format(pid)).dataobj).astype(np.uint8)
        pred = np.asarray(nib.load(f'{volumes_root}/{run}/nii/{pid}.nii.gz').dataobj).astype(np.uint8)
        spacing = tuple(float(z) for z in nib.load(CT.format(pid, pid)).header.get_zooms()[:3])
        fixed, removed = keep_largest_per_class(pred)
        for i, k in enumerate([1, 2, 3, 4]):
            before = class_metrics(pred == k, gt == k, spacing, 2.0)
            after = class_metrics(fixed == k, gt == k, spacing, 2.0)
            rows[CLASSES[k]].setdefault('dice_before', []).append(before['dice'])
            rows[CLASSES[k]].setdefault('dice_after', []).append(after['dice'])
            rows[CLASSES[k]].setdefault('hd95_before', []).append(before['hd95'])
            rows[CLASSES[k]].setdefault('hd95_after', []).append(after['hd95'])
            rows[CLASSES[k]].setdefault('removed', []).append(removed[CLASSES[k]])
    return rows


for run in sys.argv[1:]:
    print(f"\n########## {run} ##########")
    rows = score(run)
    print(f"{'class':10s} {'dice before':>11s} {'dice after':>10s} {'delta':>7s} | {'hd95 before':>11s} {'hd95 after':>10s} | {'voxels removed/pixel':>20s}")
    for c in CLASSES[1:]:
        r = rows[c]
        db, da = np.nanmean(r['dice_before']), np.nanmean(r['dice_after'])
        hb, ha = np.nanmean(r['hd95_before']), np.nanmean(r['hd95_after'])
        rm = ', '.join(str(v) for v in r['removed'])
        print(f"{c:10s} {db:11.3f} {da:10.3f} {da-db:+7.3f} | {hb:11.1f} {ha:10.1f} | [{rm}]")
