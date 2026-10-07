#!/bin/bash
# Local (non-Slurm) 3D run: train, infer on val, stitch, score 3D metrics.
# Usage, from the repo root:  scripts/train_eval_3d.sh E_F07_unet3d_adamw_cosine
set -euo pipefail
exp="$1"
config="configs/full/$exp.yaml"
dest="results/full/$exp"
PY=${PY:-.venv/bin/python}
mkdir -p "$dest"
echo "Running $exp on $(hostname), commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name --format=csv,noheader

$PY -u train3d.py --config "$config" --dest "$dest" --gpu 2> "$dest/train.err" | tee "$dest/train.log"

$PY -u infer3d.py --config "$dest/config_dump.yaml" --weights "$dest/bestweights.pt" \
  --img_folder data/SEGTHOR_FULL_huwide/val/img --dest "$dest/eval" \
  --scan_pattern 'data/segthor_train_full/train/{id_}/{id_}.nii.gz' --gpu 2>&1 | tail -5 | tee "$dest/infer.log"
$PY -u metrics3d.py --pred_folder "$dest/eval/nii" \
  --gt_pattern 'data/segthor_train_full/train/{id_}/GT.nii.gz' \
  --scan_pattern 'data/segthor_train_full/train/{id_}/{id_}.nii.gz' \
  --class_names background esophagus heart trachea aorta \
  --dest "$dest/eval/metrics" --process 8 2>&1 | tail -30
echo "DONE $exp"
