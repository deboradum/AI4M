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

$PY -u train3d.py --config "$config" --dest "$dest" --gpu --resume 2> "$dest/train.err" | tee "$dest/train.log"

# Infer on the val slices of the dataset the run trained on; resampled datasets
# need --resample so stitch.py maps the 1/1/2.5 mm grid back to the CT grid
# (same convention as train_eval_full.sbatch).
dataset=$(sed -n 's/^dataset: *"\{0,1\}\([^"]*\)"\{0,1\} *$/\1/p' "$dest/config_dump.yaml")
[ -d "data/$dataset/val/img" ] || { echo "ERROR: data/$dataset/val/img missing" >&2; exit 1; }
resample_flag=()
[[ "$dataset" == *_resampled ]] && resample_flag=(--resample)
$PY -u infer3d.py --config "$dest/config_dump.yaml" --weights "$dest/bestweights.pt" \
  --img_folder "data/$dataset/val/img" --dest "$dest/eval" "${resample_flag[@]}" \
  --scan_pattern 'data/segthor_train_full/train/{id_}/{id_}.nii.gz' --gpu 2>&1 | tail -5 | tee "$dest/infer.log"
$PY -u metrics3d.py --pred_folder "$dest/eval/nii" \
  --gt_pattern 'data/segthor_train_full/train/{id_}/GT.nii.gz' \
  --scan_pattern 'data/segthor_train_full/train/{id_}/{id_}.nii.gz' \
  --class_names background esophagus heart trachea aorta \
  --dest "$dest/eval/metrics" --process 8 2>&1 | tail -30
echo "DONE $exp"
