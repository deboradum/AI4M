#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=14:00:00
# Usage (from repo root), once per partition so the first free GPU wins:
#   for p in gpu_a100 gpu_h100; do
#     sbatch --partition=$p --job-name=<exp> --output=results/full/<exp>/slurm-%j-$p.out \
#       scripts/train_eval_full.sbatch <exp>
#   done
# Trains configs/full/<exp>.yaml, then infers, stitches and scores 3D metrics on the val set.
set -euo pipefail
trap 'status=$?; echo "ERROR: line $LINENO: $BASH_COMMAND (exit $status)" >&2' ERR
exp="$1"
cd "${SLURM_SUBMIT_DIR:?}"
config="configs/full/$exp.yaml"
dest="results/full/$exp"
# Each run is submitted to both gpu_a100 and gpu_h100 under the same job name;
# the first to start claims the run and cancels its pending twin.
mkdir -p "$dest"
if ! mkdir "$dest/.claim" 2>/dev/null; then echo "Twin already claimed $exp; exiting"; exit 0; fi
scancel --name="$SLURM_JOB_NAME" --state=PENDING --user="$USER" || true
echo "Running $exp on $(hostname), partition $SLURM_JOB_PARTITION, commit $(git rev-parse --short HEAD)"
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source ~/.venvs/ai4mi/bin/activate
# W&B logs online with the credentials in ~/.netrc (entity ai4miFrancesco)
nvidia-smi --query-gpu=name --format=csv,noheader

mkdir -p "$dest"
python -u scripts/preflight_segthor.py --config "$config"
python -u main.py --config "$config" --dest "$dest" --gpu 2> "$dest/train.err" | tee "$dest/train.log" | { grep -E "Improved|Early|Error" || true; }

# Infer on the val slices of the dataset the run trained on; its preprocess.json
# tells infer.py whether to undo a resampling when stitching.
dataset=$(sed -n 's/^dataset: *"\{0,1\}\([^"]*\)"\{0,1\} *$/\1/p' "$dest/config_dump.yaml")
[ -f "data/$dataset/preprocess.json" ] || { echo "ERROR: data/$dataset/preprocess.json missing, re-slice it" >&2; exit 1; }
python -u infer.py --config "$dest/config_dump.yaml" --weights "$dest/bestweights.pt" \
  --img_folder "data/$dataset/val/img" --dest "$dest/eval" \
  --scan_pattern 'data/segthor_train_full/train/{id_}/{id_}.nii.gz' --gpu 2>&1 | tail -5 | tee "$dest/infer.log"
python -u metrics3d.py --pred_folder "$dest/eval/nii" \
  --gt_pattern 'data/segthor_train_full/train/{id_}/GT.nii.gz' \
  --scan_pattern 'data/segthor_train_full/train/{id_}/{id_}.nii.gz' \
  --class_names background esophagus heart trachea aorta \
  --dest "$dest/eval/metrics" --process 8 2>&1 | tail -30
echo "DONE $exp"
