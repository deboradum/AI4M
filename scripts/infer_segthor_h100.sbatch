#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# Usage, from the repository root:
#   sbatch --job-name=E001_infer \
#     --output=volumes/E001_baseline/slurm-%j.out \
#     --error=volumes/E001_baseline/slurm-%j.err \
#     scripts/infer_segthor_h100.sbatch \
#     results/E001_baseline/config_dump.yaml \
#     results/E001_baseline/bestweights.pt \
#     data/SEGTHOR/val/img \
#     volumes/E001_baseline \
#     'data/segthor_part1/train/{id_}/{id_}.nii.gz'

set -euo pipefail
trap 'status=$?; echo "ERROR: line $LINENO: $BASH_COMMAND (exit $status)" >&2' ERR

if [[ $# -ne 5 ]]; then
    echo "Usage: $0 <config.yaml> <weights.pt> <img-folder> <dest> <scan-pattern>" >&2
    exit 2
fi

config="$1"
weights="$2"
img_folder="$3"
dest="$4"
scan_pattern="$5"

repo_dir="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
cd "$repo_dir"
echo "Running on $(hostname) from $repo_dir"
echo "Config: $config"
echo "Weights: $weights"
echo "Images: $img_folder"
echo "Results: $dest"

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source /home/scur0204/.venvs/ai4mi/bin/activate

test -f "$config"
test -f "$weights"
test -d "$img_folder"
mkdir -p "$dest"

python -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))"
python -u infer.py --config "$config" --weights "$weights" --img_folder "$img_folder" \
  --dest "$dest" --scan_pattern "$scan_pattern" --gpu
