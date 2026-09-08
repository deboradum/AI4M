#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=08:00:00

# Usage, from the repository root:
#   sbatch --job-name=E001_baseline \
#     --output=results/E001_baseline/slurm-%j.out \
#     --error=results/E001_baseline/slurm-%j.err \
#     scripts/train_segthor_h100.sbatch configs/E001_baseline.yaml results/E001_baseline

set -euo pipefail
trap 'status=$?; echo "ERROR: line $LINENO: $BASH_COMMAND (exit $status)" >&2' ERR

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <config.yaml> <results-dir>" >&2
    exit 2
fi

config="$1"
dest="$2"
# Slurm copies this script to a spool directory before execution, so
# BASH_SOURCE points to /var/spool/... on the compute node. The submission
# directory is the stable repository path.
repo_dir="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
cd "$repo_dir"
echo "Running on $(hostname) from $repo_dir"
echo "Config: $config"
echo "Results: $dest"

# This virtual environment was created with the system Python 3.11.13. Do not
# load the cluster Python 3.11.3 module here: mixing interpreters under one
# venv can make imports fail before training starts.
source /home/scur0204/.venvs/ai4mi/bin/activate
echo "Python: $(command -v python)"
python --version

test -f "$config"
mkdir -p "$dest"

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
python -u main.py --config "$config" --dest "$dest" --gpu
