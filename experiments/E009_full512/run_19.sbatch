#!/bin/bash
#SBATCH --job-name=E009_full512
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=E009-%j.log

set -euo pipefail
cd "$HOME/ai4mi_project"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ai4mi

python -u -O train_19.py
