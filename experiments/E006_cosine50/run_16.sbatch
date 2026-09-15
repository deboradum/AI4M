#!/bin/bash
#SBATCH --job-name=E006_cosine50
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=08:00:00
#SBATCH --output=E006-%j.log

set -euo pipefail

cd "$HOME/ai4mi_project"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ai4mi

python -u -O train_16.py \
  --epochs 50 \
  --seed 123 \
  --dest "results/E006_cosine50_${SLURM_JOB_ID}"
