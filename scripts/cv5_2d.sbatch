#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=12:00:00
# 5-fold CV of the 2D (E_F06) and 2.5D (E_F05) U-Nets: one fold per queue entry, both
# models trained side by side on the same GPU (run_cv5.sh: the GPU saturates at 2 jobs,
# 2 x 111 slices/s vs 195 for one). The 3D folds (E_F07) run elsewhere.
# Usage (from repo root), once per fold and partition so the first free GPU wins:
#   for k in 0 1 2 3 4; do for p in gpu_h100 gpu_a100; do
#     sbatch --partition=$p --job-name=cv5_2d_f$k --output=results/cv5/slurm-cv5_2d_f$k-%j-$p.out \
#       scripts/cv5_2d.sbatch $k
#   done; done
set -euo pipefail
k="$1"
cd "${SLURM_SUBMIT_DIR:?}"
# Each fold is submitted to both gpu_h100 and gpu_a100 under the same job name;
# the first to start claims the fold and cancels its pending twin.
mkdir -p results/cv5
if ! mkdir "results/cv5/.claim_2d_f$k" 2>/dev/null; then echo "Twin already claimed fold $k; exiting"; exit 0; fi
scancel --name="$SLURM_JOB_NAME" --state=PENDING --user="$USER" || true
echo "cv5 2D/2.5D fold $k on $(hostname), partition $SLURM_JOB_PARTITION, commit $(git rev-parse --short HEAD)"
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source ~/.venvs/ai4mi/bin/activate
# W&B logs online with the credentials in ~/.netrc (entity ai4miFrancesco)
nvidia-smi --query-gpu=name --format=csv,noheader

PY=python scripts/run_cv5.sh -j 2 E_F05_unet25d_adamw_cosine_150ep_f$k E_F06_unet2d_adamw_cosine_150ep_f$k
# run_cv5.sh reports failures without failing; make them visible to sacct
for exp in E_F05_unet25d_adamw_cosine_150ep_f$k E_F06_unet2d_adamw_cosine_150ep_f$k; do
  [ -f "results/cv5/$exp/eval/pp_final/metrics/dice.npz" ] || { echo "ERROR: $exp not finished" >&2; rm -rf "results/cv5/.claim_2d_f$k"; exit 1; }
done
echo "DONE fold $k"
