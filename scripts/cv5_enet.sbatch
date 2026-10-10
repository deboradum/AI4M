#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
# 5-fold CV of the ENet (E_F00 recipe + U-Net data and CE+Dice loss) (E_F11), so the baseline is scored with the same protocol as the
# U-Nets (each of the 40 patients held out once, final-epoch weights, 3D metrics on the CT grid).
# Uses data/SEGTHOR_FULL_huwide_resampled_cv0..4 (make cv5_data), the same data as the U-Nets: ENet gets the
# same preprocessing, so ENet vs U-Net isolates the model.
# Usage (from repo root): for k in 0 1 2 3 4; do sbatch --partition=gpu_a100 --job-name=cv5_enet_f$k \
#   --output=results/cv5/slurm-cv5_enet_f$k-%j.out scripts/cv5_enet.sbatch $k; done
set -euo pipefail
k="$1"
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p results/cv5
echo "cv5 ENet fold $k on $(hostname), commit $(git rev-parse --short HEAD)"
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source ~/.venvs/ai4mi/bin/activate
exp=E_F11_enet_ce_dice_f$k
PY=python scripts/run_cv5.sh -j 1 "$exp"
[ -f "results/cv5/$exp/eval/pp_final/metrics/dice.npz" ] || { echo "ERROR: $exp not finished" >&2; exit 1; }
echo "DONE fold $k"
