#!/bin/bash
# Post-processing variants on the 5-fold CV raw predictions (results/cv5/<run>/eval/nii):
# each variant writes results/cv5/<run>/eval/pp_<name>/{nii,metrics}. CPU only.
#   scripts/cv5_pp_sweep.sh            # all finished folds, all variants
set -euo pipefail
PY=${PY:-.venv/bin/python}
declare -A V=(
  [t10]="--lcc 2 --keep_near 3:10 --min_ml 4:1"
  [t20]="--lcc 2 --keep_near 3:20 --min_ml 4:1"
  [t10e10]="--lcc 2 --keep_near 3:10 1:10 --min_ml 4:1"
  [t20e20]="--lcc 2 --keep_near 3:20 1:20 --min_ml 4:1"
)
MET='--gt_pattern data/segthor_train_full/train/{id_}/GT.nii.gz --scan_pattern data/segthor_train_full/train/{id_}/{id_}.nii.gz --class_names background esophagus heart trachea aorta --process 4'
jobs_list() {
  for d in results/cv5/E_F0*_f[0-4]; do
    [ -f "$d/eval/metrics/nsd.npz" ] || continue
    for v in "${!V[@]}"; do echo "$d $v ${V[$v]}"; done
  done
}
run_one() {
  local d=$1 v=$2; shift 2
  local out=$d/eval/pp_$v
  [ -f "$out/metrics/nsd.npz" ] && return 0
  $PY postprocess.py --input_folder "$d/eval/nii" --output_folder "$out/nii" "$@" --process 2 > "$out.log" 2>&1
  $PY metrics3d.py --pred_folder "$out/nii" --dest "$out/metrics" $MET >> "$out.log" 2>&1
  echo "done $out"
}
export -f run_one; export PY MET
jobs_list | xargs -P ${J:-24} -L 1 bash -c 'run_one "$@"' _
