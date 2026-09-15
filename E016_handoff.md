# E016_aug_noelastic — handoff for the next agent

Status: **registered, not yet run**. This file is committed so the next agent
can pick up exactly here. Everything below is verified from the actual runs in
this repo (results/ and volumes/ are gitignored but present locally).

## Goal of this job

Prepare E016 (config + registry + this handoff), commit, and **push the branch
`francesco/augmentation` to GitHub** (`origin` = deboradum/AI4M). The next
agent then runs E016, records the result, and pushes again.

## Context: why E016 exists

E015_augment (parent E001) was the on-the-fly augmentation ablation
(rotation ±15°, scale 0.9–1.1, intensity ±0.1, elastic α=10, no h-flip,
25 epochs). Its 3D result:

| | esophagus | heart | trachea | fg mean |
|---|---|---|---|---|
| E001 (parent) | 0.610 | 0.808 | 0.377 | 0.598 |
| E015_augment | 0.640 | 0.883 | **0.081** | 0.534 |

Augmentation helped heart (0.883, best of all runs) and esophagus (0.640 >
 E001)
and narrowed the train/val gap (confirmed), but the **elastic term collapsed
the thin trachea** (0.081 vs 0.377; even train present-only trachea was 0.023 —
the model never fits trachea under elastic warp). E016 drops elastic and runs
50 epochs to let the augmented model converge.

## Two bugs found and fixed during E015 (do not re-introduce)

1. **DataLoader shared-memory crash** (macOS/MPS, `num_workers=5`): the run
   died at epoch 4 with `RuntimeError: Shared memory manager connection has
   timed out`. Fix: `num_workers: 0` in the config (throughput-only change;
   data/split/seed/model unchanged). E015 and E016 configs already have it.
2. **Elastic augmentation mis-scaling** (the big one): `_sample_elastic_displacement`
   in `dataset.py` added the field directly to torchvision's identity grid,
   which is normalized to [-1, 1] — so α=10 meant up to ±600 px warp on 512 px
   images, i.e. scrambled noise, and the model learned nothing (all-background
   predictions; the "0.71 val Dice" was an artifact of scoring empty slices
   as 1.0). **Fixed** in `dataset.py`: the field is now scaled by
   `[2.0/w, 2.0/h]` so α is true pixel displacement. Verified by a 5-epoch
   A/B: heart val present-only 0.535 (aug) vs 0.618 (no-aug) at e4 — the
   fixed model learns organs. **This fix is committed — keep it.**

## What is prepared (committed)

- `configs/E016_aug_noelastic.yaml` — E001 + `augment=true`, rotation 15,
   scale 0.9–1.1, intensity 0.1, **`aug_elastic_alpha: 0.0`**, `epochs: 50`,
   `num_workers: 0`, seed 123, ENet 8/2, in_slices 1.
- `experiments/E016_aug_noelastic/config.json` — status `registered`,
   hypothesis, `what_to_check` (trachea recovery, fg mean vs E001 0.598 and
   E014_hu_wide 0.737, heart/esophagus gains hold, 50 vs 25 epochs,
   present-only val Dice > 0 for all organs).
- `EXPERIMENTS.md` — E016 row added, status `registered`.
- This file.

## How to run E016 (local MPS, ~2 h for 50 epochs)

Repo root: `/Users/francesco/Desktop/UVA MSC/AI4MI/AI4M`
Venv python: `./ai4mi/bin/python` (absolute path works too).

1. **Cheap gate first** (5-epoch probe, ~12 min) — same pattern that validated
   the elastic fix. It must show trachea present-only val Dice > 0 at e4–5
   (E015's broken run was 0.000; E015 fixed was heart 0.535 but trachea ~0 at
   e4 — trachea comes later, so also check train present-only):
```
rm -rf results/E016_probe
./ai4mi/bin/python -O main.py --config <(sed 's/epochs: 50/epochs: 5/' configs/E016_aug_noelastic.yaml) \
  --dest results/E016_probe --mps
./ai4mi/bin/python -c "
import numpy as np
dv=np.load('results/E016_probe/dice_val.npy'); pv=np.load('results/E016_probe/present_val.npy')
print([round(float(dv[-1,pv[-1,:,k],k].mean()),3) if pv[-1,:,k].any() else 'n/a' for k in range(1,5)])
"
```
   Gate: exit 0, no traceback, and **heart present-only > 0.3** at final epoch.
   (Trachea may still be ~0 at e4 — it's the slowest organ; the full run must
   show it > 0 by the end. If heart is 0.000 too, stop and re-check the
   elastic fix / config before the long run.)

2. **Full run** (supervised background so it survives):
```
hub start name=e016-train \
  application=/Users/francesco/Desktop/UVA MSC/AI4MI/AI4M/ai4mi/bin/python \
  args=[-O, main.py, --config, configs/E016_aug_noelastic.yaml, --dest, results/E016_aug_noelastic, --mps] \
  cwd=/Users/francesco/Desktop/UVA MSC/AI4MI/AI4M \
  ready.log=">>> Setting up to train"
hub wait name=e016-train for=exit timeout=7200
```
   ~50 epochs × ~2.5 min ≈ 2 h. If `hub start` is unusable, use `bash`
   `async: true` with the same command and poll
   `jq '.epochs_done' results/E016_aug_noelastic/stats.json` until 50.
   Do NOT kill while epochs_done is advancing.

3. **Verify training**:
```
jq '{epochs_done, best_epoch, total_min, n_params}' results/E016_aug_noelastic/stats.json   # 50
./ai4mi/bin/python -c "
import numpy as np
dv=np.load('results/E016_aug_noelastic/dice_val.npy'); pv=np.load('results/E016_aug_noelastic/present_val.npy')
print('val present-only:', [round(float(dv[-1,pv[-1,:,k],k].mean()),3) if pv[-1,:,k].any() else 'n/a' for k in range(1,5)])
"
```
   **All three organs must be > 0** (the E015-broken failure mode was all 0).
   Copy the hub output.log over train.log if needed:
```
cp ~/.omp/run/daemons/*/daemons/e016-train/output.log results/E016_aug_noelastic/train.log
```

4. **Infer + stitch** (CPU; `infer.py` has no --mps; num_workers from config
   dump = 0):
```
./ai4mi/bin/python -O infer.py --config results/E016_aug_noelastic/config_dump.yaml \
  --weights results/E016_aug_noelastic/bestweights.pt \
  --img_folder data/SEGTHOR/val/img --dest volumes/E016_aug_noelastic \
  --scan_pattern 'data/segthor_part1/train/{id_}/{id_}.nii.gz'
```
   Expect 915 slices, 5 patients, `volumes/E016_aug_noelastic/nii/` with real
   organ voxels (E015's broken run had zero — check
   `./ai4mi/bin/python -c "import nibabel as nib, numpy as np; v=nib.load('volumes/E016_aug_noelastic/nii/Patient_01.nii.gz').get_fdata().astype(int); print({k:int((v==k).sum()) for k in range(5)})"`).

5. **3D metrics**:
```
./ai4mi/bin/python -O metrics3d.py --pred_folder volumes/E016_aug_noelastic/nii \
  --gt_pattern 'data/segthor_part1/train/{id_}/GT.nii.gz' \
  --scan_pattern 'data/segthor_part1/train/{id_}/{id_}.nii.gz' \
  --class_names background esophagus heart trachea aorta \
  --dest results/E016_aug_noelastic/metrics3d --process 4
```
   Aorta = n/a (absent from segthor_part1).

## Success criteria (the actual experiment result)

- Trachea 3D Dice recovers toward E001 (0.377) / E014_hu_wide (0.653) — the
   whole point of E016.
- Heart holds ~0.88 and esophagus ≥ 0.64 (E015's gains).
- Foreground mean beats E001 (0.598); ideally approaches E014_hu_wide (0.737).
- If trachea stays ~0.1 even without elastic, report it as a real negative
   result ( note it — do not fabricate a win).

## Record the result (mirror E015/E014 exactly)

- `experiments/E016_aug_noelastic/config.json`: status → `complete`,
   `git_commit` = the commit that will contain the record (see commit step),
   fill `results` block (epochs_completed 50, best_epoch, training total_min/
   device/params, inference seconds_per_patient/device/validation_patients 5,
   metrics_3d dice/hd95/assd/nsd per class + aorta null + foreground_mean,
   comparison_to_parent deltas vs E001, notes incl. "elastic dropped after
   E015 trachea collapse", artifacts paths), answer `what_to_check` in place.
- `EXPERIMENTS.md`: E016 row status → `complete`, fill Final result + Notes.
- `python summarize.py results/E016_aug_noelastic --volumes volumes` — row
   must be fully populated.

## Commit + push (the goal of this job)

Only source/registry files. Never add `results/`, `volumes/`, `data/`.
```
git add EXPERIMENTS.md configs/E016_aug_noelastic.yaml \
        experiments/E016_aug_noelastic/config.json
git commit -m "Run E016_aug_noelastic (no elastic, 50 epochs); record result"
git push origin francesco/augmentation
```
The `git_commit` field is self-referential: commit once, read the hash, set
`git_commit` in config.json, commit again (as done for E015:
`0bba233` + `ece6049`). Then push.

## Knowns / gotchas for the next agent

- Local MPS is ~2.5 min/epoch (vs ~40 s on Snellius). 50 epochs ≈ 2 h.
- `main.py` has no resume; a crash means restart from scratch (dest dir is
   overwritten each run; epoch 0 always beats best_dice=0 so bestweights
   re-saved).
- The "val Dice ~0.7" in stats.json is NOT trustworthy on its own — always
   check present-only dice (dice_val.npy + present_val.npy) and the 3D
   metrics.csv. That artifact hid the E015 broken run.
- `dataset.py` elastic fix is committed and required for correct aug.
- Tests: `python -m unittest discover -s tests -v` and `git diff --check`
   before the final push (TEAM_WORKFLOW.md step 6/9).
