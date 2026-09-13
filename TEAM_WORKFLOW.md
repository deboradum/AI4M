# Team experiment workflow

The original [readme.md](readme.md) is the course README. This guide explains
the practical group workflow for reproducible SegTHOR experiments.

```text
choose a parent → create a config → test → GPU preflight → train
→ infer and stitch → compute 3D metrics → record the result
```

## 1. Work on a branch

Do not develop directly on `main`.

```bash
cd /gpfs/home3/<username>/projects/AI4M
git switch -c <your-branch>
git status --short --branch
```

Keep the worktree clean before starting a new run. Make small, reviewable
commits rather than one large commit at the end.

## 2. Environment

Keep the virtual environment outside the repository so Git never sees it. On
Snellius, create it with the Python module used by the GPU launchers:

```bash
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
python -m venv /home/<username>/.venvs/ai4mi
source /home/<username>/.venvs/ai4mi/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a later session, load the same modules and activate the same environment.
The current Slurm launchers activate `/home/scur0204/.venvs/ai4mi`; teammates
must use an approved launcher setup that activates their own environment.

## 3. Prepare data once

Source data and generated slices are ignored by Git. Do not commit the ZIP,
NIfTI source files, or `data/SEGTHOR/`. All experiments share one prepared
SegTHOR dataset; do not create a copied 2.5D dataset.

```bash
sha256sum -c data/segthor_part1.sha256

python slice_segthor.py \
  --source_dir data/segthor_part1 \
  --dest_dir data/SEGTHOR_tmp \
  --shape 256 256 \
  --retains 5 \
  --seed 0 \
  --fold 0 \
  --process 1

mv data/SEGTHOR_tmp data/SEGTHOR
```

Only promote the temporary directory if `data/SEGTHOR/` does not already
exist. Every comparable experiment must use the same split and preprocessing.

## 4. Register an ablation before running it

Choose a parent experiment, normally `E001` for an isolated comparison. Use a
unique name in the existing style, for example `E010_large_enet`.

Before training, create:

- `configs/E010_large_enet.yaml` — the runnable configuration;
- `experiments/E010_large_enet/config.json` — parent, hypothesis, and result;
- one matching row in `EXPERIMENTS.md`.

Preserve the parent data, split, seed, loss, optimizer, batch size, epochs,
and metrics. Change only the variable being tested.

### Input-slice rule

```yaml
in_slices: 1  # normal 2D experiment; also the default
in_slices: 3  # explicit 2.5D input: [z-1, z, z+1]
```

Shared code does not make every experiment 2.5D. E001 and E010 are one-slice
experiments; E008 is the explicit 2.5D experiment.

## 5. Shared code versus ablation files

`main.py`, `dataset.py`, `infer.py`, and `configType.py` are shared
infrastructure. Do not create copies such as `main_E010.py`.

Most ablations need only a config and experiment record. If a new reusable
capability is genuinely needed, make the smallest backward-compatible change,
add a focused test, and ensure inference supports it too. Defaults must keep
the original one-slice behavior.

## 6. Test and preflight

From the repository root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m unittest discover -s tests -v

git diff --check
git status --short
```

Then run a real one-batch GPU preflight. It tests data loading, model, loss,
backward pass, optimizer, CUDA, and target shape:

```bash
mkdir -p results/E010_large_enet

sbatch \
  --job-name=E010_large_preflight \
  --output=results/E010_large_enet/preflight-%j.out \
  --error=results/E010_large_enet/preflight-%j.err \
  scripts/train_segthor_h100.sbatch \
  configs/E010_large_enet.yaml \
  results/E010_large_enet \
  --preflight-only
```

Continue only when the output reports `Preflight passed` and contains no
exception.

## 7. Train

Use a unique results folder for every experiment:

```bash
sbatch \
  --job-name=E010_large_enet \
  --output=results/E010_large_enet/slurm-%j.out \
  --error=results/E010_large_enet/slurm-%j.err \
  scripts/train_segthor_h100.sbatch \
  configs/E010_large_enet.yaml \
  results/E010_large_enet
```

After completion, verify the planned epoch count in `stats.json` and the
presence of `bestweights.pt`. Progress bars in `.err` are normal; a traceback,
CUDA error, or `ERROR:` marker is not.

## 8. Infer, stitch, and evaluate in 3D

Inference uses the saved `config_dump.yaml`, so it recreates the trained model:

```bash
mkdir -p volumes/E010_large_enet

sbatch \
  --job-name=E010_large_infer \
  --output=volumes/E010_large_enet/slurm-%j.out \
  --error=volumes/E010_large_enet/slurm-%j.err \
  scripts/infer_segthor_h100.sbatch \
  results/E010_large_enet/config_dump.yaml \
  results/E010_large_enet/bestweights.pt \
  data/SEGTHOR/val/img \
  volumes/E010_large_enet \
  'data/segthor_part1/train/{id_}/{id_}.nii.gz'
```

Compare stitched patient volumes, not only training's slice-level Dice:

```bash
python metrics3d.py \
  --pred_folder volumes/E010_large_enet/nii \
  --gt_pattern 'data/segthor_part1/train/{id_}/GT.nii.gz' \
  --scan_pattern 'data/segthor_part1/train/{id_}/{id_}.nii.gz' \
  --class_names background esophagus heart trachea aorta \
  --dest results/E010_large_enet/metrics3d \
  --process 4
```

Use the same command pattern for every run. Aorta is absent in `segthor_part1`,
so report it as `n/a`. Summarize completed runs with:

```bash
python summarize.py results/E001_baseline results/E010_large_enet \
  results/E010_unet2d --volumes volumes
```

## 9. Record and commit results

After 3D metrics finish, update the matching registry row and
`experiments/<experiment-id>/config.json` with status, commit, cost, and mean
3D metrics.

Commit source, tests, configs, and compact experiment records. Do not commit
generated artifacts: `results/`, `volumes/`, `data/SEGTHOR/`, or source ZIPs.

Before opening a merge request, confirm tests and `git diff --check` pass,
every run has a unique config and record, and no ignored runtime files were
force-added.
