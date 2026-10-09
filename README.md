# AI4MI: SegTHOR organ segmentation

Group project for AI for Medical Imaging (Fall 2026). We start from the course baseline, a small 2D ENet, and improve it one change at a time on the SegTHOR data: CT scans of 40 patients, four organs (esophagus, heart, trachea, aorta).

The course's original README (setup, viewers, submission format, known issues) is now at [docs/COURSE_README.md](docs/COURSE_README.md).

## Results

5-fold cross-validation over all 40 labelled patients. Every patient is held out exactly once and scored with the final-epoch weights, so no checkpoint was chosen on the patients it is scored on. 3D Dice per organ, after post-processing.

| Model | Esophagus | Heart | Trachea | Aorta | Mean Dice [95% CI] | HD95 (mm) | NSD |
|---|---|---|---|---|---|---|---|
| E_F11 ENet (baseline) | 0.650 | 0.900 | 0.815 | 0.850 | 0.803 [0.772, 0.830] | 18.0 | 0.657 |
| E_F06 U-Net 2D | 0.778 | 0.934 | 0.880 | 0.919 | 0.878 [0.864, 0.891] | 10.3 | 0.772 |
| E_F05 U-Net 2.5D | 0.784 | 0.938 | 0.873 | 0.917 | 0.878 [0.864, 0.891] | 10.6 | 0.776 |
| **E_F07 U-Net 3D** | **0.789** | **0.945** | **0.897** | **0.923** | **0.888 [0.880, 0.897]** | **7.8** | **0.795** |

Paired differences per patient (B − A, 40 patients). ✓ means the 95% bootstrap CI excludes zero.

| B − A | Dice | HD95 (mm) | NSD | B better on Dice |
|---|---|---|---|---|
| 2D − ENet | +0.074 ✓ | −7.6 ✓ | +0.115 ✓ | 38/40 |
| 2.5D − ENet | +0.075 ✓ | −7.4 ✓ | +0.119 ✓ | 38/40 |
| 3D − ENet | +0.085 ✓ | −10.2 ✓ | +0.138 ✓ | 39/40 |
| 2.5D − 2D | +0.000 | +0.3 | +0.003 | 20/40 |
| 3D − 2D | +0.011 ✓ | −2.5 ✓ | +0.023 ✓ | 26/40 |
| 3D − 2.5D | +0.010 ✓ | −2.8 ✓ | +0.019 ✓ | 31/40 |

What this tells us:

- **Every U-Net beats the baseline clearly**, on all four organs, for Dice and NSD. The biggest gain is on the esophagus, the thinnest and hardest organ (0.650 → 0.78–0.79).
- **The 3D U-Net (E_F07) is our best model** and the one we submit. Its lead over the slice models is small on Dice (+0.01) and larger on boundaries (HD95 −2.5 to −2.8 mm). In 3D renders the 2D model leaves visible steps where neighbouring slices disagree; E_F07 does not.
- **2.5D gives nothing over 2D.** Three adjacent slices as input did not help; the gain only came with a real 3D network.

Full tables (IoU, ASSD, raw vs post-processed): [results/cv5/cv5_pooled.md](results/cv5/cv5_pooled.md).

## Process

Each change was tested against its parent run, mostly on the ENet, then carried over to the U-Nets. Early tests used 5 to 8 validation patients and one seed, so the effect sizes are rough; the direction is what we relied on. The full log of every run is in [EXPERIMENTS.md](EXPERIMENTS.md) and [experiments/README.md](experiments/README.md).

| Change | Why | Effect | Kept? |
|---|---|---|---|
| HU window −1000 to 1000 (instead of per-patient min–max) | Same intensity scale for every patient; keeps air and soft tissue apart | ENet Dice 0.741 → 0.795 (trachea +0.10) | Yes |
| Resampling to 1.0 × 1.0 × 2.5 mm | Organs have the same size in pixels for every scanner | ENet Dice 0.741 → 0.801 (esophagus +0.10) | Yes |
| Recovering the aorta as its own class | The first data release merged it into the esophagus | Aorta becomes scorable; esophagus target becomes the thin organ alone | Yes |
| CE + soft Dice loss | Plain CE is dominated by background and under-segments thin organs | Esophagus +0.20, trachea +0.12 (paired, significant) | Yes |
| Weighted CE, focal loss | Other answers to class imbalance | No clear win; never compared to CE + Dice on equal terms | No |
| Augmentation: rotation ±15°, scale 0.9–1.1, intensity ±0.1 | Only 32 training patients | Every organ up, but only with enough epochs; at 25 epochs it looked harmful | Yes |
| Elastic deformation | Standard augmentation | Trachea collapsed (0.377 → 0.081): warping destroys thin tubes | No |
| Larger ENet | More capacity | +0.02 Dice | Superseded by U-Net |
| U-Net instead of ENet | Skip connections keep fine detail | +0.02 to +0.09 Dice depending on recipe | Yes |
| 512 × 512 input | More resolution for thin organs | Inconclusive (test was confounded) | No |
| AdamW + cosine LR decay | Every constant-LR run still peaked at its last epoch | Not isolated | Yes |
| 2.5D input (3 slices) | More context along z | No difference vs 2D | Tested, not used |
| 3D U-Net on patches | Full 3D context | Best model, see Results | **Final** |
| Post-processing: keep largest piece for every organ | Remove stray false positives | Hurt (ENet Dice 0.741 → 0.702): predictions with gaps lose real parts | No |
| Post-processing: largest piece for heart and trachea, aorta pieces < 1 ml removed | Only for organs that are reliably one piece | Little Dice change, removes far-off fragments (one heart fragment 163 mm away) | Yes |
| Gap-tolerant variant (`--keep_near`) | Slice models sometimes split the trachea in two | No supported gain | No |

Statistics: per-patient paired differences with 95% bootstrap CIs (10,000 resamples). Metrics are computed on volumes stitched back to native CT resolution with [metrics3d.py](metrics3d.py): Dice, IoU, HD95, ASSD, NSD at 2 mm.

## Final models

All four share the preprocessing above, CE + Dice loss, lr 5e-4 and seed 123.

| Model | Input | Parameters | Training | Config |
|---|---|---|---|---|
| E_F11 ENet (baseline) | 2D slice | 0.28 M | Adam, constant LR, 100 epochs, no augmentation | [configs/full/E_F11_enet_ce_dice.yaml](configs/full/E_F11_enet_ce_dice.yaml) |
| E_F06 U-Net | 2D slice | 7.8 M | AdamW + cosine, 150 epochs, augmentation | [configs/full/E_F06_unet2d_adamw_cosine_150ep.yaml](configs/full/E_F06_unet2d_adamw_cosine_150ep.yaml) |
| E_F05 U-Net | 3 adjacent slices | 7.8 M | same as E_F06 | [configs/full/E_F05_unet25d_adamw_cosine_150ep.yaml](configs/full/E_F05_unet25d_adamw_cosine_150ep.yaml) |
| E_F07 U-Net | 3D patches | 16.5 M | AdamW + cosine, 300 epochs × 250 iterations | [configs/full/E_F07_unet3d_adamw_cosine.yaml](configs/full/E_F07_unet3d_adamw_cosine.yaml) |

## Reproducing

Setup (Python ≥ 3.10):

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Put the course archives `segthor_train_full.zip` and `test.zip` in `data/`, then build the datasets:

```
make data/SEGTHOR_FULL_huwide_resampled     # fixed 32/8 split
make cv5_data                               # the 5 CV folds
make data/SEGTHOR_TEST_huwide_resampled     # test set
```

Train and score one model on the 32/8 split:

```
# 2D / 2.5D
python -O main.py --config configs/full/E_F06_unet2d_adamw_cosine_150ep.yaml \
    --dest results/full/E_F06_unet2d_adamw_cosine_150ep --gpu
# 3D: trains, infers on validation, stitches and scores in one go
scripts/train_eval_3d.sh E_F07_unet3d_adamw_cosine
```

Full 5-fold cross-validation (training, inference, post-processing, 3D metrics), then the tables:

```
scripts/run_cv5.sh --make-configs
scripts/run_cv5.sh -j 2                     # resumable; finished folds are skipped
python scripts/cv5_pooled.py                # writes results/cv5/cv5_pooled.md
```

Post-processing used for every reported number:

```
python postprocess.py --input_folder <pred>/nii --output_folder <pred>/pp_final/nii \
    --lcc 2 3 --min_ml 4:1
```

Runs are logged to W&B (`ai4miFrancesco/AI4M`), CV runs tagged `cv5` and `fold<k>`.

## Repository layout

| Path | What |
|---|---|
| `main.py`, `train3d.py` | Training (2D/2.5D and 3D) |
| `infer.py`, `infer3d.py`, `stitch.py` | Inference and stitching back to CT resolution |
| `postprocess.py` | Connected-component post-processing |
| `metrics3d.py` | 3D metrics |
| `slice_segthor.py` | Slicing, HU windowing, resampling |
| `segthor/` | Networks, losses, data loading |
| `configs/` | One YAML per experiment; `full/` final models, `cv5/` CV folds, `splits/` patient splits |
| `experiments/` | Records of every run, grouped by model and type of change |
| `scripts/` | CV, statistics, figures and 3D renders |
| `totalsegmentator/` | Slicing and quality control of TotalSegmentator scans for pre-training |
| `docs/` | Plan, model notes, presentation material, course README |

## Limitations

- Single seed for every run; the CV covers patient variation, not seed variation.
- Design choices were tested mostly on the ENet and on few patients, then reused for the U-Nets without re-testing.
- HU window and resampling each helped alone; their combination was never isolated.
