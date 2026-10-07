# Presentation documentation: 3D U-Net and post-processing

Everything behind the 3D part of the project, written so that the slides can be
built from it and every claim can be traced to code, a run, or a number.

> **Status (7 Oct 2026): interim numbers.** All results here come from the
> **epoch-24 checkpoint** of E_F07 (`results/snapshots/E_F07_ep24/`). E_F07 trains
> to epoch 300 (ETA ~21:30, 7 Oct). The brief requires the presentation to show
> the same results as the submission, so once the final checkpoint exists,
> regenerate every table and figure with the commands in
> [Regenerating the numbers](#regenerating-the-numbers) before making slides.

## Contents

| File | What it covers |
|---|---|
| [01_motivation.md](01_motivation.md) | **Motivation** (3 pts): each change from an observation, and the alternatives we did not pick |
| [02_method_3d_unet.md](02_method_3d_unet.md) | The 3D U-Net (E_F07): data path, model, training, inference |
| [03_results.md](03_results.md) | Validation results vs the ENet baseline, all five 3D metrics, per patient |
| [04_postprocessing.md](04_postprocessing.md) | Post-processing: the final rule, why size not distance, two rejected ideas |
| [05_critical_analysis.md](05_critical_analysis.md) | **Critical analysis** (3 pts): how each change was judged (paired per patient, bootstrap CI) and how well each conclusion is supported |
| [06_slide_plan.md](06_slide_plan.md) | Proposed 10-minute story, slide by slide, mapped to the rubric |
| [references.md](references.md) | Citations used in the slides |
| [figures/](figures/) | Slide-ready figures (generated, see below) |

## The story in one paragraph

SegTHOR scans are 3D volumes, and the organs that the 2D networks struggle with,
the esophagus and the trachea, are long thin tubes along the z axis. A 2D network
sees one slice at a time and has to guess continuity; a 3D network sees it. After
the team fixed the input (HU window, resampling to a common 1.0 x 1.0 x 2.5 mm
grid, which is what makes 3D convolutions meaningful across patients), we trained
a 3D U-Net with the same recipe as the 2D/2.5D U-Nets (E_F05/E_F06), so the
comparison isolates the dimensionality. On the 8 validation patients it beats the
ENet baseline on every organ and every metric, with the largest gain on the
esophagus (Dice 0.52 -> 0.80). Its remaining failures are of two kinds: stray
false positives far from the organ, which targeted post-processing removes, and
gaps in the thin tubes, which post-processing cannot fix (we tested three ways
and measured why), pointing to a topology-aware loss as the next step.

## Where things are

| What | Path |
|---|---|
| Branch | `francesco/3d` on github.com/deboradum/AI4M |
| 3D model | `segthor/models/UNet3D.py` |
| 3D data path, sampler, augmentation, sliding window | `segthor/volumes.py` |
| 3D training / inference | `train3d.py`, `infer3d.py` |
| Config | `configs/full/E_F07_unet3d_adamw_cosine.yaml` |
| Run script (train, infer, stitch, metrics) | `scripts/train_eval_3d.sh` |
| Post-processing | `postprocess.py` (`--lcc`, `--min_ml`, `--keep_near`), `infer3d.py --hysteresis` |
| 3D render videos | `scripts/render_3d.py` |
| Tables / figures for these docs | `scripts/presentation_tables.py`, `scripts/presentation_figures.py` |
| Paired per-patient comparison of two runs | `scripts/paired_stats.py` |
| Run outputs (not in git) | `results/full/E_F07_unet3d_adamw_cosine/`, `results/snapshots/E_F07_ep24/` |
| W&B run | https://wandb.ai/ai4miFrancesco/AI4M/runs/qzwj6s46 |

## Regenerating the numbers

From the repo root, with the final checkpoint (`R=results/full/E_F07_unet3d_adamw_cosine`):

```bash
# val + test predictions (sliding window, stitched back to the CT grid)
python infer3d.py --config $R/config_dump.yaml --weights $R/bestweights.pt --gpu --resample \
  --img_folder data/SEGTHOR_FULL_huwide_resampled/val/img --dest $R/val \
  --scan_pattern 'data/segthor_train_full/train/{id_}/{id_}.nii.gz'
python infer3d.py --config $R/config_dump.yaml --weights $R/bestweights.pt --gpu --resample \
  --img_folder data/SEGTHOR_TEST_huwide_resampled/test/img --dest $R/test \
  --scan_pattern 'data/segthor_test/test/{id_}.nii.gz'
# final post-processing, val and test
for s in val test; do python postprocess.py --input_folder $R/$s/nii --output_folder $R/$s/pp_final/nii \
  --lcc 2 3 --min_ml 4:1 --process 8; done
# metrics (raw and post-processed)
for d in $R/val $R/val/pp_final; do python metrics3d.py --pred_folder $d/nii --dest $d/metrics \
  --gt_pattern 'data/segthor_train_full/train/{id_}/GT.nii.gz' \
  --scan_pattern 'data/segthor_train_full/train/{id_}/{id_}.nii.gz' \
  --class_names background esophagus heart trachea aorta --process 8; done
# tables and figures for these docs
python scripts/presentation_tables.py --run $R
python scripts/presentation_figures.py --run $R --dest docs/presentation/figures
```
