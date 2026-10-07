# Experiment records

Layout: `experiments/<model>/<kind of change>/<experiment>`. Each record is
a `config.json` folder (early runs, sometimes with per-patient CSVs) or a
single `E_F*.json` file (full-dataset runs). Configs live in `configs/`,
outputs in the ignored `results/`; the running log is `EXPERIMENTS.md`.

**Only compare rows on the same dataset.** Three datasets were used:

| Tag | Data | Classes scored |
|---|---|---|
| `part1` | 20 patients (15/5), aorta folded into esophagus | 3 (esophagus∪aorta, heart, trachea) |
| `part1+aorta` | same 20 patients, aorta split back out (`archive/aorta_recovery`) | 4 |
| `full` | 40 patients (32/8, `configs/splits/segthor_full_32_8.json`), course labels | 4 |

Foreground mean = 3D Dice averaged over the scored organs, validation set.

## ENet (kernels 8, factor 2)

| Kind | Experiment | Data | What changed | FG Dice |
|---|---|---|---|---|
| baseline | E001_baseline | part1 | seeded course baseline | 0.598 |
| baseline | E_F00_baseline.json | full | course baseline on the full set | 0.741 |
| preprocessing | E009_full512 | part1 | 512² instead of 256² (confounded) | 0.558 |
| preprocessing | E014_hu_wide | part1 | HU window [-1000, 1000] | 0.737 |
| preprocessing | E014_hu_soft | part1 | HU window [-200, 300] | 0.724 |
| preprocessing | E_F01_hu_wide.json | full | HU window [-1000, 1000] | 0.795 |
| preprocessing | E_F02_hu_mid.json | full | HU window [-1000, 600] | 0.784 |
| preprocessing | E_F03_size_normalized.json | full | resampled to fixed spacing | 0.801 |
| augmentation | E015_augment | part1 | rotation/scale/intensity + elastic | 0.534 |
| augmentation | E016_aug_noelastic | part1 | same, no elastic (never run) | – |
| augmentation | E018_aorta_augment | part1+aorta | augmentation, 25 epochs | 0.606 |
| augmentation | E020_aorta_augment_50ep | part1+aorta | augmentation, 50 epochs | 0.706 |
| loss | E019_aorta_ce_dice | part1+aorta | CE + Dice | 0.693 |
| optimizer | E006_cosine50 | part1 | cosine LR, 50 epochs (2D val only, forked code) | – |
| input_2.5d | E008_25d | part1 | 3 adjacent slices | 0.486 |
| labels | E017_aorta_huwide | part1+aorta | aorta as its own class | 0.615 |
| combined | E022_aorta_ce_dice_augment_75ep | part1+aorta | CE+Dice + augmentation, 75 epochs | 0.757 |
| postprocessing | E021_aorta_lcc_postprocess | part1+aorta | largest connected component (rejected) | – |
| postprocessing | E_F09_baseline_pp_*.json | full | LCC / min-size / morphology / hole filling on E_F00 | 0.702–0.750 |

## ENet_large (kernels 16, factor 4)

| Kind | Experiment | Data | What changed | FG Dice |
|---|---|---|---|---|
| architecture | E010_large_enet | part1 | larger ENet | 0.618 |

## UNet (base 32, 4 levels)

| Kind | Experiment | Data | What changed | FG Dice |
|---|---|---|---|---|
| architecture | E010_unet2d | part1 | U-Net instead of ENet | 0.686 |
| architecture | E023_unet_aorta_75ep | part1+aorta | U-Net, CE, no augmentation | 0.752 |
| combined | E024_unet_aorta_ce_dice_augment_75ep | part1+aorta | U-Net + E022's recipe | 0.781 |
| combined | E_F04_unet_ce_dice_augment_150ep_es20.json | full | U-Net + CE+Dice + augmentation + HU window | **0.854** |
| optimizer | E_F06_unet2d_adamw_cosine_150ep.json | full | E_F04 + AdamW (wd 1e-4) + cosine LR | running |
| input_2.5d | E_F05_unet25d_adamw_cosine_150ep.json | full | E_F06 + 3 adjacent slices | running |

The loss sweep (weighted CE, focal, focal+Dice) is kept in `archive/loss_sweep/`.
