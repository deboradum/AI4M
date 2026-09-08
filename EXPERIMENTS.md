# Experiment index

This is the shared registry for scientific runs. Create or update a row before
starting a run; generated outputs stay under ignored `results/` and `volumes/`.

| ID | Parent | Ablation / change | Hypothesis | Important configuration | Status | Final result | Notes |
|---|---|---|---|---|---|---|---|
| E000 | – | Original course baseline | Reference implementation | ENet, CE, single slice, 25 epochs, unseeded | historical | – | Original run predates the YAML configuration workflow. |
| E001 | E000 | Reproducible baseline | A fixed seed provides a stable parent for ablations | `in_slices=1`, ENet 8/2, CE, seed 123, 25 epochs, SegTHOR 15/5 split | complete | 3D Dice: esophagus 0.610, heart 0.808, trachea 0.377; foreground mean 0.598 | Commit `617e3f1`; 21.4 min training; aorta unavailable in `segthor_part1`. |
| E008 | E001 | 2.5D input | Adjacent axial slices improve z-context, especially for trachea and esophagus | `in_slices=3`; otherwise identical to E001 | complete | 3D Dice: esophagus 0.508, heart 0.705, trachea 0.245; foreground mean 0.486 | Commit `b859009`; lower than E001 by 0.112 foreground mean Dice; aorta unavailable in `segthor_part1`. |
| E010_large_enet | E001 | Larger ENet capacity | A larger ENet improves segmentation under the E001 protocol, or exposes a compute/overfitting trade-off | ENet `kernels=16`, `factor=4`, `in_slices=1`; otherwise identical to E001 | planned | – | Runtime: `results/E010_large_enet/`; committed record: `experiments/E010_large_enet/`. |
| E010_unet2d | E001 | Plain 2D U-Net architecture/capacity | U-Net skip connections improve segmentation under the E001 protocol, or expose a compute/overfitting trade-off | U-Net base channels 32, 4 levels, `in_slices=1`; otherwise identical to E001 | planned | – | Runtime: `results/E010_unet2d/`; committed record: `experiments/E010_unet2d/`. |
