# Experiment index

This is the shared registry for scientific runs. Create or update a row before
starting a run; generated outputs stay under ignored `results/` and `volumes/`.

| ID | Parent | Ablation / change | Hypothesis | Important configuration | Status | Final result | Notes |
|---|---|---|---|---|---|---|---|
| E000 | – | Original course baseline | Reference implementation | ENet, CE, single slice, 25 epochs, unseeded | historical | – | Original run predates the YAML configuration workflow. |
| E001 | E000 | Reproducible baseline | A fixed seed provides a stable parent for ablations | `in_slices=1`, ENet 8/2, CE, seed 123, 25 epochs, SegTHOR 15/5 split | planned | – | Runtime: `results/E001_baseline/`; committed record: `experiments/E001/`. |
| E008 | E001 | 2.5D input | Adjacent axial slices improve z-context, especially for trachea and esophagus | `in_slices=3`; otherwise identical to E001 | planned | – | Runtime: `results/E008_25d/`; committed record: `experiments/E008/`. |
