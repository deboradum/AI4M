# 5. Critical analysis: are the conclusions supported?

Rubric: *"How are the tried modifications analysed? Are the conclusions of
(non-)improvement well supported?"* This file states how we judged each change
and how strong the evidence is. Several of our own gains turn out to rest on a
single patient.

## How a modification is analysed

1. **Same data, same split, same metrics.** All runs use the 8 validation
   patients and `metrics3d.py` on the original CT grid (Dice, IoU, HD95, ASSD,
   NSD).
2. **One change at a time where possible.** E_F05 / E_F06 / E_F07 share data and
   recipe, so their differences isolate the input dimension (2D / 2.5D / 3D).
   Post-processing steps are applied one at a time to the same predictions.
3. **Paired per patient, not means.** With 8 patients, one patient can carry a
   mean. For every change we compute the per-patient difference, a **95 %
   bootstrap CI** over patients, and **how many patients got better / worse**
   (`scripts/paired_stats.py`).
4. **Verdict scale.**
   - *supported*: the CI excludes 0 and no patient got worse.
   - *supported, not uniform*: the CI excludes 0, but some patients got worse.
   - *inconclusive*: the CI crosses 0.
   - *1-2 patients only*: too few patients moved to say anything general.

## Evidence per modification

| Modification | What changed | Evidence | Conclusion |
|---|---|---|---|
| **3D U-Net pipeline vs ENet baseline** | Everything: preprocessing, architecture, loss, optimizer, augmentation, post-processing | Dice: all 8 patients above the baseline's *mean* on every organ (worst patient: esophagus 0.69 vs baseline mean 0.52). HD95: 8/8 esophagus and heart, 7/8 trachea and aorta. Paired test pending (needs the baseline's per-patient files) | **Supported** that the final pipeline beats the baseline. It does **not** show that 3D is the reason: too many things changed |
| **3D vs 2D / 2.5D** (E_F07 vs E_F06 / E_F05) | Input dimension only | E_F05 / E_F06 still training | **Open.** This is the comparison that supports or refutes "3D helps the tubes". Run `paired_stats.py` when they finish |
| LCC heart | Remove all but the largest heart piece | HD95 -19.4 mm mean, but only **1 / 8** patients changed (Patient_37: 160 -> 5.5 mm) | **1 patient only.** Fixes one real failure; it says nothing about whether it helps in general |
| LCC trachea | Remove all but the largest trachea piece | HD95 -9.6 mm mean, CI [-34, +5]; **1 better** (Patient_17: 103 -> 11 mm), **2 worse** (Patient_12: 11 -> 25 mm; Patient_37: 2.8 -> 3.5 mm) | **Inconclusive.** Kept as a judgement call: one large fix against one moderate loss. Say so on the slide |
| Aorta pieces < 1 ml | Remove small aorta pieces | 1 / 8 patients changed (Patient_09: 10.0 -> 7.7 mm), none worse | **1 patient only.** Safe on validation; on test it removes 21 pieces with median distance 288 mm |
| Keep pieces within 10 mm (rejected) | Distance rule on esophagus / aorta | Esophagus HD95 worse on 2 / 8, better on 0 | **Rejection supported**: it removes real esophagus beyond a > 10 mm gap |
| Hysteresis gap filling (rejected) | Low-confidence voxels connected to the organ | Esophagus Dice +0.001, CI crosses 0 (4 better / 1 worse); aorta 2 better / 5 worse | **No measurable benefit.** Supports the explanation that gaps are confident errors |
| Class-first patch sampling | How forced patches are drawn | No run without it | **Not measured.** We cannot claim it helped; present it as a design choice, not a result |
| HU window, resampling (team, E_F01 / E_F03) | Preprocessing | Means only in this folder | Run `paired_stats.py` on the team's per-patient metrics to state how well supported they are |

Paired tables for the post-processing rows: `python scripts/paired_stats.py
--a <before>/metrics --b <after>/metrics` (outputs under
`results/snapshots/E_F07_ep24/val/`).

## What the evidence cannot tell us

| Limitation | Effect on the conclusions | What would fix it |
|---|---|---|
| **8 validation patients** | Only large or consistent effects can be shown. Most post-processing effects are 1-2 patients | 5-fold CV over all 40 patients (splits ready: `scripts/make_folds.py`); 40 paired patients instead of 8 |
| **Choices made on the same 8 patients** (HU window, resampling, checkpoint, post-processing rule) | Absolute numbers are optimistic. Rankings between our runs are fair because all were chosen the same way | The course's test-set score is the unbiased number; we cannot compute it (no test labels) |
| **One seed per run** | Seed-to-seed variation is unknown, so small differences between runs may be noise | 2-3 seeds of the final configuration, or CV |
| **Resampled 256 mm crop** | 6 / 40 patients lose part of an organ at the image edge (heart up to ~6 %); Patient_10 is in validation. Affects every model trained on this data equally | Larger in-plane size after resampling, or crop around the body |
| **Selection metric differs from the reported one** | Checkpoint chosen on 3D Dice of the resampled volumes, reported on the CT grid. Small effect | — |

## What the analysis points to

The remaining esophagus and aorta errors are **gaps the network is confident
about**. Hysteresis could not fill them, and removing the scraps around them made
HD95 worse. Dice and CE barely penalise a thin gap. So the next modification is a
topology-aware loss: **Skeleton Recall loss** (Kirchhoff et al., ECCV 2024) or
clDice (Shit et al., CVPR 2021). For the stray far predictions that currently
need post-processing, the boundary loss (Kervadec et al., MIDL 2019 / MedIA
2021) penalises them during training.
