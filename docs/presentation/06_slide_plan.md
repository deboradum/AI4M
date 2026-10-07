# 6. Slide plan (10 minutes + 3 for questions)

A proposed story for the whole presentation, with the 3D part placed in it. The
course asks for "a consistent story, and coherent project", not a catalogue of
attempts. So each slide answers the problem the previous slide exposed.
Sections marked *(team)* are other members' work. Fill them from
`EXPERIMENTS.md` and `experiments/`.

## Hard requirements from the brief

- [ ] Slide numbers on every slide (0.5 pt on its own)
- [ ] Group number and members on the first slide (and/or the last)
- [ ] End on a take-home / summary slide, **not** "Thank you" / "Questions?"
- [ ] Numbers identical to the submission. Regenerate everything from the final checkpoint first (README)
- [ ] Every speaker presents a share (balance between speakers, 1 pt)
- [ ] Citations on the slides where an idea comes from a paper (1 pt), see [references.md](references.md)
- [ ] Short slides; split anything dense

## Flow (~12 slides, ~50 s each)

The two 3-point rubric items, motivation and critical analysis, shape the flow.
Every method slide opens with the observation that motivated it
([01](01_motivation.md)). Every result slide says how well it is supported
([05](05_critical_analysis.md)), not just the mean.

| # | Slide | Content | Figure / table | Rubric |
|---|---|---|---|---|
| 1 | Title | Group number, members, one-line result | — | — |
| 2 | What makes SegTHOR hard | Thin, low-contrast esophagus/trachea; spacing varies 1.5x; contrast in 12/40 scans | 2D slice with the esophagus marked; data table in [01](01_motivation.md) | Motivation |
| 3 | Baseline and how we judge a change | ENet per organ; 3D metrics on the CT grid; **paired per patient + bootstrap CI**, because 8 patients | E_F00 row; verdict scale from [05](05_critical_analysis.md) | Critical analysis |
| 4 | Fix the input *(team)* | Observation -> HU window, resampling | E_F00-E_F03 | Motivation |
| 5 | Training improvements *(team)* | Observation -> change for augmentation, loss, optimizer, 2D U-Net; why no elastic, no flips | team tables | Motivation |
| 6 | Why 3D | Weak organs are tubes along z; why not 2.5D only, a bigger 2D net, transformers, ensembling | Patient_15 render; decision rows 3-4 of [01](01_motivation.md) | Motivation |
| 7 | The 3D U-Net | Same recipe as 2D/2.5D; only what 3D forces changes, each with its reason. nnU-Net ideas cited | [02](02_method_3d_unet.md) | Motivation |
| 8 | Results vs baseline | Better on all organs; all 8 patients above the baseline mean. What this does *not* show: that 3D is the cause | `figures/results_vs_baseline.png` | Results, critical analysis |
| 9 | 2D vs 2.5D vs 3D | The controlled comparison, paired | `paired_stats.py` output, when E_F05/E_F06 finish | Critical analysis |
| 10 | Post-processing | Two failure modes; final rule; size not distance; **most gains are 1 patient** | Patient_37 before/after; `leftover_pieces.png` | Motivation, critical analysis |
| 11 | What the evidence supports, and next steps | Supported / inconclusive / not measured; 8 patients, one seed, no CV; gaps -> topology-aware loss | Evidence table of [05](05_critical_analysis.md), trimmed | Critical analysis |
| 12 | Take-home | 3 bullets; group members again | — | Summary |

Backup slides (for questions): full 5-metric table, per-patient table, paired
tables, rejected post-processing ideas, compute cost, test-set gallery frames.

## Suggested speaking split (fill in names)

| Part | Slides | Speaker |
|---|---|---|
| Problem, data, baseline | 1-3 | … |
| Input and training improvements | 4-5 | … |
| 3D model and results | 6-9 | … |
| Post-processing and limitations | 10-11 | … |
| Take-home | 12 | … |

## Likely questions and short answers

- **"Is the val score optimistic?"** Yes, slightly. The checkpoint and the design
  choices were picked on those 8 patients. The test set (scored by the course
  staff) is the unbiased number. Rankings between our runs are fair because all
  of them were chosen the same way.
- **"Why not nnU-Net?"** Forbidden by the brief. We borrowed specific, cited ideas.
- **"Why is heart HD95 worse than the baseline before post-processing?"** One
  patient: a stray heart prediction 160 mm away (Patient_37). The largest
  connected component removes it.
- **"Is the post-processing gain real?"** It fixes specific failures (Patient_37's
  heart, Patient_17's trachea). On 8 patients we cannot claim a general
  improvement: the heart gain is one patient, and the trachea is one better and
  two worse.
- **"Why leave the esophagus noise?"** Measured: those pieces sit 5-13 mm from the
  true esophagus inside a gap. Removing them makes HD95 worse. They are a symptom
  of gaps, which need a training-time fix.
- **"Is 3D slower?"** Not in total: a full 3D run (~6.9 h on an A100) takes about as
  long as a 2D run (7.4 h on an H100), and per voxel it is ~17x faster. We did not
  profile 2D; per-batch asserts and PNG I/O are the likely cost.
