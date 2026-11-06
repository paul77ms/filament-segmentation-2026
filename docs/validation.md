# Validation protocol

Every number in the README is an out-of-fold (OOF) Panoptic Quality on the 707 training images (1,154 annotation-set
records). Every stage of the final pipeline was judged by a rule written down before its result
(`records/registry.csv`). Comments of the form "D-0xx" in the code refer to the team's internal decision log.

## Cross-validation split (`scripts/make_splits.py`, `outputs/splits/folds_v1.csv`)

- **One fold per image.** An image and all of its 1–3 annotation sets always stay in the same fold.
- **14-day groups.** Observations are linked into one group when they are less than 14 days apart, which gives 57
  groups. Same-day images from different GONG sites therefore stay together.
- **Balanced folds.** Groups are assigned to 5 folds, balanced for files, sets, instances, site, year and annotation
  batch. No group spans two folds. The minimum time gap between images of different folds is 14.1–15.0 days. No plain
  KFold is used anywhere.

## Scorers

- **V0: the organisers' self-evaluation notebook.**
  - One record per (annotation set, image).
  - Every IoU > 0.5 pair is a true positive.
  - PQ is pooled over the records.
  - Our implementation matches the official notebook exactly (`scripts/official_eval/`, `scripts/mv_verification/`).
- **V1: a hypothesis about the hidden leaderboard scorer.**
  - Per image, the ground truth is the concatenation of all its annotation sets.
  - Matching is one-to-one at IoU > 0.5, and PQ is pooled over images (`src/filseg/scorers.py`,
    `scripts/scorer_v1_check.py`).
  - The public scores of our earlier submissions matched V1 within rounding; for the final CSV, V1 is 0.381 out of
    fold and the public score 0.37 (the public part is a subset of the 180 test images).
  - Our predictions are always disjoint, so nothing exploits this scorer.
- **Leave-one-annotator-out (LOAO).** On the 296 multi-set images, each annotator in turn is scored against each of
  the other sets, and so is the model (`scripts/beyond_human.py`, `scripts/eval_dual.py`, `scripts/a_eval.py loao`).
  This gives a human-agreement reference for every metric.

## Nested validation of the stacked stages (R2 and SE2)

**Why nesting is needed.** R2 and the SE2 scorer are trained on stage-1 outputs, so a normal OOF would leak.

**How it is done.** For each outer fold k:
- four inner stage-1 models are trained on the three folds outside {k, j}:
  `scripts/train_seg.py --exp b0n_r101_s1024 --holdout k,j`;
- they give every outer-train image an out-of-fold stage 1;
- they give fold k the mean of the four models (`B_ens`).

The stacked models are then trained and scored as follows:
- **R2n_k** is trained on the outer-train windows only, and scored on fold k with exactly the test-time recipe.
- **The SE2 scorer for fold k** is trained only on fragments produced by inner models that never saw fold k (see
  "SE2 and its threshold" below). Fold k's fragments come from `B_ens` and R2n_k.

**Checks.** Machine checks per outer fold confirm that no fold-k label reaches fold k's predictions
(`scripts/nested_stage1.py verify`, `records/nested_r2b0/verify_o*.json`).

**Cost.** 15 GPU runs, 7.8 GPU-hours for stage 1 and R2; SE2 adds 20 inner R2 models (about 10 GPU-hours).

## How each stage was adopted

| Stage | Rule (registered before the result) | Result |
|---|---|---|
| Stage-1 post-processing (thr 0.5, >= 500 px) | chosen by nested CV (`scripts/tune_post.py`): selected on the training folds, scored on the held-out fold | 0.410 5-fold OOF (0.409 with the nested stage 1 `B_ens`); a plateau between 300 and 500 px |
| R2 refinement | beat `B_ens` on PQ, thin-region recall and precision, a per-file sign test, the review set, and LOAO | PQ +0.0150 [+0.0104, +0.0197], 5/5 folds positive, 426 files better and 253 worse (sign test p = 3e-11), review set 0.369 -> 0.380. **Two conditions failed** (see below) |
| Cleanup (20 px, then 5 %) | V0 and V1 >= 0, or a CI that includes 0 with lower bound > -0.003; overlap diagnostics, review set and LOAO not worse | V0 +0.0006 and V1 +0.0006 in total (20 px: +0.0003 / +0.0003; 5 %: +0.0004 / +0.0003), i.e. neutral; N:1 364 -> 277, 1:N 451 -> 346, multi-part filaments 27.6 % -> 14.7 % |
| SE2 | E1 V0 CI lower > -0.002 and >= 3/5 folds; E2 V1 not lower; E3 no-match predictions up <= 10 %; E4 LOAO small detection up, nothing newly below human; E5 review set not lower; E6 N:1 up <= 10 % (the rule's reference was an earlier small-filament stage that scored the raw pieces) | at the registered threshold (PQ/2): **E3 fails** (+13.7 %). At the final threshold (m x PQ/2, m chosen by V1, see below): **E1, E3, E4 and E5 fail**, E2 and E6 pass. Adopted by a post-hoc decision. Against the same pipeline without a small-filament stage: V0 +0.0067 [+0.0028, +0.0106], V1 +0.0220 [+0.0179, +0.0259], 5/5 folds each |
| Constants (all 7) | switch only if nested automatic tuning gives V0 with CI lower bound > 0 | +0.0003 [-0.0015, +0.0023] -> current values kept; all on a plateau (`docs/figures/fig_constants.png`) |

### R2 was adopted by a post-hoc decision (stated openly)

**The failure.** R2 failed two of its pre-registered conditions:
- Thin-region precision fell from 0.510 to 0.493, while thin-region recall rose from 0.427 to 0.454.
- The LOAO aspect `thin_precision_colocated` fell below the human level (human 0.566, `B_ens` 0.576, R2 0.548).

These are mask-derived thin-region metrics, not barb accuracy.

**What the rule said and what we did.** Under the rule, R2 was "not adopted". After seeing this result, the team
decided to adopt R2 anyway, and recorded that as a post-hoc decision. R2_final was then trained once, with the
validated recipe, on all 707 images.

**What came after.** The public leaderboard was not used for this decision. The small-filament stage on top of R2 was
then judged under its own pre-registered rule (below).

### SE2 and its threshold (post hoc, stated openly)

**Why SE2.** Missed filaments that two or more annotators drew have a median stage-1 probability of 0.95; at 1024²
thin filaments break into pieces under 500 px, which the size rule discards. Judged as raw pieces, their IoU with the
annotation stays below 0.5 even when the filament is real. SE2 redraws these pieces with R2 at native resolution and
scores the redrawn mask.

**Strict nesting.** The scorer of outer fold k is trained on fragments of folds j != k that were refined by 20 inner
R2 models (folds outside {k, j}; stage 1 of their training files from models that never saw k), with fold-j stage 1
from the inner model that never saw k or j (`scripts/run_r2nn.sh`, `records/se2/R2nn_context/`).

**Result at the registered threshold (PQ/2 = 0.2124).** V0 +0.0019 [-0.0004, +0.0041], V1 +0.0068 [+0.0044, +0.0090],
small-filament detection .435 -> .546, but predictions that match no annotation set +13.7 % (limit 10 %): the rule
fails on E3, so SE2 was not adopted as registered (`records/se2/summary.json`).

**The final threshold: chosen by the leaderboard-type scorer V1.**
- *Why V1.* The SE2 score predicts the gain averaged over the image's annotation sets, which is the per-set metric V0.
  The leaderboard pools the sets of an image (V1): a fragment that one annotator of three drew is a full match there,
  but scores a third of that under V0. A threshold set by V0 therefore leaves out filaments that only some
  annotators drew.
- *Rule, registered before it was computed.* Grid m in {0.6, 0.8, 1.0, 1.25, 1.5} (threshold m x tau_k); for each
  outer fold k, m_k maximises V1 on the other four folds, and fold k is scored with m_k
  (`scripts/se2_analysis.py select`). Chosen: m = 0.6 in folds 0, 2, 3, 4 and 0.8 in fold 1. At test time m = 0.6,
  the V1 maximum over all 707 images and the choice of 4 of the 5 folds: threshold 0.6 x 0.2124 = 0.12744
  (`records/se2/analysis/threshold_v1.json`).
- *All folds, every m* (exploratory context, `records/se2/analysis/sweep.csv`):

  | m | V0 | V1 | predictions | matching no annotation set |
  |---|---|---|---|---|
  | no small-fragment stage | 0.4249 | 0.3595 | 4,339 | 1,074 |
  | 0.6 | 0.4316 | 0.3819 | 5,299 | 1,550 |
  | 0.8 | 0.4343 | 0.3815 | 5,117 | 1,427 |
  | 1.0 | 0.4359 | 0.3780 | 4,876 | 1,292 |
  | 1.25 | 0.4344 | 0.3736 | 4,680 | 1,194 |
  | 1.5 | 0.4330 | 0.3700 | 4,547 | 1,132 |

- *The registered SE2 rule at this threshold* (nested, against the rule's reference): E1 fails (V0 -0.0024
  [-0.0054, +0.0006]); E2 passes (V1 +0.0102 [+0.0071, +0.0133], 5/5 folds); E3 fails (predictions matching no
  annotation set 1,136 -> 1,533, +35 %); E4 fails (small-filament detection .435 -> .619, a left-out human .555, but
  three aspects newly below the human level: thin-region precision, spurious-prediction rate and share of spurious
  predictions); E5 fails (review set V0 .394 -> .393); E6 passes (many-to-one 277 -> 289, +4 %).

**What is post hoc.** The threshold was first chosen by the first-order V0 gain on out-of-fold scores (m 1.25). The
criterion was changed to V1 after the all-fold sweep above had been seen (V1 highest at m 0.6); the fold-separated
rule was registered before it ran. Both thresholds received the same public score (0.37), so the leaderboard did not
decide; the team chose m 0.6 after reviewing its additions on the test and validation images by eye. The final
threshold does not pass the registered SE2 rule, so the outer estimate carries selection optimism. 0.6 is the lowest
value of the grid; smaller multipliers were not evaluated.

**What the additions are** (`records/se2/analysis/added.csv`, the final pipeline's additions): of 928 additions, 469
match at least one annotation set; of the 459 that match none, 302 lie more than 10 px from any annotation (68 % of
them in single-annotator images), 137 overlap an annotated filament with IoU 0.1–0.5, 18 lie within 10 px of one and
2 are pieces of an already matched filament. A visual review of the additions on validation and test images found
mostly small, dark, elongated structures; a few are faint bands near the limb or compact blobs.

**Robustness.** Against the pipeline without a small-filament stage, V1 rises at all 6 sites and in all 4 periods;
V0 rises in all 4 periods and at 5 of the 6 sites (site B: -0.0001) (`records/se2/analysis/se2_vs_no_small_stage.json`).

## Leakage checks

- **No model saw its evaluation fold.** The training-file lists of all 10 inner stage-1 models, the 5 `B_ens`
  ensembles, the 5 stage-1 fold models, the 5 R2n and 20 inner R2 models, and the sources of every outer SE2 scorer
  exclude the evaluated fold (`scripts/nested_stage1.py verify`, `records/nested_r2b0/verify_o*.json`).
- **Folds are separated in time.** Images of different folds are at least 14.1 days apart (median 45.6 days).
- **Inference reads only the test images.** A system-call trace of `src/filseg_cli.py` (strace) shows that, besides
  the code, inference opens only `weights/` and the test images; the annotation file, the training images and the
  split file are never opened. No test image is byte-identical to a training image.

## Reproduction checks

- **GPU.** The self-contained inference (`src/filseg_infer*.py`) reproduces the submitted CSV byte for byte on an
  RTX A6000 (`records/release_checks/`). The SE2 port was checked against the validation code (`scripts/se2.py`) on
  33 test images with fragments: identical fragments and crops in 33/33.
- **CPU.** On CPU the pipeline runs in fp32, the GPU reference in bf16, so pixels with a probability close to a
  threshold can flip. The notebook judges this with tolerances fixed in advance (`src/reproduce_utils.py`,
  `VERDICT_TOLERANCE`: >= 97 % of the reference filaments matched, <= 3 % appearing or disappearing, median IoU
  >= 0.95).
- **Code changes.** After the repository was reduced to the final pipeline, the nested out-of-fold predictions of all
  707 training images, their V0 / V1 records and the leave-one-annotator-out aspects were recomputed and are identical
  to those of the validated code; the training data streams of stage 1 and R2 are sample-for-sample identical.
  After the SE2 threshold was set to its final value, the nested records were recomputed with this repository's code
  and equal those of the registered V1 selection (V0 0.4316, V1 0.3814, 707 images).
