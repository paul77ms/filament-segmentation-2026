# The 1–3 independent annotations per image

## What the data look like

The 707 training images carry 1, 2 or 3 annotation sets (411 / 145 / 151 images, 1,154 sets). Each set was drawn
independently by a different annotator. The MAGFiLO authors deliberately keep the natural disagreements and do not
aggregate them (Ahmadzadeh et al., Scientific Data 2024). The disagreement is large:

**Agreement between annotators.**
- Two annotators of the same image agree with PQ 0.34. Their matched-filament IoU has a median of 0.63.
- Two annotators almost never trace the same filament to IoU > 0.8: 1.0 % of best-match IoUs fall in 0.8–0.9 and
  none above 0.9.

**Each annotator has a systematic style.**
- Width differs by ±20–35 %.
- 6–33 % of the other annotators' filaments are skipped, mostly faint or small ones.
- About 39 % of multi-set images differ on whether a filament is one object or several.

**Agreement falls with size** (multi-set images; `docs/figures/fig_beyond_human.png`, left):

| filament size | drawn by all annotators | drawn by one annotator only |
|---|---|---|
| < 250 px | 19 % | 40 % |
| 250–500 px | 30 % | 33 % |
| 500–1,000 px | 41 % | 21 % |
| 1,000–2,000 px | 50 % | 15 % |
| > 2,000 px | 52–53 % | 14–19 % |

**Small filaments are drawn by only some annotators.** In multi-set images, 77 % of the small filaments the model can
recover were drawn by only some of the annotators.

## The principle: never fuse, treat every set as a valid interpretation

The official metric scores each annotation set separately (V0), so the goal is the prediction with the best
expected score over the annotators. The final pipeline uses the sets as follows.

| Component | How the sets are used | Evidence |
|---|---|---|
| Stage 1 (U-Net) | each visit of an image draws one of its sets (seeded); BCE + soft Dice | over the epochs the expected loss is the loss averaged over the image's sets: every annotator's interpretation is trained, none is preferred |
| R2 refinement | each window visit draws one set's instance that matches the candidate | the same expectation, per candidate |
| SE2 scorer | target y = mean over the image's sets of (IoU with that set's best instance if > 0.5, else 0); SE2 computes it on the R2-refined fragment | this is the exact per-component V0 term: an added disjoint component raises the pooled V0 iff y > PQ/2 (checked with the real evaluator in `tests/test_small_exist.py`) |
| Evaluation | V0 (sets separate), V1 (sets pooled), LOAO (one annotator vs the rest); consensus vs single-annotator ground truth reported separately | `docs/validation.md` |

## Why not a fused label

**Majority vote, union and intersection** are not used:
- With two sets, a majority vote degenerates into either the union or the intersection.
- The union fuses differently-split filaments and widens them.
- The intersection deletes the faint and small filaments that only some annotators drew.

## What a single submission cannot do

**One interpretation per filament.** A submission is one segmentation. Where annotators disagree on extent, on one
filament vs two, or on whether a faint filament exists at all, the pipeline outputs a single interpretation. The
losses and the SE2 decision rule are set up so that this is the interpretation with the highest expected
per-annotator score. The one exception is SE2's threshold, which was set for the leaderboard-type score (sets
pooled), so it keeps filaments that only some annotators drew more often than the per-annotator score alone would.

**Remaining gaps against a human.** The leave-one-annotator-out comparison shows four (`docs/figures/fig_human_vs_model.png`):
- spurious predictions (matching no remaining annotator): 0.186 vs 0.179;
- thin-region precision: 0.484 vs 0.494;
- the share of split filaments: 0.041 vs 0.029;
- the number of filaments: 10 % more than an annotator draws (count ratio 1.10).
