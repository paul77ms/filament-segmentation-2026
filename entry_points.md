# Entry points

All commands run from the repository root. Paths are in `SETTINGS.json`.

**Before you start:**
- Install `pip install -r requirements.txt` (Python 3.13, CUDA 12.4).
- Unzip the competition data to `MAGFiLO_1.0_Kaggle_2026/`.
- Prefix every command with `PYTHONPATH=src`.
- One GPU with 14 GB is enough.
- Training also needs torchvision's ImageNet ResNet-101 checkpoint in `~/.cache/torch/hub/checkpoints/`:
  `wget -P ~/.cache/torch/hub/checkpoints https://download.pytorch.org/models/resnet101-5d3b4d8f.pth`.

**What the training runs reproduce.** The outputs of sections 2 and 3 are the checkpoints in `weights/`, which are
byte-identical to the files the runs wrote. Re-training on another machine gives equivalent models, not
bit-identical ones (cuDNN).

## 1. Predict with the stored weights (no training)

```bash
jupyter nbconvert --to notebook --execute notebooks/reproduce_submission.ipynb \
    --output reproduce_submission.executed.ipynb --ExecutePreprocessor.timeout=-1
```

- **Output:** `outputs/submission_r2b0_se2.csv`. The notebook compares it with `reference/submission_r2b0_se2.csv`.
- **Expected result:** byte-identical on an RTX A6000.
- **Time:** about 10 minutes for the 180 test images.

## 2. Prepare data

```bash
python scripts/make_splits.py           # -> outputs/splits/folds_v1.csv (already in the repository)
```

This step builds the 5-fold split: an image and all of its annotation sets stay in one fold, and observations less
than 14 days apart form one group. Full run: CPU, about 2 min with 16 workers. Add `--skip-image-analysis` for the time-only part.

## 3. Train the three models of the final pipeline

| # | Command | Output | Time (RTX A6000) |
|---|---|---|---|
| 3.1 | `python scripts/train_seg.py --exp b0_r101_s1024 --fold k --epochs 60 --val_every 5 --workers 6` for k = 0…4 | `outputs/exps/b0_r101_s1024/fold<k>/last.pt` and `val_probs_last.npz` (out-of-fold stage 1) | 57–81 min per fold, 14 GB |
| 3.2 | `python scripts/build_refine_cache.py` | `outputs/cache/refine_v1/`: R2 windows built from the out-of-fold stage 1 of 3.1 | CPU, minutes |
| 3.3 | `python scripts/train_refine.py --variant r2 --all --epochs 30 --exp R2_final_context` | `outputs/exps/R2_final_context/all/last.pt` | 36 min, 5.2 GB |
| 3.4 | `python scripts/small_exist.py cands` | small pieces of the out-of-fold stage 1 (inputs of 3.5) | CPU, under 1 min |
| 3.5 | `python scripts/small_exist.py train` | the per-fold break-even tau_k = PQ/2 of the pipeline before SE2 (`train_log.json`), used by SE2's threshold; the scorer architecture and recipe that SE2 reuses | about 1 min |
| 3.6 | nested stage 1 and R2 (section 5, `scripts/nested_queue.py`) | inner stage-1 maps and R2n models: the out-of-fold refined fragments that train SE2 | 7.8 GPU-h |
| 3.7 | `python scripts/se2.py data --source eval`, then `python scripts/se2.py train` (the train step needs the inner sources of 3.8) | `outputs/research/se2/scorer_final.pt` (SE2, final) | ~10 min CPU + ~2 min |
| 3.8 | `bash scripts/run_r2nn.sh 0` then `python scripts/se2.py data --source inner` | 20 inner R2 models for the nested SE2 scorers | ~10 GPU-h |
| 3.9 | `python scripts/se2.py eval`; `python scripts/a_eval.py loao`; `python scripts/se2_analysis.py sweep`, `select` | per-fold and final threshold chosen by V1 (m = 0.6 -> 0.12744) | ~20 min CPU |

**Mapping to `weights/`:**
- `b0_fold<k>.pt` = 3.1 `last.pt`;
- `r2_final.pt` = 3.3 `last.pt`;
- `se2_scorer.pt` = 3.7 `outputs/research/se2/scorer_final.pt`.

To store new weights in the repository the same way (parts under 50 MB plus a manifest), run:

```bash
cp outputs/exps/b0_r101_s1024/fold0/last.pt weights/b0_fold0.pt   # likewise for the other files
python src/weight_parts.py split weights/b0_fold0.pt weights/
(cd weights && sha256sum *.pt *.part?? > SHA256SUMS)
```

## 4. Predict from the training outputs (scripts)

```bash
python scripts/predict_r2b0.py test --out outputs/submissions/r2b0_5fold_t05_a500_r2a300_fc20_isl5.csv
python src/filseg_cli.py --data MAGFiLO_1.0_Kaggle_2026 --out outputs/submission_r2b0_se2.csv   # final (SE2)
python scripts/check_submission_csv.py outputs/submission_r2b0_se2.csv
```

**What each command does:**
- `predict_r2b0.py` reads the models of 3.1 and 3.3 from `outputs/exps/`. It applies the existence gate, the
  candidates, R2, assembly and the cleanup (defaults `--min_part 20 --island_frac 0.05`): the CSV before the
  small-filament stage.
- `src/filseg_cli.py` (and the notebook) run the whole final pipeline, including SE2 at threshold 0.12744, from the
  checkpoints in `weights/`. Copy newly trained files there first (section 3, mapping above). Its SE2 path was
  checked identical to the validation code `scripts/se2.py` on 33 test images.

## 5. Validation (optional): reproduces the numbers in `docs/validation.md`

The order below follows the dependencies. All CPU steps use at most 4 workers.

| Step | Commands | What it gives |
|---|---|---|
| Stage-1 post-processing | `scripts/tune_post.py --exp b0_r101_s1024 --folds 0,1,2,3,4 --probs val_probs_last.npz`; `scripts/eval_oof.py --exp b0_r101_s1024 --folds 0,1,2,3,4 --thr 0.5 --min_area 500` | threshold 0.5 and gate 500 px by nested CV; OOF PQ 0.409 |
| Nested stage 1 and R2 | `scripts/nested_queue.py --gpus 0 --inner_first`: 10 inner `train_seg.py --exp b0n_r101_s1024 --holdout k,j`, then `nested_stage1.py ens/cache/verify`, then 5 `train_refine.py --variant r2 --fold k --exp R2n_context` | inner out-of-fold stage 1, `B_ens`, R2n models (15 GPU runs, 7.8 GPU-hours) |
| R2 evaluation | `scripts/nested_r2b0_eval.py table`, `loao`, `analyse`; `scripts/scorer_v1_check.py`; `scripts/eval_dual.py table`, `loao`, `analyse` | R2 vs `B_ens` (V0, V1, LOAO, review set) |
| Cleanup | `scripts/fragment_cleanup.py table`, `loao`, `analyse`; `scripts/island_cleanup.py table`, `loao`, `analyse` | 20 px and 5 % cleanup records |
| SE2 | `scripts/run_r2nn.sh`; `scripts/se2.py data`, `train`, `eval`, `loao`, `analyse`; `scripts/se2_analysis.py sweep` | nested SE2 scorers, the registered rule, every threshold multiplier (`records/se2/`) |
| Final pipeline | `scripts/a_eval.py eval`, `loao`; `scripts/se2_analysis.py select`, `classify` | the final pipeline's nested out-of-fold records (V0 0.4316, V1 0.3814), the V1 threshold choice and its leave-one-annotator-out aspects |
| Constants | `scripts/tune_nested.py sweep`, `select`, `combined`, `loao`, `analyse` | sensitivity and nested automatic tuning of the post-processing constants |
| Report analyses | `scripts/error_budget.py`; `scripts/rubric_metrics.py run`, `figure`; `scripts/axis_analysis.py run`, `figure`; `scripts/human_missed.py run`; `scripts/miss_decomposition.py run`, `figure`; `scripts/make_figures_a.py`; `scripts/fig_annotator_cases.py` (after the rows above) | where stage 1 loses filaments, the judging-rubric metrics, the central axis vs the annotators' spines (`records/report_analyses/`), filaments one annotator missed, the figures in `docs/figures/` |

**Tests.** Each `tests/test_*.py` runs its tests from its `__main__` block (`python tests/test_metrics.py`).
`scripts/run_tests_forkserver.py` runs the core suites host-safely.
