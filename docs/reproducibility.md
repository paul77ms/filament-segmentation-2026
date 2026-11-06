# Reproducibility

## 1. Reproduce the submitted CSV (no training)

Python 3.12 or newer is required (numpy 2.5.4 and scipy 1.18.1 in `requirements.txt` need it); tested with
Python 3.13.13 on Linux x86_64. With an older system Python, create the environment from a newer one, e.g.
`conda create -n filseg python=3.13`.

```bash
git clone https://github.com/paul77ms/filament-segmentation-2026.git && cd filament-segmentation-2026
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt            # every package the repository uses, pinned (==)

# competition data from the Kaggle page; unzip in the repository root -> MAGFiLO_1.0_Kaggle_2026/{train,test}/
unzip filament-segmentation-2026.zip       # or: export FILSEG_DATA=/path/to/MAGFiLO_1.0_Kaggle_2026
# found automatically also next to the repository, in the current folder and in /kaggle/input/filament-segmentation-2026

jupyter nbconvert --to notebook --execute notebooks/reproduce_submission.ipynb \
    --output reproduce_submission.executed.ipynb --ExecutePreprocessor.timeout=-1
```

**What the notebook does**
1. Rebuilds the 7 checkpoints from `weights/*.partNN` and checks every SHA-256 (`weights/README.md`).
2. Shows each stage on a test image.
3. Shows the training and validation records with the training commands (training is not re-run).
4. Predicts all 180 test images -> `outputs/submission_r2b0_se2.csv`. Only the test images are read; no annotations.
5. Compares the CSV with the submitted one (`reference/submission_r2b0_se2.csv`), matching filaments by overlap.
6. Evaluates on held-out training images: each stage-1 fold model on the fold it never saw (stage 1 + existence gate;
   `FILSEG_EVAL_N` images per fold, default 20, 0 = all 707), with the official metric definitions; the nested
   out-of-fold results of the whole pipeline are printed from `records/` (their 40 inner models are not stored).

**Training from scratch inside the notebook:** set `RUN_TRAINING = True` in cell 3c; it runs every training command
in order and section 4 then predicts with the newly trained models instead of the stored weights.

**Expected result**
- **GPU:** on an NVIDIA RTX A6000 (bf16 autocast, deterministic cuDNN) the output is byte-identical to the submitted
  CSV (`records/release_checks/`); about 10 minutes for 180 images (3.1–3.4 s per image, peak GPU memory 2.4 GB).
- **CPU:** fp32, about 14 s per image with 4 threads. A few pixels near a threshold can flip; the notebook reports the
  IoU distribution and a verdict whose tolerances were fixed in advance (`src/reproduce_utils.py`).

**Options** (environment variables): `FILSEG_N_LIMIT=N` (first N images, a quick check), `FILSEG_THREADS` (CPU
threads), `FILSEG_OUT` (output CSV), `FILSEG_DATA` (data folder).

**Without the notebook**

```bash
python src/filseg_cli.py --data MAGFiLO_1.0_Kaggle_2026 --out outputs/submission_r2b0_se2.csv
```

```python
import sys; sys.path.insert(0, "src")
import filseg_infer_r2b0 as R2B0, reproduce_utils as U
U.prepare_weights("weights")                     # rebuilds the .pt files from their parts, checks SHA-256
b0 = [f"weights/b0_fold{k}.pt" for k in range(5)]
R2B0.run("MAGFiLO_1.0_Kaggle_2026/test/test_images", b0, "weights/r2_final.pt", "weights/se2_scorer.pt",
         "outputs/submission_r2b0_se2.csv")      # SE2 threshold 0.12744 (filseg_infer_se.SE2_TAU)
```

## 2. Packages

- `requirements.txt`: every package the repository imports (inference, notebook execution, training, evaluation and
  analysis), with exact versions. Inference alone needs only torch, torchvision, numpy and opencv.
- `requirements-lock.txt`: the fully resolved environment of the tested run (all transitive dependencies).
- Tested with Python 3.13 on Linux x86_64 (torch 2.6.0 + CUDA 12.4).

## 3. Train from scratch

`entry_points.md` gives every command in order; `SETTINGS.json` holds the paths. In summary:

| Step | Command | Time (RTX A6000) |
|---|---|---|
| CV split | `scripts/make_splits.py` | < 1 min (CPU) |
| stage 1, 5 folds (also the test models) | `scripts/train_seg.py --exp b0_r101_s1024 --fold k --epochs 60` | 57–81 min per fold, 14 GB |
| post-processing choice (nested) | `scripts/tune_post.py` | minutes (CPU) |
| R2 cache and final R2 | `scripts/build_refine_cache.py`, then `scripts/train_refine.py --variant r2 --all --exp R2_final_context` | minutes (CPU) + 36 min, 5.2 GB |
| nested stage 1 and R2 (also the SE2 training data) | `scripts/nested_queue.py` (10 inner stage-1 + 5 R2n models) | 7.8 GPU-h |
| SE2 nested validation (inner R2 models) | `scripts/run_r2nn.sh` (20 R2 models, folds outside {k, j}) | ~10 GPU-h |
| SE2 scorers and threshold | `scripts/small_exist.py cands`, `train` (break-even tau_k); `scripts/se2.py data`, `train`, `eval`; `scripts/a_eval.py loao`; `scripts/se2_analysis.py sweep`, `select` | ~30 min CPU + ~2 min GPU |
| test CSV | the notebook (or `src/filseg_cli.py`) | ~10 min |

Re-training on another machine gives equivalent models, not bit-identical ones (cuDNN). Training uses only the
competition's training masks and torchvision's ImageNet ResNet-101 initialisation.

## 4. Hardware and execution time

| | |
|---|---|
| GPU | 1 x NVIDIA RTX A6000 (48 GB; driver 535); every run used a single GPU |
| CPU / RAM | Intel Xeon w5-3433 (32 threads; jobs pinned to 4–6 cores); 125 GB RAM |
| OS | Ubuntu 20.04.6 LTS |

| | Train (RTX A6000) | Predict, 180 test images |
|---|---|---|
| final pipeline | ~6.2 GPU-hours for the test models (stage 1: 5 x 57–81 min; R2: 36 min; SE2 scorer: < 1 min) + the nested models that produce the SE2 training data (~18 GPU-hours) | ~10 min on GPU (3.1–3.4 s per image end to end incl. reading, post-processing and RLE; peak GPU memory 2.4 GB); about 14 s per image on CPU (4 threads) |

## 5. Checks behind the claims

- Byte-identical reproduction of the submitted CSV: `records/release_checks/`.
- Leakage checks (no model saw its evaluation fold; inference opens only the weights and the test images):
  `docs/validation.md`.
- Tests: `tests/` (metric against the official notebook, RLE, scorer rule, nested design);
  `scripts/run_tests_forkserver.py` runs the core suites.
