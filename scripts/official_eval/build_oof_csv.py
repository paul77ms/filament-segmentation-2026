"""Build the inputs for the OFFICIAL self-evaluation notebook from the b0 OOF probabilities.

  * pred CSV  : one row per predicted filament for all 707 train files (each file predicted by the fold
                model that did NOT train on it), in submission format ``filament_id,segmentation_rle``,
                ids ``<stem>_<k>`` (k 0-based, as scripts/predict.py), post-processing identical to the
                final submission: seg.prob_to_instances(p, PostCfg(thr=0.5, min_area=500)) -> rle.encode.
  * GT CSV    : the train COCO json in the notebook's GT layout, one row per GT filament,
                ``filament_id = "<annotator_id>-<image_id>_<k>"`` = ``<COCO image id>_<k>`` (COCO image ids are
                already "<GGGGAA>-<stem>"), segmentation_rle = pycocotools compressed counts of the polygon
                (filseg.rle.ann_to_rle, which is pycocotools frPyObjects -> merge), k = JSON order.

Usage: NUMPY_MADVISE_HUGEPAGE=0 CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/official_eval/build_oof_csv.py --out DIR
"""
import os

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filseg import data as D  # noqa: E402
from filseg import rle as R  # noqa: E402
from filseg import seg  # noqa: E402
from filseg.metrics import single_thread_worker  # noqa: E402

EXP = ROOT / "outputs" / "exps" / "b0_r101_s1024"
THR, MIN_AREA = 0.5, 500


def pred_rows(args):
    stem, npz = args
    p = np.load(npz)[stem].astype(np.float32)
    inst = seg.prob_to_instances(p, seg.PostCfg(THR, MIN_AREA))
    rows = []
    cover = np.zeros((seg.H, seg.W), bool)
    for k in range(len(inst)):
        m = inst.full_mask(k)
        assert not np.any(cover & m)
        cover |= m
        c = R.encode(m)
        assert np.array_equal(R.decode(c).astype(bool), m)
        rows.append((R.make_filament_id(stem, k), c))
    return stem, rows


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import multiprocessing as mp
    single_thread_worker()

    jobs, fold_of = [], {}
    for f in range(5):
        for fn in (EXP / f"fold{f}" / "val_files.txt").read_text().split():
            stem = Path(fn).stem
            assert stem not in fold_of, stem
            fold_of[stem] = f
            jobs.append((stem, str(EXP / f"fold{f}" / "val_probs_last.npz")))
    assert len(jobs) == 707, len(jobs)
    with mp.get_context("forkserver").Pool(a.workers, initializer=single_thread_worker) as pool:
        res = dict(pool.map(pred_rows, jobs, chunksize=2))
    rows = [r for stem, _ in jobs for r in res[stem]]
    n_pred = R.write_submission(rows, out / "oof_pred_b0_t05_a500.csv")

    coco = D.load_coco()
    by_img = {}
    for ann in coco["annotations"]:
        by_img.setdefault(ann["image_id"], []).append(ann)
    gt_rows = []
    n_sets = 0
    for im in coco["images"]:
        anns = by_img.get(im["id"], [])
        n_sets += 1
        assert anns, f"empty set {im['id']}"
        assert Path(im["file_name"]).stem in fold_of
        for k, ann in enumerate(anns):
            c = R.ann_to_rle(ann)["counts"]
            gt_rows.append((f"{im['id']}_{k}", c.decode("ascii") if isinstance(c, bytes) else c))
    n_gt = R.write_submission(gt_rows, out / "gt_train_sets.csv")
    meta = dict(exp=str(EXP), probs="val_probs_last.npz", thr=THR, min_area=MIN_AREA, n_files=len(jobs),
                n_pred_rows=n_pred, n_gt_rows=n_gt, n_gt_sets=n_sets,
                files_without_pred=sum(1 for s, _ in jobs if not res[s]),
                fold_of=fold_of,
                sha256={"oof_pred_b0_t05_a500.csv": sha(out / "oof_pred_b0_t05_a500.csv"),
                        "gt_train_sets.csv": sha(out / "gt_train_sets.csv")})
    json.dump(meta, open(out / "build_meta.json", "w"), indent=1)
    print({k: v for k, v in meta.items() if k != "fold_of"})


if __name__ == "__main__":
    main()
