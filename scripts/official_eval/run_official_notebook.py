"""Execute the OFFICIAL organiser self-evaluation notebook on a (GT CSV, prediction CSV) pair.

The code cells are read from self-evaluation-notebook.ipynb at run time (sha256 checked) and
exec'd unchanged, with exactly these adaptations:
  * code cell 1 (the "@TODO" placeholder ``pred_df = None; gt_df = None``) is replaced by reading the two CSVs
    (``pd.read_csv(..., dtype=str, keep_default_na=False)``);
  * code cells 5-8 are the commented-out calls; their ``# `` prefixes are removed (the resulting statements are
    asserted to be exactly the expected ones);
  * matplotlib uses the Agg backend; each plot cell draws into its own figure, which is saved as a PNG and whose
    displayed numbers (bar heights, legend text) are read back from the axes;
  * --workers > 1: the statement ``overlap_df = get_overlap_df(gt_df, pred_df)`` is evaluated by calling the
    notebook's own ``get_overlap_df`` once per annotator-image (GT rows of that annotator-image only, full pred_df)
    in a forkserver pool, then concatenating the rows in the notebook's order (``gt_annotator_image_ids.unique()``).
    Rows are independent per annotator-image, so this is the same table; --check_serial N verifies it on the first N
    records against one serial call. Worker matrices travel as float32 numpy and are converted back with
    torch.from_numpy (lossless). torch uses 1 thread per worker (integer-valued float32 sums < 2^24 are exact in any
    order, so this cannot change the matrices).
  * --subset N --seed S restricts gt_df to N annotator-images drawn with numpy default_rng(S) (sorted draw order
    is irrelevant: the notebook's order of the remaining rows is kept).

CPU only. Outputs (JSON + PNGs + per-record CSV) go to --out.
"""
import os

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

NB_PATH = ROOT / "self-evaluation-notebook.ipynb"
NB_SHA256 = "d4956c0e92cf999dc4417da40f3b61f6e5705e7d596c021ad22cf272507101c8"
EXPECTED_CALLS = {
    5: ["overlap_df = get_overlap_df(gt_df, pred_df)", "pq_score = get_pq_score(overlap_df)",
        'print(f"PQ Score: {pq_score:.4f}")'],
    6: ['plot_distribution(overlap_df["iou_matrix"], "IoU Distribution", "IoU")'],
    7: ['plot_distribution(overlap_df["dice_matrix"], "Dice Distribution", "Dice")'],
    8: ['plot_m2n_counts(overlap_df, "Many to One and One to Many Distribution")'],
}


def notebook_code_cells():
    raw = open(NB_PATH, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == NB_SHA256, f"notebook changed: {sha}"
    nb = json.loads(raw)
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(cells) == 9, len(cells)
    return cells, sha


def uncomment(src: str) -> str:
    """Remove the '# ' prefix from the commented-out call lines (not from the '@TODO' prose lines)."""
    out = []
    for line in src.splitlines():
        if re.match(r"^# (overlap_df|pq_score|print|plot_)", line):
            out.append(line[2:])
    return "\n".join(out)


def load_inputs(ns: dict, gt_csv: str, pred_csv: str) -> None:
    """Replacement for code cell 1 (the only input cell)."""
    pd = ns["pd"]
    ns["pred_df"] = pd.read_csv(pred_csv, dtype=str, keep_default_na=False)
    ns["gt_df"] = pd.read_csv(gt_csv, dtype=str, keep_default_na=False)


def make_namespace(gt_csv, pred_csv):
    import matplotlib
    matplotlib.use("Agg")
    cells, _ = notebook_code_cells()
    ns: dict = {"__name__": "official_notebook"}
    exec(compile(cells[0], "nb_cell_1_imports", "exec"), ns)          # imports (verbatim)
    load_inputs(ns, gt_csv, pred_csv)                                  # replaces the placeholder cell
    for i, name in ((2, "nb_cell_3_helpers"), (3, "nb_cell_4_pq"), (4, "nb_cell_5_plots")):
        exec(compile(cells[i], name, "exec"), ns)                      # verbatim
    return ns, cells


# ------------------------------------------------------------------ worker side
_W: dict = {}


def _init_worker(gt_csv, pred_csv):
    from filseg.metrics import single_thread_worker
    single_thread_worker()
    import torch
    torch.set_num_threads(1)
    ns, _ = make_namespace(gt_csv, pred_csv)
    ns["_gt_ai"] = ns["gt_df"]["filament_id"].str.split("_", n=1).str[0]
    _W["ns"] = ns


def _one(ai: str):
    ns = _W["ns"]
    t0 = time.time()
    sub = ns["gt_df"][ns["_gt_ai"] == ai]
    od = ns["get_overlap_df"](sub, ns["pred_df"])                     # the notebook's own function
    assert len(od) == 1 and od.iloc[0]["annotator_image"] == ai
    r = od.iloc[0]
    return (ai, r["iou_matrix"].numpy().astype("float32", copy=True), r["dice_matrix"].numpy().astype("float32", copy=True),
            int(r["n_gt"]), int(r["n_pred"]), time.time() - t0)


def overlap_df_parallel(ns, records, gt_csv, pred_csv, workers):
    import multiprocessing as mp

    import pandas as pd
    import torch
    from filseg.metrics import single_thread_worker
    single_thread_worker()
    rows, secs = [], {}
    with mp.get_context("forkserver").Pool(workers, initializer=_init_worker, initargs=(gt_csv, pred_csv)) as pool:
        for k, (ai, iou, dice, ng, npred, dt) in enumerate(pool.imap(_one, records, chunksize=1)):
            rows.append((ai, torch.from_numpy(iou), torch.from_numpy(dice), ng, npred))
            secs[ai] = dt
            if (k + 1) % 50 == 0:
                print(f"  {k + 1}/{len(records)} records", flush=True)
    od = pd.DataFrame({"annotator_image": [r[0] for r in rows]})
    od["iou_matrix"], od["dice_matrix"], od["n_gt"], od["n_pred"] = zip(*[r[1:] for r in rows])
    return od, secs


# ------------------------------------------------------------------ plot readers
def read_distribution_axes(ax):
    bars = [p for p in ax.patches]
    heights = [float(p.get_height()) for p in bars]
    centers = [float(p.get_x() + p.get_width() / 2) for p in bars]
    leg = ax.get_legend()
    labels = [t.get_text() for t in leg.get_texts()] if leg else []
    return dict(bar_centers=centers, bar_heights=heights, legend=labels, title=ax.get_title())


def read_m2n_axes(ax):
    bars = [(float(p.get_x() + p.get_width() / 2), float(p.get_height())) for p in ax.patches
            if p.get_width() > 0 and p.get_height() > 0 and p.get_x() > -1e5]
    ticks = [(float(t), lab.get_text()) for t, lab in zip(ax.get_xticks(), ax.get_xticklabels())]
    leg = ax.get_legend()
    labels = [t.get_text() for t in leg.get_texts()] if leg else []
    return dict(bars=bars, ticks=ticks, legend=labels, title=ax.get_title())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--check_serial", type=int, default=0, help="also run the serial cell on the first N records")
    ap.add_argument("--tag", default="full")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import numpy as np

    t_all = time.time()
    ns, cells = make_namespace(a.gt, a.pred)
    plt = ns["plt"]
    calls = {i: uncomment(cells[i]) for i in EXPECTED_CALLS}
    for i, exp in EXPECTED_CALLS.items():
        assert calls[i].splitlines() == exp, (i, calls[i])

    gt_df, pred_df = ns["gt_df"], ns["pred_df"]
    gt_ai = gt_df["filament_id"].str.split("_", n=1).str[0]
    all_records = list(gt_ai.unique())
    if a.subset:
        rng = np.random.default_rng(a.seed)
        pick = set(rng.choice(len(all_records), size=a.subset, replace=False).tolist())
        keep = {r for k, r in enumerate(all_records) if k in pick}
        gt_df = gt_df[gt_ai.isin(keep)].reset_index(drop=True)
        ns["gt_df"] = gt_df
    records = list(gt_df["filament_id"].str.split("_", n=1).str[0].unique())
    print(f"records={len(records)} gt_rows={len(gt_df)} pred_rows={len(pred_df)}", flush=True)

    t0 = time.time()
    if a.workers <= 1:
        import torch
        torch.set_num_threads(4)
        exec(compile(calls[5].splitlines()[0], "nb_cell_8_overlap", "exec"), ns)   # overlap_df = get_overlap_df(...)
        secs = {}
    else:
        ns["overlap_df"], secs = overlap_df_parallel(ns, records, a.gt, a.pred, a.workers)
    t_overlap = time.time() - t0
    od = ns["overlap_df"]
    assert list(od["annotator_image"]) == records

    serial_check = None
    if a.check_serial:
        import torch
        sub_records = records[: a.check_serial]
        sub_gt = gt_df[gt_df["filament_id"].str.split("_", n=1).str[0].isin(set(sub_records))]
        od_s = ns["get_overlap_df"](sub_gt, pred_df)
        same = all(torch.equal(x, y) for x, y in zip(od_s["iou_matrix"], od["iou_matrix"][: a.check_serial])) and \
            all(torch.equal(x, y) for x, y in zip(od_s["dice_matrix"], od["dice_matrix"][: a.check_serial])) and \
            list(od_s["annotator_image"]) == sub_records and \
            list(od_s["n_gt"]) == list(od["n_gt"][: a.check_serial]) and list(od_s["n_pred"]) == list(od["n_pred"][: a.check_serial])
        serial_check = dict(n=len(sub_records), identical=bool(same))
        print("serial check:", serial_check, flush=True)

    # cell 8 remaining statements: pq_score = get_pq_score(overlap_df); print(...)
    exec(compile("\n".join(calls[5].splitlines()[1:]), "nb_cell_8_pq", "exec"), ns)
    pq_score = float(ns["pq_score"])

    # extra (not in the notebook): TP/FP/FN exactly as get_pq_score counts them, for comparison
    tp = fp = fn = 0
    tp_sum = 0.0
    for row in od.itertuples(index=False):
        if row.n_gt == 0:
            fp += row.n_pred; continue
        if row.n_pred == 0:
            fn += row.n_gt; continue
        hit = row.iou_matrix > 0.5
        tp += int(hit.sum()); tp_sum += float(row.iou_matrix[hit].sum())
        fp += ns["fp_count_hit"](hit); fn += ns["fn_count_hit"](hit)

    shown = {}
    for i, fname in ((6, "iou_distribution"), (7, "dice_distribution"), (8, "m2n_counts")):
        fig = plt.figure(figsize=(8, 5) if i < 8 else (10, 5))
        exec(compile(calls[i], f"nb_cell_{i + 3}", "exec"), ns)
        ax = plt.gca()
        shown[fname] = read_m2n_axes(ax) if i == 8 else read_distribution_axes(ax)
        fig.tight_layout()
        fig.savefig(out / f"official_nb_{a.tag}_{fname}.png", dpi=110)
        plt.close(fig)

    # full-precision versions of what the plots show (same expressions as plot_distribution / plot_m2n_counts)
    def scores(col):
        s = np.concatenate([np.asarray(m, dtype=float).ravel() for m in od[col]] or [np.zeros(0)])
        return s[s > 0]
    iou_s, dice_s = scores("iou_matrix"), scores("dice_matrix")
    from collections import Counter
    gdeg, pdeg = [], []
    for row in od.itertuples(index=False):
        if row.n_gt == 0:
            pdeg.extend([0] * row.n_pred); continue
        if row.n_pred == 0:
            gdeg.extend([0] * row.n_gt); continue
        h = row.iou_matrix > 0
        gdeg.extend(h.sum(dim=1).tolist()); pdeg.extend(h.sum(dim=0).tolist())

    def hist(s):
        c, e = np.histogram(s, bins=np.linspace(0.0, 1.0, 51))
        return dict(counts=c.tolist(), probabilities=(c / c.sum()).tolist() if c.sum() else c.tolist(),
                    bin_edges=e.tolist())

    per_rec = []
    for row in od.itertuples(index=False):
        h = (row.iou_matrix > 0.5) if (row.n_gt and row.n_pred) else None
        per_rec.append(dict(annotator_image=row.annotator_image, n_gt=row.n_gt, n_pred=row.n_pred,
                            tp=int(h.sum()) if h is not None else 0,
                            fp=(ns["fp_count_hit"](h) if h is not None else (row.n_pred if row.n_gt == 0 else 0)),
                            fn=(ns["fn_count_hit"](h) if h is not None else (row.n_gt if row.n_pred == 0 else 0)),
                            sum_iou_f32=float(row.iou_matrix[h].double().sum()) if h is not None else 0.0,
                            n_overlap_pairs=int((row.iou_matrix > 0).sum()) if h is not None else 0,
                            seconds=secs.get(row.annotator_image)))
    import pandas as pd
    pd.DataFrame(per_rec).to_csv(out / f"official_nb_{a.tag}_per_record.csv", index=False)
    res = dict(
        notebook=str(NB_PATH), notebook_sha256=NB_SHA256, gt_csv=a.gt, pred_csv=a.pred,
        subset=a.subset or None, seed=a.seed if a.subset else None, n_records=len(records),
        records=records if a.subset else None,
        n_gt=int(od["n_gt"].sum()), n_pred_record_weighted=int(od["n_pred"].sum()),
        pq_score=pq_score, pq_printed=f"PQ Score: {pq_score:.4f}",
        tp=tp, fp=fp, fn=fn, tp_iou_sum_f32=tp_sum,
        iou_overlap_pairs=dict(n=int(iou_s.size), mean=float(iou_s.mean()), median=float(np.median(iou_s)),
                               hist=hist(iou_s)),
        dice_overlap_pairs=dict(n=int(dice_s.size), mean=float(dice_s.mean()), median=float(np.median(dice_s)),
                                hist=hist(dice_s)),
        gt_one_to_n=dict(sorted(Counter(gdeg).items())), pred_n_to_one=dict(sorted(Counter(pdeg).items())),
        plots_as_displayed=shown, serial_check=serial_check,
        seconds_overlap=t_overlap, seconds_total=time.time() - t_all, workers=a.workers,
        per_record_seconds=dict(median=float(np.median(list(secs.values()))) if secs else None,
                                max=float(max(secs.values())) if secs else None,
                                sum=float(sum(secs.values())) if secs else None),
    )
    json.dump(res, open(out / f"official_nb_{a.tag}.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k not in ("records", "plots_as_displayed", "iou_overlap_pairs",
                                                                 "dice_overlap_pairs")}, indent=1, default=str))
    print("iou mean", res["iou_overlap_pairs"]["mean"], "dice mean", res["dice_overlap_pairs"]["mean"])


if __name__ == "__main__":
    main()
