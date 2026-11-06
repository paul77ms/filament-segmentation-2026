"""CPU wiring test of the nested refinement design (D-035). No GPU, no training; <= 4 cores.

Stand-in inner runs are built from the EXISTING b0 OOF (inner run hold{a}{b} 'predicts' a fold-f file with the b0
fold-f probability). With these stand-ins the nested stage 1 of outer fold 0 must reproduce refine_v1 exactly, so the
whole chain (train_seg file selection -> ens -> cache -> R2 dataset -> verify -> evaluation) has a known answer:
  1 train_seg.select_files: every holdout pair trains on exactly the other three folds
  2 nested cache == refine_v1 for the same files (prob_u8, candidates, boxes, windows, candidate->GT matches)
  3 R2 input tensors / targets from the nested cache == from refine_v1
  4 verify passes; a planted leak (a fold-0 file in an inner run's train_files) and a wrong R2n cache make it fail
  5 nested_r2b0_eval: B_ens / R2nb0 / R2nfull per-record counts == b0 / R2b0_any_a300 / R2full_a300 of the earlier
    analysis (outputs/refine_candidate/records.csv) on the same files
  6 compare_paired refine / refineb0 instances unchanged by the config-driven cache lookup (snapshot before the edit)
  7 the production cache path (static_from=refine_v1, all 707 files) would accept refine_v1's GT row order
  8 evaluation gates: an R2n trained from refine_v1 is refused; LOAO keeps the consensus aspects; without the 5
    post-R2n verify files ADOPT is False
Usage: CUDA_VISIBLE_DEVICES="" taskset -c 0-3 .venv/bin/python tests/test_nested_design.py <scratch_dir>
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

N = 6                                    # files per fold (fold 0: 3 multi-set files, two with 3 sets -> LOAO consensus)
SCR = Path(sys.argv[1]).resolve()
EX = SCR / "exps"
PY = str(ROOT / ".venv" / "bin" / "python")
ENV = dict(os.environ, FILSEG_EXPS=str(EX), CUDA_VISIBLE_DEVICES="", NUMPY_MADVISE_HUGEPAGE="0")
COMMON = ["--inner_exp", "b0n_fake", "--ens_exp", "ens_fake", "--r2_exp", "R2n_fake",
          "--cache_prefix", str(SCR / "cache_o"), "--files_per_fold", str(N)]
results = {}


def ok(name, cond, detail=""):
    results[name] = bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail else ""), flush=True)


def run(args, expect=0):
    r = subprocess.run([PY] + args, cwd=ROOT, env=ENV, capture_output=True, text=True)
    if r.returncode != expect:
        print(r.stdout[-3000:], r.stderr[-3000:])
    return r


shutil.rmtree(SCR, ignore_errors=True)
EX.mkdir(parents=True)
(EX / "b0_r101_s1024").symlink_to(ROOT / "outputs" / "exps" / "b0_r101_s1024")
sp = pd.read_csv(ROOT / "outputs" / "splits" / "folds_v1.csv")
sub = sp.sort_values("file_name").groupby("fold", group_keys=False).head(N)

# 1 file selection of the inner runs
from train_seg import select_files  # noqa: E402
from nested_stage1 import PAIR_ORDER, run_name  # noqa: E402
good = True
for k, j in PAIR_ORDER:
    lo, hi = sorted((k, j))
    tr, va, d = select_files(sp, 0, f"{lo},{hi}")
    trf, vaf = set(sp[sp.file_name.isin(tr)].fold), set(sp[sp.file_name.isin(va)].fold)
    good &= (trf == set(range(5)) - {k, j} and vaf == {k, j} and not set(tr) & set(va) and len(tr) + len(va) == len(sp)
             and d == run_name(k, j))
ok("1_select_files_holdout_pairs", good and len(PAIR_ORDER) == 10 and len({run_name(*p) for p in PAIR_ORDER}) == 10)

# stand-in inner runs
b0 = {f: np.load(ROOT / "outputs/exps/b0_r101_s1024" / f"fold{f}" / "val_probs_last.npz") for f in range(5)}
for k, j in PAIR_ORDER:
    lo, hi = sorted((k, j))
    d = EX / "b0n_fake" / run_name(k, j)
    d.mkdir(parents=True)
    vs = sub[sub.fold.isin([lo, hi])]
    np.savez_compressed(d / "val_probs_last.npz", **{r.file_name[:-5]: b0[r.fold][r.file_name[:-5]] for r in vs.itertuples()})
    tr, va, _ = select_files(sp, 0, f"{lo},{hi}")
    (d / "train_files.txt").write_text("\n".join(tr)); (d / "val_files.txt").write_text("\n".join(va))
    json.dump(dict(train_folds=[f for f in range(5) if f not in (lo, hi)], holdout=f"{lo},{hi}"), open(d / "config.json", "w"))
    json.dump({}, open(d / "summary.json", "w"))

r = run(["scripts/nested_stage1.py", "ens", "--outer", "0"] + COMMON)
ok("ens_runs", r.returncode == 0, r.stdout.strip()[-120:])
r = run(["scripts/nested_stage1.py", "cache", "--outer", "0", "--static_from", "", "--workers", "4"] + COMMON)
ok("cache_runs", r.returncode == 0, r.stdout.strip()[-160:])

# 2 nested cache == refine_v1 on the same files
from filseg import refine as RF  # noqa: E402
c_new, c_old = SCR / "cache_o0", RF.CACHE
i_new, i_old = json.load(open(c_new / "index.json")), json.load(open(c_old / "index.json"))
P_new, P_old = np.load(c_new / "prob_u8.npy", mmap_mode="r"), np.load(c_old / "prob_u8.npy", mmap_mode="r")
C_new, C_old = np.load(c_new / "cand.npy", mmap_mode="r"), np.load(c_old / "cand.npy", mmap_mode="r")
G_new, G_old = np.load(c_new / "gt.npy", mmap_mode="r"), np.load(c_old / "gt.npy", mmap_mode="r")
same_p = same_c = same_meta = same_gt = True
for i, f in enumerate(i_new["files"]):
    io = i_old["files"].index(f)
    same_p &= np.array_equal(P_new[i], P_old[io])
    same_c &= np.array_equal(C_new[i], C_old[io])
    mn, mo = i_new["meta"][f], i_old["meta"][f]
    rn = {i_new["set_ids"][r]: r for r in mn["set_rows"]}
    ro = {i_old["set_ids"][r]: r for r in mo["set_rows"]}
    same_meta &= (mn["boxes"] == mo["boxes"] and mn["windows"] == mo["windows"] and set(rn) == set(ro)
                  and all(mn["match"][f"{b[0]}|{rn[s]}"] == mo["match"][f"{b[0]}|{ro[s]}"] for b in mn["boxes"] for s in rn))
    same_gt &= all(np.array_equal(G_new[rn[s]], G_old[ro[s]]) for s in rn)
ok("2_prob_u8_equal_refine_v1", same_p); ok("2_candidates_equal", same_c); ok("2_meta_equal", same_meta)
ok("2_gt_equal", same_gt)
ok("2_sources_recorded", all(len(v) == 1 for v in i_new["stage1_sources"].values()) and i_new["outer"] == 0)

# 3 R2 inputs / targets identical
import train_refine as TR  # noqa: E402
files = sorted(i_new["files"])
dn = TR.WinSet(i_new, files, "r2", train=False, cache=c_new)
do = TR.WinSet(i_old, files, "r2", train=False, cache=c_old)
same_x = len(dn) == len(do) and all(dn.items[t][1:5] == do.items[t][1:5] for t in range(len(dn)))
for t in range(0, len(dn), max(1, len(dn) // 12)):
    xn, tn, _ = dn[t]; xo, to, _ = do[t]
    same_x &= bool((xn == xo).all() and (tn == to).all())
ok("3_r2_inputs_targets_equal", same_x, f"{len(dn)} windows")

# 4 verify, planted leaks
r = run(["scripts/nested_stage1.py", "verify", "--outer", "0", "--out", str(SCR / "verify")] + COMMON)
ok("4_verify_passes", r.returncode == 0)
d = EX / "b0n_fake" / "hold01"
orig = (d / "train_files.txt").read_text()
(d / "train_files.txt").write_text(orig + "\n" + sub[sub.fold == 0].file_name.iloc[0])
r = run(["scripts/nested_stage1.py", "verify", "--outer", "0", "--out", str(SCR / "verify_leak")] + COMMON, expect=1)
v = json.load(open(SCR / "verify_leak" / "verify_o0.json"))
ok("4_planted_train_leak_detected", r.returncode == 1 and not v["A_hold01_no_outer_val_file_in_train"]["ok"])
(d / "train_files.txt").write_text(orig)

# stand-in R2n fold 0 = the existing R2 fold-0 refined candidates (same candidates as refine_v1)
z = np.load(ROOT / "outputs/exps/R2_refine_context/fold0/val_cand_probs.npz")
keep = {f[:-5] for f in sub[sub.fold == 0].file_name}
rd = EX / "R2n_fake" / "fold0"
rd.mkdir(parents=True)
np.savez_compressed(rd / "val_cand_probs.npz", **{k: z[k] for k in z.files if k.split("|")[0] in keep})
(rd / "val_files.txt").write_text((EX / "ens_fake" / "fold0" / "val_files.txt").read_text())
(rd / "train_files.txt").write_text("\n".join(sorted(sp.file_name[sp.fold != 0])))
json.dump(dict(cache_dir=str(c_new), exist_exp="ens_fake", train_folds=[1, 2, 3, 4]), open(rd / "config.json", "w"))
json.dump({}, open(rd / "summary.json", "w"))
r = run(["scripts/nested_stage1.py", "verify", "--outer", "0", "--out", str(SCR / "verify2")] + COMMON)
v = json.load(open(SCR / "verify2" / "verify_o0.json"))
ok("4_verify_with_r2n_passes", r.returncode == 0 and v["E_r2_cache"]["ok"] and v["E_r2_train_has_no_outer_val_file"]["ok"])
json.dump(dict(cache_dir=str(RF.CACHE), exist_exp="ens_fake", train_folds=[1, 2, 3, 4]), open(rd / "config.json", "w"))
r = run(["scripts/nested_stage1.py", "verify", "--outer", "0", "--out", str(SCR / "verify3")] + COMMON, expect=1)
ok("4_wrong_r2n_cache_detected", r.returncode == 1)
r = run(["scripts/nested_r2b0_eval.py", "table", "--folds", "0", "--ens_exp", "ens_fake", "--r2_exp", "R2n_fake",
         "--cache_prefix", str(SCR / "cache_o"), "--files_per_fold", str(N), "--workers", "4", "--out", str(SCR / "eval_bad")],
        expect=1)
ok("8_eval_refuses_r2n_from_refine_v1", r.returncode != 0 and "not the nested cache" in r.stderr)
json.dump(dict(cache_dir=str(c_new), exist_exp="ens_fake", train_folds=[1, 2, 3, 4]), open(rd / "config.json", "w"))

# 5 evaluation reproduces the earlier R2b0 analysis on these files
r = run(["scripts/nested_r2b0_eval.py", "table", "--folds", "0", "--ens_exp", "ens_fake", "--r2_exp", "R2n_fake",
         "--cache_prefix", str(SCR / "cache_o"), "--files_per_fold", str(N), "--workers", "4", "--out", str(SCR / "eval")])
ok("5_eval_table_runs", r.returncode == 0, r.stdout.strip()[-100:])
new = pd.read_csv(SCR / "eval" / "records.csv")
old = pd.read_csv(ROOT / "outputs/refine_candidate/records.csv")
cols = ["tp_host", "fp_host", "fn_host", "sum_iou_host", "fn_split", "fn_merged", "fp_merge", "thin_gt_covered_px",
        "thin_pred_px"]
for a_, b_ in (("B_ens", "b0"), ("R2nb0", "R2b0_any_a300"), ("R2nfull", "R2full_a300"), ("b0_single", "b0"),
               ("B_ens_a300", "b0_a300")):
    x = new[new.method == a_].set_index("rid").sort_index()[cols]
    y = old[(old.method == b_) & old.rid.isin(x.index)].set_index("rid").sort_index()[cols]
    ok(f"5_{a_}_equals_old_{b_}", len(x) > 0 and x.index.equals(y.index) and np.allclose(x.values, y.values),
       f"{len(x)} records")
r = run(["scripts/nested_r2b0_eval.py", "loao", "--folds", "0", "--ens_exp", "ens_fake", "--r2_exp", "R2n_fake",
         "--cache_prefix", str(SCR / "cache_o"), "--files_per_fold", str(N), "--workers", "4", "--out", str(SCR / "eval")])
ok("8_eval_loao_runs", r.returncode == 0, r.stdout.strip()[-80:])
r = run(["scripts/nested_r2b0_eval.py", "analyse", "--ens_exp", "ens_fake", "--out", str(SCR / "eval"),
         "--verify_dir", str(SCR / "verify2"), "--n_boot", "200"])
ok("5_eval_analyse_runs", r.returncode == 0)
sm = json.load(open(SCR / "eval" / "summary.json"))
gt_l = pd.read_csv(SCR / "eval" / "loao_ref_instances.csv")
asp = sm["loao"]["aspects"]
ok("8_loao_consensus_aspect_present", gt_l.c_best_iou.notna().any()
   and all(np.isfinite(asp["consensus_matched_rate"][k]) for k in ("human", "B_ens", "R2nb0")))
ok("8_loao_covers_multiset_files", sm["rule"]["loao_covers_all_multiset_files"] is True
   and len(sm["loao"]["stems"]) == int(((sub.fold == 0) & (sub.n_sets >= 2)).sum()))
ok("8_no_adopt_without_5_verifies", sm["rule"]["verify_all_ok_incl_r2n"] is False and sm["rule"]["ADOPT"] is False)
ok("8_table_coverage_gate", sm["rule"]["table_covers_all_val_files"] is True)

# 6 compare_paired unchanged for the pre-D-035 runs
import compare_paired as C  # noqa: E402
before = json.load(open(SCR.parent / "nested_reg" / "refine_instances_before.json"))
same = True
for key, val in before.items():
    g, s = key.split("|")
    m = dict(C.parse_method(f"R2_refine_context:{g}:0.5:300"), b0_exist=g == "refineb0")
    P = C._refine_instances(m, s, 1)
    same &= [len(P), int(P.areas.sum()), hashlib.sha256(P.union().tobytes()).hexdigest()] == val
ok("6_compare_paired_refine_unchanged", same, f"{len(before)} cases")

# 7 production cache build: refine_v1 covers exactly the files / GT rows the nested build would create
allf = sorted(sp.file_name)
_, _, set_ids = RF.file_sets(allf)
ok("7_static_from_refine_v1_rows_match", i_old["files"] == allf and i_old["set_ids"] == set_ids, f"{len(set_ids)} rows")

print(json.dumps(dict(passed=sum(results.values()), total=len(results)), indent=1))
json.dump(results, open(SCR / "results.json", "w"), indent=1)
sys.exit(0 if all(results.values()) else 1)
