"""Run the nested R2b0 experiment (D-035) on the GPUs we are allowed to use, in dependency order.

  GPU jobs : 10 inner b0 runs (train_seg.py --exp b0n_r101_s1024 --holdout k,j, unchanged b0 recipe, 60 epochs,
             validation only at the last epoch) -> per outer fold k, once its 4 inner runs exist: R2n_k
             (train_refine.py --variant r2 --fold k --cache outputs/cache/refine_nested_o<k> --exp R2n_context).
  CPU jobs : per outer fold, in the background: nested_stage1.py ens / cache / verify (a failed verify blocks that
             fold's R2n); after all R2n: verify again (incl. the R2n training-file checks) and, only if all 5 pass,
             nested_r2b0_eval.py table / loao / analyse.
  Not run  : R2_final (all 707 files) -- only with --final AND outputs/nested/eval/summary.json rule.ADOPT == true.

Safety: never kills or signals any process. A GPU gets a new job only if it is in --gpus, has no job of ours and has
>= need + margin GB free; no new launch before --start or after --last_launch; touching outputs/nested/PAUSE stops new
launches (running jobs finish). Each job runs in its own session with CPU affinity (--cores_gpu0/1, --cores_cpu).
Done jobs (summary.json) are skipped; a job is launched at most twice (a failed run folder is renamed
*_failed_<time>). After a queue restart, jobs recorded in outputs/nested/queue_running.json whose process still runs
(/proc/<pid>/cmdline matches) are tracked, not relaunched. A prep is trusted only if its verify file is newer than the
inner runs, the ensemble and the cache it checked.

  setsid nohup .venv/bin/python scripts/nested_queue.py --gpus 0,1 --start "2026-10-28 20:00" \
      --last_launch "2026-10-29 08:00" > logs/nested_queue.out 2>&1 < /dev/null &
  .venv/bin/python scripts/nested_queue.py --dry_run          # print the plan only
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from nested_stage1 import PAIR_ORDER, inner_runs, run_name  # noqa: E402

PY = str(ROOT / ".venv" / "bin" / "python")
EXPS = ROOT / "outputs" / "exps"
STATE = ROOT / "outputs" / "nested"
RUNNING = STATE / "queue_running.json"
NEED_GB = {"b0": 14.0, "r2": 5.5}
MAX_LAUNCH = 2
INNER, ENS, R2N, R2F = "b0n_r101_s1024", "b0n_ens", "R2n_context", "R2_final_context"


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--margin_gb", type=float, default=4.0)
    p.add_argument("--start", default="", help="'YYYY-mm-dd HH:MM' local time; wait until then")
    p.add_argument("--last_launch", default="", help="no new GPU job after this time")
    p.add_argument("--cores_gpu0", default="16-21")
    p.add_argument("--cores_gpu1", default="22-27")
    p.add_argument("--cores_cpu", default="28-31")
    p.add_argument("--inner_first", action="store_true", help="queue inner b0 runs before R2n (better with ONE GPU: "
                   "the CPU prep of the last outer folds then overlaps with R2n training)")
    p.add_argument("--final", action="store_true", help="also train R2_final on all files (only after ADOPT)")
    p.add_argument("--no_eval", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def now():
    return dt.datetime.now()


def log(msg):
    line = f"{now():%F %T} {msg}"
    print(line, flush=True)
    with open(ROOT / "logs" / "nested_queue.log", "a") as fh:
        fh.write(line + "\n")


def free_gb(gpu):
    """Free memory in GB; 0 on any nvidia-smi problem (then nothing is launched on that GPU)."""
    try:
        out = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        return float(out) / 1024
    except Exception:
        return 0.0


def env(gpu=None):
    e = dict(os.environ, NUMPY_MADVISE_HUGEPAGE="0", CUBLAS_WORKSPACE_CONFIG=":4096:8", CUDA_DEVICE_ORDER="PCI_BUS_ID")
    e["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)    # same numbering as nvidia-smi --id
    return e


class Tracked:
    """A job started by an earlier queue process (still running): polled through /proc, never signalled."""

    def __init__(self, pid, needle):
        self.pid, self.needle, self.returncode = pid, needle, None

    def poll(self):
        if self.returncode is None and not alive(self.pid, self.needle):
            self.returncode = -999      # exit code unknown; success is judged by summary.json
        return self.returncode


def alive(pid, needle):
    try:
        return needle in Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except OSError:
        return False


def gpu_jobs(a):
    """GPU jobs in priority order: (name, kind, run dir, argv, dependencies). R2n_k comes first so it starts as soon as
    outer fold k is prepared (blocked jobs are skipped, so inner b0 runs proceed meanwhile)."""
    jobs, inner = [], []
    for k, j in PAIR_ORDER:
        lo, hi = sorted((k, j))
        r = run_name(k, j)
        inner.append((f"{INNER}/{r}", "b0", EXPS / INNER / r,
                      [PY, "scripts/train_seg.py", "--exp", INNER, "--holdout", f"{lo},{hi}", "--epochs", "60",
                       "--val_every", "60", "--workers", "6", "--no_best"], []))
    for k in (2, 3, 4, 0, 1):
        jobs.append((f"{R2N}/fold{k}", "r2", EXPS / R2N / f"fold{k}",
                     [PY, "scripts/train_refine.py", "--variant", "r2", "--fold", str(k), "--epochs", "30",
                      "--exp", R2N, "--cache", str(ROOT / "outputs" / "cache" / f"refine_nested_o{k}")],
                     [f"prep{k}"]))
    jobs = inner + jobs if a.inner_first else jobs + inner
    if a.final:
        jobs.append((f"{R2F}/all", "r2", EXPS / R2F / "all",
                     [PY, "scripts/train_refine.py", "--variant", "r2", "--all", "--epochs", "30", "--exp", R2F],
                     ["adopted"]))
    return jobs


def prep_cmds(k):
    s = [PY, "scripts/nested_stage1.py"]
    return [s + ["ens", "--outer", str(k)], s + ["cache", "--outer", str(k), "--workers", "4"],
            s + ["verify", "--outer", str(k)]]


def prep_valid(k):
    """verify_o<k>.json passed and is newer than everything it checked (inner runs, ensemble, cache)."""
    v = STATE / f"verify_o{k}.json"
    if not v.exists() or json.load(open(v)).get("all_ok") is not True:
        return False
    deps = [EXPS / INNER / r / "summary.json" for _, r in inner_runs(k)] + [
        EXPS / ENS / f"fold{k}" / "val_probs_last.npz", ROOT / "outputs" / "cache" / f"refine_nested_o{k}" / "index.json"]
    return all(d.exists() and d.stat().st_mtime < v.stat().st_mtime for d in deps)


def start_cpu(cmds, cores, tag):
    """Background CPU chain (cmd1 && cmd2 && ...) under taskset; returns the Popen."""
    chain = " && ".join(" ".join(["taskset", "-c", cores] + c) for c in cmds)
    logf = open(ROOT / "logs" / f"nested_{tag}.log", "a")
    log(f"CPU start {tag}: {chain}")
    return subprocess.Popen(["bash", "-c", chain], cwd=ROOT, env=env(None), stdout=logf, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)


def run_cpu(cmds, cores, tag):
    p = start_cpu(cmds, cores, tag)
    rc = p.wait()
    log(f"CPU end {tag} rc={rc}")
    return rc == 0


def save_running(running):
    json.dump({str(g): dict(name=v[0], pid=v[2].pid, needle=v[3]) for g, v in running.items()},
              open(RUNNING, "w"), indent=1)


def main():
    a = parse()
    gpus = [int(g) for g in a.gpus.split(",") if g != ""]
    cores = {0: a.cores_gpu0, 1: a.cores_gpu1}
    assert set(gpus) <= set(cores), f"core sets are defined for GPUs {sorted(cores)} only"
    jobs = gpu_jobs(a)
    if a.dry_run:
        for name, kind, d, argv, deps in jobs:
            print(f"{name:32s} {kind} need {NEED_GB[kind]} GB deps={deps} done={(d / 'summary.json').exists()}\n    "
                  + " ".join(argv[1:]))
        for k in range(5):
            print(f"prep{k}: after {[r for _, r in inner_runs(k)]}: " + " ; ".join(" ".join(c[2:]) for c in prep_cmds(k)))
        return
    STATE.mkdir(parents=True, exist_ok=True)
    start = dt.datetime.strptime(a.start, "%Y-%m-%d %H:%M") if a.start else None
    last = dt.datetime.strptime(a.last_launch, "%Y-%m-%d %H:%M") if a.last_launch else None
    while start and now() < start:
        time.sleep(30)
    log(f"queue start gpus={gpus} last_launch={a.last_launch or '-'} pid={os.getpid()}")
    running, launches, flags = {}, {}, set()
    if RUNNING.exists():     # jobs of a previous queue process that are still alive: track, do not relaunch
        for g, v in json.load(open(RUNNING)).items():
            if alive(v["pid"], v["needle"]):
                d = next(j[2] for j in jobs if j[0] == v["name"])
                running[int(g)] = (v["name"], d, Tracked(v["pid"], v["needle"]), v["needle"])
                launches[v["name"]] = 1
                log(f"tracking still-running {v['name']} pid={v['pid']} on GPU{g}")
    flags |= {f"prep{k}" for k in range(5) if prep_valid(k)}
    ev = STATE / "eval" / "summary.json"
    if ev.exists() and json.load(open(ev))["rule"].get("ADOPT") is True:
        flags.add("adopted")
    preps, failed_prep = {}, set()
    while True:
        for g, (name, d, proc, _) in list(running.items()):          # finished GPU jobs
            if proc.poll() is not None:
                ok = (d / "summary.json").exists()
                log(f"GPU{g} end {name} rc={proc.returncode} ok={ok}")
                del running[g]
                save_running(running)
                if not ok and d.exists():
                    d.rename(d.with_name(d.name + f"_failed_{now():%m%d%H%M%S}"))
                    log(f"  {name}: failed run folder renamed")
        for k, p in list(preps.items()):                              # finished CPU preps
            if p.poll() is not None:
                del preps[k]
                if p.returncode == 0 and prep_valid(k):
                    flags.add(f"prep{k}")
                    log(f"CPU end prep_o{k}: verify passed")
                else:
                    failed_prep.add(k)
                    log(f"PREP/VERIFY FAILED for outer {k} (rc={p.returncode}): its R2n job is blocked")
        for k in range(5):                                            # start preps whose inner runs are complete
            if f"prep{k}" in flags or k in failed_prep or k in preps:
                continue
            if all((EXPS / INNER / r / "summary.json").exists() for _, r in inner_runs(k)):
                preps[k] = start_cpu(prep_cmds(k), a.cores_cpu, f"prep_o{k}")
        busy = {v[0] for v in running.values()}
        todo = [j for j in jobs if not (j[2] / "summary.json").exists() and j[0] not in busy
                and launches.get(j[0], 0) < MAX_LAUNCH]
        if not todo and not running and not preps:
            break
        launch_ok = not (STATE / "PAUSE").exists() and not (last and now() > last)
        for name, kind, d, argv, deps in todo:
            if not launch_ok:
                break
            if any(x not in flags for x in deps):
                continue
            g = next((g for g in gpus if g not in running and free_gb(g) >= NEED_GB[kind] + a.margin_gb), None)
            if g is None:
                continue
            if d.exists():          # partial folder of an attempt that died (e.g. before a queue restart): keep it aside
                d.rename(d.with_name(d.name + f"_failed_{now():%m%d%H%M%S}"))
                log(f"  {name}: leftover partial run folder renamed before launch")
            logf = open(ROOT / "logs" / f"{name.replace('/', '_')}.log", "a")
            proc = subprocess.Popen(["taskset", "-c", cores[g]] + argv, cwd=ROOT, env=env(g), stdout=logf,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            needle = " ".join(argv[1:])
            running[g] = (name, d, proc, needle)
            launches[name] = launches.get(name, 0) + 1
            save_running(running)
            log(f"GPU{g} start {name} (launch {launches[name]}/{MAX_LAUNCH}) pid={proc.pid} cores={cores[g]}")
            time.sleep(60)          # let the job allocate memory before the next free-memory check
        if not running and not preps and todo:
            if not launch_ok:
                log("no launch allowed (PAUSE file or past --last_launch) and nothing running: stopping")
                break
            if all(any(x not in flags for x in j[4]) for j in todo):
                log(f"remaining jobs blocked by {sorted({x for j in todo for x in j[4]} - flags)}: stopping")
                break
        time.sleep(30)
    r2_done = [k for k in range(5) if (EXPS / R2N / f"fold{k}" / "summary.json").exists()]
    verified = [k for k in r2_done   # verify again, now including the R2n training-file checks (E)
                if run_cpu([[PY, "scripts/nested_stage1.py", "verify", "--outer", str(k)]], a.cores_cpu, f"verify_o{k}")]
    if a.no_eval:
        pass
    elif verified == [0, 1, 2, 3, 4]:
        e = [PY, "scripts/nested_r2b0_eval.py"]
        run_cpu([e + ["table", "--workers", "4"], e + ["loao", "--workers", "4"], e + ["analyse"]], a.cores_cpu, "eval")
    else:
        log(f"evaluation NOT run: R2n done {r2_done}, post-training verify passed {verified}")
    log(f"queue end; R2n done for outer folds {r2_done}, verified {verified}")


if __name__ == "__main__":
    main()
