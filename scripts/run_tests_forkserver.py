"""Run the filseg test suites the way their `__main__` blocks do, but host-safe (helper).

Usage:
  NUMPY_MADVISE_HUGEPAGE=0 CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/run_tests_forkserver.py \
      [--real-metrics --audit_dir DIR] [--workers 8]

Why a wrapper: the test files and filseg.metrics.evaluate_dataset / filseg.diagnostics create
multiprocessing pools with the 'fork' context. The shared-host rule is: call
filseg.metrics.single_thread_worker() first and use the 'forkserver' context. This wrapper does both
(multiprocessing.get_context('fork') is redirected to 'forkserver') and otherwise calls the same test
functions in the same order as each file's `__main__` block:
  tests/test_rle.py --skip-real   -> all test_* except test_real_train_roundtrip
  tests/test_metrics.py           -> all test_* except test_real_smoke (+ test_real_smoke with --real-metrics,
                                     writing its two audit files into --audit_dir instead of outputs/audit)
  tests/test_diagnostics.py       -> all test_*
  tests/test_adversarial.py       -> all test_*
Exit code 1 if any test fails; every failure is printed with its traceback.
"""
from __future__ import annotations

import os
import sys

os.environ["NUMPY_MADVISE_HUGEPAGE"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLBACKEND", "Agg")

import multiprocessing as _mp  # noqa: E402

_orig_get_context = _mp.get_context


def _get_context(method=None):
    return _orig_get_context("forkserver" if method == "fork" else method)


_mp.get_context = _get_context

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    import argparse
    import importlib
    import inspect
    import time
    import traceback

    ap = argparse.ArgumentParser()
    ap.add_argument("--real-metrics", action="store_true")
    ap.add_argument("--audit_dir", default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    from filseg.metrics import single_thread_worker
    single_thread_worker()

    plan = [("test_rle", {"test_real_train_roundtrip"}), ("test_metrics", {"test_real_smoke"}),
            ("test_diagnostics", set()), ("test_adversarial", set())]
    results = []
    for modname, skip in plan:
        mod = importlib.import_module(modname)
        fns = [(k, v) for k, v in vars(mod).items()
               if k.startswith("test_") and inspect.isfunction(v) and v.__module__ == modname and k not in skip]
        if modname == "test_metrics" and a.real_metrics:
            if a.audit_dir:
                os.makedirs(a.audit_dir, exist_ok=True)
                mod.AUDIT = a.audit_dir
            fns.append(("test_real_smoke", lambda: mod.test_real_smoke(n_workers=a.workers)))
        for name, fn in fns:
            t = time.time()
            try:
                out = fn()
                results.append((modname, name, "PASS", time.time() - t, out))
                print(f"PASS {modname}.{name} ({time.time() - t:.1f}s){'  ' + str(out) if out is not None else ''}",
                      flush=True)
            except Exception:
                results.append((modname, name, "FAIL", time.time() - t, None))
                print(f"FAIL {modname}.{name} ({time.time() - t:.1f}s)\n{traceback.format_exc()}", flush=True)
    by_mod = {}
    for m, _, st, _, _ in results:
        d = by_mod.setdefault(m, {"PASS": 0, "FAIL": 0})
        d[st] += 1
    print("SUMMARY", by_mod, flush=True)
    sys.exit(1 if any(r[2] == "FAIL" for r in results) else 0)


if __name__ == "__main__":
    main()
