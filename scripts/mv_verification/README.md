# MV metric/RLE adversarial verification scripts (CPU only)

Owner: task MV. Results and interpretation: docs/metrics_verification.md.
Run with `OMP_NUM_THREADS=1 .venv/bin/python <script>` from this directory.

| script | what it does |
|---|---|
| bf_pq.py | independent brute-force PQ (python pixel sets, exhaustive matching) + documented split/merge/causes |
| synth_compare.py [N] | N randomised synthetic cases: brute force vs filseg.metrics vs verbatim host notebook (torch CPU) |
| synth_more.py | 3 more seeds x 2000 small cases + 60 large (2048^2, 517x300, 1023x2047) filament-like cases (10-13 GT, 30 preds) |
| real_compare.py [out_dir] | all 707 train stems: RLE checks on all 8199 masks; self (1154) and ordered-pair (1196) records: brute force vs filseg vs notebook. Env NW = workers (default 12). Writes mv_real_rle_checks.csv, mv_real_pq_records.csv, mv_real_summary.json |
| rle_adv.py | RLE edge shapes + mutation tests (C-order encode/decode) against tests/test_rle.py |
| splitmerge_cases.py | crafted split/merge cases (prints counters) |
| split_alt.py | containment vs coverage-based split/merge counts on inter-annotator pairs |
| timing.py | timing of a 10-GT / 30-pred record (filseg vs host notebook) |
| diag_dense.py | boundary F (d=2), thin region (r=3) and semantic Dice vs a dense scipy reference on ~400 synthetic cases |
