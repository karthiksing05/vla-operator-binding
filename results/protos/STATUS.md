# results/protos STATUS (P2 / P3 development pre-checks on pi05_libero)

Updated 2026-10-22 05:40.  DONE.  Plan: docs/PLAN_V4.md P2 / P3; write-up: docs/FRAMEWORK.md section 5.
Code: lifelong_ood/protos/{store,relation}.py, experiments/pp_{prep,fields}.py, pp2_dev.py, pp3_dev.py, tests/test_protos.py.
Ledger: ledger/pp2_dev/, ledger/pp3_dev/ (tag pi05_libero, dirty=true).  Payloads: pp2_dev.json, pp3_dev.json, obs/ (412 npz).

| step | what | job | state |
|---|---|---|---|
| tests | tests/test_protos.py | 4188051 | 10 passed |
| prep | index (200 train + 100 LIBERO-90 + 120 transfer obs) + raw hdf5 fetch + match | 4189290 | done; 39/40 training demos matched exactly (task 38 libero_spatial:2: no raw match, its 5 obs lack object positions) |
| poses | object positions at the chosen frames (LIBERO venv) | 4189408 | done, 295 frames; replayed eef vs dataset eef mm-level, max 5 cm |
| binds | splice source bindings | 4189378 | done |
| fields | modes / extents / cached fields | 4189379-82, 4189711 | done, 412 obs (LIBERO-90 task 48 has 2 first frames: 8 transfer obs skipped) |
| analysis | pp2_dev (4189725), pp3_dev (4189720) | | done |

Headline (development):
- P2 kill as specified: separation passes only degenerately (starts collapse; 1.2 clusters at rms 0.15); attribution 1.3x chance and
  the null prompt points at the same objects; transfer r 0.07 (bar 0.5).
- P3 below bar: R0 explains 0.51 of the instruction-vs-null gap (bar 0.70; plain parts), stability 0.86 (bar 0.8), equal weights 0.34,
  Comp-3 0.59; without libero_spatial (post hoc) R0 0.73.  R1 no gain.  Readout: object AND, goal THEN(post) on rotation.

Rerun on the split checkpoint: pp_fields.py --checkpoint <dir> --tag <t> (4 shards), then pp2_dev.py --tag=<t> --record, pp3_dev.py --tag=<t> --record.
