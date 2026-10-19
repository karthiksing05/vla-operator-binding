# experiments/

One flat directory, because scripts import each other by name and every ledger row records the script that wrote it. The
prefix of a script names the plan it belongs to. Plans are in `docs/` (current) and `docs/history/` (superseded).

## Infrastructure (used by everything)

| scripts | role |
|---|---|
| `serve_guided.py`, `launch_pack.py`, `libero_client.py`, `roll*.sbatch`, `roll.sh` | serve a policy with a composition mode (`--mode`); launch rollouts packed per GPU; the LIBERO client |
| `fresh_states.py` | the fresh start-state sets (`results/robot/fresh_*`) |
| `audit_numbers.py`, `audit_coverage.py`, `make_numbers.py` | every number in the paper checked against the ledger |
| `paper_page.py`, `report/`, `site_media.py`, `fig_*.py` | the paper page and its figures |
| `prereg/` | one file per pre-registered read (each records an `m3_prereg` ledger row before any episode) |
| `split_grades.py` | the compositional split graded by unseen parts and relations (docs/SPLIT.md) |

## The current paper, by plan

| prefix | plan | what |
|---|---|---|
| `c0_fit.py` | all | fits a unit (prototype / leaf). `--cache` (frozen prefix cached once; 8.6× faster, same losses), `--init-best`, `--heldout` (docs/FIT_TIME.md) |
| `e0`–`e5` | D / E (docs/history/PLAN_FACTORED.md) | the **supervised scheme**: object, instance and goal prototypes; the hierarchy; the stream |
| `g0`–`g23` | G (docs/history/PLAN_GENERALIZE.md) | the system iterated and tested: recombination, baselines, other policies, Li's OOD suites, TLI, operators, streams, leaves, ablations, LIBERO-PRO |
| `h1`–`h4` | H (docs/history/PLAN_H.md) | interpretation (what units donate), cost, scaling, new backbones |
| `w0`–`w2`, `t0` | W (docs/history/PLAN_WANG.md) | Wang et al.'s selection; transfer limits; streams |
| `i1`–`i7` | interpretation of the library (docs/INTERPRETATION.md) | prototypes, primitives, behaviour, factors, novelty, anatomy |
| `z1`–`z4` | Z (docs/PLAN_Z.md) | composition with nothing fitted: FLOWER (Z1), word splice (Z2), LIBERO-90 pointers (Z3), pointer programs (Z4) |
| `zu_*` | Z, track U | the **unsupervised scheme**: likelihood scores of demos (`zu_score`), boundary tests (`zu_sweep`, `zu_kcut`, `zu_read`); docs/UNSUPERVISED.md |
| `zi_*` | Z, track I | donation analysis: expert velocities (`zi_velocities`) and the soft blend with Shapley donations (`zi_blend`) |
| `zf_*` | Z, F1 | fit time: profile, levers, the fast-fit confirmatory test (docs/FIT_TIME.md) |

Renamed on 2026-10-19 so that families do not collide. These were the names when they were written:
- `u1_score|sweep|kcut|read.py` → `zu_*`
- `i1_velocities|blend.py` → `zi_*`
- `f1_common|profile|levers|fast_fit|leaf_fit.py` → `zf_*`

Ledger rows written before the rename (`f1_profile`, `f1_levers`) record the old paths. Their data directories keep their names:
`results/robot/u1/`, `results/robot/i1/`, `results/robot/f1/`.

## Earlier programs (kept: their ledger rows cite them)

| prefix | program |
|---|---|
| `b*`, `c1`/`c2`, `d1` | role codes, the scheme, grammar, the first streams (docs/history/METHOD_2026-10-01.md) |
| `f0`, `f1_binding`, `f1_factor_probe`, `f1_fresh`, `f4` | factor coverage, binding reads, token folding |
| `p*`, `q*`, `r*` | the transfer program, prototype PoE, retrieval and borrowing programs (archive/2026-10-program/) |
| `u1_joint.py` | the joint stream of the W program (not the segmentation) |

Scripts that no doc, figure, ledger row or other script uses are in `archive/experiments/`.
