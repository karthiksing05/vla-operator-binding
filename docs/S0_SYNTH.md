# S0: the composition machinery on a synthetic manipulation domain (2026-10-22)

This is the robot-independent core of docs/FRAMEWORK.md: TRELLIS-style symbolic composition over continuous
Gaussian prototypes, tested on a synthetic domain with known units. All code is numpy only.

- Code: `lifelong_ood/compose/` (`gcobweb.py`, `store.py`, `parse.py`, `generate.py`, `learn.py`, `synth.py`).
- Tests: `tests/test_compose.py`, 12 tests, about 15 s.
- Experiment: `experiments/s0_synth.py`, which writes `results/s0/{config}_seed{s}.json` and
  `results/s0/summary.json`.
- Sweeps: `results/s0/sweep_*/`.
- SLURM scripts: `slurm/s0/`. Every run went through overcap. One run is about 40 s on one CPU.

`lifelong_ood/compose/` is a package, so it shadows the old module file `lifelong_ood/compose.py`. That file is
unchanged. The package's `__getattr__` loads it lazily and re-exports its names, so
`from lifelong_ood.compose import poe_teacher` still works (there is a test for this).

## The domain (`synth.py`)

- Codes z are in R^16, split into three blocks: T (dims 0-5, unit type), O (dims 6-10, object) and G (dims
  11-15, goal).
- Units:
  - approach(o) = [t_app, e_o, 0]
  - grasp(o) = [t_gr, e_o, 0]
  - carry(o,g) = [t_car, e_o, f_g]
  - release(g) = [t_rel, 0, f_g]
- Each unit lasts 5-15 frames. Each frame adds a drift of 0.3 along a type direction and Gaussian noise with
  sigma 0.15.
- Grammar: Task -> PP | PP PP (probability 0.3 for two). PP -> Pick(o) Place(o,g). Pick -> approach grasp.
  Place -> carry release.
- Observation features (12 dims): gripper closure, holding, e_o and f_g of the current pick-place. They carry
  the contact events used by the contact-event baseline.
- Splits:
  - objects 0-4 and goals 0-3 are seen;
  - G-rec holds out four object x goal combinations: (0,3), (1,0), (2,1), (3,2). Every object and goal is
    still seen in other combinations;
  - G-new uses object 5, which is never seen.
- The ground truth (units, constituents, an oracle nearest-unit decoder) is used only for scoring and for the
  closed-loop Env stand-in.

## Design choices, thresholds and scoring formulas

| piece | choice (Config default) | source / reason |
|---|---|---|
| Cobweb | A port of `cobweb_torch.py`. var = meanSq/n + prior_var (0.01). The score is KL(child‖parent), summed over dims (weighted) and over nominal attributes (Laplace alpha). PU = mean_c p(c)·KL(c‖parent). Operators: best / new / merge / split, plus the fringe split and the exact-match leaf. Categorisation is best-first with score log p(c\|x) + log p(x\|c). The basic level is argmax over the path of p(c)·KL(c‖root). | torch code |
| nominal bags | P(bag\|C,a) = Σ_v w_v P(v\|C,a), a mixture: the bag is one uncertain value | TRELLIS App. B ("small bag over concept identifiers") |
| context description | [self code \| left \| right \| obs], with block weights 1 / 0.5 / 0.5 / 0.5. left and right are 1/distance-weighted means of up to W=3 surface frames outside the span, with zeros at the sequence ends (EMPTYNULL). obs is taken at the span start. **self is visible**, because a continuous primitive has no symbol: its code is its identity (TRELLIS keeps a hidden content-ref instead). | TRELLIS 4.1 / App. B |
| content description | then: {rel, L: bag, R: bag, Lc, Rc}. with: {rel, P: bag}. A bag holds the 3 deepest nodes on the part's context path with count ≥ 2, uniformly weighted. Tags: then-nesting depth; a with-composite fills one span and gets tag 0. | TRELLIS ("up to three concept ids"); the depth-4 rule is replaced because depth moves under merge |
| prototype pool | every non-root context node with count ≥ 5; each node's self-block marginal | FRAMEWORK O3 |
| with selection | greedy submodular F(S) = Σ_t Σ_r max_{j∈S} l_{j,r}(z_t) over the span's frames, at most K=3 prototypes. A prototype is added only if its gain per frame exceeds 8 nats. | Wang et al. Eqs. 10-11 (summed over frames) |
| PoE | closed form: Λ = Σ_j w_j/σ²_j, μ = Λ⁻¹ Σ_j w_j μ_j/σ²_j, with w_j(r) = softmax_j(mean_t l_{j,r}/τ), τ = 0.5 | Wang et al. Eqs. 9, 12 |
| span coverage | mean per-frame, per-dimension log-likelihood of the span's frames under its PoE | *my choice*: using the pooled code alone let broad nodes cover whole demos |
| primitive merge (parse) | Delta = LL(A) + LL(B) − LL(A∪B), where LL is total frame log-likelihood under the span's own cover. A merge is recognised iff Delta ≤ 32 nats and the union's coverage ≥ θ_cover. The cheapest merge is committed first. | *my choice*: a likelihood-ratio segmentation test. The pool's count ≥ 5 is the TRELLIS context-path gate for primitives. |
| θ_cover | the 0.5 % quantile of single calibration frames' coverage, minus 0.25 | calibrated, no labels |
| NEW (P4) | a final span with coverage < θ_new is NEW. θ_new is the 1 % frame-weighted quantile of the learner's own spans on 10 calibration demos, minus 0.1. Adjacent NEW spans merge into one remainder. | calibrated, no labels. The margin is a trade-off (see below). |
| then candidates | adjacent non-NEW elements. Each candidate is sorted through both taxonomies, and each taxonomy's score is s = log P(C\|i) + log p(i\|C)/n_attrs at the halting node. Rank = s_c + s_x. | TRELLIS App. B ranks by the posterior. I added the per-attribute likelihood because path posteriors saturate near 0 and do not separate candidates. |
| recognition gate | walk up from the content halting node to the first non-root ancestor with count > τ_parse = 5. Recognised iff that ancestor exists and the instance's per-attribute log-likelihood under it ≥ θ_content. θ_content is the 1 % quantile on calibration composites, minus 1. | TRELLIS App. B uses the count gate alone (τ = 30, "transparent after training"), which would never reject anything here. τ is scaled to a 40-demo corpus. |
| learner's parse | the same parse with the gate off (every then-candidate is admitted), so the content taxonomy bootstraps without supervision | TRELLIS trains on gold parses; S0 has none |
| L1 | windows go to ctx. with-windows go to cnt as well. Committed composites go to cnt and ctx, after their parts. | TRELLIS 3.4 / 4.4 |
| L2 (create) | the frames of a NEW remainder are incorporated into ctx, so Cobweb's new / fringe operators make the prototype. The demo is then parsed again. | FRAMEWORK L2 (code only; no adapter in S0) |
| consolidate | a split removes a context node; content counts on its id move to its surviving ancestor | TRELLIS App. B (canonicalise references) |
| generation | the pool is every stored then-decomposition under the deepest ancestor of the seed's content leaf with count > τ_gen = 10, restricted to the slot's complexity tag (the class the parent expects; without this rule expansion recursed forever). Candidates are ranked by a context callback. The default callback scores [left = last emitted code, obs = the controller's current observation] against the candidate's stored context, using one shared per-dimension variance (the median over the pool). The arg-max is taken. Emitted codes are the covers' PoE means, executed by `Env` (oracle decode). | TRELLIS P9 / App. B |

## Results (3 seeds, mean ± sd)

Setup: 40 discovery and learning demos, 10 calibration demos and 20 in-distribution test demos per seed.
Continual runs use a separate domain seed.

| metric | full | flat | no split | no with |
|---|---|---|---|---|
| **(a)** boundary F1 (±2 frames), in-dist test | **1.000 ± 0** | 0.766 ± 0.165 | 1.000 ± 0 | 1.000 ± 0 |
| contact-event baseline F1 | 0.360 ± 0.019 | (same) | | |
| in-dist parse complete (one root, no NEW) | 0.967 ± 0.047 | 0.917 | 0.983 | 0.967 |
| **(b)** unit-concept recall (a node ≥ 90 % pure holding ≥ 80 % of a unit) | 0.887 ± 0.018 | 0.070 | 0.910 | 0.887 |
| type-concept recall | 0.417 ± 0.312 | 0.000 | 0.583 | 0.417 |
| basic level: depth / type purity / unit purity | 1.33 / 0.75 / 0.20 | 1.0 / 0.55 / 0.32 | 1.34 / 0.73 / 0.17 | same as full |
| constituent precision of the learner's commits | 0.664 ± 0.047 | 0.563 | 0.572 | 0.670 |
| **(c)** G-rec: NEW fraction of the one demo | 0.043 ± 0.033 | 0.000 | 0.043 | 0.221 |
| G-rec: carry span covered by a with-composite | **1.000** | 0.50 | 1.000 | 0 |
| G-rec: parse complete (stored elements only) | 0.833 ± 0.118 | 1.000 | 0.833 | **0.000** |
| G-rec generation, exact sequence, zero demos (generic seed) | 0.000 | 0.000 | 0.000 | 0.000 |
| G-rec generation, exact, after one demo (seed = its parse root) | **0.917 ± 0.118** | 0.000 | 0.917 | 0.917 |
| G-rec generation, unit accuracy after one demo | 0.979 ± 0.029 | 0.167 | 0.979 | 0.917 |
| G-rec generation, generic seed after one demo (recombination) | 0.167 ± 0.236 | 0.000 | 0.000 | 0.333 |
| **(d)** G-new demo 1: NEW fraction of new-object units / other units | **1.000 / 0.000** | 0.000 / 0.000 | 0.949 / 0 | 1.000 / 0 |
| prototypes created | 5.3 ± 0.5 | 0 | 5.3 | 5.3 |
| demo 2 (new object, other goal): NEW fraction of new-object units | 0.293 ± 0.254 | 0.111 | 0.206 | 0.440 |
| without creation: demo-2 NEW fraction of new-object units | 1.000 | 0.078 | 1.000 | 1.000 |
| demo 2: new-object spans that reuse a created prototype | **1.000** | 0 | 0.889 | 1.000 |
| demo 2: parse complete / generation exact after learning it | 0.333 / 1.000 | 0.667 / 0 | 0.667 / 1.000 | 0 / 1.000 |
| **(e)** stream of 30: drops in earlier-task frame coverage (of 435 pairs) | 5.3 ± 4.1 (worst drop 0.40) | 70 ± 97 | 2.7 | 1.3 |
| drops in earlier-task complete parse (of 435) | **30.7 ± 7.9** | 45.7 | **1.0 ± 1.4** | 28.7 |
| final mean recognition of all 30 tasks: coverage / complete | 0.990 / 0.878 | 0.884 / 0.600 | 0.997 / 0.978 | 0.998 / 0.889 |
| prototypes added per task, when the task brings new parts / no new parts | 3.55 / 2.25 | 0 / 0 | 4.05 / 2.29 | 5.03 / 1.99 |
| corr(prototypes, tasks) / corr(prototypes, parts seen) | 0.99-1.00 / 0.94-0.97 | n/a | same | same |

Context-taxonomy purity by depth (full, seed 0, window instances only):

| depth | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| type purity | 0.75 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| unit purity (type + argument) | 0.17 | 0.40 | 0.63 | 0.87 | 0.91 | 0.98 | 1.00 |

Unit type separates at depth 2, the argument (object or goal) by depth 4-6. The content taxonomy's purity over
unit-type sequences is 0.38 at depth 1, 0.81 at depth 2 and ≥ 0.98 from depth 4. The learner's commits match
Pick 23, Place 39, PP 15 and Task 40 times out of 168.

### Sweeps (full config, `results/s0/sweep_*`)

| setting | G-rec complete | G-rec 1-demo exact | G-new demo-1 detected | G-new demo-2 NEW | stream: complete drops / coverage drops |
|---|---|---|---|---|---|
| default (bag of 3, θ_new margin 0.1) | 0.83 | 0.92 | 1.00 | 0.29 | 30.7 / 5.3 |
| bag = full path | 0.75 | 0.75 | 1.00 | 0.39 | 15.3 / 4.3 |
| θ_new margin 0.3 | 1.00 | 0.75 | **0.67** (misses one seed) | 0.00 | 33.3 / 3.7 |
| both | 0.92 | 0.58 | 0.67 | 0.00 | 18.0 / 0.0 |

## What works

- **Segmentation emerges from recognition.** The parse recovers every unit boundary on in-distribution demos
  (F1 = 1.00, 3 seeds). The contact-event baseline scores 0.36. The flat store drops to 0.77, because its
  Cobweb collapses to 3-6 broad clusters.
- **The hierarchy recovers the units.** Context nodes are type-pure at depth 2 and unit-pure by depth 6. 89 %
  of units exist as a concept. The CU basic level sits at depth about 1.3, which is roughly unit type, not
  unit + argument.
- **"with" carries G-rec.** The unseen carry(o, g) is covered by a PoE of carry(o, ·) and carry(·, g) in 100 %
  of the held-out cells, so the one demo is parsed from stored prototypes only. Without "with", every G-rec
  carry span is NEW and no G-rec parse completes (0.00); creation then has to make a new primitive for it.
- **One demo is enough to generate the new combination.** After incorporating one demo, top-down generation
  ranked against the closed-loop observation reproduces the new unit sequence in 11 of 12 cells (unit
  accuracy 0.98).
- **G-new is detected and reused.**
  - The new object's units are 100 % NEW and the other units 0 % NEW. Creation adds about 5 prototypes.
  - In a second task with the new object, every covered new-object span reuses a created prototype. carry(new,
    g') is covered by "with" from a created prototype and a stored one.
  - Without creation, the second task stays 100 % NEW on those units.

## What fails or is weaker than claimed

1. **Preservation is not guaranteed at the composite level.**
   - Codes never change, but earlier tasks lose their complete parse in about 31 of 435 (task, later-time)
     pairs. In seed 0 the mean complete rate over the stream sinks to 0.39 at task 13 before it recovers to 0.68.
   - The cause is reference drift. As the context taxonomy deepens and restructures, a part's bag of
     deepest-node ids changes, and its content instance no longer matches what was stored. Consolidation
     handles only split, not merge or deepening.
   - Evidence for this cause:
     - the no-split ablation, whose single taxonomy also sees the continuous context, drops only 1.0 ± 1.4
       times;
     - full-path bags halve the drops.
   - Coverage-level drops (5 of 435, worst 0.40 of a task's frames) come from routing: new pool nodes change
     which prototypes are selected.
   - This is exactly the L3 point: preservation has to be measured, and here it is violated through routing.
2. **The store grows with tasks, not only with parts.**
   - Prototypes grow by 2.3 per task even when a task brings no new part, versus 3.5 when it does.
     corr(prototypes, tasks) is 0.99.
   - Cobweb stores every instance, and repeated windows push new nodes over the count-5 pool threshold.
   - Rules (content leaves with count ≥ 2) grow slowly, by about 0.2 per task.
   - The claim "grows with parts, not tasks" does not hold as built.
3. **Per-dimension PoE can fake-cover a new object.**
   - A new object's O block can be mosaicked from other objects' dimensions. Its coverage then overlaps that of
     legitimate with-composites:
     - seed 1: new-object windows reach 0.15, against G-rec with-windows at 0.13-0.45;
     - in-distribution windows are ≥ 0.29.
   - No single θ_new separates "new part" from "new combination". The margin sweep shows this trade-off directly:
     - with margin 0.1, G-new is detected in 3 of 3 seeds, but 4 % of G-rec frames and 29 % of demo-2
       new-object frames (the with-composite of a created and a stored prototype) are wrongly NEW;
     - with margin 0.3, G-rec and demo 2 are clean, but G-new is missed in 1 of 3 seeds.
4. **Zero-demo recombination does not work.**
   - Generation from a generic training seed under the new scene gets the unit types right but the arguments
     wrong. Seed 0's unit accuracy is 0.0-0.25 per cell with no demo; across seeds the exact rate is 0.00 with no demo and 0.17 after the demo.
   - The pools are narrow (recall anchors with count > 10 and the same tag), and the context ranking cannot
     build a carry(o*, g*) that was never stored.
   - Generation succeeds only when the demo's own decomposition is in the pool.
5. **Bracketing is arbitrary.** Learner commits match a ground-truth constituent 66 % of the time. Adjacency
   alone cannot tell [[A G][C R]] from [A [G C] R]. Generation does not depend on the bracketing.
6. **The flat ablation is degenerate.** Flat CU partitions collapse to 3-6 clusters, so its G-new "non-detection"
   and its 0.77 F1 measure that collapse, not a tuned flat baseline.
7. **Several mechanisms are mine, not TRELLIS's** (listed in the table above): the span-coverage score, the
   Delta merge test, the calibrated content floor, the per-attribute ranking term, the complexity-tag slot
   filter and the shared context scale. Each was added after a measured failure:
   - the pooled-code coverage merged whole demos into one span;
   - the count gate alone never rejected anything;
   - generation without the tag filter recursed forever;
   - per-candidate variances let broad concepts win the ranking.
   θ_cover, θ_new and θ_content are calibrated on unlabeled in-distribution demos. Everything else was set by
   hand, mostly on seed-0 diagnostics: gain_min = 8 and seg_pen = 32 nats, the calibration margins, n_proto,
   τ_parse, τ_gen, the bag size and the block weights.

## Reproduce

```
sbatch slurm/s0/s0_full.sbatch                      # 4 configs x 3 seeds -> results/s0/
sbatch slurm/s0/s0_args.sbatch --configs full --out results/s0/sweep_bagfull --set "bag_size=1000"
python -m pytest -q tests/test_compose.py           # ~15 s
```
