# Interpretation: what the prototypes and operators do, and what that says about composition (Plan H1)

All three analyses are exploratory and none was pre-registered. Every number below comes from a ledger row:
`ledger/h1_factor` (tags `l90_g5`, `l90_i7`, `l90_g20`, `l90_g20_ood`, `ood_pi05`, `ood_pi0`, `ood_smolvla`, `ood_xvla`,
`likelihood`), `ledger/h1_donate` (tag `read`) and `ledger/h1_tli` (tag `read`). The scripts are `experiments/h1_factor.py`,
`h1_donate.py` and `h1_tli.py`; their docstrings give the method and the command. The figures are in `figures/robot/h1_*.png`.
The rows carry `dirty=true` because other sessions had uncommitted edits to files these scripts never touch.

## 1 Composed success is a product of a grasp competence and a place competence (h1_factor)

**The model.** A composed step runs one unit until the policy's own grasp and another unit after it. If the grasp is the only
interface between the two units, then P(success) = g(approach unit) x p(carry unit), where g is the probability that the first
grasp takes the named object and p is the probability of success given that grasp. The episode JSONs record the object each
first grasp took (`bound`).

**The approach unit fixes the grasp, and only the grasp.** On Li's OOD suites, arms that share an object source but have
different goal sources took the same first object in 1,803 of 1,805 trial groups on pi0.5. The counts are 1,810 of 1,815 on
pi0, 1,765 of 1,775 on SmolVLA and 1,781 of 1,785 on X-VLA (`ood_*`, `grasp_same_object_across_goal_sources`). On LIBERO-90,
the object-only arm and the both-units arm grasp the named object first 676 and 678 times out of 1,060 (G5), and 1,120 and
1,129 times out of 1,320 (G20). The approach is therefore a function of the approach unit alone.

**A wrong first grasp ends the task once a unit is acting.** After a non-named first grasp, the both-units arm succeeds 7 times
in G5 and 3 times in G20. The policy alone succeeds 85 times in that situation in G5 (`l90_g5`, `arms`). Most of those 85
successes come from six scenes where the grasp tracker confuses adjacent objects. The tracker labels the object nearest the
gripper, and these six tasks are flagged in `tracker_confused_tasks`. With w = 0, the model predicts the both-units arm on
G5's 53 held-out tasks, using fold concepts that saw no demonstration of the task: 366 successes predicted against 365
observed, Pearson r 0.86, calibration slope 1.01, intercept -0.004 (`l90_g5`, `in_sample_zero_w`). On the 33 tasks where the
goal-only arm made at least three named grasps, so that p is measured and not only shrunk toward the pooled rate, r is 0.93
and the counts are 246 observed against 260 predicted (`in_sample_estimable_zero_w`). The full model, which adds w from the
goal-only arm, predicts 408. All of its excess sits in the w term: 41.9 predicted against 7 observed (`decomposition`).
Estimated on the practice start states and scored on set b, the product gives r 0.84 (`out_of_sample`).

**On Li's suites the product predicts start states that practice never saw, on all four policies.** G10's practice on set a
crosses 16 object sources with one goal source and 16 goal sources with the chosen object source, five episodes per pair.
From set a alone, the product g(a) x p(g) + (1 - g(a)) x w(g) predicts every grasp-switch arm run on sets c, d, e and f:

| policy | test cells | Pearson r | slope | observed / predicted successes |
|---|---|---|---|---|
| pi0.5 | 129 | 0.93 | 1.01 | 1,566 / 1,415 |
| pi0 | 120 | 0.94 | 1.04 | 1,574 / 1,390 |
| SmolVLA | 48 | 0.91 | 0.88 | 137 / 148 |
| X-VLA | 53 | 0.97 | 1.02 | 445 / 424 |

The grasp competence transfers almost exactly between start-state sets: mean absolute error 0.025 on pi0.5 and 0.026 on pi0,
with bias at most 0.01 except on SmolVLA (-0.054) (`grasp_transfer`). The remaining miss is in the place competence. On pi0
and pi0.5, P(success | named grasp) on the test sets exceeds the practice estimate by 0.087 and 0.059 (`place_transfer`, bias).
Almost all of the misfit cells (24 of 25 on pi0, 24 of 29 on pi0.5) are therefore under-predictions. Within practice, leaving
one cell out, the product fits with r 0.89 to 0.94 (`loo_practice`). A goal source's place competence does not depend on which
object source brought the object: the pooled G-test of homogeneity across object sources gives p = 0.58 (pi0), 0.11 (pi0.5),
1.0 (SmolVLA) and 1.0 (X-VLA) (`stage1_homogeneity`).

**Where the product fails: the hand-off can help.** The interface test compares, task by task, P(success | named grasp) under
the carry unit after the approach unit's grasp with the same quantity after the policy's own grasp:
- G5 fold concepts (no demonstration of the task): +0.002, 95% interval -0.082 to 0.093 over 36 tasks. Independence holds.
- G20, the full library (each unit fitted on demonstrations that include the task's own): +0.117, 95% interval 0.031 to 0.213
  over 43 tasks (`l90_g20`, `interface_both_vs_goal`). The product then under-predicts: 897 observed against 763 predicted,
  slope 0.85, with 17 of 22 misfit tasks above their prediction. Task 34, for example, reaches 0.89 success after the object
  unit's grasp and 0.17 after the policy's own.
- G20 on Li's suites (pi0.5, set e: the chosen pair's object source alone and goal source alone): +0.071, 95% interval 0.013 to
  0.140. The fit is r 0.95, with 282 observed against 254 predicted (`l90_g20_ood`).

So the product is a lower bound, and it is tight when the two units were learned apart (fold concepts). When the units come
from the same demonstrations (the library), or from training tasks that share a grasp (the policy's own fields), the approach
unit ends in the state the carry unit was trained to start from, and success exceeds the product. The one interval that goes
the other way: under G5 fold concepts, the policy's own carry does slightly worse after the object unit's grasp than after its
own (-0.080, 95% interval -0.159 to -0.002; `interface_obj_vs_zs`). A unit's grasp suits a carry that expects that grasp. In
I7 (10 tasks, 5 episodes), the product over-predicts (49 against 40), and its p rests on 12 named grasps on four tasks, which
is too few to read further.

**Does the policy's likelihood predict the failures?** Partly, and only through the grasp. On G5, the frozen pi0.5's flow loss
on an object concept's LIBERO-90 frames (ledger i6_novelty) correlates with the object unit's grasp competence at Spearman
-0.49 (p = 0.0002) and with composed success at -0.28 (p = 0.040). It does not predict the product's residual: -0.16
(p = 0.26) for the object loss and 0.00 for the goal loss. The goal concept's loss does not predict the goal unit's place
competence (-0.13, p = 0.34) (`likelihood`, `l90_g5`). On Li's suites, the held concepts' losses (13 of 20 tasks have both)
correlate with the residuals inconsistently across policies, between -0.43 and +0.47. The likelihood says a concept is
unfamiliar and harder to grasp. It does not say where the product will miss.

**What this means for composition in general.** For these flow policies, composing a pick-and-place reduces to two independent
competences joined at one observable event: which object is in hand. The approach unit determines g completely, the carry unit
determines p, and a wrong grasp gives zero. Therefore (i) composed success can be predicted from cheap per-component
measurements, here five-episode practice cells, on start states practice never saw (r 0.91 to 0.97 on four architectures).
(ii) Choosing sources by practice is choosing the argmax of a product, and the two factors can be searched one after the other,
as G10 does. (iii) Composition fails in two ways: the grasp fails (the factor that transfers exactly), or the hand-off state
differs from the state the carry was learned from. The second failure can run in either direction, and units fitted together
show positive coupling. Grant et al. (arXiv 2603.19233) find that goal selection and motor generation split between the VLM
and the action expert. The split here is at a different boundary, approach versus carry, and it holds whether the unit is an
action-expert LoRA (the library) or a VLM instruction (the grasp switch).

Figures: `figures/robot/h1_factor_l90.png` (predicted against observed; each arm's g, p and w; residual against concept loss)
and `figures/robot/h1_factor_ood.png` (out-of-sample prediction on the four policies).

## 2 What a prototype donates (h1_donate; pi0.5, the 48 prototypes of the E3 / E5 library)

Each prototype was fitted on demonstration 0 of its tasks. It is measured on the held-out demonstration 1 of up to four of
those tasks, at frames of its own phase and of the other phase, with 4 shared noise draws (method in the script's docstring).

**Object and instance prototypes donate WHERE; goal prototypes donate WHERE and WHEN.** Error of the sampled chunk to the
held-out demonstration's chunk, in normalised action units, prototype off -> on (own phase; `by_kind.*.own`):

| kind (n) | position | orientation | gripper | release / close timing (steps) |
|---|---|---|---|---|
| object type (19) | 1.205 -> 0.897 | 1.026 -> 1.006 | 1.321 -> 1.257 | 2.29 -> 2.67 |
| instance (8) | 1.282 -> 1.158 | 1.333 -> 1.427 | 1.090 -> 1.133 | 1.73 -> 2.35 |
| goal (21) | 1.995 -> 1.242 | 1.516 -> 1.570 | 1.489 -> 1.117 | 4.31 -> 3.32 |

No kind of prototype corrects orientation. Object prototypes cut the position error by 26% and leave the gripper alone; their
close timing gets slightly worse (better on 7 of 19). Goal prototypes cut the position error by 38% and the gripper error by
25%, and they move the release closer to the demonstration's (better on 9 of 15). Direction: the cosine between the executed
displacement (the first 5 actions) and the vector from the end effector to the phase's target (the demonstration's grasp point
for objects, its release point for goals) rises from 0.66 to 0.76 for object prototypes, against 0.79 for the demonstration
itself. For goal prototypes it rises from 0.22 to 0.53, against 0.58 for the demonstration. The change a goal prototype makes
points at the release point (cosine 0.33, positive for 19 of 21). The change an object prototype makes is nearly orthogonal to
the grasp point (0.07): it redirects an approach that already heads roughly the right way. Instances add little over that
(cosine of the change -0.09). A prototype's change is larger in its own phase than in the other (|change| 2.21 against 1.63 for
objects, 3.11 against 1.35 for goals), so the phase gate removes a change that would otherwise also act out of phase.

**The flow loss: goals lower it, objects do not on held-out demonstrations.** The frozen policy's flow-matching loss on own-phase
frames, off -> on: goals 0.095 -> 0.082 (lower for 14 of 21), objects 0.055 -> 0.060 (lower for 9 of 19), instances
0.060 -> 0.086 (lower for 1 of 8). By flow time, the prototypes correct the coarse part of the field and damage the fine part.
The velocity error to the target falls at t = 0.9 for objects (2.69 -> 2.43) and goals (3.83 -> 2.99) and rises at t = 0.1
for objects (5.89 -> 7.28), goals (6.12 -> 7.56) and instances (6.66 -> 10.80). The change relative to the field also grows as
t falls (objects 0.25 at t = 0.9, 0.63 at t = 0.1). A rank-4 prototype fitted on one demonstration per task learns where to
send the hand, which shows in the early, high-noise steps of sampling. It also memorises the fine detail of its training chunks,
which does not carry over to a new demonstration of the same task. The scale sweep agrees (Osc1, Sch2, Gscplate, XG1;
`scale_sweep`). The own-phase flow loss is lowest at weight 0.5 to 0.75 for all four (Osc1 0.049 at 0.75 against 0.066 at 1).
The chunk change grows monotonically with the weight (Osc1 0.34, 0.69, 1.38, 2.84 at 0.25, 0.5, 1, 2). The goal prototypes'
alignment with the release point peaks at weight 1.25 to 1.5 (Gscplate 0.28 at 0, 0.53 at 1, 0.63 at 1.25) and collapses at
3 (0.11). A negative weight does not mirror the change (|change| 0.69 at -1 against 1.38 at +1 for Osc1). The prototypes act
as directions in a nonlinear field, effective in a band around their fitted scale.

**Subspaces.** Mean over the action expert's 163 adapted Linears of the output-subspace overlap (mean cos^2 of the principal
angles between the B column spaces), against chance (rank / dimension = 0.0050) (`subspaces`):

| pairs | n | overlap | / chance | Frobenius cosine of B A |
|---|---|---|---|---|
| instance and its own type | 8 | 0.120 | 23.9x | 0.130 |
| object type and object type | 171 | 0.075 | 14.9x | 0.058 |
| instance and sibling instance | 9 | 0.075 | 14.9x | 0.064 |
| goal and goal | 210 | 0.074 | 14.6x | 0.073 |
| instance and another object | 163 | 0.067 | 13.4x | 0.048 |
| object and goal | 567 | 0.055 | 10.9x | 0.026 |

Every prototype shares a common low-rank region of the expert, 11 to 15 times chance. This is the subspace that LoRA on this
expert reaches. Above that floor, the structure follows the concept hierarchy. An instance child overlaps its own type most
(24x, cosine 0.13), although it was fitted independently and not initialised from the type. Object and goal prototypes overlap
least (11x, cosine 0.026). Sibling instances do not overlap more than unrelated object types. So the hierarchy shows up in
weight space as a child sharing directions with its parent, not as siblings sharing a type subspace. All three kinds put their
update mainly in the expert's middle and late layers (layers 5 to 15 hold 81% to 84% of ||BA||^2) and about 80% in the MLPs
(attention share 0.20 objects, 0.19 instances, 0.21 goals). They differ in direction, not in location.

Figures: `figures/robot/h1_donate.png`, `h1_subspaces.png`, `h1_scale.png`.

## 3 What the operators do (h1_tli; four policies, Li's 20 OOD tasks and source pairs)

Offline, on frames of the object source task's 26th demonstration (the latents were identified on the first 20), prompted with
the OOD instruction, 2 noise draws. "Transfer" is the fraction of the distance from the instruction-alone chunk to the object
source's own chunk that the shift closes (1 means the source's behaviour is recalled).

| policy | Li's TLI online (zs) | shift / residual | effect of the shift / gap | transfer |
|---|---|---|---|---|
| pi0 | 326 / 400 (38) | 0.28 | 0.82 | +0.23 |
| pi0.5 | 259 / 400 (155) | 0.23 | 1.83 | -1.10 |
| X-VLA | 142 / 400 (80) | 1.01 | 1.29 | -0.46 |
| SmolVLA | 0 / 400 (5) | 1.14 | 0.76 | +0.07 |

The online counts are copied from ledger g13_tli_sources (pi0, pi0.5; set d) and g18_four_policies (SmolVLA, X-VLA; set e)
into the h1_tli row.

**Where TLI acts.** On pi0, the only policy where the shift moves the chunk toward the source, the early and middle thirds of
the layers carry it: shifting one band alone gives transfer 0.30 (early), 0.26 (middle) and 0.09 (late), and layer 6 alone
gives 0.25. Most of the effect sits on one instruction token: token 7 gives 0.36 and token 4 gives 0.15, against at most 0.06
for the others. Which words these positions hold was not checked. On
SmolVLA and X-VLA the shift is as large as the residual it is added to (|T1 - T2| / |h| = 1.14 and 1.01 at every layer, against
0.28 and 0.23 on pi0 and pi0.5). The added difference then replaces the instruction's representation instead of nudging it. On
SmolVLA it moves the chunk far (0.76 of the gap) but not toward the source (transfer 0.07), which is consistent with TLI's 0 of
400 there. Across policies the ratio separates the two that fail or half-fail (about 1) from the two where TLI works (0.23 to
0.28). It does not explain pi0.5. There the offline shift at alpha = 0 moves the chunk away from the source (transfer -1.10),
yet TLI gains 104 successes online. Offline transfer on the source's scene is the wrong proxy for pi0.5, where the policy's
reading of the OOD scene seems to matter more than recalling the source. Within a policy, the per-task offline measures predict
little: Spearman of transfer against per-task TLI success is +0.28 (p = 0.24) on pi0 and -0.54 (p = 0.014) on pi0.5, and the
shift/residual ratio against success is +0.51 (p = 0.021) on X-VLA.

**The grasp switch selects the conditional field that reproduces each phase's trained motion.** On all four policies, on the
object source's approach frames, the object source's instruction gives a chunk closer to the demonstration's than the OOD
instruction does: 5.1 against 9.7 (pi0), 0.69 against 0.99 (pi0.5), 2.1 against 10.6 (SmolVLA) and 0.62 against 1.32 (X-VLA).
On the goal source's carry frames, the goal source's instruction does the same: 2.7 against 5.9, 0.90 against 1.07, 1.47 against
7.76, and 1.06 against 1.34. The wrong source is worst in both phases. The switch therefore replaces a field that does not know
the phase's motion with one that does. TLI's shift does not get as close (approach 7.5 on pi0, 8.9 on SmolVLA). On pi0, the
field change at the switch (goal source against object source on carry frames) is 87% position and 11% gripper. On the other
policies it is spread over position (0.34 to 0.54), orientation (0.34 to 0.37) and gripper (0.12 to 0.29). This is why the
switch works on every architecture and TLI does not: the switch needs only the policy's own conditional field under a held
instruction, while TLI needs the instruction's residual to be linear enough for a fixed difference to stand in for a different
task.

Figures: `figures/robot/h1_tli_layers.png` and `figures/robot/h1_switch.png`.

## 4 What was not done
- Swann et al.'s reusable-versus-memorised score (arXiv 2603.19183) was not computed: the procedure is unverified here. The
  scale sweep and the held-out-demonstration loss stand in for it, and they show memorised fine detail (the loss rises at
  t = 0.1 and is lowest below weight 1).
- Direction toward the named object is measured against the demonstration's grasp point, not against every object's pose. The
  LeRobot LIBERO-90 frames carry no object poses. Which bowl an instance child picks online is in i7_anatomy.
- The TLI and switch analyses use the source tasks' training scenes, because Li's OOD tasks have no demonstrations. The
  online effect on the OOD scene is not reproduced offline (pi0.5 above).
- The likelihood of an OOD task's concepts is the held concept's loss on its training frames (ledger i6_novelty). The
  likelihood in the OOD scene was not measured.
