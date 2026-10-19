# Unsupervised segmentation and identities (plan Z, track U)

Exploratory, offline. Data: `experiments/zu_score.py`, which records pi0.5's flow loss at every 2nd frame of one demo per task,
under each of its 40 training instructions (4 shared noise draws per frame).
- LIBERO-90: tasks 0–89.
- LIBERO-10: tasks 0–9.
- LIBERO-Goal / Spatial: training tasks 10–19 and 30–39, with the demo's own instruction excluded.

Truth is the demo's sustained grasp and release, and it is used for scoring only.

## 1. Likelihood does not find the boundaries

At matched cut counts, the policy's likelihood places cuts no better than random.

| segmenter (72 pick-and-place demos; about 2 cuts per demo) | grasp within 10 frames | release within 10 frames |
|---|---|---|
| label switch (Viterbi over the best field, λ 4; 2.07 cuts per demo) | 0.36 | 0.14 |
| random cuts, same count | 0.28 | 0.24 |
| exact 2-cut partition of the loss signature over the held-out instructions | 0.24 | 0.10 |
| exact 2-cut partition of the prefix latent | 0.25 | 0.11 |
| exact 2-cut partition of the demo's arm actions (dims 0–5, a non-policy baseline) | 0.58 | 0.11 |
| random 2 cuts | 0.30 | 0.25 |

`experiments/zu_sweep.py`, `experiments/zu_kcut.py`; ledger `zu_read` tags `sweep` and `kcut`. The plan's kill gate was grasp recall ≥ 0.6. The label-switch segmenter
fails it, and so do the other likelihood-based segmenters.

**Targeted check.** On the 20 LIBERO-90 demos whose object and goal are both held, take the loss under the best object source
minus the loss under the best goal source. Its mean rises across the grasp in 0.65 of the demos (chance 0.5). A single best cut
lands within 10 frames of the grasp in 0.05 of them. In the 4 testable demos from familiar scenes (LIBERO-Goal), the same contrast
rises across the grasp in all 4: after the grasp the goal source's log-loss is lower by 0.89 to 2.19, and a single cut lands within 10 frames of the grasp in 2 of the 4 (`experiments/zu_identity.py`, ledger `zu_read` tag `identity`).

## 2. Likelihood does name the segments

This holds once the segments are cut at contact events. The test: among the 40 training instructions (own excluded), does the one
with the lowest mean flow loss over a segment hold that segment's concept? For the approach, that is the object concept; for the
carry, the goal concept.

| demos | segment | top-1 holds the concept | chance | median rank of a holder |
|---|---|---|---|---|
| familiar scenes (LIBERO-Goal / Spatial, n 7 / 15) | approach | 0.43 | 0.04 | 2 |
| | carry | 0.60 | 0.22 | 0 |
| LIBERO-10 (n 4) | carry | 0.75 | 0.24 | 0 |
| LIBERO-90 (n 44 / 35) | approach | 0.18 | 0.03 | 8.5 |
| | carry | 0.31 | 0.17 | 2 |

## Reading

The frozen policy's likelihood identifies WHICH held concept explains a stretch of behaviour, and it does so best in scenes it
knows. It does not say WHERE one unit ends: per frame, the flow loss is dominated by the observation and by the noise draw, not by
the instruction.

This is the interface result again (success ≈ g × p): composition happens at contact events. So the unsupervised recipe splits:
- **boundaries** come from the demonstration's own contact events (gripper close and open; task-agnostic, no grammar);
- **identities** come from the policy's likelihood, averaged per segment.

`lifelong_ood/policy/program.py` turns this into a pointer program: a list of training instructions, advanced online at the
policy's own gripper events. The grasp switch is the case where the grammar names the two segments.

In unfamiliar scenes (LIBERO-90), identification is weak (0.18 approach). There the policy barely reads its prompt (cf.
f1_binding: it runs the scene's habit), which bounds any prompt-level composition.

Next: pilot U3 (`experiments/z4_programs.py`). It runs programs on the 23 LIBERO-90 tasks outside the template.
