# Probe OF: object-frame prototypes and internal anchors in a frozen flow VLA (written 2026-10-25, before any D1/D2/D3 number)

**User direction (2026-10-25).** Prototypes must come from the internals of an off-the-shelf post-trained flow VLA: no
composition-aware training, no new demos. Route 1: action-space prototypes may exist, but P2/P3 measured them in the wrong
coordinates. An approach to an object is invariant only in the object's frame. Addition (D3): a detector is acceptable, but the
preferred anchor is the model's own grounding. Ideally the object's identity and location come from the vision tower alone
("the mug" dissociated from "where the mug is").

Code: `experiments/of_frame.py`. Pre-registration: `experiments/prereg/pof_frame.py` (ledger `m3_prereg`, tag `pof_frame`).
Read rows: `of_read` tags `d1`, `d2`, `d3`. Payloads: `results/of/` (read.json and sanity.json are tracked; the rest stays local).
Models: pi05_l90c@5000 (`l90c`) and the public pi05_libero (`libero`). Both are pi0.5 checkpoints: absolute OSC deltas, not the
pi0 state-relative actions (memory gotcha), so a chunk is integrated from the recorded hand position with the demo-calibrated
scale (pp2_dev.scale_from_demos, protos.store.interpret). Offline only: no rollout. Demo states are set in the simulator only to
read poses and segmentation (as pp_prep.py poses does). No demo is executed or replayed.

## 0. What P2's "transfer" measured (experiments/pp2_dev.py, transfer block)

- Data: the transfer set, the FIRST frames of 10 LIBERO init states of 12 tasks (6 suite + 6 LIBERO-90).
- Per task, across its 10 init states: Pearson r between the endpoint (x, y) of the named-object part-prompt mode (`obj:named`,
  spliced; chunk mean over the 4 shared starts, integrated with the demo scale) and the named object's position (x, y). x and y
  are averaged (corr2). The headline is the median over tasks, best over the 6 levels: 0.071 (pi05_libero), 0.026 (pi05_l90c).
  A displacement version (endpoint minus hand vs object minus hand) is also stored.
- So P2's target is the object's own position. Pearson r ignores a constant offset, so the statistic already asks whether the
  endpoint follows the object across init states, i.e. a translation-only object frame with the object centre as the aim. It did
  not test rotation, pooling across scenes, or a hand-relative aim point. Its world-frame analogue (a fixed world target) is
  undefined (a constant predictor). Within one task the init states move objects by a few cm, while a 10-step chunk moves the
  hand by a few cm from a fixed home pose.
- Sanity check (`--sanity`): P2's number recomputed by this code path from P2's stored files must match its ledger row to 1e-6.

## 1. D1: do object-frame prototypes exist?

**Observations.** Each model's TRAINING tasks with a raw demo match (pp_prep index: one demo per task;
results/protos/match/tr<k>.npz, results/robot/p0_demo/<tid>_match.npz). The task must have a pick-and-place first step whose
object is a movable instance. Candidate frames: every 3rd frame before the demo's first gripper-close command. Kept frames: the
hand is at least 5 cm (xy) from the named object (sim pose), at most 12 per task, evenly spaced. Object fillers = the named
object's BDDL type, kept if it appears in at least 2 scenes (factors' scene). The selection uses poses only, never the model.

**Behaviour at each frame** (pp_fields' machinery, protos.store.Live):
- `part`: mode of the `obj:named` part prompt (spliced, P2's source); `part_plain` / `part_tmpl` variants;
- `instr`: mode of the full training instruction; `instr_alt`: up to 2 other training instructions of the same scene that name
  the same object type (mean endpoint);
- `own`: the mean of 8 own samples under the instruction (seed 21); `null`: the null-prompt mode (a scene-default control).
- Modes: 4 shared starts (2 instruction drafts seed 11, 2 null drafts seed 12) at u in {0.1, 0.3, 0.55}, Adam ascent, 80
  iterations, lr 0.02, Tweedie x0; Laplace extent with 1 probe (unused).

**Frames.** A frame is an origin c and rotation R in the table plane (xy).
- world: c = 0, R = I;
- object: c = the named object's position, R = its yaw (from the body quaternion), ORACLE sim pose;
- object_trans: the object's position, no rotation (descriptive);
- control: the position and yaw of a different, non-named, non-goal movable object of the same scene, fixed per task by a hash of
  the task id (fixtures only if the scene has no other movable object).

**Statistic (P2's corr2, same definition, made frame-generic).** The model's displacement d = endpoint - hand (xy) is
predicted by a 3-parameter approach model d = lam (c - h) + R b. Here b is the aim point in the frame (times the gain) and h is
the hand. Every frame has the same 3 parameters. Per object type, the model is fitted by least squares on the type's frames from
the OTHER scenes (leave-scene-out) and predicts the held-out scene. Transfer T_frame = P2's corr2 (Pearson r of x and of y,
averaged) between d and its prediction, pooled over the type's frames, then averaged over types (types with at least 5 frames).
The P2-centre variant (d predicted by o - h, P2's displacement form, pooled) is reported beside it. As in P2, each frame takes its
best level; `own` has none.

**CIs.** Task-cluster bootstrap (resample tasks with replacement, B 2000, seed 0, predictions fixed), the same resample for every
frame.

**Hypotheses** (Holm over H1-H4 at alpha 0.05):
- H1: l90c, part.
- H2: l90c, instr.
- H3: libero, part.
- H4: libero, instr.

Each passes if T_object >= 0.5, T_object - T_world >= 0.3 and T_object > T_control (point estimates), and
p = max(P*(T_obj - T_world <= 0), P*(T_obj - T_control <= 0)) passes Holm.

**Readout.**
- Route 1 is ALIVE if H1 or H3 passes: part-prompt prototypes exist in the object frame.
- It is INSTRUCTION-LEVEL ONLY if only H2 or H4 passes.
- It is DEAD otherwise.

The other sources and the per-object table are descriptive.

## 2. D2: is the model's own grounding an internal anchor?

**Observations.** R2a's zs observations of pi05_l90c (results/r2a/obs, index; 1,305 pre-grasp and 372 post-grasp). ORACLE regions
come from R3b's segmentation (results/rs3/regions; >= 5% pixel coverage). The prompt is the task instruction. The object-word and
goal-word token rows come from R3b's parse (rs3_steer.role_rows, asserted equal to the policy's own tokens).

**Attention.** The post-softmax weights of the VLM's prefix pass, captured by wrapping transformers' eager Gemma attention (the
outputs are unchanged). The primary map is word -> agentview image tokens (256): the mean over heads, over the role's word rows
(object words pre-grasp, goal words post-grasp), and over layers {0, 1, 2}. L0b found that a word's effect stays at its own
positions through layer 1 and moves into the image tokens by layer 2.

**Localisation.**
- hit = the argmax token lies in the named object's region (pre-grasp) or the own target's region (post-grasp);
- chance = region size / 256;
- the mass share and the reverse direction (image queries -> word keys) are descriptive, and so is every layer 0-17.

**Groups.**
- pi05_l90c: zs episode types (success, wrong_object pre-grasp; success, wrong_goal post-grasp). The unit is the episode mean, with
  the two-level task-cluster bootstrap (R2a's ci_rate, B 10,000).
- pi05_libero, on the same images: its own 12 samples (seed 11) label each observation by the plurality target of R2a's readout:
  `hit` (the named object / own target) or `wrong` (another candidate). The unit is the observation. This proxy is used because
  pi05_libero has no zs episodes on these tasks and rollouts are not allowed.

**Gate (estimate gate, not in Holm).** Object role: hit(wrong) >= hit(success) - 0.10 and hit(wrong) >= 2 x chance(wrong). The
goal role is gated the same way. If the gate passes, the model localises the named object even when it acts on another, so the
misbinding is downstream of grounding.

## 3. D3: what / where in the vision tower (added by the user before any number; pre-registered with D1/D2 in one row)

**Features.**
- primary `sig`: the SigLIP (projected) agentview tokens. They do not depend on the prompt;
- `l0` and `l2`: the VLM's image tokens after layers 0 and 2 under the null prompt (descriptive).

**Region pooling.** The mean over the tokens of a body's segmentation region (at least 2 tokens). Bodies are the task's movable
objects, at every D1 training frame. Labels are BDDL object types. Prototypes are built from training-scene frames only (sim
segmentation).

**What test.**
- Leave-scene-out nearest-prototype classification (cosine) of the region features of types seen in at least 2 training scenes,
  against prototypes from the other scenes. Chance = 1 / number of prototype types.
- Gate: accuracy >= 0.8 and >= 3 x chance (task-cluster CI).
- Same-type vs different-type cosine across scenes (AUC), descriptive.

**Where test.**
- Is position recoverable separately? The region centroid is the patch index (exact by construction). A leave-scene-out ridge
  read (lambda 1) from the pooled feature to the centroid gives R^2 (descriptive).
- Stability: accuracy on frames whose region centroid lies far from the type's mean centroid (above the median shift) vs near.
  Stable if |far - near| <= 0.10.
- Dissociation = the what gate passes AND stable.

**Vision anchor.**
- On R2a's observations, the per-token cosine of the SigLIP tokens to the named object's type prototype (all training frames of
  that model); the argmax token is the anchor. hit = inside the oracle region. The candidate-level read (the candidate region with
  the highest mean similarity) is descriptive.
- Observations whose named type has no prototype are excluded (coverage reported).
- Gate (as D2): hit(wrong) >= hit(success) - 0.10 and >= 2 x chance, with the same groups as D2. It is compared with D2's
  attention anchor.

## 4. Cost (planned)

Index and select: CPU, minutes. Poses: 1 EGL job. D1 GPU: about 1,000 frames x about 12 s, 4 shards per model. Dev: 1,677
observations x about 3 s per model, 2 shards. Read: CPU.
