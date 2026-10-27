"""Pre-registration of plan L's probe L2 (ledger m3_prereg tag pl2_compose; docs/PLAN_LANG.md), written 2026-10-22 BEFORE any L2
flow loss or closed-loop episode existed (the code: lifelong_ood/lang/transplant.py, experiments/pl2_compose.py).

Question.  Can a failing new instruction be repaired with no gradient and NO PRACTICE, by composing the policy's own grounded
word states (language prototypes) with sources chosen on the one demo alone?  This answers the design review's practice-budget
threat (docs/DESIGN_REVIEW.md T4: the practice-chosen operators spend 340 rewarded episodes per task) and, with the controls,
part of T8 (position / trajectory selection).

Demos.  Li's 20 OOD tasks have none, so two are recorded per task on a NEW start-state set (fresh_ood_pl2d, LIBERO's sampler at
BASE 3300000, 8 per task) from the OTHER policy's strongest earlier arm on set d (G13: TLI with Li's pairs, the practice-chosen
grasp switch, or pi0.5 alone): pi0.5's demos from pi0 and pi0's from pi0.5; where the other policy succeeds fewer than twice, the
same policy's own arm (flagged "self").  Demo 0 = the first success, demo 1 = the second.

Operators (each on the instruction's object and goal phrases; candidates = training sentences that contain the phrase or its head
noun with identical token ids):
  word     the bind.py delta splice generalised: S_src - S_new|src at the phrase's tokens, layers in a span;
  tmpl     the new instruction re-expressed in a training template ("put the {O} {G}" / "pick up the {O} and place it {G}"), then word;
  tli_sel  Li's TLI with the pair (t1 holds the object phrase, t2 the goal phrase; single-step training sentences) and lambda in
           {12, 24, 36} chosen on demo 0;
  role     word with the object edit before the grasp and the goal edit while carrying (online: held_ops._Grasp).
Selection on demo 0 only: the demo's flow-matching loss (real action dims, 36 frames, 4 fixed draws per frame shared by all
candidates); coordinate order: object source, goal source given it, then the layer span in {1-17, 1-6, 7-12, 13-17}; tmpl
first picks the template by the loss with word's sources.  l2sel (THE TESTED ARM) = per task, the operator with the lowest
demo-0 loss (always composes; no fall-back to zero-shot).  No practice episode and no reward are used anywhere.

Offline (descriptive, reported before the closed loop is read): held-out demo-1 flow loss of every arm against zs (per-task
sign counts, geometric-mean ratio), on Li's 20 tasks (pi0.5, pi0) and 16 LIBERO-90 tasks (own IPEC demos 0 / 1).

Closed loop.  Dev tasks = the first 8 of Li's 20 by sha1('pl2:' + key): goal:1, goal:2, goal:4, goal:5, goal:6, goal:7, spatial:6,
spatial:7; the other 12 are kept for a confirm read with these rules frozen.  Start states: fresh_ood_pl2 (BASE 3400000, 20 per
task, never used), noise seed 13, paired across arms; launch_pack --bind track (first grasped object).
Arms per policy (pi0.5, pi0): zs, tli_li (Li's pairs and lambda), word, tmpl, tli_sel, role (l2sel is one of them per task),
rand, pos.
  H1  l2sel > zs: exact sign test on discordant episode pairs, alpha 0.05, per policy.
  H2  l2sel non-inferior to tli_li: task-level bootstrap 95% lower bound of the success difference above -0.05, per policy.
  Controls (descriptive, on l2sel's form: same operator, prompt, span and schedule; for TLI the same lambda):
    rand  a seeded random single-step training sentence sharing no content word with the phrase, its own role phrase
          right-aligned onto the new phrase's tokens;
    pos   the T8 control: the single-step training sentence with no content-word overlap whose training demos grasp (object role)
          / release (goal role) nearest to where demo 0 grasps / releases.  If pos matches l2sel, the gain is position /
          trajectory selection, not word grounding.
  Also descriptive: each operator against zs; part correctness g = P(first grasp takes the named object), p = P(success | named
  first grasp); the per-repair source sentence, layer span and the action-mode change (endpoint, nearest object, gripper timing;
  lifelong_ood/protos/store.py interpret) at demo-1 frames."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/vla-operator-binding")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_ood", tag="pl2_compose", seed=13,
    metrics={"backbones": ["pi05_libero", "pi0_libero"],
             "demos": "2 recorded successes per task on fresh_ood_pl2d (BASE 3300000) by the other policy's strongest set-d arm (G13); "
                      "own-policy fallback flagged; demo 0 selects, demo 1 scores",
             "operators": ["word", "tmpl", "tli_sel", "role"], "tested_arm": "l2sel = per task the operator with the lowest demo-0 flow loss",
             "selection": "demo-0 flow loss only (36 frames, 4 fixed draws); object source, goal source, layer span {1-17,1-6,7-12,13-17}; "
                          "TLI pair at lambda 24 then lambda {12,24,36}; no practice, no reward",
             "dev_tasks": ["goal:1", "goal:2", "goal:4", "goal:5", "goal:6", "goal:7", "spatial:6", "spatial:7"],
             "confirm_tasks": ["goal:0", "goal:3", "goal:8", "goal:9", "spatial:0", "spatial:1", "spatial:2", "spatial:3", "spatial:4",
                               "spatial:5", "spatial:8", "spatial:9"],
             "states": "fresh_ood_pl2 (BASE 3400000, 20 per task, never used), noise seed 13, paired",
             "arms": ["zs", "tli_li", "word", "tmpl", "tli_sel", "role", "rand", "pos"],
             "hypotheses": {
                 "H1": "per policy, l2sel > zs on the 8 dev tasks x 20 episodes (exact sign test on discordant pairs, alpha 0.05)",
                 "H2": "per policy, l2sel non-inferior to TLI with Li's hand pairs and lambda (task-level bootstrap 95% lower bound of the "
                       "difference above -0.05)",
                 "controls": "rand (seeded random no-overlap sentence) and pos (T8: no-overlap sentence whose training grasp / release is "
                             "nearest demo 0's), each on l2sel's operator form; descriptive comparisons l2sel vs rand / pos",
                 "descriptive": "offline demo-1 flow loss per arm vs zs (Li's 20 x 2 policies, 16 LIBERO-90 tasks); each operator vs zs in "
                                "closed loop; part correctness g and p|g; per-repair source, span and action-mode change"},
             "written_before": "any L2 flow loss and any L2 closed-loop episode"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/pl2_compose.py",
    command="pre-registration of plan L's probe L2 (demo-selected language-prototype composition, no practice)"))
