"""Pre-registration of read G21 (plan H3 baselines; ledger m3_prereg tag g21_operators), written after the 4-episode smoke of each
new mode (pi0.5, libero_goal_ood task 1, start states set a) and before any other G21 episode (experiments/g21_operators.py).

Two more operators for held concepts, run with the same source pairs as the grasp switch and TLI so the operator is the only
thing that varies (lifelong_ood/policy/held_ops.py):
  GPC     Cao et al. (arXiv 2510.01068, ICLR 2026): the convex mix w v(.|instruction a) + (1 - w) v(.|instruction g) of the frozen
          policy's fields at every Euler step; their test-time search over w is practice over w in {0.3, 0.5, 0.7}, 5 episodes
          each on start states set a (most successes; ties to 0.5, then the lower w).
  inject  Grant et al. (arXiv 2603.19233): action-expert activations of the source training tasks, mirrored on TLI: the mean
          residual at every expert layer input per denoising step and token over frames 0-119 of the source's first 20
          demonstrations; the difference (M_a - M_g) added at every layer >= 1 before the grasp, (M_g - M_a) from the grasp to the
          release.  Deviation: additive (TLI's form) in place of their replacement write-back, because a frame-mean replacement
          erases the noisy chunk from the expert and passes the test noise into the action.
Pairs: G10's practice-chosen pair (sel) and Li et al.'s pair (li).  Test on set e (BASE 800000, 20 per task, noise seed 13),
paired with G15's set-e files.  Per policy (pi0.5, pi0)."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_ood", tag="g21_operators", seed=13,
    metrics={"backbones": ["pi05_libero", "pi0_libero"], "states": "practice fresh_ood_a (5 per w, 10 for the phase check); "
             "test fresh_ood_e (BASE 800000, 20 per task)", "w_grid": [0.3, 0.5, 0.7],
             "hypotheses": {
                 "H1_gpc": "per policy, GPC with the practice-chosen pair and practice-chosen w is non-inferior to G15's practice-chosen "
                           "operator on set e (task-level bootstrap 95% lower bound of the difference above -0.05)",
                 "H1_inject": "per policy, activation injection with the practice-chosen pair is non-inferior to G15's practice-chosen "
                              "operator on set e (same test)",
                 "H2_gpc": "per policy, GPC with the practice-chosen pair > the policy alone on set e (exact sign test, alpha 0.05)",
                 "H2_inject": "per policy, activation injection with the practice-chosen pair > the policy alone on set e (same test)",
                 "descriptive": "each operator against the grasp switch and TLI with the same pairs (chosen pair: G15's psel and TLI "
                                "lambda 24 with G10's pair; Li's pair: the grasp switch pfield@li and Li's TLI); Li's-pair arms against "
                                "the policy alone; chosen w per task; per task",
                 "check": "GPC with w = 1 before the grasp and 0 from it equals the grasp switch: 10 episodes per task on set a, "
                          "per-episode agreement with G10's acceptance episodes of the same pair (equal up to the batched VLM pass)"},
             "smoke": "4 episodes per mode on pi0.5 libero_goal_ood task 1, set a (ledger-free files g21_smoke_*)",
             "written_before": "any non-smoke G21 episode"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g21_operators.py",
    command="pre-registration of read G21 (plan H3: GPC velocity composition and activation injection as operators for held concepts)"))
