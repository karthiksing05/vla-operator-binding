"""Pre-registration of plan H4's read G22 (ledger m3_prereg tag g22_libero_pro), written before any non-smoke G22 episode
(experiments/g22_libero_pro.py; the split mapping in docs/LIBERO_PRO.md).

Benchmark: LIBERO-PRO (Zhou et al., arXiv 2510.03827; code eafdb80, data c86fc3b), its TASK split, the one split whose new tasks
recombine held concepts in a known scene (Position, Object, Semantic, Environment and the seven robustness cases change what the
policy sees or where things are under the instruction that taught the task; they are not compositional and are not run).
Suites libero_goal_task, libero_spatial_task, libero_object_task, libero_10_task (40 tasks), in LIBERO-PRO's own LIBERO package.
The policy receives the BDDL's new instruction, lower-cased (LIBERO-PRO keeps the old file names, so LIBERO's Task.language is the
old instruction; its paper defines the perturbation as the old instruction with a new success predicate).

System: iteration 12, applied unchanged.  Per eligible task, practice chooses the object source (stage 1, the word rule's goal source
fixed) and then the goal source (stage 2), 5 episodes per candidate on fresh set a, under the grasp switch (G10) and under TLI
(G14, lambda 24); ties to the word rule's source, then the lowest index; then 10 practice episodes of each chosen pair on set a
choose the operator (ties to the switch).  Eligible = one step, verb put, a preposition, one In/On goal atom (27 tasks: Goal 6,
Spatial 10, Object 10, LIBERO-10 1).  Pool: G10's 16 instructions for the Goal and Spatial splits (unchanged); G10's rule over
LIBERO-Object (10) and LIBERO-10 (5) for those splits, because G10's pool holds none of their scenes (the one change, stated).
Where the word rule finds no concept, the fixed partner of stage 1 is the pool member sharing most content words.

Splits without held-concept structure for the system: 13 tasks (Goal 4: a drawer opened, two steps, a push, the stove turned
off; LIBERO-10 9: two objects, two steps, or a goal atom the instruction omits) run the policy alone only.

Arms and start states (noise seed 13, paired):
  fresh set t (seeds 900000+, 20 per task): the policy alone, the switch with its pair, TLI with its pair (the chosen arm is one)
  LIBERO-PRO's 50 init states per task: the policy alone, the policy under the OLD instruction (origprompt: the published
  protocol), and the chosen arm; LIBERO-PRO's published number for the same policy beside them.
Policies: pi0.5 first, then pi0 if time allows (same design; G22's read per policy)."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
from lifelong_ood.policy import factors as F
print(ledger.record(
    "m3_prereg", bench="libero_pro", tag="g22_libero_pro", seed=13,
    metrics={"backbones": ["pi05_libero", "pi0_libero"],
             "suites": ["libero_goal_task", "libero_spatial_task", "libero_object_task", "libero_10_task"],
             "splits_run": ["task"], "splits_not_run": {"position (_swap)": "layout shift of a held task, not compositional",
                                                         "object (_object)": "visual shift", "semantic (_lan)": "lexical shift",
                                                         "environment (_env)": "environment shift; not shipped",
                                                         "seven robustness cases": "visual or physical shift"},
             "eligible": 27, "policy_alone_only": 13,
             "hypotheses": {"H1": "per policy, the system's chosen arm > the policy alone on fresh set t over the 27 eligible tasks "
                                  "(exact sign test on discordant pairs, alpha 0.05)",
                            "H1b": "the same on LIBERO-PRO's 50 init states per eligible task (exact sign test, alpha 0.05)",
                            "descriptive": "per split (Goal, Spatial, Object, LIBERO-10): the system and the policy alone on both start-"
                                           "state sets; the policy under the old instruction against LIBERO-PRO's published Task "
                                           "number; operator counts; the 13 ineligible tasks' policy-alone rates; the switch and "
                                           "TLI arms each against the policy"},
             "parser": F.parser_hash(),
             "written_before": "any non-smoke G22 episode (smoke: 4 episodes per split, experiment g22_smoke, not counted)"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g22_libero_pro.py",
    command="pre-registration of plan H4's read G22 (the OOD system on LIBERO-PRO's task split)"))
