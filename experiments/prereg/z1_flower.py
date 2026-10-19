"""Pre-registration of plan Z's read Z1 (ledger m3_prereg tag z1_flower), written before any Z1 episode.  FLOWER (mbreuss/
flower_libero_goal: LIBERO-Goal's 10 tasks only; lifelong_ood/policy/backbone_flower.py) with NO fine-tuning: on the 6 OOD tasks
whose object and goal are both held inside its suite (goal:4-9), held concepts are pointers to its own training instructions,
switched at the grasp (our rule's sources, all in task indices 10-19); nothing is fitted.  Test on set e, 20 per task, noise seed 13
(experiments/z1_flower.py)."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_ood", tag="z1_flower", seed=13,
    metrics={"backbones": ["flower"], "tasks": [f"goal:{i}" for i in range(4, 10)],
             "hypotheses": {"H1": "pointers switched at the grasp (our rule's sources, zero fitting) > FLOWER alone under the new "
                                  "instruction on set e (exact sign test, alpha 0.05)",
                            "H2": "ours non-inferior to Li's hand-chosen pairs at the grasp (task-level bootstrap 95% lower bound above -0.05)",
                            "descriptive": "per-task counts; the 14 other OOD tasks are outside FLOWER-goal's library and are not run"},
             "written_before": "any Z1 episode"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/z1_flower.py", command="pre-registration of plan Z's read Z1 (FLOWER, no fine-tuning)"))
