"""Pre-registration of plan G's read G20 (ledger m3_prereg tag g20_ablation), written before any G20 episode
(experiments/g20_ablation.py).  One ablation table per side on pi0.5, every row paired with the full system on the same start
states, all descriptive (exact sign test and task-level bootstrap per row, no correction claimed):

LIBERO-90, one demonstration per new concept, G8's 66 tasks, start states set d (G8's episodes reused):
  full = task leaves (G8); minus leaves = library kept by practice; minus practice = the composition always on; minus
  composition = per-task LoRA with acceptance; and on the composition itself (always on, against the full composition always on):
  object unit only (policy after the grasp), goal unit only (policy before the grasp), the type prototype in place of the instance
  child (instance tasks), both units on for the whole episode (no switch at the grasp).
Li's OOD suites, no demonstration, start states set e (G15's episodes reused):
  full = practice chooses sources and operator (G15); minus operator choice = grasp switch only / TLI only; minus source practice =
  word-matched sources at the grasp; minus the grasp trigger = practice-chosen sources switched at Li's fixed step; the object
  source only (test instruction after the grasp) and the goal source only (test instruction before the grasp); the policy alone."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero", tag="g20_ablation", seed=13,
    metrics={"backbone": "pi05_libero", "states": {"libero_90": "fresh_states_d", "ood": "fresh_ood_e"},
             "hypotheses": {"descriptive": "each component's contribution as the paired difference between the full system (or the full "
                                           "composition) and the system without it"},
             "written_before": "any G20 episode"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g20_ablation.py", command="pre-registration of plan G's read G20 (ablation on pi0.5)"))
