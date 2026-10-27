"""Pre-registration of plan G's read G3 on SmolVLA (ledger m3_prereg tag g3_crossmodel), written before its fits and episodes
(experiments/g3_crossmodel.py).  E3's hierarchical factored library (the same concepts and demonstrations) refit on SmolVLA
(lerobot/smolvla_libero; rank 4), against SmolVLA's per-task one-demo LoRA (rank 16) given the same acceptance test, on the stream's
30 tasks: acceptance on stored states 0-9 (10 episodes, seed 13), test on 20 fresh start states per task (set b, seed 13) -- E3's
states and seed."""
import json
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/vla-operator-binding")
from lifelong_ood import ledger
pl = json.load(open("/coc/flash10/ksingara3/vla-operator-binding/results/robot/g3_plan.json"))
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="g3_crossmodel", seed=13,
    metrics={"plan": pl, "backbone": "smolvla",
             "hypotheses": {"H1": "on SmolVLA, the kept arm > zero-shot (exact sign test, alpha 0.05)",
                            "H2": "on SmolVLA, non-inferiority to its per-task LoRA with acceptance (task-level bootstrap 95% lower "
                                  "bound above -0.05), E3's criterion",
                            "correction": "H1 alone in the Holm family; H2 a separate criterion"},
             "context": "E3 (pi0.5) read; a five-step SmolVLA fit and a one-episode SmolVLA serving smoke preceded this registration",
             "written_before": "any fit (other than the smoke) or episode of g3"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g3_crossmodel.py", command="pre-registration of plan G's read G3 (SmolVLA)"))
