"""Pre-registration of plan G's read G3 on pi0 (ledger m3_prereg tag g3_crossmodel_pi0), written before its fits and episodes
(experiments/g3_crossmodel.py).  E3's hierarchical factored library (the same concepts and demonstrations) refit on pi0
(openpi pi0_libero, PyTorch; rank 4), against pi0's per-task one-demo LoRA (rank 16) given the same acceptance test, on the stream's
30 tasks: acceptance on stored states 0-9 (10 episodes, seed 13), test on 20 fresh start states per task (set b, seed 13) -- E3's
states and seed."""
import json
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/vla-operator-binding")
from lifelong_ood import ledger
pl = json.load(open("/coc/flash10/ksingara3/vla-operator-binding/results/robot/g3_plan_pi0.json"))
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="g3_crossmodel_pi0", seed=13,
    metrics={"plan": pl, "backbone": "pi0_libero",
             "hypotheses": {"H1": "on pi0, the kept arm > zero-shot (exact sign test, alpha 0.05)",
                            "H2": "on pi0, non-inferiority to its per-task LoRA with acceptance (task-level bootstrap 95% lower "
                                  "bound above -0.05), E3's criterion",
                            "correction": "H1 alone in the Holm family; H2 a separate criterion"},
             "context": "E3 (pi0.5) and G3 on SmolVLA read (SmolVLA: 203 vs 20 zero-shot, per-task LoRA leads); a five-step pi0 fit and a one-episode pi0 serving smoke preceded this registration",
             "written_before": "any fit (other than the smoke) or episode of g3 on pi0"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g3_crossmodel_pi0.py", command="pre-registration of plan G's read G3 (pi0)"))
