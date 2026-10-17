"""Pre-registration of plan G's read G3 on X-VLA (ledger m3_prereg tag g3_crossmodel_xvla), written before its fits and episodes
(experiments/g3_crossmodel.py).  E3's hierarchical factored library (the same concepts and demonstrations) refit on X-VLA
(lerobot/xvla-libero, trained on the same 40 LIBERO tasks; rank 4), against X-VLA's per-task one-demo LoRA (rank 16) given the same acceptance test, on the stream's
30 tasks: acceptance on stored states 0-9 (10 episodes, seed 13), test on 20 fresh start states per task (set b, seed 13) -- E3's
states and seed."""
import json
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
pl = json.load(open("/coc/flash10/ksingara3/lifelong-ood-learner/results/robot/g3_plan_xvla.json"))
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="g3_crossmodel_xvla", seed=13,
    metrics={"plan": pl, "backbone": "xvla",
             "hypotheses": {"H1": "on X-VLA, the kept arm > zero-shot (exact sign test, alpha 0.05)",
                            "H2": "on X-VLA, non-inferiority to its per-task LoRA with acceptance (task-level bootstrap 95% lower "
                                  "bound above -0.05), E3's criterion",
                            "correction": "H1 alone in the Holm family; H2 a separate criterion"},
             "context": "E3 (pi0.5) and G3 on SmolVLA read (SmolVLA: 203 vs 20 zero-shot, per-task LoRA leads); a 50-step X-VLA prototype fit and a two-episode X-VLA serving smoke (zero-shot and composition, task 13, seed 99) preceded this registration; pi0 read: 324 vs 85",
             "written_before": "any fit (other than the smoke) or episode of g3 on X-VLA"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g3_crossmodel_xvla.py", command="pre-registration of plan G's read G3 (X-VLA)"))
