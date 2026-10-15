"""Pre-registration of plan G's reads G0 + G1 (ledger m3_prereg tag g0_recombine), written before any of their fits and episodes
(experiments/g0_recombine.py; docs/PLAN_GENERALIZE.md).  The 67 single-step pick-and-place LIBERO-90 tasks in 5 folds (seeded);
per fold, every concept its tasks need is refit with NO demonstration of the fold's tasks (rank 4, c0_fit --joint); a task is
eligible when its object (instance child if named and demonstrated outside the fold, else type) and its goal keep demonstrations
outside the fold.  Per fold, a monolithic multitask LoRA (rank 16, 4000 steps, every frame with its own instruction) on the pool
of demonstrations the concepts may draw on, minus the fold.  Arms on each eligible task: zs; gfac (the fold's composed concepts);
mono (the fold's monolithic LoRA); lora (per-task LoRA on the task's own demonstration: the upper reference, not a competitor).
Acceptance on stored states 0-9 (10 episodes, seed 15); test on 20 fresh start states per task (set b), seed 15."""
import json
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
pl = json.load(open("/coc/flash10/ksingara3/lifelong-ood-learner/results/robot/g0_plan.json"))
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="g0_recombine", seed=15,
    metrics={"plan": pl, "eligible": sorted(int(t) for t, v in pl["tasks"].items() if v["eligible"]),
             "hypotheses": {"H1": "RECOMBINATION: gfac (no acceptance) > zero-shot on the eligible tasks (exact sign test)",
                            "H2": "FACTORISATION: gfac > mono on the eligible tasks (exact sign test)",
                            "correction": "Holm over H1 and H2; H1b (kept gfac vs zs) and H2b (kept gfac vs kept mono, each with "
                                          "the acceptance test) reported as secondary",
                            "gate": "if H1 fails, the claim narrows to consolidated prototypes with one demonstration per task "
                                    "(docs/PLAN_GENERALIZE.md); if H2 fails, the claim that factorisation matters is dropped"},
             "context": "E0's leave-one-out arm (exploratory, before the data fix: 125 vs 117 on 25 tasks) and E3, E5 (running) were "
                        "seen; the folds, arms, steps and seeds are fixed here",
             "written_before": "any fit or episode of g0"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g0_recombine.py", command="pre-registration of plan G's reads G0 + G1"))
