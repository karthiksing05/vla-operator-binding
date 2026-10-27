"""Pre-registration of plan G's read G2 (ledger m3_prereg tag g2_baselines), written before any of its episodes
(experiments/g2_baselines.py; docs/PLAN_GENERALIZE.md).  On G0's eligible held-out tasks that have an object donor and a goal donor
outside their fold: task arithmetic (the two donors' per-task LoRAs both on at weight 1) and LoRA averaging (both at weight 1/2),
always on.  Acceptance on stored states 0-9 (seed 15), test on G0's fresh start states (seed 15), paired with G0's arms."""
import json
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/vla-operator-binding")
from lifelong_ood import ledger
pl = json.load(open("/coc/flash10/ksingara3/vla-operator-binding/results/robot/g2_plan.json"))
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="g2_baselines", seed=15,
    metrics={"plan": pl, "tasks": sorted(int(t) for t, v in pl.items() if v["ok"]),
             "hypotheses": {"H1": "the composed concepts (G0's gfac) > task arithmetic (lorasum), exact sign test",
                            "H2": "the composed concepts > LoRA averaging (loraavg), exact sign test",
                            "correction": "Holm over H1 and H2; each baseline against zero-shot and the kept arms reported as "
                                          "descriptive"},
             "context": "a two-episode smoke run of the serving mode (experiment g2_smoke) preceded this registration",
             "written_before": "any episode of g2_accept or g2_main"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g2_baselines.py", command="pre-registration of plan G's read G2"))
