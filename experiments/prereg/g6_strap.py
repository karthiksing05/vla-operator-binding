"""Pre-registration of plan G's read G6 (ledger m3_prereg tag g6_strap), written before any STRAP fit or episode
(experiments/g6_strap.py).  STRAP (arXiv 2412.15182, its own segmentation, S-DTW and K = 100) on E3's 29 tasks with a
demonstration; prior = pi0.5's 40 training tasks' 1,693 demonstrations + the one demonstration of every other LIBERO-90 task
(the library's own data); a rank-16 LoRA, 2,000 steps; the acceptance test on stored states 0-9 (10 episodes, seed 13) as for the
per-task LoRA; test on E3's fresh start states (set b, 20 per task, seed 13), paired with E3's recorded arms."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="g6_strap", seed=13,
    metrics={"tasks": [9, 10, 12, 13, 14, 15, 25, 30, 31, 36, 41, 43, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 71, 72],
             "hypotheses": {"H1": "E3's kept library is non-inferior to STRAP with acceptance (task-level bootstrap 95% lower bound "
                                  "of library minus STRAP+acc above -0.05), E3's criterion",
                            "descriptive": "STRAP alone vs zero-shot; STRAP+acc vs the per-task LoRA+acc; library vs STRAP alone"},
             "context": "E3 read (kept library 480 vs 494 for LoRA+acc); STRAP on the earlier 21-task set was a baseline of the "
                        "token era (ledger b_baselines tags strap, strap2)",
             "written_before": "any STRAP fit or episode of g6"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g6_strap.py", command="pre-registration of plan G's read G6 (STRAP)"))
