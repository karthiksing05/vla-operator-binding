"""Pre-registration of plan Z / F1's confirmatory read (ledger m3_prereg tag f1_fast_fit), written after the lever read (ledger
f1_levers tag read: E5's 26 units, held-out flow loss only, no episode) and before any fast-fit episode.  Recipe: the frozen prefix
cached once (c0_fit --cache; equal losses, 39.4 -> 4.6 min per 1,000 steps), from scratch, 200 steps (held-out loss 47% below the
1,000-step fit on average, 32% in the worst unit).  experiments/zf_fast_fit.py."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/vla-operator-binding")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="f1_fast_fit", seed=13,
    metrics={"recipe": {"steps": 200, "init": "scratch", "cache": True},
             "hypotheses": {
                 "H1": "E3's 22 prototypes refit with the recipe; their composition on E3's 29 tasks with a per-task LoRA, set b, 20 per "
                       "task, noise seed 13, paired with E3's library episodes: task-level bootstrap 95% lower bound of (fast - E3) above -0.05",
                 "H2": "G8's leaves rule with RANK-4 leaves (E3: G9's rank-4 LoRAs; E5: new rank-4 leaves) against G8's rank-16 rule on the "
                       "66 tasks, set d, paired: lower bound above -0.05",
                 "descriptive": "fit minutes per unit (job wall); acceptance-kept arms; the 30th task"},
             "written_before": "any f1_main / f1_accept / f1_leaf episode"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/zf_fast_fit.py", command="pre-registration of the fast-fit confirmatory test"))
