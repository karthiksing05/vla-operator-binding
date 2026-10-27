"""Pre-registration of plan G's read G4 (ledger m3_prereg tag g4_ood), written before its fits (other than a five-step smoke) and
any scored episode (experiments/g4_ood.py).  Li's LIBERO-Goal-OOD and LIBERO-Spatial-OOD (arXiv 2505.03500), 20 tasks; the
library's concepts fitted on pi0.5's own training demonstrations (5 per task, rank 4), matched to each test instruction by
experiments/g4_ood.py's rule (exact, else content-word Jaccard >= 0.5); acceptance on start states set a (10 per task, sampler
seeds 300000+), test on set b (20 per task, seeds 400000+), noise seed 13, Li's modified LIBERO, 300 steps."""
import json
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/vla-operator-binding")
from lifelong_ood import ledger
pl = json.load(open("/coc/flash10/ksingara3/vla-operator-binding/results/robot/g4_plan.json"))
print(ledger.record(
    "m3_prereg", bench="libero_ood", tag="g4_ood", seed=13,
    metrics={"plan": pl,
             "hypotheses": {"H1": "over the 20 tasks, the kept arm > the frozen policy (exact sign test on discordant pairs, alpha 0.05)",
                            "descriptive": "per suite success rates of the kept arm, the composition alone and the policy; the "
                                           "published rates (pi0.5 35.5 / 46.6, pi0 + text latent interpolation 81.0 / 85.0, "
                                           "pi0.5 trained on synthesized compositions 88.2 / 95.6 on Spatial-OOD / Goal-OOD, "
                                           "arXiv 2607.00351 Table I) are context only: their start states differ from ours"},
             "context": "E3, G0 (recombination from other tasks only), G5b read; the matching rule was written from the 20 "
                        "instructions and the 40 training instructions before any episode; it maps 'bbq source' to 'bbq sauce'",
             "written_before": "any fit (other than a smoke) or scored episode of g4; a two-episode serving smoke on libero_goal_ood task 8 (seed 99) ran"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/g4_ood.py", command="pre-registration of plan G's read G4 (Li's OOD suites)"))
