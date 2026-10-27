"""Pre-registration of plan L's probe L1 (ledger m3_prereg tag pl1_predict; docs/PLAN_LANG.md), written 2026-10-22 BEFORE the L0a
lexicon audit (experiments/pl0_audit.py) had computed any grounding score and before experiments/pl1_predict.py existed.

Question.  Does the lexicon audit's word grounding in the test scene predict which of Li's 20 OOD instructions the policy fails
zero-shot?

Data.  Per (policy, task): zero-shot success of the policy alone (arm "zs") pooled over the existing start-state sets c, d, e, f
(20 episodes each, noise seed 13: experiments g4b_main / g4c_main, g13_main, g15_main, g17_base), 80 episodes per task, for
pi0_libero and pi05_libero: 40 (policy, task) units.  Failure = pooled success rate below 0.50.

Predictor (fixed now).  For each content word w of the OOD instruction (the object phrase, the goal phrase, the verb and, when the
instruction has one, the spatial relation of the object phrase; parse: lifelong_ood/lang/audit.py parse(), built on
lifelong_ood/policy/primitives.step_key), its grounding in the task's own OOD scene
    G(w) = mean over 10 observations (first frames of start-state set a, fresh_ood_a) and over same-category alternatives w' of
           mean_t [ mean_points ||eps(instr[w -> w']) - eps(instr)|| / mean_points ||eps(instr) - eps(null)|| ]
with flow times t in (0.2, 0.4, 0.6, 0.8), shared noise and shared draft chunks.  PRIMARY predictor: min_w G(w) (lower predicts
failure).  SECONDARY: mean_w G(w).

Test.  PRIMARY: AUC of -min_w G(w) for failure over the 40 pooled units.  GO if AUC >= 0.75 (the audit is then a free failure
detector, run before any rollout).  Descriptive: the same per policy; the secondary predictor; Spearman of min / mean G with the
pooled success rate (two-sided p); bootstrap 95% interval of the AUC over tasks.
Known before computing: pi0 alone succeeds on 149 of 1,600 pooled episodes and pi0.5 on 596, so pi0 contributes mostly failures and the
pooled AUC partly separates the two policies; the per-policy AUCs are therefore reported next to it (descriptive).
LIBERO-90-C dev zero-shot episodes are not part of this test (noted for later)."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/vla-operator-binding")
from lifelong_ood import ledger
print(ledger.record(
    "m3_prereg", bench="libero_ood", tag="pl1_predict", seed=13,
    metrics={"backbones": ["pi0_libero", "pi05_libero"],
             "outcome": "zero-shot success of the policy alone pooled over start-state sets c, d, e, f (80 episodes per task); failure = rate < 0.50",
             "predictor_primary": "min over the instruction's content words (object, goal, verb, relation if present) of the word grounding G(w) in the "
                                  "task's own OOD scene (10 first frames of fresh_ood_a; G = mean over alternatives of the swap's ||delta eps|| / "
                                  "||eps(instr) - eps(null)||, t in 0.2/0.4/0.6/0.8, shared noise); lower predicts failure",
             "predictor_secondary": "mean over the same words of G(w)",
             "hypotheses": {"GO": "AUC of -min G for failure over the 40 pooled (policy, task) units >= 0.75",
                            "descriptive": "per policy; secondary predictor; Spearman with the pooled success rate; bootstrap 95% CI over tasks"},
             "written_before": "any L0a grounding score and experiments/pl1_predict.py"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/pl1_predict.py", command="pre-registration of plan L's probe L1 (grounding predicts zero-shot failure)"))
