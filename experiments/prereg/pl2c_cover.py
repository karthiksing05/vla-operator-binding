"""PRE-REGISTRATION of plan L probe L2c (docs/PLAN_LANG.md; code experiments/pl2c_cover.py), written 2026-10-23 AFTER the offline
picks (results/lang/pl2c/picks.json) and the demo-0 lambda scores (results/lang/pl2c/demo_lam_<policy>.json) and BEFORE any L2c
rollout (no pl2c_* episode file exists when this row is recorded; asserted below).

Question.  Can a SYMBOLIC, demo-free rule recover (or beat) Li's hand-chosen TLI pairs?  The rule is the framework's composition
step: parse the new instruction into role-filler parts, retrieve the training sentences that cover those parts in the same role,
combine the two by TLI.

Rule (experiments/pl2c_cover.py docstring; designed from principles before any per-pick success number was looked at):
  parse   (V, verb), (O, head content words), (L, landmark content words: the object phrase after its first relation word),
          (G, goal-phrase content words), by transplant.roles / primitives.step_key; STOP words out.
  pool    the 27 single-step pick-and-place training prompts of the 40 training tasks (transplant.template_of; pl2's TLI pool).
  cover   greedy coverage, a filler counts only in the SAME role; ties: total pairs covered, same frame, word Jaccard, lower index;
          2 sources.
  slots   t1 = the pick leaning to the object roles ((#O + #L) - #G covered), t2 = the other; ties: first pick is t1.
  lambda  auto_li = Li's lambda for the task (as tli_li uses it); auto_dl = argmin demo-0 flow loss over {24, 12, 36} (pl2's
          scorer, K 4, seed 1, demo 0 of results/lang/pl2/demos.json).
Offline agreement with Li's pairs (no success data): exact (t1, t2) on 9 / 20 tasks (goal:7, goal:8, goal:9, spatial:4..9);
t1 agrees on 10, t2 on 15.  The picks are identical for both policies (same training set).

Disclosure.  Dev / confirm tasks of pl2 are previously used tasks (their pl2 / G12 / G13 success counts exist and were reported to
the agent in aggregate: tli_li 169 / 240 pi0.5 confirm, 106 / 160 pi0 dev; demo-chosen TLI 143 / 40).  The rule reads no success
number; demo 0 enters only the secondary arm's lambda.  All arms are rolled on a NEW start-state set fresh_ood_pl2c (LIBERO's
sampler at BASE 5100000, 20 per task, asserted unused: no logged fresh-states job used a base within 1000 of it, and every state
is > 1e-6 from every state of every earlier fresh_ood_* set; results/lang/pl2c/fresh_check.json), noise seed 13, paired.
Where auto pair and lambda equal Li's (auto_li on the 9 exact tasks), the arm string is identical and ONE fresh rollout serves both.

Arms per policy (pi05_libero, pi0_libero) x 20 tasks (dev + confirm pooled) x 20 episodes: zs, tli_li, auto_li, auto_dl.
  H1  auto_li >= tli_li - 0.05: task-cluster bootstrap 95% lower bound of the success difference >= -0.05 (e1_read.compare, the
      function pl2 used), per policy.
  H2  auto_li > zs: exact sign test on discordant episode pairs, alpha 0.05, and more successes, per policy.
  Secondary (same tests, reported, not a claim): auto_dl.  Descriptive: tli_li vs zs; auto_li vs tli_li on the 11 differing
  tasks; per-task picks and why they differ."""
import glob
import hashlib
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/vla-operator-binding"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

OUT = os.path.join(ROOT, "results", "lang", "pl2c")
assert not glob.glob(os.path.join(ROOT, "results", "robot", "pl2c_*.json")), "an L2c rollout already exists"
assert os.path.exists(os.path.join(OUT, "fresh_check.json")), "start states not checked"
files = {n: os.path.join(OUT, n) for n in ("picks.json", "demo_lam_pi05.json", "demo_lam_pi0.json", "fresh_check.json")}
sha = {n: hashlib.sha256(open(f, "rb").read()).hexdigest() for n, f in files.items()}
pk = json.load(open(files["picks.json"]))
dl = {b: json.load(open(files[f"demo_lam_{b}.json"])) for b in ("pi05", "pi0")}
assert all(len(d) == 20 for d in dl.values())
frozen = {k: dict(auto_li=r["arm_auto_li"], tli_li=r["arm_li"], exact=r["exact"],
                  auto_dl={b: f"tli@{r['auto']['t1']}.{r['auto']['t2']}.{dl[b][k]['lam']}" for b in dl},
                  auto_text=[r["auto"]["t1_text"], r["auto"]["t2_text"]], li_text=[r["li"]["t1_text"], r["li"]["t2_text"]])
          for k, r in pk["per_task"].items()}
print(json.dumps(frozen, indent=1))
print(ledger.record(
    "m3_prereg", bench="libero_ood", tag="pl2c_cover", seed=13,
    metrics={"rule": "role-filler coverage (V, O head, L landmark, G goal; same-role only) over the 27 single-step training prompts; "
                     "greedy, ties total-cover > same frame > word Jaccard > lower index; 2 sources; t1 = object-leaning pick",
             "lambda": {"auto_li": "Li's per-task lambda (g12_tli.MAP, as tli_li)", "auto_dl": "argmin demo-0 flow loss over {24,12,36}"},
             "agreement": pk["agreement"], "frozen_arms": frozen,
             "H1": "auto_li - tli_li: task-cluster bootstrap 95% lower bound >= -0.05 (e1_read.compare), per policy, 20 tasks pooled",
             "H2": "auto_li > zs: exact sign test on discordant pairs, alpha 0.05, per policy",
             "secondary": "auto_dl, same tests, descriptive",
             "states": "fresh_ood_pl2c (BASE 5100000, 20 per task, asserted unused), noise seed 13, paired",
             "backbones": ["pi05_libero", "pi0_libero"],
             "disclosure": "dev/confirm tasks were used before (pl2, G12, G13); aggregate tli_li / tli_sel totals known to the agent; "
                           "the rule reads no success data; demo 0 only sets auto_dl's lambda; identical arm strings share one rollout",
             "files_sha256": sha, "written_before": "any pl2c rollout"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/pl2c_cover.py",
    command="pre-registration: L2c symbolic coverage rule, picks frozen, before any L2c rollout"))
