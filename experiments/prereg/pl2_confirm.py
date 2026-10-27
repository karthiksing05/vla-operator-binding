"""ADDENDUM to the pre-registration of plan L's probe L2 (ledger m3_prereg tag pl2_compose, experiments/prereg/pl2_compose.py):
the CONFIRM read, pi0.5 (pi05_libero) only.  Written 2026-10-22, after the dev read (commit 426df7fe, ledger pl2_dev) and BEFORE
any closed-loop episode on the 12 confirm tasks existed.

Why an addendum.  The pl2_compose row names the 12 confirm tasks and says they are "kept for a confirm read with these rules
frozen", but states H1 / H2 on the 8 dev tasks only.  This row states the confirm read: exactly the dev protocol, applied to
the 12 confirm tasks, nothing changed.

Frozen (unchanged from pl2_compose):
  demos      results/lang/pl2/demos.json (recorded before the dev read, cross-policy, fresh_ood_pl2d BASE 3300000); demo 0
             selects, demo 1 scores.  No new demos.
  selection  the offline files results/lang/pl2/offline/pi05_<task>.json, written 12:45-14:10 on 2026-10-22 by the same run as
             the dev tasks' (before any L2 closed-loop episode); their sha256 are recorded below.  l2sel = per task the
             operator in {word, tmpl, tli_sel, role} with the lowest demo-0 flow loss; no zero-shot fallback, no guard; same
             operators, layer spans {1-17,1-6,7-12,13-17}, lambda grid {12,24,36}.  Nothing re-selected or tuned.
             (Their demo-1 losses are already in ledger pl2_offline as descriptive offline numbers; no closed-loop number exists.)
  arms       the dev arms: zs, tli_li (Li's pairs and lambda), word, tmpl, tli_sel, role, rand, pos (l2sel = one of them).
  states     fresh_ood_pl2 (BASE 3400000), 20 per task; the confirm tasks' starts in this set have never been rolled (only the
             8 dev tasks have pl2_dev episodes).  Noise seed 13, paired across arms; launch_pack --bind track.
Hypotheses (pi0.5, 12 confirm tasks x 20 episodes = 240 paired episodes per arm):
  H1  l2sel > zs: exact sign test on discordant episode pairs, alpha 0.05 (and more successes).
  H2  l2sel non-inferior to tli_li: 95% lower bound of the pooled success difference (two-level paired cluster bootstrap over
      tasks then starts, B 10000, seed 0; lifelong_ood/stats.bootstrap_diff, the dev read's) above -0.05.
  Descriptive: l2sel vs rand / pos; each arm vs zs; part correctness g and p|g; per-task picks.
Scope: pi0 is not run on the confirm set in this read (the lead's instruction: pi0.5 only)."""
import hashlib
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/vla-operator-binding"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

CONFIRM = ["goal:0", "goal:3", "goal:8", "goal:9", "spatial:0", "spatial:1", "spatial:2", "spatial:3", "spatial:4", "spatial:5",
           "spatial:8", "spatial:9"]
OUT = os.path.join(ROOT, "results", "lang", "pl2")


def sha(f):
    return hashlib.sha256(open(f, "rb").read()).hexdigest()


files = {k: os.path.join(OUT, "offline", f"pi05_{k.replace(':', '_')}.json") for k in CONFIRM}
picks = {k: json.load(open(f))["l2sel"] for k, f in files.items()}
used = [k for k in CONFIRM if any(n.startswith(f"pl2_dev_{k.split(':')[0]}_libero_{k.split(':')[0]}_ood_{k.split(':')[1]}_")
                                    or n.startswith(f"pl2_conf_{k.split(':')[0]}_libero_{k.split(':')[0]}_ood_{k.split(':')[1]}_")
                                    for n in os.listdir(os.path.join(ROOT, "results", "robot")))]
assert not used, f"confirm starts already used: {used}"
print(ledger.record(
    "m3_prereg", bench="libero_ood", tag="pl2_confirm", seed=13,
    metrics={"addendum_to": "m3_prereg__libero_ood__pl2_compose__s13__3e41ecc1",
             "backbones": ["pi05_libero"],
             "confirm_tasks": CONFIRM,
             "protocol": "the dev protocol of pl2_compose / pl2_dev applied unchanged to the 12 confirm tasks",
             "demos": "results/lang/pl2/demos.json (unchanged; cross-policy, fresh_ood_pl2d BASE 3300000); demo 0 selects, demo 1 scores",
             "demos_sha256": sha(os.path.join(OUT, "demos.json")),
             "selection": "frozen offline files (demo-0 flow loss only; l2sel = lowest demo-0 loss among word/tmpl/tli_sel/role; no zero-shot "
                          "fallback, no guard; spans {1-17,1-6,7-12,13-17}; lambda {12,24,36}); not re-run",
             "offline_sha256": {k: sha(f) for k, f in files.items()},
             "l2sel_picks": picks,
             "arms": ["zs", "tli_li", "word", "tmpl", "tli_sel", "role", "rand", "pos"],
             "states": "fresh_ood_pl2 (BASE 3400000, 20 per task; confirm tasks never rolled on it), noise seed 13, paired, bind track",
             "hypotheses": {
                 "H1": "pi0.5, l2sel > zs on the 12 confirm tasks x 20 episodes (exact sign test on discordant pairs, alpha 0.05)",
                 "H2": "pi0.5, l2sel non-inferior to tli_li (Li's hand pairs and lambda): 95% lower bound of the pooled success difference, "
                       "two-level paired cluster bootstrap (tasks, then starts; B 10000, seed 0) above -0.05",
                 "descriptive": "l2sel vs rand / pos; each arm vs zs; g and p|g; per-task picks"},
             "scope": "pi0.5 only; pi0 confirm not run in this read",
             "written_before": "any closed-loop episode on the 12 confirm tasks"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/pl2_confirm.py",
    command="addendum pre-registration: plan L L2 confirm read (pi0.5, 12 confirm tasks, dev protocol frozen)"))
