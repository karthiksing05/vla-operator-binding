"""PRE-REGISTRATION of plan L probe L2d (docs/PLAN_LANG.md; code experiments/pl2d_scene.py), written 2026-10-23 AFTER the frozen
test set (results/lang/pl2d/test_set.json), the offline picks (results/lang/pl2d/picks.json) and the start-state check
(results/lang/pl2d/fresh_check.json), and BEFORE any L2d rollout (no pl2d_* episode file exists when this row is recorded; asserted).

Question.  Does a HIERARCHICAL retrieval key (scene, then frame, then same-role coverage) make demo-free TLI pairs work on new
instructions in familiar scenes?  L2c's coverage was scene-blind and lost to Li's hand pairs; L0b says language prototypes are
scene-bound keys.

Rule (experiments/pl2d_scene.py docstring; fixed from that principle, no success number enters it):
  scene       factors.parse's fixed scene rule (parser_hash below), read with the task's base suite: LIBERO-10 the SCENE name,
              LIBERO-Object the object set, Goal / Spatial the suite.
  candidates  the 27-prompt pl2c pool restricted to the scene; < 2 -> the base suite; < 2 -> the whole pool.
  cover       pl2c's greedy same-role coverage inside the candidates (ties total, frame, Jaccard, index); 2 sources; pl2c's slots.
  lambda      Li's map has no entry for new tasks: the modal Li lambda 24 (17 of 20 Li tasks) for every TLI arm.
Design check (descriptive, no success data; Li's 20 tasks are spent and are not the test): the scene rule reproduces Li's exact
pair on 3 / 20 (pl2c's scene-blind rule: 9 / 20); t1 on 10, t2 on 8.  Li's own two sources lie in the task's scene (this rule's
scene) on 10 / 20 tasks, all Goal-OOD (picks.json li_sources_in_task_scene); every Spatial-OOD pair of Li's takes one source from
LIBERO-Goal ("put the bowl on the stove / on top of the cabinet", or the cream-cheese / wine-bottle prompt), which the scene rule
cannot reach.  So the L2c reading "Li's sources come from the task's own scene" holds for the Goal-OOD tasks only.

Test set: LIBERO-PRO's task split (new instruction, objects and layout of a training scene), the 27 G22-eligible tasks, pi0.5's G22
zero-shot on fresh set t <= 15 / 20, no plan-L use (ledger scan): 21 qualify, all taken (cap 24), order sha256('pl2d:13:' + key).
Frozen list sha256 in test_set.json.  Start states fresh_pro_pl2d (LIBERO's sampler on LIBERO-PRO's BDDLs, BASE 5300000, 20 per
task; no logged fresh-states job used a base within 1000 of it, and every state is > 1e-6 from every state of fresh_pro_a /
fresh_pro_t), noise seed 13, paired.  Identical arm strings share one rollout.

Disclosure.
  * The 21 tasks were used by G22 (ledger g22_*): zs and the old instruction on LIBERO-PRO's 50 init states, zs on set t, and the
    practice-chosen grasp switch and TLI pairs (5- and 10-episode practice on set a, 20 test episodes on set t).  Selection reads
    ONLY per_task[k].fresh.zs.  The agent's context also displayed G22's per-task composition counts (the read file was printed
    while locating the zs field); the selection rule is mechanical (every eligible task with zs <= 15 is taken), so those counts
    could not steer it.
  * Offline the scene rule and pl2c's blind rule pick the SAME arm on 20 of 21 test tasks (LIBERO-PRO's new instructions reuse
    words of their own scene's training prompts), so H2 has discordant pairs only where the arms differ (object:0).  H2 is kept
    as stated; it is expected to have almost no power on this set.  scene_random never equals scene_cover.
  * Candidate level: scene 11 tasks (Goal 6, Spatial 5), suite 9 (Object: one layout per task), all 1 (ten:5, the only LIBERO-10
    prompt of its scene is in the pool).  For ten:5 scene_random draws from the whole pool.

Arms per policy (pi05_libero first, then pi0_libero) x 21 tasks x 20 episodes: zs, scene_cover, blind_cover, scene_random.
  H1  scene_cover > zs            exact two-sided sign test on discordant episode pairs (e1_read.compare)
  H2  scene_cover > blind_cover   same
  H3  scene_cover > scene_random  same
  Holm step-down over {H1, H2, H3} per policy at family alpha 0.05; a hypothesis passes iff Holm rejects AND scene_cover has more
  successes.  Descriptive: blind vs zs, random vs zs, scene vs blind on the differing tasks, per-suite totals, task-cluster
  bootstrap intervals."""
import glob
import hashlib
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/vla-operator-binding"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger
from lifelong_ood.policy import factors as F

OUT = os.path.join(ROOT, "results", "lang", "pl2d")
assert not glob.glob(os.path.join(ROOT, "results", "robot", "pl2d_*.json")), "an L2d rollout already exists"
assert os.path.exists(os.path.join(OUT, "fresh_check.json")), "start states not checked"
files = {n: os.path.join(OUT, n) for n in ("test_set.json", "picks.json", "fresh_check.json")}
sha = {n: hashlib.sha256(open(f, "rb").read()).hexdigest() for n, f in files.items()}
ts = json.load(open(files["test_set.json"]))
pk = json.load(open(files["picks.json"]))
fc = json.load(open(files["fresh_check.json"]))
assert sorted(pk["test"]) == ts["test"] and len(fc) == len(ts["test"])
frozen = {k: dict(r["arms"], level=r["level"], text=r["text"], scene_text=r["texts"]["scene_cover"],
                  blind_text=r["texts"]["blind_cover"], random_text=r["texts"]["scene_random"]) for k, r in pk["test"].items()}
print(json.dumps(frozen, indent=1))
print(ledger.record(
    "m3_prereg", bench="libero_pro", tag="pl2d_scene", seed=13,
    metrics={"rule": "scene (factors.parse rule, base suite) -> base suite -> all, each only with >= 2 candidates of the 27-prompt "
                     "pl2c pool; inside: pl2c greedy same-role coverage, 2 sources, pl2c slots; lambda 24 (modal Li lambda)",
             "design_check_li20": pk["li_design_check"]["agreement"], "design_check_pl2c": pk["li_design_check"]["pl2c_agreement"],
             "li_sources_in_task_scene": pk["li_design_check"]["li_sources_in_task_scene"],
             "test_set": ts["test"], "test_sha256": ts["test_sha256"], "qualifying": ts["qualifying"], "selection": ts["rule"],
             "zs_pi05_fresh_t": {k: ts["per_task"][k]["zs_pi05_fresh_t"] for k in ts["test"]},
             "frozen_arms": frozen, "overlap": pk["overlap"],
             "H1": "scene_cover > zs: exact two-sided sign test on discordant pairs",
             "H2": "scene_cover > blind_cover: same (arms identical on 20/21 tasks offline; expected low power)",
             "H3": "scene_cover > scene_random: same",
             "multiplicity": "Holm over {H1,H2,H3} per policy, family alpha 0.05; pass = Holm-rejected and more successes",
             "states": "fresh_pro_pl2d (BASE 5300000, 20 per task, asserted unused), noise seed 13, paired",
             "backbones": ["pi05_libero", "pi0_libero"],
             "disclosure": "the 21 tasks were used by G22 (zs, origprompt, practice-chosen switch / TLI); selection reads only G22's "
                           "pi0.5 fresh-t zs; G22 composition counts were visible to the agent, selection is mechanical (all "
                           "qualifying tasks); Li's 20 tasks are spent and only enter the descriptive design check",
             "parser": F.parser_hash(), "files_sha256": sha, "written_before": "any pl2d rollout"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/pl2d_scene.py",
    command="pre-registration: L2d hierarchical scene key, test set and arms frozen, before any L2d rollout"))
