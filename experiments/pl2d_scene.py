"""PLAN L, PROBE L2d (docs/PLAN_LANG.md; pre-registration experiments/prereg/pl2d_scene.py, ledger m3_prereg tag pl2d_scene):
a HIERARCHICAL retrieval key for demo-free TLI pairs.  L2c (experiments/pl2c_cover.py) matched the new instruction's role-filler
parts against all 27 single-step training prompts and lost to Li's hand pairs because coverage is scene-blind: Li's sources come
from the OOD task's own scene and template family.  L0b agrees: a word's effect moves into the scene's image tokens by layer 2,
so a language prototype is a scene-bound key.  L2d keys the store by scene first, then frame, then role-filler coverage.

THE RULE (fixed from that principle before any L2d rollout; nothing is tuned, no success number enters it):
  scene     the repo's fixed scene of a LIBERO task (lifelong_ood/policy/factors.parse; parser_hash in the prereg): LIBERO-10 / 90
            the SCENE name in the file name; LIBERO-Object one layout per task, its scene its object set; Spatial and Goal one
            kitchen table each, the suite.  A derived suite (LIBERO-PRO's *_task, Li's *_ood) is read with its BASE suite's rule
            (LIBERO-PRO keeps the original task's objects and layout; Li's OOD suites extend the Goal / Spatial tables).
  candidates  the pl2c pool (single-step pick-and-place training prompts, 27) restricted to the new task's scene; if fewer than 2,
            the pool restricted to the new task's base suite; if still fewer than 2, the whole pool.
  cover     inside the candidates, pl2c's greedy same-role coverage unchanged (parse V / O head / L landmark / G goal; key: new
            pairs covered, ties total covered, same frame, word Jaccard, lower index); 2 sources.
  slots     pl2c's: t1 = the pick leaning to the object roles ((#O + #L) - #G covered), ties: first pick is t1.
  lambda    pl2c auto_li's rule is Li's per-task lambda; Li's map has no entry for a new task, so the modal Li lambda 24 (17 of
            Li's 20 tasks use 24) is used for every TLI arm here.
Arms (per task): zs; scene_cover (the rule); blind_cover (pl2c's rule over the whole pool, the ablation); scene_random (2 sources
drawn from the SAME candidate set as scene_cover by a seeded hash of the task key, pl2c's slot rule, lambda 24: isolates coverage
from scene).  LIBERO-PRO texts are parsed through g22_libero_pro.reader_text ("pick X" -> "pick up X", for reading only); the
policy receives the benchmark's own text.  pl2c excludes a source whose text equals the new text (a task cannot be its own
source); a LIBERO-PRO task is never a training task, so the exclusion is applied to the benchmark's own text (it excludes nothing).

TEST SET (fresh): LIBERO-PRO's task split (new instructions in the training scenes; docs/LIBERO_PRO.md), the 27 G22-eligible tasks
(one step, put, one In/On atom), never used by any plan-L read (ledger scan below), with pi0.5's G22 zero-shot on fresh set t
<= 15 / 20 (< 0.8), read ONLY from G22's zs counts (ledger g22_libero_pro read_pi05, per_task[k].fresh.zs); all qualifying tasks,
ordered by sha256('pl2d:13:' + key), capped at 24; frozen in results/lang/pl2d/test_set.json before any composition rollout.
Start states: fresh_pro_pl2d (LIBERO's sampler on LIBERO-PRO's BDDLs, BASE 5300000, 20 per task, asserted unused), seed 13, paired.

    sbatch ... --wrap "$PY experiments/pl2d_scene.py --select --picks"         test set + rule picks -> results/lang/pl2d/
    $PY experiments/pl2d_scene.py --fresh-launch [--dry] | --fresh-check       start states
    $PY experiments/pl2d_scene.py --launch [--dry] [--bbs pi05_libero]         packs (writes .packs + a submit list)
    $PY experiments/pl2d_scene.py --read [--record]                            closed loop, H1-H3 with Holm, ledger pl2d_read
"""
import argparse
import glob
import hashlib
import json
import os
import random
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
R = os.path.join(ROOT, "results", "robot")
OUT = os.path.join(ROOT, "results", "lang", "pl2d")
SEED, LAM, K = 13, 24, 2
BBS = ("pi05_libero", "pi0_libero")
TEST_DIR, BASE, N_STATES = "fresh_pro_pl2d", 5300000, 20
USED_BASES = (100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1300000, 3300000, 3400000, 5100000)
ZS_MAX, CAP, MIN_TASKS = 15, 24, 12
ARMS = ("zs", "scene_cover", "blind_cover", "scene_random")
SHORT = {"libero_goal_task": "goal", "libero_spatial_task": "spatial", "libero_object_task": "object", "libero_10_task": "ten"}
EXCLUDE = "sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw"


def short(bb):
    return bb.split("_")[0]


def hsh(s):
    return hashlib.sha256(s.encode()).hexdigest()


# ── scenes ───────────────────────────────────────────────────────────────────────────────────────────────────────
def base_suite(suite):
    return re.sub(r"_(task|ood)$", "", suite)


def scene_of(t):
    """factors.parse's scene rule, applied with the task's BASE suite (see the docstring)."""
    m = re.match(r"((KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+)_", t.name)
    if m:
        return m.group(1)
    b = base_suite(t.suite)
    if b == "libero_object":
        return "libero_object:" + "+".join(sorted(set(t.objects.values())))
    return b


def train_scenes(tx):
    """training index t -> (suite, scene), matched by instruction text inside the task's suite (asserted unique)."""
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy import factors as F
    ft = F.load(F.TRAIN_SUITES)
    out = {}
    for t in range(40):
        s = D.suite_of(t)
        m = [x for x in ft if x.suite == s and x.language == tx[t]]
        assert len(m) == 1, (t, tx[t], len(m))
        out[t] = dict(suite=s, scene=scene_of(m[0]), name=m[0].name)
    return out


# ── the rule ─────────────────────────────────────────────────────────────────────────────────────────────────────
def cover_within(text, tx, cands, exclude_text, k=K):
    """pl2c_cover.cover with the source set restricted to `cands` (identical key, trace and slot rule)."""
    import pl2c_cover as C2
    N = C2.parse(text)
    S = {t: C2.parse(tx[t]) for t in cands if tx[t] != exclude_text}
    picks, covered, trace = [], set(), []

    def key(t):
        c = N["pairs"] & S[t]["pairs"]
        jac = len(N["words"] & S[t]["words"]) / len(N["words"] | S[t]["words"])
        return (len(c - covered), len(c), S[t]["frame"] == N["frame"], round(jac, 6), -t)
    for _ in range(k):
        cand = [t for t in S if t not in picks]
        ks = sorted(cand, key=key, reverse=True)
        best = ks[0]
        trace.append(dict(pick=best, key=list(key(best)), runner_up=ks[1] if len(ks) > 1 else None,
                          runner_up_key=list(key(ks[1])) if len(ks) > 1 else None,
                          tie_on=next((n for n, a, b in zip(("gain", "total", "frame", "jaccard", "index"), key(best), key(ks[1]))
                                       if a != b), None) if len(ks) > 1 else None))
        picks.append(best)
        covered |= N["pairs"] & S[best]["pairs"]
    t1, t2 = slots(N, S, picks[0], picks[1])
    return dict(picks=picks, t1=int(t1), t2=int(t2), trace=trace, pairs=sorted(map(list, N["pairs"])), parts=N["parts"],
                frame=N["frame"], uncovered=sorted(map(list, N["pairs"] - covered)),
                covered_by={str(t): sorted(map(list, N["pairs"] & S[t]["pairs"])) for t in picks})


def slots(N, S, a, b):
    def lean(t):
        c = N["pairs"] & S[t]["pairs"]
        return sum(r in ("O", "L") for r, _ in c) - sum(r == "G" for r, _ in c)
    return (b, a) if lean(b) > lean(a) else (a, b)


def candidates(scene, suite, pool, ts):
    """-> (candidate list, level): same scene, else same base suite, else the whole pool (each only if >= 2)."""
    c = [t for t in pool if ts[t]["scene"] == scene]
    if len(c) >= 2:
        return c, "scene"
    c = [t for t in pool if ts[t]["suite"] == base_suite(suite)]
    if len(c) >= 2:
        return c, "suite"
    return list(pool), "all"


def scene_random(key, text, tx, cands):
    import pl2c_cover as C2
    rng = random.Random(int(hsh(f"pl2d:random:{SEED}:{key}")[:16], 16))
    a, b = rng.sample(sorted(cands), 2)
    N = C2.parse(text)
    S = {t: C2.parse(tx[t]) for t in (a, b)}
    t1, t2 = slots(N, S, a, b)
    return dict(drawn=[a, b], t1=int(t1), t2=int(t2))


def rule(key, text_parse, text_raw, scene, suite, tx, ts):
    import pl2c_cover as C2
    pool = C2.pool(tx)
    cands, level = candidates(scene, suite, pool, ts)
    sc = cover_within(text_parse, tx, cands, text_raw)
    bl = cover_within(text_parse, tx, pool, text_raw)
    rn = scene_random(key, text_parse, tx, cands)
    arm = lambda c: f"tli@{c['t1']}.{c['t2']}.{LAM}"                         # noqa: E731
    return dict(scene=scene, level=level, candidates=cands, scene_cover=sc, blind_cover=bl, scene_random=rn,
                arms=dict(zs="zs", scene_cover=arm(sc), blind_cover=arm(bl), scene_random=arm(rn)),
                texts=dict(scene_cover=[tx[sc["t1"]], tx[sc["t2"]]], blind_cover=[tx[bl["t1"]], tx[bl["t2"]]],
                           scene_random=[tx[rn["t1"]], tx[rn["t2"]]]))


# ── test set ─────────────────────────────────────────────────────────────────────────────────────────────────────
def g22_zs():
    """pi0.5's G22 zero-shot successes on fresh set t, per task: ONLY per_task[k]['fresh']['zs'] is read (ledger row), checked
    against results/robot/g22_read_pi05.json."""
    f = glob.glob(os.path.join(ROOT, "ledger", "g22_libero_pro", "*read_pi05*.json"))
    assert len(f) == 1, f
    row = json.load(open(f[0]))
    zs = {k: int(v["fresh"]["zs"]) for k, v in row["metrics"]["per_task"].items()}
    rd = json.load(open(os.path.join(R, "g22_read_pi05.json")))["per_task"]
    assert all(int(rd[k]["fresh"]["zs"]) == z for k, z in zs.items())
    return zs, os.path.relpath(f[0], ROOT)


def usage(v):
    """Every ledger row / episode file naming this LIBERO-PRO task (suite + index): the experiments that used it."""
    pat = re.compile(rf"{v['suite']}_{v['index']}_|{v['suite']}__.*tasks-{v['index']}__")
    exps = set()
    for d in os.listdir(os.path.join(ROOT, "ledger")):
        for f in os.listdir(os.path.join(ROOT, "ledger", d)):
            if v["suite"] in f and pat.search(f):
                exps.add(d)
    files = sorted({re.sub(rf"_{v['suite']}_.*", "", f) for f in os.listdir(R) if f.endswith(".json") and f"_{v['suite']}_{v['index']}_" in f})
    return sorted(exps), files


def select():
    pl = json.load(open(os.path.join(R, "g22_plan.json")))["tasks"]
    zs, src = g22_zs()
    cand, why = [], {}
    for k, v in sorted(pl.items()):
        exps, files = usage(v)
        plan_l = [e for e in exps + files if re.match(r"^(pl\d|l\d)", e)]
        ok = bool(v["eligible"]) and zs[k] <= ZS_MAX and not plan_l
        why[k] = dict(language=v["language"], original=v["original"], eligible=bool(v["eligible"]), zs_pi05_fresh_t=zs[k],
                      used_by_ledger=exps, used_by_episode_files=files, plan_l_use=plan_l, qualifies=ok)
        if ok:
            cand.append(k)
    order = sorted(cand, key=lambda k: hsh(f"pl2d:{SEED}:{k}"))
    test = sorted(order[:CAP])
    assert len(test) >= MIN_TASKS, len(test)
    res = dict(rule=f"G22-eligible LIBERO-PRO task-split task, pi0.5 G22 fresh-t zs <= {ZS_MAX}/20, no plan-L use; all qualifying "
                    f"ordered by sha256('pl2d:{SEED}:'+key), first {CAP}", zs_source=src, qualifying=len(cand), hash_order=order,
               test=test, test_sha256=hsh(json.dumps(test)), per_task=why)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "test_set.json"), "w", encoding="utf-8"), indent=1)
    print(f"[pl2d] {len(cand)} qualify, test {len(test)}: {test}  sha256 {res['test_sha256'][:16]}", flush=True)
    for k in sorted(pl):
        w = why[k]
        print(f"  {k:10} {'Q' if w['qualifies'] else '-'} elig {int(w['eligible'])} zs {w['zs_pi05_fresh_t']:2d}  "
              f"ledger {w['used_by_ledger']}  '{w['language']}'", flush=True)


# ── picks (offline, no success data) ─────────────────────────────────────────────────────────────────────────────
def picks():
    import g12_tli as T
    import g22_libero_pro as G22
    import pl2c_cover as C2
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy import factors as F
    tx = D.Demos().tasks
    ts = train_scenes(tx)
    pool = C2.pool(tx)
    tm = F.task_map()
    out = dict(pool=pool, train_scenes={str(t): ts[t] for t in pool}, lam=LAM)
    # design check on Li's 20 (descriptive only; these tasks are spent and are NOT the test)
    li_pl = json.load(open(os.path.join(R, "g4_plan.json")))["tasks"]
    agree, li, same_as_pl2c = dict(exact=0, same_set=0, t1=0, t2=0, n=0), {}, 0
    old = json.load(open(os.path.join(ROOT, "results", "lang", "pl2c", "picks.json")))["per_task"]
    for k, v in sorted(li_pl.items()):
        t = F.parse(v["suite"], v["index"], tm[v["suite"]][v["index"]])
        assert t.language == v["language"], (k, t.language, v["language"])
        a, b, lam = T.MAP[v["language"].replace(" ", "_")]
        lp = (T.KEY[a], T.KEY[b])
        r = rule(k, v["language"], v["language"], scene_of(t), v["suite"], tx, ts)
        bl = r["blind_cover"]
        c0 = C2.cover(v["language"], tx)
        assert (bl["t1"], bl["t2"]) == (c0["t1"], c0["t2"]) == (old[k]["auto"]["t1"], old[k]["auto"]["t2"]), k   # blind == pl2c
        same_as_pl2c += 1
        sc = r["scene_cover"]
        e = dict(exact=(sc["t1"], sc["t2"]) == lp, same_set={sc["t1"], sc["t2"]} == set(lp), t1=sc["t1"] == lp[0], t2=sc["t2"] == lp[1])
        for q in e:
            agree[q] += int(e[q])
        agree["n"] += 1
        li[k] = dict(text=v["language"], scene=r["scene"], level=r["level"], n_candidates=len(r["candidates"]),
                     li=list(lp), li_text=[tx[lp[0]], tx[lp[1]]], li_scenes=[ts[lp[0]]["scene"], ts[lp[1]]["scene"]],
                     scene_cover=[sc["t1"], sc["t2"]], scene_text=r["texts"]["scene_cover"], blind=[bl["t1"], bl["t2"]],
                     pl2c_exact=bool(old[k]["exact"]), **e)
        print(f"[li] {k:10} {r['level']:5} ({len(r['candidates'])}) scene ({sc['t1']},{sc['t2']}) li {lp} blind ({bl['t1']},{bl['t2']})"
              f" {'EXACT' if e['exact'] else 'same set' if e['same_set'] else 't1' if e['t1'] else 't2' if e['t2'] else 'differ'}"
              f" | pl2c {'EXACT' if old[k]['exact'] else '-'}", flush=True)
    out["li_design_check"] = dict(agreement=agree, pl2c_agreement=json.load(open(os.path.join(ROOT, "results", "lang", "pl2c",
                                                                                                "picks.json")))["agreement"],
                                  blind_equals_pl2c=same_as_pl2c, per_task=li,
                                  li_sources_in_task_scene=sum(all(s == x["scene"] for s in x["li_scenes"]) for x in li.values()))
    print("[li] scene rule vs Li:", json.dumps(agree), "| pl2c:", json.dumps(out["li_design_check"]["pl2c_agreement"]), flush=True)
    # the frozen test set
    test = json.load(open(os.path.join(OUT, "test_set.json")))
    pl = json.load(open(os.path.join(R, "g22_plan.json")))["tasks"]
    per = {}
    for k in test["test"]:
        v = pl[k]
        t = F.parse(v["suite"], v["index"], tm[v["suite"]][v["index"]])
        assert t.language == v["language"], (k, t.language)
        orig = F.parse(base_suite(v["suite"]), v["index"], tm[v["suite"]][v["index"]])        # the original task (same file name)
        r = rule(k, G22.reader_text(v["language"]), v["language"], scene_of(t), v["suite"], tx, ts)
        r.update(text=v["language"], parse_text=G22.reader_text(v["language"]), suite=v["suite"], index=v["index"],
                 original=v["original"], original_scene=scene_of(orig), scene_equals_original=scene_of(orig) == scene_of(t),
                 distinct_arms=len(set(r["arms"].values())))
        per[k] = r
        print(f"[test] {k:10} '{v['language']}' scene {r['scene'][:40]} ({r['level']}, {len(r['candidates'])} cands; = orig "
              f"{r['scene_equals_original']}) | scene {r['arms']['scene_cover']} {r['texts']['scene_cover']} | blind "
              f"{r['arms']['blind_cover']} | random {r['arms']['scene_random']}", flush=True)
    out["test"] = per
    out["test_sha256"] = test["test_sha256"]
    out["overlap"] = dict(scene_eq_blind=sum(r["arms"]["scene_cover"] == r["arms"]["blind_cover"] for r in per.values()),
                          scene_eq_random=sum(r["arms"]["scene_cover"] == r["arms"]["scene_random"] for r in per.values()),
                          levels={lv: sum(r["level"] == lv for r in per.values()) for lv in ("scene", "suite", "all")})
    print("[test] overlap", json.dumps(out["overlap"]), flush=True)
    json.dump(out, open(os.path.join(OUT, "picks.json"), "w", encoding="utf-8"), indent=1)


def record_picks():
    from lifelong_ood import ledger
    pk = json.load(open(os.path.join(OUT, "picks.json")))
    ts = json.load(open(os.path.join(OUT, "test_set.json")))
    m = dict(li_design_check=dict(agreement=pk["li_design_check"]["agreement"], pl2c_agreement=pk["li_design_check"]["pl2c_agreement"],
                                  li_sources_in_task_scene=pk["li_design_check"]["li_sources_in_task_scene"],
                                  per_task={k: dict(scene=r["scene_cover"], li=r["li"], level=r["level"], exact=r["exact"])
                                            for k, r in pk["li_design_check"]["per_task"].items()}),
             test_set=dict(tasks=ts["test"], sha256=ts["test_sha256"], qualifying=ts["qualifying"], rule=ts["rule"],
                           zs_pi05_fresh_t={k: ts["per_task"][k]["zs_pi05_fresh_t"] for k in ts["test"]}),
             arms={k: dict(r["arms"], level=r["level"], n_candidates=len(r["candidates"])) for k, r in pk["test"].items()},
             overlap=pk["overlap"])
    print(ledger.record("pl2d_picks", bench="libero_pro", tag="offline", seed=SEED, metrics=m,
                        bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=dict(lam=LAM, k=K),
                        script="experiments/pl2d_scene.py", command=" ".join(sys.argv)))


# ── start states ─────────────────────────────────────────────────────────────────────────────────────────────────
def by_suite():
    test = json.load(open(os.path.join(OUT, "test_set.json")))["test"]
    pl = json.load(open(os.path.join(R, "g22_plan.json")))["tasks"]
    out = {}
    for k in test:
        out.setdefault(pl[k]["suite"], []).append(pl[k]["index"])
    return {s: sorted(v) for s, v in out.items()}


def fresh_launch(dry):
    assert not any(b <= BASE < b + 1000 or BASE <= b < BASE + N_STATES for b in USED_BASES), BASE
    assert not os.path.exists(os.path.join(R, TEST_DIR)) or not os.listdir(os.path.join(R, TEST_DIR)), "set exists"
    for suite, ids in by_suite().items():
        cmd = ["sbatch", "--parsable", "-p", "overcap", "-A", "overcap", "--requeue", f"--exclude={EXCLUDE}",
               f"--export=ALL,SUITE={suite},TASKS={'+'.join(map(str, ids))},N={N_STATES},BASE={BASE},FRESH_DIR={TEST_DIR}",
               "experiments/fresh_states.sbatch"]
        print(" ".join(cmd))
        if not dry:
            print(subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=ROOT).stdout.strip())


def fresh_check():
    """Assert the new set is unused: 20 states per task, no state within 1e-6 of any state of an earlier fresh set of that task."""
    out = {}
    for suite, ids in by_suite().items():
        for t in ids:
            x = np.load(os.path.join(R, TEST_DIR, f"{suite}_{t}.npy"))
            assert len(x) == N_STATES, (suite, t, x.shape)
            assert len({hashlib.sha256(s.tobytes()).hexdigest() for s in x}) == N_STATES, "duplicate states"
            dmin, sets = np.inf, []
            for f in sorted(glob.glob(os.path.join(R, "fresh_*", f"{suite}_{t}.npy"))):
                if os.path.basename(os.path.dirname(f)) == TEST_DIR:
                    continue
                y = np.load(f)
                if y.shape[1] != x.shape[1]:
                    continue
                dmin = min(dmin, float(np.min(np.linalg.norm(x[:, None] - y[None], axis=-1))))
                sets.append(os.path.basename(os.path.dirname(f)))
            assert sets and dmin > 1e-6, (suite, t, dmin, sets)
            out[f"{suite}_{t}"] = dict(min_dist=dmin, compared=sets)
            print(suite, t, f"min distance to earlier sets {dmin:.4f} over {sets}", flush=True)
    json.dump(out, open(os.path.join(OUT, "fresh_check.json"), "w", encoding="utf-8"), indent=1)


# ── closed loop ──────────────────────────────────────────────────────────────────────────────────────────────────
def exp_of(v):
    return f"pl2d_{SHORT[v['suite']]}"


def launch(dry, bbs=BBS, lanes=(("tail-lab", 2), ("overcap", 4))):
    """pl2's pack machinery (pl2_compose._spec; 4 instances per A40), lanes split over tail-lab (at most 2 at once) and overcap.
    Writes the packs and results/lang/pl2d/submit_<tag>.sh (sbatch lines only), which is run on the login node."""
    import pl2_compose as PC
    assert os.path.exists(os.path.join(OUT, "fresh_check.json")), "run --fresh-check first"
    pk = json.load(open(os.path.join(OUT, "picks.json")))["test"]
    specs = []
    for bb in bbs:
        for k, r in sorted(pk.items()):
            v = dict(suite=r["suite"], index=r["index"])
            for arm in dict.fromkeys(r["arms"][a] for a in ARMS):
                specs.append(PC._spec(v["index"], arm, v["suite"], bb, exp_of(v), N_STATES, TEST_DIR))
    assert all(s is not None for s in specs)
    todo = [s for s in specs if not os.path.exists(s["OUT"])]
    print(len(specs), "instances,", len(todo), "to run", flush=True)
    os.makedirs(os.path.join(R, ".packs"), exist_ok=True)
    n_pack = len(os.listdir(os.path.join(R, ".packs")))
    lane_list = [p for p, n in lanes for _ in range(n)]
    lines, li = ["#!/bin/bash", "set -e", f"cd {ROOT}"], [None] * len(lane_list)
    for j, i in enumerate(range(0, len(todo), 4)):
        grp = todo[i:i + 4]
        while True:
            n_pack += 1
            path = os.path.join(R, ".packs", f"p_{n_pack:04d}.tsv")
            try:
                os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                break
            except FileExistsError:
                continue
        with open(path, "w", encoding="utf-8") as f:
            for sp in grp:
                f.write("\t".join(f"{a}={b}" for a, b in sp.items()) + "\n")
        lane = j % len(lane_list)
        part = lane_list[lane]
        acct = f"--partition={part} --account={part} --qos=short"
        dep = f"--dependency=afterany:${{J{lane}}} " if li[lane] else ""
        lines.append(f"J{lane}=$(sbatch --parsable {dep}{acct} --requeue --time=04:00:00 --gres=gpu:a40:1 --exclude={EXCLUDE} "
                     f"--export=ALL,SPECS={path} experiments/roll_pack.sbatch); echo \"{os.path.basename(path)} lane {lane} {part} "
                     f"[{' '.join(str(s['TASKS']) + ':' + s['TAG'] for s in grp)}] -> $J{lane}\"")
        li[lane] = 1
    tag = "_".join(short(b) for b in bbs)
    sh = os.path.join(OUT, f"submit_{tag}.sh")
    if not dry:
        open(sh, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines[3:]) if dry else f"wrote {sh} ({len(lines) - 3} packs)", flush=True)


def holm(ps, alpha=0.05):
    """Holm step-down over a dict name -> p; -> name -> rejected."""
    order = sorted(ps, key=lambda n: ps[n])
    rej, stop = {}, False
    for i, n in enumerate(order):
        if not stop and ps[n] <= alpha / (len(order) - i):
            rej[n] = True
        else:
            stop, rej[n] = True, False
    return rej


def read(record, bbs=BBS):
    import e1_read as E1
    import pl2_compose as PC
    pk = json.load(open(os.path.join(OUT, "picks.json")))["test"]
    res = {}
    for bb in bbs:
        x, missing, per = {}, [], {}
        for k, r in sorted(pk.items()):
            v = dict(suite=r["suite"], index=r["index"])
            z = {a: PC._eps(exp_of(v), v, r["arms"][a], bb) for a in ARMS}
            if None in z.values() or any(len(e) != N_STATES for e in z.values()):
                missing.append(k)
                continue
            x[k] = {a: {t: int(bool(e["success"])) for t, e in ep.items()} for a, ep in z.items()}
            per[k] = {a: int(sum(s.values())) for a, s in x[k].items()} | dict(arms=r["arms"], level=r["level"])
        out = dict(backbone=bb, tasks=len(x), missing=missing, per_task=per)
        if x and not missing:
            C = E1.compare
            out["totals"] = {a: int(sum(per[k][a] for k in per)) for a in ARMS}
            out["n"] = N_STATES * len(x)
            hy = dict(H1=("scene_cover", "zs"), H2=("scene_cover", "blind_cover"), H3=("scene_cover", "scene_random"))
            for h, (a, b) in hy.items():
                out[h] = C({k: {"x": z[a], "y": z[b]} for k, z in x.items()})
                out[h]["arms"] = [a, b]
            rej = holm({h: out[h]["sign_p"] for h in hy})
            for h in hy:
                out[h]["holm_rejected"] = rej[h]
                out[h]["passes"] = bool(rej[h] and out[h]["x"] > out[h]["y"])
            out["desc"] = dict(blind_vs_zs=C({k: {"x": z["blind_cover"], "y": z["zs"]} for k, z in x.items()}),
                               random_vs_zs=C({k: {"x": z["scene_random"], "y": z["zs"]} for k, z in x.items()}))
            dif = [k for k in x if pk[k]["arms"]["scene_cover"] != pk[k]["arms"]["blind_cover"]]
            if dif:
                out["desc"]["scene_vs_blind_differing"] = C({k: {"x": x[k]["scene_cover"], "y": x[k]["blind_cover"]} for k in dif})
            out["by_suite"] = {SHORT[s]: {a: int(sum(per[k][a] for k in per if pk[k]["suite"] == s)) for a in ARMS}
                               | dict(tasks=sum(pk[k]["suite"] == s for k in per)) for s in SHORT if any(pk[k]["suite"] == s for k in per)}
            print(f"\n[pl2d] {short(bb)}: {len(x)} tasks; totals {out['totals']} of {out['n']}", flush=True)
            for h in list(hy) + list(out["desc"]):
                d = out[h] if h in hy else out["desc"][h]
                print(f"  {h:26} {d['x']} vs {d['y']} of {d['n']}, discordant {d['discordant']}, sign p {d['sign_p']:.3g}, diff "
                      f"{d['diff']['diff']:+.3f} [{d['diff']['lo']:+.3f}, {d['diff']['hi']:+.3f}]"
                      + (f" holm {d['holm_rejected']} passes {d['passes']}" if h in hy else ""), flush=True)
            for k in per:
                print(f"  {k:10} {per[k]['level']:5} " + " ".join(f"{a} {per[k][a]:2d}" for a in ARMS), flush=True)
        else:
            print(f"[pl2d] {short(bb)}: missing {missing}", flush=True)
        res[short(bb)] = out
    json.dump(res, open(os.path.join(OUT, "read.json"), "w", encoding="utf-8"), indent=1)
    if record:
        assert all(not r["missing"] for r in res.values()), "incomplete"
        from lifelong_ood import ledger
        print(ledger.record("pl2d_read", bench="libero_pro", tag="read", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(states=TEST_DIR, base=BASE, episodes=N_STATES, arms=ARMS, backbones=list(bbs), lam=LAM,
                                        prereg="m3_prereg tag pl2d_scene"),
                            script="experiments/pl2d_scene.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in ("select", "picks", "record-picks", "fresh-launch", "fresh-check", "launch", "read", "dry", "record"):
        ap.add_argument("--" + f, action="store_true")
    ap.add_argument("--bbs", default=",".join(BBS))
    a = ap.parse_args()
    bbs = tuple(a.bbs.split(","))
    if a.select:
        select()
    if a.picks:
        picks()
    if a.record_picks:
        record_picks()
    if a.fresh_launch:
        fresh_launch(a.dry)
    if a.fresh_check:
        fresh_check()
    if a.launch:
        launch(a.dry, bbs)
    if a.read:
        read(a.record, bbs)
