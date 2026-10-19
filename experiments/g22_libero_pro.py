"""Plan H4, read G22 (ledger m3_prereg tag g22_libero_pro; docs/LIBERO_PRO.md): the finished OOD system (iteration 12) on LIBERO-PRO's
TASK split (Zhou et al., arXiv 2510.03827), the one split whose new tasks recombine concepts the policy holds (a held object, or a
held instance description, with a held goal in a known scene).  Nothing in the system changes: practice chooses each held concept's
source under the grasp switch (G10's two stages) and under TLI (G14's two stages, lambda 24), then the operator per task by 10
practice episodes each on start states set a (ties to the switch); the chosen arm is tested against the policy alone.

Suites: libero_goal_task, libero_spatial_task, libero_object_task, libero_10_task (LIBERO-PRO's own BDDLs and init states, run in
LIBERO-PRO's LIBERO package; experiments/libero_client.py).  The instruction is the BDDL's new (:language ...) (LIBERO-PRO keeps
the old file names, so LIBERO's file-name language is the OLD instruction; arm origprompt gives the policy that old instruction,
the protocol of LIBERO-PRO's own table).

Eligible tasks (the system reads one object and one goal around one grasp): one step, verb put (or "pick ... and place it"), a
preposition, and one In/On goal atom.  Others run the policy alone only.  Source pool: G10's (the single-step pick-and-place
training instructions of LIBERO-Goal and LIBERO-Spatial) for the Goal and Spatial splits, as for Li's suites built on the same two
suites; for the Object and LIBERO-10 splits the same rule over that suite's own training instructions (G10's pool holds none of
their scenes).

Start states: practice on fresh set a (LIBERO's sampler on LIBERO-PRO's BDDLs, seeds 300000+), test on fresh set t (seeds 900000+,
20 per task) and on LIBERO-PRO's own 50 init states (comparable with the published table).

    $PY experiments/g22_libero_pro.py --plan
    $PY experiments/g22_libero_pro.py --smoke [--backbone pi0_libero]          # 4 episodes per split (not counted)
    setsid nohup $PY experiments/g22_libero_pro.py --run [--backbone pi0_libero] > results/robot/g22_run_<short>.log 2>&1 &
    $PY experiments/g22_libero_pro.py --read [--backbone pi0_libero] [--record]
"""
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
R = os.path.join(ROOT, "results", "robot")
PY = os.path.expanduser("~/flash/miniconda3/envs/chunks/bin/python")
SEED, LAM = 13, 24
PLAN = os.path.join(R, "g22_plan.json")
EXCLUDE = "hal,tars,t1000,vicki,johnny5,c3po,olivaw"
SUITES = ("libero_goal_task", "libero_spatial_task", "libero_object_task", "libero_10_task")
SHORT = {"libero_goal_task": "goal", "libero_spatial_task": "spatial", "libero_object_task": "object", "libero_10_task": "ten"}
FRESH_A, FRESH_T = "fresh_pro_a", "fresh_pro_t"
N_PRO = 50                                                    # LIBERO-PRO's (and LIBERO's) 50 init states per task
# LIBERO-PRO's published success (README leaderboard at eafdb80, read 2026-10-19; Task and Position columns), per suite
# (the leaderboard's Task column; "original" = the README's motivation table, unperturbed LIBERO)
PUBLISHED = {"pi05_libero": {"task": dict(goal=0.00, spatial=0.01, ten=0.01, object=0.01), "original": dict(goal=0.97, spatial=0.96,
                                                                                                             ten=0.93, object=0.98)},
             "pi0_libero": {"task": dict(goal=0.00, spatial=0.00, ten=0.00, object=0.00), "original": dict(goal=0.92, spatial=0.90,
                                                                                                           ten=0.82, object=0.98)}}


def reader_text(lang):
    """The concept reader's grammar names "pick up X and place it ..."; LIBERO-PRO writes "pick X and place it ...".  For READING
    concepts only (eligibility and the word rule); the policy receives the benchmark's own text."""
    return re.sub(r"^pick (?!up )", "pick up ", lang)


def eligible(t):
    from lifelong_ood.policy import primitives as P
    r = reader_text(t.language)
    return (len(P.steps_of(r)) == 1 and P.object_key(r).startswith("put|") and bool(P.goal_key(r).split("|")[0])
            and len(t.slots) == 1 and t.slots[0][0] in ("In", "On"))


def pool(suite):
    """G10's source pool for the Goal and Spatial splits (unchanged); G10's rule over the suite's own training instructions else."""
    import g4_ood as G
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy import primitives as P
    if suite in ("libero_goal_task", "libero_spatial_task"):
        return G.source_pool()
    tx = D.Demos().tasks
    base = suite.replace("_task", "")
    return [i for i in D.SUITES[base] if len(P.steps_of(tx[i])) == 1 and P.object_key(tx[i]).startswith("put|")
            and P.goal_key(tx[i]).split("|")[0]]


def plan():
    """-> results/robot/g22_plan.json: every task of the task split, its eligibility, pool and the word rule's sources (G4's rule:
    exact concept, else content-word Jaccard >= 0.5, the concept's source the first of its tasks in the test task's base suite;
    where it finds none, the pool member whose key shares most content words: the fixed partner of stage 1 must exist)."""
    import g4_ood as G
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy import factors as F
    from lifelong_ood.policy import primitives as P
    dm = D.Demos()
    obj, goal = {}, {}
    for ti, text in sorted(dm.tasks.items()):
        if len(P.steps_of(text)) != 1:
            continue
        obj.setdefault(P.object_key(text), []).append(ti)
        if P.goal_key(text).split("|")[0]:
            goal.setdefault(P.goal_key(text), []).append(ti)
    tasks = {}
    for t in F.load(F.PRO_SUITES):
        el = eligible(t)
        ent = dict(suite=t.suite, index=t.index, language=t.language, original=F.prompt_of(F.task_map()[t.suite][t.index]),
                   eligible=el, slots=[list(s) for s in t.slots])
        if el:
            pl_ = pool(t.suite)
            base = D.SUITES[t.suite.replace("_task", "")]
            r = reader_text(t.language)
            src = {}
            for kind, key, lib, kf in (("object", P.object_key(r), obj, P.object_key), ("goal", P.goal_key(r), goal, P.goal_key)):
                m = G.match(key, lib)
                if m is not None:
                    own = [i for i in lib[m] if i in base]
                    src[kind], rule = (own or lib[m])[0], "word rule"
                else:
                    sc = sorted(((len(G.words(key) & G.words(kf(dm.tasks[i]))), i in base, -i) for i in pl_), reverse=True)
                    src[kind], rule = -sc[0][2], "most shared words in the pool"
                ent[f"{kind}_key"], ent[f"{kind}_src"], ent[f"{kind}_rule"] = key, src[kind], rule
            ent["pool"] = pl_
        tasks[f"{SHORT[t.suite]}:{t.index}"] = ent
    json.dump(dict(tasks=tasks, parser=F.parser_hash()), open(PLAN, "w", encoding="utf-8"), indent=1)
    for k, v in tasks.items():
        print(f"[g22] {k:10} {'E' if v['eligible'] else '-'} {v['language'][:70]:70} "
              + (f"O {v['object_src']} ({v['object_rule']}) G {v['goal_src']} ({v['goal_rule']}) pool {len(v['pool'])}" if v["eligible"] else ""),
              flush=True)
    print(f"[g22] {len(tasks)} tasks, {sum(v['eligible'] for v in tasks.values())} eligible", flush=True)


def _pl():
    return json.load(open(PLAN))["tasks"]


def _bsfx(backbone):
    return "" if backbone == "pi05_libero" else "_" + backbone.split("_")[0]


def ld(exp, v, arm, backbone):
    f = os.path.join(R, f"{exp}_{v['suite']}_{v['index']}_{arm.replace('@', '_')}_seed{SEED}{_bsfx(backbone)}.json")
    if not os.path.exists(f):
        return None
    eps = {int(e["trial"]): int(bool(e["success"])) for t_ in json.load(open(f))["tasks"].values() for e in t_["episodes"]}
    return eps


def _launch(exp, v, arms, trials, fresh, backbone, per=4):
    """launch_pack (packs of `per` instances per GPU), then its packs kept off the excluded nodes.  fresh None: LIBERO-PRO's own init
    states 0..trials-1."""
    cmd = [PY, "experiments/launch_pack.py", "--tasks", str(v["index"]), "--arms", ",".join(arms), "--experiment", exp, "--first",
           "0", "--trials", str(trials), "--seed", str(SEED), "--bind", "track", "--partition", "overcap", "--per", str(per),
           "--suite", v["suite"]] + (["--fresh", "--fresh-dir", fresh] if fresh else []) \
        + (["--backbone", backbone] if backbone != "pi05_libero" else [])
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()
    jobs = re.findall(r"-> job (\d+)", out)
    for j in jobs:
        subprocess.run(["scontrol", "update", f"JobId={j}", f"ExcNodeList={EXCLUDE}"], capture_output=True)
    return jobs


def op_arm(op, a, g):
    return f"pfield@{a}.{g}" if op == "p" else f"tli@{a}.{g}.{LAM}"


def choose(op, stage, k, v, backbone):
    """G10 / G14's choice for one task: stage 1 the object source (the word rule's goal source fixed), stage 2 the goal source (the
    chosen object source fixed); the most successes in 5 practice episodes on set a, ties to the word rule's source, then the lowest
    index.  None while practice is incomplete."""
    pool_ = v["pool"]
    if stage == 1:
        arms, pref = {a: op_arm(op, a, v["goal_src"]) for a in pool_}, v["object_src"]
    else:
        a1 = choose(op, 1, k, v, backbone)
        if a1 is None:
            return None
        arms, pref = {g: op_arm(op, a1, g) for g in pool_}, v["goal_src"]
    sc = {c: ld(f"g22_s{stage}{op}", v, arm, backbone) for c, arm in arms.items()}
    if None in sc.values():
        return None
    return max(pool_, key=lambda c: (sum(sc[c].values()), c == pref, -c))


def pair(op, k, v, backbone):
    a, g = choose(op, 1, k, v, backbone), choose(op, 2, k, v, backbone)
    return None if a is None or g is None else op_arm(op, a, g)


def operator(k, v, backbone):
    """G15's choice: TLI if its pair's 10 practice episodes on set a beat the switch's, else the switch.  -> (arm, counts) or None."""
    ps, ts = pair("p", k, v, backbone), pair("t", k, v, backbone)
    if ps is None or ts is None:
        return None
    pa, ta = ld("g22_accp", v, ps, backbone), ld("g22_acct", v, ts, backbone)
    if pa is None or ta is None:
        return None
    cp, ct = sum(pa.values()), sum(ta.values())
    return (ts if ct > cp else ps), dict(psel=cp, tsel=ct, op="tsel" if ct > cp else "psel", psel_arm=ps, tsel_arm=ts)


def smoke(backbone):
    """4 episodes per split (experiment g22_smoke, not counted): the policy alone and the old instruction on LIBERO-PRO's init states
    of every split's first eligible task (task 0 for the LIBERO-10 split's policy-alone check), and the two operators with the word
    rule's pair on fresh set a."""
    pl = _pl()
    for s in SUITES:
        ks = [k for k, v in pl.items() if v["suite"] == s]
        el = [k for k in ks if pl[k]["eligible"]]
        k = (el or ks)[0]
        v = pl[k]
        print(k, "pro-init", _launch("g22_smoke", v, ["zs", "origprompt"], 4, None, backbone, per=2), flush=True)
        if el:
            print(k, "fresh a", _launch("g22_smoke", v, [op_arm("p", v["object_src"], v["goal_src"]),
                                                         op_arm("t", v["object_src"], v["goal_src"])], 4, FRESH_A, backbone, per=2),
                  flush=True)
        else:
            print(k, "fresh t", _launch("g22_smoke", v, ["zs"], 4, FRESH_T, backbone, per=1), flush=True)


def _state_file(backbone):
    return os.path.join(R, f"g22_state_{backbone.split('_')[0]}.json")


def run(backbone):
    """Per task, each stage launched once its inputs exist (a state file records what is launched, so --run resumes)."""
    pl = _pl()
    short = backbone.split("_")[0]
    sf = _state_file(backbone)
    st = json.load(open(sf)) if os.path.exists(sf) else {}

    def once(key, fn):
        if key in st:
            return
        st[key] = fn()
        json.dump(st, open(sf, "w", encoding="utf-8"), indent=1)

    while True:
        for k, v in pl.items():
            once(f"{k}|base", lambda: _launch("g22_pro", v, ["zs", "origprompt"], N_PRO, None, backbone)
                 + _launch("g22_test", v, ["zs"], 20, FRESH_T, backbone))
            if not v["eligible"]:
                continue
            for op in ("p", "t"):
                once(f"{k}|s1{op}", lambda: _launch(f"g22_s1{op}", v, [op_arm(op, a, v["goal_src"]) for a in v["pool"]], 5, FRESH_A,
                                                    backbone))
                a1 = choose(op, 1, k, v, backbone)
                if a1 is not None:
                    once(f"{k}|s2{op}", lambda: _launch(f"g22_s2{op}", v, [op_arm(op, a1, g) for g in v["pool"]], 5, FRESH_A,
                                                        backbone))
                arm = pair(op, k, v, backbone)
                if arm is not None:
                    once(f"{k}|acc{op}", lambda: _launch(f"g22_acc{op}", v, [arm], 10, FRESH_A, backbone))
                    once(f"{k}|test{op}", lambda: _launch("g22_test", v, [arm], 20, FRESH_T, backbone))
            ch = operator(k, v, backbone)
            if ch is not None:
                once(f"{k}|pro_chosen", lambda: _launch("g22_pro", v, [ch[0]], N_PRO, None, backbone))
        if all(f"{k}|pro_chosen" in st for k, v in pl.items() if v["eligible"]):
            break
        time.sleep(300)
    print(f"[g22 {short}] G22_TEST_LAUNCHED {time.strftime('%F %T')}", flush=True)
    while read(backbone, False, quiet=True)["missing"]:
        time.sleep(300)
    read(backbone, False)
    print(f"[g22 {short}] G22_DONE backbone={backbone} read=results/robot/g22_read_{short}.json {time.strftime('%F %T')}", flush=True)


def read(backbone, record, quiet=False):
    import e1_read as E1
    pl = _pl()
    short = backbone.split("_")[0]
    C = E1.compare
    fresh, pro, missing, choice = {}, {}, [], {}
    for k, v in pl.items():
        zt = ld("g22_test", v, "zs", backbone)
        zp, op_ = ld("g22_pro", v, "zs", backbone), ld("g22_pro", v, "origprompt", backbone)
        if v["eligible"]:
            ch = operator(k, v, backbone)
            if ch is None:
                missing.append(k)
                continue
            choice[k] = ch[1]
            ft = {"zs": zt, "chosen": ld("g22_test", v, ch[0], backbone), "psel": ld("g22_test", v, ch[1]["psel_arm"], backbone),
                  "tsel": ld("g22_test", v, ch[1]["tsel_arm"], backbone)}
            fp = {"zs": zp, "origprompt": op_, "chosen": ld("g22_pro", v, ch[0], backbone)}
        else:
            ft, fp = {"zs": zt}, {"zs": zp, "origprompt": op_}
        if None in ft.values() or None in fp.values():
            missing.append(k)
            continue
        fresh[k], pro[k] = ft, fp
    el = [k for k in fresh if pl[k]["eligible"]]
    cmp = lambda tab, a, b, ks: C({k: {"x": tab[k][a], "y": tab[k][b]} for k in ks})   # noqa: E731
    res = dict(backbone=backbone, missing=missing, choice=choice, eligible=sorted(el),
               not_eligible=sorted(k for k in fresh if not pl[k]["eligible"]))
    if el:
        res["H1_chosen_vs_zs_fresh"] = cmp(fresh, "chosen", "zs", el)
        res["H1_chosen_vs_zs_fresh"]["passes"] = bool(res["H1_chosen_vs_zs_fresh"]["x"] > res["H1_chosen_vs_zs_fresh"]["y"]
                                                      and res["H1_chosen_vs_zs_fresh"]["sign_p"] < 0.05)
        res["H1b_chosen_vs_zs_pro"] = cmp(pro, "chosen", "zs", el)
        res["H1b_chosen_vs_zs_pro"]["passes"] = bool(res["H1b_chosen_vs_zs_pro"]["x"] > res["H1b_chosen_vs_zs_pro"]["y"]
                                                     and res["H1b_chosen_vs_zs_pro"]["sign_p"] < 0.05)
        res["n_tsel"] = sum(c["op"] == "tsel" for c in choice.values())
        for s in SUITES:
            ks = [k for k in el if pl[k]["suite"] == s]
            if ks:
                res[f"split_{SHORT[s]}"] = dict(fresh=cmp(fresh, "chosen", "zs", ks), pro=cmp(pro, "chosen", "zs", ks),
                                                psel_fresh=cmp(fresh, "psel", "zs", ks), tsel_fresh=cmp(fresh, "tsel", "zs", ks))
    rate = lambda tab, a, ks: (sum(sum(tab[k][a].values()) for k in ks) / max(1, sum(len(tab[k][a]) for k in ks)))   # noqa: E731
    res["per_suite"] = {}
    for s in SUITES:
        ks = [k for k in pro if pl[k]["suite"] == s]
        if not ks:
            continue
        e_ = [k for k in ks if pl[k]["eligible"]]
        res["per_suite"][SHORT[s]] = dict(
            tasks=len(ks), eligible=len(e_),
            pro_zs=rate(pro, "zs", ks), pro_origprompt=rate(pro, "origprompt", ks),
            pro_system=(sum(sum(pro[k]["chosen" if k in e_ else "zs"].values()) for k in ks) / (N_PRO * len(ks))),
            fresh_zs=rate(fresh, "zs", ks),
            fresh_system=(sum(sum(fresh[k]["chosen" if k in e_ else "zs"].values()) for k in ks) / (20 * len(ks))),
            published_task=PUBLISHED.get(backbone, {}).get("task", {}).get(SHORT[s]))
    res["per_task"] = {k: dict(fresh={a: int(sum(z.values())) for a, z in fresh[k].items()},
                               pro={a: int(sum(z.values())) for a, z in pro[k].items()}) for k in fresh}
    if not quiet:
        for n_ in ("H1_chosen_vs_zs_fresh", "H1b_chosen_vs_zs_pro"):
            if n_ in res:
                h = res[n_]
                print(f"[g22 {short}] {n_:22} {h['x']} vs {h['y']} of {h['n']}, discordant {h['discordant']}, sign p {h['sign_p']:.3g}, "
                      f"diff {h['diff']['diff']:+.3f} [{h['diff']['lo']:+.3f}, {h['diff']['hi']:+.3f}]", flush=True)
        for s, d in res["per_suite"].items():
            print(f"[g22 {short}] {s:8} {d['eligible']}/{d['tasks']} eligible | LIBERO-PRO init: policy {d['pro_zs']:.2f}, old instruction "
                  f"{d['pro_origprompt']:.2f}, system {d['pro_system']:.2f} (published {d['published_task']}) | fresh t: policy "
                  f"{d['fresh_zs']:.2f}, system {d['fresh_system']:.2f}", flush=True)
        print(f"[g22 {short}] TLI chosen on {res.get('n_tsel')} | missing {missing}", flush=True)
        json.dump(res, open(os.path.join(R, f"g22_read_{short}.json"), "w", encoding="utf-8"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("g22_libero_pro", bench="libero_pro", tag=f"read_{short}", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=dict(backbone=backbone),
                            script="experiments/g22_libero_pro.py", command=" ".join(sys.argv)))
    return res


if __name__ == "__main__":
    bb = sys.argv[sys.argv.index("--backbone") + 1] if "--backbone" in sys.argv else "pi05_libero"
    if "--plan" in sys.argv:
        plan()
    if "--smoke" in sys.argv:
        smoke(bb)
    if "--run" in sys.argv:
        run(bb)
    if "--read" in sys.argv:
        read(bb, "--record" in sys.argv)
