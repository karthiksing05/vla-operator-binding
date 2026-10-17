"""Plan G, read G4: COMPOSITIONAL OOD on a published benchmark -- Li's LIBERO-Goal-OOD and LIBERO-Spatial-OOD (arXiv 2505.03500,
IJCAI 2026; reused by arXiv 2607.00351), 20 tasks that pair the object of one of pi0.5's training tasks with the target of
another.  No demonstration of any test task: every concept is fitted on pi0.5's own training demonstrations (LIBERO-Spatial,
-Object, -Goal, -10), the object prototype on the frames before the grasp of every training task whose object concept it is, the
goal prototype on the frames from the grasp to the release of every training task whose goal concept it is (ndemo
demonstrations per task).  A test task's concepts are read from its instruction (lifelong_ood/policy/primitives.py) and matched
to the library's: exactly, else the concept whose content words overlap most (Jaccard >= 0.5, prepositions kept); a factor with
no match is served by the policy alone.  Acceptance: 10 practice episodes on start states set a (LIBERO's sampler, seeds 300000+)
keep the composition only if it beats the policy; test on set b (seeds 400000+, 20 per task), paired; Li's modified LIBERO
(its looser On, its flat stove; experiments/libero_client.py), max 300 steps.

    $PY experiments/g4_ood.py --plan [--fit] | --accept --test | --read [--record]
"""
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
R = os.path.join(ROOT, "results", "robot")
PY = os.path.expanduser("~/flash/miniconda3/envs/chunks/bin/python")
PLAN = os.path.join(R, "g4_plan.json")
SEED, RANK, NDEMO = 13, 4, 5
SUITES = {"goal": "libero_goal_ood", "spatial": "libero_spatial_ood"}
STOP = {"the", "a", "of", "it", "and"}


def words(key):
    return {w for w in re.split(r"[ |]+", key) if w and w not in STOP}


def match(key, lib):
    """The library concept for `key`: exact, else the best content-word Jaccard >= 0.5 (unique best), else None."""
    if key in lib:
        return key
    sc = sorted(((len(words(key) & words(k)) / len(words(key) | words(k)), k) for k in lib), reverse=True)
    if sc and sc[0][0] >= 0.5 and (len(sc) == 1 or sc[1][0] < sc[0][0]):
        return sc[0][1]
    return None


def plan():
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy import factors as F
    from lifelong_ood.policy import primitives as P
    dm = D.Demos()
    obj, goal = {}, {}
    for ti, text in sorted(dm.tasks.items()):
        if len(P.steps_of(text)) != 1:
            continue
        obj.setdefault(P.object_key(text), []).append(ti)
        if P.goal_key(text).split("|")[0]:                          # a goal needs a preposition (not "turn on the stove")
            goal.setdefault(P.goal_key(text), []).append(ti)
    names, tasks = {}, {}
    for t in F.load(F.OOD_SUITES):
        short = next(k for k, v in SUITES.items() if v == t.suite)
        ok, gk = P.object_key(t.language), P.goal_key(t.language)
        mo, mg = match(ok, obj), match(gk, goal)
        ent = dict(suite=t.suite, index=t.index, language=t.language, object_key=ok, goal_key=gk, object_concept=mo,
                   goal_concept=mg, object_proto=None, goal_proto=None)
        for kind, m_, lib, flag in (("object", mo, obj, "--approach"), ("goal", mg, goal, "--goal")):
            if m_ is None:
                continue
            n = names.setdefault((kind, m_), dict(name=f"Q{kind[0].upper()}{len(names):02d}", kind=kind, key=m_, tasks=lib[m_],
                                                   flag=flag))["name"]
            ent[f"{kind}_proto"] = n
        tasks[f"{short}:{t.index}"] = ent
    out = dict(tasks=tasks, protos={v["name"]: v for v in names.values()}, ndemo=NDEMO, rank=RANK)
    json.dump(out, open(PLAN, "w"), indent=1)
    for k, v in tasks.items():
        print(f"[g4] {k:11} {v['language'][:46]:46} O {v['object_concept']!s:44} G {v['goal_concept']}", flush=True)
    print(f"[g4] {len(tasks)} tasks, {len(names)} prototypes", flush=True)


def sb(script):
    out = subprocess.run(["sbatch", "--parsable", "--exclude=hal,tars,t1000,vicki,johnny5,c3po", "--partition=overcap",
                          "--account=overcap", "--qos=short", f"--export=ALL,ENV=openpi,SCRIPT={script}", "slurm/jobs/robot_gpu.sbatch"],
                         capture_output=True, text=True, cwd=ROOT)
    return out.stdout.strip() or out.stderr.strip()


def fit():
    for p in json.load(open(PLAN))["protos"].values():
        if not os.path.exists(os.path.join(R, f"prim_{p['name']}_0.pt")):
            print(p["name"], sb(f"experiments/c0_fit.py --task 0 --joint {'+'.join(map(str, p['tasks']))} --key {p['key'].replace(' ', '_')} "
                                f"--name {p['name']} --rank {RANK} {p['flag']} --source d40 --ndemo {NDEMO}"), flush=True)


def arms(v):
    a = {"zs": "zs"}
    if v["object_proto"] or v["goal_proto"]:
        a["fac"] = f"fac@{v['object_proto'] or 'x'}.{v['goal_proto'] or 'x'}"
    return a


def tag(a):
    return a.replace("@", "_") + f"_seed{SEED}"


def launch(stage, trials, fresh_dir):
    for k, v in json.load(open(PLAN))["tasks"].items():
        short = k.split(":")[0]
        cmd = [PY, "experiments/launch_pack.py", "--tasks", str(v["index"]), "--arms", ",".join(arms(v).values()),
               "--experiment", f"g4_{stage}_{short}", "--first", "0", "--trials", str(trials), "--seed", str(SEED), "--bind", "track",
               "--partition", "overcap", "--per", "2", "--suite", v["suite"], "--fresh", "--fresh-dir", fresh_dir]
        print(k, subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-120:], flush=True)


def load(exp, suite, t, a):
    f = os.path.join(R, f"{exp}_{suite}_{t}_{tag(a)}.json")                # launch_pack names a non-default suite in the file
    if not os.path.exists(f):
        return None
    return {int(e["trial"]): int(bool(e["success"])) for v in json.load(open(f))["tasks"].values() for e in v["episodes"]}


def read(record):
    import e1_read as E1
    pl = json.load(open(PLAN))
    acc, test, kept, missing = {}, {}, {}, []
    for k, v in pl["tasks"].items():
        short = k.split(":")[0]
        A = arms(v)
        s = {a: load(f"g4_accept_{short}", v["suite"], v["index"], x) for a, x in A.items()}
        x_ = {a: load(f"g4_main_{short}", v["suite"], v["index"], x) for a, x in A.items()}
        if None in s.values() or None in x_.values():
            missing.append(k)
            continue
        acc[k] = {a: int(sum(z.values())) for a, z in s.items()}
        kept[k] = "fac" if "fac" in acc[k] and acc[k]["fac"] > acc[k]["zs"] else "zs"
        test[k] = x_
    C = E1.compare
    res = dict(missing=missing, acceptance=acc, kept=kept)
    for name, sel in (("all", lambda k: True), ("goal", lambda k: k.startswith("goal")), ("spatial", lambda k: k.startswith("spatial"))):
        ts = {k: x for k, x in test.items() if sel(k)}
        res[name] = dict(H1=C({k: {"x": x[kept[k]], "y": x["zs"]} for k, x in ts.items()}),
                         fac_vs_zs=C({k: {"x": x["fac"], "y": x["zs"]} for k, x in ts.items() if "fac" in x}),
                         rate_kept=sum(sum(x[kept[k]].values()) for k, x in ts.items()) / max(1, 20 * len(ts)),
                         rate_zs=sum(sum(x["zs"].values()) for k, x in ts.items()) / max(1, 20 * len(ts)))
    res["H1"] = res["all"]["H1"]
    res["H1"]["passes"] = bool(res["H1"]["x"] > res["H1"]["y"] and res["H1"]["sign_p"] < 0.05)
    res["per_task"] = {k: {a: int(sum(z.values())) for a, z in x.items()} | {"kept": kept[k]} for k, x in test.items()}
    for n in ("all", "goal", "spatial"):
        h = res[n]["H1"]
        print(f"[g4] {n:8} kept {h['x']} vs zs {h['y']} of {h['n']} (sign p {h['sign_p']:.3g}); rates {res[n]['rate_kept']:.3f} vs "
              f"{res[n]['rate_zs']:.3f}; fac alone {res[n]['fac_vs_zs']['x']}", flush=True)
    if missing:
        print("[g4] missing", missing, flush=True)
    json.dump(res, open(os.path.join(R, "g4_read.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("g4_ood", bench="libero_ood", tag="read", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=dict(ndemo=NDEMO),
                            script="experiments/g4_ood.py", command=" ".join(sys.argv)))


def source(tasks, suite):
    """Plan G4b: the training task whose instruction stands for a concept in this scene -- the concept's source tasks from the test
    task's own base suite (libero_goal_ood -> libero_goal), else all of them; the first in benchmark order."""
    from lifelong_ood.policy import data as D
    base = D.SUITES[suite.replace("_ood", "")]
    own = [t for t in tasks if t in base]
    return (own or tasks)[0]


def pfield(v, pl):
    o = source(pl["protos"][v["object_proto"]]["tasks"], v["suite"]) if v["object_proto"] else "x"
    g = source(pl["protos"][v["goal_proto"]]["tasks"], v["suite"]) if v["goal_proto"] else "x"
    return f"pfield@{o}.{g}"


def screen(launch_it):
    """Plan G4b screen (exploratory): the policy's own fields under the concepts' source instructions, switched at the grasp, on
    start states set a (10 per task, noise seed 13), against the policy's recorded episodes there."""
    pl = json.load(open(PLAN))
    if launch_it:
        for k, v in pl["tasks"].items():
            cmd = [PY, "experiments/launch_pack.py", "--tasks", str(v["index"]), "--arms", pfield(v, pl), "--experiment",
                   f"g4_accept_{k.split(':')[0]}", "--first", "0", "--trials", "10", "--seed", str(SEED), "--bind", "track",
                   "--partition", "overcap", "--per", "1", "--suite", v["suite"], "--fresh", "--fresh-dir", "fresh_ood_a"]
            print(k, pfield(v, pl), subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-60:], flush=True)
        return
    tot, rows = dict(zs=0, fac=0, pf=0), {}
    for k, v in pl["tasks"].items():
        sh = k.split(":")[0]
        z, f, p_ = (load(f"g4_accept_{sh}", v["suite"], v["index"], a) for a in ("zs", arms(v)["fac"], pfield(v, pl)))
        if p_ is None:
            continue
        rows[k] = dict(zs=sum(z.values()), fac=sum(f.values()), pf=sum(p_.values()), arm=pfield(v, pl))
        for a in tot:
            tot[a] += rows[k][a]
        print(f"[g4b] {k:11} {v['language'][:44]:44} zs {rows[k]['zs']:2} fac {rows[k]['fac']:2} pfield {rows[k]['pf']:2} "
              f"({rows[k]['arm']})", flush=True)
    print(f"[g4b] {len(rows)} tasks: zs {tot['zs']} fac {tot['fac']} pfield {tot['pf']}", flush=True)
    return rows


def arms_b(v, pl):
    a = arms(v)
    a["pfield"] = pfield(v, pl)
    return a


def launch_b():
    """G4b test: zero-shot, the prototype composition and the policy-field composition on start states set c (seeds 500000+)."""
    pl = json.load(open(PLAN))
    for k, v in pl["tasks"].items():
        cmd = [PY, "experiments/launch_pack.py", "--tasks", str(v["index"]), "--arms", ",".join(arms_b(v, pl).values()),
               "--experiment", f"g4b_main_{k.split(':')[0]}", "--first", "0", "--trials", "20", "--seed", str(SEED), "--bind",
               "track", "--partition", "overcap", "--per", "3", "--suite", v["suite"], "--fresh", "--fresh-dir", "fresh_ood_c"]
        print(k, subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-100:], flush=True)


def read_b(record):
    """G4b (m3_prereg g4b_field): acceptance on set a keeps the best of {policy, prototype composition, policy-field composition}
    (ties to the policy, then the prototypes); H1 kept > policy on set c."""
    import e1_read as E1
    pl = json.load(open(PLAN))
    order = ("zs", "fac", "pfield")
    acc, test, kept, missing = {}, {}, {}, []
    for k, v in pl["tasks"].items():
        sh, A = k.split(":")[0], arms_b(v, pl)
        s_ = {a: load(f"g4_accept_{sh}", v["suite"], v["index"], x) for a, x in A.items()}
        x_ = {a: load(f"g4b_main_{sh}", v["suite"], v["index"], x) for a, x in A.items()}
        if None in s_.values() or None in x_.values():
            missing.append(k)
            continue
        acc[k] = {a: int(sum(z.values())) for a, z in s_.items()}
        kept[k] = max(order, key=lambda a: (acc[k][a], -order.index(a)))
        test[k] = x_
    C = E1.compare
    res = dict(missing=missing, acceptance=acc, kept=kept,
               H1=C({k: {"x": x[kept[k]], "y": x["zs"]} for k, x in test.items()}),
               pfield_vs_zs=C({k: {"x": x["pfield"], "y": x["zs"]} for k, x in test.items()}),
               fac_vs_zs=C({k: {"x": x["fac"], "y": x["zs"]} for k, x in test.items()}),
               per_task={k: {a: int(sum(z.values())) for a, z in x.items()} | {"kept": kept[k]} for k, x in test.items()})
    for sub in ("goal", "spatial"):
        res[sub] = C({k: {"x": x[kept[k]], "y": x["zs"]} for k, x in test.items() if k.startswith(sub)})
    res["H1"]["passes"] = bool(res["H1"]["x"] > res["H1"]["y"] and res["H1"]["sign_p"] < 0.05)
    for n_ in ("H1", "pfield_vs_zs", "fac_vs_zs", "goal", "spatial"):
        h = res[n_]
        print(f"[g4b] {n_:13} {h['x']} vs {h['y']} of {h['n']}, discordant {h['discordant']}, sign p {h['sign_p']:.3g}, "
              f"diff {h['diff']['diff']:+.3f} [{h['diff']['lo']:+.3f}, {h['diff']['hi']:+.3f}]", flush=True)
    print("[g4b] kept", {a: sum(v == a for v in kept.values()) for a in order}, "| missing", missing, flush=True)
    json.dump(res, open(os.path.join(R, "g4b_read.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("g4_ood", bench="libero_ood", tag="read_field", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/g4_ood.py", command=" ".join(sys.argv)))


def launch_c(backbone, stage):
    """G4c: G4b's policy-field composition on another flow-matching policy (pi0_libero, smolvla): zero-shot and pfield, acceptance
    on set a (stage accept) and test on set c (stage main)."""
    pl = json.load(open(PLAN))
    trials, fresh = (10, "fresh_ood_a") if stage == "accept" else (20, "fresh_ood_c")
    for k, v in pl["tasks"].items():
        cmd = [PY, "experiments/launch_pack.py", "--tasks", str(v["index"]), "--arms", f"zs,{pfield(v, pl)}", "--experiment",
               f"g4c_{stage}_{k.split(':')[0]}", "--first", "0", "--trials", str(trials), "--seed", str(SEED), "--bind", "track",
               "--partition", "overcap", "--per", "2", "--suite", v["suite"], "--fresh", "--fresh-dir", fresh, "--backbone", backbone]
        print(k, subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-90:], flush=True)


def read_c(backbone, record):
    """G4c (m3_prereg g4c_field): acceptance on set a keeps pfield over the policy only if it beats it; H1 kept > policy on set c."""
    import e1_read as E1
    pl, short = json.load(open(PLAN)), backbone.split("_")[0]
    acc, test, kept, missing = {}, {}, {}, []
    for k, v in pl["tasks"].items():
        sh = k.split(":")[0]
        A = {"zs": "zs", "pfield": pfield(v, pl)}
        f = lambda st, a: os.path.join(R, f"g4c_{st}_{sh}_{v['suite']}_{v['index']}_{a.replace('@', '_')}_seed{SEED}_{short}.json")   # noqa: E731
        ld = lambda st, a: ({int(e["trial"]): int(bool(e["success"])) for t_ in json.load(open(f(st, a)))["tasks"].values()
                             for e in t_["episodes"]} if os.path.exists(f(st, a)) else None)   # noqa: E731
        s_, x_ = {a: ld("accept", x) for a, x in A.items()}, {a: ld("main", x) for a, x in A.items()}
        if None in s_.values() or None in x_.values():
            missing.append(k)
            continue
        acc[k] = {a: int(sum(z.values())) for a, z in s_.items()}
        kept[k] = "pfield" if acc[k]["pfield"] > acc[k]["zs"] else "zs"
        test[k] = x_
    C = E1.compare
    res = dict(backbone=backbone, missing=missing, acceptance=acc, kept=kept,
               H1=C({k: {"x": x[kept[k]], "y": x["zs"]} for k, x in test.items()}),
               pfield_vs_zs=C({k: {"x": x["pfield"], "y": x["zs"]} for k, x in test.items()}),
               per_task={k: {a: int(sum(z.values())) for a, z in x.items()} | {"kept": kept[k]} for k, x in test.items()})
    for sub in ("goal", "spatial"):
        res[sub] = C({k: {"x": x[kept[k]], "y": x["zs"]} for k, x in test.items() if k.startswith(sub)})
    res["H1"]["passes"] = bool(res["H1"]["x"] > res["H1"]["y"] and res["H1"]["sign_p"] < 0.05)
    for n_ in ("H1", "pfield_vs_zs", "goal", "spatial"):
        h = res[n_]
        print(f"[g4c {short}] {n_:13} {h['x']} vs {h['y']} of {h['n']}, discordant {h['discordant']}, sign p {h['sign_p']:.3g}, "
              f"diff {h['diff']['diff']:+.3f} [{h['diff']['lo']:+.3f}, {h['diff']['hi']:+.3f}]", flush=True)
    print(f"[g4c {short}] kept pfield on {sum(v == 'pfield' for v in kept.values())} | missing {missing}", flush=True)
    json.dump(res, open(os.path.join(R, f"g4c_read_{short}.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("g4_ood", bench="libero_ood", tag=f"read_field_{short}", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=dict(backbone=backbone),
                            script="experiments/g4_ood.py", command=" ".join(sys.argv)))


LI_MAPPING = "/coc/flash10/ksingara3/ext_dl/pi0tl/pi0-text-latent/examples/libero/mapping.json"
LI_STEPS = dict(                    # Li et al.'s task_hidden_states_mapping (examples/libero/main.py), its last entry: the switch
    put_the_cream_cheese_on_the_plate=24, put_the_cream_cheese_on_the_stove=24, put_the_cream_cheese_on_top_of_the_cabinet=24,
    put_the_wine_bottle_in_the_bowl=12, put_the_wine_bottle_on_the_plate=24, put_the_wine_bottle_on_the_stove=24,
    put_the_orange_juice_on_the_stove=24, put_the_tomato_sauce_on_top_of_the_cabinet=24, put_the_bbq_source_on_the_plate=24,
    put_the_cream_cheese_in_the_basket=24, put_the_bowl_at_table_center_on_the_stove=24,
    put_the_bowl_at_table_center_on_the_cabinet=24, put_the_bowl_next_to_the_plate_on_the_stove=30,
    put_the_bowl_next_to_the_plate_on_the_cabinet=24, put_the_bowl_on_cookie_box_on_the_stove=24,
    put_the_bowl_on_cookie_box_on_the_cabinet=14, put_the_orange_juice_on_the_plate=24, put_the_milk_on_the_plate=24,
    put_the_chocolate_pudding_on_the_plate=24, put_the_butter_on_the_plate=24)


def li_arms(v, pl):
    """G4d: Li et al.'s pi0^S (arXiv 2505.03500) and its crossings with ours.  Li switches the prompt at environment step
    N / 2 * 5 (N from LI_STEPS, counting the 10 settle steps); our client queries every 5 steps after them, so that is query
    N / 2 - 2."""
    from lifelong_ood.policy import data as D
    inv = {t: i for i, t in D.Demos().tasks.items()}
    row = next(r for r in json.load(open(LI_MAPPING))[v["suite"]] if r[0] == v["language"])
    la, lg = inv[row[1]], inv[row[2]]
    q = LI_STEPS[v["language"].replace(" ", "_")] // 2 - 2
    ours = pfield(v, pl)
    return {"li_fixed": f"pfield@{la}.{lg}.q{q}", "li_grasp": f"pfield@{la}.{lg}", "ours_fixed": f"{ours}.q{q}"}


def launch_d(backbone):
    pl = json.load(open(PLAN))
    for k, v in pl["tasks"].items():
        cmd = [PY, "experiments/launch_pack.py", "--tasks", str(v["index"]), "--arms", ",".join(li_arms(v, pl).values()),
               "--experiment", f"g4d_main_{k.split(':')[0]}", "--first", "0", "--trials", "20", "--seed", str(SEED), "--bind",
               "track", "--partition", "overcap", "--per", "3", "--suite", v["suite"], "--fresh", "--fresh-dir", "fresh_ood_c"]
        cmd += ["--backbone", backbone] if backbone != "pi05_libero" else []
        print(k, subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-90:], flush=True)


def read_d(backbone, record):
    """G4d (m3_prereg g4d_li): unfiltered success of pi0^S (li_fixed), Li's sources at our grasp (li_grasp), our sources at Li's
    fixed step (ours_fixed) and ours (ours_grasp: G4b / G4c's pfield arm), each against the policy, on set c."""
    import e1_read as E1
    pl = json.load(open(PLAN))
    short = backbone.split("_")[0]
    sfx = "" if backbone == "pi05_libero" else f"_{short}"

    def ld(exp, v, a):
        f = os.path.join(R, f"{exp}_{v['suite']}_{v['index']}_{a.replace('@', '_')}_seed{SEED}{sfx}.json")
        return ({int(e["trial"]): int(bool(e["success"])) for t_ in json.load(open(f))["tasks"].values() for e in t_["episodes"]}
                if os.path.exists(f) else None)
    x, missing = {}, []
    for k, v in pl["tasks"].items():
        sh = k.split(":")[0]
        main = "g4b_main" if backbone == "pi05_libero" else "g4c_main"
        z = dict(zs=ld(f"{main}_{sh}", v, "zs"), ours_grasp=ld(f"{main}_{sh}", v, pfield(v, pl)),
                 **{a: ld(f"g4d_main_{sh}", v, arm) for a, arm in li_arms(v, pl).items()})
        if None in z.values():
            missing.append(k)
            continue
        x[k] = z
    C = E1.compare
    res = dict(backbone=backbone, missing=missing, tasks=len(x))
    for a in ("li_fixed", "li_grasp", "ours_fixed", "ours_grasp"):
        res[f"{a}_vs_zs"] = C({k: {"x": z[a], "y": z["zs"]} for k, z in x.items()})
    res["li_fixed_vs_ours_grasp"] = C({k: {"x": z["li_fixed"], "y": z["ours_grasp"]} for k, z in x.items()})
    res["li_grasp_vs_li_fixed"] = C({k: {"x": z["li_grasp"], "y": z["li_fixed"]} for k, z in x.items()})
    res["ours_grasp_vs_ours_fixed"] = C({k: {"x": z["ours_grasp"], "y": z["ours_fixed"]} for k, z in x.items()})
    res["per_task"] = {k: {a: int(sum(v.values())) for a, v in z.items()} for k, z in x.items()}
    for n_, h in res.items():
        if isinstance(h, dict) and "sign_p" in h:
            print(f"[g4d {short}] {n_:26} {h['x']} vs {h['y']} of {h['n']}, discordant {h['discordant']}, sign p {h['sign_p']:.3g}",
                  flush=True)
    print(f"[g4d {short}] missing {missing}", flush=True)
    json.dump(res, open(os.path.join(R, f"g4d_read_{short}.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("g4_ood", bench="libero_ood", tag=f"read_li_{short}", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=dict(backbone=backbone),
                            script="experiments/g4_ood.py", command=" ".join(sys.argv)))


def source_pool():
    """G10: every single-step pick-and-place training instruction of LIBERO-Goal and LIBERO-Spatial (task indices of pi0.5's
    training set): the candidates practice chooses among for a held concept's source."""
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy import primitives as P
    tx = D.Demos().tasks
    return [i for i in D.SUITES["libero_goal"] + D.SUITES["libero_spatial"] if len(P.steps_of(tx[i])) == 1
            and P.object_key(tx[i]).startswith("put|") and P.goal_key(tx[i]).split("|")[0]]


def _bsfx(backbone):
    return "" if backbone == "pi05_libero" else "_" + backbone.split("_")[0]


def _ld(exp, v, arm, backbone):
    f = os.path.join(R, f"{exp}_{v['suite']}_{v['index']}_{arm.replace('@', '_')}_seed{SEED}{_bsfx(backbone)}.json")
    return ({int(e["trial"]): int(bool(e["success"])) for t_ in json.load(open(f))["tasks"].values() for e in t_["episodes"]}
            if os.path.exists(f) else None)


def _launch(exp, k, v, arms, trials, fresh, backbone):
    cmd = [PY, "experiments/launch_pack.py", "--tasks", str(v["index"]), "--arms", ",".join(arms), "--experiment", exp, "--first",
           "0", "--trials", str(trials), "--seed", str(SEED), "--bind", "track", "--partition", "overcap", "--per", "4", "--suite",
           v["suite"], "--fresh", "--fresh-dir", fresh]
    cmd += ["--backbone", backbone] if backbone != "pi05_libero" else []
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-60:]


def ours_sources(v, pl):
    a, g = pfield(v, pl).split("@", 1)[1].split(".")
    return int(a), int(g)


def g10_choose(stage, backbone):
    """-> {task: chosen source} after stage 1 (object source, our goal source fixed) or 2 (goal source, chosen object source
    fixed): the most practice successes on set a (5 episodes), ties to our rule's source, then the lowest index."""
    pl = json.load(open(PLAN))
    pool = source_pool()
    out = {}
    for k, v in pl["tasks"].items():
        oa, og = ours_sources(v, pl)
        if stage == 1:
            arms = {a: f"pfield@{a}.{og}" for a in pool}
            pref = oa
        else:
            a1 = g10_choose(1, backbone)[k]
            arms = {g: f"pfield@{a1}.{g}" for g in pool}
            pref = og
        sc = {c: _ld(f"g10_s{stage}_{k.split(':')[0]}", v, arm, backbone) for c, arm in arms.items()}
        if None in sc.values():
            out[k] = None
            continue
        out[k] = max(pool, key=lambda c: (sum(sc[c].values()), c == pref, -c))
    return out


def g10_launch(stage, backbone):
    pl = json.load(open(PLAN))
    pool = source_pool()
    s1 = g10_choose(1, backbone) if stage == 2 else None
    for k, v in pl["tasks"].items():
        oa, og = ours_sources(v, pl)
        arms = [f"pfield@{a}.{og}" for a in pool] if stage == 1 else [f"pfield@{s1[k]}.{g}" for g in pool]
        print(k, _launch(f"g10_s{stage}_{k.split(':')[0]}", k, v, arms, 5, "fresh_ood_a", backbone), flush=True)


def g10_test(backbone):
    """The chosen pair: acceptance on set a (10 episodes; the policy's are recorded in g4c_accept / g4_accept) and test on set c."""
    pl = json.load(open(PLAN))
    s2 = g10_choose(2, backbone)
    s1 = g10_choose(1, backbone)
    for k, v in pl["tasks"].items():
        arm = f"pfield@{s1[k]}.{s2[k]}"
        sh = k.split(":")[0]
        print(k, arm, _launch(f"g10_acc_{sh}", k, v, [arm], 10, "fresh_ood_a", backbone),
              _launch(f"g10_main_{sh}", k, v, [arm], 20, "fresh_ood_c", backbone), flush=True)


def g10_read(backbone, record):
    import e1_read as E1
    pl = json.load(open(PLAN))
    short = backbone.split("_")[0]
    s1, s2 = g10_choose(1, backbone), g10_choose(2, backbone)
    acc_zs_exp = "g4c_accept" if backbone != "pi05_libero" else "g4_accept"
    li_exp_sfx = short if backbone != "pi05_libero" else "pi05"
    x, missing, kept, chosen = {}, [], {}, {}
    for k, v in pl["tasks"].items():
        sh = k.split(":")[0]
        arm = f"pfield@{s1[k]}.{s2[k]}"
        chosen[k] = arm
        la = li_arms(v, pl)
        main_zs = "g4c_main" if backbone != "pi05_libero" else "g4b_main"
        z = dict(sel=_ld(f"g10_main_{sh}", v, arm, backbone), sel_acc_a=_ld(f"g10_acc_{sh}", v, arm, backbone),
                 zs_a=_ld(f"{acc_zs_exp}_{sh}", v, "zs", backbone), zs=_ld(f"{main_zs}_{sh}", v, "zs", backbone),
                 li_grasp=_ld(f"g4d_main_{sh}", v, la["li_grasp"], backbone), li_fixed=_ld(f"g4d_main_{sh}", v, la["li_fixed"], backbone),
                 ours=_ld(f"{main_zs}_{sh}", v, pfield(v, pl), backbone))
        if None in z.values():
            missing.append(k)
            continue
        kept[k] = "sel" if sum(z["sel_acc_a"].values()) > sum(z["zs_a"].values()) else "zs"
        z["kept"] = z["sel"] if kept[k] == "sel" else z["zs"]
        x[k] = z
    C = E1.compare
    res = dict(backbone=backbone, missing=missing, chosen=chosen, kept={a: sum(v == a for v in kept.values()) for a in ("sel", "zs")},
               same_as_li={k: (s1[k], s2[k]) == tuple(int(c) for c in li_arms(pl["tasks"][k], pl)["li_grasp"].split("@")[1].split("."))
                           for k in chosen},
               sel_vs_li_grasp=C({k: {"x": z["sel"], "y": z["li_grasp"]} for k, z in x.items()}),
               sel_vs_li_fixed=C({k: {"x": z["sel"], "y": z["li_fixed"]} for k, z in x.items()}),
               sel_vs_ours=C({k: {"x": z["sel"], "y": z["ours"]} for k, z in x.items()}),
               sel_vs_zs=C({k: {"x": z["sel"], "y": z["zs"]} for k, z in x.items()}),
               kept_vs_zs=C({k: {"x": z["kept"], "y": z["zs"]} for k, z in x.items()}),
               per_task={k: {a: int(sum(v.values())) for a, v in z.items()} for k, z in x.items()})
    res["sel_vs_li_grasp"]["noninferior"] = bool(res["sel_vs_li_grasp"]["diff"]["lo"] > -0.05)
    for n_ in ("sel_vs_li_grasp", "sel_vs_li_fixed", "sel_vs_ours", "sel_vs_zs", "kept_vs_zs"):
        h = res[n_]
        print(f"[g10 {short}] {n_:16} {h['x']} vs {h['y']} of {h['n']}, discordant {h['discordant']}, sign p {h['sign_p']:.3g}, "
              f"diff {h['diff']['diff']:+.3f} [{h['diff']['lo']:+.3f}, {h['diff']['hi']:+.3f}]", flush=True)
    print(f"[g10 {short}] same pair as Li on {sum(res['same_as_li'].values())} of {len(chosen)} | kept {res['kept']} | missing {missing}",
          flush=True)
    json.dump(res, open(os.path.join(R, f"g10_read_{short}.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("g4_ood", bench="libero_ood", tag=f"read_g10_{short}", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=dict(backbone=backbone),
                            script="experiments/g4_ood.py", command=" ".join(sys.argv)))


def diagnose(record):
    """Why the library adds nothing here: each prototype's flow loss on its first logged batch and its last three (ledger c0_fit), G4's
    prototypes (pi0.5's own training demonstrations) against E3's (LIBERO-90 demonstrations the policy never saw)."""
    import glob
    import numpy as np

    def curve(n):
        fs = sorted(glob.glob(os.path.join(ROOT, "ledger", "c0_fit", f"*joint_{n}__*.json")))
        c = json.load(open(fs[-1]))["metrics"]["curve"]
        loss = c["loss"] if isinstance(c, dict) else c
        return float(loss[0]), float(np.mean(loss[-3:]))                    # 11 logged batches: the first, the mean of the last 3
    e3 = json.load(open(os.path.join(R, "e3_plan.json")))
    names = {"g4": sorted(json.load(open(PLAN))["protos"]),
             "e3": sorted({v["object_proto"] for v in e3.values()} | {v["goal_proto"] for v in e3.values()})}
    res = {}
    for k, ns in names.items():
        c = [curve(n) for n in ns]
        res[k] = dict(n=len(c), first=float(np.mean([a for a, _ in c])), last=float(np.mean([b for _, b in c])))
    print(json.dumps(res, indent=1), flush=True)
    if record:
        from lifelong_ood import ledger
        print(ledger.record("g4_ood", bench="libero_ood", tag="diagnosis", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/g4_ood.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    BKC = sys.argv[sys.argv.index("--backbone") + 1] if "--backbone" in sys.argv else None
    if "--accept-c" in sys.argv:
        launch_c(BKC, "accept")
    if "--test-c" in sys.argv:
        launch_c(BKC, "main")
    if "--read-c" in sys.argv:
        read_c(BKC, "--record" in sys.argv)
    if "--g10-s1" in sys.argv:
        g10_launch(1, BKC or "pi05_libero")
    if "--g10-s2" in sys.argv:
        g10_launch(2, BKC or "pi05_libero")
    if "--g10-test" in sys.argv:
        g10_test(BKC or "pi05_libero")
    if "--g10-read" in sys.argv:
        g10_read(BKC or "pi05_libero", "--record" in sys.argv)
    if "--test-d" in sys.argv:
        launch_d(BKC or "pi05_libero")
    if "--read-d" in sys.argv:
        read_d(BKC or "pi05_libero", "--record" in sys.argv)
    if "--test-b" in sys.argv:
        launch_b()
    if "--read-b" in sys.argv:
        read_b("--record" in sys.argv)
    if "--screen" in sys.argv:
        screen("--launch" in sys.argv)
    if "--diagnose" in sys.argv:
        diagnose("--record" in sys.argv)
    if "--plan" in sys.argv:
        plan()
    if "--fit" in sys.argv:
        fit()
    if "--accept" in sys.argv:
        launch("accept", 10, "fresh_ood_a")
    if "--test" in sys.argv:
        launch("main", 20, "fresh_ood_b")
    if "--read" in sys.argv:
        read("--record" in sys.argv)
