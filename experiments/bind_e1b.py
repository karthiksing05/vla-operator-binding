"""PLAN_BIND E1b (docs/PLAN_E12.md section 6; pre-registration experiments/prereg/pbind_e1b.py, ledger m3_prereg tag pbind_e1b):
what predicts the WRONG OBJECT?  E1's object-role units (results/bind/e1.json: zs wrong_object failures, the bound object = the
filler), minus units with a single candidate (no choice).  CPU only; no policy.

Accounts (each predicts one candidate among the dev task's movable objects minus the named one; ties fractional):
  VIS  visual similarity: the candidate whose type's SigLIP prototype (results/rb4/protos.npz, OF D3 / R4) is most cosine-similar
       to the named object's type prototype (undefined if the named type has no prototype; candidates without one never win)
  FREQ global training exposure: the candidate whose type is the object-role filler of the most training cells
  SRC  source-bound slot (2609.39971): the candidate whose start xy (the zs episode's own start_xy) is nearest a training grasp source
       of similar scenes: the start xy of the object-role fillers of the training cells of the same scene (else the same scene family
       KITCHEN / LIVING_ROOM / STUDY), from LIBERO's init states 0-4 of each training cell (stage --sources, LIBERO env)
Baseline (PRIMARY, as E1): frequency-matched, P(filler) proportional to 1 + the global training count of its type key; uniform
descriptive.  An account PREDICTS the wrong object above baseline if the two-level task-cluster bootstrap 95% CI of
mean(hit - P_freq) has lo > 0.  Units where an account is undefined are excluded from that account (counted).

    --sources    (LIBERO env, CPU)  -> results/bind/e1b_sources.json
    --read [--record] [--dry]  (chunks env)  -> results/bind/e1b.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
OUT = os.path.join(ROOT, "results", "bind")
N_INIT = 5


def family(scene):
    return re.sub(r"_SCENE\d+$", "", scene or "")


def sources():
    import pathlib
    import m1_compose_create as M1
    from lifelong_ood import l90c
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    tt = M1.task_table()
    bd = benchmark.get_benchmark_dict()
    out = {}
    for tid in l90c.load_split()["train"]:
        t = tt[tid]
        objs = sorted({str(a[1]) for a in t.goal if a[0] in ("In", "On") and len(a) >= 3})
        if not objs:
            continue
        suite_name, i = tid.split(":")
        suite = bd[suite_name]()
        task = suite.get_task(int(i))
        init = suite.get_task_init_states(int(i))
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = ControlEnv(bddl_file_name=str(bddl), use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False)
        env.seed(0)
        env.reset()
        pts = {o: [] for o in objs}
        for k in range(min(N_INIT, len(init))):
            env.set_init_state(init[k])
            for o in objs:
                pts[o].append([float(v) for v in env.env.object_states_dict[o].get_geom_state()["pos"][:2]])
        out[tid] = dict(scene=t.scene, objects=pts)
        env.close()
        print(f"[bind e1b sources] {tid} ({t.scene}): {objs}", flush=True)
    json.dump(out, open(os.path.join(OUT, "e1b_sources.json"), "w", encoding="utf-8"), indent=1)


def read(record=False, dry=False):
    import d4_read
    import m1_compose_create as M1
    from lifelong_ood import l90c
    import bind_e1 as B1
    tt = M1.task_table()
    train = [tt[x] for x in l90c.load_split()["train"]]
    gcount = Counter()
    for t in train:
        for x in set(B1.roles(t)[0]):
            gcount[B1.tkey(x, t)] += 1
    pz = np.load(os.path.join(ROOT, "results", "rb4", "protos.npz"), allow_pickle=True)
    proto = {str(c): v / np.linalg.norm(v) for c, v in zip(pz["cats"], pz["sig"])}
    src = json.load(open(os.path.join(OUT, "e1b_sources.json"), encoding="utf-8"))
    e1 = json.load(open(os.path.join(OUT, "e1.json"), encoding="utf-8"))
    units, excl = [], Counter()
    starts = {}
    for u in e1["units"]:
        if u["role"] != "object":
            continue
        t = tt[u["tid"]]
        if u["tid"] not in starts:
            eps = json.load(open(os.path.join(ROOT, "results", "fm", "types", u["tid"].replace(":", "_") + ".json"), encoding="utf-8"))
            starts[u["tid"]] = {int(e["trial"]): e.get("start_xy", {}) for e in eps["arms"]["zs"]["episodes"]}
        named = B1.roles(t)[0]
        cands = sorted(set(t.objects) - set(named))
        if len(cands) < 2:
            excl["single_candidate"] += 1
            continue
        filler = u["filler"]
        w = np.array([1.0 + gcount[B1.tkey(c, t)] for c in cands])
        p_freq = float(w[cands.index(filler)] / w.sum()) if filler in cands else 0.0
        # VIS
        nt = B1.inst_type(named[0])
        vis = None
        if nt in proto:
            sims = {c: float(proto[B1.inst_type(c)] @ proto[nt]) for c in cands if B1.inst_type(c) in proto}
            vis = B1.argmax_set(sims) if sims else None
        # FREQ
        fq = B1.argmax_set({c: gcount[B1.tkey(c, t)] for c in cands})
        # SRC
        same = [v for k, v in src.items() if v["scene"] == t.scene]
        fam = same or [v for k, v in src.items() if family(v["scene"]) == family(t.scene)]
        pts = np.array([p for v in fam for ps in v["objects"].values() for p in ps]) if fam else np.zeros((0, 2))
        sx = starts[u["tid"]].get(int(u["trial"]), {})
        srcp = None
        if len(pts) and all(c in sx for c in cands):
            dmin = {c: -float(np.min(np.linalg.norm(pts - np.asarray(sx[c])[None], axis=1))) for c in cands}
            srcp = B1.argmax_set(dmin)
        units.append(dict(tid=u["tid"], trial=u["trial"], filler=filler, n_cands=len(cands), p_freq=p_freq, p_unif=1.0 / len(cands),
                          vis=vis, freq=fq, src=srcp, src_level=("scene" if same else "family" if fam else None),
                          hit_vis=None if vis is None else float(vis.get(filler, 0.0)), hit_freq=float(fq.get(filler, 0.0)),
                          hit_src=None if srcp is None else float(srcp.get(filler, 0.0))))
    res = dict(n_units=len(units), excluded=dict(excl), by_task=dict(Counter(u["tid"] for u in units)),
               defined=dict(vis=sum(u["hit_vis"] is not None for u in units), src=sum(u["hit_src"] is not None for u in units)))
    if dry:
        print(f"[bind e1b dry] {json.dumps(res)}", flush=True)
        return

    def rate(key, base="p_freq"):
        per = {}
        for u in units:
            if u[key] is None:
                continue
            per.setdefault(u["tid"], []).append(u[key] - u[base])
        return d4_read.rate(per)
    for acc in ("vis", "freq", "src"):
        res[acc] = dict(minus_freq=rate("hit_" + acc), minus_unif=rate("hit_" + acc, "p_unif"))
        r = res[acc]["minus_freq"]
        res[acc]["verdict"] = "ABOVE baseline" if r["lo"] is not None and r["lo"] > 0 else "not above baseline"
    res["units"] = units
    json.dump(res, open(os.path.join(OUT, "e1b.json"), "w", encoding="utf-8"), indent=1, default=float)
    for k in ("n_units", "excluded", "by_task", "defined"):
        print(f"[bind e1b] {k}: {json.dumps(res[k])}", flush=True)
    for acc in ("vis", "freq", "src"):
        r, q = res[acc]["minus_freq"], res[acc]["minus_unif"]
        print(f"[bind e1b] {acc}: hit - P_freq {r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] (n {r['n']}, {r['tasks']} tasks); "
              f"hit - uniform {q['mean']:+.3f}; {res[acc]['verdict']}", flush=True)
    for tid in sorted({u["tid"] for u in units}):
        us = [u for u in units if u["tid"] == tid]
        print(f"[bind e1b]   {tid} (n {len(us)}): fillers {dict(Counter(u['filler'] for u in us))}; VIS {us[0]['vis']}; FREQ {us[0]['freq']}; "
              f"SRC ({us[0]['src_level']}) {dict(Counter(max(u['src'], key=u['src'].get) for u in us if u['src']))}", flush=True)
    if record:
        from lifelong_ood import ledger
        print(ledger.record("bind_e1", bench="libero_90_c", tag="e1b", seed=0, metrics={k: v for k, v in res.items() if k != "units"},
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/bind_e1b.py", command="python experiments/bind_e1b.py --read --record"))


def main():
    ap = argparse.ArgumentParser()
    for s in ("sources", "read", "record", "dry"):
        ap.add_argument(f"--{s}", action="store_true")
    a = ap.parse_args()
    if a.sources:
        sources()
    if a.read:
        read(a.record, a.dry)


if __name__ == "__main__":
    main()
