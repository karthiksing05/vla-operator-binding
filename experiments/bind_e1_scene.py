"""PLAN_BIND E1 scene stage (LIBERO env, CPU, no renderer, no policy): per dev task with an In / On goal, the scene's BDDL goal
candidates for the named object (fm_types.Scene.alternatives: every region / object except init regions, tables, the object and its
own targets) with their positions, and, for every zs wrong_goal / carried_elsewhere episode, the named object's final xy (from the
recorded fresh_m1 rollout).  -> results/bind/e1_scene.json.  Read by experiments/bind_e1.py."""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import fm_types as FT                                             # noqa: E402
import m1_compose_create as M1                                    # noqa: E402

out = {}
tt = M1.task_table()
for tid, _ in M1.dev_tasks():
    t = tt[tid]
    pl = [a for a in t.goal if a[0] in ("In", "On") and len(a) >= 3]
    if not pl:
        continue
    named = str(pl[0][1])
    own = {str(a[2]) for a in pl if str(a[1]) == named}
    sc = FT.Scene(tid)
    sc.set(0)
    alts = sc.alternatives(named, own)
    pos = {n: sc.pos_of(n).tolist() for n in alts + sorted(own)}
    modes = FT.arm_modes_all(tid)
    rf, rd, eps = FT.episodes(tid, "zs", modes)
    typed = {int(e["trial"]): e for e in json.load(open(os.path.join(ROOT, "results", "fm", "types", tid.replace(":", "_") + ".json"),
                                                         encoding="utf-8"))["arms"]["zs"]["episodes"]}
    final = {}
    for k, e in typed.items():
        if e["type"] == "wrong_goal" and e.get("sub") == "carried_elsewhere":
            rec = FT.load_rec(rd, tid, k)
            names = [str(x) for x in rec["obj_names"]]
            b = e["bound"]
            final[str(k)] = np.asarray(rec["obj_pos"][-1, names.index(b)], float).tolist()
    out[tid] = dict(named=named, own=sorted(own), alternatives=alts, pos=pos, carried_final=final)
    print(f"[bind e1 scene] {tid}: {len(alts)} alternatives, {len(final)} carried_elsewhere episodes", flush=True)
    sc.env.close()
os.makedirs(os.path.join(ROOT, "results", "bind"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "results", "bind", "e1_scene.json"), "w", encoding="utf-8"), indent=1)
