"""S1 (docs/PLAN_SOLVE.md): dump the training / dev tasks (language, scene, BDDL goal, parse) -> results/bind/s1/tasks.json."""
import json, os, sys
R = "/coc/flash10/ksingara3/vla-operator-binding"
sys.path[:0] = [R, R + "/experiments"]
import m1_compose_create as M1
from lifelong_ood import l90c
tt = M1.task_table()
sp = l90c.load_split()
print("split keys", list(sp.keys()))
dev = json.load(open(R + "/results/bind/e1_scene.json"))
out = {}
for grp, tids in (("train", sp["train"]), ("dev", list(dev))):
    for tid in tids:
        t = tt[tid]
        p = M1.parse(t.language)
        out[tid] = dict(set=grp, lang=t.language, scene=t.scene, objects=t.objects, fixtures=t.fixtures, goal=t.goal, slots=t.slots,
                        steps=p["steps"])
        print(grp, tid, t.scene, "|", t.language, "|", t.goal, "|", [(s["head"], s["landmark"], s["goal"]) for s in p["steps"]])
json.dump(out, open(os.path.join(R, "results/bind/s1/tasks.json"), "w"), indent=1, default=str)
