"""INTERPRETATION 7: the anatomy of a composed step, from recorded end-effector trajectories (exploratory).

Ten E3 tasks chosen to show each mechanism (wrong-object tasks 53, 52, 48; the drawer-habit task 25; the three black-bowl
instance tasks 12, 13, 14; goals plate / tray / cabinet in 71, 58, 43), five start states of E3's set b at noise seed 13, arms:
the policy alone (zs), object unit only (fac@O.x), goal unit only (fac@x.G), both (fac@O.G), and on the instance tasks the type
prototype in place of the child (fac@T.G).  Episodes recorded with states, actions and every movable object's pose
(experiments/launch_pack.py --record --light --all-objects; experiment i7_anat).

Figures (figures/robot/, copied to the site media):
  i7_paths.png      top-down end-effector paths per arm from one start state on four showcase tasks: approach solid until the
                    policy's own grasp, transport dashed; the named object starred, the other objects grey.
  i7_centred.png    the low-dimensional view: distance from the end effector to the NAMED object over the approach (time
                    normalised to the grasp), and from the grasped object to the GOAL over the transport, median and quartiles per
                    arm over all tasks and episodes.
  i7_instances.png  the hierarchy: on the three black-bowl tasks, which bowl each unit grasps first (instance child vs type).
Metrics (ledger i7_anatomy tag read): per arm, the share of episodes whose first grasp takes the named object, and successes.

    $PY experiments/i7_anatomy.py [--record]
"""
import glob
import json
import os
import re
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
R = os.path.join(ROOT, "results", "robot")
FIG = os.path.join(ROOT, "figures", "robot")
TASKS = ["53", "52", "48", "25", "12", "13", "14", "71", "58", "43"]
SHOW = ["53", "25", "13", "58"]
COL = {"zs": "#6b7280", "obj": "#1d5e86", "goal": "#a35d2b", "both": "#2f7d4f", "type": "#8b5cf6"}
LABEL = {"zs": "policy alone", "obj": "object unit only", "goal": "goal unit only", "both": "object + goal units",
         "type": "type prototype + goal"}


def arms(v):
    o, g, ty = v["object_proto"], v["goal_proto"], v["type_proto"]
    a = {"zs": "zs", "both": f"fac@{o}.{g}", "obj": f"fac@{o}.x", "goal": f"fac@x.{g}"}
    if v["instance_key"]:
        a["type"] = f"fac@{ty}.{g}"
    return a


def episodes(t, arm):
    """-> [(trial, success, rec dict)] for one task and arm."""
    tag = arm.replace("@", "_") + "_seed13"
    js = os.path.join(R, f"i7_anat_{t}_{tag}.json")
    if not os.path.exists(js):
        return []
    succ = {int(e["trial"]): bool(e["success"]) for v in json.load(open(js))["tasks"].values() for e in v["episodes"]}
    out = []
    for f in sorted(glob.glob(os.path.join(R, f"i7_anat_{t}_{tag}_rec", f"task{t}_trial*.npz"))):
        k = int(re.search(r"trial(\d+)", f).group(1))
        z = np.load(f, allow_pickle=True)
        out.append((k, succ.get(k, False), {n: z[n] for n in z.files}))
    return out


def grasp_index(rec, run=15):
    """The first step from which the gripper command stays closed for `run` steps (LIBERO: +1 closes), else None."""
    c = np.asarray(rec["action"])[:, 6] > 0
    for i in range(len(c) - run):
        if c[i:i + run].all():
            return i
    return None


def release_index(rec, g):
    c = np.asarray(rec["action"])[:, 6] > 0
    for i in range(g, len(c)):
        if not c[i]:
            return i
    return len(c)


def names_of(rec):
    return [str(x) for x in rec["obj_names"]]


def target_names(task):
    """(named object instance, goal object instance) from the task's BDDL objects of interest."""
    it = list(task.interest)
    return it[0], (it[-1] if len(it) > 1 else None)


def nearest(rec, i):
    p = np.asarray(rec["state"])[i, :3]
    op = np.asarray(rec["obj_pos"])[i]
    d = np.linalg.norm(op - p[None], axis=-1)
    return names_of(rec)[int(np.argmin(d))]


def main(record):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lifelong_ood.policy import factors as F
    plan = json.load(open(os.path.join(R, "e3_plan.json")))
    tasks = {t.index: t for t in F.load(("libero_90",))}
    os.makedirs(FIG, exist_ok=True)
    data, metrics = {}, {}
    for t in TASKS:
        task = tasks[int(t)]
        obj, goal = target_names(task)
        data[t] = {k: episodes(t, a) for k, a in arms(plan[t]).items()}
        for k, eps in data[t].items():
            m = metrics.setdefault(k, dict(episodes=0, successes=0, grasped=0, named_first=0))
            for _, ok, rec in eps:
                m["episodes"] += 1
                m["successes"] += int(ok)
                g = grasp_index(rec)
                if g is not None and "obj_pos" in rec:
                    m["grasped"] += 1
                    m["named_first"] += int(nearest(rec, g) == obj)
    for m in metrics.values():
        m["named_first_share"] = m["named_first"] / max(1, m["episodes"])

    # Figure 1: top-down paths on the showcase tasks, one start state
    fig, axes = plt.subplots(1, len(SHOW), figsize=(4.2 * len(SHOW), 4.4))
    for ax, t in zip(axes, SHOW):
        task = tasks[int(t)]
        obj, goal = target_names(task)
        trial0 = None
        for k, eps in data[t].items():
            ep = next((e for e in eps if e[0] == 0), None)
            if ep is None:
                continue
            _, ok, rec = ep
            if trial0 is None:
                trial0 = rec
            s = np.asarray(rec["state"])[:, :3]
            g = grasp_index(rec)
            g = len(s) if g is None else g
            r_ = release_index(rec, g) if g < len(s) else len(s)
            ax.plot(s[:g + 1, 0], s[:g + 1, 1], "-", color=COL[k], lw=1.8, label=f"{LABEL[k]} ({'success' if ok else 'fail'})")
            if g < len(s):
                ax.plot(s[g:r_, 0], s[g:r_, 1], "--", color=COL[k], lw=1.5)
                ax.plot(s[g, 0], s[g, 1], "o", color=COL[k], ms=6)
        if trial0 is not None and "obj_pos" in trial0:
            op0 = np.asarray(trial0["obj_pos"])[0]
            for n, p in zip(names_of(trial0), op0):
                star = n == obj
                isgoal = n == goal
                ax.plot(p[0], p[1], "*" if star else ("s" if isgoal else "x"), color="#111827" if (star or isgoal) else "#9ca3af",
                        ms=13 if star else 8)
                ax.annotate(re.sub(r"_\d+$", "", n).replace("_", " "), (p[0], p[1]), fontsize=6.5,
                            color="#111827" if (star or isgoal) else "#6b7280", xytext=(3, 3), textcoords="offset points")
        ax.set_title(task.language, fontsize=8.5)
        ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    h, l_ = [], []
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            key = ll.split(" (")[0]
            if key not in [x.split(" (")[0] for x in l_]:
                h.append(hh); l_.append(key)
    fig.legend(h, l_, loc="lower center", ncol=len(l_), fontsize=8, frameon=False)
    fig.suptitle("End-effector paths from one start state: approach solid until the policy's own grasp (dot), transport dashed",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(os.path.join(FIG, "i7_paths.png"), dpi=150)
    plt.close(fig)

    # Figure 2: concept-centred low-dimensional view
    grid = np.linspace(0, 1, 41)
    curves = {k: dict(app=[], tr=[]) for k in COL}
    for t in TASKS:
        task = tasks[int(t)]
        obj, goal = target_names(task)
        for k, eps in data[t].items():
            for _, ok, rec in eps:
                if "obj_pos" not in rec:
                    continue
                nm = names_of(rec)
                s = np.asarray(rec["state"])[:, :3]
                op = np.asarray(rec["obj_pos"])
                g = grasp_index(rec)
                end = len(s) if g is None else g + 1
                if obj in nm:
                    d = np.linalg.norm(s[:end] - op[:end, nm.index(obj)], axis=-1)
                    curves[k]["app"].append(np.interp(grid, np.linspace(0, 1, len(d)), d))
                if g is not None and goal in nm:
                    r_ = release_index(rec, g)
                    held = op[g:r_ + 1, nm.index(obj)] if obj in nm else s[g:r_ + 1]
                    d = np.linalg.norm(held - op[g:r_ + 1, nm.index(goal)], axis=-1)
                    if len(d) > 1:
                        curves[k]["tr"].append(np.interp(grid, np.linspace(0, 1, len(d)), d))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, ph, title, xl in ((axes[0], "app", "approach: end effector to the NAMED object", "start -> the policy's grasp"),
                              (axes[1], "tr", "transport: grasped object to the GOAL", "grasp -> release")):
        for k in COL:
            c = np.array(curves[k][ph])
            if len(c) < 2:
                continue
            q1, med, q3 = np.percentile(c, [25, 50, 75], axis=0)
            ax.plot(grid, med, color=COL[k], lw=2, label=f"{LABEL[k]} (n={len(c)})")
            ax.fill_between(grid, q1, q3, color=COL[k], alpha=0.15)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xl, fontsize=8)
        ax.set_ylabel("distance (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "i7_centred.png"), dpi=150)
    plt.close(fig)

    # Figure 3: the hierarchy -- which bowl each unit grasps first on the instance tasks
    inst = [t for t in TASKS if plan[t]["instance_key"]]
    fig, axes = plt.subplots(1, len(inst), figsize=(3.6 * len(inst), 3.2), sharey=True)
    first = {}
    for ax, t in zip(np.atleast_1d(axes), inst):
        task = tasks[int(t)]
        obj, _ = target_names(task)
        rows = {}
        for k in ("zs", "type", "both"):
            for _, ok, rec in data[t].get(k, []):
                g = grasp_index(rec)
                n = "no grasp" if g is None or "obj_pos" not in rec else nearest(rec, g)
                rows.setdefault(k, {}).setdefault(n, 0)
                rows[k][n] += 1
        first[t] = rows
        bowls = sorted({n for r in rows.values() for n in r if "bowl" in n or n == "no grasp"} | {obj})
        x = np.arange(len(bowls))
        w = 0.27
        for j, k in enumerate(("zs", "type", "both")):
            ax.bar(x + (j - 1) * w, [rows.get(k, {}).get(b, 0) for b in bowls], w, color=COL[k],
                   label={"zs": "policy alone", "type": "type prototype", "both": "instance child"}[k])
        ax.set_xticks(x)
        ax.set_xticklabels([("NAMED " if b == obj else "") + re.sub(r"akita_black_bowl_", "bowl ", b) for b in bowls], fontsize=7,
                           rotation=20)
        ax.set_title(plan[t]["instance_key"].split("|")[-1] + " black bowl (task " + t + ")", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("episodes: first grasp", fontsize=8)
    np.atleast_1d(axes)[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "i7_instances.png"), dpi=150)
    plt.close(fig)

    media = os.path.join(R, "site", "media")
    for f in ("i7_paths.png", "i7_centred.png", "i7_instances.png"):
        shutil.copy(os.path.join(FIG, f), os.path.join(media, f))
    res = dict(metrics=metrics, first_grasp_instances=first,
               centred_medians={k: {ph: (np.median(np.array(v[ph]), axis=0)[[0, 20, 40]].round(3).tolist() if len(v[ph]) else None)
                                    for ph in ("app", "tr")} for k, v in curves.items()})
    print(json.dumps(res["metrics"], indent=1), flush=True)
    print(json.dumps(res["first_grasp_instances"], indent=1), flush=True)
    json.dump(res, open(os.path.join(R, "i7_anatomy.json"), "w"), indent=1)
    if record:
        from lifelong_ood import ledger
        print(ledger.record("i7_anatomy", bench="libero_90", tag="read", seed=13, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=dict(tasks=TASKS),
                            script="experiments/i7_anatomy.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    main("--record" in sys.argv)
