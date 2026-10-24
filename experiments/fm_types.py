"""FAILURE MAP, stage A (docs/PLAN_FM.md, docs/PLAN_STORY.md section 2): type every recorded episode of M1's (and the failure map's
own few-shot arms') rollouts on LIBERO-90-C's 29 dev tasks, from the simulator state alone.  No new rollout.

Per episode the rollout recorded (experiments/libero_client.py --record --light --objects --all-objects --bind track) the 8-D
EEF/finger state, the actions, every movable object's pos + quat at every policy step, and `bound`: the first movable object
moved > 3 cm while the gripper was commanded closed.  The goal and its regions are the task's own BDDL (LIBERO's predicates,
evaluated by LIBERO's own code): a CPU LIBERO env (no renderer) is set to the episode's start state (results/robot/fresh_m1), every
movable object is put at its LAST recorded pose, and the goal atoms of the named object are evaluated.

Types of a FAILED episode (rule fixed before any aggregate; the order is the precedence):
  fixture       the task's goal has no In / On atom (open / close a drawer, turn on the stove): articulated fixtures are read
                only by success, so these failures are not typed
  no_commit     no movable object was ever bound
  wrong_object  the bound object is not the instruction's named object (an In / On goal atom's first argument);
                subtype wrong_instance when it is another instance of the named object's type
  wrong_goal    the named object was bound, and at its final pose it satisfies an In / On atom with ANOTHER BDDL region or
                object of the scene (not an init region, not a table, not its own target) -- or it was carried >= 10 cm (xy)
                from where it was bound while getting less than half way (xy) towards its target
  motion        the named object was bound and either got at least half way towards the target, or was not carried 10 cm
                (grasp lost, dropped, placement failed); subtype in_region when its own goal atom holds at the final pose (the
                episode failed on another atom, e.g. the articulated step of a two-step task, or on the last step)

Axes per task (docs/PLAN_STORY.md 2): CD bin (split.json dev bins 0.1 / 0.3 / 0.5; the part set is "part"), task CD
(dev_per_task), part vs combination, scene familiar (at least one training cell of split.json "train" in the task's scene,
factors.py scene) / new, instance (the instruction names a qualifier left / right / front / back / middle AND the BDDL holds >= 2
instances of a moved or target object's type: the qualifier picks an instance; caddy compartments and "right of the caddy" are
regions, not instances).

    sbatch ... --wrap "$LPY experiments/fm_types.py --probe libero_90:14 zs 0+1+2"     print the traces of a few episodes
    sbatch ... --wrap "$LPY experiments/fm_types.py --type [--arms zs+create] [--shard 0/4]"   type every episode -> results/fm/types/
    (the aggregate and the tests are experiments/fm_fewshot.py --read)
$LPY = /coc/flash10/ksingara3/robot/openpi/examples/libero/.venv/bin/python with PYTHONPATH=.../third_party/libero (LIBERO's env).
"""
import argparse
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import m1_compose_create as M1                                          # noqa: E402

OUT = os.path.join(ROOT, "results", "fm")
TYPES = ("no_commit", "wrong_object", "wrong_goal", "motion", "fixture")
CARRY_MIN, HALF = 0.10, 0.5                                             # the wrong_goal / motion split (fixed before any aggregate)
SETTLE = 10                                                             # control steps of physics after placing the final poses
HOLD = [0.0] * 6 + [-1.0]                                               # zero delta, gripper open
QUALS = ("left", "right", "front", "back", "middle")


def slug(tid):
    return tid.replace(":", "_")


# ── axes ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def axes():
    """-> {tid: dict(bin, cd, kind, scene, scene_familiar, instance, named, atoms)} for the 29 dev tasks."""
    from lifelong_ood import l90c
    s = l90c.load_split()
    tt = M1.task_table()
    train_scenes = {tt[t].scene for t in s["train"]}
    out = {}
    for tid, b in M1.dev_tasks():
        t = tt[tid]
        inst = dict(t.objects, **t.fixtures)
        cnt = {}
        for n, ty in t.objects.items():
            cnt[ty] = cnt.get(ty, 0) + 1
        place = [a for a in t.goal if a[0] in ("In", "On") and len(a) == 3]
        roles = set()
        for a in place:
            roles.add(inst.get(a[1], a[1]))
            if a[2] in inst:
                roles.add(inst[a[2]])
        quals = [q for q in QUALS if re.search(rf"\b{q}\b", t.language)]
        out[tid] = dict(bin=b, cd=s["dev_per_task"].get(tid, {}).get("cd"), kind="part" if b == "part" else "combination",
                        scene=t.scene, scene_familiar=t.scene in train_scenes,
                        instance=bool(quals) and any(cnt.get(r, 0) >= 2 for r in roles), quals=quals,
                        named=sorted({a[1] for a in place}), atoms=[list(a) for a in t.goal], text=t.language)
    return out


# ── episodes ─────────────────────────────────────────────────────────────────────────────────────────────────────
def episodes(tid, arm, modes):
    """-> (roll json path, rec dir, [episode dicts]) of one task x arm (M1's rollout naming; the failure map's own arms under
    results/fm/roll), or None if not rolled."""
    mode = modes[arm]
    if arm in M1.ARMS:
        f = M1.roll_out("m1_main", tid, mode)
    else:
        import fm_fewshot as FF
        f = FF.roll_out(tid, mode)
    if not os.path.exists(f):
        return None
    z = json.load(open(f, encoding="utf-8"))
    eps = [e for t in z["tasks"].values() for e in t["episodes"]]
    return f, f[:-5] + "_rec", eps


# ── the LIBERO env (CPU, no renderer) ────────────────────────────────────────────────────────────────────────────
class Scene:
    def __init__(self, tid):
        import pathlib
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs.env_wrapper import ControlEnv
        suite, i = tid.split(":")
        task = benchmark.get_benchmark_dict()[suite]().get_task(int(i))
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        self.env = ControlEnv(bddl_file_name=str(bddl), use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False)
        self.env.seed(0)
        self.env.reset()
        self.e = self.env.env
        self.fresh = np.load(os.path.join(M1.R, M1.FRESH_DIR, f"{suite}_{i}.npy"))
        self.goal = [list(g) for g in self.e.parsed_problem["goal_state"]]

    def set(self, k, names=None, pos=None, quat=None, settle=0):
        """The start state k, then (optionally) every listed movable object at the given pose (pos xyz, quat xyzw as recorded), at
        rest, and `settle` control steps of physics with the arm holding its home pose (the last record precedes the last action:
        a bowl being lowered onto the plate is still 1 cm above it, out of contact)."""
        self.env.set_init_state(self.fresh[k])
        if names is not None:
            sim = self.e.sim
            for n, p, q in zip(names, pos, quat):
                j = self.e.objects_dict[n].joints[0]
                sim.data.set_joint_qpos(j, np.r_[np.asarray(p, np.float64), q[3], q[0], q[1], q[2]])
                sim.data.set_joint_qvel(j, np.zeros(6))
            sim.forward()
            self.e.timestep, self.e.done = 0, False
            for _ in range(settle):
                self.env.step(HOLD)

    def pred(self, name, a, b):
        try:
            return bool(self.e._eval_predicate([name, a, b]))
        except Exception:                                                # noqa: BLE001  (a pair the predicate does not define)
            return None

    def pos_of(self, name):
        st = self.e.object_states_dict[name]
        return np.array(st.get_geom_state()["pos"], np.float64)

    def alternatives(self, obj, own):
        """Every BDDL region / object of the scene a placement could land in, except init regions, tables, the object itself
        and its own targets."""
        out = []
        for n in self.e.object_states_dict:
            if n == obj or n in own or "init_region" in n or n.endswith("table") or n in ("kitchen_table", "study_table",
                                                                                           "living_room_table", "floor"):
                continue
            out.append(n)
        return out


def type_episode(sc, ax, e, rec):
    """-> dict(type, sub, ...) of one episode (successes too: their own atom at the final pose validates the typer)."""
    out = dict(trial=int(e["trial"]), success=bool(e["success"]), bound=e.get("bound"), bound_step=e.get("bound_step"),
               steps=int(e["steps"]))
    if rec is None:
        out.update(type=None, sub="no_record")
        return out
    names = [str(x) for x in rec["obj_names"]]
    P, Q = rec["obj_pos"], rec["obj_quat"]
    out["start_xy"] = {n: [round(float(P[0, j, 0]), 4), round(float(P[0, j, 1]), 4)] for j, n in enumerate(names)}
    sc.set(out["trial"], names, P[-1], Q[-1], settle=SETTLE)
    own = {}                                                            # named object -> its In / On goal atoms
    place = [a for a in sc.goal if len(a) == 3 and a[0] in ("in", "on")]
    for a in place:
        own.setdefault(a[1], []).append(a)
    out["own_atoms_final"] = {a[1] + ">" + a[2]: sc.pred(a[0], a[1], a[2]) for a in place}
    out["success_settled"] = bool(sc.env.check_success())                # every atom (articulated ones: start state, not read)
    if out["success"]:
        out.update(type="success", sub="")
        return out
    if not place:
        out.update(type="fixture", sub="bound" if out["bound"] else "")
        return out
    b = out["bound"]
    if b is None:
        out.update(type="no_commit", sub="")
        return out
    if b not in own:
        ty = lambda n: re.sub(r"_\d+$", "", n)                          # noqa: E731
        out.update(type="wrong_object", sub="wrong_instance" if any(ty(b) == ty(n) for n in own) else "other_type")
        return out
    i = names.index(b)
    t0 = min(int(out["bound_step"] or 0) + 1, len(P) - 1)
    if all(out["own_atoms_final"][a[1] + ">" + a[2]] for a in own[b]):
        out.update(type="motion", sub="in_region")
        return out
    tgt = own[b][0][2]
    pT = sc.pos_of(tgt)
    xy = P[t0:, i, :2].astype(np.float64)
    d = np.linalg.norm(xy - pT[None, :2], axis=-1)
    progress = float(1.0 - d.min() / max(d[0], 1e-6))
    carried = float(np.linalg.norm(xy - xy[0][None], axis=-1).max())
    alt = [n for n in sc.alternatives(b, {a[2] for a in own[b]})
           if any(sc.pred(p, b, n) for p in ("in", "on"))]
    out.update(progress=round(progress, 3), carried=round(carried, 3), d_bind=round(float(d[0]), 3), d_min=round(float(d.min()), 3),
               d_final=round(float(d[-1]), 3), alt=alt, target=tgt)
    if alt:
        out.update(type="wrong_goal", sub="other_region")
    elif carried >= CARRY_MIN and progress < HALF:
        out.update(type="wrong_goal", sub="carried_elsewhere")
    else:
        out.update(type="motion", sub="toward_target" if progress >= HALF else "not_carried")
    return out


def load_rec(rec_dir, tid, trial):
    f = os.path.join(rec_dir, f"task{tid.split(':')[1]}_trial{trial}.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f, allow_pickle=True)
    return {k: z[k] for k in ("obj_names", "obj_pos", "obj_quat", "state", "action")}


def arm_modes_all(tid):
    """M1's arms + the failure map's few-shot arms (experiments/fm_fewshot.py), as {arm: MODE}."""
    am = dict(M1.arm_modes(tid) or {})
    try:
        import fm_fewshot as FF
        am.update(FF.arm_modes(tid) or {})
    except ImportError:
        pass
    return am


def type_task(tid, arms, ax):
    """Type every recorded episode of one task under the given arms -> results/fm/types/<task>.json (merged with earlier arms)."""
    sc = Scene(tid)
    modes = arm_modes_all(tid)
    f = os.path.join(OUT, "types", f"{slug(tid)}.json")
    res = json.load(open(f, encoding="utf-8")) if os.path.exists(f) else dict(task=tid, axes=ax[tid], arms={})
    res["axes"] = ax[tid]
    done_mode = {}
    for arm in arms:
        if arm not in modes:
            continue
        got = episodes(tid, arm, modes)
        if got is None:
            continue
        rf, rd, eps = got
        if modes[arm] in done_mode:                                     # arms served identically share one rollout
            res["arms"][arm] = dict(res["arms"][done_mode[modes[arm]]], shared_with=done_mode[modes[arm]])
            continue
        typed = [type_episode(sc, ax[tid], e, load_rec(rd, tid, int(e["trial"]))) for e in eps]
        res["arms"][arm] = dict(roll=os.path.relpath(rf, ROOT), mode=modes[arm], episodes=typed)
        done_mode[modes[arm]] = arm
        print(f"[fm types] {tid} {arm}: " + " ".join(f"{x['trial']}:{x['type']}" for x in typed), flush=True)
    os.makedirs(os.path.dirname(f), exist_ok=True)
    tmp = f + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    os.replace(tmp, f)
    sc.env.close()


def probe(tid, arm, trials):
    """Print an episode's trace for a hand check: the start / bind / final positions of the named and bound objects, the gripper
    command over time, the goal atoms at the start and at the final pose, and the type."""
    ax = axes()
    a = ax[tid]
    print(f"[probe] {tid} '{a['text']}' axes {json.dumps({k: a[k] for k in ('bin', 'cd', 'kind', 'scene', 'scene_familiar', 'instance')})}")
    print(f"[probe] goal {a['atoms']} named {a['named']}")
    sc = Scene(tid)
    print(f"[probe] parsed goal {sc.goal}; object states {sorted(sc.e.object_states_dict)}")
    modes = arm_modes_all(tid)
    rf, rd, eps = episodes(tid, arm, modes)
    if isinstance(trials, str):                                         # "fail3": the first 3 failed episodes
        trials = [int(e["trial"]) for e in eps if not e["success"]][:int(trials[4:])]
    for e in eps:
        if int(e["trial"]) not in trials:
            continue
        rec = load_rec(rd, tid, int(e["trial"]))
        names = [str(x) for x in rec["obj_names"]]
        P = rec["obj_pos"]
        sc.set(int(e["trial"]))
        start = {n: np.round(sc.pos_of(n), 3).tolist() for n in names}
        drift = max(float(np.linalg.norm(np.array(start[n]) - P[0, j])) for j, n in enumerate(names))
        sc.set(int(e["trial"]), names, P[0], rec["obj_quat"][0])
        rt = max(float(np.linalg.norm(sc.pos_of(n) - P[0, j])) for j, n in enumerate(names))
        print(f"\n[probe] trial {e['trial']} success {e['success']} steps {e['steps']} bound {e.get('bound')} @ {e.get('bound_step')}; "
              f"init-state vs first record max drift {drift:.4f} m; set-pose round trip {rt:.2e} m")
        g = rec["action"][:, 6]
        ch = np.flatnonzero(np.diff((g > 0).astype(int))) + 1
        print(f"[probe]   gripper closes/opens at steps {ch.tolist()[:12]}; T {len(g)}")
        for n in sorted(set(a["named"]) | ({e.get("bound")} - {None})):
            j = names.index(n)
            print(f"[probe]   {n}: start {np.round(P[0, j], 3).tolist()} final {np.round(P[-1, j], 3).tolist()} "
                  f"max z {P[:, j, 2].max():.3f}")
        for t in range(0, len(P), max(1, len(P) // 8)):
            print(f"[probe]   t {t:3d} eef {np.round(rec['state'][t, :3], 3).tolist()} grip {g[t]:+.0f} "
                  + " ".join(f"{n} {np.round(P[t, names.index(n)], 3).tolist()}" for n in a["named"] if n in names))
        x = type_episode(sc, a, e, rec)
        print(f"[probe]   TYPE {json.dumps(x)}")
    sc.env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", nargs=3, metavar=("TASK", "ARM", "TRIALS"))
    ap.add_argument("--type", action="store_true")
    ap.add_argument("--arms", default="+".join(M1.ARMS))
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--axes", action="store_true")
    a = ap.parse_args()
    if a.axes:
        ax = axes()
        for k, v in ax.items():
            print(k, json.dumps({x: v[x] for x in ("bin", "cd", "kind", "scene", "scene_familiar", "instance", "quals", "named")}))
    if a.probe:
        p3 = a.probe[2]
        probe(a.probe[0], a.probe[1], p3 if p3.startswith("fail") else {int(x) for x in p3.split("+")})
    if a.type:
        ax = axes()
        i, n = map(int, a.shard.split("/"))
        tasks = list(ax) if a.tasks == "all" else a.tasks.split("+")
        for j, tid in enumerate(tasks):
            if j % n == i:
                type_task(tid, a.arms.split("+"), ax)
