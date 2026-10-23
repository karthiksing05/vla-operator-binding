"""P2 / P3 development pre-checks, step 0 (docs/PLAN_V4.md section 8): the observation index, the raw LIBERO states of the chosen
demonstrations, and every object's position at the chosen frames.

    index  (numpy; reads metadata and parquet actions)  -> results/protos/index.json
           train     40 pi05_libero training tasks x 5 evenly spaced frames of the task's first demo (physical-intelligence/libero)
           l90       25 LIBERO-90 tasks (held out from pi05_libero) x 4 frames of the IPEC demo already matched to its raw hdf5
                     (results/robot/p0_demo/<task>_match.npz)
           transfer  6 training + 6 LIBERO-90 tasks x the 10 practice init states rendered by experiments/f0b_firstframes_seg.py
                     (objects' positions already stored there)
           Each observation carries its conditions (protos.store.conditions_for); training observations never use their own
           task as a splice source (leave-task-out).
    fetch  (h5py + huggingface_hub; SLURM)  download the 40 training tasks' raw hdf5 (yifengzhu-hf/LIBERO-datasets) and match the
           chosen demo by its no-op-filtered arm actions (experiments/p0_demo_init.py's rule) -> results/protos/match/tr<k>.npz
    poses  (LIBERO venv; SLURM)  set each chosen frame's simulator state and read every object's and fixture's position
           -> results/protos/poses/<key>.npz

    python experiments/pp_prep.py index
    sbatch ... --export=ALL,SCRIPT="experiments/pp_prep.py fetch" slurm/jobs/cpu.sbatch          (chunks env)
    sbatch ... --export=ALL,ENV=libero,SCRIPT="experiments/pp_prep.py poses" slurm/jobs/robot_gpu.sbatch
"""
import json
import os
import pathlib
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results", "protos")
RAW = "/coc/flash10/ksingara3/robot/data/libero_raw"
P0 = os.path.join(ROOT, "results", "robot", "p0_demo")
F0B = os.path.join(ROOT, "results", "robot", "f0b_firstframes")
TF_TRAIN = (10, 11, 13, 21, 26, 30)        # HF task ids: bowl->plate, wine bottle->rack, cream cheese->bowl, ketchup, dressing, spatial
TF_L90 = (18, 26, 48, 52, 58, 62)          # LIBERO-90 pick-and-place tasks (q1's class a)
N_TRAIN_FRAMES, N_L90_FRAMES = 5, 4


def save_atomic(path, **arrs):
    tmp = path + f".{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)


def write_json(path, obj):
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def context(actions, state, f):
    """Phase (1 = gripper commanded closed), gripper opening (finger qpos gap), frames since the last gripper event."""
    g = np.sign(np.asarray(actions)[:, 6])
    ev = np.where(np.diff(g[: f + 1]) != 0)[0]
    since = f - (int(ev[-1]) + 1) if len(ev) else f
    return int(g[f] > 0), float(state[f][6] - state[f][7]), int(since)


def resolve_l90(d90, tid, m, text):
    """The IPEC episode a p0_demo match file was made from: the one whose 8-d state equals the file's ipec_state (older files
    store no rank, and some ranks index the text's episodes rather than the task's)."""
    ref = np.asarray(m["ipec_state"], np.float32)
    cands = list(d90.episodes_of_task(tid))
    r = int(m["ipec_rank"]) if "ipec_rank" in m.files else 0
    order = ([cands[r]] if r < len(cands) else []) + cands + list(d90.episodes_of_text(text))
    length = {int(x["episode_index"]): int(x["length"]) for x in d90.eps}
    for e in order:
        if length.get(int(e)) != len(ref):
            continue
        st = d90.load(e)["state"]
        if len(st) == len(ref) and np.abs(st - ref).max() < 1e-5:
            return int(e)
    return None


def index():
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy import factors as F
    from lifelong_ood.protos import store as S
    demos, d90 = D.Demos(), D.Demos90()
    allt = F.load()
    train_texts = {k: demos.tasks[k] for k in range(40)}
    by_lang = {t.language: t for t in allt if t.suite != "libero_90"}
    l90 = {t.index: t for t in allt if t.suite == "libero_90"}

    def conds(task, exclude, kinds="full"):
        ph = F.scene_phrases(task, allt)
        scene = [t for t in allt if t.scene == task.scene and t.tid != task.tid]
        return [c.to_json() for c in S.conditions_for(task, scene, ph, train_texts, exclude, kinds)]

    obs = []
    for k in range(40):
        t = by_lang[train_texts[k]]
        e = demos.episodes(k)[0]
        ep = demos.load(e)
        T = len(ep["actions"])
        for f in np.unique(np.linspace(0, T - 10, N_TRAIN_FRAMES).round().astype(int)):
            ph, grip, since = context(ep["actions"], ep["state"], int(f))
            obs.append(dict(key=f"tr{k:02d}_e{e}_f{int(f)}", set="train", hf_task=k, suite=t.suite, index=t.index, episode=int(e),
                            frame=int(f), T=T, language=t.language, phase=ph, grip=grip, since=since, conds=conds(t, {k})))
    have = sorted(int(m.group(1)) for m in (re.match(r"^(\d+)_match\.npz$", x) for x in os.listdir(P0)) if m)
    pick = have[::2][:25]
    for tid in pick:
        t = l90[tid]
        m = np.load(os.path.join(P0, f"{tid}_match.npz"))
        e = resolve_l90(d90, tid, m, t.language)
        if e is None:
            print(f"[pp_prep] LIBERO-90 task {tid}: no IPEC episode matches its raw match file; skipped", flush=True)
            continue
        ep = d90.load(e)
        T = len(ep["actions"])
        assert T == len(m["kept"]), (tid, T, len(m["kept"]))
        for f in np.unique(np.linspace(0, T - 10, N_L90_FRAMES).round().astype(int)):
            ph, grip, since = context(ep["actions"], ep["state"], int(f))
            obs.append(dict(key=f"l90_{tid:02d}_e{e}_f{int(f)}", set="l90", l90_task=tid, suite="libero_90", index=tid, episode=int(e),
                            frame=int(f), T=T, language=t.language, phase=ph, grip=grip, since=since, conds=conds(t, set())))
    for k in TF_TRAIN:
        t = by_lang[train_texts[k]]
        for s in range(10):
            obs.append(dict(key=f"tf_{t.suite}_{t.index}_s{s}", set="transfer", hf_task=k, suite=t.suite, index=t.index, state=s,
                            language=t.language, phase=0, grip=None, since=0, conds=conds(t, {k}, "transfer")))
    for tid in TF_L90:
        t = l90[tid]
        for s in range(10):
            obs.append(dict(key=f"tf_libero_90_{tid}_s{s}", set="transfer", l90_task=tid, suite="libero_90", index=tid, state=s,
                            language=t.language, phase=0, grip=None, since=0, conds=conds(t, set(), "transfer")))
    os.makedirs(OUT, exist_ok=True)
    write_json(os.path.join(OUT, "index.json"), dict(train_texts=train_texts, obs=obs))
    n = {s: sum(o["set"] == s for o in obs) for s in ("train", "l90", "transfer")}
    nb = sum(1 for o in obs for c in o["conds"] if c["binds"])
    nc = sum(len(o["conds"]) for o in obs)
    print(f"[pp_prep] index: {n}; conditions {nc} ({nb} spliced); phases train post-grasp "
          f"{sum(o['phase'] for o in obs if o['set'] == 'train')}/{n['train']}", flush=True)
    for o in obs[:1] + [x for x in obs if x["set"] == "l90"][:1]:
        print(o["key"], o["language"])
        for c in o["conds"]:
            print("   ", c["name"], "|", c["prompt"], "|", [(b[0], b[1]) for b in c["binds"]], "|", c["target"])


def index_path(tag=""):
    """results/protos/index.json, or index_<tag>.json for another checkpoint's index (e.g. l90c: experiments/pp_prep.py index_l90c)."""
    return os.path.join(OUT, f"index_{tag}.json" if tag else "index.json")


def _tag():
    return next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--tag=")), "")


def index_l90c():
    """The index for pi05_l90c (docs/L90C.md): its TRAINING observations are its own training tasks, its l90 observations its
    held-out DEV tasks (combination bins and the part set; the sealed third is never read: lifelong_ood.l90c.check_task).
      train    the split's suite tasks (physical-intelligence/libero; the same keys and frames as index.json, so their poses exist)
               + its LIBERO-90 training cells that have a raw match (results/robot/p0_demo/<tid>_match.npz), N_TRAIN_FRAMES frames
      l90      its dev test and dev part tasks with a raw match, N_L90_FRAMES frames
      transfer as index.json (first frames of 10 init states)
    train_texts are POSITIONAL over pi05_l90c's training instructions (pp_fields' E_instr rows; pp3's hf_task is that position).
    Splice sources (binds) stay physical-intelligence/libero task ids, restricted to the suite tasks pi05_l90c trains on.
    Each training observation carries src = "hf" | "ipec" (src_hf / l90_task: its source)."""
    from lifelong_ood import l90c
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy import factors as F
    from lifelong_ood.protos import store as S
    demos, d90 = D.Demos(), D.Demos90()
    allt = F.load()
    split = l90c.load_split()
    train = set(split["train"])
    hf_of_text = {v: k for k, v in demos.tasks.items()}
    tasks = {t.tid: t for t in allt}
    bind_src = {hf_of_text[tasks[tid].language]: tasks[tid].language for tid in sorted(train) if not tid.startswith("libero_90:")}
    order = sorted(train, key=lambda x: (x.split(":")[0], int(x.split(":")[1])))
    texts = []
    for tid in order:
        if tasks[tid].language not in texts:
            texts.append(tasks[tid].language)
    pos = {t: i for i, t in enumerate(texts)}
    l90 = {t.index: t for t in allt if t.suite == "libero_90"}

    def conds(task, exclude, kinds="full"):
        ph = F.scene_phrases(task, allt)
        scene = [t for t in allt if t.scene == task.scene and t.tid != task.tid]
        return [c.to_json() for c in S.conditions_for(task, scene, ph, bind_src, exclude, kinds)]

    base = json.load(open(index_path()))
    obs = []
    for o in base["obs"]:
        if o["set"] == "train":
            t = tasks[f"{o['suite']}:{o['index']}"]
            if t.tid in train:
                obs.append(dict(o, hf_task=pos[t.language], src="hf", src_hf=o["hf_task"], conds=conds(t, {o["hf_task"]})))
    have = sorted(int(m.group(1)) for m in (re.match(r"^(\d+)_match\.npz$", x) for x in os.listdir(P0)) if m)
    dev = [x for b in split["bins"] for x in split["dev"][b]] + list(split["dev"]["part"])
    for tid in have:
        key = f"libero_90:{tid}"
        role = "train" if key in train else "l90" if key in dev else None
        if role is None:
            continue
        l90c.check_task(key)
        t = l90[tid]
        m = np.load(os.path.join(P0, f"{tid}_match.npz"))
        e = resolve_l90(d90, tid, m, t.language)
        if e is None:
            print(f"[pp_prep] LIBERO-90 task {tid}: no IPEC episode matches its raw match file; skipped", flush=True)
            continue
        ep = d90.load(e)
        T = len(ep["actions"])
        nf = N_TRAIN_FRAMES if role == "train" else N_L90_FRAMES
        for f in np.unique(np.linspace(0, T - 10, nf).round().astype(int)):
            ph, grip, since = context(ep["actions"], ep["state"], int(f))
            o = dict(key=f"{'tr90' if role == 'train' else 'l90'}_{tid:02d}_e{e}_f{int(f)}", set=role, l90_task=tid, suite="libero_90",
                     index=tid, episode=int(e), frame=int(f), T=T, language=t.language, phase=ph, grip=grip, since=since,
                     conds=conds(t, set()))
            if role == "train":
                o.update(hf_task=pos[t.language], src="ipec")
            obs.append(o)
    for o in base["obs"]:
        if o["set"] == "transfer":
            t = tasks[f"{o['suite']}:{o['index']}"]
            ex = {o["hf_task"]} if "hf_task" in o else set()
            obs.append(dict(o, conds=conds(t, ex, "transfer")))
    write_json(index_path("l90c"), dict(train_texts={str(i): x for i, x in enumerate(texts)}, bind_src=bind_src, obs=obs,
                                        split_sha=split.get("sealed_sha256")))
    n = {s: sum(o["set"] == s for o in obs) for s in ("train", "l90", "transfer")}
    print(f"[pp_prep] index_l90c: {n}; train tasks {len({o['hf_task'] for o in obs if o['set'] == 'train'})} of {len(texts)} "
          f"instructions; l90 tasks {sorted({o['l90_task'] for o in obs if o['set'] == 'l90'})}; splice sources {len(bind_src)}",
          flush=True)


def noop_mask(actions):
    keep, prev = [], None
    for a in actions:
        keep.append(not (np.linalg.norm(a[:-1]) < 1e-4 and prev is not None and a[-1] == prev[-1]))
        prev = a
    return np.asarray(keep)


def fetch():
    import h5py
    from huggingface_hub import hf_hub_download
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy import factors as F
    idx = json.load(open(os.path.join(OUT, "index.json")))
    tm = F.task_map()
    demos = D.Demos()
    os.makedirs(os.path.join(OUT, "match"), exist_ok=True)
    jobs = sorted({(o["hf_task"], o["suite"], o["index"], o["episode"]) for o in idx["obs"] if o["set"] == "train"})
    for k, suite, ti, e in jobs:
        out = os.path.join(OUT, "match", f"tr{k:02d}.npz")
        if os.path.exists(out):
            continue
        name = tm[suite][ti]
        p = hf_hub_download("yifengzhu-hf/LIBERO-datasets", f"{suite}/{name}_demo.hdf5", repo_type="dataset", local_dir=RAW)
        ep = demos.load(e)
        hf = np.asarray(ep["actions"], np.float64)[:, :6]
        best = None
        with h5py.File(p, "r") as f:
            for key in f["data"].keys():
                acts = np.asarray(f["data"][key]["actions"], np.float64)
                keep = noop_mask(acts)
                A = acts[keep][:, :6]
                if len(A) != len(hf):
                    continue
                err = float(np.abs(A - hf).max())
                if best is None or err < best[0]:
                    best = (err, key, np.where(keep)[0], np.asarray(f["data"][key]["states"])[keep])
        if best is None or best[0] > 1e-4:
            print(f"[pp_prep] task {k} ({suite}:{ti}): no raw demo matches episode {e} (best {None if best is None else best[0]})",
                  flush=True)
            continue
        err, key, kept, states = best
        save_atomic(out, raw_key=key, kept=kept, states=states, max_err=err, hf_state=np.asarray(ep["state"], np.float32))
        print(f"[pp_prep] task {k} ({suite}:{ti}): episode {e} = raw {key}, {len(kept)} steps, max |action diff| {err:.1e}", flush=True)


def poses():
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    idx = json.load(open(index_path(_tag())))
    os.makedirs(os.path.join(OUT, "poses"), exist_ok=True)
    bench = benchmark.get_benchmark_dict()
    groups = {}
    for o in idx["obs"]:
        if o["set"] in ("train", "l90"):
            k_ = o.get("src_hf", o.get("hf_task")) if o.get("src") != "ipec" else None      # index_l90c: hf_task is positional
            groups.setdefault((o["suite"], o["index"], k_, o.get("l90_task")), []).append(o)
    for (suite, ti, k, tid), obs in sorted(groups.items(), key=lambda x: str(x[0])):
        todo = [o for o in obs if not os.path.exists(os.path.join(OUT, "poses", o["key"] + ".npz"))]
        if not todo:
            continue
        mp = os.path.join(OUT, "match", f"tr{k:02d}.npz") if k is not None else os.path.join(P0, f"{tid}_match.npz")
        if not os.path.exists(mp):
            print(f"[pp_prep] poses: no match for {suite}:{ti}", flush=True)
            continue
        m = np.load(mp)
        task = bench[suite]().get_task(ti)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=64, camera_widths=64)
        env.seed(0)
        env.reset()
        inner = env.env
        names = list(inner.objects_dict.keys()) + list(inner.fixtures_dict.keys())
        kinds = ["object"] * len(inner.objects_dict) + ["fixture"] * len(inner.fixtures_dict)
        for o in todo:
            ob = env.set_init_state(m["states"][o["frame"]])
            P = [np.asarray(ob[f"{n}_pos"], np.float64) if f"{n}_pos" in ob else np.array(inner.sim.data.body_xpos[inner.obj_body_id[n]])
                 for n in names]
            eef = np.asarray(ob["robot0_eef_pos"], np.float32)
            save_atomic(os.path.join(OUT, "poses", o["key"] + ".npz"), names=np.array(names), kinds=np.array(kinds),
                        pos=np.asarray(P, np.float32), eef=eef)
        env.close()
        print(f"[pp_prep] poses {suite}:{ti}: {len(todo)} frames, {len(names)} bodies", flush=True)


if __name__ == "__main__":
    for cmd in sys.argv[1:]:
        if cmd.startswith("--"):
            continue
        {"index": index, "index_l90c": index_l90c, "fetch": fetch, "poses": poses}[cmd]()
