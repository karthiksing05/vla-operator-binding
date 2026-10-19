"""Plan H, H2 (cost): every method on ACCURACY, SPACE and TIME, the way robotics papers report cost (OpenVLA-OFT: latency and Hz
over 100 queries on one GPU; CLARE: parameter growth per task; SmolVLA: GPU-hours).  Login node, no torch: everything is read
from the ledger, the stored units in results/robot and the slurm logs; the latency comes from experiments/h2_latency.py's rows.

  SPACE   trainable parameters and bf16 bytes of every stored unit, per backbone, measured from the files (a .pt is a zip of raw
          storages: their sizes and dtypes are read without torch; a .npy from its header), cross-checked against the bytes the
          fitting scripts recorded (c0_fit, r3_fewshot).  Demonstration stores (experience replay, STRAP's prior) are charged
          at their size on disk.
  TIME    (a) fit time per unit: the optimisation loop as the fitting script logged it (c0_fit's '[c0j] ... in Ns', r3_fewshot's
          train_secs, g6_strap's train_secs, b_continual's secs, DSRL's practice-run secs) and the whole slurm job (model load,
          prefix caching, training) from its log's start and exit stamps; all on A40s.  (b) practice episodes per task, summed
          from the practice runs' ledger rows (n_episodes).  (c) test-time latency from ledger h2_latency.
  ACCURACY  the headline success counts from each read's ledger row.

    $PY experiments/h2_cost.py [--record]      # -> results/robot/h2_cost.json, figures/robot/h2_*.png, docs/COST.md
Ledger rows (with --record): h2_space tag units, h2_fit tag units, h2_practice tag counts, h2_cost tags libero90 and ood.
"""
import datetime as dt
import glob
import json
import os
import pickletools
import re
import subprocess
import sys
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from lifelong_ood import ledger                                       # noqa: E402

R = os.path.join(ROOT, "results", "robot")
SL = os.path.join(ROOT, "slurm")
FIG = os.path.join(ROOT, "figures", "robot")
BACKBONES = ("pi05_libero", "pi0_libero", "smolvla", "xvla")
NICE = {"pi05_libero": "pi0.5", "pi0_libero": "pi0", "smolvla": "SmolVLA", "xvla": "X-VLA"}
SUF = {"pi05_libero": "", "pi0_libero": "_pi0", "smolvla": "_smolvla", "xvla": "_xvla"}     # practice-row tag suffixes
PROTO_PREFIX = {"pi05_libero": "", "pi0_libero": "P", "smolvla": "V", "xvla": "Y"}          # G3/G19 prototype names
LORA_TAG = {"pi05_libero": "", "pi0_libero": "_p0", "smolvla": "_sv", "xvla": "_xv"}
L90_ROOT = "/coc/flash10/ksingara3/robot/data/hf/IPEC-COMMUNITY__libero_90_no_noops_lerobot"
D40_ROOT = "/coc/flash10/ksingara3/robot/data/hf/physical-intelligence__libero"
TLI_STD = (13, 17)                    # the latent pair the latency job times


_LOAD = {}


def load(exp, **f):
    """ledger.load cached per experiment (the read re-filters the same directories many times)."""
    if exp not in _LOAD:
        _LOAD[exp] = ledger.load(exp)
    return [r for r in _LOAD[exp] if ledger._match(r, f)]


def rel(p):
    return os.path.relpath(p, ROOT)


def latest(rows):
    return max(rows, key=lambda r: (r.get("started") or "", r["_path"]))


def one(exp, **f):
    rows = load(exp, **f)
    if not rows:
        raise KeyError(f"no ledger row {exp} {f}")
    return latest(rows)


# ── files without torch ──────────────────────────────────────────────────────────────────────────────────────────────────
DT_SIZE = {"HalfStorage": 2, "BFloat16Storage": 2, "FloatStorage": 4, "DoubleStorage": 8, "LongStorage": 8, "IntStorage": 4,
           "BoolStorage": 1, "ByteStorage": 1}


def pt_info(path):
    """A torch .pt (zip) -> raw storage bytes by dtype, parameters, bf16 bytes, file size.  The storages are the zip's data/<k>
    entries; each one's dtype is the Storage class named before its key in data.pkl."""
    z = zipfile.ZipFile(path)
    names = z.namelist()
    pkl = next(n for n in names if n.endswith("data.pkl"))
    sizes = {n.rsplit("/", 1)[1]: z.getinfo(n).file_size for n in names if "/data/" in n and not n.endswith(".pkl")}
    dtypes, last = {}, None
    for op, arg, _ in pickletools.genops(z.read(pkl)):
        if op.name in ("GLOBAL", "STACK_GLOBAL") and isinstance(arg, str) and "Storage" in arg:
            last = arg.split()[-1]
        elif isinstance(arg, str) and "Storage" in arg and op.name.endswith("UNICODE"):
            last = arg
        elif isinstance(arg, str) and arg in sizes and last:
            dtypes.setdefault(arg, last)
    params, by = 0, {}
    for k, s in sizes.items():
        d = dtypes.get(k, "FloatStorage")
        by[d] = by.get(d, 0) + s
        params += s // DT_SIZE.get(d, 4)
    return dict(params=int(params), bf16_bytes=int(2 * params), stored_bytes=int(sum(sizes.values())), by_dtype=by,
                file_bytes=int(os.path.getsize(path)), path=rel(path))


def npy_shape(path):
    a = np.load(path, mmap_mode="r")
    return tuple(int(x) for x in a.shape), str(a.dtype)


# ── slurm logs ───────────────────────────────────────────────────────────────────────────────────────────────────────────
DATE = re.compile(r"(\w{3}) (\w{3}) +(\d+) (\d+):(\d+):(\d+) (AM|PM) \w+ (\d{4})")


def _when(s):
    m = DATE.search(s)
    if not m:
        return None
    _, mon, d, h, mi, se, ap, y = m.groups()
    return dt.datetime.strptime(f"{mon} {d} {h}:{mi}:{se} {ap} {y}", "%b %d %I:%M:%S %p %Y")


_LOG = {}


def job_log(job):
    """-> dict(text, wall_s, gpu, host) for slurm/<job>.out (the last start stamp before the last exit stamp)."""
    if job in _LOG:
        return _LOG[job]
    p = os.path.join(SL, f"{job}.out")
    out = dict(text="", wall_s=None, gpu=None, path=rel(p))
    if os.path.exists(p):
        txt = open(p, errors="ignore").read()
        out["text"] = txt
        starts = [(i, ln) for i, ln in enumerate(txt.splitlines()) if ln.startswith("[robot] ") and " on " in ln and DATE.search(ln)]
        exits = [(i, ln) for i, ln in enumerate(txt.splitlines()) if ln.startswith("exit=")]
        if starts and exits:
            ie, le = exits[-1]
            st = [x for x in starts if x[0] < ie]
            if st:
                t0, t1 = _when(st[-1][1]), _when(le)
                out["wall_s"] = (t1 - t0).total_seconds() if t0 and t1 else None
                out["gpu"] = st[-1][1].rsplit(": ", 1)[-1].strip()
    _LOG[job] = out
    return out


# ── fits ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
C0 = None


def c0_rows():
    global C0
    if C0 is None:
        C0 = load("c0_fit")
    return C0


def proto_fit(name, backbone="pi05_libero", rank=4):
    rows = [r for r in c0_rows() if r["config"].get("name") == name and r["config"].get("rank") == rank
            and r["config"].get("backbone", "pi05_libero") == backbone]
    if not rows:
        return None
    r = latest(rows)
    lg = job_log(r["env"].get("slurm_job"))
    m = re.findall(rf"\[c0j\] {re.escape(name)}: .*? (\d+) demos, (\d+) frames, .* in (\d+)s", lg["text"])
    return dict(name=name, train_s=float(m[-1][2]) if m else None, wall_s=lg["wall_s"], gpu=lg["gpu"] or r["env"].get("device"),
                demos=int(m[-1][0]) if m else None, frames=int(m[-1][1]) if m else None, bytes=int(r["metrics"]["bytes"]),
                joint=[int(x) for x in str(r["config"].get("joint", "")).split("+") if x], row=rel(r["_path"]), log=lg["path"])


def lora_fit(task, tag, backbone="pi05_libero"):
    rows = [r for r in load("r3_fewshot") if r["keys"].get("task") == int(task) and r["config"].get("tag", "") == tag
            and r["config"].get("backbone", "pi05_libero") == backbone and not r["config"].get("full")]
    if not rows:
        return None
    r = latest(rows)
    lg = job_log(r["env"].get("slurm_job"))
    return dict(task=int(task), train_s=r["metrics"].get("train_secs"), wall_s=lg["wall_s"], gpu=r["env"].get("device"),
                bytes=int(r["bytes"]["total"]), params=r["metrics"].get("n_params"), demo_episode=r["metrics"].get("demo_episode"),
                row=rel(r["_path"]))


def wrong_scene():
    return set(one("l90_scenes", tag="map")["metrics"]["wrong_scene_tasks"])


def lorasc_tag(t, wrong):
    return "_sc" if int(t) in wrong and os.path.exists(os.path.join(R, f"r3_lora_{t}_sc.pt")) else ""


def tli_identify():
    """Per backbone: A40 wall seconds per identified latent, over every finished g12_tli --identify job."""
    out = subprocess.run(["grep", "-l", "-r", "--include=*.out", "g12_tli.py --identify", SL], capture_output=True,
                         text=True).stdout.split()
    acc = {}
    for p in out:
        job = os.path.basename(p)[:-4]
        lg = job_log(job)
        m = re.search(r"g12_tli.py --identify(?: --backbone (\S+))?", lg["text"])
        bb = (m.group(1) if m and m.group(1) else "pi0_libero")
        n = len(re.findall(r"^\[g12\] task \d+ .*latent", lg["text"], re.M))
        if lg["wall_s"] and n and "exit=0" in lg["text"]:
            a = acc.setdefault(bb, dict(wall_s=0.0, latents=0, jobs=[]))
            a["wall_s"] += lg["wall_s"]
            a["latents"] += n
            a["jobs"].append(lg["path"])
    for a in acc.values():
        a["wall_s_per_latent"] = a["wall_s"] / a["latents"]
    return acc


# ── the plans ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def g8_plan():
    e3p, e5p = json.load(open(os.path.join(R, "e3_plan.json"))), json.load(open(os.path.join(R, "e5_plan.json")))
    tasks = [int(t) for t, v in e3p.items() if v["lora"]] + [int(t) for t in e5p["tasks"]]
    protos = sorted({v["object_proto"] for v in e3p.values()} | {v["goal_proto"] for v in e3p.values()}
                    | {v["type_proto"] for v in e3p.values()} | {v["object_proto"] for v in e5p["tasks"].values()}
                    | {v["goal"] for v in e5p["tasks"].values()})
    e3_protos = sorted({v["object_proto"] for v in e3p.values()} | {v["goal_proto"] for v in e3p.values()})
    return tasks, protos, [int(t) for t in e3p], [int(t) for t, v in e3p.items() if v["lora"]], e3_protos


def ood_arms():
    """-> {backbone: {task key: {arm: arm string}}} for the served OOD arms (G15 / G18)."""
    import g12_tli as T
    import g14_tli_practice as G14
    pl = json.load(open(os.path.join(R, "g4_plan.json")))
    out = {}
    for bb in ("pi0_libero", "pi05_libero"):
        g10 = json.load(open(os.path.join(R, f"g10_read_{bb.split('_')[0]}.json")))
        s1, s2 = G14.choose(1, bb), G14.choose(2, bb)
        out[bb] = {k: dict(tli_li=T.arm(v), psel=g10["chosen"][k], tsel=G14._arm(s1[k], s2[k])) for k, v in pl["tasks"].items()}
    import g4_ood as G
    for bb in ("smolvla", "xvla"):
        st = json.load(open(os.path.join(R, f"g18_state_{bb}.json")))
        out[bb] = {k: dict(tli_li=T.arm(v), li_grasp=G.li_arms(v, pl)["li_grasp"], **st["pairs"][k]) for k, v in pl["tasks"].items()}
    return out, G.source_pool()


def tli_sources(arm):
    if not arm.startswith("tli@"):
        return set()
    a, b, _ = arm.split("@", 1)[1].split(".")
    return {int(a), int(b)}


# ── SPACE ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def space():
    u = {}
    for bb in BACKBONES:
        pre = PROTO_PREFIX[bb]
        p = os.path.join(R, f"prim_{pre}Osc1_0.pt")
        f = proto_fit(f"{pre}Osc1", bb)
        info = pt_info(p)
        u[f"prototype_r4/{bb}"] = dict(info, ledger_bytes=f["bytes"], row=f["row"])
        lp = os.path.join(R, f"r3_lora_10{LORA_TAG[bb]}.pt")
        lf = lora_fit(10, LORA_TAG[bb], bb)
        u[f"lora_r16/{bb}"] = dict(pt_info(lp), ledger_bytes=lf["bytes"], row=lf["row"])
        lat = os.path.join(R, "tli", f"{bb}_13.npy")
        shp, dty = npy_shape(lat)
        n = int(np.prod(shp))
        u[f"tli_latent/{bb}"] = dict(shape=list(shp), dtype=dty, params=n, bf16_bytes=2 * n, file_bytes=os.path.getsize(lat),
                                     path=rel(lat), files=len(glob.glob(os.path.join(R, "tli", f"{bb}_*.npy"))))
    lf = lora_fit(10, "_r4")
    u["lora_r4/pi05_libero"] = dict(pt_info(os.path.join(R, "r3_lora_10_r4.pt")), ledger_bytes=lf["bytes"], row=lf["row"])
    for k in ("dsrlz", "dsrls", "dsrlf", "dsrl", "dsrlo"):
        fs = sorted(glob.glob(os.path.join(R, f"{k}_*_actor.pt")))
        infos = [pt_info(x) for x in fs]
        u[f"dsrl_actor/{k}"] = dict(infos[0], files=len(fs), params_set=sorted({i["params"] for i in infos}))
    for m in ("seq", "er"):
        u[f"cl_adapter/{m}"] = pt_info(os.path.join(R, f"bcl_{m}_g11.pt"))
    # STRAP's retrieval prior: its DINOv2 indexes, its frame dump, and the training demonstrations it retrieves from
    d40 = sum(os.path.getsize(x) for x in glob.glob(os.path.join(D40_ROOT, "data", "*", "*.parquet")))
    n40 = len(glob.glob(os.path.join(D40_ROOT, "data", "*", "*.parquet")))
    idx = {n: os.path.getsize(os.path.join(R, n)) for n in ("strap_emb.npz", "strap90_emb.npz", "strap90_frames.npz")}
    u["strap_prior"] = dict(indexes=idx, d40_parquet_bytes=int(d40), d40_episodes=n40, total=int(d40 + sum(idx.values())))
    u["strap_lora"] = pt_info(sorted(glob.glob(os.path.join(R, "strap90_*_lora.pt")))[0])
    return u


def er_demo_bytes(tasks, wrong):
    """Experience replay keeps each arrival's one demonstration: its parquet episode (states, actions) and the two camera videos,
    as stored (LeRobot format).  A task without a recorded demo episode is charged the mean of the others (listed as missing)."""
    info = json.load(open(os.path.join(L90_ROOT, "meta", "info.json")))
    out, miss = {}, []
    for t in tasks:
        lf = lora_fit(t, lorasc_tag(t, wrong))
        e = lf and lf["demo_episode"]
        p = e is not None and os.path.join(L90_ROOT, info["data_path"].format(episode_chunk=int(e) // info["chunks_size"], episode_index=int(e)))
        if p and os.path.exists(p):
            vids = [os.path.join(L90_ROOT, info["video_path"].format(episode_chunk=int(e) // info["chunks_size"], video_key=k,
                                                                     episode_index=int(e)))
                    for k in ("observation.images.image", "observation.images.wrist_image")]
            out[int(t)] = int(os.path.getsize(p) + sum(os.path.getsize(v) for v in vids if os.path.exists(v)))
        else:
            miss.append(int(t))
    return out, miss


# ── PRACTICE ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def episodes(exp, tasks=None, tag_re=".*"):
    """Episodes of the matching practice rows, one row per (tag, task, first trial): a rerun replaces, never adds."""
    best = {}
    for r in load(exp):
        if not re.fullmatch(tag_re, r["tag"]):
            continue
        tk = str(r["keys"].get("tasks", ""))
        if tasks is not None and (not tk.isdigit() or int(tk) not in tasks):
            continue
        k = (r["tag"], tk, r["keys"].get("first_trial"))
        if k not in best or (r.get("started") or "") > (best[k].get("started") or ""):
            best[k] = r
    return int(sum(int(r["metrics"].get("n_episodes", 0)) for r in best.values())), len(best)


def ood_practice():
    out = {}
    for bb in BACKBONES:
        s = SUF[bb]
        neg = "(?!.*(_pi0|_smolvla|_xvla)$)" if not s else ""
        rx = neg + r".*" + re.escape(s) + "$" if s else neg + ".*"
        c = {}
        for sh in ("goal", "spatial"):
            for st in ("g10_s1", "g10_s2", "g10_acc", "g14_s1", "g14_s2", "g15_acc"):
                n, k = episodes(f"{st}_{sh}", tag_re=rx)
                c[st] = c.get(st, 0) + n
        secs = []
        for sh in ("goal", "spatial"):
            for r in load(f"g10_s1_{sh}"):
                if re.fullmatch(rx, r["tag"]) and r["metrics"].get("secs") and r["metrics"].get("n_episodes"):
                    secs.append(r["metrics"]["secs"] / r["metrics"]["n_episodes"])
        c["switch"] = c["g10_s1"] + c["g10_s2"] + c["g10_acc"]
        c["tli"] = c["g14_s1"] + c["g14_s2"] + c["g15_acc"]
        c["system"] = c["switch"] + c["tli"]
        c["episode_wall_s"] = float(np.mean(secs)) if secs else None
        out[bb] = c
    return out


def l90_practice(g8_tasks, e3_tasks, e3_lora):
    s8 = set(g8_tasks)
    zs = episodes("e3_accept", s8, r"zs_seed\d+")[0] + episodes("e5_accept", s8, r"zs_seed\d+")[0]
    fac = episodes("e3_accept", s8, r"fac_.*")[0] + episodes("e5_accept", s8, r"fac_.*")[0]
    lora = episodes("e3_accept", s8, r"lorasc_seed\d+")[0] + episodes("e5_accept", s8, r"lorasc_seed\d+")[0]
    sb = set(e3_lora)
    zs_b = episodes("e3_accept", sb, r"zs_seed\d+")[0]
    zs_b30 = episodes("e3_accept", set(e3_tasks), r"zs_seed\d+")[0]
    fac_b = episodes("e3_accept", sb, r"fac_.*")[0]
    lora_b = episodes("e3_accept", sb, r"lorasc_seed\d+")[0]
    fac_b30 = episodes("e3_accept", set(e3_tasks), r"fac_.*")[0]
    c = dict(g8_zs=zs, g8_fac=fac, g8_lora=lora, g8_tasks=len(s8), e3_zs=zs_b, e3_fac=fac_b, e3_lora=lora_b, e3_zs30=zs_b30, e3_fac30=fac_b30,
             strap=episodes("g6_accept")[0], g9_r4=episodes("g9_accept")[0],
             dsrlz_train=episodes("g7_train", set(e3_tasks), r"dsrlz_train.*")[0], dsrlz_acc=episodes("g7_accept", None, r"dsrlz.*")[0],
             dsrls_train=episodes("g7_train", sb, r"dsrls_train.*")[0], dsrls_acc=episodes("g7_accept", None, r"dsrls.*")[0],
             dsrlf_train=episodes("g7b_train")[0], dsrlf_acc=episodes("g7b_accept")[0])
    w = one("w1_stream", tag="read")["metrics"]["a"]
    c["w1"] = int(sum(10 * sum(k in a for k in ("zs", "acc1", "acc2")) for a in w["arrivals_log"]))
    c["w1_arrivals"] = len(w["arrivals_log"])
    return c


# ── LATENCY ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def latency():
    out = {}
    for bb in BACKBONES:
        rows = load("h2_latency", tag=bb)
        if rows:
            r = latest(rows)
            out[bb] = dict(arms=r["metrics"]["arms"], row=rel(r["_path"]), gpu=r["metrics"]["gpu"])
    return out


def lat_of(lat, bb, mix):
    """mix: {arm: weight}; -> (ms, Hz, peak GB, arms used) or Nones when not measured.  Hz of the mix = 1000 / mean ms."""
    if bb not in lat:
        return None
    a = lat[bb]["arms"]
    mix = {k: w for k, w in mix.items() if w}
    if not all(k in a for k in mix):
        return None
    W = sum(mix.values())
    ms = sum(a[k]["ms_mean"] * w for k, w in mix.items()) / W
    return dict(ms=ms, hz=1000.0 / ms, peak_gb=max(a[k]["peak_alloc_gb"] for k in mix), arms=mix)


# ── the read ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def build():
    wrong = wrong_scene()
    g8_tasks, protos, e3_tasks, e3_lora, e3_protos = g8_plan()
    U = space()
    P4, L16, L4 = U["prototype_r4/pi05_libero"]["bf16_bytes"], U["lora_r16/pi05_libero"]["bf16_bytes"], U["lora_r4/pi05_libero"]["bf16_bytes"]
    fits = dict(protos={p: proto_fit(p) for p in protos}, loras={t: lora_fit(t, lorasc_tag(t, wrong)) for t in g8_tasks},
                loras_r4={t: lora_fit(t, "_r4") for t in e3_lora})
    for bb in BACKBONES[1:]:
        names = sorted({r["config"]["name"] for r in c0_rows() if r["config"].get("backbone") == bb and r["config"].get("rank") == 4
                        and "smoke" not in r["config"]["name"].lower()})
        fits[f"protos/{bb}"] = {n: proto_fit(n, bb) for n in names}
        fits[f"loras/{bb}"] = {r["keys"]["task"]: lora_fit(r["keys"]["task"], LORA_TAG[bb], bb)
                               for r in load("r3_fewshot") if r["config"].get("backbone") == bb}
    fits["strap"] = {}
    for r in load("g6_strap"):
        if r["tag"] == "fit":
            lg = job_log(r["env"].get("slurm_job"))
            fits["strap"][r["keys"]["task"]] = dict(train_s=r["metrics"]["train_secs"], wall_s=lg["wall_s"], row=rel(r["_path"]))
    fits["cl"] = {}
    for m in ("seq", "er"):
        r = one("b_continual", tag=f"{m}_g11")
        fits["cl"][m] = dict(train_s=r["metrics"]["secs"], wall_s=job_log(r["env"].get("slurm_job"))["wall_s"], row=rel(r["_path"]))
    fits["dsrl"] = {}
    for k, exp in (("dsrlz", "g7_train"), ("dsrls", "g7_train"), ("dsrlf", "g7b_train")):
        best = {}
        for r in load(exp):
            if r["tag"].startswith(k + "_train") and int(r["keys"]["tasks"]) in set(e3_tasks):
                kk = int(r["keys"]["tasks"])
                if kk not in best or (r.get("started") or "") > (best[kk].get("started") or ""):
                    best[kk] = r
        rs = list(best.values())
        fits["dsrl"][k] = dict(wall_s=float(sum(r["metrics"]["secs"] for r in rs)), runs=len(rs),
                               episodes=int(sum(r["metrics"]["n_episodes"] for r in rs)))
    fits["tli"] = tli_identify()
    w1 = one("w1_stream", tag="read")
    w1_names = set()
    for a in w1["metrics"]["a"]["arrivals_log"]:
        for k in ("comp1", "comp2", "kept"):
            v = a.get(k) or ""
            if v.startswith("fac@"):
                w1_names |= {x for x in v[4:].split(".") if x != "x"}
    fits["w1"] = {n: proto_fit(n) for n in sorted(w1_names)}
    pr = l90_practice(g8_tasks, e3_tasks, e3_lora)
    po = ood_practice()
    er_bytes, er_miss = er_demo_bytes(json.load(open(os.path.join(R, "w1_state.json")))["a"]["order"], wrong)
    lat = latency()
    arms, pool = ood_arms()
    return dict(U=U, P4=P4, L16=L16, L4=L4, fits=fits, practice=pr, ood_practice=po, er_bytes=er_bytes, er_missing=er_miss,
                latency=lat, ood_arms=arms, pool=pool, g8_tasks=g8_tasks, protos=protos, e3_tasks=e3_tasks, e3_lora=e3_lora,
                e3_protos=e3_protos, wrong=sorted(wrong))


def s_(fs, key):
    """-> (total over the units, units measured); a unit without a measurement (its slurm log gone) is charged the mean of the
    measured ones."""
    v = [f[key] for f in fs if f and f.get(key) is not None]
    if not v:
        return 0.0, 0
    return float(sum(v)) * len(fs) / len(v), len(v)


def table(B):
    """-> (LIBERO-90 rows, OOD rows); every row: method, set, success k / n, bytes per task and cumulative, fit A40-minutes per
    task (train / job), practice episodes per task, demonstrations per task, latency, sources."""
    P4, L16, L4, U, F, pr = B["P4"], B["L16"], B["L4"], B["U"], B["fits"], B["practice"]
    lat = B["latency"]
    g8 = one("g8_leaves", tag="read")
    m8 = g8["metrics"]
    n8 = m8["tasks"]
    kept = m8["kept"]
    pf = [F["protos"][p] for p in B["protos"]]
    lf = [F["loras"][t] for t in B["g8_tasks"]]
    leaf_tasks = [int(t) for t, v in m8["per_task"].items() if v["kept"] == "lora"]
    proto_tr, proto_wall = s_(pf, "train_s")[0], s_(pf, "wall_s")[0]
    lora_tr, lora_wall = s_(lf, "train_s")[0], s_(lf, "wall_s")[0]
    demos_g8 = sorted(set(B["g8_tasks"]) | {t for f in pf for t in f["joint"]})
    src8 = [rel(g8["_path"]), "h2_space/units", "h2_fit/units", "h2_practice/counts", "h2_latency/pi05_libero"]
    rows = []

    def row(method, set_, k, n, ntask, cum, fit_tr, fit_wall, practice, demos, mix, src, note="", bb="pi05_libero", family=""):
        L = lat_of(lat, bb, mix) if mix else None
        rows.append(dict(method=method, set=set_, k=int(k), n=int(n), rate=k / n, tasks=int(ntask), bytes_cum=int(cum),
                         bytes_task=cum / ntask, fit_train_min_task=None if fit_tr is None else fit_tr / 60 / ntask,
                         fit_job_min_task=None if fit_wall is None else fit_wall / 60 / ntask,
                         fit_job_min_cum=None if fit_wall is None else fit_wall / 60,
                         practice_task=practice / ntask, demos_task=demos / ntask, latency=L, sources=src, note=note, backbone=bb,
                         family=family))

    row("Hierarchy with task leaves (ours, G8)", "d", m8["H1"]["x"], m8["H1"]["n"], n8, len(B["protos"]) * P4 + kept["lora"] * L16,
        proto_tr + lora_tr, proto_wall + lora_wall, pr["g8_zs"] + pr["g8_fac"] + pr["g8_lora"], len(demos_g8),
        {"policy": kept["zs"], "prototype": kept["hfac"], "lora16": kept["lora"]}, src8,
        f"{len(B['protos'])} rank-4 prototypes + {kept['lora']} rank-16 leaves; every task's LoRA is fitted for the acceptance test",
        family="ours")
    row("Library without leaves (ours, G8)", "d", m8["library_vs_lora_acc"]["x"], n8 * 20, n8, len(B["protos"]) * P4, proto_tr, proto_wall,
        pr["g8_zs"] + pr["g8_fac"], len(demos_g8), {"policy": 1, "prototype": 1}, src8,
        "prototypes only; latency the mean of the policy and a prototype", family="ours")
    nl = m8["bytes"]["per_task_loras"]
    row("Per-task LoRA r16 + acceptance (G8)", "d", m8["H1"]["y"], m8["H1"]["n"], n8, nl * L16, lora_tr, lora_wall,
        pr["g8_zs"] + pr["g8_lora"], n8, {"policy": n8 - nl, "lora16": nl}, src8, f"{nl} adapters kept of {n8} fitted",
        family="adapter")
    row("Policy alone (pi0.5)", "d", m8["leaves_vs_zs"]["y"], m8["H1"]["n"], n8, 0, 0, 0, 0, 0, {"policy": 1}, src8, family="policy")
    # set b: E3's tasks
    g9 = one("g9_rank4lora", tag="read")["metrics"]
    g6 = one("g6_strap", tag="read")
    g7 = one("g7_dsrl", tag="read")
    g7b = one("g7_dsrl", tag="read_library")
    srcb = [rel(one("g9_rank4lora", tag="read")["_path"]), "h2_space/units", "h2_fit/units", "h2_practice/counts",
            "h2_latency/pi05_libero"]
    nb = g9["tasks"]
    e3f = [F["protos"][p] for p in B["e3_protos"]]
    e3_tr, e3_wall = s_(e3f, "train_s")[0], s_(e3f, "wall_s")[0]
    lb = [F["loras"][t] for t in B["e3_lora"]]
    lb_tr, lb_wall = s_(lb, "train_s")[0], s_(lb, "wall_s")[0]
    e3_demos = sorted(set(B["e3_lora"]) | {t for f in e3f for t in f["joint"]})
    nlb = g9["kept_lora"]
    e3k = one("e3_hier", tag="read")["metrics"]["kept"]
    nh = sum(e3k[str(t)] == "hfac" for t in B["e3_lora"])
    row("Library (ours, E3)", "b", g9["H1"]["y"], g9["H1"]["n"], nb, g9["bytes"]["library"], e3_tr, e3_wall, pr["e3_zs"] + pr["e3_fac"],
        len(e3_demos), {"policy": nb - nh, "prototype": nh}, srcb, f"{g9['bytes']['prototypes']} prototypes", family="ours")
    e3h = one("e3_hier", tag="read")
    kl = sum(v == "lora" for v in e3h["metrics"]["kept_lora"].values())
    row("Per-task LoRA r16 + acceptance (E3)", "b", e3h["metrics"]["hypotheses"]["H2"]["y"], e3h["metrics"]["hypotheses"]["H2"]["n"], nb,
        kl * L16, lb_tr, lb_wall, pr["e3_zs"] + pr["e3_lora"], nb, {"policy": nb - kl, "lora16": kl}, [rel(e3h["_path"])] + srcb[1:],
        f"{kl} adapters kept of {nb}", family="adapter")
    r4 = [F["loras_r4"][t] for t in B["e3_lora"]]
    row("Per-task LoRA r4 + acceptance (G9)", "b", g9["H1"]["x"], g9["H1"]["n"], nb, nlb * L4, s_(r4, "train_s")[0], s_(r4, "wall_s")[0],
        pr["e3_zs"] + pr["g9_r4"], nb, {"policy": nb - nlb, "lora4": nlb}, srcb, f"{nlb} kept of {nb}", family="adapter")
    sp = U["strap_prior"]["total"]
    nks = sum(v == "strap" for v in g6["metrics"]["kept_strap"].values())
    st = list(F["strap"].values())
    srcs = [rel(g6["_path"])] + srcb[1:]
    row("STRAP", "b", g6["metrics"]["descriptive"]["strap_vs_zs"]["x"], g9["H1"]["n"], nb, sp + nb * L16, s_(st, "train_s")[0],
        s_(st, "wall_s")[0], 0, nb, {"lora16": 1}, srcs, "retrieval prior (training demos + DINOv2 indexes) charged once; retrieval "
        "and embedding stages not timed", family="strap")
    row("STRAP + acceptance (G6)", "b", g6["metrics"]["H1"]["y"], g9["H1"]["n"], nb, sp + nks * L16, s_(st, "train_s")[0],
        s_(st, "wall_s")[0], pr["e3_zs"] + pr["strap"], nb, {"policy": nb - nks, "lora16": nks}, srcs, f"{nks} adapters kept",
        family="strap")
    A = U["dsrl_actor/dsrlz"]["bf16_bytes"]
    hz7 = g7["metrics"]
    nz = sum(v.get("dsrlz") == "dsrlz" for v in hz7["kept"].values())
    ns = sum(v.get("dsrls") == "dsrls" for v in hz7["kept"].values())
    src7 = [rel(g7["_path"]), "h2_space/units", "h2_fit/units", "h2_practice/counts", "h2_latency/pi05_libero"]
    n30 = hz7["descriptive"]["dsrlz_vs_zs"]["n"] // 20
    row("DSRL on the policy (G7, dsrlz) + acceptance", "b30", hz7["hypotheses"]["lib_vs_dsrlz_acc"]["y"], hz7["descriptive"]["dsrlz_vs_zs"]["n"],
        n30, nz * A, None, F["dsrl"]["dsrlz"]["wall_s"], pr["e3_zs30"] + pr["dsrlz_train"] + pr["dsrlz_acc"], 0,
        {"policy": n30 - nz, "dsrl": nz}, src7, f"{nz} actors kept; fit = the 100 practice episodes with SAC updates (wall, shared "
        "GPU)", family="dsrl")
    row("DSRL on the per-task LoRA (G7, dsrls) + acceptance", "b", hz7["hypotheses"]["lib_vs_dsrls_acc"]["y"], g9["H1"]["n"], nb,
        ns * (A + L16), lb_tr, lb_wall + F["dsrl"]["dsrls"]["wall_s"], pr["e3_zs"] + pr["dsrls_train"] + pr["dsrls_acc"], nb,
        {"policy": nb - ns, "lora16": ns}, src7, f"{ns} actors + LoRAs kept; latency: the LoRA's (the actor adds about 1 ms "
        "on the policy, h2_latency arm dsrl)", family="dsrl")
    nf = g7b["metrics"]["kept"]["dsrlf"]
    row("DSRL on the library (G7b, dsrlf) + acceptance", "b30", g7b["metrics"]["H2"]["x"], g7b["metrics"]["H2"]["n"], n30,
        g9["bytes"]["library"] + nf * A, e3_tr, e3_wall + F["dsrl"]["dsrlf"]["wall_s"],
        pr["e3_zs30"] + pr["e3_fac30"] + pr["dsrlf_train"] + pr["dsrlf_acc"], len(e3_demos), {"policy": n30 - nf, "prototype": nf},
        [rel(g7b["_path"])] + src7[1:], f"{nf} actors kept", family="dsrl")
    row("Policy alone (pi0.5)", "b", g6["metrics"]["descriptive"]["strap_vs_zs"]["y"], g9["H1"]["n"], nb, 0, 0, 0, 0, 0, {"policy": 1},
        [rel(g6["_path"])], family="policy")
    # W1 stream (G11)
    g11 = one("g11_cl_baselines", tag="read")
    w1 = one("w1_stream", tag="read")["metrics"]["a"]
    nw = g11["metrics"]["tasks"]
    src11 = [rel(g11["_path"]), rel(one("w1_stream", tag="read")["_path"]), "h2_space/units", "h2_fit/units", "h2_practice/counts",
             "h2_latency/pi05_libero"]
    wf = list(F["w1"].values())
    req = [a["task"] for a in w1["arrivals_log"] if a.get("demo")]
    erm = np.mean(list(B["er_bytes"].values()))
    w1_demo_b = int(sum(B["er_bytes"].get(int(t), erm) for t in req))
    row("W1 learner (ours, stream)", "b30", g11["metrics"]["w1_vs_seq"]["x"], g11["metrics"]["w1_vs_seq"]["n"], nw,
        w1["prototypes"] * P4 + w1_demo_b,
        s_(wf, "train_s")[0], s_(wf, "wall_s")[0], pr["w1"], w1["demos"], {"policy": nw - w1["kept_library"], "prototype": w1["kept_library"]},
        src11, f"{w1['prototypes']} prototypes + the {len(req)} requested demonstrations it keeps for refits ({w1_demo_b / 1e6:.1f} MB); "
        "fit = every prototype the stream fitted or refitted", family="ours")
    cl = F["cl"]
    row("Sequential fine-tuning (G11)", "b30", g11["metrics"]["w1_vs_seq"]["y"], g11["metrics"]["w1_vs_seq"]["n"], nw,
        U["cl_adapter/seq"]["bf16_bytes"], cl["seq"]["train_s"], cl["seq"]["wall_s"], 0, nw, {"lora16": 1}, src11, "one adapter",
        family="cl")
    erb = int(sum(B["er_bytes"].get(int(t), erm) for t in json.load(open(os.path.join(R, "w1_state.json")))["a"]["order"]))
    row("Experience replay (G11)", "b30", g11["metrics"]["w1_vs_er"]["y"], g11["metrics"]["w1_vs_er"]["n"], nw,
        U["cl_adapter/er"]["bf16_bytes"] + erb, cl["er"]["train_s"], cl["er"]["wall_s"], 0, nw, {"lora16": 1}, src11,
        f"one adapter + {nw} stored demonstrations ({erb / 1e6:.1f} MB, parquet + videos)", family="cl")
    row("Policy alone (pi0.5)", "b30", g11["metrics"]["seq_vs_zs"]["y"], g11["metrics"]["seq_vs_zs"]["n"], nw, 0, 0, 0, 0, 0, {"policy": 1},
        [rel(g11["_path"])], family="policy")

    # ── OOD side ──
    orows = []
    po, arms_all, tli = B["ood_practice"], B["ood_arms"], F["tli"]
    for bb in BACKBONES:
        sh = NICE[bb]
        if bb in ("pi0_libero", "pi05_libero"):
            g = one("g15_operator", tag=f"read_{bb.split('_')[0]}")
        else:
            g = one("g18_four_policies", tag=f"read_{bb}")
        m = g["metrics"]
        pt = m["per_task"]
        tot = {a: sum(v[a] for v in pt.values()) for a in next(iter(pt.values()))}
        n = 20 * len(pt)
        T = len(pt)
        arms = arms_all[bb]
        LB = U[f"tli_latent/{bb}"]["bf16_bytes"]
        per_lat = tli.get(bb, {}).get("wall_s_per_latent")
        ch = m["choice"]
        src = [rel(g["_path"]), "h2_space/units", "h2_fit/units", "h2_practice/counts", f"h2_latency/{bb}"]

        def lat_bytes(keys):
            s = set()
            for k in keys:
                s |= keys[k]
            return s
        sys_src = {k: tli_sources(arms[k][ch[k]]) for k in arms}
        tsel_src = {k: tli_sources(arms[k]["tsel"]) for k in arms}
        li_src = {k: tli_sources(arms[k]["tli_li"]) for k in arms}
        ns_sys, ns_tsel, ns_li = len(lat_bytes(sys_src)), len(lat_bytes(tsel_src)), len(lat_bytes(li_src))
        ntli = sum(c == "tsel" for c in ch.values())
        pool = len(B["pool"])
        fit_pool = None if per_lat is None else per_lat * pool
        orow = lambda **kw: orows.append(dict(backbone=bb, set="e", n=n, tasks=T, demos_task=0.0, **kw))   # noqa: E731

        def L_(mix):
            return lat_of(lat, bb, mix)
        orow(method="Practice-chosen operator (ours, G15/G18)", k=tot["chosen"], rate=tot["chosen"] / n, bytes_cum=ns_sys * LB,
             bytes_task=ns_sys * LB / T, fit_job_min_cum=None if fit_pool is None else fit_pool / 60,
             fit_job_min_task=None if fit_pool is None else fit_pool / 60 / T, practice_task=po[bb]["system"] / T,
             latency=L_({"switch_carry": T - ntli, "tli": ntli}), sources=src, family="ours",
             note=f"TLI chosen on {ntli} tasks: {ns_sys} source latents kept; practice identifies the {pool}-source pool")
        orow(method="Grasp switch, practice-chosen pair (G10)", k=tot["psel"], rate=tot["psel"] / n, bytes_cum=0, bytes_task=0,
             fit_job_min_cum=0, fit_job_min_task=0, practice_task=po[bb]["switch"] / T, latency=L_({"switch_carry": 1}), sources=src,
             family="ours", note="pointers to two training instructions: no parameters")
        orow(method="TLI, practice-chosen pair (G14)", k=tot["tsel"], rate=tot["tsel"] / n, bytes_cum=ns_tsel * LB, bytes_task=ns_tsel * LB / T,
             fit_job_min_cum=None if fit_pool is None else fit_pool / 60, fit_job_min_task=None if fit_pool is None else fit_pool / 60 / T,
             practice_task=po[bb]["tli"] / T, latency=L_({"tli": 1}), sources=src, family="tli",
             note=f"{ns_tsel} source latents kept")
        orow(method="Li's TLI, Li's hand pairs", k=tot["tli_li"], rate=tot["tli_li"] / n, bytes_cum=ns_li * LB, bytes_task=ns_li * LB / T,
             fit_job_min_cum=None if per_lat is None else per_lat * ns_li / 60,
             fit_job_min_task=None if per_lat is None else per_lat * ns_li / 60 / T, practice_task=0.0, latency=L_({"tli": 1}),
             sources=src, family="tli", note=f"{ns_li} source latents; hand-chosen pairs and lambda")
        if "li_grasp" in tot:
            orow(method="Li's hand pairs at the grasp", k=tot["li_grasp"], rate=tot["li_grasp"] / n, bytes_cum=0, bytes_task=0,
                 fit_job_min_cum=0, fit_job_min_task=0, practice_task=0.0, latency=L_({"switch_carry": 1}), sources=src, family="tli",
                 note="hand-chosen pointers")
        orow(method=f"Policy alone ({sh})", k=tot["zs"], rate=tot["zs"] / n, bytes_cum=0, bytes_task=0, fit_job_min_cum=0,
             fit_job_min_task=0, practice_task=0.0, latency=L_({"policy": 1}), sources=src, family="policy", note="")
        if bb in ("pi0_libero", "pi05_libero"):
            g16 = one("g16_stream_ood", tag=f"read_{bb.split('_')[0]}")
            for order in ("plan", "reversed"):
                z = g16["metrics"][order]
                orow(method=f"Stream with reuse (ours, G16, {order} order)", k=z["stream_vs_g15"]["x"], rate=z["stream_vs_g15"]["x"] / n,
                     bytes_cum=ns_sys * LB, bytes_task=ns_sys * LB / T, fit_job_min_cum=None if fit_pool is None else fit_pool / 60,
                     fit_job_min_task=None if fit_pool is None else fit_pool / 60 / T, practice_task=z["practice"] / T,
                     latency=L_({"switch_carry": T - z["tli_served"], "tli": z["tli_served"]}), sources=[rel(g16["_path"])] + src[1:],
                     family="ours", note=f"TLI served on {z['tli_served']}; bytes bounded by G15's kept latents")
    return rows, orows


# ── outputs ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def mb(b):
    return "0" if not b else (f"{b / 1e9:.2f} GB" if b >= 1e9 else f"{b / 1e6:.1f} MB" if b >= 1e5 else f"{b / 1e3:.1f} KB")


def fmin(x):
    return "n/a" if x is None else ("0" if x == 0 else f"{x:.1f}")


def lat_s(L):
    return "pending" if L is None else f"{L['ms']:.0f} / {L['hz']:.1f}"


def mem_s(L):
    return "pending" if L is None else f"{L['peak_gb']:.1f}"


SET_NAME = {"d": "set d, 66 tasks", "b": "set b, 29 tasks", "b30": "set b, 30 tasks", "e": "set e, 20 tasks"}


def md_table(rows, ood=False):
    h = ["method"] + (["policy"] if ood else []) + ["set", "success", "bytes / task", "cumulative", "fit A40-min / task (job)",
                                                    "practice ep. / task", "demos / task", "latency ms / Hz", "peak GB", "ledger"]
    out = ["| " + " | ".join(h) + " |", "|" + "---|" * len(h)]
    for r in rows:
        c = [r["method"]] + ([NICE[r["backbone"]]] if ood else []) + [SET_NAME[r["set"]], f"{r['k']} / {r['n']} ({100 * r['rate']:.1f}%)",
                                                                      mb(r["bytes_task"]), mb(r["bytes_cum"]),
                                                                      fmin(r.get("fit_job_min_task")), f"{r['practice_task']:.0f}",
                                                                      f"{r['demos_task']:.2f}".rstrip("0").rstrip("."),
                                                                      lat_s(r["latency"]), mem_s(r["latency"]),
                                                                      "; ".join(f"`{s}`" for s in r["sources"][:1])]
        out.append("| " + " | ".join(c) + " |")
    return "\n".join(out)


COL = {"ours": "#2a78d6", "adapter": "#eb6834", "tli": "#1baf7a", "dsrl": "#eda100", "strap": "#e87ba4", "cl": "#6250d6",
       "policy": "#8a8985"}
FAM = {"ours": "ours", "adapter": "per-task adapter", "tli": "Li's TLI / hand pairs", "dsrl": "DSRL", "strap": "STRAP",
       "cl": "continual learning", "policy": "policy alone"}


def figure(rows, path, title, mark_key, marks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                         "ytick.color": "#52514e"})
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), facecolor="#fcfcfb")
    for ax, (xk, xl) in zip(axs, (("bytes_cum", "stored bytes per task (bf16 parameters + stored data), log"),
                                  ("cost_ep", "practice episodes + demonstrations per task, log"))):
        ax.set_facecolor("#fcfcfb")
        for r in rows:
            x = r["bytes_task"] if xk == "bytes_cum" else r["practice_task"] + r["demos_task"]
            x = max(x, 1e3 if xk == "bytes_cum" else 0.5)              # zero cost drawn at the axis floor
            ax.scatter([x], [100 * r["rate"]], s=60, marker=marks[r[mark_key]], color=COL[r["family"]], edgecolor="#fcfcfb",
                       linewidth=1.5, zorder=3)
            if r["family"] in ("ours",) or r.get("label"):
                ax.annotate(r.get("label") or r["method"].split(" (")[0], (x, 100 * r["rate"]), textcoords="offset points",
                            xytext=(5, 4), fontsize=7, color="#52514e")
        ax.set_xscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel("success (%)")
        ax.grid(True, color="#e6e5e0", linewidth=0.6, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    hs = [Line2D([], [], marker="o", linestyle="", color=COL[f], label=FAM[f]) for f in COL if any(r["family"] == f for r in rows)]
    hs += [Line2D([], [], marker=m, linestyle="", color="#52514e", label=k) for k, m in marks.items()
           if any(r[mark_key] == k for r in rows)]
    fig.legend(handles=hs, loc="lower center", ncol=min(len(hs), 7), frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=10, color="#0b0b0b")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def main(record):
    B = build()
    rows, orows = table(B)
    os.makedirs(FIG, exist_ok=True)
    for r in rows:
        r["setname"] = SET_NAME[r["set"]]
    figure(rows, os.path.join(FIG, "h2_pareto_libero90.png"),
           "LIBERO-90 tasks never seen (pi0.5): success against storage and against practice + demonstrations", "setname",
           {SET_NAME["d"]: "o", SET_NAME["b"]: "s", SET_NAME["b30"]: "^"})
    for r in orows:
        r["policy"] = NICE[r["backbone"]]
    figure(orows, os.path.join(FIG, "h2_pareto_ood.png"),
           "Li's compositional OOD tasks (set e), four flow policies: success against storage and against practice", "policy",
           {"pi0.5": "o", "pi0": "s", "SmolVLA": "^", "X-VLA": "D"})
    res = dict(libero90=rows, ood=orows, units=B["U"], practice=B["practice"], ood_practice=B["ood_practice"],
               tli_identify=B["fits"]["tli"], er_bytes=B["er_bytes"], er_missing=B["er_missing"],
               latency={bb: v["arms"] for bb, v in B["latency"].items()})
    fit_summary = {}
    for k, v in B["fits"].items():
        if isinstance(v, dict) and v and all(isinstance(x, dict) or x is None for x in v.values()) and k not in ("tli", "dsrl", "cl"):
            fs = [x for x in v.values() if x]
            tr, n1 = s_(fs, "train_s")
            wl, n2 = s_(fs, "wall_s")
            fit_summary[k] = dict(units=len(v), found=len(fs), train_s_mean=tr / max(n1, 1), job_s_mean=wl / max(n2, 1),
                                  train_n=n1, job_n=n2)
    fit_summary["dsrl"], fit_summary["cl"], fit_summary["tli"] = B["fits"]["dsrl"], B["fits"]["cl"], B["fits"]["tli"]
    res["fit_summary"] = fit_summary
    json.dump(res, open(os.path.join(R, "h2_cost.json"), "w"), indent=1, default=str)
    for k, v in fit_summary.items():
        print("[h2 fit]", k, json.dumps(v, default=str)[:240], flush=True)
    for r in rows + orows:
        print(f"[h2] {r.get('policy', ''):7} {r['method'][:48]:48} {r['k']:4}/{r['n']:4} bytes/task {mb(r['bytes_task']):>9} "
              f"fit/task {fmin(r.get('fit_job_min_task')):>6} min practice {r['practice_task']:6.1f} demos {r['demos_task']:.2f} "
              f"lat {lat_s(r['latency'])}", flush=True)
    write_doc(B, rows, orows, fit_summary)
    if record:
        Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
        kw = dict(seed=0, script="experiments/h2_cost.py", command=" ".join(sys.argv), bytes=Z)
        print(ledger.record("h2_space", bench="libero", tag="units", metrics=dict(units=B["U"], er_bytes=B["er_bytes"]), config={}, **kw))
        print(ledger.record("h2_fit", bench="libero", tag="units", metrics=dict(summary=fit_summary, fits=B["fits"]), config={}, **kw))
        print(ledger.record("h2_practice", bench="libero", tag="counts", metrics=dict(libero90=B["practice"], ood=B["ood_practice"]),
                            config={}, **kw))
        print(ledger.record("h2_cost", bench="libero_90", tag="libero90", metrics=dict(rows=rows), config={}, **kw))
        print(ledger.record("h2_cost", bench="libero_ood", tag="ood", metrics=dict(rows=orows), config={}, **kw))


def write_doc(B, rows, orows, fs):
    U = B["U"]
    lat = B["latency"]
    unit_rows = []
    for bb in BACKBONES:
        unit_rows.append(f"| {NICE[bb]} | {U[f'prototype_r4/{bb}']['params']:,} / {mb(U[f'prototype_r4/{bb}']['bf16_bytes'])} | "
                         f"{U[f'lora_r16/{bb}']['params']:,} / {mb(U[f'lora_r16/{bb}']['bf16_bytes'])} | "
                         + (f"{U['lora_r4/pi05_libero']['params']:,} / {mb(U['lora_r4/pi05_libero']['bf16_bytes'])}" if bb == "pi05_libero" else "n/a")
                         + f" | {' x '.join(map(str, U[f'tli_latent/{bb}']['shape']))} = {U[f'tli_latent/{bb}']['params']:,} / "
                         f"{mb(U[f'tli_latent/{bb}']['bf16_bytes'])} | 0 |")
    fit_rows = []
    for k, lab in (("protos", "prototype r4, pi0.5"), ("protos/pi0_libero", "prototype r4, pi0"), ("protos/smolvla", "prototype r4, SmolVLA"),
                   ("protos/xvla", "prototype r4, X-VLA"), ("loras", "per-task LoRA r16, pi0.5"), ("loras_r4", "per-task LoRA r4, pi0.5"),
                   ("loras/pi0_libero", "per-task LoRA r16, pi0"), ("loras/smolvla", "per-task LoRA r16, SmolVLA"),
                   ("loras/xvla", "per-task LoRA r16, X-VLA"), ("strap", "STRAP LoRA r16 (2,000 steps), pi0.5"),
                   ("w1", "W1 stream prototypes, pi0.5")):
        v = fs.get(k)
        if v:
            fit_rows.append(f"| {lab} | {v['found']} | {v['train_s_mean'] / 60:.1f} | {v['job_s_mean'] / 60:.1f} |")
    for k, v in fs["dsrl"].items():
        fit_rows.append(f"| DSRL {k} actor (practice + SAC, wall) | {v['runs']} | n/a | {v['wall_s'] / v['runs'] / 60:.1f} |")
    for k, v in fs["cl"].items():
        fit_rows.append(f"| {'sequential FT' if k == 'seq' else 'experience replay'} (whole 30-task stream) | 1 | "
                        f"{v['train_s'] / 60:.1f} | {v['wall_s'] / 60 if v['wall_s'] else float('nan'):.1f} |")
    for bb, v in fs["tli"].items():
        fit_rows.append(f"| TLI latent identification, {NICE[bb]} | {v['latents']} | n/a | {v['wall_s_per_latent'] / 60:.1f} |")
    lrows = []
    for bb in BACKBONES:
        if bb not in lat:
            lrows.append(f"| {NICE[bb]} | pending |" + " |" * 7)
            continue
        a = lat[bb]["arms"]
        cell = lambda k: (f"{a[k]['ms_mean']:.0f} / {a[k]['hz']:.1f} / {a[k]['peak_alloc_gb']:.1f}" if k in a else "n/a")   # noqa: E731
        lrows.append(f"| {NICE[bb]} | " + " | ".join(cell(k) for k in ("policy", "switch_carry", "tli", "prototype", "lora16", "lora4", "dsrl",
                                                                        "policy_wrapped")) + " |")
    po = B["ood_practice"]
    doc = f"""# Cost: accuracy, space and time of every method

Generated by `experiments/h2_cost.py` (space, fit time, practice, accuracy; login node) and `experiments/h2_latency.py` (latency,
one A40 per backbone); figures `figures/robot/h2_pareto_libero90.png` and `figures/robot/h2_pareto_ood.png`. Every cell is in the
ledger: the tables below are `ledger/h2_cost` tags `libero90` and `ood` (one row per method, with the source rows it was read
from), built from `ledger/h2_space` tag `units`, `ledger/h2_fit` tag `units`, `ledger/h2_practice` tag `counts` and
`ledger/h2_latency` tags `pi05_libero`, `pi0_libero`, `smolvla`, `xvla`; the success counts come from the read row named in the
last column. Reported the way robotics papers report cost: OpenVLA-OFT's latency and Hz (batch 1, 100 queries after a warm-up,
one GPU), CLARE's parameter growth per task (here bytes per task and cumulative over the task set), SmolVLA's GPU time (here A40
minutes).

Conventions. Bytes are trainable parameters at 2 bytes (bf16; the adapters are stored in fp16, the same width; the TLI latents
and DSRL actors are stored in fp32 and charged at bf16), plus stored data at its size on disk. "Per task" divides the cumulative
store by the tasks in the set; shared units (prototypes, TLI source latents) are counted once. Fit time is the slurm job's wall
time on an A40 (model load, prefix caching and training), divided over the tasks; for DSRL it is the wall time of its 100 practice
episodes with SAC updates (servers sharing a GPU). Practice episodes count every rollout a method ran before the test,
including the policy's own 10 acceptance episodes where its rule compares against the policy. Demonstrations count the distinct
demonstrations a method trained on, per task (a prototype consumes one demonstration from each task that shares its concept,
some outside the test set). Latency is per action chunk (one query): pi0 / pi0.5 chunks of 10 actions, SmolVLA 50, X-VLA 30;
a method that serves different units on different tasks reports the task-weighted mean of the units' latencies and the largest
peak memory.

## Stored units (parameters / bf16 bytes)

| policy | prototype (rank-4 LoRA) | per-task LoRA rank 16 | per-task LoRA rank 4 | TLI latent per source task (layers x 9 x width) | pointer |
|---|---|---|---|---|---|
{chr(10).join(unit_rows)}

DSRL actor (authors' configuration, 7-dim noise): {U['dsrl_actor/dsrlz']['params']:,} parameters, {mb(U['dsrl_actor/dsrlz']['bf16_bytes'])}
(our 320-dim variant {U['dsrl_actor/dsrl']['params']:,}). STRAP's retrieval prior: the {U['strap_prior']['d40_episodes']:,} training
demonstrations it retrieves from ({mb(U['strap_prior']['d40_parquet_bytes'])} as parquet) plus its DINOv2 indexes and frame dump
({mb(sum(U['strap_prior']['indexes'].values()))}). Experience replay keeps each arrival's demonstration: {len(B['er_bytes'])}
episodes, {mb(sum(B['er_bytes'].values()))} as stored (parquet + two camera videos){(' (no recorded demo episode, charged the mean: tasks ' + str(B['er_missing']) + ')') if B['er_missing'] else ''}.
The W1 learner keeps the demonstrations it requested (its refits use all of a concept's demonstrations), charged the same way.

## Fit time per unit (A40)

| unit | units timed | train min (mean) | job min (mean) |
|---|---|---|---|
{chr(10).join(fit_rows)}

## Test-time latency (ms per chunk / Hz / peak GB allocated; batch 1, 100 queries after 10 warm-up, A40)

| policy | alone | grasp switch | TLI | prototype on | per-task LoRA r16 | LoRA r4 | DSRL actor | adapters injected, off |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(lrows)}

Reading the latency. Every adapter is served as an unmerged keyed LoRA (KeyedLoRA wrappers on every Linear of the action
expert, eager PyTorch), so switching one on adds the low-rank matmuls of every layer: about +50% per chunk on pi0 / pi0.5 and
SmolVLA, +20% on X-VLA; with the adapters injected and all off the overhead is under 2%. A single always-on adapter (the per-task
LoRA, a leaf, STRAP, the continual-learning adapters) could be merged into the weights and then costs exactly the policy's
latency; a prototype switched at the grasp could be merged and unmerged at the switch. The operators on held concepts (the grasp
switch, TLI) cost nothing measurable: the prefix is re-encoded every query anyway and the TLI shift is one addition per layer.

## LIBERO-90: tasks never seen, pi0.5

{md_table(rows)}

Notes: {' '.join(f"{r['method']}: {r['note']}." for r in rows if r['note'])}

## Li's compositional OOD tasks (no demonstrations), four flow policies

{md_table(orows, ood=True)}

Practice episodes per backbone (20 tasks; ledger practice rows g10_s1/s2/acc, g14_s1/s2, g15_acc): {', '.join(f"{NICE[b]} {po[b]['system']} (switch {po[b]['switch']}, TLI {po[b]['tli']})" for b in BACKBONES)}.
Notes: {' '.join(sorted({r['note'] for r in orows if r['note']}))}.
"""
    open(os.path.join(ROOT, "docs", "COST.md"), "w", encoding="utf-8").write(doc)


if __name__ == "__main__":
    main("--record" in sys.argv)
