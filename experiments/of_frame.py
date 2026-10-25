"""PROBE OF (docs/PLAN_OF.md; pre-registration experiments/prereg/pof_frame.py, ledger m3_prereg tag pof_frame): do object-frame
action prototypes exist inside an off-the-shelf post-trained flow VLA, and is the model's own grounding a usable internal anchor?

  D1  object-frame transfer of approach behaviour (part-prompt modes, instruction modes, own samples) across scenes, in the world /
      object / control frame, on pre-grasp demo frames of each model's TRAINING tasks (sim poses = ORACLE, diagnostic only).
  D2  word -> image-patch attention of the instruction's object (goal) words, localisation inside the named object's (own target's)
      ORACLE region (R3b segmentation), on R2a's zs observations: success vs wrong-object vs wrong-goal.
  D3  what / where in the vision tower: region-pooled SigLIP (and early VLM, null prompt) features identify an object across scenes and
      positions (leave-scene-out prototypes, training scenes only); the per-token match to the named object's prototype is a
      vision-only anchor, scored on R2a's zs observations like D2.

Models: pi05_l90c@5000 (tag l90c) and the public pi05_libero (tag libero).  Offline only: no rollout, no demo executed or replayed (demo
STATES are set in the simulator only to read poses / segmentation, as pp_prep.py poses does).

Stages (each its own SLURM job; nothing here runs on the login node):
  --index  --model m            (chunks env, CPU)  training tasks, demo episode, phase-0 candidate frames -> results/of/index_<m>.json
  --poses  --shard i/n          (LIBERO env, EGL)  poses (pos + quat), eef, segmentation, a 64-px sim render -> results/of/poses/
  --select --model m            (chunks env)       pre-grasp frames (hand >= 5 cm xy from the named object), <= 12 per task
  --gpu    --model m --shard    (openpi GPU)       modes / samples / region-pooled features at every selected frame -> obs_<m>/
  --protos --model m            (chunks env)       identity prototypes per object type from training frames -> protos_<m>.npz
  --dev    --model m --shard    (openpi GPU)       R2a zs observations: attention maps, own samples, prototype similarity -> dev_<m>/
  --sanity                      (chunks env)       P2's transfer statistic recomputed from P2's stored files vs its ledger value
  --read [--record]             (chunks env)       D1 / D2 / D3 -> results/of/read.json, ledger of_read
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")

OUT = os.path.join(ROOT, "results", "of")
PROTOS = os.path.join(ROOT, "results", "protos")
P0 = os.path.join(ROOT, "results", "robot", "p0_demo")
R2A_OUT = os.path.join(ROOT, "results", "r2a")
RS3_OUT = os.path.join(ROOT, "results", "rs3")
MODELS = {
    "l90c": dict(cfg="pi05_l90c", ckpt="/coc/flash10/ksingara3/robot/checkpoints_l90c/pytorch/pi05_l90c_5000",
                 pp_index="index_l90c.json", cfgname="pi05_l90c_l90c", pp_obs="obs_l90c", pp2_tag="l90c"),
    "libero": dict(cfg="pi05_libero", ckpt=None, pp_index="index.json", cfgname="pi05_libero", pp_obs="obs", pp2_tag="pi05_libero"),
}
# ── settings (fixed before any model output was computed; docs/PLAN_OF.md) ──────────────────────────────────────────
STRIDE = 3                      # candidate frames: every 3rd phase-0 frame of the demo
HAND_CLEAR = 0.05               # pre-grasp: the hand >= 5 cm (xy) from the NAMED object (R2a's clearance)
MAX_PER_TASK = 12               # evenly spaced among the eligible frames
LEVELS = (0.1, 0.3, 0.55)       # mode levels (R2a's), ascent as pp_fields (Adam, 80 iterations, lr 0.02)
N_OWN = 8                       # own samples under the instruction, seed 21
N_DEV_SAMP, DEV_SEED = 12, 11   # dev observations: R2a's policy_sample draws
MAX_ALT = 2                     # other training instructions of the same scene naming the same object type
COV_MIN = 0.05                  # a token belongs to a body's region if >= 5% of its pixels show it (R3b)
FEAT_LAYERS = {"l0": 1, "l2": 3}   # VLM image tokens under the null prompt: the INPUT of layer k = the output of layer k-1
D2_LAYERS = (0, 1, 2)           # primary layer set of the word -> patch attention (L0b: words live at their positions to layer 1)
N_LAYERS = 18
B_BOOT = 2000
SOURCES = ("part", "part_plain", "part_tmpl", "instr", "instr_alt", "own", "null")
COND_OF = {"part": "obj:named", "part_plain": "obj:named:plain", "part_tmpl": "obj:named:tmpl", "instr": "instr", "null": "null"}
FRAMES = ("world", "object", "object_trans", "control")
EXCLUDE = "sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw"


def slug(tid):
    return tid.replace(":", "_")


def save_atomic(path, **arrs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def jload(path):
    return json.load(open(path, encoding="utf-8"))


def pose_key(tid, ep, f):
    return f"{slug(tid)}_e{ep}_f{f}"


def h8(s):
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


# ── index (chunks env) ─────────────────────────────────────────────────────────────────────────────────────────────
def index(model):
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy import factors as F
    from lifelong_ood.protos import store as S
    M = MODELS[model]
    allt = F.load()
    tasks = {t.tid: t for t in allt}
    pp = jload(os.path.join(PROTOS, M["pp_index"]))
    demos, d90 = D.Demos(), D.Demos90()
    if model == "l90c":
        from lifelong_ood import l90c
        train = sorted(l90c.load_split()["train"])
    else:
        texts = set(demos.tasks[k] for k in range(40))
        train = sorted(t.tid for t in allt if t.suite != "libero_90" and t.language in texts)
    first = {}
    for o in pp["obs"]:
        if o["set"] == "train":
            first.setdefault(f"{o['suite']}:{o['index']}", o)
    out = []
    for tid, o in sorted(first.items()):
        t = tasks[tid]
        assert tid in train, tid
        pt = S.parts_of(t, F.scene_phrases(t, allt))
        inst = pt["obj_inst"]
        from lifelong_ood.policy import primitives as P
        first_verb = S.verb_of(P.steps_of(t.language)[0].strip())
        if not inst or inst not in t.objects or not pt["goal_type"] or first_verb not in ("put", "pick up", "stack"):
            print(f"[of index] {tid}: no pick-and-place first step ({inst}, first verb '{first_verb}'); skipped", flush=True)
            continue
        ipec = o.get("src") == "ipec"
        if ipec:
            mp, ldr = os.path.join(P0, f"{o['l90_task']}_match.npz"), d90
        else:
            mp, ldr = os.path.join(PROTOS, "match", f"tr{o.get('src_hf', o['hf_task']):02d}.npz"), demos
        if not os.path.exists(mp):
            print(f"[of index] {tid}: no raw match ({mp}); skipped", flush=True)
            continue
        ep = ldr.load(o["episode"])
        A = ep["actions"]
        T = len(A)
        close = np.flatnonzero(A[:, 6] > 0)
        g = int(close[0]) if len(close) else T
        frames = [int(f) for f in range(0, min(g, T - 10 + 1), STRIDE)]
        conds = {c["name"]: c for c in o["conds"]}
        keep = [conds[n] for n in ("null", "instr", "obj:named", "obj:named:plain", "obj:named:tmpl") if n in conds]
        alts = []
        for t2 in sorted(train):
            u = tasks[t2]
            if t2 == tid or u.scene != t.scene or u.language == t.language:
                continue
            if S.parts_of(u, F.scene_phrases(u, allt))["obj_type"] == pt["obj_type"] and u.language not in [a.prompt for a in alts]:
                alts.append(S.Condition(f"instr:alt{len(alts)}", "instr", u.language))
        keep += [a.to_json() for a in alts[:MAX_ALT]]
        goal_inst = {a[2] for a in t.goal if len(a) == 3} | {a[1] for a in t.goal if len(a) >= 2}
        movable = sorted(n for n in t.objects if n != inst and n not in goal_inst)
        pool = movable or sorted(n for n in t.fixtures if n not in goal_inst)
        ctrl = pool[h8("ofctrl:" + tid) % len(pool)] if pool else None
        out.append(dict(tid=tid, suite=t.suite, index=t.index, scene=t.scene, language=t.language, cat=pt["obj_type"], inst=inst,
                        goal_type=pt["goal_type"], ctrl=ctrl, ipec=ipec, episode=int(o["episode"]), match=os.path.relpath(mp, ROOT),
                        T=T, first_close=g, frames=frames, conds=keep, objects=dict(t.objects), fixtures=dict(t.fixtures)))
        print(f"[of index] {tid} ({t.scene}) '{t.language}': {pt['obj_type']} / {inst}, ctrl {ctrl}, {len(frames)} candidate frames, "
              f"alts {[a.prompt for a in alts[:MAX_ALT]]}", flush=True)
    dump(os.path.join(OUT, f"index_{model}.json"), dict(model=model, tasks=out))
    print(f"[of index] {model}: {len(out)} tasks, {sum(len(x['frames']) for x in out)} candidate frames", flush=True)


# ── poses (LIBERO env, EGL) ───────────────────────────────────────────────────────────────────────────────────────
def poses(shard):
    import pathlib
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import libero_client as LC
    i, n = map(int, shard.split("/"))
    jobs = {}
    for m in MODELS:
        p = os.path.join(OUT, f"index_{m}.json")
        if os.path.exists(p):
            for x in jload(p)["tasks"]:
                jobs.setdefault((x["tid"], x["episode"]), x)
    od = os.path.join(OUT, "poses")
    os.makedirs(od, exist_ok=True)
    rp = ("robot0_", "gripper0_", "mount0_")
    for j, ((tid, e), x) in enumerate(sorted(jobs.items())):
        if j % n != i:
            continue
        todo = [f for f in x["frames"] if not os.path.exists(os.path.join(od, pose_key(tid, e, f) + ".npz"))]
        if not todo:
            continue
        m = np.load(os.path.join(ROOT, x["match"]))
        task = benchmark.get_benchmark_dict()[x["suite"]]().get_task(int(x["index"]))
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=64, camera_widths=64)
        env.seed(0)
        env.reset()
        inner = env.env
        sim = inner.sim
        names = list(inner.objects_dict.keys()) + list(inner.fixtures_dict.keys())
        kinds = ["object"] * len(inner.objects_dict) + ["fixture"] * len(inner.fixtures_dict)
        tables = LC.seg_tables(env)
        assert list(tables[0]) == names
        mm = sim.model
        robot = np.array([g for g in range(mm.ngeom) if (mm.body_id2name(int(mm.geom_bodyid[g])) or "").startswith(rp)])
        for f in todo:
            ob = env.set_init_state(m["states"][f])
            bid = [inner.obj_body_id[q] for q in names]
            P = np.asarray([sim.data.body_xpos[b] for b in bid], np.float64)
            Q = np.asarray([sim.data.body_xquat[b] for b in bid], np.float64)       # w, x, y, z
            eef = np.asarray(ob["robot0_eef_pos"], np.float64)
            rgb = sim.render(camera_name="agentview", width=256, height=256)
            thumb = rgb.reshape(64, 4, 64, 4, 3).mean((1, 3)).astype(np.uint8)
            old = mm.geom_group[robot].copy()
            mm.geom_group[robot] = 5
            try:
                cov = LC.seg_maps(env, tables, "agentview").astype(np.float16)       # [N, 256], client orientation
            finally:
                mm.geom_group[robot] = old
            save_atomic(os.path.join(od, pose_key(tid, e, f) + ".npz"), names=np.array(names), kinds=np.array(kinds), pos=P, quat=Q,
                        eef=eef, cov=cov, thumb=thumb)
        env.close()
        print(f"[of poses] {tid} e{e}: {len(todo)} frames, {len(names)} bodies", flush=True)


# ── select (chunks env) ───────────────────────────────────────────────────────────────────────────────────────────
def select(model):
    ix = jload(os.path.join(OUT, f"index_{model}.json"))
    sel, miss = [], 0
    for x in ix["tasks"]:
        ok = []
        for f in x["frames"]:
            p = os.path.join(OUT, "poses", pose_key(x["tid"], x["episode"], f) + ".npz")
            if not os.path.exists(p):
                miss += 1
                continue
            z = np.load(p, allow_pickle=True)
            names = [str(q) for q in z["names"]]
            d = float(np.linalg.norm(z["pos"][names.index(x["inst"])][:2] - z["eef"][:2]))
            if d >= HAND_CLEAR:
                ok.append(f)
        if len(ok) > MAX_PER_TASK:
            ok = [ok[q] for q in np.unique(np.linspace(0, len(ok) - 1, MAX_PER_TASK).round().astype(int))]
        for f in ok:
            sel.append(dict(key=pose_key(x["tid"], x["episode"], f), tid=x["tid"], frame=f))
    cats = {}
    for x in ix["tasks"]:
        if any(s["tid"] == x["tid"] for s in sel):
            cats.setdefault(x["cat"], set()).add(x["scene"])
    multi = {c: sorted(s) for c, s in cats.items() if len(s) >= 2}
    dump(os.path.join(OUT, f"sel_{model}.json"), dict(model=model, obs=sel, multi_scene_cats=multi, missing_poses=miss))
    print(f"[of select] {model}: {len(sel)} frames over {len({s['tid'] for s in sel})} tasks; missing poses {miss}; "
          f"types in >= 2 scenes: { {c: len(s) for c, s in multi.items()} }", flush=True)


# ── GPU helpers (openpi env) ──────────────────────────────────────────────────────────────────────────────────────
def backbone(model):
    from lifelong_ood.policy.backbone import PolicyBackbone
    M = MODELS[model]
    return PolicyBackbone(M["cfg"], checkpoint_dir=M["ckpt"], device="cuda")


def vision_feats(bk, obs_null):
    """-> {"sig": [256, D] SigLIP (projected) agentview tokens, "l0"/"l2": VLM image tokens under the null prompt}, and the null prefix."""
    import torch
    m = bk.model
    with torch.no_grad():
        images, img_masks, lang_tokens, lang_masks, _ = m._preprocess_observation(obs_null, train=False)
        embs, _, _ = m.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        out = {"sig": embs[0, :256].float().cpu().numpy()}
        cap = {}

        def fn(i, h, span):
            for k, li in FEAT_LAYERS.items():
                if i == li:
                    cap[k] = h[0, :256].float().cpu().numpy()
            return None
        hs = bk.cond_hook(fn)
        try:
            pre_n = bk.prefix(obs_null)
        finally:
            for q in hs:
                q.remove()
    out.update(cap)
    return out, pre_n


def pool(F_, cov):
    """region-pooled features: [N bodies, D] (zeros where a body covers no token), token counts [N]."""
    msk = (np.asarray(cov, np.float32) >= COV_MIN).astype(np.float32)
    cnt = msk.sum(1)
    P = (msk @ F_) / np.maximum(cnt, 1)[:, None]
    return P.astype(np.float16), cnt.astype(np.int16)


def gpu(model, shard):
    import torch
    from lifelong_ood.policy import data as D
    from lifelong_ood.protos import store as S
    M = MODELS[model]
    ix = {x["tid"]: x for x in jload(os.path.join(OUT, f"index_{model}.json"))["tasks"]}
    sel = jload(os.path.join(OUT, f"sel_{model}.json"))["obs"]
    i, n = map(int, shard.split("/"))
    mine = sel[i::n] if n > 0 else sel[:2]
    smoke = n == 0
    od = os.path.join(OUT, f"obs_{model}" if not smoke else f"scratch/smoke_{model}")
    bk = backbone(model)
    demos, d90 = D.Demos(), D.Demos90()
    splice = S.RowSplice(bk, M["cfgname"], demos)
    print(f"[of gpu] {bk.describe()}; {len(mine)} frames, shard {shard}", flush=True)
    cache, t_all, done = {}, time.time(), 0
    for s in mine:
        outp = os.path.join(od, s["key"] + ".npz")
        if os.path.exists(outp):
            continue
        t0 = time.time()
        x = ix[s["tid"]]
        ck = (x["ipec"], x["episode"])
        if ck not in cache:
            cache.clear()
            cache[ck] = (d90 if x["ipec"] else demos).load(x["episode"], **({} if x["ipec"] else {"images": True}))
        ep = cache[ck]
        odict = (d90 if x["ipec"] else demos).observation(ep, s["frame"])
        odict = dict(odict, prompt=x["language"])
        z = np.load(os.path.join(OUT, "poses", s["key"] + ".npz"), allow_pickle=True)
        conds = [S.Condition.from_json(c) for c in x["conds"]]
        with torch.no_grad():
            live = S.Live(bk, odict, conds, splice, block=16)
            pre_i = bk.prefix(bk.observation(odict))
            feats, pre_n = vision_feats(bk, bk.observation(dict(odict, prompt="")))
            drafts = np.concatenate([bk.sample(2, pre_i, seed=11), bk.sample(2, pre_n, seed=12)])
            starts = S.shared_starts(drafts, LEVELS, seed=0)
            Mo = live.modes(starts, iters=80, lr=0.02, probes=1)
            C, Sn = len(conds), len(drafts)
            raw = np.stack([bk.unnormalize(Mo[u]["x0"].reshape(C * Sn, -1)).reshape(C, Sn, bk.H, -1) for u in LEVELS])
            own = bk.unnormalize(bk.sample(N_OWN, pre_i, seed=21)).reshape(N_OWN, bk.H, -1)
        img = np.asarray(odict["observation/image"], np.float32)
        th = img.reshape(64, 4, 64, 4, 3).mean((1, 3))
        sim = z["thumb"].astype(np.float32)
        variants = {"id": sim, "flip_both": sim[::-1, ::-1], "flip_ud": sim[::-1], "flip_lr": sim[:, ::-1]}
        align = {k: float(np.corrcoef(th.reshape(-1), v.reshape(-1))[0, 1]) for k, v in variants.items()}
        pf = {}
        for k, Fk in feats.items():
            pf[f"feat_{k}"], cnt = pool(Fk, z["cov"])
        save_atomic(outp, names=np.array([c.name for c in conds]), levels=np.array(LEVELS), x0_raw=raw.astype(np.float32),
                    own_raw=own.astype(np.float32), draft_raw=bk.unnormalize(drafts).astype(np.float32), bodies=z["names"],
                    ntok=cnt, align=np.array(json.dumps(align)), **pf)
        done += 1
        print(f"[of gpu] {s['key']}: {C} conditions, {time.time() - t0:.1f}s ({done} done, {time.time() - t_all:.0f}s); align "
              f"{ {k: round(v, 3) for k, v in align.items()} }" if smoke or done <= 3 else
              f"[of gpu] {s['key']}: {time.time() - t0:.1f}s ({done} done)", flush=True)
    print(f"[of gpu] shard {shard} finished: {done} new", flush=True)


# ── identity prototypes (chunks env) ──────────────────────────────────────────────────────────────────────────────
def train_rows(model):
    """Every movable object's region-pooled features at every training frame of `model` -> list of dict."""
    ix = {x["tid"]: x for x in jload(os.path.join(OUT, f"index_{model}.json"))["tasks"]}
    rows = []
    for s in jload(os.path.join(OUT, f"sel_{model}.json"))["obs"]:
        p = os.path.join(OUT, f"obs_{model}", s["key"] + ".npz")
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        q = np.load(os.path.join(OUT, "poses", s["key"] + ".npz"), allow_pickle=True)
        x = ix[s["tid"]]
        cov = (q["cov"].astype(np.float32) >= COV_MIN)
        tok = np.stack(np.meshgrid(np.arange(16), np.arange(16), indexing="ij"), -1).reshape(256, 2).astype(np.float64)
        for b, nm in enumerate([str(u) for u in z["bodies"]]):
            if nm not in x["objects"] or int(z["ntok"][b]) < 2:
                continue
            rows.append(dict(key=s["key"], tid=s["tid"], scene=x["scene"], inst=nm, cat=x["objects"][nm], ntok=int(z["ntok"][b]),
                             rc=tok[cov[b]].mean(0), xy=q["pos"][b][:2], feats={k: z[f"feat_{k}"][b].astype(np.float32)
                                                                                for k in ("sig", "l0", "l2")}))
    return rows


def unit(v):
    v = np.asarray(v, np.float64)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def protos(model):
    rows = train_rows(model)
    cats = sorted({r["cat"] for r in rows})
    out = {}
    for k in ("sig", "l0", "l2"):
        out[k] = np.stack([unit(np.mean([unit(r["feats"][k]) for r in rows if r["cat"] == c], 0)) for c in cats]).astype(np.float32)
    cnt = [sum(r["cat"] == c for r in rows) for c in cats]
    sc = [len({r["scene"] for r in rows if r["cat"] == c}) for c in cats]
    save_atomic(os.path.join(OUT, f"protos_{model}.npz"), cats=np.array(cats), count=np.array(cnt), scenes=np.array(sc), **out)
    print(f"[of protos] {model}: {len(rows)} object regions, {len(cats)} types: {dict(zip(cats, cnt))}", flush=True)


# ── dev observations (openpi GPU): attention, own samples, prototype similarity ─────────────────────────────────────
class AttnCap:
    """Wraps transformers' Gemma eager attention: while `store` is a dict, the VLM layers' post-softmax weights [heads, Q, K] are
    kept by layer index.  The policy's outputs are unchanged (the original function runs and its result is returned)."""

    def __init__(self, bk):
        from transformers.models.gemma import modeling_gemma as G
        self.ids = {id(ly.self_attn): li for li, ly in enumerate(bk.model.paligemma_with_expert.paligemma.language_model.layers)}
        self.store = None
        orig = G.eager_attention_forward
        cap = self

        def f(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
            out = orig(module, query, key, value, attention_mask, scaling, dropout, **kw)
            if cap.store is not None and id(module) in cap.ids:
                cap.store[cap.ids[id(module)]] = out[1].detach()[0].float()
            return out
        G.eager_attention_forward = f


def dev_meta():
    import r2a_null_modes as R2A
    meta = {}
    for tid, x in R2A.load_index().items():
        for e in x["episodes"]:
            for m in e["obs"]:
                meta[m["key"]] = dict(m, task=tid, trial=e["trial"], type=e["type"], success=e["success"])
    return meta


def dev(model, shard):
    import torch
    import rs3_steer as RS
    meta = dev_meta()
    keys = sorted(meta)
    i, n = map(int, shard.split("/"))
    smoke = n == 0
    mine = keys[:2] if smoke else keys[i::n]
    od = os.path.join(OUT, f"dev_{model}" if not smoke else f"scratch/smoke_dev_{model}")
    pz = np.load(os.path.join(OUT, f"protos_{model}.npz"), allow_pickle=True)
    PR = {k: torch.as_tensor(pz[k], device="cuda") for k in ("sig", "l0", "l2")}
    bk = backbone(model)
    cap = AttnCap(bk)
    sp = RS._tokenizer()
    tc, done, t_all = {}, 0, time.time()
    print(f"[of dev] {bk.describe()}; {len(mine)} observations, shard {shard}", flush=True)
    for key in mine:
        outp = os.path.join(od, key + ".npz")
        if os.path.exists(outp):
            continue
        t0 = time.time()
        z = np.load(os.path.join(R2A_OUT, "obs", key + ".npz"), allow_pickle=True)
        text = str(z["text"])
        if text not in tc:
            tc[text] = RS.role_rows(sp, text)
        ids, rows, parse = tc[text]
        odict = {"observation/image": z["image"], "observation/wrist_image": z["wrist"],
                 "observation/state": np.asarray(z["state"], np.float32), "prompt": text}
        with torch.no_grad():
            obs = bk.observation(odict)
            nt = int(obs.tokenized_prompt_mask.sum())
            assert list(np.asarray(obs.tokenized_prompt[0][:nt].cpu())) == ids, key
            off = int(bk.text_span(obs)[0])
            cap.store = {}
            try:
                pre = bk.prefix(obs)
            finally:
                st, cap.store = cap.store, None
            assert len(st) == N_LAYERS, len(st)
            arr = {}
            for role in ("O", "G"):
                r = [off + q for q in rows[role]]
                if not r:
                    arr[f"w2i_{role}"] = np.zeros((N_LAYERS, 256), np.float32)
                    arr[f"i2w_{role}"] = np.zeros((N_LAYERS, 256), np.float32)
                    continue
                arr[f"w2i_{role}"] = np.stack([st[l][:, r][:, :, :256].mean((0, 1)).cpu().numpy() for l in range(N_LAYERS)])
                arr[f"i2w_{role}"] = np.stack([st[l][:, :256][:, :, r].sum(-1).mean(0).cpu().numpy() for l in range(N_LAYERS)])
            X = bk.sample(N_DEV_SAMP, pre, seed=DEV_SEED)
            samp = bk.unnormalize(X).reshape(N_DEV_SAMP, bk.H, -1)
            feats, _ = vision_feats(bk, bk.observation(dict(odict, prompt="")))
            for k, Fk in feats.items():
                Ft = torch.nn.functional.normalize(torch.as_tensor(Fk, device="cuda"), dim=-1)
                arr[f"sim_{k}"] = (PR[k] @ Ft.T).cpu().numpy().astype(np.float16)                   # [K types, 256]
        save_atomic(outp, samples=samp.astype(np.float32), parse=np.array(json.dumps(dict(rows=rows, parse=parse))), **arr)
        done += 1
        if smoke or done <= 3:
            print(f"[of dev] {key}: rows {rows} parse {parse}; {time.time() - t0:.1f}s", flush=True)
    print(f"[of dev] shard {shard} finished: {done} new ({time.time() - t_all:.0f}s)", flush=True)


# ── statistics ────────────────────────────────────────────────────────────────────────────────────────────────────
def corr2(A, B):
    """P2's statistic (pp2_dev.corr2): Pearson r of x and of y, averaged."""
    A, B = np.asarray(A), np.asarray(B)
    rs = [np.corrcoef(A[:, k], B[:, k])[0, 1] for k in range(2) if A[:, k].std() > 1e-9 and B[:, k].std() > 1e-9]
    return float(np.mean(rs)) if rs else float("nan")


def yaw_of(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def design(r, frame):
    """The frame's 3-parameter approach model d = lam (c - h) + R b (xy): rows [2, 3]."""
    h = r["h"]
    if frame == "world":
        c, R = np.zeros(2), np.eye(2)
    elif frame == "object":
        c, R = r["o"], rot(r["o_yaw"])
    elif frame == "object_trans":
        c, R = r["o"], np.eye(2)
    else:
        c, R = r["c"], rot(r["c_yaw"])
    return np.column_stack([c - h, R])


def lso_predict(rows, frame):
    """Leave-scene-out predictions of the displacement for every row (per object type); rows of types in < 2 scenes get None."""
    pred = [None] * len(rows)
    by = {}
    for j, r in enumerate(rows):
        by.setdefault(r["cat"], []).append(j)
    for cat, js in by.items():
        scenes = sorted({rows[j]["scene"] for j in js})
        if len(scenes) < 2:
            continue
        for s in scenes:
            tr = [j for j in js if rows[j]["scene"] != s]
            X = np.concatenate([design(rows[j], frame) for j in tr])
            y = np.concatenate([rows[j]["d"] for j in tr])
            th = np.linalg.lstsq(X, y, rcond=None)[0]
            for j in js:
                if rows[j]["scene"] == s:
                    pred[j] = design(rows[j], frame) @ th
    return pred


def transfer_stat(rows, pred, idx=None):
    """mean over object types (in >= 2 scenes of the sample) of P2's corr2 between the displacement and its prediction."""
    idx = np.arange(len(rows)) if idx is None else np.asarray(idx)
    by = {}
    for j in idx:
        if pred[j] is not None:
            by.setdefault(rows[j]["cat"], []).append(j)
    rs = {}
    for cat, js in by.items():
        if len({rows[j]["scene"] for j in js}) < 2 or len(js) < 5:
            continue
        rs[cat] = corr2([rows[j]["d"] for j in js], [pred[j] for j in js])
    v = [x for x in rs.values() if np.isfinite(x)]
    return (float(np.mean(v)) if v else float("nan")), rs


def boot_tasks(rows, preds, B=B_BOOT, seed=0):
    """Task-cluster bootstrap: resample tasks with replacement; -> {frame: samples [B]} of transfer_stat."""
    rng = np.random.default_rng(seed)
    tasks = sorted({r["tid"] for r in rows})
    of = {t: [j for j, r in enumerate(rows) if r["tid"] == t] for t in tasks}
    out = {f: np.empty(B) for f in preds}
    for b in range(B):
        pick = rng.integers(0, len(tasks), len(tasks))
        idx = np.concatenate([of[tasks[q]] for q in pick])
        for f, p in preds.items():
            out[f][b] = transfer_stat(rows, p, idx)[0]
    return out


def holm(ps, alpha=0.05):
    order = sorted(ps, key=lambda k: ps[k])
    passed, ok = {}, True
    for r, k in enumerate(order):
        ok = ok and ps[k] <= alpha / (len(ps) - r)
        passed[k] = bool(ok)
    return passed


def scale_for(model):
    import pp2_dev
    return np.asarray(pp2_dev.scale_from_demos(jload(os.path.join(PROTOS, MODELS[model]["pp_index"]))), np.float64)


def d1_rows(model, scale):
    from lifelong_ood.protos import store as S
    ix = {x["tid"]: x for x in jload(os.path.join(OUT, f"index_{model}.json"))["tasks"]}
    rows = {}
    align = []
    for s in jload(os.path.join(OUT, f"sel_{model}.json"))["obs"]:
        p = os.path.join(OUT, f"obs_{model}", s["key"] + ".npz")
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        q = np.load(os.path.join(OUT, "poses", s["key"] + ".npz"), allow_pickle=True)
        x = ix[s["tid"]]
        align.append(json.loads(str(z["align"])))
        names = [str(u) for u in q["names"]]
        o = names.index(x["inst"])
        c = names.index(x["ctrl"]) if x["ctrl"] in names else None
        h = np.asarray(q["eef"], np.float64)
        base = dict(key=s["key"], tid=s["tid"], scene=x["scene"], cat=x["cat"], h=h[:2], o=q["pos"][o][:2], o_yaw=yaw_of(q["quat"][o]),
                    c=q["pos"][c][:2] if c is not None else None, c_yaw=yaw_of(q["quat"][c]) if c is not None else None)
        cn = [str(u) for u in z["names"]]
        beh = {}
        for src, cname in COND_OF.items():
            if cname in cn:
                for l, u in enumerate(LEVELS):
                    beh[(src, u)] = z["x0_raw"][l, cn.index(cname)].mean(0)
        alts = [k for k, nm in enumerate(cn) if nm.startswith("instr:alt")]
        if alts:
            for l, u in enumerate(LEVELS):
                beh[("instr_alt", u)] = np.mean([z["x0_raw"][l, k].mean(0) for k in alts], 0)
        beh[("own", None)] = z["own_raw"].mean(0)
        for k, raw in beh.items():
            it = S.interpret(raw, h, q["pos"][[o]], [x["inst"]], scale)
            rows.setdefault(k, []).append(dict(base, d=(it["endpoint"] - h)[:2]))
    return rows, align


def read_d1(model):
    scale = scale_for(model)
    rows, align = d1_rows(model, scale)
    res = dict(scale=scale.tolist(), n_obs=len(rows.get(("own", None), [])), sources={})
    al = {k: float(np.mean([a[k] for a in align])) for k in align[0]} if align else {}
    best_share = float(np.mean([max(a, key=a.get) == "flip_both" for a in align])) if align else None
    res["image_alignment"] = dict(mean_corr=al, share_best_flip_both=best_share)
    for src in SOURCES:
        lv = [u for (s, u) in rows if s == src]
        if not lv:
            continue
        per = {}
        for u in lv:
            R = [r for r in rows[(src, u)] if r["c"] is not None]          # tasks without any control body drop out of every frame
            preds = {f: lso_predict(R, f) for f in FRAMES}
            pt = {f: transfer_stat(R, preds[f]) for f in FRAMES}
            centre = [r["o"] - r["h"] for r in R]
            per[str(u)] = dict(T={f: pt[f][0] for f in FRAMES}, per_cat={f: pt[f][1] for f in FRAMES},
                               p2_centre=transfer_stat(R, centre)[0], n=len(R), R=R, preds=preds)
        # P2's convention: each frame at its own best level (point estimate)
        best = {f: max(per, key=lambda u: (per[u]["T"][f] if np.isfinite(per[u]["T"][f]) else -9)) for f in FRAMES}
        T = {f: per[best[f]]["T"][f] for f in FRAMES}
        # bootstrap: the same task resample for every frame (each at its chosen level)
        R0 = per[best["object"]]["R"]
        same = all(per[best[f]]["R"] is R0 or len(per[best[f]]["R"]) == len(R0) for f in FRAMES)
        assert same
        bs = boot_tasks(R0, {f: per[best[f]]["preds"][f] for f in FRAMES})
        ci = {f: [float(np.nanpercentile(bs[f], 2.5)), float(np.nanpercentile(bs[f], 97.5))] for f in FRAMES}
        dw, dc = bs["object"] - bs["world"], bs["object"] - bs["control"]
        tab = {}
        for f in FRAMES:
            for cat, r in per[best[f]]["per_cat"][f].items():
                tab.setdefault(cat, dict(scenes=len({x["scene"] for x in R0 if x["cat"] == cat}),
                                         tasks=len({x["tid"] for x in R0 if x["cat"] == cat}),
                                         n=sum(x["cat"] == cat for x in R0)))[f] = r
        res["sources"][src] = dict(
            best_level=best, T=T, ci=ci, diff_obj_world=float(T["object"] - T["world"]),
            diff_obj_world_ci=[float(np.nanpercentile(dw, 2.5)), float(np.nanpercentile(dw, 97.5))],
            diff_obj_ctrl=float(T["object"] - T["control"]),
            diff_obj_ctrl_ci=[float(np.nanpercentile(dc, 2.5)), float(np.nanpercentile(dc, 97.5))],
            p_obj_le_world=float(np.mean(dw <= 0)), p_obj_le_ctrl=float(np.mean(dc <= 0)),
            p2_centre={u: d["p2_centre"] for u, d in per.items()},
            by_level={u: dict(T=d["T"], n=d["n"]) for u, d in per.items()}, per_object=tab, n=len(R0))
    return res


def ci_rate(per_task):
    import r2a_null_modes as R2A
    return R2A.ci_rate(per_task, B=10000, seed=0)


def region_of(g, names_list):
    cn = [str(q) for q in g["cand_names"]]
    rr = [g["region"][cn.index(q)] for q in names_list if q in cn]
    return (np.max(rr, 0) > 0) if rr else None


def proxy_label(samples, z, named):
    import r2a_null_modes as R2A
    cn = [str(q) for q in z["cand_names"]]
    pts = [R2A.pointed(r, z["state"][:3], z["cand_pos"])[0] for r in samples]
    tg = [cn[p] for p in pts if p >= 0]
    if not tg:
        return None
    vals, cnt = np.unique(tg, return_counts=True)
    top = vals[np.argmax(cnt)]
    return "hit" if top in named else "wrong"


def read_dev(model):
    """D2 (attention anchor) and D3 (vision anchor) on R2a's zs observations."""
    from lifelong_ood.policy import factors as F
    meta = dev_meta()
    tasks = {t.tid: t for t in F.load()}
    pz = np.load(os.path.join(OUT, f"protos_{model}.npz"), allow_pickle=True)
    cats = [str(c) for c in pz["cats"]]
    per_obs = []
    for key, m in sorted(meta.items()):
        p = os.path.join(OUT, f"dev_{model}", key + ".npz")
        gp = os.path.join(RS3_OUT, "regions", key + ".npz")
        if not os.path.exists(p) or not os.path.exists(gp):
            continue
        d = np.load(p, allow_pickle=True)
        g = np.load(gp, allow_pickle=True)
        z = np.load(os.path.join(R2A_OUT, "obs", key + ".npz"), allow_pickle=True)
        reg = region_of(g, m["named"])
        if reg is None or reg.sum() == 0:
            continue
        role = m["role"]
        rk = "O" if role == "pre" else "G"
        rows = json.loads(str(d["parse"]))["rows"][rk]
        rec = dict(key=key, task=m["task"], trial=m["trial"], role=role, type=m["type"], success=m["success"],
                   chance=float(reg.sum() / 256.0), proxy=proxy_label(d["samples"], z, set(m["named"])))
        if rows:
            W = d[f"w2i_{rk}"]
            I = d[f"i2w_{rk}"]
            a = W[list(D2_LAYERS)].mean(0)
            rec["att_hit"] = float(reg[int(np.argmax(a))])
            rec["att_mass"] = float(a[reg].sum() / max(a.sum(), 1e-12))
            rec["att_hit_layer"] = [float(reg[int(np.argmax(W[l]))]) for l in range(N_LAYERS)]
            rec["rev_hit_layer"] = [float(reg[int(np.argmax(I[l]))]) for l in range(N_LAYERS)]
            b = I[list(D2_LAYERS)].mean(0)
            rec["rev_hit"] = float(reg[int(np.argmax(b))])
        # D3 vision anchor: the named object's type prototype (pre-grasp: the named object; post-grasp: the own target, when it is an
        # object / fixture instance of the task)
        t = tasks[m["task"]]
        inst = {**t.objects, **t.fixtures}
        ncat = []
        for nm in m["named"]:
            base = max([q for q in inst if nm == q or nm.startswith(q + "_")], key=len, default=None)
            if base is not None and inst[base] in cats:
                ncat.append(cats.index(inst[base]))
        rec["has_proto"] = bool(ncat)
        if ncat:
            cn = [str(q) for q in g["cand_names"]]
            for k in ("sig", "l0", "l2"):
                sim = d[f"sim_{k}"].astype(np.float32)[ncat].max(0)
                rec[f"vis_hit_{k}"] = float(reg[int(np.argmax(sim))])
                cand = [(q, g["region"][j] > 0) for j, q in enumerate(cn) if (g["region"][j] > 0).sum() > 0]
                if len(cand) >= 2:
                    sc = [float(sim[r].mean()) for _, r in cand]
                    rec[f"vis_cand_{k}"] = float(cand[int(np.argmax(sc))][0] in m["named"])
                    rec["cand_chance"] = 1.0 / len(cand)
        per_obs.append(rec)

    def rate(sub, field, unit="episode"):
        per = {}
        if unit == "episode":
            ep = {}
            for r in sub:
                if field in r:
                    ep.setdefault((r["task"], r["trial"]), []).append(r[field])
            for (tk, _), v in ep.items():
                per.setdefault(tk, []).append(float(np.mean(v)))
        else:
            for r in sub:
                if field in r:
                    per.setdefault(r["task"], []).append(float(r[field]))
        return ci_rate(per)

    groups = {
        "pre_success": [r for r in per_obs if r["role"] == "pre" and r["success"]],
        "pre_wrong_object": [r for r in per_obs if r["role"] == "pre" and r["type"] == "wrong_object"],
        "pre_all": [r for r in per_obs if r["role"] == "pre"],
        "post_success": [r for r in per_obs if r["role"] == "post" and r["success"]],
        "post_wrong_goal": [r for r in per_obs if r["role"] == "post" and r["type"] == "wrong_goal"],
        "post_all": [r for r in per_obs if r["role"] == "post"],
        "pre_proxy_hit": [r for r in per_obs if r["role"] == "pre" and r["proxy"] == "hit"],
        "pre_proxy_wrong": [r for r in per_obs if r["role"] == "pre" and r["proxy"] == "wrong"],
        "post_proxy_hit": [r for r in per_obs if r["role"] == "post" and r["proxy"] == "hit"],
        "post_proxy_wrong": [r for r in per_obs if r["role"] == "post" and r["proxy"] == "wrong"],
    }
    fields = ["att_hit", "att_mass", "rev_hit", "chance", "vis_hit_sig", "vis_hit_l0", "vis_hit_l2", "vis_cand_sig", "vis_cand_l0",
              "vis_cand_l2", "cand_chance"]
    out = {}
    for gname, sub in groups.items():
        unit = "obs" if "proxy" in gname else "episode"
        out[gname] = {f: rate(sub, f, unit) for f in fields}
        out[gname]["n_obs"] = len(sub)
        out[gname]["chance_vis"] = rate([r for r in sub if "vis_hit_sig" in r], "chance", unit)
        out[gname]["att_hit_by_layer"] = [float(np.mean([r["att_hit_layer"][l] for r in sub if "att_hit_layer" in r]))
                                          if any("att_hit_layer" in r for r in sub) else None for l in range(N_LAYERS)]
        out[gname]["rev_hit_by_layer"] = [float(np.mean([r["rev_hit_layer"][l] for r in sub if "rev_hit_layer" in r]))
                                          if any("rev_hit_layer" in r for r in sub) else None for l in range(N_LAYERS)]
    out["coverage"] = dict(n_obs=len(per_obs), with_rows=sum("att_hit" in r for r in per_obs),
                           with_proto=sum(r["has_proto"] for r in per_obs))

    def gate(su, wr, f, chance="chance"):
        a, b, c = out[su][f]["mean"], out[wr][f]["mean"], out[wr][chance]["mean"]
        if a is None or b is None or c is None:
            return dict(go=False, reason="empty group")
        return dict(go=bool(b >= a - 0.10 and b >= 2 * c), success=a, wrong=b, chance=c)
    lab = ("pre_success", "pre_wrong_object", "post_success", "post_wrong_goal") if model == "l90c" else \
        ("pre_proxy_hit", "pre_proxy_wrong", "post_proxy_hit", "post_proxy_wrong")
    out["gates"] = dict(
        d2_object=gate(lab[0], lab[1], "att_hit"), d2_goal=gate(lab[2], lab[3], "att_hit"),
        d3_anchor_object=gate(lab[0], lab[1], "vis_hit_sig", "chance_vis"), labels=list(lab))
    return out


def read_d3_what(model):
    rows = train_rows(model)
    res = {}
    for k in ("sig", "l0", "l2"):
        multi = {}
        for r in rows:
            multi.setdefault(r["cat"], set()).add(r["scene"])
        elig = {c for c, s in multi.items() if len(s) >= 2}
        scenes = sorted({r["scene"] for r in rows})
        hits, ch, dist, task_of = [], [], [], []
        Xp, Yp, Ptr = [], [], []
        for s in scenes:
            tr = [r for r in rows if r["scene"] != s]
            te = [r for r in rows if r["scene"] == s and r["cat"] in elig]
            if not te:
                continue
            cs = sorted({r["cat"] for r in tr})
            P = np.stack([unit(np.mean([unit(r["feats"][k]) for r in tr if r["cat"] == c], 0)) for c in cs])
            mrc = {c: np.mean([r["rc"] for r in tr if r["cat"] == c], 0) for c in cs}
            # ridge position read (leave-scene-out): features -> region centroid (row, col)
            Xt = np.stack([unit(r["feats"][k]) for r in tr])
            Yt = np.stack([r["rc"] for r in tr])
            mu, ym = Xt.mean(0), Yt.mean(0)
            A = Xt - mu
            W = np.linalg.solve(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ (Yt - ym))
            for r in te:
                if r["cat"] not in cs:
                    continue
                pred = cs[int(np.argmax(P @ unit(r["feats"][k])))]
                hits.append(float(pred == r["cat"]))
                ch.append(1.0 / len(cs))
                dist.append(float(np.linalg.norm(r["rc"] - mrc[r["cat"]])))
                task_of.append(r["tid"])
                Xp.append(r["rc"])
                Yp.append((unit(r["feats"][k]) - mu) @ W + ym)
        hits, dist = np.asarray(hits), np.asarray(dist)
        per = {}
        for h, t in zip(hits, task_of):
            per.setdefault(t, []).append(h)
        acc = ci_rate(per)
        med = float(np.median(dist)) if len(dist) else 0.0
        near, far = hits[dist <= med], hits[dist > med]
        Xp, Yp = np.asarray(Xp), np.asarray(Yp)
        r2 = float(1 - ((Xp - Yp) ** 2).sum() / max(((Xp - Xp.mean(0)) ** 2).sum(), 1e-9)) if len(Xp) else None
        # same vs different type cosine (pairs across different scenes), AUC
        rng = np.random.default_rng(0)
        U = np.stack([unit(r["feats"][k]) for r in rows])
        cat = np.array([r["cat"] for r in rows])
        sc = np.array([r["scene"] for r in rows])
        a = rng.integers(0, len(rows), 20000)
        b = rng.integers(0, len(rows), 20000)
        ok = sc[a] != sc[b]
        a, b = a[ok], b[ok]
        cos = (U[a] * U[b]).sum(1)
        same = cat[a] == cat[b]
        auc = None
        if same.any() and (~same).any():
            from scipy.stats import mannwhitneyu
            auc = float(mannwhitneyu(cos[same], cos[~same]).statistic / (same.sum() * (~same).sum()))
        res[k] = dict(acc=acc, chance=float(np.mean(ch)) if ch else None, n=int(len(hits)), types_eligible=sorted(elig),
                      acc_near=float(near.mean()) if len(near) else None, acc_far=float(far.mean()) if len(far) else None,
                      median_shift_tokens=med, position_r2_from_features=r2, same_diff_auc=auc,
                      what_go=bool(acc["mean"] is not None and acc["mean"] >= 0.8 and acc["mean"] >= 3 * float(np.mean(ch))),
                      stable=bool(len(near) and len(far) and abs(float(far.mean()) - float(near.mean())) <= 0.10))
    return res


def sanity():
    """P2's transfer (pp2_dev.py's transfer block, verbatim logic) recomputed from P2's stored files, vs its ledger value."""
    import pp2_dev
    from lifelong_ood import ledger
    from lifelong_ood.policy import factors as F
    from lifelong_ood.protos import store as S
    out = {}
    for model, M in MODELS.items():
        idx = jload(os.path.join(PROTOS, M["pp_index"]))
        scale = pp2_dev.scale_from_demos(idx)
        TT = pp2_dev.task_types()
        od = os.path.join(PROTOS, M["pp_obs"])
        tr = {}
        for o in idx["obs"]:
            if o["set"] != "transfer":
                continue
            p = os.path.join(od, o["key"] + ".npz")
            if not os.path.exists(p):
                continue
            z = np.load(p, allow_pickle=True)
            f = np.load(os.path.join(pp2_dev.F0B, f"{o['suite']}_{o['index']}.npz"), allow_pickle=True)
            types, task = TT[(o["suite"], o["index"])]
            pt = S.parts_of(task, F.scene_phrases(task, [v[1] for v in TT.values()]))
            bodies = [str(x) for x in f["names"]]
            if pt["obj_inst"] not in bodies:
                continue
            j = bodies.index(pt["obj_inst"])
            s = o["state"]
            eef = np.asarray(f["state8"][s][:3], np.float64)
            objp = np.asarray(f["pos"][s][j], np.float64)
            names = [str(x) for x in z["names"]]
            if "obj:named" not in names:
                continue
            c = names.index("obj:named")
            for l in range(len(S.LEVELS)):
                it = S.interpret(z["x0_raw"][l, c].mean(0), eef, objp[None], [pt["obj_inst"]], scale)
                tr.setdefault((o["suite"], o["index"]), {}).setdefault(l, []).append((it["endpoint"][:2], objp[:2]))
        best = -1.0
        for l in range(len(S.LEVELS)):
            r = [corr2([x[0] for x in d[l]], [x[1] for x in d[l]]) for d in tr.values() if len(d.get(l, [])) >= 5]
            if r:
                best = max(best, float(np.nanmedian(r)))
        rows = ledger.load("pp2_dev", tag=M["pp2_tag"])
        led = rows[-1]["metrics"]["go"]["transfer_endpoint_r_best"] if rows else None
        out[model] = dict(recomputed=best, ledger=led, match=bool(led is not None and abs(best - led) < 1e-6), tasks=len(tr))
        print(f"[of sanity] {model}: P2 transfer endpoint r best recomputed {best:.4f} vs ledger {led}", flush=True)
    dump(os.path.join(OUT, "sanity.json"), out)


def read(record):
    from lifelong_ood import ledger
    res = dict(sanity=jload(os.path.join(OUT, "sanity.json")) if os.path.exists(os.path.join(OUT, "sanity.json")) else None)
    d1, d2, d3 = {}, {}, {}
    for model in MODELS:
        d1[model] = read_d1(model)
        d2[model] = read_dev(model)
        d3[model] = dict(what=read_d3_what(model), anchor={g: {f: d2[model][g].get(f) for f in
                                                               ("vis_hit_sig", "vis_hit_l0", "vis_hit_l2", "vis_cand_sig", "cand_chance",
                                                                "chance_vis", "att_hit")} for g in d2[model] if g.startswith(("pre", "post"))})
    # D1 hypotheses: H1 l90c part, H2 l90c instr, H3 libero part, H4 libero instr
    H = {"H1": ("l90c", "part"), "H2": ("l90c", "instr"), "H3": ("libero", "part"), "H4": ("libero", "instr")}
    ps, pt = {}, {}
    for h, (m, s) in H.items():
        x = d1[m]["sources"][s]
        ps[h] = max(x["p_obj_le_world"], x["p_obj_le_ctrl"])
        pt[h] = bool(x["T"]["object"] >= 0.5 and x["diff_obj_world"] >= 0.3 and x["diff_obj_ctrl"] > 0)
    hp = holm(ps)
    verdict = {h: dict(model=H[h][0], source=H[h][1], point_criteria=pt[h], p=ps[h], holm=hp[h], go=bool(pt[h] and hp[h])) for h in H}
    route1 = "alive" if (verdict["H1"]["go"] or verdict["H3"]["go"]) else \
        "instruction-level only" if (verdict["H2"]["go"] or verdict["H4"]["go"]) else "dead"
    d3g = {m: dict(what_go=d3[m]["what"]["sig"]["what_go"], stable=d3[m]["what"]["sig"]["stable"],
                   dissociation=bool(d3[m]["what"]["sig"]["what_go"] and d3[m]["what"]["sig"]["stable"]),
                   anchor=d2[m]["gates"]["d3_anchor_object"]) for m in MODELS}
    for m in MODELS:
        for s in d1[m]["sources"].values():
            s.pop("by_level_raw", None)
    res.update(d1=d1, d1_verdict=verdict, route1=route1, d2={m: d2[m] for m in MODELS},
               d2_gates={m: d2[m]["gates"] for m in MODELS}, d3=d3, d3_gates=d3g)
    dump(os.path.join(OUT, "read.json"), res)
    brief = dict(d1={m: {s: dict(T=v["T"], ci=v["ci"], best_level=v["best_level"], d_ow=v["diff_obj_world"], d_oc=v["diff_obj_ctrl"],
                                 n=v["n"]) for s, v in d1[m]["sources"].items()} for m in MODELS},
                 d1_verdict=verdict, route1=route1, d2_gates=res["d2_gates"], d3_gates=d3g, sanity=res["sanity"],
                 align={m: d1[m]["image_alignment"] for m in MODELS})
    print(json.dumps(brief, indent=1, default=float))
    if record:
        for tag, payload in (("d1", dict(d1=d1, verdict=verdict, route1=route1, sanity=res["sanity"])),
                             ("d2", dict(d2=res["d2"], gates=res["d2_gates"])), ("d3", dict(d3=d3, gates=d3g))):
            print(ledger.record("of_read", bench="libero_90_c", tag=tag, seed=0, metrics=payload,
                                bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                                config=dict(levels=LEVELS, d2_layers=D2_LAYERS, cov_min=COV_MIN, hand_clear=HAND_CLEAR),
                                script="experiments/of_frame.py", command=" ".join(sys.argv),
                                backbone=dict(models={m: MODELS[m]["cfg"] for m in MODELS}), artifacts=[os.path.join(OUT, "read.json")]))


def diag():
    """Print the descriptive parts of read.json and the argmax-token histogram of the D2 maps (a check for attention sinks)."""
    r = jload(os.path.join(OUT, "read.json"))
    for m in MODELS:
        print(f"== {m}")
        for s in ("part", "instr", "own", "null"):
            x = r["d1"][m]["sources"][s]
            print(f"D1 {s}: p2_centre {x['p2_centre']}; per_object " + json.dumps(
                {c: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()} for c, d in x["per_object"].items()}))
        for g, x in r["d2"][m].items():
            if g in ("gates", "coverage"):
                print(f"D2 {g}: {x}")
                continue
            print(f"D2 {g} n {x['n_obs']}: " + json.dumps({f: (None if x[f]["mean"] is None else [round(x[f]["mean"], 3),
                                                              round(x[f]["lo"], 3), round(x[f]["hi"], 3), x[f]["n"]])
                                                          for f in x if isinstance(x[f], dict) and "mean" in x[f]}))
            print(f"   att_hit_by_layer {[None if v is None else round(v, 3) for v in x['att_hit_by_layer']]}")
            print(f"   rev_hit_by_layer {[None if v is None else round(v, 3) for v in x['rev_hit_by_layer']]}")
        print("D3 what " + json.dumps({k: {q: v for q, v in d.items() if q != "types_eligible"} for k, d in r["d3"][m]["what"].items()},
                                      default=float))
        print("D3 eligible types " + json.dumps(r["d3"][m]["what"]["sig"]["types_eligible"]))
        hist = {}
        for f in sorted(glob.glob(os.path.join(OUT, f"dev_{m}", "*.npz")))[:400]:
            d = np.load(f, allow_pickle=True)
            for rk in ("O",):
                W = d[f"w2i_{rk}"]
                if W.sum() == 0:
                    continue
                for l in (0, 1, 2, 8, 14):
                    hist.setdefault(l, []).append(int(np.argmax(W[l])))
                hist.setdefault("mass_img", []).append(float(W[list(D2_LAYERS)].sum(1).mean()))
        for l, v in hist.items():
            if l == "mass_img":
                print(f"D2 check: word->agentview total attention mass (layers 0-2 mean) {np.mean(v):.4f}")
            else:
                u, c = np.unique(v, return_counts=True)
                o = np.argsort(-c)[:5]
                print(f"D2 check layer {l}: top argmax tokens {list(zip(u[o].tolist(), c[o].tolist()))} of {len(v)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", action="store_true")
    for s in ("index", "poses", "select", "gpu", "protos", "dev", "sanity", "read"):
        ap.add_argument(f"--{s}", action="store_true")
    ap.add_argument("--model", default="l90c")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    if a.index:
        index(a.model)
    if a.poses:
        poses(a.shard)
    if a.select:
        select(a.model)
    if a.gpu:
        gpu(a.model, a.shard)
    if a.protos:
        protos(a.model)
    if a.dev:
        dev(a.model, a.shard)
    if a.sanity:
        sanity()
    if a.read:
        read(a.record)
    if a.diag:
        diag()


if __name__ == "__main__":
    main()
