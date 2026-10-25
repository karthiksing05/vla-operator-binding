"""PROBE R3c (docs/PLAN_R3C.md; pre-registration experiments/prereg/pr3c_splice.py, ledger m3_prereg tag pr3c_splice):
can the frozen pi05_l90c@5000 be re-bound by splicing its OWN correctly-bound image-token K/V, region by region, from same-scene
training instructions?  No demo, no weight change, no new token.

R3b (docs/PLAN_R3B.md) found early attention steering inert: the misbinding lives in the image-token K/V of the prefix cache
(L0b: the instruction's effect is in the image tokens by layer 2).  The frozen model binds correctly on its TRAINING instructions
in the same scene.  So for a target "put X on W": find same-scene training cells A (moves X) and B (puts something on/in W) from
the BDDL goal atoms (ORACLE source retrieval, v1, labelled), run the prefix under the target and under A / B, and in the cache
replace the K/V at X's region tokens with A's (pre-grasp) or at W's region tokens with B's (post-grasp), in the chosen layers.
The text tokens stay the target's.  Whole-image splicing (every image position from A / B) is the sentence-composition control.

Stages (each its own job; nothing here runs on the login node):
  --sources               (chunks env, CPU) per R2a observation: same-scene sources A / B, counterfactual targets; coverage of the
                          15 recombination dev tasks -> results/rs4/sources.json
  --smoke                 (openpi GPU) identity splice == u11 exactly; cache shapes; movement magnitude (no readout) -> scratch/
  --tune --shard i/n      (openpi GPU) tuning pool (R3b's): u11, u21, full prompts, every grid config toward the cf / named target
  --stage0 [--record]     (chunks env) movability gate + the pre-registered choice -> results/rs4/stage0.json, ledger rs4_stage0
  --test --shard i/n      (openpi GPU) R3b's test sets: the chosen config and its comparators -> results/rs4/test/
  --read [--record]       (chunks env) H1-H4, rates, ledger rs4_read
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import r2a_null_modes as R2A                                            # noqa: E402

OUT = os.path.join(ROOT, "results", "rs4")
RS3 = os.path.join(ROOT, "results", "rs3")
R2 = R2A.OUT
CFG, CKPT = R2A.CFG, R2A.CKPT
# ── settings (fixed before the tuning run; docs/PLAN_R3C.md) ───────────────────────────────────────────────────────
N_SAMP = 12
SEED_A, SEED_B = 11, 21                                                 # R2a policy_sample seed (parity) / noise floor
LAYER_SETS = ("all", "ge2", "ge4")                                      # spliced cache layers: 0-17, 2-17, 4-17
DILATIONS = (0, 1)                                                      # region dilation in tokens (3 x 3 per step)
MOVE_BAR, FLOOR_MIN = 0.10, 0.03                                        # stage 0: g_cf >= 0.10 + max(0.03, |u21 - u11 at cf|)
ADMIT_NAMED, ADMIT_STILL, TIE = -0.02, 0.10, 0.005                      # choice rule (tuning pool)
GATE_RISE, GATE_HARM, ALPHA = 0.20, -0.05, 0.05
N_IMG_TOK = 256                                                         # agentview = prefix positions 0-255
N_ALL_IMG = 768                                                         # 3 image slots x 256 (whole-image splice)


def lstart(ls):
    return {"all": 0, "ge2": 2, "ge4": 4}[ls]


def grid():
    out = [dict(name=f"region|{ls}|d{d}", extent="region", layer_set=ls, dil=d) for ls in LAYER_SETS for d in DILATIONS]
    out += [dict(name=f"whole|{ls}", extent="whole", layer_set=ls, dil=0) for ls in LAYER_SETS]
    return out


def dump(path, obj):
    R2A.dump(path, obj)


def h(s):
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


# ── sources (chunks env): BDDL goal atoms of the same-scene training cells (ORACLE retrieval) ──────────────────────
def _atoms(t):
    """[(pred, obj, target)] of a factors.Task's In / On goal atoms (instance names, as R2a's candidates)."""
    return [(a[0].lower(), a[1], a[2]) for a in t.goal if len(a) == 3 and a[0].lower() in ("in", "on")]


def sources():
    import m1_compose_create as M1
    from lifelong_ood import l90c
    tt = M1.task_table()
    sp = l90c.load_split()
    train = [c for c in sp["train_l90"]]
    ix = R2A.load_index()
    dev = dict(M1.dev_tasks())
    res = dict(obs={}, tasks={})
    for tid in sorted(ix):
        t = tt[tid]
        cells = [c for c in train if tt[c].scene == t.scene and tt[c].language != t.language]
        movers, placers = {}, {}                                        # object -> cells moving it; target -> cells placing on it
        for c in cells:
            for p, o, g in _atoms(tt[c]):
                movers.setdefault(o, set()).add(c)
                placers.setdefault(g, set()).add(c)
        own = _atoms(t)

        def pick(pool, tag):
            return min(sorted(pool), key=lambda c: h(f"rs4src:{tid}:{tag}:{c}")) if pool else None
        res["tasks"][tid] = dict(bin=dev.get(tid), scene=t.scene, text=t.language, n_scene_cells=len(cells),
                                 objects={o: pick(movers.get(o, ()), "O:" + o) for _, o, _ in own},
                                 targets={g: pick(placers.get(g, ()), "G:" + g) for _, _, g in own})
        for e in ix[tid]["episodes"]:
            for m in e["obs"]:
                cn = list(m["cand"])
                if m["role"] == "pre":
                    pool = movers
                    src = {x: pick(pool.get(x, ()), "O:" + x) for x in m["named"]}
                else:
                    pool = placers
                    src = {x: pick(pool.get(x, ()), "G:" + x) for x in m["named"]}
                src = {x: c for x, c in src.items() if c is not None}
                cfp = [q for q in cn if q not in m["named"] and pool.get(q)]
                cf = cfp[h("rs4cf:" + m["key"]) % len(cfp)] if cfp else None
                cf_src = pick(pool[cf], ("O:" if m["role"] == "pre" else "G:") + cf) if cf else None
                # import diagnostic (post-grasp): the object source A's own goal for the held object, if it is a candidate
                imp = None
                if m["role"] == "post":
                    held = [o for _, o, g in own if g in m["named"]]
                    a = pick(movers.get(held[0], ()), "O:" + held[0]) if held else None
                    if a is not None:
                        ga = [g for _, o, g in _atoms(tt[a]) if o == held[0] and g in cn and g not in m["named"]]
                        imp = dict(src=a, text=tt[a].language, goal=ga[0]) if ga else dict(src=a, text=tt[a].language, goal=None)
                # descriptive only: a same-scene cell moving an object of the named object's TYPE (another instance counts)
                tys = {t.objects.get(x) for x in m["named"]} if m["role"] == "pre" else set()
                type_src = sorted(c for c in cells if tys & set(tt[c].moved)) if m["role"] == "pre" else None
                res["obs"][m["key"]] = dict(task=tid, role=m["role"], named=m["named"], type_src=type_src,
                                            src={x: dict(cell=c, text=tt[c].language) for x, c in src.items()},
                                            cf=cf, cf_src=dict(cell=cf_src, text=tt[cf_src].language) if cf_src else None, imp=imp)
    rec = [tid for tid, v in res["tasks"].items() if v["bin"] in ("0.1", "0.3", "0.5")]
    cov = dict(n_recombination=len(rec),
               A_all=sum(all(v is not None for v in res["tasks"][t]["objects"].values()) and bool(res["tasks"][t]["objects"]) for t in rec),
               B_all=sum(all(v is not None for v in res["tasks"][t]["targets"].values()) and bool(res["tasks"][t]["targets"]) for t in rec),
               per_task={t: dict(objects=res["tasks"][t]["objects"], targets=res["tasks"][t]["targets"]) for t in rec})
    cov["both"] = sum(all(v is not None for v in list(res["tasks"][t]["objects"].values()) + list(res["tasks"][t]["targets"].values()))
                      and bool(res["tasks"][t]["objects"]) and bool(res["tasks"][t]["targets"]) for t in rec)
    res["coverage_recombination"] = cov
    sp3 = json.load(open(os.path.join(RS3, "split.json"), encoding="utf-8"))
    cnt = {"tune": dict(n=len(sp3["tune"]), named_src=sum(bool(res["obs"][k]["src"]) for k in sp3["tune"]),
                        cf_src=sum(res["obs"][k]["cf"] is not None for k in sp3["tune"]))}
    for s, ks in sp3["test"].items():
        ep = lambda f: len({(res["obs"][k]["task"], k.split("_t")[1].split("_")[0]) for k in ks if f(res["obs"][k])})   # noqa: E731
        cnt[s] = dict(n=len(ks), named_src=sum(bool(res["obs"][k]["src"]) for k in ks),
                      named_src_type=sum(bool(res["obs"][k]["type_src"]) for k in ks),
                      episodes=ep(lambda o: True), episodes_named_src=ep(lambda o: bool(o["src"])),
                      tasks=sorted({res["obs"][k]["task"] for k in ks}),
                      tasks_named_src=sorted({res["obs"][k]["task"] for k in ks if res["obs"][k]["src"]}),
                      cf_src=sum(res["obs"][k]["cf"] is not None for k in ks))
    res["counts"] = cnt
    dump(os.path.join(OUT, "sources.json"), res)
    print("[rs4 sources] coverage of the recombination tasks: " + json.dumps({k: v for k, v in cov.items() if k != "per_task"}))
    for t, v in cov["per_task"].items():
        print(f"[rs4 sources]   {t:13s} {res['tasks'][t]['bin']:4s} '{res['tasks'][t]['text']}' objects {v['objects']} targets {v['targets']}")
    print("[rs4 sources] observation counts " + json.dumps(cnt), flush=True)


# ── the splice (openpi env) ──────────────────────────────────────────────────────────────────────────────────────
def dilate(mask, d):
    m = mask.reshape(16, 16).astype(bool)
    for _ in range(d):
        p = np.pad(m, 1)
        m = np.max([p[1 + a:17 + a, 1 + b:17 + b] for a in (-1, 0, 1) for b in (-1, 0, 1)], 0)
    return m.reshape(256)


def splice(pre_t, parts, l0):
    """New Prefix: pre_t's cache with, in layers >= l0, the K / V at each part's positions taken from that part's source prefix.
    parts = [(positions LongTensor, source Prefix)].  pre_t and the sources are left untouched."""
    import torch
    from lifelong_ood.policy.backbone import Prefix
    kv = pre_t.kv
    kv2 = type(kv)()
    for li, (k, v) in enumerate(zip(kv.key_cache, kv.value_cache)):
        if li >= l0 and parts:
            k, v = k.clone(), v.clone()
            for pos, src in parts:
                ks, vs = src.kv.key_cache[li], src.kv.value_cache[li]
                assert ks.shape == k.shape and k.shape[2] == pre_t.pad_masks.shape[1], (ks.shape, k.shape)
                k[:, :, pos, :] = ks[:, :, pos, :]
                v[:, :, pos, :] = vs[:, :, pos, :]
        kv2.update(k, v, li)
    return Prefix(state=pre_t.state, pad_masks=pre_t.pad_masks, kv=kv2, n=pre_t.n)


class Runner:
    def __init__(self):
        from lifelong_ood.policy.backbone import PolicyBackbone
        self.bk = PolicyBackbone(CFG, checkpoint_dir=CKPT, device="cuda")
        self.src = json.load(open(os.path.join(OUT, "sources.json"), encoding="utf-8"))["obs"]

    def obs(self, key):
        import torch
        bk = self.bk
        z = np.load(os.path.join(R2, "obs", key + ".npz"), allow_pickle=True)
        g = np.load(os.path.join(RS3, "regions", key + ".npz"), allow_pickle=True)
        cn = [str(q) for q in z["cand_names"]]
        assert cn == [str(q) for q in g["cand_names"]], key
        cache = {}

        def pre(text):
            if text not in cache:
                od = {"observation/image": z["image"], "observation/wrist_image": z["wrist"],
                      "observation/state": np.asarray(z["state"], np.float32), "prompt": text}
                with torch.no_grad():
                    cache[text] = bk.prefix(bk.observation(od))
            return cache[text]

        def pos(names, d, extent):
            if extent == "whole":
                return torch.arange(N_ALL_IMG, device=bk.device)
            m = np.max([g["region"][cn.index(q)] for q in names], 0) > 0
            return torch.as_tensor(np.flatnonzero(dilate(m, d)), device=bk.device, dtype=torch.long)
        return z, cn, str(z["text"]), pre, pos

    def draw(self, p, seed):
        X = self.bk.sample(N_SAMP, p, seed=seed)
        return self.bk.unnormalize(X).reshape(N_SAMP, self.bk.H, -1).astype(np.float32)

    def run(self, key, cfgs, extra):
        """-> {arm: [12, H, 7] chunks}.  cfgs: grid configs run toward 'named' and 'cf'; extra: the comparator arm names."""
        import torch
        s = self.src[key]
        z, cn, text, pre, pos = self.obs(key)
        out = {}
        with torch.no_grad():
            pt = pre(text)
            out["u11"] = self.draw(pt, SEED_A)
            if extra.get("u21"):
                out["u21"] = self.draw(pt, SEED_B)
            tg = {}
            if s["src"]:
                tg["named"] = [(x, v["text"]) for x, v in s["src"].items()]
            if s["cf"] is not None:
                tg["cf"] = [(s["cf"], s["cf_src"]["text"])]
            for t, lst in tg.items():
                if extra.get(f"full|{t}"):                              # the whole prompt replaced by the (first) source
                    out[f"full|{t}"] = self.draw(pre(lst[0][1]), SEED_A)
                for c in cfgs:
                    parts = [(pos([x], c["dil"], c["extent"]), pre(st)) for x, st in lst]
                    if c["extent"] == "whole":
                        parts = parts[:1]
                    out[f"{c['name']}|{t}"] = self.draw(splice(pt, parts, lstart(c["layer_set"])), SEED_A)
            if "imp" in extra and s.get("imp"):                         # post-grasp: whole / region splice from the object source A
                imp = s["imp"]
                pa = pre(imp["text"])
                c = extra["imp"]
                held_region = [(torch.arange(N_ALL_IMG, device=self.bk.device), pa)]
                out["imp_whole"] = self.draw(splice(pt, held_region, lstart(c["layer_set"])), SEED_A)
                out["imp_full"] = self.draw(pa, SEED_A)
        return out


def gpu_stage(stage, shard):
    sp = json.load(open(os.path.join(RS3, "split.json"), encoding="utf-8"))
    rn = Runner()
    if stage in ("tune", "smoke"):
        keys = list(sp["tune"])
        cfgs, extra = grid(), {"u21": 1, "full|named": 1, "full|cf": 1}
    else:
        ch = json.load(open(os.path.join(OUT, "stage0.json"), encoding="utf-8"))
        assert ch["stage0_pass"], "stage 0 failed: no test run"
        c = next(q for q in grid() if q["name"] == ch["chosen"])
        w = next(q for q in grid() if q["name"] == f"whole|{c['layer_set']}")
        keys = sorted(set(k for v in sp["test"].values() for k in v))
        cfgs, extra = [c, w], {"u21": 1, "full|named": 1, "full|cf": 0, "imp": c}
    od = os.path.join(OUT, stage if stage != "smoke" else "scratch/smoke")
    i, n = map(int, shard.split("/"))
    mine = keys[i::n] if stage != "smoke" else [k for k in keys if rn.src[k]["src"] and rn.src[k]["cf"]][:2]
    os.makedirs(od, exist_ok=True)
    print(f"[rs4 {stage}] {rn.bk.describe()}; {len(mine)} observations x {len(cfgs)} configs, shard {shard}", flush=True)
    t_all, done = time.time(), 0
    for key in mine:
        outp = os.path.join(od, key + ".npz")
        if os.path.exists(outp):
            continue
        t0 = time.time()
        if stage == "smoke":
            _smoke(rn, key)
        raw = rn.run(key, cfgs, extra)
        info = dict(key=key, src=rn.src[key], secs=time.time() - t0)
        if stage == "smoke":
            moves = {k: float(np.abs(v - raw["u11"]).mean()) for k, v in raw.items() if k != "u11"}
            print(f"[rs4 smoke] {key}: mean |arm - u11| {json.dumps(moves)}", flush=True)
            continue
        R2A.save_atomic(outp, arms=np.array(list(raw)), raw=np.stack([raw[k] for k in raw]), info=np.array(json.dumps(info)))
        done += 1
        print(f"[rs4 {stage}] {key}: {info['secs']:.1f}s ({done} done, {time.time() - t_all:.0f}s)", flush=True)
    print(f"[rs4 {stage}] shard {shard} finished: {done} new", flush=True)


def _smoke(rn, key):
    """Identity splice (source = the target itself, every layer, whole image) reproduces u11 exactly; parity of u11 with R2a's stored
    draft_instr draws; the K / V of the agentview tokens differ between the target and a source prompt (by layer)."""
    import torch
    z, cn, text, pre, pos = rn.obs(key)
    with torch.no_grad():
        pt = pre(text)
        u = rn.draw(pt, SEED_A)
        ident = rn.draw(splice(pt, [(pos(None, 0, "whole"), pre(text))], 0), SEED_A)
        zm = np.load(os.path.join(R2, "modes", key + ".npz"), allow_pickle=True)
        a, b = json.loads(str(zm["slices"]))["draft_instr"]
        st = list(rn.src[key]["src"].values())[0]["text"]
        ps = pre(st)
        dk = [float((ps.kv.key_cache[l][:, :, :N_IMG_TOK] - pt.kv.key_cache[l][:, :, :N_IMG_TOK]).norm()
                    / pt.kv.key_cache[l][:, :, :N_IMG_TOK].norm()) for l in range(len(pt.kv.key_cache))]
    print(f"[rs4 smoke] {key}: identity splice max|.-u11| {float(np.abs(ident - u).max()):.2e}; parity max|u11 - R2a draft_instr| "
          f"{float(np.abs(zm['raw'][a:b] - u).max()):.2e}; layers {len(pt.kv.key_cache)} cache {tuple(pt.kv.key_cache[0].shape)}; "
          f"source '{st}' vs target '{text}': rel |dK| agentview by layer {[round(x, 3) for x in dk]}", flush=True)


# ── reads (chunks env) ───────────────────────────────────────────────────────────────────────────────────────────
META = {}


def _load_meta():
    for tid, x in R2A.load_index().items():
        for e in x["episodes"]:
            for m in e["obs"]:
                META[m["key"]] = dict(m, task=tid, trial=e["trial"], type=e["type"], success=e["success"])


def obs_shares(key, f):
    """-> {arm: (hit share at the arm's target, share pointing at nothing)}; u11@cf / u21@cf: unsteered shares at the cf target;
    imp arms: (share at A's goal, share at the own target)."""
    z = np.load(os.path.join(R2, "obs", key + ".npz"), allow_pickle=True)
    s = np.load(f, allow_pickle=True)
    info = json.loads(str(s["info"]))
    src = info["src"]
    cn = [str(q) for q in z["cand_names"]]
    eef, cp = z["state"][:3], z["cand_pos"]
    named, cf = set(src["named"]), src["cf"]
    out = {}
    for arm, raw in zip([str(q) for q in s["arms"]], s["raw"]):
        pts = np.array([R2A.pointed(r, eef, cp)[0] for r in raw])
        at = lambda S: float(np.mean([p >= 0 and cn[p] in S for p in pts]))   # noqa: E731
        still = float((pts < 0).mean())
        if arm.startswith("imp"):
            ga = (src.get("imp") or {}).get("goal")
            out[arm] = (at({ga}) if ga else None, at(named))
            continue
        out[arm] = (at({cf}) if arm.endswith("|cf") else at(named), still)
        if arm in ("u11", "u21"):
            if cf is not None:
                out[arm + "@cf"] = (at({cf}), still)
            ga = (src.get("imp") or {}).get("goal")
            if ga:
                out[arm + "@imp"] = (at({ga}), at(named))
    return out, info


def _ep_pairs(rows, a, b, ia=0, ib=0):
    """rows [(key, shares)] -> {task: (episode means of arm a, of arm b)} over observations that have both arms."""
    ep = {}
    for key, sh in rows:
        if a in sh and b in sh and sh[a][ia] is not None and sh[b][ib] is not None:
            m = META[key]
            ep.setdefault((m["task"], m["trial"]), []).append((sh[a][ia], sh[b][ib]))
    out = {}
    for (t, _), v in ep.items():
        pa, pb = out.setdefault(t, ([], []))
        pa.append(float(np.mean([q[0] for q in v])))
        pb.append(float(np.mean([q[1] for q in v])))
    return {t: (np.asarray(p[0]), np.asarray(p[1])) for t, p in out.items()}


def _mean_diff(pairs):
    n = sum(len(p[0]) for p in pairs.values())
    return float(sum(p[0].sum() - p[1].sum() for p in pairs.values()) / n) if n else None


def stage0(record):
    from lifelong_ood import stats
    _load_meta()
    sp = json.load(open(os.path.join(RS3, "split.json"), encoding="utf-8"))
    rows = []
    for key in sp["tune"]:
        f = os.path.join(OUT, "tune", key + ".npz")
        assert os.path.exists(f), f"tuning incomplete: {key}"
        rows.append((key, obs_shares(key, f)[0]))
    floor = _mean_diff(_ep_pairs(rows, "u21@cf", "u11@cf"))
    floor_named = _mean_diff(_ep_pairs(rows, "u21", "u11"))
    thr = MOVE_BAR + max(FLOOR_MIN, abs(floor))
    res = {}
    for c in grid():
        nm, cf = c["name"] + "|named", c["name"] + "|cf"
        pc = _ep_pairs(rows, cf, "u11@cf")
        pn = _ep_pairs(rows, nm, "u11")
        st = _ep_pairs(rows, nm, "u11", 1, 1)
        g_cf = stats.bootstrap_diff(pc, B=10000, seed=0)
        res[c["name"]] = dict(g_cf=g_cf["diff"], g_cf_ci=[g_cf["lo"], g_cf["hi"]], g_named=_mean_diff(pn), still=_mean_diff(st),
                              n_cf=g_cf["n_pairs"], n_named=sum(len(p[0]) for p in pn.values()), extent=c["extent"])
        res[c["name"]]["movable"] = bool(res[c["name"]]["g_cf"] >= thr)
        res[c["name"]]["admissible"] = bool(c["extent"] == "region" and res[c["name"]]["movable"]
                                            and res[c["name"]]["g_named"] >= ADMIT_NAMED and res[c["name"]]["still"] <= ADMIT_STILL)
    ref = dict(full_cf=stats.bootstrap_diff(_ep_pairs(rows, "full|cf", "u11@cf"), B=10000, seed=0),
               full_named=_mean_diff(_ep_pairs(rows, "full|named", "u11")))
    s0 = any(v["movable"] for k, v in res.items() if v["extent"] == "region")
    cg = {c["name"]: c for c in grid()}
    adm = [k for k, v in res.items() if v["admissible"]]
    chosen = None
    if adm:
        best = max(res[k]["g_cf"] for k in adm)
        near = [k for k in adm if res[k]["g_cf"] >= best - TIE]
        chosen = min(near, key=lambda k: (-lstart(cg[k]["layer_set"]), cg[k]["dil"]))
    out = dict(stage0_pass=bool(s0 and chosen is not None), any_movable_region=bool(s0), threshold=thr, floor_cf_u21_minus_u11=floor,
               floor_named_u21_minus_u11=floor_named, chosen=chosen, table=res, reference=ref, n_tune=len(rows),
               unsteered_named=float(np.mean([sh["u11"][0] for _, sh in rows])),
               unsteered_at_cf=float(np.mean([sh["u11@cf"][0] for _, sh in rows if "u11@cf" in sh])))
    dump(os.path.join(OUT, "stage0.json"), out)
    for k, v in res.items():
        print(f"[rs4 stage0] {k:16s} g_cf {v['g_cf']:+.3f} [{v['g_cf_ci'][0]:+.3f}, {v['g_cf_ci'][1]:+.3f}] (n {v['n_cf']}) "
              f"g_named {v['g_named']:+.3f} still {v['still']:+.3f} movable {v['movable']} adm {v['admissible']}")
    print(f"[rs4 stage0] full prompt from the cf source: g_cf {ref['full_cf']['diff']:+.3f} [{ref['full_cf']['lo']:+.3f}, "
          f"{ref['full_cf']['hi']:+.3f}]; full from the named source g_named {ref['full_named']:+.3f}")
    print(f"[rs4 stage0] floor u21 - u11 at cf {floor:+.3f} (named {floor_named:+.3f}); threshold {thr:.3f}; "
          f"stage 0 {'PASSES' if out['stage0_pass'] else 'FAILS'}; chosen {chosen}", flush=True)
    if record:
        from lifelong_ood import ledger
        print(ledger.record("rs4_stage0", bench="libero_90_c", tag="stage0", seed=SEED_A, metrics=out,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=_settings(),
                            script="experiments/rs4_splice.py", command=" ".join(sys.argv), backbone=dict(config=CFG, checkpoint=CKPT),
                            artifacts=[os.path.join(OUT, "stage0.json")]))


def _settings():
    return dict(n_samp=N_SAMP, seeds=[SEED_A, SEED_B], layer_sets=LAYER_SETS, dilations=DILATIONS, move_bar=MOVE_BAR,
                floor_min=FLOOR_MIN, admit_named=ADMIT_NAMED, admit_still=ADMIT_STILL, tie=TIE, gate_rise=GATE_RISE,
                gate_harm=GATE_HARM, sources="ORACLE: BDDL goal atoms of same-scene training cells (train_l90)")


def _bd(pairs, **kw):
    """stats.bootstrap_diff, or an empty result (diff None) when no task has a covered episode."""
    from lifelong_ood import stats
    if not pairs:
        return dict(diff=None, lo=None, hi=None, lo90=None, hi90=None, n_pairs=0, n_tasks=0)
    return stats.bootstrap_diff(pairs, **kw)


def read(record):
    _load_meta()
    sp = json.load(open(os.path.join(RS3, "split.json"), encoding="utf-8"))
    ch = json.load(open(os.path.join(OUT, "stage0.json"), encoding="utf-8"))
    c = ch["chosen"]
    w = "whole|" + c.split("|")[1]
    res = dict(chosen=c, settings=_settings(), sets={})
    for sname, keys in sp["test"].items():
        rows = []
        for key in keys:
            f = os.path.join(OUT, "test", key + ".npz")
            assert os.path.exists(f), f"test incomplete: {key}"
            rows.append((key, obs_shares(key, f)[0]))
        s, cf = f"{c}|named", f"{c}|cf"
        d = _bd(_ep_pairs(rows, s, "u11"), B=10000, seed=0, return_samples=True)
        smp = d.pop("samples") if d["diff"] is not None else np.ones(1)   # no covered episode: p = 1, the hypothesis fails
        itt = [(k, dict(sh, **({s: sh["u11"]} if s not in sh else {}))) for k, sh in rows]   # uncovered -> unsteered
        r = dict(n_obs=len(rows), n_covered=sum(s in sh for _, sh in rows), diff=d, p_rise=float(np.mean(smp <= 0.0)),
                 p_harm=float(np.mean(smp <= GATE_HARM)),
                 itt=_bd(_ep_pairs(itt, s, "u11"), B=10000, seed=0),
                 floor_u21_minus_u11=_bd(_ep_pairs(rows, "u21", "u11"), B=10000, seed=0),
                 whole_minus_u11=_bd(_ep_pairs(rows, f"{w}|named", "u11"), B=10000, seed=0),
                 full_minus_u11=_bd(_ep_pairs(rows, "full|named", "u11"), B=10000, seed=0),
                 region_minus_whole=_bd(_ep_pairs(rows, s, f"{w}|named"), B=10000, seed=0),
                 cf_control=(_bd(_ep_pairs(rows, cf, "u11@cf"), B=10000, seed=0)
                             if _ep_pairs(rows, cf, "u11@cf") else None))
        rates = {}
        for a in ("u11", "u21", s, f"{w}|named", "full|named"):
            per = {}
            for (t, _), v in _group([(k, sh) for k, sh in rows if s in sh], a).items():
                per.setdefault(t, []).append(v)
            rates[a] = R2A.ci_rate(per)
        r["rates_covered"] = rates
        r["n_episodes"] = len({(META[k]["task"], META[k]["trial"]) for k, sh in rows if s in sh})
        r["n_tasks"] = len({META[k]["task"] for k, sh in rows if s in sh})
        if sname.endswith("post"):                                      # import of the object source's goal (descriptive)
            r["import"] = {a: _mean_diff(_ep_pairs(rows, a, "u11@imp")) for a in ("imp_whole", "imp_full")}
            r["import_unsteered_at_A_goal"] = float(np.mean([sh["u11@imp"][0] for _, sh in rows if "u11@imp" in sh])) \
                if any("u11@imp" in sh for _, sh in rows) else None
        res["sets"][sname] = r
    hyp = {"H1": ("wo_pre", "rise"), "H2": ("su_pre", "harm"), "H3": ("wg_post", "rise"), "H4": ("su_post", "harm")}
    ps = {k: res["sets"][s]["p_rise" if q == "rise" else "p_harm"] for k, (s, q) in hyp.items()}
    H, prev = {}, True
    for rank, k in enumerate(sorted(ps, key=ps.get)):
        thr = ALPHA / (len(ps) - rank)
        ok = prev and ps[k] < thr
        prev = ok
        s, q = hyp[k]
        d = res["sets"][s]["diff"]
        est = (d["diff"] is not None and d["diff"] >= GATE_RISE and d["lo"] > 0) if q == "rise" else d["diff"] is not None
        H[k] = dict(set=s, kind=q, p=ps[k], holm_threshold=thr, holm_ok=bool(ok), estimate_ok=bool(est), passed=bool(ok and est), diff=d)
    res["hypotheses"] = H
    res["stage1_gate"] = bool(H["H1"]["passed"] and H["H2"]["passed"])
    path = os.path.join(OUT, "read.json")
    dump(path, res)
    for s_, v in res["sets"].items():
        print(f"\n[rs4 read] {s_}: {v['n_obs']} obs ({v['n_covered']} covered), {v['n_episodes']} ep, {v['n_tasks']} tasks; " + json.dumps({k: v[k] for k in ("diff", "itt", "floor_u21_minus_u11", "whole_minus_u11", "full_minus_u11", "region_minus_whole", "cf_control")}))
        print("[rs4 read]   " + "  ".join(f"{a} {r_['mean']:.3f} [{r_['lo']:.3f}, {r_['hi']:.3f}]" for a, r_ in v["rates_covered"].items()
                                          if r_["mean"] is not None) + ("  import " + json.dumps(v.get("import")) if "import" in v else ""))
    print("\n[rs4 read] hypotheses " + json.dumps({k: {q: v[q] for q in ("p", "holm_threshold", "holm_ok", "estimate_ok", "passed")}
                                                   for k, v in H.items()}))
    print(f"[rs4 read] stage-1 gate {'PASSES' if res['stage1_gate'] else 'FAILS'}", flush=True)
    if record:
        from lifelong_ood import ledger
        print(ledger.record("rs4_read", bench="libero_90_c", tag="stage1", seed=SEED_A, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=_settings(),
                            script="experiments/rs4_splice.py", command=" ".join(sys.argv), backbone=dict(config=CFG, checkpoint=CKPT),
                            artifacts=[path]))


def _group(rows, a):
    ep = {}
    for key, sh in rows:
        if a in sh and sh[a][0] is not None:
            m = META[key]
            ep.setdefault((m["task"], m["trial"]), []).append(sh[a][0])
    return {k: float(np.mean(v)) for k, v in ep.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in ("sources", "smoke", "tune", "stage0", "test", "read", "record"):
        ap.add_argument("--" + f, action="store_true")
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    if a.sources:
        sources()
    if a.smoke:
        gpu_stage("smoke", a.shard)
    if a.tune:
        gpu_stage("tune", a.shard)
    if a.stage0:
        stage0(a.record)
    if a.test:
        gpu_stage("test", a.shard)
    if a.read:
        read(a.record)
