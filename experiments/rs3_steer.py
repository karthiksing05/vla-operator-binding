"""PROBE R3b (docs/PLAN_R3B.md; pre-registration experiments/prereg/pr3b_steer.py, ledger m3_prereg tag pr3b_steer):
can the frozen pi05_l90c@5000 be re-bound at the site of its binding error by an attention-logit bias, with no demo, no weight change
and no new token?

zs fails the dev recombinations mainly by binding (R2a: wrong goal 152, wrong object 56 of 580); L0b: binding leaves the word
positions by layer 2 and lives in the image tokens.  So at an observation o, in the early layers, add +b to the attention logits from
  - the instruction's OBJECT-word tokens (pre-grasp) / GOAL-word tokens (post-grasp) -- the VLM prefix pass ("word" site), and/or
  - the action expert's 10 action queries -- the denoising pass ("expert" site),
to the agentview image tokens that cover the named object / goal region.  Region v1 = ORACLE: the simulator's segmentation of the
scene with every movable object at its recorded pose (the robot hidden), on the policy's 16 x 16 agentview tokens (privileged, as
ReGuide-oracle).  Region v2 (the model's own word->patch attention) runs only if v1 passes.

Stage 1 is offline, on R2a's stored observations (results/r2a/obs, index): the policy's 12-chunk sample with and without steering
(the same noise, seed 11 = R2a's policy_sample draws), scored with R2a's readout (r2a_null_modes.pointed, 5 cm corridor rule).

Stages (each its own job; nothing here runs on the login node):
  --regions --shard i/n   (LIBERO env + EGL GPU) oracle region maps of every candidate at every R2a observation -> results/rs3/regions/
  --check                 (chunks env) overlay PNGs of the named regions on the policy's recorded images (hand check) -> scratch/
  --split                 (chunks env) tuning pool and test sets from the R2a index (episode types only) -> results/rs3/split.json
  --smoke                 (openpi GPU) parity of the unsteered arm with R2a's stored draws + a steered arm moves the chunk (2 tuning obs)
  --tune --shard i/n      (openpi GPU) tuning pool: unsteered + every grid config (named / counterfactual target) -> results/rs3/tune/
  --choose [--record]     (chunks env) the pre-registered selection rule -> results/rs3/choice.json, ledger rs3_choice
  --test --shard i/n      (openpi GPU) test sets: the chosen config and its comparators -> results/rs3/test/
  --read [--record]       (chunks env) H1-H4, rates, ledger rs3_read
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
import r2a_null_modes as R2A                                            # noqa: E402

OUT = os.path.join(ROOT, "results", "rs3")
R2 = R2A.OUT
CFG, CKPT = R2A.CFG, R2A.CKPT
# ── settings (fixed before the tuning run; docs/PLAN_R3B.md section 2) ──────────────────────────────────────────
N_SAMP = 12                                                             # R2a's policy_sample: 12 draws
SEED_A, SEED_B = 11, 21                                                 # SEED_A = R2a's draft_instr seed (parity); SEED_B = noise floor
SITES = ("word", "expert", "both")
LAYER_SETS = {"L0-1": (0, 1), "L0-3": (0, 1, 2, 3), "L0-7": tuple(range(8))}
BIASES = (2.0, 4.0, 8.0)
COV_MIN = 0.05                                                          # a token is in an object's region if >= 5% of its pixels show it
DISK_PX = 24                                                            # fallback / region-site disk radius (256-px client image)
ADMIT_NAMED, ADMIT_STILL, TIE = -0.02, 0.10, 0.005                      # selection rule (tuning pool)
GATE_RISE, GATE_HARM, ALPHA = 0.20, -0.05, 0.05
N_IMG_TOK = 256                                                         # agentview = the first 256 prefix positions
ROBOT_PREFIX = ("robot0_", "gripper0_", "mount0_")


def slug(tid):
    return tid.replace(":", "_")


def dump(path, obj):
    R2A.dump(path, obj)


def cf_pick(key, names, excl):
    """The counterfactual steering target: a non-named candidate, chosen by a hash of the observation key (no outcome)."""
    pool = [n for n in names if n not in excl]
    if not pool:
        return None
    h = int(hashlib.sha1(("rs3cf:" + key).encode()).hexdigest(), 16)
    return pool[h % len(pool)]


# ── regions (LIBERO env, EGL) ─────────────────────────────────────────────────────────────────────────────────────
TRANSFORMS = [(sw, fr, fc) for sw in (0, 1) for fr in (0, 1) for fc in (0, 1)]


def _xf(px, t):
    """raw projected pixel (a, b) of robosuite's camera_utils -> (row, col) in the client's flipped 256 image under transform t."""
    a, b = (px[..., 1], px[..., 0]) if t[0] else (px[..., 0], px[..., 1])
    a = 255 - a if t[1] else a
    b = 255 - b if t[2] else b
    return np.stack([a, b], -1)


TOK_RC = np.stack(np.meshgrid(np.arange(16) * 16 + 7.5, np.arange(16) * 16 + 7.5, indexing="ij"), -1).reshape(256, 2)   # row, col


def disk(rc, r=DISK_PX):
    return (np.linalg.norm(TOK_RC - np.asarray(rc, float)[None], axis=1) <= r).astype(np.float32)


class _CU:
    """robosuite's camera_utils needs h5py (absent in the LIBERO venv): the same pinhole projection from the MuJoCo camera, returning
    (row, col) of the renderer's raw image.  The orientation into the client's image is calibrated per task (TRANSFORMS)."""
    @staticmethod
    def get_camera_transform_matrix(sim, cam, h, w):
        return (sim, cam, h, w)

    @staticmethod
    def project_points_from_world_to_camera(pts, M, h, w):
        sim, cam = M[0], M[1]
        cid = sim.model.camera_name2id(cam)
        f = 0.5 * h / np.tan(float(sim.model.cam_fovy[cid]) * np.pi / 360.0)
        pc = (np.asarray(pts, np.float64) - np.asarray(sim.data.cam_xpos[cid])) @ np.asarray(sim.data.cam_xmat[cid]).reshape(3, 3)
        z = np.maximum(-pc[..., 2], 1e-6)                               # MuJoCo cameras look down -z, y up
        return np.stack([-f * pc[..., 1] / z + h / 2.0, f * pc[..., 0] / z + w / 2.0], -1)


def regions(shard):
    import pathlib
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    CU = _CU
    import libero_client as LC
    ix = R2A.load_index()
    i, n = map(int, shard.split("/"))
    od = os.path.join(OUT, "regions")
    os.makedirs(od, exist_ok=True)
    report = {}
    for j, tid in enumerate(sorted(ix)):
        if j % n != i:
            continue
        x = ix[tid]
        suite, ti = tid.split(":")
        task = benchmark.get_benchmark_dict()[suite]().get_task(int(ti))
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        env.seed(0)
        env.reset()
        fresh = np.load(os.path.join(R2A.R, R2A.FRESH_DIR, f"{suite}_{ti}.npy"))
        inner = env.env
        sim = inner.sim
        names, groot = LC.seg_tables(env)
        m = sim.model
        robot_geoms = np.array([g for g in range(m.ngeom) if (m.body_id2name(int(m.geom_bodyid[g])) or "").startswith(ROBOT_PREFIX)])
        M = CU.get_camera_transform_matrix(sim, "agentview", 256, 256)
        rec_dir = R2A.roll_path(tid)[:-5] + "_rec"
        rows = []                                                       # per observation: (meta, cov, cand names, cand pos, proj raw)
        calib = []                                                      # (raw projection, seg centroid) of visible movable objects
        for e in x["episodes"]:
            if not e["obs"]:
                continue
            rz = np.load(os.path.join(rec_dir, f"task{ti}_trial{e['trial']}.npz"), allow_pickle=True)
            onames = [str(q) for q in rz["obj_names"]]
            for mm in e["obs"]:
                jj = mm["j"]
                env.set_init_state(fresh[int(e["trial"])])
                for nm, p, q in zip(onames, rz["obj_pos"][jj], rz["obj_quat"][jj]):
                    jt = inner.objects_dict[nm].joints[0]
                    sim.data.set_joint_qpos(jt, np.r_[np.asarray(p, np.float64), q[3], q[0], q[1], q[2]])
                    sim.data.set_joint_qvel(jt, np.zeros(6))
                sim.forward()
                old = m.geom_group[robot_geoms].copy()
                m.geom_group[robot_geoms] = 5
                try:
                    cov = LC.seg_maps(env, (names, groot), "agentview").astype(np.float32)   # [N names, 256], client orientation
                finally:
                    m.geom_group[robot_geoms] = old
                zo = np.load(os.path.join(R2, "obs", mm["key"] + ".npz"), allow_pickle=True)
                cn = [str(q) for q in zo["cand_names"]]
                cp = np.asarray(zo["cand_pos"], np.float64)
                raw = CU.project_points_from_world_to_camera(cp, M, 256, 256).astype(np.float64)
                for nm in onames:                                       # calibration pairs: movable objects with a visible mask
                    if nm in names:
                        c = cov[names.index(nm)]
                        if c.sum() > 0.5:
                            w = c / c.sum()
                            body = sim.data.body_xpos[inner.obj_body_id[nm]]
                            pr = CU.project_points_from_world_to_camera(np.asarray(body)[None], M, 256, 256)[0].astype(np.float64)
                            calib.append((pr, (w[:, None] * TOK_RC).sum(0)))
                rows.append((mm, cov, cn, cp, raw))
                if len(rows) == 1:
                    print(f"[rs3 regions] {tid} first obs {mm['key']}: seg names {len(names)}, covered tokens per name "
                          f"{ {nm: round(float(c.sum()), 1) for nm, c in zip(names, cov) if c.sum() > 0} }; rec names {onames}; "
                          f"robot geoms hidden {len(robot_geoms)}", flush=True)
        if not rows:
            print(f"[rs3 regions] {tid}: no observation", flush=True)
            env.close()
            continue
        if not calib:
            print(f"[rs3 regions] {tid}: NO calibration pair ({len(rows)} obs)", flush=True)
            raise SystemExit(3)
        # projection orientation: the transform with the smallest median distance between projected body centre and mask centroid
        P = np.array([c[0] for c in calib])
        C = np.array([c[1] for c in calib])
        errs = {t: float(np.median(np.linalg.norm(_xf(P, t) - C, axis=1))) for t in TRANSFORMS}
        tbest = min(errs, key=errs.get)
        report[tid] = dict(calib_pairs=len(calib), median_err_px=errs[tbest], transform=list(tbest))
        print(f"[rs3 regions] {tid}: {len(rows)} obs; projection transform {tbest} median error {errs[tbest]:.1f} px over "
              f"{len(calib)} pairs (next best {sorted(errs.values())[1]:.1f})", flush=True)
        assert tbest == (0, 0, 1) and errs[tbest] < 30.0, (tid, errs)   # one convention on every task (body origin vs mask centroid)
        for mm, cov, cn, cp, raw in rows:
            rc = _xf(raw, tbest)
            out, how = [], []
            for c_i, nm in enumerate(cn):
                if nm in names:
                    r_ = (cov[names.index(nm)] >= COV_MIN).astype(np.float32)
                    if r_.sum() == 0:
                        r_, h = disk(rc[c_i]), "disk_fallback"
                    else:
                        h = "seg"
                else:
                    par = [q for q in names if nm.startswith(q + "_")]
                    d = disk(rc[c_i])
                    if par:
                        pq = max(par, key=len)
                        r_ = d * (cov[names.index(pq)] > 0)
                        h = "site_in_" + pq
                        if r_.sum() == 0:
                            r_, h = d, "site_disk"
                    else:
                        r_, h = d, "site_disk"
                out.append(r_)
                how.append(h)
            R2A.save_atomic(os.path.join(od, mm["key"] + ".npz"), region=np.asarray(out, np.uint8), cand_names=np.array(cn),
                            how=np.array(how), proj_rc=rc.astype(np.float32))
        env.close()
    dump(os.path.join(OUT, "regions", f"_report_{i}of{n}.json"), report)


def check(keys=None):
    """Overlay of the named object's / own target's region on the recorded policy image, for a hand check (no policy, no outcome)."""
    import imageio
    ix = R2A.load_index()
    os.makedirs(os.path.join(OUT, "scratch"), exist_ok=True)
    picks = []
    for tid in sorted(ix):
        for e in ix[tid]["episodes"]:
            for mm in e["obs"]:
                picks.append((tid, mm))
    rng = np.random.default_rng(0)
    sel = [picks[q] for q in rng.choice(len(picks), 12, replace=False)] if keys is None else [p for p in picks if p[1]["key"] in keys]
    for tid, mm in sel:
        z = np.load(os.path.join(R2, "obs", mm["key"] + ".npz"), allow_pickle=True)
        g = np.load(os.path.join(OUT, "regions", mm["key"] + ".npz"), allow_pickle=True)
        cn = [str(q) for q in g["cand_names"]]
        reg = np.zeros(256)
        for nm in mm["named"]:
            reg = np.maximum(reg, g["region"][cn.index(nm)])
        up = np.kron(reg.reshape(16, 16), np.ones((14, 14)))[..., None]
        img = z["image"].astype(np.float32)
        ov = (img * (1 - 0.45 * up) + 0.45 * up * np.array([255, 0, 0])).astype(np.uint8)
        imageio.imwrite(os.path.join(OUT, "scratch", f"check_{mm['key']}.png"), np.concatenate([z["image"], ov], 1))
        print(f"[rs3 check] {mm['key']} '{ix[tid]['axes']['text']}' named {mm['named']} how "
              f"{[str(g['how'][cn.index(nm)]) for nm in mm['named']]} tokens {int(reg.sum())}", flush=True)


# ── split (episode types only) ────────────────────────────────────────────────────────────────────────────────────
def split():
    ix = R2A.load_index()
    tune, test = [], dict(wo_pre=[], su_pre=[], wg_post=[], su_post=[])
    for tid in sorted(ix):
        for e in ix[tid]["episodes"]:
            pre = [m["key"] for m in e["obs"] if m["role"] == "pre"]
            post = [m["key"] for m in e["obs"] if m["role"] == "post"]
            if e["success"]:
                test["su_pre"] += pre
                test["su_post"] += post
            elif e["type"] == "wrong_object":
                test["wo_pre"] += pre
            else:
                if e["type"] == "wrong_goal":
                    test["wg_post"] += post
                if pre:
                    tune.append(pre[len(pre) // 2])                     # one pre-grasp observation per episode: its middle one
    res = dict(tune=tune, test=test, n_tune=len(tune), n_test={k: len(v) for k, v in test.items()})
    dump(os.path.join(OUT, "split.json"), res)
    print("[rs3 split]", json.dumps(res["n_tune"]), json.dumps(res["n_test"]))
    return res


# ── the steering hook (openpi env) ───────────────────────────────────────────────────────────────────────────────
class Steer:
    """Opt-in forward pre-hooks on the self-attention modules: + b on the attention logits (the additive 4-D mask) from the chosen
    queries to the chosen image-token keys, in the chosen layers.  cfg None -> exact no-op."""

    def __init__(self, bk):
        self.bk, self.cfg, self.calls = bk, None, {"word": 0, "expert": 0}
        pwe = bk.model.paligemma_with_expert
        self.handles = []
        for site, layers in (("word", pwe.paligemma.language_model.layers), ("expert", pwe.gemma_expert.model.layers)):
            for li, ly in enumerate(layers):
                self.handles.append(ly.self_attn.register_forward_pre_hook(self._hook(site, li), with_kwargs=True))

    def _hook(self, site, li):
        import torch

        def f(mod, args, kwargs):
            c = self.cfg
            if c is None or li not in c["layers"] or site not in c["sites"]:
                return None
            msk = kwargs.get("attention_mask")
            assert msk is not None and msk.dim() == 4, "rs3: no 4-D additive mask reached the attention"
            B, _, Q, K = msk.shape
            hs = kwargs["hidden_states"]
            assert hs.shape[1] == Q, (site, hs.shape, msk.shape)
            cols = torch.as_tensor(c["cols"], device=msk.device, dtype=torch.long)
            bias = torch.zeros((1, 1, Q, K), dtype=msk.dtype, device=msk.device)
            if site == "word":
                assert Q == K, "the word site runs in the prefix pass"
                rows = torch.as_tensor(c["rows"], device=msk.device, dtype=torch.long)
                if len(rows) == 0 or len(cols) == 0:
                    return None
                bias[0, 0, rows[:, None], cols[None, :]] = c["b"]
            else:
                assert Q == self.bk.H, "the expert site runs in the denoising pass"
                if len(cols) == 0:
                    return None
                bias[0, 0, :, cols] = c["b"]
            self.calls[site] += 1
            return args, dict(kwargs, attention_mask=msk + bias)
        return f


def _tokenizer():
    from openpi.models.tokenizer import PaligemmaTokenizer
    return PaligemmaTokenizer(48)._tokenizer


def role_rows(sp, text):
    """-> (tokens, {"O": token positions of the object-phrase content words, "G": of the goal-phrase content words}, words).
    Positions index the prompt's tokens (BOS = 0) as the pi0-format tokenizer builds them: encode(text, bos) + encode("\\n")."""
    from lifelong_ood.policy import primitives as P
    import m1_compose_create as M1
    cleaned = text.strip().replace("_", " ").replace("\n", " ")
    ids = list(sp.encode(cleaned, add_bos=True)) + list(sp.encode("\n"))
    pieces = [sp.id_to_piece(int(q)) for q in ids]
    word_of, w = [], -1
    for k, pc in enumerate(pieces):
        if k == 0 or k == len(pieces) - 1:
            word_of.append(-1)
            continue
        if k == 1 or pc.startswith("▁"):                       # the first word carries no word-start mark
            w += 1
        word_of.append(w)
    words = cleaned.lower().split()
    assert w + 1 == len(words), (text, pieces)
    O, G = set(), set()
    for cl in P.steps_of(cleaned.lower()):
        cl = cl.strip()
        cw = cl.split()
        s = next((q for q in range(len(words) - len(cw) + 1) if words[q:q + len(cw)] == cw), None)
        if s is None:
            continue
        verb, prep, target = P.step_key(cl).split("|")
        if cl.startswith("pick up ") and (" and place it " in cl or " and put it " in cl):
            o0 = s + 2
            o1 = s + cw.index("and")
        else:
            o0 = s + len(verb.split())
            o1 = s + len(cw)
        if prep and target and cl.endswith(f"{prep} {target}"):
            g0 = s + len(cw) - len(target.split())
            G.update(range(g0, s + len(cw)))
            if not cl.startswith("pick up "):
                o1 = g0 - len(prep.split())
        O.update(range(o0, o1))
    keep = lambda S: sorted(q for q in S if M1.content_words(words[q]))     # noqa: E731
    O, G = keep(O), keep(G)
    rows = {"O": [k for k, q in enumerate(word_of) if q in O], "G": [k for k, q in enumerate(word_of) if q in G]}
    return ids, rows, dict(words=words, O=[words[q] for q in O], G=[words[q] for q in G])


def grid():
    return [dict(name=f"{s}|{ln}|{b:g}", sites=(("word", "expert") if s == "both" else (s,)), site=s, layer_set=ln,
                 layers=LAYER_SETS[ln], b=float(b)) for s in SITES for ln in LAYER_SETS for b in BIASES]


def run_obs(bk, st, sp, key, arms, tok_cache):
    """One observation: every arm's 12 unnormalised chunks.  arms = [(name, cfg or None, target 'named' | 'cf', seed)]."""
    import torch
    z = np.load(os.path.join(R2, "obs", key + ".npz"), allow_pickle=True)
    g = np.load(os.path.join(OUT, "regions", key + ".npz"), allow_pickle=True)
    role = str(z["role"])
    cn = [str(q) for q in z["cand_names"]]
    assert cn == [str(q) for q in g["cand_names"]], key
    text = str(z["text"])
    if text not in tok_cache:
        tok_cache[text] = role_rows(sp, text)
    ids, rows, parse = tok_cache[text]
    m = META[key]
    named = list(m["named"])
    cf = cf_pick(key, cn, set(named))
    reg = lambda nms: np.flatnonzero(np.max([g["region"][cn.index(q)] for q in nms], 0) > 0).astype(np.int64)   # noqa: E731
    cols = {"named": reg(named), "cf": reg([cf]) if cf else np.zeros(0, np.int64)}
    od = {"observation/image": z["image"], "observation/wrist_image": z["wrist"],
          "observation/state": np.asarray(z["state"], np.float32), "prompt": text}
    out, info = {}, {}
    with torch.no_grad():
        obs = bk.observation(od)
        n_tok = int(obs.tokenized_prompt_mask.sum())
        assert list(np.asarray(obs.tokenized_prompt[0][:n_tok].cpu())) == ids, (key, ids)
        st.cfg = None
        pre0 = bk.prefix(obs)
        off = bk.text_span(obs)[0]
        for name, cfg, tg, seed in arms:
            if tg == "cf" and cf is None:
                continue
            if cfg is None:
                pre = pre0
            else:
                st.cfg = dict(sites=cfg["sites"], layers=set(cfg["layers"]), b=cfg["b"], cols=cols[tg],
                              rows=[off + q for q in rows["O" if role == "pre" else "G"]])
                pre = bk.prefix(obs) if "word" in cfg["sites"] else pre0
            try:
                X = bk.sample(N_SAMP, pre, seed=seed)
            finally:
                st.cfg = None
            out[name] = bk.unnormalize(X).reshape(N_SAMP, bk.H, -1).astype(np.float32)
    info = dict(key=key, role=role, named=named, cf=cf, n_cols=dict(named=int(len(cols["named"])), cf=int(len(cols["cf"]))),
                rows=rows["O" if role == "pre" else "G"], parse=parse)
    return out, info


META = {}


def _load_meta():
    ix = R2A.load_index()
    for tid, x in ix.items():
        for e in x["episodes"]:
            for m in e["obs"]:
                META[m["key"]] = dict(m, task=tid, trial=e["trial"], type=e["type"], success=e["success"])


def gpu_stage(stage, shard):
    import torch
    from lifelong_ood.policy.backbone import PolicyBackbone
    _load_meta()
    sp = json.load(open(os.path.join(OUT, "split.json"), encoding="utf-8"))
    if stage in ("tune", "smoke"):
        keys = list(sp["tune"])
        arms = [("u11", None, "named", SEED_A)] + [(c["name"] + "|" + tg, c, tg, SEED_A) for c in grid() for tg in ("named", "cf")]
    else:
        ch = json.load(open(os.path.join(OUT, "choice.json"), encoding="utf-8"))
        c = next(q for q in grid() if q["name"] == ch["chosen"])
        keys = sorted(set(k for v in sp["test"].values() for k in v))
        arms = [("u11", None, "named", SEED_A), ("u21", None, "named", SEED_B), ("s", c, "named", SEED_A), ("s_cf", c, "cf", SEED_A)]
        for b in BIASES:
            if b != c["b"]:
                q = next(q for q in grid() if q["site"] == c["site"] and q["layer_set"] == c["layer_set"] and q["b"] == b)
                arms.append((f"sweep_b{b:g}", q, "named", SEED_A))
    od = os.path.join(OUT, stage if stage != "smoke" else "scratch/smoke")
    i, n = map(int, shard.split("/"))
    mine = keys[i::n]
    if stage == "smoke":
        mine = keys[:2]
        arms = [arms[0]] + [a for a in arms if a[0] in ("expert|L0-7|8|named", "word|L0-1|8|named", "both|L0-3|4|cf")]
    os.makedirs(od, exist_ok=True)
    bk = PolicyBackbone(CFG, checkpoint_dir=CKPT, device="cuda")
    st = Steer(bk)
    sptok = _tokenizer()
    print(f"[rs3 {stage}] {bk.describe()}; {len(mine)} observations x {len(arms)} arms, shard {shard}", flush=True)
    tc, t_all, done = {}, time.time(), 0
    for key in mine:
        outp = os.path.join(od, key + ".npz")
        if os.path.exists(outp):
            continue
        t0 = time.time()
        raw, info = run_obs(bk, st, sptok, key, arms, tc)
        info["secs"] = time.time() - t0
        if stage == "smoke":
            zm = np.load(os.path.join(R2, "modes", key + ".npz"), allow_pickle=True)
            a, b = json.loads(str(zm["slices"]))["draft_instr"]
            par = float(np.abs(zm["raw"][a:b] - raw["u11"]).max())
            moves = {k: float(np.abs(v - raw["u11"]).mean()) for k, v in raw.items() if k != "u11"}
            print(f"[rs3 smoke] {key}: parity max|u11 - R2a draft_instr| = {par:.2e}; mean |steered - u11| {json.dumps(moves)}; "
                  f"hook calls {st.calls}; parse {json.dumps(info['parse'])} rows {info['rows']} cols {info['n_cols']}", flush=True)
        R2A.save_atomic(outp, arms=np.array(list(raw)), raw=np.stack([raw[k] for k in raw]), info=np.array(json.dumps(info)))
        done += 1
        print(f"[rs3 {stage}] {key}: {info['secs']:.1f}s ({done} done, {time.time() - t_all:.0f}s)", flush=True)
    print(f"[rs3 {stage}] shard {shard} finished: {done} new", flush=True)


# ── reads (chunks env) ───────────────────────────────────────────────────────────────────────────────────────────
def obs_shares(key, f):
    """-> {arm: (share at named or cf target, share pointing at nothing)} for one stored observation."""
    z = np.load(os.path.join(R2, "obs", key + ".npz"), allow_pickle=True)
    s = np.load(f, allow_pickle=True)
    info = json.loads(str(s["info"]))
    cn = [str(q) for q in z["cand_names"]]
    eef, cp = z["state"][:3], z["cand_pos"]
    out = {}
    for arm, raw in zip([str(q) for q in s["arms"]], s["raw"]):
        pts = np.array([R2A.pointed(r, eef, cp)[0] for r in raw])
        tgt = {info["cf"]} if arm.endswith("cf") else set(info["named"])
        hit = np.array([p >= 0 and cn[p] in tgt for p in pts], float)
        out[arm] = (float(hit.mean()), float((pts < 0).mean()))
        if arm == "u11" and info["cf"] is not None:                    # the unsteered share at the counterfactual target
            out["u11@cf"] = (float(np.mean([p >= 0 and cn[p] == info["cf"] for p in pts])), float((pts < 0).mean()))
    return out, info


def choose(record):
    _load_meta()
    sp = json.load(open(os.path.join(OUT, "split.json"), encoding="utf-8"))
    rows = []
    for key in sp["tune"]:
        f = os.path.join(OUT, "tune", key + ".npz")
        assert os.path.exists(f), f"tuning incomplete: {key}"
        rows.append(obs_shares(key, f)[0])
    res = {}
    for c in grid():
        nm, cf = c["name"] + "|named", c["name"] + "|cf"
        g_named = float(np.mean([r[nm][0] - r["u11"][0] for r in rows]))
        rc = [r for r in rows if cf in r]
        g_cf = float(np.mean([r[cf][0] - r["u11@cf"][0] for r in rc]))
        still = float(np.mean([r[nm][1] - r["u11"][1] for r in rows]))
        res[c["name"]] = dict(g_named=g_named, g_cf=g_cf, still=still, admissible=bool(g_named >= ADMIT_NAMED and still <= ADMIT_STILL),
                              n=len(rows), n_cf=len(rc))
    order = {s: i for i, s in enumerate(SITES)}
    lorder = {ln: i for i, ln in enumerate(LAYER_SETS)}
    adm = [k for k, v in res.items() if v["admissible"]]
    pool = adm if adm else list(res)
    crit = "g_cf" if adm else "g_named"
    best = max(res[k][crit] for k in pool)
    near = [k for k in pool if res[k][crit] >= best - TIE]
    cg = {c["name"]: c for c in grid()}
    chosen = min(near, key=lambda k: (cg[k]["b"], len(cg[k]["layers"]), order[cg[k]["site"]]))
    out = dict(chosen=chosen, criterion=crit, any_admissible=bool(adm), table=res, unsteered_named=float(np.mean([r["u11"][0] for r in rows])),
               unsteered_still=float(np.mean([r["u11"][1] for r in rows])), n_tune=len(rows))
    dump(os.path.join(OUT, "choice.json"), out)
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]["g_cf"]):
        print(f"[rs3 choose] {k:22s} g_cf {v['g_cf']:+.3f} g_named {v['g_named']:+.3f} still {v['still']:+.3f} adm {v['admissible']}")
    print(f"[rs3 choose] chosen {chosen} by {crit}; unsteered named {out['unsteered_named']:.3f} on {len(rows)} tuning obs", flush=True)
    if record:
        from lifelong_ood import ledger
        print(ledger.record("rs3_choice", bench="libero_90_c", tag="choice", seed=SEED_A, metrics=out,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=_settings(),
                            script="experiments/rs3_steer.py", command=" ".join(sys.argv), backbone=dict(config=CFG, checkpoint=CKPT),
                            artifacts=[os.path.join(OUT, "choice.json")]))


def _settings():
    return dict(n_samp=N_SAMP, seeds=[SEED_A, SEED_B], sites=SITES, layer_sets=LAYER_SETS, biases=BIASES, cov_min=COV_MIN,
                disk_px=DISK_PX, admit_named=ADMIT_NAMED, admit_still=ADMIT_STILL, tie=TIE, gate_rise=GATE_RISE, gate_harm=GATE_HARM)


def read(record):
    from lifelong_ood import stats
    _load_meta()
    sp = json.load(open(os.path.join(OUT, "split.json"), encoding="utf-8"))
    ch = json.load(open(os.path.join(OUT, "choice.json"), encoding="utf-8"))
    r2a = json.load(open(os.path.join(R2, "read.json"), encoding="utf-8"))
    res = dict(chosen=ch["chosen"], settings=_settings(), sets={})
    parity = []
    for sname, keys in sp["test"].items():
        ep = {}
        for key in keys:
            f = os.path.join(OUT, "test", key + ".npz")
            assert os.path.exists(f), f"test incomplete: {key}"
            sh, info = obs_shares(key, f)
            zm = np.load(os.path.join(R2, "modes", key + ".npz"), allow_pickle=True)
            r2 = R2A.obs_readout(META[key], np.load(os.path.join(R2, "obs", key + ".npz"), allow_pickle=True), zm)
            parity.append(abs(r2["policy_sample"] - sh["u11"][0]))
            for k in ("ratio_top", "coverage", "instr_mode_top", "cag_mode_top", "null_mass_top", "random", "policy_exec"):
                sh["r2a_" + k] = (r2[k], 0.0)
            m = META[key]
            ep.setdefault((m["task"], m["trial"]), []).append(sh)
        arms = sorted({a for v in ep.values() for s in v for a in s})
        per = {}                                                        # arm -> task -> [episode means]
        for (t, _), v in ep.items():
            for a in arms:
                vals = [s[a][0] for s in v if a in s]
                if vals:
                    per.setdefault(a, {}).setdefault(t, []).append(float(np.mean(vals)))
        still = {a: float(np.mean([np.mean([s[a][1] for s in v if a in s]) for v in ep.values() if any(a in s for s in v)])) for a in arms}

        def pairs(a, b):
            out = {}
            for (t, _), v in ep.items():
                if all(a in s for s in v) and all(b in s for s in v):
                    pa, pb = out.setdefault(t, ([], []))
                    pa.append(float(np.mean([s[a][0] for s in v])))
                    pb.append(float(np.mean([s[b][0] for s in v])))
            return {t: (np.asarray(p[0]), np.asarray(p[1])) for t, p in out.items()}
        d = stats.bootstrap_diff(pairs("s", "u11"), B=10000, seed=0, return_samples=True)
        smp = d.pop("samples")
        floor = stats.bootstrap_diff(pairs("u21", "u11"), B=10000, seed=0)
        cfp = {}
        for (t, _), v in ep.items():
            if all("s_cf" in s for s in v):
                pa, pb = cfp.setdefault(t, ([], []))
                pa.append(float(np.mean([s["s_cf"][0] for s in v])))
                pb.append(float(np.mean([s["u11@cf"][0] for s in v])))
        cfd = stats.bootstrap_diff({t: (np.asarray(p[0]), np.asarray(p[1])) for t, p in cfp.items()}, B=10000, seed=0) if cfp else None
        res["sets"][sname] = dict(n_obs=len(keys), n_episodes=len(ep), n_tasks=len({t for t, _ in ep}),
                                  rates={a: R2A.ci_rate(per[a]) for a in arms}, still=still, diff=d,
                                  p_rise=float(np.mean(smp <= 0.0)), p_harm=float(np.mean(smp <= GATE_HARM)),
                                  noise_floor_u21_minus_u11=floor, cf_control=cfd)
    res["parity_policy_sample_max_abs"] = float(max(parity))
    # hypotheses: H1 / H3 rise >= 0.20 with CI excluding 0; H2 / H4 non-inferiority at -0.05; Holm over the four
    hyp = {"H1": ("wo_pre", "rise"), "H2": ("su_pre", "harm"), "H3": ("wg_post", "rise"), "H4": ("su_post", "harm")}
    ps = {h: res["sets"][s]["p_rise" if k == "rise" else "p_harm"] for h, (s, k) in hyp.items()}
    order = sorted(ps, key=ps.get)
    H, prev = {}, True
    for rank, h in enumerate(order):
        thr = ALPHA / (len(order) - rank)
        ok = prev and ps[h] < thr
        prev = ok
        s, k = hyp[h]
        d = res["sets"][s]["diff"]
        extra = (d["diff"] >= GATE_RISE and d["lo"] > 0) if k == "rise" else True
        H[h] = dict(set=s, kind=k, p=ps[h], holm_threshold=thr, holm_ok=bool(ok), estimate_ok=bool(extra), passed=bool(ok and extra),
                    diff=d)
    res["hypotheses"] = H
    res["stage1_gate"] = bool(H["H1"]["passed"] and H["H2"]["passed"])
    res["r2a_zs_type_tables"] = {role: {t: {a: v["mean"] for a, v in tb.items()} for t, tb in r2a[role]["zs_type"].items()}
                                 for role in ("pre", "post") if role in r2a}
    path = os.path.join(OUT, "read.json")
    dump(path, res)
    for s, v in res["sets"].items():
        print(f"\n[rs3 read] {s}: {v['n_obs']} obs, {v['n_episodes']} ep, {v['n_tasks']} tasks; diff s - u11 {v['diff']['diff']:+.3f} "
              f"[{v['diff']['lo']:+.3f}, {v['diff']['hi']:+.3f}]; noise floor u21 - u11 {v['noise_floor_u21_minus_u11']['diff']:+.3f} "
              f"[{v['noise_floor_u21_minus_u11']['lo']:+.3f}, {v['noise_floor_u21_minus_u11']['hi']:+.3f}]")
        print("[rs3 read]   " + "  ".join(f"{a} {r['mean']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]" for a, r in v["rates"].items()
                                          if r["mean"] is not None))
        print("[rs3 read]   still " + json.dumps({a: round(x, 3) for a, x in v["still"].items()}) + "  cf control " +
              json.dumps(v["cf_control"]))
    print("\n[rs3 read] hypotheses " + json.dumps({h: {k: v[k] for k in ("p", "holm_threshold", "holm_ok", "estimate_ok", "passed")}
                                                   for h, v in H.items()}))
    print(f"[rs3 read] stage-1 gate {'PASSES' if res['stage1_gate'] else 'FAILS'}; parity max |policy_sample - u11| "
          f"{res['parity_policy_sample_max_abs']:.3g}")
    if record:
        from lifelong_ood import ledger
        print(ledger.record("rs3_read", bench="libero_90_c", tag="stage1", seed=SEED_A, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=_settings(),
                            script="experiments/rs3_steer.py", command=" ".join(sys.argv), backbone=dict(config=CFG, checkpoint=CKPT),
                            artifacts=[path]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in ("regions", "check", "split", "smoke", "tune", "choose", "test", "read", "record"):
        ap.add_argument("--" + f, action="store_true")
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    if a.regions:
        regions(a.shard)
    if a.check:
        check()
    if a.split:
        split()
    if a.smoke:
        gpu_stage("smoke", a.shard)
    if a.tune:
        gpu_stage("tune", a.shard)
    if a.choose:
        choose(a.record)
    if a.test:
        gpu_stage("test", a.shard)
    if a.read:
        read(a.record)
