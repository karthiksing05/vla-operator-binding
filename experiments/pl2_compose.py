"""PLAN L, PROBE L2 (docs/PLAN_LANG.md; pre-registration experiments/prereg/pl2_compose.py, ledger m3_prereg tag pl2_compose):
fast composition from language prototypes, no gradient, the source chosen by the ONE demo's flow loss (no practice, no reward).

Operators (lifelong_ood/lang/transplant.py): (a) word transplant, (b) template-matched transplant, (c) TLI with the pair and lambda
chosen on the demo, (d) the per-role schedule.  Per task: demo 0 selects (sources, layer span, template, pair, lambda, and the
operator: l2sel = the operator with the lowest demo-0 loss); demo 1 is held out and only scored.  Baselines: the new instruction
(zs), the null prompt, TLI with Li's hand pairs (tli_li).  Controls on l2sel's form: a random source sentence (rand) and a
position-matched source with no word overlap (pos; the T8 control: the training sentence whose demos grasp / release nearest to
where demo 0 grasps / releases, sharing no content word with the phrase).

DEMOS.  Li's OOD tasks have none, so L2 records them: successful rollouts on a NEW start-state set (fresh_ood_pl2d, BASE 3300000)
by the OTHER policy's strongest earlier arm on set d (G13: TLI with Li's pairs or the practice-chosen grasp switch); demos for
pi0.5 come from pi0, demos for pi0 from pi0.5 (own-policy fallback only where the other policy has fewer than 2 successes; flagged).
Demo 0 = the first success, demo 1 = the second.  LIBERO-90: the task's own IPEC demos 0 and 1 (offline only).

    $PY experiments/pl2_compose.py --split                         dev / confirm split of Li's 20 tasks (seeded hash)
    $PY experiments/pl2_compose.py --demos-launch | --demos-read   record the demos (packs on overcap) / index them
    sbatch ... SCRIPT="experiments/pl2_compose.py --offline --backbone pi05_libero --set ood"     (GPU; resumable per task)
    $PY experiments/pl2_compose.py --offline-read [--record]       tables, figure, ledger pl2_offline
    $PY experiments/pl2_compose.py --launch-dev [--dry]            closed loop: 8 dev tasks x 20 episodes x arms, fresh_ood_pl2
    $PY experiments/pl2_compose.py --read-dev [--record]           success, part correctness, ledger pl2_dev, figures
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
R = os.path.join(ROOT, "results", "robot")
OUT = os.path.join(ROOT, "results", "lang", "pl2")
FIG = os.path.join(ROOT, "figures", "robot")
PY = os.path.expanduser("~/flash/miniconda3/envs/chunks/bin/python")
SEED = 13
BBS = ("pi05_libero", "pi0_libero")
DEMO_DIR, DEMO_N, TEST_DIR = "fresh_ood_pl2d", 8, "fresh_ood_pl2"
EXCLUDE = "sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw"
K_DRAWS, F_TARGET = 4, 36
LAMS = (12, 24, 36)
N_L90 = 16
OPS = ("word", "tmpl", "tli_sel", "role")
ARMS_DEV = ("zs", "tli_li", "word", "tmpl", "tli_sel", "role", "rand", "pos")


def short(bb):
    return bb.split("_")[0]


def bsfx(bb):
    return "" if bb == "pi05_libero" else "_" + short(bb)


def plan():
    return json.load(open(os.path.join(R, "g4_plan.json")))


def sh(k):
    return k.replace(":", "_")


# ── split ────────────────────────────────────────────────────────────────────────────────────────────────────────
def split():
    """Li's 20 tasks -> 8 dev / 12 confirm, by sha1('pl2:' + key) (fixed before any L2 number)."""
    ks = sorted(plan()["tasks"], key=lambda k: hashlib.sha1(f"pl2:{k}".encode()).hexdigest())
    return dict(dev=sorted(ks[:8]), confirm=sorted(ks[8:]))


# ── demos ────────────────────────────────────────────────────────────────────────────────────────────────────────
def gen_arm(k, v, gen_bb):
    """The generator policy's strongest earlier arm on set d (G13 per-task counts): TLI with Li's pairs or the practice-chosen
    grasp switch (and the policy alone for pi0.5); ties to TLI."""
    import g12_tli as T
    g13 = json.load(open(os.path.join(R, f"g13_read_{short(gen_bb)}.json")))["per_task"][k]
    g10 = json.load(open(os.path.join(R, f"g10_read_{short(gen_bb)}.json")))["chosen"][k]
    cand = [("tli_li", T.arm(v)), ("psel", g10)] + ([("zs", "zs")] if gen_bb == "pi05_libero" else [])
    name, arm = max(cand, key=lambda c: (g13[c[0]], c[0] == "tli_li"))
    return arm, int(g13[name])


def _spec(tid, arm, suite, bb, exp, trials, fresh, record=False):
    """A roll_pack instance line, built as launch_pack.main builds it (same MODE, TAG, OUT conventions)."""
    import launch_pack as LP
    LP.SUITE_OF.update(suite=suite, backbone=bb if bb != "pi05_libero" else "")
    s = LP.arm_spec(tid, arm, False)
    if s is None:
        return None
    tag = arm.replace("@", "_").replace("-", "_") + f"_seed{SEED}" + bsfx(bb)
    out = os.path.join(R, f"{exp}_{suite}_{tid}_{tag}.json")
    sp = dict(MODE=s[0], SUITE=suite, TASKS=tid, TRIALS=trials, FIRST=0, EXPERIMENT=exp, TAG=tag, PROGRAM_BYTES=s[1], OUT=out,
              SERVE_SEED=str(SEED), BIND="track", FRESH=os.path.join(R, fresh))
    if record:
        sp.update(RECORD="1", OBJECTS="1", ALL_OBJECTS="1")
    if bb != "pi05_libero":
        sp.update(BACKBONE_CONFIG=bb, BACKBONE_CKPT=f"/coc/flash10/ksingara3/robot/openpi_cache/openpi-assets/checkpoints/{bb}_pytorch")
    return sp


def submit(specs, lanes=4, per=4, dry=False, after=None):
    """Pack instances 4 per GPU and submit them in `lanes` chains (afterany), so at most `lanes` of these jobs run at once."""
    import launch_pack as LP
    todo = [s for s in specs if s is not None and not os.path.exists(s["OUT"])]
    os.makedirs(os.path.join(R, ".packs"), exist_ok=True)
    n_pack = len(os.listdir(os.path.join(R, ".packs")))
    last = [after] * lanes if after else [None] * lanes
    jobs = []
    for j, i in enumerate(range(0, len(todo), per)):
        grp = todo[i:i + per]
        while True:
            n_pack += 1
            path = os.path.join(R, ".packs", f"p_{n_pack:04d}.tsv")
            try:
                os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                break
            except FileExistsError:
                continue
        with open(path, "w", encoding="utf-8") as f:
            for sp in grp:
                f.write("\t".join(f"{k}={v}" for k, v in sp.items()) + "\n")
        lane = j % lanes
        args = ([f"--dependency=afterany:{last[lane]}"] if last[lane] else []) + [
            "--partition=overcap", "--account=overcap", "--qos=short", "--time=04:00:00", "--gres=gpu:a40:1",
            f"--exclude={EXCLUDE}", f"--export=ALL,SPECS={path}", "experiments/roll_pack.sbatch"]
        jid = LP.sbatch(args, dry)
        last[lane] = jid
        jobs.append(jid)
        print(f"pack {os.path.basename(path)} [{', '.join(str(s['TASKS']) + ':' + s['TAG'] for s in grp)}] lane {lane} -> {jid}", flush=True)
    return jobs


def demos_launch(dry):
    pl = plan()
    specs = []
    for gen in BBS:
        for k, v in pl["tasks"].items():
            arm, c = gen_arm(k, v, gen)
            specs.append(_spec(v["index"], arm, v["suite"], gen, f"pl2_demo_{k.split(':')[0]}", DEMO_N, DEMO_DIR, record=True))
            print(k, short(gen), arm, f"(set d {c}/20)")
    submit(specs, dry=dry)


def demos_read():
    """-> results/lang/pl2/demos.json {backbone: {task: dict(gen, arm, self, d0, d1)}} (recorded npz paths)."""
    pl = plan()
    rec = {}
    for gen in BBS:
        for k, v in pl["tasks"].items():
            arm, _ = gen_arm(k, v, gen)
            tag = arm.replace("@", "_").replace("-", "_") + f"_seed{SEED}" + bsfx(gen)
            out = os.path.join(R, f"pl2_demo_{k.split(':')[0]}_{v['suite']}_{v['index']}_{tag}.json")
            if not os.path.exists(out):
                rec[(gen, k)] = None
                continue
            eps = sorted((e for t in json.load(open(out))["tasks"].values() for e in t["episodes"]), key=lambda e: e["trial"])
            d = out[:-5] + "_rec"
            rec[(gen, k)] = dict(arm=arm, n=len(eps), succ=[int(e["trial"]) for e in eps if e["success"]],
                                 files={int(e["trial"]): os.path.join(d, f"task{v['index']}_trial{e['trial']}.npz") for e in eps})
    idx = {}
    for bb in BBS:
        other = [g for g in BBS if g != bb][0]
        idx[bb] = {}
        for k in pl["tasks"]:
            pick = None
            for gen, own in ((other, False), (bb, True)):
                r = rec.get((gen, k))
                if r is not None and len(r["succ"]) >= 2:
                    pick = dict(gen=gen, arm=r["arm"], self=own, d0=r["files"][r["succ"][0]], d1=r["files"][r["succ"][1]],
                                successes=f"{len(r['succ'])}/{r['n']}")
                    break
            idx[bb][k] = pick
            print(short(bb), k, None if pick is None else (short(pick["gen"]), pick["arm"], pick["successes"], "SELF" if pick["self"] else ""))
    os.makedirs(OUT, exist_ok=True)
    json.dump(idx, open(os.path.join(OUT, "demos.json"), "w", encoding="utf-8"), indent=1)
    miss = [(short(g), k) for (g, k), r in rec.items() if r is None]
    print("missing generator files:", miss)


# ── demo frames ──────────────────────────────────────────────────────────────────────────────────────────────────
def carry_of(actions, replan=5):
    """Carrying frames from the demo's own gripper command: from 3 queries after the first sustained close (>= 10 frames)
    until the next opening (held_ops._Grasp's rule, offline).  -> bool [T], first close frame, release frame."""
    g = np.asarray(actions)[:, 6] > 0
    T = len(g)
    c0 = next((f for f in range(T) if g[f:f + 10].all() and len(g[f:f + 10]) == 10), None)
    out = np.zeros(T, bool)
    if c0 is None:
        return out, -1, -1
    rel = next((f for f in range(c0, T) if not g[f]), T)
    out[min(T, c0 + 3 * replan):rel] = True
    return out, int(c0), int(rel)


def load_demo(src, H=10):
    """A demo -> dict(frames [raw obs dicts at F subsampled frames], chunks [F, H, 7], states [F, 8], carry [F], q [F], idx [F],
    actions, state, obj (dict or None), c0, rel)."""
    if isinstance(src, str):                                             # a recorded OOD episode (npz)
        z = np.load(src, allow_pickle=True)
        act, st = z["action"].astype(np.float32), z["state"].astype(np.float32)
        img, wr = z["image"], z["wrist"]
        obj = dict(pos=z["obj_pos"], names=[str(x) for x in z["obj_names"]]) if "obj_pos" in z else None
        get = lambda i: {"observation/image": img[i], "observation/wrist_image": wr[i], "observation/state": st[i]}   # noqa: E731
    else:                                                                # (Demos90, episode index): LIBERO-90's own demo
        d90, e = src
        ep = d90.load(e)
        ep["image"], ep["wrist_image"] = d90.frames(e)
        act, st, obj = ep["actions"], ep["state"], None
        get = lambda i: {k_: v_ for k_, v_ in d90.observation(ep, i).items() if k_ != "prompt"}                      # noqa: E731
    T = len(act)
    stride = max(1, int(round(T / F_TARGET)))
    idx = np.arange(0, T, stride)
    ci = np.minimum(idx[:, None] + np.arange(H)[None], T - 1)
    carry, c0, rel = carry_of(act)
    return dict(frames=[get(int(i)) for i in idx], chunks=act[ci], states=st[idx], carry=carry[idx], q=idx // 5, idx=idx,
                actions=act, state=st, obj=obj, c0=c0, rel=rel, T=T, _get=get)


# ── the training side: candidates, contacts ─────────────────────────────────────────────────────────────────────
def train_contacts(demos):
    """-> {task: dict(grasp [3], release [3])}: mean eef position at the first sustained close / the next opening over the first 5
    demos of each single-step pick-and-place training task (the T8 control's position match).  Cached."""
    import lifelong_ood.lang.transplant as LT
    f = os.path.join(OUT, "train_contacts.json")
    if os.path.exists(f):
        return {int(k): v for k, v in json.load(open(f)).items()}
    out = {}
    for t in range(40):
        if LT.template_of(demos.tasks[t]) is None:
            continue
        G, Rl = [], []
        for e in demos.episodes(t)[:5]:
            ep = demos.load(e)
            _, c0, rel = carry_of(ep["actions"])
            if c0 < 0:
                continue
            G.append(ep["state"][c0, :3])
            Rl.append(ep["state"][min(rel, len(ep["state"]) - 1), :3])
        if G:
            out[t] = dict(grasp=np.mean(G, 0).tolist(), release=np.mean(Rl, 0).tolist())
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(f, "w", encoding="utf-8"), indent=1)
    return out


def l90_tasks():
    """N_L90 LIBERO-90 single-step pick-and-place tasks with a distinct text not among the 40 training texts, a content word held in a
    training sentence, >= 2 own-scene demos; chosen by sha1 rank (fixed)."""
    import importlib.util
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy import primitives as P
    import lifelong_ood.lang.transplant as LT
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    texts, _ = m.libero90_texts()
    tx = D.Demos().tasks
    se = json.load(open(os.path.join(R, "l90_scene_episodes.json")))
    seen, ok = set(), []
    for t, s in texts.items():
        if s in seen or s in tx.values() or len(P.steps_of(s)) != 1 or s.startswith("stack") or len(se.get(str(t), [])) < 2:
            continue
        r = LT.roles(s)
        if not r["o"] or not re.match(r"^(on|in|into|onto|to)\b", r["g"]):
            continue
        if not any(LT.contains(tx[k], r["o"]) or LT.contains(tx[k], r["g"]) for k in range(40)):
            continue
        seen.add(s)
        ok.append((t, s))
    ok.sort(key=lambda x: hashlib.sha1(f"pl2l90:{x[0]}".encode()).hexdigest())
    return sorted(ok[:N_L90])


# ── offline selection and scoring ───────────────────────────────────────────────────────────────────────────────
def _tli_lat(bk, cfg, demos, t):
    from lifelong_ood.policy import tli as TL
    f = TL.path(cfg, t, ROOT)
    if not os.path.exists(f):
        lat, _ = TL.identify(bk, demos, t)
        tmp = f"{f}.{os.getpid()}.tmp.npy"
        np.save(tmp, lat)
        os.replace(tmp, f)
    return np.load(f)


def select_task(bk, cfg, demos, key, text, d0, d1, li=None, named=None, log=print):
    """Demo-0 selection of every operator and control; demo-1 (held-out) scores.  -> result dict, plus the arms' specs."""
    import torch
    import lifelong_ood.lang.transplant as LT
    tx = demos.tasks
    nl = len(bk.cond_layers())
    SP = [(1, nl - 1), (1, 6), (7, 12), (13, nl - 1)]
    s0 = LT.DemoScorer(bk, d0["frames"], d0["chunks"], d0["states"], d0["carry"], d0["q"], K=K_DRAWS, seed=1)
    s1 = LT.DemoScorer(bk, d1["frames"], d1["chunks"], d1["states"], d1["carry"], d1["q"], K=K_DRAWS, seed=2)
    r = LT.roles(text)
    n_eval = [0]
    cache = {}

    def L(sc, prompt, edits=(), lo=1, hi=nl - 1, sched="og", tli=None):
        n_eval[0] += 1
        comp = LT.Composer(bk, edits, lo=lo, hi=hi, schedule=sched, tli=tli) if (edits or tli is not None) else None
        return sc.loss(prompt, comp)

    def edit(prompt, role, src, src_phrase=None):
        kk = (prompt, role, src, src_phrase)
        if kk not in cache:
            cache[kk] = LT.make_edit(bk, cfg, demos, prompt, r[role], src, src_phrase=src_phrase, role=role)
        return cache[kk]
    cands = {w: [t for t in range(40) if LT.contains(tx[t], r[w]) and t != demos.task_of_text.get(text)] for w in "og"}
    cands = {w: [t for t in c if edit(text, w, t) is not None] for w, c in cands.items()}
    log(f"[pl2] {key} '{text}': o '{r['o']}' {cands['o']} | g '{r['g']}' {cands['g']}")
    res = dict(key=key, text=text, roles=r, cands=cands, d0=dict(T=d0["T"], c0=d0["c0"], rel=d0["rel"]), sel={}, l0={}, l1={})
    arms = {}                                                            # name -> (prompt, edits, lo, hi, sched, tli meta)
    arms["zs"] = (text, [], 1, nl - 1, "og", None)
    arms["null"] = ("", [], 1, nl - 1, "og", None)
    if li is not None:
        arms["tli_li"] = (text, [], 1, nl - 1, "og", dict(t1=li[0], t2=li[1], lam=li[2]))

    def transplant(prompt, sched, cand):
        """Coordinate selection on demo 0: object source, then goal source given it, then the layer span."""
        chosen, trace = {}, {}
        for w in "og":
            best = None
            for t in cand[w]:
                e = edit(prompt, w, t)
                if e is None:
                    continue
                es = [edit(prompt, x, chosen[x]) for x in chosen] + [e]
                l = L(s0, prompt, es, sched=sched)
                trace[f"{w}:{t}"] = l
                if best is None or l < best[0]:
                    best = (l, t)
            if best is not None:
                chosen[w] = best[1]
        es = [edit(prompt, w, t) for w, t in chosen.items()]
        span, sl = SP[0], {}
        if es:
            for lo, hi in SP:
                sl[f"{lo}-{hi}"] = L(s0, prompt, es, lo, hi, sched)
            span = tuple(int(x) for x in min(sl, key=sl.get).split("-"))
        return dict(src=chosen, span=list(span), trace=trace, span_loss=sl), es
    # (a) word
    sa, ea = transplant(text, "og", cands)
    arms["word"] = (text, ea, sa["span"][0], sa["span"][1], "og", None)
    res["sel"]["word"] = sa
    # (d) role
    sd, ed = transplant(text, "ph", cands)
    arms["role"] = (text, ed, sd["span"][0], sd["span"][1], "ph", None)
    res["sel"]["role"] = sd
    # (b) template-matched
    tmpls = sorted({LT.template_of(tx[t]) for t in range(40)} - {None})
    tl = {}
    for tm in tmpls:
        p2 = LT.fill(tm, r["o"], r["g"])
        es = [e for e in (edit(p2, w, t) for w, t in sa["src"].items()) if e is not None]
        tl[tm] = dict(prompt=p2, with_word=L(s0, p2, es, sa["span"][0], sa["span"][1]), alone=L(s0, p2))
    tm = min(tl, key=lambda x: tl[x]["with_word"])
    p2 = tl[tm]["prompt"]
    c2 = {w: [t for t in cands[w] if edit(p2, w, t) is not None] for w in "og"}
    sb, eb = transplant(p2, "og", c2)
    sb.update(template=tm, prompt=p2, templates=tl)
    arms["tmpl"] = (p2, eb, sb["span"][0], sb["span"][1], "og", None)
    arms["tmpl_only"] = (p2, [], 1, nl - 1, "og", None)
    res["sel"]["tmpl"] = sb
    # (c) TLI chosen on the demo: pairs at lambda 24, then lambda
    pool = [t for t in range(40) if LT.template_of(tx[t]) is not None]
    tc = {w: [t for t in pool if LT.contains(tx[t], r[w])] or [t for t in pool if tx[t].startswith("put the")] for w in "og"}
    if not any(t1 != t2 for t1 in tc["o"] for t2 in tc["g"]):         # one sentence holds both phrases: add the "put the" pool
        tc["g"] = sorted(set(tc["g"]) | {t for t in pool if tx[t].startswith("put the")})
    lat = {}
    pairs, best = {}, None
    for t1 in tc["o"]:
        for t2 in tc["g"]:
            if t1 == t2:
                continue
            for t in (t1, t2):
                if t not in lat:
                    lat[t] = _tli_lat(bk, cfg, demos, t)
            l = L(s0, text, tli=dict(d=lat[t1] - lat[t2], lam=24))
            pairs[f"{t1}.{t2}"] = l
            if best is None or l < best[0]:
                best = (l, t1, t2)
    lams = {24: best[0]}
    for lam in LAMS:
        if lam not in lams:
            lams[lam] = L(s0, text, tli=dict(d=lat[best[1]] - lat[best[2]], lam=lam))
    lam = min(lams, key=lams.get)
    res["sel"]["tli_sel"] = dict(pair=[best[1], best[2]], lam=int(lam), pairs=pairs, lams={str(k): v for k, v in lams.items()},
                                 cands=tc)
    arms["tli_sel"] = (text, [], 1, nl - 1, "og", dict(t1=best[1], t2=best[2], lam=int(lam)))
    # demo-0 losses of every arm so far; l2sel = the operator with the lowest
    for name, a in arms.items():
        res["l0"][name] = _score(bk, cfg, demos, s0, a, L, lat)
    l2 = min(OPS, key=lambda o: res["l0"][o])
    res["l2sel"] = l2
    # controls on l2sel's form
    rng = np.random.default_rng(int(hashlib.sha1(f"pl2rand:{key}".encode()).hexdigest()[:8], 16))
    tcon = train_contacts(demos)
    p_, e_, lo_, hi_, sch_, tli_ = arms[l2]
    roles_used = [e["role"] for e in e_] if tli_ is None else ["o", "g"]
    rnd, pos, info = {}, {}, {}
    for w in roles_used:
        nov = [t for t in pool if not LT.overlaps(tx[t], r[w])]
        rnd[w] = int(rng.choice(nov))
        tgt = np.asarray(d0["state"][max(d0["c0"], 0) if w == "o" else min(max(d0["rel"], 0), d0["T"] - 1), :3])
        pc = [t for t in nov if t in tcon]
        pos[w] = int(min(pc, key=lambda t: np.linalg.norm(np.asarray(tcon[t]["grasp" if w == "o" else "release"]) - tgt)))
        info[w] = dict(rand=rnd[w], pos=pos[w], rand_text=tx[rnd[w]], pos_text=tx[pos[w]])
    for name, pick in (("rand", rnd), ("pos", pos)):
        if tli_ is not None:
            t1, t2 = pick["o"], pick["g"]
            if t1 == t2:
                t2 = [t for t in pool if t != t1 and not LT.overlaps(tx[t], r["g"])][0]
            for t in (t1, t2):
                if t not in lat:
                    lat[t] = _tli_lat(bk, cfg, demos, t)
            arms[name] = (p_, [], 1, nl - 1, "og", dict(t1=int(t1), t2=int(t2), lam=tli_["lam"]))
        else:
            es = [edit(p_, w, pick[w], src_phrase=LT.roles(tx[pick[w]])[w]) for w in roles_used]
            arms[name] = (p_, [e for e in es if e is not None], lo_, hi_, sch_, None)
        res["l0"][name] = _score(bk, cfg, demos, s0, arms[name], L, lat)
    res["controls"] = info
    # held-out demo 1
    for name, a in arms.items():
        res["l1"][name] = _score(bk, cfg, demos, s1, a, L, lat)
    res["l1"]["l2sel"] = res["l1"][l2]
    res["l0"]["l2sel"] = res["l0"][l2]
    res["n_eval"] = n_eval[0]
    res["arms"] = {n: dict(prompt=a[0], edits=[{k: v for k, v in e.items() if k != "v"} for e in a[1]], lo=a[2], hi=a[3], sched=a[4],
                           tli=a[5]) for n, a in arms.items()}
    # interpretation: the action mode at demo-1 frames before the grasp and while carrying (only where object poses exist)
    if d1.get("obj") is not None:
        res["modes"] = modes(bk, demos, d1, {n: arms[n] for n in ("zs", "word", "tmpl", "tli_sel", "role", "tli_li", "rand", "pos")
                                             if n in arms}, lat, named)
    return res, arms, lat


def _score(bk, cfg, demos, sc, a, L, lat):
    p, es, lo, hi, sch, tm = a
    tli = None if tm is None else dict(d=lat[tm["t1"]] - lat[tm["t2"]], lam=tm["lam"]) if tm["t1"] in lat and tm["t2"] in lat else None
    if tm is not None and tli is None:
        for t in (tm["t1"], tm["t2"]):
            lat[t] = _tli_lat(bk, cfg, demos, t)
        tli = dict(d=lat[tm["t1"]] - lat[tm["t2"]], lam=tm["lam"])
    return L(sc, p, es, lo, hi, sch, tli)


def modes(bk, demos, d, arms, lat, named, n=8):
    """At demo frames (first, pre-grasp middle, carrying middle) sample n chunks per arm; the integrated endpoint's nearest object
    (and the object the motion points at) and the gripper's first change (lifelong_ood/protos/store.py interpret)."""
    import torch
    import lifelong_ood.lang.transplant as LT
    from lifelong_ood.protos import store as S
    T, c0, rel = d["T"], d["c0"], d["rel"]
    pts = {"start": 0, "approach": max(0, c0 // 2) if c0 > 0 else T // 4,
           "carry": (c0 + rel) // 2 if c0 > 0 and rel > c0 else T // 2}
    scale = S.displacement_scale([d["state"]], [d["actions"]])
    names = d["obj"]["names"]
    out = {}
    for pn, f in pts.items():
        f = int(min(f, T - 1))
        out[pn] = dict(frame=f, carry=bool(carry_of(d["actions"])[0][f]))
        for name, (p, es, lo, hi, sch, tm) in arms.items():
            tli = None if tm is None else dict(d=lat[tm["t1"]] - lat[tm["t2"]], lam=tm["lam"])
            comp = LT.Composer(bk, es, lo=lo, hi=hi, schedule=sch, tli=tli) if (es or tli is not None) else None
            # the frame's raw observation: re-read it from the demo arrays (DemoScorer subsampled; rebuild for this frame)
            ob = d["_get"](f)
            if comp is not None:
                comp.attach()
                comp.carry = torch.tensor([out[pn]["carry"]])
                comp.q = np.array([f // 5])
            try:
                with torch.no_grad():
                    o = bk.observation(dict(ob, prompt=p))
                    pre = bk.prefix(o)
                    rows = bk.sample(n, pre, seed=7 + f)
                    raw = bk.unnormalize(rows, state=pre.state[0].float().cpu().numpy())
            finally:
                if comp is not None:
                    comp.detach()
            its = [S.interpret(x, d["state"][f, :3], d["obj"]["pos"][f], names, scale) for x in raw]
            near = [names[i["nearest"]] for i in its]
            point = [names[i["pointed"]] for i in its]
            out[pn][name] = dict(endpoint=np.mean([i["endpoint"] for i in its], 0).tolist(),
                                 nearest=max(set(near), key=near.count), pointed=max(set(point), key=point.count),
                                 named_near=None if named is None else float(np.mean([x == named for x in near])),
                                 named_point=None if named is None else float(np.mean([x == named for x in point])),
                                 grip_first=float(np.mean([i["grip_first_change"] for i in its])),
                                 grip_closed=float(np.mean([i["grip_closed_frac"] for i in its])))
    return out


def offline(bb, which, only=None, shard=(0, 1)):
    import torch  # noqa: F401
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy.backbone import PolicyBackbone
    import lifelong_ood.lang.transplant as LT
    import g12_tli as T
    bk = PolicyBackbone(bb, device="cuda")
    demos = D.Demos()
    os.makedirs(os.path.join(OUT, "offline"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "specs"), exist_ok=True)
    jobs = []
    if which == "ood":
        idx = json.load(open(os.path.join(OUT, "demos.json")))[bb]
        import h1_factor as HF
        named = HF.named_objects()
        for k, v in plan()["tasks"].items():
            if idx.get(k) is None:
                print(f"[pl2] {short(bb)} {k}: no demos; skipped", flush=True)
                continue
            a, b, lam = T.MAP[v["language"].replace(" ", "_")]
            jobs.append((k, v["language"], idx[k]["d0"], idx[k]["d1"], (T.KEY[a], T.KEY[b], lam), named.get((v["suite"], v["index"])), v))
    else:
        d90 = D.Demos90()
        for t, s in l90_tasks():
            e = d90.episodes_of_task(t)
            jobs.append((f"l90:{t}", s, (d90, e[0]), (d90, e[1]), None, None, None))
    for j, (k, text, s0, s1, li, named, v) in enumerate(jobs):
        if (only and k not in only) or j % shard[1] != shard[0]:
            continue
        f = os.path.join(OUT, "offline", f"{short(bb)}_{sh(k)}.json")
        if os.path.exists(f):
            continue
        t0 = time.time()
        d0, d1 = load_demo(s0, bk.H), load_demo(s1, bk.H)
        res, arms, lat = select_task(bk, bb, demos, k, text, d0, d1, li=li, named=named)
        res["seconds"] = time.time() - t0
        res["demo_src"] = [s0 if isinstance(s0, str) else int(s0[1]), s1 if isinstance(s1, str) else int(s1[1])]
        if v is not None:                                                # serve specs for the closed loop (lt@<name>)
            for name, (p, es, lo, hi, sch, tm) in arms.items():
                if tm is None and name not in ("zs", "null", "tmpl_only"):
                    LT.save_spec(spec_path(bb, v["suite"], v["index"], name), es, p, lo=lo, hi=hi, schedule=sch,
                                 meta=dict(task=k, arm=name))
        tmp = f + f".{os.getpid()}.tmp"
        json.dump(res, open(tmp, "w", encoding="utf-8"), indent=1, default=float)
        os.replace(tmp, f)
        l1 = res["l1"]
        print(f"[pl2] {short(bb)} {k} in {res['seconds']:.0f}s, {res['n_eval']} evals; l2sel={res['l2sel']}; demo-1 loss "
              + " ".join(f"{n} {x:.4f}" for n, x in l1.items()), flush=True)
        torch_empty()
    print("[pl2] offline done", flush=True)


def torch_empty():
    import torch
    torch.cuda.empty_cache()


def spec_path(bb, suite, tid, name):
    return os.path.join(OUT, "specs", f"{bb}_{suite}_{tid}_{name}.json")


# ── offline read ────────────────────────────────────────────────────────────────────────────────────────────────
def offline_read(record):
    from lifelong_ood import stats
    rows, out = {}, {}
    for f in sorted(glob.glob(os.path.join(OUT, "offline", "*.json"))):
        d = json.load(open(f))
        b, kk = os.path.basename(f)[:-5].split("_", 1)
        s = "l90" if kk.startswith("l90") else "ood"
        rows.setdefault((b, s), []).append(d)
    names = ("zs", "null", "tli_li", "word", "tmpl", "tmpl_only", "tli_sel", "role", "l2sel", "rand", "pos")
    for (b, s), ds in sorted(rows.items()):
        tab = {}
        for n in names:
            v = [(d["l1"][n], d["l1"]["zs"]) for d in ds if n in d["l1"]]
            if not v:
                continue
            x = np.array(v)
            wins = int((x[:, 0] < x[:, 1]).sum())
            loses = int((x[:, 0] > x[:, 1]).sum())
            tab[n] = dict(n=len(v), mean=float(x[:, 0].mean()), ratio_to_zs=float(np.exp(np.mean(np.log(x[:, 0] / x[:, 1])))),
                          wins_vs_zs=wins, losses_vs_zs=loses, sign_p=float(stats.sign_test(wins, loses)))
        cmp = {}
        for a_, b_ in (("l2sel", "tli_li"), ("l2sel", "rand"), ("l2sel", "pos"), ("word", "tmpl"), ("word", "role"), ("tmpl", "tmpl_only")):
            v = np.array([(d["l1"][a_], d["l1"][b_]) for d in ds if a_ in d["l1"] and b_ in d["l1"]])
            if len(v):
                w, l = int((v[:, 0] < v[:, 1]).sum()), int((v[:, 0] > v[:, 1]).sum())
                cmp[f"{a_}_vs_{b_}"] = dict(n=len(v), wins=w, losses=l, sign_p=float(stats.sign_test(w, l)),
                                            ratio=float(np.exp(np.mean(np.log(v[:, 0] / v[:, 1])))))
        chosen = {n: sum(d["l2sel"] == n for d in ds) for n in OPS}
        per = {d["key"]: dict(chosen=d["l2sel"], **{n: round(d["l1"][n], 5) for n in names if n in d["l1"]}) for d in ds}
        spans = {o: {} for o in ("word", "tmpl", "role")}
        for d in ds:
            for o in spans:
                k_ = "-".join(map(str, d["sel"][o]["span"]))
                spans[o][k_] = spans[o].get(k_, 0) + 1
        out[f"{b}_{s}"] = dict(tasks=len(ds), table=tab, compare=cmp, chosen=chosen, spans=spans, per_task=per)
        out[f"{b}_{s}"]["modes"] = mode_summary(ds)
        out[f"{b}_{s}"]["repairs"] = repairs(ds)
        print(f"\n[pl2 offline] {b} {s}: {len(ds)} tasks; l2sel chose {chosen}; spans {spans}")
        for n, t in tab.items():
            print(f"  {n:10} demo-1 loss {t['mean']:.4f}  x zs {t['ratio_to_zs']:.3f}  wins/losses vs zs {t['wins_vs_zs']}/{t['losses_vs_zs']}"
                  f"  sign p {t['sign_p']:.3g}")
        for n, c in cmp.items():
            print(f"  {n:20} {c['wins']}/{c['losses']} of {c['n']} (ratio {c['ratio']:.3f}, sign p {c['sign_p']:.3g})")
    json.dump(out, open(os.path.join(OUT, "offline_read.json"), "w", encoding="utf-8"), indent=1)
    fig_offline(out)
    if record:
        from lifelong_ood import ledger
        print(ledger.record("pl2_offline", bench="libero_ood+libero_90", tag="read", seed=SEED, metrics=out,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(K=K_DRAWS, frames=F_TARGET, lams=LAMS, demo_dir=DEMO_DIR, ndemo_bind=3),
                            script="experiments/pl2_compose.py", command=" ".join(sys.argv)))


def mode_summary(ds):
    """Mean over tasks of the action-mode readouts at demo-1 frames: P(endpoint nearest the named object), P(motion points at it),
    the gripper's first change (chunk step), per arm and frame kind; the chosen operator as 'l2sel'."""
    acc = {}
    for d in ds:
        if "modes" not in d:
            continue
        for pt, m in d["modes"].items():
            for arm, x in m.items():
                if not isinstance(x, dict):
                    continue
                for nm in ([arm] + (["l2sel"] if arm == d["l2sel"] else [])):
                    for q in ("named_near", "named_point", "grip_first", "grip_closed"):
                        if x.get(q) is not None:
                            acc.setdefault(pt, {}).setdefault(nm, {}).setdefault(q, []).append(x[q])
    return {pt: {a: {q: float(np.mean(v)) for q, v in m.items()} | {"n": len(next(iter(m.values())))} for a, m in arms.items()}
            for pt, arms in acc.items()}


def repairs(ds):
    """Per task: the chosen operator, its sources (sentences), layer span / TLI pair and lambda, and its mode change against zs."""
    from lifelong_ood.policy import data as D
    tx = D.Demos().tasks
    out = {}
    for d in ds:
        a = d["arms"][d["l2sel"]]
        r = dict(op=d["l2sel"], prompt=a["prompt"], l1=d["l1"]["l2sel"], l1_zs=d["l1"]["zs"])
        if a["tli"] is not None:
            r.update(t1=tx[a["tli"]["t1"]], t2=tx[a["tli"]["t2"]], lam=a["tli"]["lam"])
        else:
            r.update(sources={e["role"]: tx[e["src"]] for e in a["edits"]}, span=f"{a['lo']}-{a['hi']}", sched=a["sched"])
        if "modes" in d:
            r["mode"] = {pt: {k: {q: m[k].get(q) for q in ("nearest", "pointed", "grip_first")} for k in ("zs", d["l2sel"]) if k in m}
                         for pt, m in d["modes"].items()}
        out[d["key"]] = r
    return out


def fig_offline(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    names = ("null", "tli_li", "tmpl_only", "word", "tmpl", "tli_sel", "role", "l2sel", "rand", "pos")
    keys = sorted(out)
    fig, ax = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.8), squeeze=False)
    for a, k in zip(ax[0], keys):
        t = out[k]["table"]
        ns = [n for n in names if n in t]
        y = [t[n]["ratio_to_zs"] for n in ns]
        col = ["#9aa5b1" if n in ("null", "rand", "pos", "tmpl_only") else "#c28e2b" if n == "tli_li" else "#2f6db5" for n in ns]
        a.barh(range(len(ns)), y, color=col)
        a.axvline(1.0, color="k", lw=0.8)
        a.set_yticks(range(len(ns)))
        a.set_yticklabels([f"{n} ({t[n]['wins_vs_zs']}/{t[n]['n']})" for n in ns], fontsize=8)
        a.invert_yaxis()
        a.set_xscale("log")
        a.set_title(f"{k}: {out[k]['tasks']} tasks", fontsize=9)
        a.set_xlabel("held-out demo-1 flow loss / zero-shot (geo. mean)", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "pl2_offline.png"), dpi=140)
    print("figure", os.path.join(FIG, "pl2_offline.png"))


# ── closed loop ─────────────────────────────────────────────────────────────────────────────────────────────────
def dev_arms(bb, k, v):
    """-> {arm name: launch arm string} for one dev task from its offline selection file."""
    import g12_tli as T
    d = json.load(open(os.path.join(OUT, "offline", f"{short(bb)}_{sh(k)}.json")))
    out = {"zs": "zs", "tli_li": T.arm(v)}
    for n in ("word", "tmpl", "tli_sel", "role", "rand", "pos"):
        a = d["arms"][n]
        out[n] = f"tli@{a['tli']['t1']}.{a['tli']['t2']}.{a['tli']['lam']}" if a["tli"] is not None else f"lt@{n}"
    out["l2sel"] = out[d["l2sel"]]
    return out, d


def launch_dev(dry, bbs=BBS):
    pl = plan()
    specs = []
    for bb in bbs:
        for k in split()["dev"]:
            v = pl["tasks"][k]
            if not os.path.exists(os.path.join(OUT, "offline", f"{short(bb)}_{sh(k)}.json")):
                print("no offline selection yet:", bb, k)
                continue
            arms, _ = dev_arms(bb, k, v)
            for arm in dict.fromkeys(arms[n] for n in ARMS_DEV):
                specs.append(_spec(v["index"], arm, v["suite"], bb, f"pl2_dev_{k.split(':')[0]}", 20, TEST_DIR))
    submit(specs, dry=dry)


def _eps(exp, v, arm, bb):
    f = os.path.join(R, f"{exp}_{v['suite']}_{v['index']}_{arm.replace('@', '_').replace('-', '_')}_seed{SEED}{bsfx(bb)}.json")
    if not os.path.exists(f):
        return None
    return {int(e["trial"]): e for t in json.load(open(f))["tasks"].values() for e in t["episodes"]}


def read_dev(record):
    import e1_read as E1
    import h1_factor as HF
    pl = plan()
    named = HF.named_objects()
    res = {}
    for bb in BBS:
        x, parts, missing, sel = {}, {}, [], {}
        for k in split()["dev"]:
            v = pl["tasks"][k]
            if not os.path.exists(os.path.join(OUT, "offline", f"{short(bb)}_{sh(k)}.json")):
                missing.append(k)
                continue
            arms, d = dev_arms(bb, k, v)
            sel[k] = d["l2sel"]
            z = {n: _eps(f"pl2_dev_{k.split(':')[0]}", v, arms[n], bb) for n in ARMS_DEV + ("l2sel",)}
            if None in z.values():
                missing.append(k)
                continue
            x[k] = {n: {t: int(bool(e["success"])) for t, e in ep.items()} for n, ep in z.items()}
            nm = named.get((v["suite"], v["index"]))
            parts[k] = {}
            for n, ep in z.items():
                g = [e.get("bound") == nm for e in ep.values()]
                pg = [bool(e["success"]) for e in ep.values() if e.get("bound") == nm]
                parts[k][n] = dict(s=int(sum(bool(e["success"]) for e in ep.values())), g=int(sum(g)), n=len(g), p_s=int(sum(pg)),
                                   p_n=len(pg))
        C = E1.compare
        r = dict(backbone=bb, tasks=len(x), missing=missing, l2sel=sel)
        if x:
            r["H1"] = C({k: {"x": z["l2sel"], "y": z["zs"]} for k, z in x.items()})
            r["H1"]["passes"] = bool(r["H1"]["x"] > r["H1"]["y"] and r["H1"]["sign_p"] < 0.05)
            r["H2"] = C({k: {"x": z["l2sel"], "y": z["tli_li"]} for k, z in x.items()})
            r["H2"]["noninferior"] = bool(r["H2"]["diff"]["lo"] > -0.05)
            for n in ("rand", "pos", "tli_li"):
                r[f"l2sel_vs_{n}"] = C({k: {"x": z["l2sel"], "y": z[n]} for k, z in x.items()})
            for n in ARMS_DEV:
                r[f"{n}_vs_zs"] = C({k: {"x": z[n], "y": z["zs"]} for k, z in x.items()})
            tot = {n: {q: sum(parts[k][n][q] for k in parts) for q in ("s", "g", "n", "p_s", "p_n")} for n in ARMS_DEV + ("l2sel",)}
            r["parts"] = {n: dict(success=t["s"], n=t["n"], g=t["g"] / max(t["n"], 1), p=t["p_s"] / max(t["p_n"], 1),
                                  named_grasps=t["g"]) for n, t in tot.items()}
            r["per_task"] = parts
        res[short(bb)] = r
        if x:
            print(f"\n[pl2 dev] {short(bb)}: {len(x)} tasks, missing {missing}; l2sel {sel}")
            for n in ("H1", "H2", "l2sel_vs_rand", "l2sel_vs_pos"):
                h = r[n]
                print(f"  {n:14} {h['x']} vs {h['y']} of {h['n']}, discordant {h['discordant']}, sign p {h['sign_p']:.3g}, "
                      f"diff {h['diff']['diff']:+.3f} [{h['diff']['lo']:+.3f}, {h['diff']['hi']:+.3f}]")
            for n, p in r["parts"].items():
                print(f"  {n:8} success {p['success']}/{p['n']}  g {p['g']:.2f}  p|g {p['p']:.2f}")
    json.dump(res, open(os.path.join(OUT, "dev_read.json"), "w", encoding="utf-8"), indent=1)
    fig_dev(res)
    if record:
        from lifelong_ood import ledger
        print(ledger.record("pl2_dev", bench="libero_ood", tag="read", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(states=TEST_DIR, episodes=20, split=split()), script="experiments/pl2_compose.py",
                            command=" ".join(sys.argv)))


def fig_dev(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = [k for k in res if "parts" in res[k]]
    if not ks:
        return
    fig, ax = plt.subplots(1, len(ks), figsize=(5.2 * len(ks), 3.6), squeeze=False)
    for a, k in zip(ax[0], ks):
        P = res[k]["parts"]
        ns = [n for n in ("zs", "tli_li", "l2sel", "word", "tmpl", "tli_sel", "role", "rand", "pos") if n in P]
        xs = np.arange(len(ns))
        a.bar(xs - 0.27, [P[n]["success"] / P[n]["n"] for n in ns], 0.27, label="success", color="#2f6db5")
        a.bar(xs, [P[n]["g"] for n in ns], 0.27, label="g: named first grasp", color="#7fb2e5")
        a.bar(xs + 0.27, [P[n]["p"] for n in ns], 0.27, label="p: place | named grasp", color="#c28e2b")
        a.set_xticks(xs)
        a.set_xticklabels(ns, rotation=40, fontsize=8)
        a.set_ylim(0, 1)
        a.set_title(f"{k}: {res[k]['tasks']} dev tasks x 20 episodes", fontsize=9)
    ax[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "pl2_dev.png"), dpi=140)
    print("figure", os.path.join(FIG, "pl2_dev.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in ("split", "demos-launch", "demos-read", "offline", "offline-read", "launch-dev", "read-dev", "dry", "record"):
        ap.add_argument("--" + f, action="store_true")
    ap.add_argument("--backbone", default="pi05_libero")
    ap.add_argument("--set", default="ood", choices=("ood", "l90"))
    ap.add_argument("--only", default="")
    ap.add_argument("--shard", default="0/1", help="i/n: this job's share of the tasks")
    a = ap.parse_args()
    if a.split:
        print(json.dumps(split(), indent=1))
    if a.demos_launch:
        demos_launch(a.dry)
    if a.demos_read:
        demos_read()
    if a.offline:
        offline(a.backbone, a.set, [x for x in a.only.split("+") if x], tuple(int(x) for x in a.shard.split("/")))
    if a.offline_read:
        offline_read(a.record)
    if a.launch_dev:
        launch_dev(a.dry)
    if a.read_dev:
        read_dev(a.record)
