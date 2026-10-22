"""Plan L, probe L0a: the LEXICON AUDIT (docs/PLAN_LANG.md; library lifelong_ood/lang/audit.py).

Sentences: pi0.5's 40 training instructions and Li's 20 OOD instructions.  Observations of each sentence's own scene:
    training  10 first frames (stored init states 0-9, results/robot/f0b_firstframes) + the 5 demonstration frames of the P2 / P3
              observation index (results/protos/index.json; poses in results/protos/poses; post-grasp frames among them)
    OOD       10 first frames of practice start-state set a (experiments/pl0_render.py -> results/lang/frames)
At each observation, every word swap's field (audit.field_audit): ~34 prompts (null, instruction, verb / object / relation / goal /
preposition swaps, the article control, 3 whole-instruction swaps) -> results/lang/audit/<backbone>/<key>.npz.  Resumable, sharded.

    sbatch ... --export=ALL,ENV=openpi,SCRIPT="experiments/pl0_audit.py --run --backbone pi05_libero --shard 0 --nshards 2" \
        slurm/jobs/robot_gpu.sbatch
    $PY experiments/pl0_audit.py --read [--record]          (numpy: tables, figures/robot/pl_*.png, results/lang/pl0_audit.json)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
OUT = os.path.join(ROOT, "results", "lang")
F0B = os.path.join(ROOT, "results", "robot", "f0b_firstframes")
PROTOS = os.path.join(ROOT, "results", "protos")
FIG = os.path.join(ROOT, "figures", "robot")
NAMES = {"cookies": "cookie box", "glazed_rim_porcelain_ramekin": "ramekin", "flat_stove": "stove", "wooden_cabinet": "cabinet"}   # as
# the instructions say them (the type names would be read literally)


def log(s):
    print(s, flush=True)


def save_atomic(path, **arrs):
    tmp = path + f".{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)


def write_json(path, obj):
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


# ── sentences, scenes, observations (numpy) ──────────────────────────────────────────────────────────────────────
def sentences():
    """-> (train_texts [40], ood {plan key: language}, tasks {sentence: factors.Task}, all tasks)."""
    from lifelong_ood.policy import factors as F
    idx = json.load(open(os.path.join(PROTOS, "index.json")))
    train = [idx["train_texts"][str(k)] for k in range(40)]
    pl = json.load(open(os.path.join(ROOT, "results", "robot", "g4_plan.json")))
    ood = {k: v["language"] for k, v in pl["tasks"].items()}
    allt = F.load()
    oodt = F.load(F.OOD_SUITES)
    by_lang = {t.language: t for t in allt if t.suite != "libero_90"}
    tasks = {s: by_lang[s] for s in train}
    for k, v in pl["tasks"].items():
        tasks[v["language"]] = next(t for t in oodt if t.suite == v["suite"] and t.index == v["index"])
    return train, ood, tasks, allt + oodt


def scene_words(task, allt, voc):
    """The scene's object phrases (moved types first excluded: the sentence's own) and the goal phrases whose noun is in the scene."""
    from lifelong_ood.policy import factors as F
    ph = dict(F.scene_phrases(task, allt), **NAMES)
    own = set(task.moved)
    objs = [F.phrase_of(ty, ph) for ty in sorted(set(task.objects.values())) if ty not in own]
    nouns = {F.phrase_of(ty, ph).split()[-1] for ty in set(task.objects.values()) | set(task.fixtures.values())}
    goals = [g for g in voc["goal"] if g.split()[-1] in nouns]
    return objs, goals


def observations(train, ood, tasks):
    """-> list of dict(key, text, tid, source, ...) for every audited observation."""
    idx = json.load(open(os.path.join(PROTOS, "index.json")))
    obs = []
    for k, text in enumerate(train):
        t = tasks[text]
        for s in range(10):
            obs.append(dict(key=f"tr{k:02d}_s{s}", text=text, tid=t.tid, set="train", src="f0b", file=f"{t.suite}_{t.index}", state=s,
                            phase=0))
        for o in idx["obs"]:
            if o["set"] == "train" and o["hf_task"] == k:
                obs.append(dict(key=o["key"], text=text, tid=t.tid, set="train", src="demo", episode=o["episode"], frame=o["frame"],
                                phase=o["phase"]))
    for key, text in ood.items():
        t = tasks[text]
        for s in range(10):
            obs.append(dict(key=f"ood_{t.suite}_{t.index}_s{s}", text=text, tid=t.tid, plan=key, set="ood", src="frames",
                            file=f"{t.suite}_{t.index}", state=s, phase=0))
    return obs


def scene_of(o):
    """-> (eef [3], body names, body positions [N, 3]) of an observation."""
    if o["src"] == "demo":
        pp = os.path.join(PROTOS, "poses", o["key"] + ".npz")
        if not os.path.exists(pp):                                  # 5 index frames have no poses: no attribution there
            return np.zeros(3), [], np.zeros((0, 3))
        q = np.load(os.path.join(PROTOS, "poses", o["key"] + ".npz"), allow_pickle=True)
        return np.asarray(q["eef"], np.float64), [str(x) for x in q["names"]], np.asarray(q["pos"], np.float64)
    d = F0B if o["src"] == "f0b" else os.path.join(OUT, "frames")
    z = np.load(os.path.join(d, o["file"] + ".npz"), allow_pickle=True)
    s = o["state"]
    return np.asarray(z["state8"][s][:3], np.float64), [str(x) for x in z["names"]], np.asarray(z["pos"][s], np.float64)


# ── GPU run ──────────────────────────────────────────────────────────────────────────────────────────────────────
def run(a):
    from lifelong_ood.lang import audit as A
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy.backbone import PolicyBackbone
    train, ood, tasks, allt = sentences()
    voc = A.vocabulary(train + list(ood.values()))
    obs = observations(train, ood, tasks)
    mine = obs[a.shard::a.nshards]
    if a.limit:
        mine = mine[: a.limit]
    od_dir = os.path.join(OUT, "audit", a.backbone)
    os.makedirs(od_dir, exist_ok=True)
    todo = [o for o in mine if not os.path.exists(os.path.join(od_dir, o["key"] + ".npz"))]
    log(f"[pl0_audit] {a.backbone}: shard {a.shard}/{a.nshards}: {len(mine)} observations, {len(todo)} to do")
    if not todo:
        return
    bk = PolicyBackbone(a.backbone, device="cuda")
    log(f"[pl0_audit] {bk.describe()}")
    demos = D.Demos()
    cache = {}
    t_all = time.time()
    for n, o in enumerate(todo):
        t0 = time.time()
        if o["src"] == "demo":
            if o["episode"] not in cache:
                cache.clear()
                cache[o["episode"]] = demos.load(o["episode"], images=True)
            od = demos.observation(cache[o["episode"]], o["frame"])
        else:
            d = F0B if o["src"] == "f0b" else os.path.join(OUT, "frames")
            p = os.path.join(d, o["file"] + ".npz")
            if not os.path.exists(p):
                log(f"[pl0_audit] {o['key']}: no frames file {p}; skipped")
                continue
            z = np.load(p, allow_pickle=True)
            s = o["state"]
            od = {"observation/image": z["img"][s], "observation/wrist_image": z["wimg"][s],
                  "observation/state": np.asarray(z["state8"][s], np.float32)}
        od = {k: v for k, v in od.items() if k != "prompt"}
        so, sg = scene_words(tasks[o["text"]], allt, voc)
        conds = A.conditions(o["text"], voc, train, so, sg)
        res = A.field_audit(bk, od, [c["prompt"] for c in conds], block=a.block)
        eef, names, pos = scene_of(o)
        save_atomic(os.path.join(od_dir, o["key"] + ".npz"), E=res["E"].astype(np.float16), raw=res["raw"], rows=res["rows"].astype(np.float16),
                    drafts=res["drafts"], conds=json.dumps(conds), meta=json.dumps(o), eef=eef, names=np.array(names), pos=pos,
                    levels=np.array(A.LEVELS))
        G = A.effects(res["E"])
        log(f"[pl0_audit] {o['key']} '{o['text']}': {len(conds)} prompts, {time.time() - t0:.1f}s; G func "
            f"{np.mean([g for g, c in zip(G, conds) if c['kind'] == 'func']):.3f} obj "
            f"{np.mean([g for g, c in zip(G, conds) if c['kind'] == 'obj']):.3f} whole "
            f"{np.mean([g for g, c in zip(G, conds) if c['kind'] == 'whole']):.3f} ({n + 1}/{len(todo)}, {time.time() - t_all:.0f}s)")
    log(f"[pl0_audit] shard {a.shard}/{a.nshards} finished")


# ── read (numpy) ─────────────────────────────────────────────────────────────────────────────────────────────────
def displacement_scale():
    from lifelong_ood.policy import data as D
    from lifelong_ood.protos import store as S
    p = os.path.join(OUT, "displacement_scale.npy")
    if os.path.exists(p):
        return np.load(p)
    idx = json.load(open(os.path.join(PROTOS, "index.json")))
    demos = D.Demos()
    st, ac = [], []
    for e in sorted({o["episode"] for o in idx["obs"] if o["set"] == "train"}):
        ep = demos.load(e)
        st.append(ep["state"])
        ac.append(ep["actions"])
    sc = S.displacement_scale(st, ac)
    np.save(p, sc)
    return sc


def _instance(arg, names):
    """A goal-atom argument (instance or instance + region) -> the body name it names, or None."""
    if arg in names:
        return arg
    hit = [n for n in names if arg.startswith(n + "_")]
    return max(hit, key=len) if hit else None


def referents(task, allt):
    """-> (own object instance, own goal instance, phrase -> instance for unambiguous scene phrases)."""
    from lifelong_ood.policy import factors as F
    ph = dict(F.scene_phrases(task, allt), **NAMES)
    inst = dict(task.objects, **task.fixtures)
    names = list(inst)
    pick = [a for a in task.goal if a[0] in ("In", "On") and len(a) == 3]
    own_o = pick[0][1] if pick else None
    own_g = _instance(pick[0][2], names) if pick else None
    by_noun = {}
    for n, ty in inst.items():
        p = F.phrase_of(ty, ph)
        for key in (p, p.split()[-1]):
            by_noun.setdefault(key, set()).add(n)
    m = {k: next(iter(v)) for k, v in by_noun.items() if len(v) == 1}
    return own_o, own_g, m


def _goal_noun(g):
    return g.split()[-1]


def read(record=False):
    from lifelong_ood.lang import audit as A
    train, ood, tasks, allt = sentences()
    scale = displacement_scale()
    out = {}
    for bb in ("pi05_libero", "pi0_libero"):
        d = os.path.join(OUT, "audit", bb)
        if not os.path.isdir(d):
            continue
        files = sorted(x for x in os.listdir(d) if x.endswith(".npz") and ".tmp" not in x)
        if not files:
            continue
        per_word = {}                    # (sentence, kind, word) -> list of per-observation G (mean over alternatives)
        per_alt = {}                     # (sentence, kind, word, alt) -> list of G
        dirs = {k: [] for k in ("obj", "goal", "rel")}
        ctrl = {"func": [], "whole": []}
        dir_rows = []
        per_word_f0 = {}
        gaps = {}                        # sentence -> per-observation ||eps(instr) - eps(null)|| (the normaliser, absolute)
        for f in files:
            z = np.load(os.path.join(d, f), allow_pickle=True)
            o = json.loads(str(z["meta"]))
            conds = json.loads(str(z["conds"]))
            G = A.effects(z["E"].astype(np.float32))
            text = o["text"]
            E32 = z["E"].astype(np.float32)
            gaps.setdefault(text, []).append(float(np.linalg.norm(E32[1] - E32[0], axis=-1).mean()))
            grp = {}
            for g, c in zip(G, conds):
                if c["kind"] in ("null", "instr"):
                    continue
                grp.setdefault((c["kind"], c["word"]), []).append(g)
                per_alt.setdefault((text, c["kind"], c["word"], c["alt"]), []).append(float(g))
            for (kd, w), gs in grp.items():
                per_word.setdefault((text, kd, w), []).append(float(np.mean(gs)))
                if o["src"] != "demo":                                 # first frames only (OOD has only these)
                    per_word_f0.setdefault((text, kd, w), []).append(float(np.mean(gs)))
            # binding direction: endpoints of paired samples
            task = tasks[text]
            own_o, own_g, m = referents(task, allt)
            names = [str(x) for x in z["names"]]
            P = {n: z["pos"][i] for i, n in enumerate(names)}
            ends = A.endpoints(z["raw"], z["eef"], scale)             # [C, S, 3]
            e_i = ends[1]
            for c_i, c in enumerate(conds):
                kd = c["kind"]
                if kd == "obj" and o["phase"] == 0 and own_o in P:
                    tgt = m.get(c["alt"]) or m.get(c["alt"].split()[-1])
                    if tgt and tgt in P and tgt != own_o:
                        cos, closer = A.direction(e_i, ends[c_i], P[own_o], P[tgt])
                        dirs["obj"].append((cos, closer))
                        dir_rows.append(dict(kind=kd, set=o["set"], text=text, alt=c["alt"], cos=cos, closer=closer))
                if kd == "goal" and o["phase"] == 1 and own_g in P:
                    tgt = m.get(_goal_noun(c["alt"]))
                    if tgt and tgt in P and tgt != own_g:
                        cos, closer = A.direction(e_i, ends[c_i], P[own_g], P[tgt])
                        dirs["goal"].append((cos, closer))
                        dir_rows.append(dict(kind=kd, set=o["set"], text=text, alt=c["alt"], cos=cos, closer=closer))
        # tables
        rows = []
        for (text, kd, w), gs in per_word.items():
            rows.append(dict(text=text, set="ood" if text in ood.values() else "train", kind=kd, word=w, G=float(np.mean(gs)),
                             G_sd=float(np.std(gs)), n_obs=len(gs)))
        kinds = A.KINDS + A.CONTROLS
        by_kind = {}
        for kd in kinds:                                                # training sentences at first frames only
            v = [float(np.mean(gs)) for (text, k_, w), gs in per_word_f0.items() if k_ == kd and text not in ood.values()]
            if v:
                by_kind[f"train_f0:{kd}"] = dict(mean=float(np.mean(v)), median=float(np.median(v)), q10=float(np.quantile(v, 0.1)),
                                                 q90=float(np.quantile(v, 0.9)), n=len(v))
        for s_ in ("train", "ood"):
            for kd in kinds:
                v = [r["G"] for r in rows if r["set"] == s_ and r["kind"] == kd]
                if v:
                    by_kind[f"{s_}:{kd}"] = dict(mean=float(np.mean(v)), median=float(np.median(v)), q10=float(np.quantile(v, 0.1)),
                                                 q90=float(np.quantile(v, 0.9)), n=len(v))
        dsum = {}
        for kd in ("obj", "goal"):
            for s_ in ("train", "ood"):
                v = [(r["cos"], r["closer"]) for r in dir_rows if r["kind"] == kd and r["set"] == s_]
                if v:
                    c_ = np.asarray(v)
                    dsum[f"{s_}:{kd}"] = dict(cos_mean=float(c_[:, 0].mean()), closer=float(c_[:, 1].mean()), n=len(v),
                                              cos_se=float(c_[:, 0].std() / np.sqrt(len(v))))
        # per-sentence direction (object) for the OOD sentences
        sent_dir = {}
        for r in dir_rows:
            sent_dir.setdefault((r["text"], r["kind"]), []).append(r["cos"])
        # sentence-level grounding for L1: min / mean over the pre-registered kinds
        sent = {}
        for r in rows:
            if r["kind"] in A.PREDICT_KINDS:
                sent.setdefault(r["text"], {})[r["kind"]] = r["G"]
        sents = {t: dict(min=float(min(v.values())), mean=float(np.mean(list(v.values()))), argmin=min(v, key=v.get), words=v,
                         func=next((r["G"] for r in rows if r["text"] == t and r["kind"] == "func"), None),
                         whole=next((r["G"] for r in rows if r["text"] == t and r["kind"] == "whole"), None),
                         dir_obj=float(np.mean(sent_dir[(t, "obj")])) if (t, "obj") in sent_dir else None,
                         gap=float(np.mean(gaps[t])), set="ood" if t in ood.values() else "train")
                 for t, v in sent.items()}
        out[bb] = dict(n_files=len(files), by_kind=by_kind, direction=dsum, words=rows, sentences=sents,
                       alts={"|".join(k): float(np.mean(v)) for k, v in per_alt.items()})
        log(f"[pl0_audit read] {bb}: {len(files)} observations")
        for k, v in by_kind.items():
            log(f"   G {k:14} mean {v['mean']:.3f} median {v['median']:.3f} [q10 {v['q10']:.3f}, q90 {v['q90']:.3f}] n {v['n']}")
        for s_ in ("train", "ood"):
            gv = [v["gap"] for v in sents.values() if v["set"] == s_]
            if gv:
                log(f"   instr-null gap {s_:5} mean {np.mean(gv):.3f} median {np.median(gv):.3f} n {len(gv)}")
        for k, v in dsum.items():
            log(f"   direction {k:10} cos {v['cos_mean']:+.3f} (se {v['cos_se']:.3f}) closer {v['closer']:.2f} n {v['n']}")
    write_json(os.path.join(OUT, "pl0_audit.json"), out)
    figures(out)
    if record:
        from lifelong_ood import ledger
        for bb, r in out.items():
            print(ledger.record("pl0_audit", bench="libero_ood", tag=f"read_{bb.split('_')[0]}", seed=0,
                                metrics=dict(n_files=r["n_files"], by_kind=r["by_kind"], direction=r["direction"],
                                             sentences=r["sentences"]),
                                bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                                config=dict(backbone=bb, levels=list(A.LEVELS), caps=A.CAP),
                                script="experiments/pl0_audit.py", command=" ".join(sys.argv),
                                artifacts=dict(payload="results/lang/pl0_audit.json")))


def figures(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    kinds = ["func", "verb", "prep", "rel", "goal", "obj", "whole"]
    bbs = [b for b in ("pi05_libero", "pi0_libero") if b in out]
    if not bbs:
        return
    fig, axs = plt.subplots(1, len(bbs), figsize=(5.2 * len(bbs), 3.6), squeeze=False)
    for ax, bb in zip(axs[0], bbs):
        rows = out[bb]["words"]
        for j, s_ in enumerate(("train", "ood")):
            data = [[r["G"] for r in rows if r["set"] == s_ and r["kind"] == k] or [np.nan] for k in kinds]
            pos = np.arange(len(kinds)) + (j - 0.5) * 0.36
            bp = ax.boxplot(data, positions=pos, widths=0.3, patch_artist=True, showfliers=False)
            for b in bp["boxes"]:
                b.set(facecolor=("#4C78A8" if s_ == "train" else "#E45756"), alpha=0.75)
        ax.set_xticks(range(len(kinds)))
        ax.set_xticklabels(kinds)
        ax.set_ylabel("word effect G (fraction of instr - null gap)")
        ax.set_title(f"{bb}: lexicon audit (blue train, red OOD)", fontsize=9)
        ax.axhline(0, color="k", lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "pl_audit_kinds.png"), dpi=150)
    plt.close(fig)
    # word x scene grounding on the OOD sentences (each sentence is its own scene)
    ks = ["verb", "obj", "rel", "goal", "prep", "func", "whole"]
    fig, axs = plt.subplots(1, len(bbs), figsize=(5.6 * len(bbs), 6.0), squeeze=False)
    for ax, bb in zip(axs[0], bbs):
        rows = [r for r in out[bb]["words"] if r["set"] == "ood"]
        texts = sorted({r["text"] for r in rows})
        M = np.full((len(texts), len(ks)), np.nan)
        for r in rows:
            M[texts.index(r["text"]), ks.index(r["kind"])] = r["G"]
        im = ax.imshow(np.minimum(M, 2.0), aspect="auto", cmap="magma", vmin=0, vmax=2)
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels(ks)
        ax.set_yticks(range(len(texts)))
        ax.set_yticklabels(texts, fontsize=6)
        ax.set_title(f"{bb}: grounding G per word x OOD scene", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "pl_audit_ood_heat.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--read", action="store_true")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--backbone", default="pi05_libero")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--block", type=int, default=16)
    a = ap.parse_args()
    if a.run:
        run(a)
    if a.read:
        read(a.record)
