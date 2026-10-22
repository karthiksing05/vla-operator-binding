"""Plan L, probe L0b: CAUSAL TRACING of word binding on pi0.5 (ROME-style; docs/PLAN_LANG.md; lifelong_ood/lang/audit.py trace_case).

Cases: 20 sentences (10 training instructions over the four LIBERO suites' templates, 10 of Li's OOD instructions) x 3 word kinds
(object, goal, verb), each at 2 first-frame observations of its own scene (stored init states 0-1 / practice set a states 0-1).
The word's input embeddings get Gaussian noise of 3x the embedding std (3 noise draws); the clean state is restored at one
(layer input, position group) of the VLM's residual stream, and separately at one (layer, position group) -- or a group at every
layer -- of the prefix K/V cache the action expert reads.  Recovery of the clean field at shared points (4 flow times x 4 points).
-> results/lang/trace/<backbone>/<case>_s<state>.npz.  Resumable, sharded.

    sbatch ... --export=ALL,ENV=openpi,SCRIPT="experiments/pl0_trace.py --run --shard 0 --nshards 2" slurm/jobs/robot_gpu.sbatch
    $PY experiments/pl0_trace.py --read [--record]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
OUT = os.path.join(ROOT, "results", "lang")
FIG = os.path.join(ROOT, "figures", "robot")
TRAIN = (10, 11, 13, 14, 17, 21, 25, 30, 34, 38)          # put / pick-up templates, goal, object and spatial suites
OOD = ("goal:0", "goal:1", "goal:2", "goal:3", "goal:6", "spatial:0", "spatial:2", "spatial:4", "spatial:6", "spatial:8")
KINDS = ("obj", "goal", "verb")
CATS = ("pre", "before", "w_first", "w_mid", "w_last", "after1", "after", "state", "suffix", "img")   # pi05_libero's prompt carries
# no discretised state (its config's discrete_state_input is off), so "state" / "suffix" stay empty there


def log(s):
    print(s, flush=True)


def cases():
    import pl0_audit as P
    from lifelong_ood.lang import audit as A
    train, ood, tasks, _ = P.sentences()
    out = []
    for k in TRAIN:
        out.append((f"tr{k:02d}", train[k], tasks[train[k]], "f0b"))
    for key in OOD:
        t = tasks[ood[key]]
        out.append((f"ood_{t.suite}_{t.index}", ood[key], t, "frames"))
    res = []
    for name, text, t, src in out:
        ws = {w.kind: w for w in A.parse(text)}
        for kd in KINDS:
            if kd in ws:
                for s in (0, 1):
                    res.append(dict(case=f"{name}_{kd}", key=f"{name}_{kd}_s{s}", text=text, kind=kd, word=ws[kd].phrase,
                                    lo=ws[kd].lo, hi=ws[kd].hi, src=src, file=f"{t.suite}_{t.index}", state=s,
                                    set="train" if src == "f0b" else "ood"))
    return res


def _od(c):
    d = os.path.join(ROOT, "results", "robot", "f0b_firstframes") if c["src"] == "f0b" else os.path.join(OUT, "frames")
    z = np.load(os.path.join(d, c["file"] + ".npz"), allow_pickle=True)
    s = c["state"]
    return {"observation/image": z["img"][s], "observation/wrist_image": z["wimg"][s],
            "observation/state": np.asarray(z["state8"][s], np.float32)}


def run(a):
    from lifelong_ood.lang import audit as A
    from lifelong_ood.policy.backbone import PolicyBackbone
    C = cases()
    mine = C[a.shard::a.nshards][: a.limit or None]
    od_dir = os.path.join(OUT, "trace", a.backbone)
    os.makedirs(od_dir, exist_ok=True)
    todo = [c for c in mine if not os.path.exists(os.path.join(od_dir, c["key"] + ".npz"))]
    log(f"[pl0_trace] {len(C)} traces; shard {a.shard}/{a.nshards}: {len(todo)} to do")
    if not todo:
        return
    bk = PolicyBackbone(a.backbone, device="cuda")
    log(f"[pl0_trace] {bk.describe()}")
    sp = os.path.join(od_dir, "embed_std.json")
    if os.path.exists(sp):
        std = json.load(open(sp))["std"]
    else:                                                  # the same pairs in every shard: deterministic
        pairs = [(_od(c), c["text"]) for c in C if c["state"] == 0 and c["kind"] == "obj"]
        std = A.embed_std(bk, pairs)
        tmp = sp + f".{os.getpid()}.tmp"
        json.dump(dict(std=std, n_pairs=len(pairs)), open(tmp, "w"))
        os.replace(tmp, sp)
    log(f"[pl0_trace] embedding std {std:.4f}, noise 3x")
    t_all = time.time()
    for n, c in enumerate(todo):
        t0 = time.time()
        r = A.trace_case(bk, _od(c), c["text"], c["lo"], c["hi"], std, n_noise=a.noise, batch=a.batch)
        out = os.path.join(od_dir, c["key"] + ".npz")
        tmp = out + f".{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, meta=json.dumps(c), std=std, **r)
        os.replace(tmp, out)
        g = list(r["groups"])
        wl = f"t{int(r['word'][-1])}"
        log(f"[pl0_trace] {c['key']} '{c['word']}' in '{c['text']}': effect {r['effect']:.3f}; hid@{wl} by layer "
            f"{np.round(r['hid'][:, g.index(wl)], 2).tolist()}; kv-all-layers@{wl} {r['kv'][-1, g.index(wl)]:.2f}; "
            f"{time.time() - t0:.0f}s ({n + 1}/{len(todo)}, {time.time() - t_all:.0f}s)")
    log(f"[pl0_trace] shard {a.shard}/{a.nshards} finished")


def categorise(groups, word, sent):
    """group name -> CATS category relative to the word (ROME's first / middle / last subject token, first subsequent, further)."""
    word, sent = [int(x) for x in word], [int(x) for x in sent]
    out = []
    for g in groups:
        if not g.startswith("t"):
            out.append(g)
            continue
        p = int(g[1:])
        if p in word:
            out.append("w_last" if p == word[-1] else "w_first" if p == word[0] else "w_mid")
        elif p < word[0]:
            out.append("before")
        elif p == word[-1] + 1:
            out.append("after1")
        else:
            out.append("after")
    return out


def read(record=False):
    bb = "pi05_libero"
    d = os.path.join(OUT, "trace", bb)
    files = sorted(x for x in os.listdir(d) if x.endswith(".npz") and ".tmp" not in x) if os.path.isdir(d) else []
    if not files:
        log("[pl0_trace read] no traces")
        return
    maps = {}
    eff = {}
    for f in files:
        z = np.load(os.path.join(d, f), allow_pickle=True)
        c = json.loads(str(z["meta"]))
        cats = categorise([str(x) for x in z["groups"]], z["word"], z["sent"])
        for which in ("hid", "kv"):
            M = z[which]                                          # [L(+1), G]
            agg = np.full((M.shape[0], len(CATS)), np.nan)
            for j, cat in enumerate(CATS):
                cols = [i for i, x in enumerate(cats) if x == cat]
                if cols:
                    agg[:, j] = M[:, cols].mean(1)
            maps.setdefault((which, c["kind"]), []).append(agg)
            maps.setdefault((which, c["kind"], c["set"]), []).append(agg)
        eff.setdefault(c["kind"], []).append(float(z["effect"]))
    res = dict(n=len(files), cats=list(CATS), effect={k: dict(mean=float(np.mean(v)), median=float(np.median(v)), n=len(v))
                                                      for k, v in eff.items()}, maps={}, summary={})
    for key, v in maps.items():
        M = np.nanmean(np.stack(v), 0)
        res["maps"]["|".join(key)] = np.round(M, 4).tolist()
    for kd in KINDS:
        if ("hid", kd) not in maps:
            continue
        H = np.nanmean(np.stack(maps[("hid", kd)]), 0)
        K = np.nanmean(np.stack(maps[("kv", kd)]), 0)
        wl = CATS.index("w_last")
        L = H.shape[0]
        kv_single = K[:-1]                                      # single layers
        flat = [(float(kv_single[l, j]), l, CATS[j]) for l in range(kv_single.shape[0]) for j in range(len(CATS))
                if np.isfinite(kv_single[l, j])]
        hflat = sorted([(float(H[l, j]), l, CATS[j]) for l in range(2, L) for j in range(len(CATS)) if np.isfinite(H[l, j])],
                       reverse=True)                            # residual restores past layer 1 (where the word has left)
        flat.sort(reverse=True)
        # where the word's own positions stop carrying it: last layer whose hidden restore at the word's last token recovers >= 0.5
        carry = [l for l in range(L) if H[l, wl] >= 0.5]
        word_cols = [CATS.index(x) for x in ("w_first", "w_mid", "w_last")]
        kv_word = np.nanmax(kv_single[:, word_cols], 1)
        pos_mass = np.clip(kv_word, 0, None)
        top4 = float(np.sort(pos_mass)[-4:].sum() / max(pos_mass.sum(), 1e-9))
        res["summary"][kd] = dict(
            hid_word_last_by_layer=np.round(H[:, wl], 3).tolist(), hid_carry_layers=carry,
            hid_suffix_by_layer=np.round(H[:, CATS.index("suffix")], 3).tolist(),
            kv_all_layers=dict(zip(CATS, np.round(K[-1], 3).tolist())),
            kv_top_cells=[dict(rec=round(a, 3), layer=l, cat=c) for a, l, c in flat[:6]],
            kv_word_by_layer=np.round(kv_word, 3).tolist(), kv_word_top4_share=round(top4, 3),
            hid_top_cells_after_l1=[dict(rec=round(a, 3), layer=l, cat=c) for a, l, c in hflat[:6]],
            hid_img_by_layer=np.round(H[:, CATS.index("img")], 3).tolist(),
            kv_img_by_layer=np.round(K[:-1, CATS.index("img")], 3).tolist())
        log(f"[pl0_trace read] {kd}: effect {res['effect'][kd]['mean']:.3f}; hidden restore at word-last >= 0.5 in layers {carry}; "
            f"kv all layers {res['summary'][kd]['kv_all_layers']}; top kv cells {res['summary'][kd]['kv_top_cells'][:3]}; "
            f"top-4-layer share of word-position kv recovery {top4:.2f}")
    with open(os.path.join(OUT, "pl0_trace.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    figure(res)
    if record:
        from lifelong_ood import ledger
        print(ledger.record("pl0_trace", bench="libero_ood", tag="read_pi05", seed=0,
                            metrics=dict(n=res["n"], effect=res["effect"], summary=res["summary"], cats=res["cats"],
                                         maps={k: v for k, v in res["maps"].items() if k.count("|") == 1}),
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(backbone=bb, noise="3x embedding std, 3 draws", train=list(TRAIN), ood=list(OOD)),
                            script="experiments/pl0_trace.py", command=" ".join(sys.argv),
                            artifacts=dict(payload="results/lang/pl0_trace.json")))


def figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    kinds = [k for k in KINDS if f"hid|{k}" in res["maps"]]
    fig, axs = plt.subplots(2, len(kinds), figsize=(4.2 * len(kinds), 7.4), squeeze=False)
    for j, kd in enumerate(kinds):
        for i, which in enumerate(("hid", "kv")):
            M = np.asarray(res["maps"][f"{which}|{kd}"], float)
            ax = axs[i, j]
            im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="lower")
            ax.set_xticks(range(len(CATS)))
            ax.set_xticklabels(CATS, rotation=60, fontsize=7)
            ax.set_ylabel("layer" + (" (top row: every layer)" if which == "kv" else " input"), fontsize=8)
            ax.set_title(f"{kd}: restore {'residual' if which == 'hid' else 'prefix K/V'}", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle("pi0.5 causal tracing: recovery of the clean action field", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "pl_trace_map.png"), dpi=150)
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
    ap.add_argument("--noise", type=int, default=3)
    ap.add_argument("--batch", type=int, default=24)
    a = ap.parse_args()
    if a.run:
        run(a)
    if a.read:
        read(a.record)
