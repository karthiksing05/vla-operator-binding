"""H1 (B): what the two operators for held concepts do to the policy (exploratory, offline; docs/PLAN_H.md section H1.2).

Question.  Text-latent interpolation (TLI; lifelong_ood/policy/tli.py) adds (T1 - T2) to the conditioning encoder's residual
stream at the instruction's first 9 tokens of every layer >= 1 at the start of an episode (alpha = 0), so the test instruction is
read as the object source task.  Which layers and which tokens carry that effect, how large is the shift against the residual
it is added to, and does that ratio explain why TLI works on pi0, partly on X-VLA and not at all on SmolVLA?  And the grasp
switch: what changes in the policy's field when the instruction switches from the object source to the goal source?

Method.  For each of Li's 20 compositional OOD tasks and Li's source pair (la, lg; experiments/g12_tli.py MAP), the 26th
demonstration of the object source task la in pi0.5's training set (the latents were identified on the first 20), 4 frames
evenly spaced before its first sustained grasp; prompted with the OOD instruction; 2 fixed noise draws per frame.  Arms:
  none     the instruction alone;  src  la's own instruction (the behaviour TLI should recall);
  all      Li's shift at every layer >= 1 and all 9 tokens;
  L<l>     the shift at layer l only;  band_early / band_mid / band_late  at one third of the layers;
  tok<j>   the shift at instruction token j only, every layer.
Effect of an arm = mean |chunk(arm) - chunk(none)| in the policy's normalised action units, as a share of the `all` arm's;
transfer = 1 - |chunk(arm) - chunk(src)| / |chunk(none) - chunk(src)| (1: the arm reproduces la's chunk; 0: no closer than the
instruction alone); shift/residual = |T1 - T2| / |h| at the instruction tokens, per layer (h from the `none` pass).
The grasp switch, offline: on la's approach frames and on 3 frames of lg's demonstration between its grasp and its release, the
chunk under the OOD instruction, la's and lg's instructions, each against the demonstration's own chunk; and the size and
make-up (position / orientation / gripper) of the change between la's and lg's fields at the frames just after lg's grasp.
Run once per backbone; --read joins the four with each backbone's per-task TLI success (ledger g13_tli_sources, g18_four_policies).

    sbatch --exclude=hal,tars,t1000,vicki,johnny5,c3po,olivaw --partition=overcap --account=overcap --qos=short \
        --export=ALL,ENV=openpi,SCRIPT="experiments/h1_tli.py --run --backbone pi0_libero" slurm/jobs/robot_gpu.sbatch
        (pi05_libero likewise; ENV=smolvla for --backbone smolvla and --backbone xvla)
    ~/flash/miniconda3/envs/chunks/bin/python experiments/h1_tli.py --read [--record]
"""
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
R = os.path.join(ROOT, "results", "robot")
OUT = os.path.join(R, "h1_tli")
FIG = os.path.join(ROOT, "figures", "robot")
BACKBONES = ("pi0_libero", "pi05_libero", "smolvla", "xvla")
SHORT = dict(pi0_libero="pi0", pi05_libero="pi05", smolvla="smolvla", xvla="xvla")
DEMO, NAPP, NTR, K = 25, 4, 3, 2


def groups(a):
    """Action-dimension groups of one chunk step: (position, orientation, gripper)."""
    return (slice(0, 3), slice(3, 6), slice(6, 7)) if a == 7 else (slice(0, 3), slice(3, 9), slice(9, 10))


def run(backbone):
    import torch
    import c0_fit as C0
    import g12_tli as T
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy import primitives as P
    from lifelong_ood.policy import tli as TL
    os.makedirs(OUT, exist_ok=True)
    bk, norm = C0.backbone(backbone)
    L = len(bk.cond_layers())
    demos = D.Demos()
    pl = json.load(open(os.path.join(R, "g4_plan.json")))
    print(f"[h1t] {bk.describe()}; {L} conditioning layers", flush=True)
    third = max(1, (L - 1) // 3)
    cfgs = dict(none=None, all=(list(range(1, L)), list(range(TL.NTOK))))
    for l in range(1, L):
        cfgs[f"L{l}"] = ([l], list(range(TL.NTOK)))
    cfgs["band_early"] = (list(range(1, 1 + third)), list(range(TL.NTOK)))
    cfgs["band_mid"] = (list(range(1 + third, 1 + 2 * third)), list(range(TL.NTOK)))
    cfgs["band_late"] = (list(range(1 + 2 * third, L)), list(range(TL.NTOK)))
    for j in range(TL.NTOK):
        cfgs[f"tok{j}"] = (list(range(1, L)), [j])
    cur = dict(cfg=None, d=None, sign=1.0, hn=None)

    def hook(i, h, span):
        if span is None:
            return None
        off, nv = span
        k = min(TL.NTOK, h.shape[1] - off)
        if k <= 0:
            return None
        m = TL._mask(nv, k, h.device)
        if cur["hn"] is not None:                                             # record |h| at the instruction tokens
            x = h[:, off:off + k].float().norm(dim=-1)                        # [B, k]
            cur["hn"].setdefault(i, []).append(((x * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy())
        c = cur["cfg"]
        if c is None or i == 0 or i not in c[0]:
            return None
        tok = torch.zeros(k, device=h.device)
        tok[[j for j in c[1] if j < k]] = 1.0
        h = h.clone()
        h[:, off:off + k] += (m * tok[None])[..., None].to(h.dtype) * (cur["sign"] * cur["d"][i][:k]).to(h.dtype)[None]
        return h
    handles = bk.cond_hook(hook)

    def frames_of(task, phase):
        e = demos.episodes(task)[DEMO]
        ep = demos.load(e, images=True)
        act = ep["actions"]
        tr = P.transport(act, 0, len(act))
        g = tr[0] if tr else len(act) // 2
        rel = [r for r in P.releases(act[:, 6], P.RELEASE_RUN) if r > g]
        rel = rel[0] if rel else len(act) - 1
        if phase == "approach":
            fr = np.linspace(0, g - 1, NAPP).round().astype(int).tolist()
        else:
            fr = np.linspace(g + 2, max(g + 3, rel - 2), NTR).round().astype(int).tolist()
        idx = D.chunk_indices(len(act), bk.H)[0]
        try:
            demo = C0.chunk_rows(ep, idx[fr], fr, norm)
        except Exception as ex:                                               # noqa: BLE001
            print(f"[h1t] demo rows unavailable ({ex})", flush=True)
            demo = None
        return ep, fr, demo, int(g), int(rel)

    @torch.no_grad()
    def chunk(ep, fr, prompt, cfg=None, d=None, sign=1.0, hn=None):
        obs = []
        for i in fr:
            o = demos.observation(ep, i)
            o["prompt"] = prompt
            obs.append(o)
        cur.update(cfg=cfg, d=d, sign=sign, hn=hn)
        try:
            pre = bk.prefix(bk.observations(obs))
        finally:
            cur.update(cfg=None, d=None, hn=None)
        n = len(fr)
        preK = bk.repeat(pre, K)
        noise = torch.randn((n * K, bk.dim_full), generator=torch.Generator(device="cpu").manual_seed(0))
        rows = bk.sample(n * K, preK, noise=noise)
        return np.asarray(rows, np.float32).reshape(n, K, -1)

    texts = demos.tasks
    t0 = time.time()
    out = {}
    for k_, v in pl["tasks"].items():
        a, b, lam = T.MAP[v["language"].replace(" ", "_")]
        la, lg = T.KEY[a], T.KEY[b]
        T1, T2 = (np.load(TL.path(backbone, t, ROOT)) for t in (la, lg))
        d = torch.as_tensor(T1 - T2, device=bk.device)
        d[0] = 0.0
        ep, fr, demo, g, rel = frames_of(la, "approach")
        hn = {}
        res = dict(none=chunk(ep, fr, v["language"], hn=hn))
        hnorm = np.array([np.concatenate(hn[i]).mean() for i in sorted(hn)])
        dnorm = np.array([float(np.linalg.norm(T1[i] - T2[i], axis=-1)[np.linalg.norm(T1[i], axis=-1) > 0].mean()) for i in
                          range(len(T1))])
        res["src"] = chunk(ep, fr, texts[la])
        res["goalsrc_on_approach"] = chunk(ep, fr, texts[lg])
        for name, c in cfgs.items():
            if c is not None:
                res[name] = chunk(ep, fr, v["language"], cfg=c, d=d)
        ep2, fr2, demo2, g2, rel2 = frames_of(lg, "transport")
        tr = dict(test=chunk(ep2, fr2, v["language"]), goalsrc=chunk(ep2, fr2, texts[lg]), objsrc=chunk(ep2, fr2, texts[la]),
                  tli_end=chunk(ep2, fr2, v["language"], cfg=cfgs["all"], d=d, sign=-1.0))
        out[k_] = dict(la=la, lg=lg, lam=lam, language=v["language"])
        np.savez_compressed(os.path.join(OUT, f"{backbone}_{k_.replace(':', '_')}.npz"),
                            **{f"app_{n}": x for n, x in res.items()}, **{f"tr_{n}": x for n, x in tr.items()},
                            demo_app=demo if demo is not None else np.zeros(0), demo_tr=demo2 if demo2 is not None else np.zeros(0),
                            hnorm=hnorm, dnorm=dnorm, a=np.array(bk.a), H=np.array(bk.H), L=np.array(L))
        print(f"[h1t] {backbone} {k_} ({la}->{lg}): |shift|/|h| by layer "
              f"{np.round(dnorm[1:len(hnorm)] / np.maximum(hnorm[1:], 1e-9), 3).tolist()} ({time.time() - t0:.0f}s)", flush=True)
    for h in handles:
        h.remove()
    json.dump(dict(backbone=backbone, tasks=out, layers=L, cfgs=list(cfgs)), open(os.path.join(OUT, f"{backbone}_meta.json"), "w"),
              indent=1)


def _d(x, y):
    """mean over frames and draws of |x - y| (rows [n, K, D])."""
    return float(np.linalg.norm(x - y, axis=-1).mean())


def per_backbone(backbone):
    meta_f = os.path.join(OUT, f"{backbone}_meta.json")
    if not os.path.exists(meta_f):
        return None
    meta = json.load(open(meta_f))
    L = meta["layers"]
    rows = {}
    for k_ in meta["tasks"]:
        z = np.load(os.path.join(OUT, f"{backbone}_{k_.replace(':', '_')}.npz"), allow_pickle=True)
        a, H = int(z["a"]), int(z["H"])
        none, src, al = z["app_none"], z["app_src"], z["app_all"]
        base_gap = _d(none, src)
        e_all = _d(al, none)
        r = dict(gap_none_src=base_gap, effect_all=e_all, rel_effect_all=e_all / max(base_gap, 1e-9),
                 transfer_all=1 - _d(al, src) / max(base_gap, 1e-9),
                 ratio_by_layer=(z["dnorm"][1:len(z["hnorm"])] / np.maximum(z["hnorm"][1:], 1e-9)).tolist(),
                 ratio=float(np.mean(z["dnorm"][1:len(z["hnorm"])] / np.maximum(z["hnorm"][1:], 1e-9))))
        r["layer_effect"] = [(_d(z[f"app_L{l}"], none) / max(e_all, 1e-9)) for l in range(1, L)]
        r["layer_transfer"] = [1 - _d(z[f"app_L{l}"], src) / max(base_gap, 1e-9) for l in range(1, L)]
        r["band_effect"] = {b: _d(z[f"app_band_{b}"], none) / max(e_all, 1e-9) for b in ("early", "mid", "late")}
        r["band_transfer"] = {b: 1 - _d(z[f"app_band_{b}"], src) / max(base_gap, 1e-9) for b in ("early", "mid", "late")}
        r["token_effect"] = [_d(z[f"app_tok{j}"], none) / max(e_all, 1e-9) for j in range(9)]
        r["token_transfer"] = [1 - _d(z[f"app_tok{j}"], src) / max(base_gap, 1e-9) for j in range(9)]
        # the grasp switch, offline
        dt = z["demo_tr"]
        da = z["demo_app"]
        if da.size and da.shape[-1] == none.shape[-1]:
            da = da[:, None]
            r["approach_err"] = dict(test=_d(none, da), objsrc=_d(src, da), goalsrc=_d(z["app_goalsrc_on_approach"], da),
                                     tli=_d(al, da))
        if dt.size and dt.shape[-1] == z["tr_test"].shape[-1]:
            dt = dt[:, None]
            r["transport_err"] = {n: _d(z[f"tr_{n}"], dt) for n in ("test", "goalsrc", "objsrc", "tli_end")}
        sw = (z["tr_goalsrc"] - z["tr_objsrc"]).reshape(*z["tr_goalsrc"].shape[:2], H, a)
        e = (sw ** 2).sum((-1, -2))
        gp, gr, gg = groups(a)
        r["switch_change"] = dict(norm=_d(z["tr_goalsrc"], z["tr_objsrc"]), norm_on_approach=_d(z["app_goalsrc_on_approach"], src),
                                  share_pos=float(((sw[..., gp] ** 2).sum((-1, -2)) / np.maximum(e, 1e-12)).mean()),
                                  share_rot=float(((sw[..., gr] ** 2).sum((-1, -2)) / np.maximum(e, 1e-12)).mean()),
                                  share_grip=float(((sw[..., gg] ** 2).sum((-1, -2)) / np.maximum(e, 1e-12)).mean()))
        rows[k_] = r
    agg = dict(layers=L, tasks=len(rows))
    for m in ("gap_none_src", "effect_all", "rel_effect_all", "transfer_all", "ratio"):
        agg[m] = float(np.mean([r[m] for r in rows.values()]))
    for m in ("ratio_by_layer", "layer_effect", "layer_transfer", "token_effect", "token_transfer"):
        agg[m] = np.mean([r[m] for r in rows.values()], 0).round(4).tolist()
    for m in ("band_effect", "band_transfer"):
        agg[m] = {b: float(np.mean([r[m][b] for r in rows.values()])) for b in ("early", "mid", "late")}
    for m in ("approach_err", "transport_err", "switch_change"):
        rs = [r[m] for r in rows.values() if m in r]
        if rs:
            agg[m] = {k: float(np.mean([x[k] for x in rs])) for k in rs[0]}
    return dict(agg=agg, per_task=rows)


def success():
    """Per backbone: Li's TLI and the policy alone per task (ledger g13_tli_sources: pi0, pi0.5 on set d; g18_four_policies:
    SmolVLA, X-VLA on set e)."""
    from lifelong_ood import ledger
    out = {}
    for b in ("pi0", "pi05"):
        rr = [r for r in ledger.load("g13_tli_sources", tag=f"read_{b}")]
        r = max(rr, key=lambda x: x["started"])["metrics"]
        out[b] = {k: dict(tli=v["tli_li"], zs=v["zs"]) for k, v in r["per_task"].items()}
    for b in ("smolvla", "xvla"):
        rr = [r for r in ledger.load("g18_four_policies", tag=f"read_{b}")]
        r = max(rr, key=lambda x: x["started"])["metrics"]
        out[b] = {k: dict(tli=v["tli_li"], zs=v["zs"]) for k, v in r["per_task"].items()}
    return out


def read(record):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as S
    succ = success()
    res, missing = {}, []
    for b in BACKBONES:
        x = per_backbone(b)
        if x is None:
            missing.append(b)
            continue
        s = succ[SHORT[b]]
        ks = [k for k in x["per_task"] if k in s]
        x["agg"]["tli_successes"] = int(sum(s[k]["tli"] for k in ks))
        x["agg"]["zs_successes"] = int(sum(s[k]["zs"] for k in ks))
        x["agg"]["episodes"] = 20 * len(ks)
        tl = np.array([s[k]["tli"] for k in ks], float)
        for m in ("transfer_all", "rel_effect_all", "ratio"):
            v = np.array([x["per_task"][k][m] for k in ks])
            rho = S.spearmanr(v, tl) if np.std(v) > 0 and np.std(tl) > 0 else None
            x["agg"][f"spearman_{m}_vs_tli_success"] = (None if rho is None else dict(rho=float(rho.correlation), p=float(rho.pvalue)))
        res[SHORT[b]] = x
    os.makedirs(FIG, exist_ok=True)
    COL = dict(pi0="#1d5e86", pi05="#2f7d4f", smolvla="#a35d2b", xvla="#8b5cf6")
    if res:
        fig, ax = plt.subplots(2, 2, figsize=(12.5, 8))
        for b, x in res.items():
            a = x["agg"]
            Ls = np.arange(1, a["layers"]) / (a["layers"] - 1)
            ax[0, 0].plot(Ls, a["layer_transfer"], "o-", ms=3, color=COL[b], label=f"{b} ({a['layers']} layers)")
            ax[0, 1].plot(Ls, a["ratio_by_layer"], "o-", ms=3, color=COL[b], label=b)
            ax[1, 0].plot(range(9), a["token_transfer"], "o-", ms=3, color=COL[b], label=b)
        ax[0, 0].set_xlabel("layer (fraction of the conditioning encoder's depth)")
        ax[0, 0].set_ylabel("transfer toward the object source's chunk\n(shift at this layer only)")
        ax[0, 0].axhline(0, color="#9ca3af", lw=1)
        ax[0, 0].set_title("which layers carry TLI's effect", fontsize=10)
        ax[0, 1].set_xlabel("layer (fraction of depth)")
        ax[0, 1].set_ylabel("|T1 - T2| / |h| at the instruction tokens")
        ax[0, 1].set_yscale("log")
        ax[0, 1].set_title("the shift against the residual it is added to", fontsize=10)
        ax[1, 0].set_xlabel("instruction token position (shift at this token only, every layer)")
        ax[1, 0].set_ylabel("transfer toward the object source's chunk")
        ax[1, 0].axhline(0, color="#9ca3af", lw=1)
        ax[1, 0].set_title("which tokens carry it", fontsize=10)
        bs = list(res)
        xs = [res[b]["agg"]["transfer_all"] for b in bs]
        ys = [res[b]["agg"]["tli_successes"] / max(1, res[b]["agg"]["episodes"]) for b in bs]
        zs = [res[b]["agg"]["zs_successes"] / max(1, res[b]["agg"]["episodes"]) for b in bs]
        for b, xx, yy, z0 in zip(bs, xs, ys, zs):
            ax[1, 1].scatter(xx, yy, color=COL[b], s=60)
            ax[1, 1].annotate(f"{b} (policy alone {z0:.2f}, shift/residual {res[b]['agg']['ratio']:.3f})", (xx, yy), fontsize=7,
                              xytext=(4, 4), textcoords="offset points")
        ax[1, 1].set_xlabel("offline transfer of the full shift (approach frames)")
        ax[1, 1].set_ylabel("TLI success rate online (Li's pairs)")
        ax[1, 1].set_title("offline transfer against online success, per backbone", fontsize=10)
        for a_ in ax.ravel():
            a_.spines[["top", "right"]].set_visible(False)
        ax[0, 0].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "h1_tli_layers.png"), dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
        w = 0.2
        for j, b in enumerate(bs):
            a = res[b]["agg"]
            if "approach_err" in a and "transport_err" in a:
                ax[0].bar(np.arange(3) + (j - 1.5) * w, [a["approach_err"]["test"], a["approach_err"]["objsrc"],
                                                          a["approach_err"]["goalsrc"]], w, color=COL[b], label=b)
                ax[1].bar(np.arange(3) + (j - 1.5) * w, [a["transport_err"]["test"], a["transport_err"]["goalsrc"],
                                                          a["transport_err"]["objsrc"]], w, color=COL[b], label=b)
        ax[0].set_xticks(range(3))
        ax[0].set_xticklabels(["OOD instruction", "object source's", "goal source's"])
        ax[0].set_title("approach frames of the object source's demo", fontsize=9)
        ax[1].set_xticks(range(3))
        ax[1].set_xticklabels(["OOD instruction", "goal source's", "object source's"])
        ax[1].set_title("carry frames of the goal source's demo", fontsize=9)
        for a_ in ax:
            a_.set_ylabel("|chunk - demonstration's chunk| (normalised)")
            a_.spines[["top", "right"]].set_visible(False)
        ax[0].legend(fontsize=7, frameon=False)
        fig.suptitle("the grasp switch selects the conditional field that reproduces each phase's trained motion", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "h1_switch.png"), dpi=150)
        plt.close(fig)
    json.dump(res, open(os.path.join(R, "h1_tli.json"), "w"), indent=1)
    for b, x in res.items():
        a = x["agg"]
        print(f"[h1t] {b}: TLI {a['tli_successes']}/{a['episodes']} (zs {a['zs_successes']}); shift/residual {a['ratio']:.4f}; "
              f"gap none->src {a['gap_none_src']:.3f}; effect all {a['effect_all']:.3f} (rel {a['rel_effect_all']:.2f}); transfer "
              f"{a['transfer_all']:.3f}; bands {a['band_transfer']}; best layer "
              f"{1 + int(np.argmax(a['layer_transfer']))} ({max(a['layer_transfer']):.3f}); tokens {np.round(a['token_transfer'], 2).tolist()}"
              f"; approach err {a.get('approach_err')}; transport err {a.get('transport_err')}; switch {a.get('switch_change')}",
              flush=True)
    print(f"[h1t] missing {missing}", flush=True)
    if record and res:
        from lifelong_ood import ledger
        z0 = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
        print(ledger.record("h1_tli", bench="libero_ood", tag="read", seed=0, metrics=dict(missing=missing, **res), bytes=z0,
                            config=dict(demo=DEMO, napp=NAPP, ntr=NTR, K=K), script="experiments/h1_tli.py",
                            command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--run" in sys.argv:
        run(sys.argv[sys.argv.index("--backbone") + 1])
    if "--read" in sys.argv:
        read("--record" in sys.argv)
