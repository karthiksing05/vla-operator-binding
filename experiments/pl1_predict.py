"""Plan L, probe L1 (pre-registered: ledger m3_prereg tag pl1_predict, experiments/prereg/pl1_predict.py): does the lexicon audit's
word grounding in the test scene predict zero-shot failure on Li's 20 OOD tasks?

Outcome: the policy alone (arm zs) pooled over start-state sets c, d, e, f (experiments g4b_main / g4c_main, g13_main, g15_main,
g17_base; 80 episodes per task); failure = success rate < 0.50.  Predictor: min (primary) and mean (secondary) over the sentence's
object / goal / verb / relation words of G(w) from experiments/pl0_audit.py (results/lang/pl0_audit.json), each policy's own audit.
GO: AUC of -min G for failure over the 40 pooled (policy, task) units >= 0.75.  Numpy only.

    $PY experiments/pl1_predict.py [--record]
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
R = os.path.join(ROOT, "results", "robot")
OUT = os.path.join(ROOT, "results", "lang")
FIG = os.path.join(ROOT, "figures", "robot")
SETS = {"pi05_libero": ("g4b_main", "g13_main", "g15_main", "g17_base"), "pi0_libero": ("g4c_main", "g13_main", "g15_main", "g17_base")}


def zero_shot(bb, pl):
    """-> {plan key: (successes, episodes)} pooled over the four sets."""
    sfx = "" if bb == "pi05_libero" else "_" + bb.split("_")[0]
    out = {}
    for k, v in pl["tasks"].items():
        s = n = 0
        for exp in SETS[bb]:
            f = os.path.join(R, f"{exp}_{k.split(':')[0]}_{v['suite']}_{v['index']}_zs_seed13{sfx}.json")
            eps = [e for t in json.load(open(f))["tasks"].values() for e in t["episodes"]]
            n += len(eps)
            s += sum(bool(e["success"]) for e in eps)
        out[k] = (s, n)
    return out


def spearman(x, y):
    from scipy import stats
    r = stats.spearmanr(x, y)
    return float(r.correlation), float(r.pvalue)


def boot_auc(score, label, task, n=2000, seed=0):
    """Bootstrap over tasks (both policies' units of a task move together)."""
    from lifelong_ood.lang import audit as A
    g = np.random.default_rng(seed)
    ut = sorted(set(task))
    idx = {t: [i for i, x in enumerate(task) if x == t] for t in ut}
    v = []
    for _ in range(n):
        pick = [i for t in g.choice(ut, len(ut)) for i in idx[t]]
        a = A.auc(np.asarray(score)[pick], np.asarray(label)[pick])
        if np.isfinite(a):
            v.append(a)
    return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


def main(record=False):
    from lifelong_ood.lang import audit as A
    pl = json.load(open(os.path.join(R, "g4_plan.json")))
    au = json.load(open(os.path.join(OUT, "pl0_audit.json")))
    units = []
    for bb in ("pi05_libero", "pi0_libero"):
        if bb not in au:
            print(f"[pl1] no audit for {bb}", flush=True)
            continue
        zs = zero_shot(bb, pl)
        for k, v in pl["tasks"].items():
            g = au[bb]["sentences"].get(v["language"])
            if g is None:
                print(f"[pl1] {bb} {k}: no grounding (audit incomplete)", flush=True)
                continue
            s, n = zs[k]
            units.append(dict(policy=bb, task=k, language=v["language"], success=s, episodes=n, rate=s / n, fail=bool(s / n < 0.5),
                              gmin=g["min"], gmean=g["mean"], argmin=g["argmin"], words=g["words"], dir_obj=g.get("dir_obj")))
    res = dict(units=units)

    def block(us, name):
        if not us:
            return None
        y = [u["fail"] for u in us]
        out = dict(n=len(us), n_fail=int(sum(y)),
                   auc_min=A.auc([-u["gmin"] for u in us], y), auc_mean=A.auc([-u["gmean"] for u in us], y),
                   spearman_min=spearman([u["gmin"] for u in us], [u["rate"] for u in us]),
                   spearman_mean=spearman([u["gmean"] for u in us], [u["rate"] for u in us]))
        if 0 < sum(y) < len(y):
            out["auc_min_ci"] = boot_auc([-u["gmin"] for u in us], y, [u["task"] for u in us])
        print(f"[pl1] {name}: n {out['n']} ({out['n_fail']} failures); AUC(-min G) {out['auc_min']:.3f} CI {np.round(out.get('auc_min_ci', (np.nan, np.nan)), 3).tolist()}; "
              f"AUC(-mean G) {out['auc_mean']:.3f}; Spearman min {out['spearman_min'][0]:+.3f} (p {out['spearman_min'][1]:.3g}), "
              f"mean {out['spearman_mean'][0]:+.3f} (p {out['spearman_mean'][1]:.3g})", flush=True)
        return out
    res["pooled"] = block(units, "pooled")
    for bb in ("pi05_libero", "pi0_libero"):
        res[bb] = block([u for u in units if u["policy"] == bb], bb)
    res["go"] = bool(res["pooled"] and res["pooled"]["auc_min"] >= 0.75)
    print(f"[pl1] pre-registered GO (pooled AUC >= 0.75): {res['go']}", flush=True)
    for u in units:
        print(f"   {u['policy']:12} {u['task']:10} {u['rate']:.2f} min G {u['gmin']:.3f} ({u['argmin']}) mean {u['gmean']:.3f}  "
              f"{u['language']}", flush=True)
    with open(os.path.join(OUT, "pl1_predict.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    figure(units)
    if record:
        from lifelong_ood import ledger
        print(ledger.record("pl1_predict", bench="libero_ood", tag="read", seed=13,
                            metrics={k: v for k, v in res.items()},
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(sets=SETS, prereg="ledger/m3_prereg tag pl1_predict"),
                            script="experiments/pl1_predict.py", command=" ".join(sys.argv),
                            artifacts=dict(payload="results/lang/pl1_predict.json")))


def figure(units):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, key, lab in ((axs[0], "gmin", "min word grounding G"), (axs[1], "gmean", "mean word grounding G")):
        for bb, c in (("pi05_libero", "#4C78A8"), ("pi0_libero", "#E45756")):
            us = [u for u in units if u["policy"] == bb]
            ax.scatter([u[key] for u in us], [u["rate"] for u in us], c=c, s=18, label=bb)
        ax.axhline(0.5, color="k", lw=0.5, ls="--")
        ax.set_xlabel(lab + " (OOD scene)")
        ax.set_ylabel("zero-shot success (80 episodes)")
        ax.legend(fontsize=7)
    fig.suptitle("L1: grounding vs zero-shot success on Li's 20 OOD tasks", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "pl_l1_predict.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main("--record" in sys.argv)
