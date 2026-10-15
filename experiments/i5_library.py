"""INTERPRETATION 5: the full library as the learner holds it (exploratory, offline).  The hierarchical factored library over the 67
single-step tasks (E3's prototypes and E5's new ones): every object, instance and goal prototype, compared exactly in rank space
(experiments/i4_factors.py).  Reports
  the weight-space hierarchy -- average-linkage clustering of the prototypes by 1 - cosine of their updates, for objects and goals;
  neighbours           -- each prototype's nearest prototype of the same kind;
  instance vs type     -- the cosine of each instance child to its own type prototype, against other types.
Figures: figures/robot/i5_dendrogram.png (copied to the site media).

    sbatch --export=ALL,SCRIPT="experiments/i5_library.py --record" slurm/jobs/cpu.sbatch
"""
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import i4_factors as I4                                                # noqa: E402

R = os.path.join(ROOT, "results", "robot")


def library():
    """-> {prototype name: (kind, concept label, number of tasks it serves)} over E3's and E5's tasks."""
    e3 = json.load(open(os.path.join(R, "e3_plan.json")))
    e5 = json.load(open(os.path.join(R, "e5_plan.json")))
    out, serve = {}, {}
    for v in e3.values():
        for n, kind, lab, tasks in ((v["type_proto"], "object", v["object_key"], v["obj_tasks"]),
                                    (v["goal_proto"], "goal", v["goal_key"], v["goal_tasks"])):
            out[n] = (kind, lab.replace("|", " ") if kind == "goal" else lab.split("|", 1)[1], len(tasks))
        if v["instance_key"]:
            out[v["object_proto"]] = ("instance", v["instance_key"].split("|", 1)[1].replace("|", " · "), len(v["object_tasks"]))
    for k, e in e5["new"].items():
        kind = {"O": "object", "I": "instance", "G": "goal"}[k[0]]
        lab = k[2:].replace("|", " ") if kind == "goal" else k[2:].split("|", 1)[1].replace("|", " · ")
        out[e["name"]] = (kind, lab, len(e["tasks"]))
    return {n: v for n, v in out.items() if os.path.exists(os.path.join(R, f"prim_{n}_0.pt"))}


def main():
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lib = library()
    P = {n: I4.load(os.path.join(R, f"prim_{n}_0.pt")) for n in lib}
    sq = {n: I4.inner(p, p) for n, p in P.items()}
    names = sorted(lib)
    S = np.eye(len(names))
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            S[i, j] = S[j, i] = I4.inner(P[a], P[names[j]]) / np.sqrt(sq[a] * sq[names[j]])
    res = dict(names=names, labels={n: lib[n][1] for n in names}, kinds={n: lib[n][0] for n in names}, similarity=S.round(4).tolist())
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.5))
    for ax, kinds, title in ((axes[0], ("object", "instance"), "object and instance prototypes"), (axes[1], ("goal",), "goal prototypes")):
        idx = [i for i, n in enumerate(names) if lib[n][0] in kinds]
        D = 1 - S[np.ix_(idx, idx)]
        np.fill_diagonal(D, 0)
        Z = linkage(squareform(np.clip(D, 0, None), checks=False), method="average")
        labels = [f"{lib[names[i]][1]} ({lib[names[i]][2]})" for i in idx]
        dendrogram(Z, labels=labels, orientation="right", ax=ax, color_threshold=0, above_threshold_color="#3b464c",
                   leaf_font_size=8)
        for t in ax.get_yticklabels():
            t.set_color("#5b4fa0" if "·" in t.get_text() else ("#a35d2b" if "goal" in title else "#1d5e86"))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("1 - cosine of the updates")
        ax.spines[["top", "right", "left"]].set_visible(False)
        nn = {}
        for a in idx:
            others = [b for b in idx if b != a]
            b = max(others, key=lambda b: S[a, b])
            nn[lib[names[a]][1]] = (lib[names[b]][1], round(float(S[a, b]), 3))
        res[f"nearest_{kinds[0]}"] = nn
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", "robot", "i5_dendrogram.png")
    fig.savefig(out, dpi=150)
    shutil.copy(out, os.path.join(R, "site", "media", "i5_dendrogram.png"))
    inst = [i for i, n in enumerate(names) if lib[n][0] == "instance"]
    types = [i for i, n in enumerate(names) if lib[n][0] == "object"]
    own, other = [], []
    for i in inst:
        base = lib[names[i]][1].split(" · ")[0]
        for j in types:
            (own if lib[names[j]][1] == base else other).append(S[i, j])
    res["instance_to_own_type"] = float(np.mean(own)) if own else None
    res["instance_to_other_types"] = float(np.mean(other)) if other else None
    off = S[~np.eye(len(names), dtype=bool)]
    res["n_prototypes"] = len(names)
    res["mean_offdiag"] = float(off.mean())
    print(json.dumps({k: v for k, v in res.items() if k not in ("similarity", "labels", "kinds", "names")}, indent=1), flush=True)
    json.dump(res, open(os.path.join(R, "i5_library.json"), "w"), indent=1)
    if "--record" in sys.argv:
        from lifelong_ood import ledger
        print(ledger.record("i5_library", bench="libero_90", tag="read", seed=0,
                            metrics={k: v for k, v in res.items() if k != "similarity"},
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/i5_library.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    main()
