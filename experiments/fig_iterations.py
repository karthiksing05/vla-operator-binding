"""The iteration ladder: each iteration of the compositional system against its reference, read from the ledger.

Left, LIBERO-90 (start states no design decision saw): success rate of the kept system and of the per-task LoRA with acceptance
at iterations 1 (goal units, D1), 2 (object units, E1), 3 (instance children, E3) and 8 (task leaves, G8, 66 tasks).
Right, Li's compositional OOD suites on pi0.5 (no demonstration of any test task), all unfiltered on start states set c: the policy
alone, the policy's own fields switched at the grasp with word-matched sources and with Li's hand-chosen sources (G4d), and with
practice-chosen sources (G10).  -> figures/robot/iterations.png (copied to the site media).

    $PY experiments/fig_iterations.py
"""
import glob
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results", "robot")


def row(exp, tag):
    fs = sorted(glob.glob(os.path.join(ROOT, "ledger", exp, f"*__{tag}__*.json")))
    return json.load(open(fs[-1]))["metrics"]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    left = []
    for lab, exp in (("1 goal units", "d1_factored"), ("2 object units", "e1_factored"), ("3 instance children", "e3_hier")):
        h = row(exp, "read")["hypotheses"]["H2"]
        left.append((lab, h["x"] / h["n"], h["y"] / h["n"]))
    g8 = row("g8_leaves", "read")["H1"]
    left.append(("8 task leaves (66 tasks)", g8["x"] / g8["n"], g8["y"] / g8["n"]))
    d = row("g4_ood", "read_li_pi05")                                       # all on start states set c, unfiltered, paired
    g10 = row("g4_ood", "read_g10_pi05")
    n = d["li_grasp_vs_zs"]["n"]
    right = [("policy alone", d["li_grasp_vs_zs"]["y"] / n), ("5: own fields,\nword sources", d["ours_grasp_vs_zs"]["x"] / n),
             ("Li's hand-chosen\nsources", d["li_grasp_vs_zs"]["x"] / n), ("9: practice-chosen\nsources", g10["sel_vs_li_grasp"]["x"] / n)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axes[0]
    xs = range(len(left))
    w = 0.38
    ax.bar([x - w / 2 for x in xs], [v[1] for v in left], w, color="#2f7d4f", label="the system (kept by practice)")
    ax.bar([x + w / 2 for x in xs], [v[2] for v in left], w, color="#9ca3af", label="per-task LoRA with acceptance")
    for x, v in zip(xs, left):
        ax.text(x - w / 2, v[1] + 0.01, f"{v[1]:.2f}", ha="center", fontsize=7)
        ax.text(x + w / 2, v[2] + 0.01, f"{v[2]:.2f}", ha="center", fontsize=7, color="#4b5563")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([v[0] for v in left], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("success rate", fontsize=8)
    ax.set_title("LIBERO-90: iterations against a per-task adapter", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
    ax.set_ylim(0, 1.1)
    ax = axes[1]
    cols = ["#9ca3af", "#1d5e86", "#6b7280", "#2f7d4f"]
    ax.bar(range(len(right)), [v[1] for v in right], 0.6, color=cols)
    for x, v in enumerate(right):
        ax.text(x, v[1] + 0.01, f"{v[1]:.2f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(right)))
    ax.set_xticklabels([v[0] for v in right], fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_title("Li's compositional OOD suites, pi0.5, no demonstration", fontsize=9)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=7)
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", "robot", "iterations.png")
    fig.savefig(out, dpi=150)
    shutil.copy(out, os.path.join(R, "site", "media", "iterations.png"))
    print(out, [(v[0], round(v[1], 3), round(v[2], 3)) for v in left], [(v[0], round(v[1], 3)) for v in right])


if __name__ == "__main__":
    main()
