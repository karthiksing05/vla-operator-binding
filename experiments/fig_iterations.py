"""The iteration ladder: each iteration of the compositional system against its reference, read from the ledger.

Left, LIBERO-90 (start states no design decision saw): success rate of the kept system and of the per-task LoRA with acceptance
at iterations 1 (goal units, D1), 2 (object units, E1), 3 (instance children, E3) and 8 (task leaves, G8, 66 tasks).
Right, Li's compositional OOD suites on pi0 and pi0.5 (no demonstration of any test task), all unfiltered on start states set e
(G15): the policy alone, Li's TLI with hand-chosen pairs, the grasp switch and TLI with practice-chosen pairs (iterations 9 and
11), and practice choosing the operator per task (iteration 12).  -> figures/robot/iterations.png (copied to the site media).

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
    right = {}                                                          # set e, no demonstration, unfiltered, paired (G15)
    for b in ("pi0", "pi05"):
        g = row("g15_operator", f"read_{b}")
        n = g["chosen_vs_zs"]["n"]
        right[b] = [g["chosen_vs_zs"]["y"] / n, g["chosen_vs_tli_li"]["y"] / n, g["chosen_vs_psel"]["y"] / n,
                    g["chosen_vs_tsel"]["y"] / n, g["chosen_vs_zs"]["x"] / n]
    labs = ["policy alone", "Li's TLI\n(hand-chosen pairs)", "9: grasp switch,\npractice pairs", "11: TLI,\npractice pairs",
            "12: practice chooses\nthe operator"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.9), gridspec_kw=dict(width_ratios=[1.1, 1.2]))
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
    ax.set_ylabel("success rate", fontsize=8)
    ax.set_title("LIBERO-90: iterations against a per-task adapter", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
    ax.set_ylim(0, 1.1)
    ax = axes[1]
    xs = range(len(labs))
    for j, (b, c) in enumerate((("pi0", "#1d5e86"), ("pi05", "#2f7d4f"))):
        ax.bar([x + (j - 0.5) * w for x in xs], right[b], w, color=c, label={"pi0": "pi0", "pi05": "pi0.5"}[b])
        for x, y in zip(xs, right[b]):
            ax.text(x + (j - 0.5) * w, y + 0.01, f"{y:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labs, fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
    ax.set_title("Li's compositional OOD suites, new start states (set e), no demonstration", fontsize=9)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=7)
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", "robot", "iterations.png")
    fig.savefig(out, dpi=150)
    shutil.copy(out, os.path.join(R, "site", "media", "iterations.png"))
    print(out, [(v[0], round(v[1], 3), round(v[2], 3)) for v in left], {b: [round(y, 3) for y in v] for b, v in right.items()})


if __name__ == "__main__":
    main()
