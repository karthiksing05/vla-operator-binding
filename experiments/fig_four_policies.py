"""Composition across four flow-matching policies, read from the ledger: Li's 20 compositional OOD tasks (no demonstration of any
test task), start states set e, 20 per task, paired.  Per policy: the policy alone, Li's text-latent interpolation with Li's
hand-chosen pairs, and the system (practice chooses each held concept's sources and, per task, the operator).  pi0 and pi0.5
from G15 (ledger g15_operator), SmolVLA and X-VLA from G18 (ledger g18_four_policies).  -> figures/robot/four_policies.png.

    $PY experiments/fig_four_policies.py
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
    rows = []
    for lab, exp, tag, zs, li in (("pi0", "g15_operator", "read_pi0", "chosen_vs_zs", "chosen_vs_tli_li"),
                                  ("pi0.5", "g15_operator", "read_pi05", "chosen_vs_zs", "chosen_vs_tli_li"),
                                  ("SmolVLA", "g18_four_policies", "read_smolvla", "chosen_vs_zs", "chosen_vs_ref"),
                                  ("X-VLA", "g18_four_policies", "read_xvla", "chosen_vs_zs", "chosen_vs_ref")):
        m = row(exp, tag)
        n = m[zs]["n"]
        rows.append((lab, m[zs]["y"] / n, m[li]["y"] / n, m[zs]["x"] / n))
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    w = 0.26
    for j, (name, c) in enumerate((("the policy alone", "#9ca3af"), ("Li's TLI, hand-chosen pairs", "#1d5e86"),
                                   ("composition chosen by practice (ours)", "#2f7d4f"))):
        xs = [i + (j - 1) * w for i in range(len(rows))]
        ys = [r[1 + j] for r in rows]
        ax.bar(xs, ys, w, color=c, label=name)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.01, f"{y:.2f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success rate", fontsize=8)
    ax.set_title("Li's compositional OOD suites on four flow-matching policies (new start states, no demonstration)", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", "robot", "four_policies.png")
    fig.savefig(out, dpi=150)
    shutil.copy(out, os.path.join(R, "site", "media", "four_policies.png"))
    print(out, [(r[0], round(r[1], 3), round(r[2], 3), round(r[3], 3)) for r in rows])


if __name__ == "__main__":
    main()
