"""Figures of plan Z for the paper page, read from the ledger (the rows experiments/audit_numbers.py checks):
    figures/robot/z_grades.png        results by grade (split_grades tag read)
    figures/robot/z_unsupervised.png  boundaries vs identities (zu_read tags sweep, kcut, identity)
    figures/robot/z_blend.png         the soft blend: held-out loss by expert set, donation by relation (zi_blend tag read)
Colours: the dataviz skill's validated reference palette (slots 1-3), the policy alone in neutral grey.

    $PY experiments/fig_plan_z.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger as L                                     # noqa: E402

OUT = os.path.join(ROOT, "figures", "robot")
GREY, BLUE, ORANGE, AQUA = "#9a9a96", "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True})


def row(exp, tag):
    return L.load(exp, tag=tag)[-1]["metrics"]


def bars(ax, groups, series, ylabel, ylim=None, fmt="{:.2f}"):
    """groups: x labels; series: [(name, colour, [value or None per group])]."""
    n = len(series)
    w = 0.8 / n
    for j, (name, c, vals) in enumerate(series):
        xs = [i - 0.4 + w * (j + 0.5) for i in range(len(groups))]
        for x, v in zip(xs, vals):
            if v is None:
                continue
            ax.bar(x, v, w * 0.92, color=c, label=name if x == xs[next(k for k, u in enumerate(vals) if u is not None)] else None)
            ax.text(x, v, fmt.format(v), ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, fontsize=7, loc="upper right")


def grades():
    T = row("split_grades", "read")["table"]

    def rate(name, g, arm):
        b = T[name].get(g)
        return None if not b or arm not in b else b[arm] / (20 * b["tasks"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw=dict(width_ratios=[1.25, 1]))
    li, z3, g8 = "Li OOD G15 (pi0.5)", "Z3 pointers, zero fit (pi0.5)", "G8 LIBERO-90 set d (pi0.5)"
    groups = [f"G1 relation\n({T[li]['G1']['tasks']} tasks)", f"G2 binding\n({T[li]['G2']['tasks']} tasks)",
              f"G3 scene\n({T[z3]['G3']['tasks']} / {T[g8]['G3']['tasks']} tasks)", f"G4 part\n({T[z3]['G4']['tasks']} / {T[g8]['G4']['tasks']} tasks)"]
    bars(a1, groups, [
        ("the policy alone", GREY, [rate(li, "G1", "zs"), rate(li, "G2", "zs"), rate(g8, "G3", "zs"), rate(g8, "G4", "zs")]),
        ("composing held units, nothing fitted", BLUE, [rate(li, "G1", "chosen"), rate(li, "G2", "chosen"), rate(z3, "G3", "pointers"),
                                                        rate(z3, "G4", "pointers")]),
        ("creating units (task leaves)", ORANGE, [None, None, rate(g8, "G3", "leaves"), rate(g8, "G4", "leaves")])],
        "success rate", (0, 1.08))
    a1.set_title("pi0.5, by what the task leaves unseen", fontsize=9, color=INK, loc="left")
    pols = [("pi0.5", li), ("pi0", "Li OOD G15 (pi0)"), ("SmolVLA", "Li OOD G18 (SmolVLA)"), ("X-VLA", "Li OOD G18 (X-VLA)"),
            ("GR00T N1.5", "Li OOD G23 (GR00T N1.5)")]

    def pooled(name, arm):
        b = [T[name][g] for g in ("G1", "G2") if g in T[name]]
        return sum(x[arm] for x in b) / (20 * sum(x["tasks"] for x in b))
    bars(a2, [p for p, _ in pols], [("the policy alone", GREY, [pooled(n, "zs") for _, n in pols]),
                                    ("composing held units, nothing fitted", BLUE, [pooled(n, "chosen") for _, n in pols])],
         "success rate", (0, 1.08))
    a2.set_title("G1 + G2 (Li's 20 tasks), five policies", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "z_grades.png"), dpi=200)


def unsupervised():
    S, K, I = row("zu_read", "sweep"), row("zu_read", "kcut"), row("zu_read", "identity")["identity"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.2))
    lab = S["label/4"]
    bars(a1, ["label switch\n(Viterbi)", "loss signature\n(2 cuts)", "prefix latent\n(2 cuts)", "arm actions\n(2 cuts, non-policy)"],
         [("segmenter", BLUE, [lab["grasp"], K["sig"]["grasp"], K["lat"]["grasp"], K["act"]["grasp"]]),
          ("random cuts, same count", GREY, [lab["grasp_random"], K["random"]["grasp"], K["random"]["grasp"], K["random"]["grasp"]])],
         "grasp found within 10 frames", (0, 1.0))
    a1.set_title("the likelihood does not find the boundaries (72 demos)", fontsize=9, color=INK, loc="left")
    keys = [("familiar/approach", "familiar scenes\napproach"), ("familiar/carry", "familiar scenes\ncarry"),
            ("l90/approach", "LIBERO-90\napproach"), ("l90/carry", "LIBERO-90\ncarry")]
    bars(a2, [f"{t}\n(n {I[k]['n']})" for k, t in keys],
         [("top-1 instruction holds the concept", AQUA, [I[k]["top1"] for k, _ in keys]),
          ("chance", GREY, [I[k]["chance"] for k, _ in keys])], "share of segments", (0, 1.0))
    a2.set_title("it does name a segment cut at contact events", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "z_unsupervised.png"), dpi=200)


def blend():
    M = row("zi_blend", "read")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.2))
    key = "approach/position"
    sets = [("empty\nprompt", "empty"), ("uniform\nmix", "uniform"), ("best single\nexpert", "hard"), ("soft\nblend", "blend"),
            ("own / new\ninstruction", "own")]
    bars(a1, [s for s, _ in sets], [("familiar scenes (own = held-out answer)", BLUE, [M["l40"]["held"][key][k] for _, k in sets]),
                                    ("LIBERO-90 (new = the task's instruction)", ORANGE, [M["l90"]["held"][key][k] for _, k in sets])],
         "held-out flow loss (lower is better)", (0, 0.36), fmt="{:.3f}")
    a1.legend(frameon=False, fontsize=7, loc="upper left")
    a1.set_title("approach, position: held fields explain the motion", fontsize=9, color=INK, loc="left")
    d = M["l90"]["donation"][key]
    rel = [("shares\nthe object", "object"), ("shares\nthe goal", "goal"), ("shares\nboth", "both"), ("shares\nnothing", "neither"),
           ("the new\ninstruction", "own")]
    rel = [(t, k) for t, k in rel if k in d]
    bars(a2, [f"{t}\n(n {d[k]['n']})" for t, k in rel], [("mean Shapley share", AQUA, [d[k]["share"] for _, k in rel])],
         "mean Shapley share of the blend's donation", (0, 0.5))
    a2.get_legend().remove()
    a2.set_title("LIBERO-90: donation follows behaviour, not the concept", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "z_blend.png"), dpi=200)


if __name__ == "__main__":
    grades()
    unsupervised()
    blend()
    print("ok")
