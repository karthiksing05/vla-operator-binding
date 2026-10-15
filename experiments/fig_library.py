"""Figures of the library as a continual learner (the W1 stream, docs/METHOD.md): concept swimlanes over arrivals, and the
per-arrival narrative the results page tabulates.  Reads results/robot/w1_state.json (the stream's own state; numbers on the page
come from ledger w1_stream).

    ~/flash/miniconda3/envs/chunks/bin/python experiments/fig_library.py
-> figures/robot/stream_lanes_<order>.png (copied to the site media) and results/robot/stream_story.json
"""
import json
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
R = os.path.join(ROOT, "results", "robot")
INK, MUTED, OBJ, GOAL, INST, GOOD = "#161b1e", "#5d686f", "#1d5e86", "#a35d2b", "#5b4fa0", "#16804f"


def short(k):
    kind, rest = k.split(":", 1)
    parts = rest.split("|")
    if kind == "G":
        return f"{parts[0]} {parts[1].replace('the ', '')}"
    if kind == "I":
        return f"{parts[1].replace('the ', '')} · {parts[2]}"
    return parts[1].replace("the ", "")


def events(st, o):
    """-> (lanes, marks, story): marks[(concept, arrival)] in {prior, created, refit, used, reused}."""
    s = st[o]
    arr = s["arrivals"]
    snaps = [a["library_before"] for a in arr] + [{k: e["name"] for k, e in s["library"].items()}]
    name2key = {}
    for snap in snaps:
        for k, n in snap.items():
            if n:
                name2key[n] = k
    marks, story, uses = {}, [], set()
    for k, n in snaps[0].items():
        if n:
            marks[(k, 0)] = "prior"
    for i, a in enumerate(arr, 1):
        changed = [k for k in snaps[i] if snaps[i][k] != snaps[i - 1][k]]
        for k in changed:
            marks[(k, i)] = "created" if not snaps[i - 1][k] else "refit"
        used = []
        if a["kept"] != "zs":
            spec = a["kept"].split("@", 1)[1]
            used = [name2key[x] for x in spec.split(".") if x != "x" and x in name2key]
            for k in used:
                uses.add((k, i, a.get("demo") is False))
        story.append(dict(arrival=i, task=a["task"], policy=a["zs"], composition=a.get("acc1"), demo=bool(a.get("demo")),
                          refit=[short(k) for k in changed], kept="policy" if a["kept"] == "zs" else "library",
                          after=a.get("acc2"), uses=[short(k) for k in used]))
    lanes = sorted({k for k, _ in marks}, key=lambda k: ({"G": 0, "O": 1, "I": 2}[k[0]], short(k)))
    lanes = sorted(set(lanes) | {k for k, _, _ in uses}, key=lambda k: ({"G": 0, "O": 1, "I": 2}[k[0]], short(k)))
    return lanes, marks, story, uses


def draw(st, o):
    lanes, marks, story, uses = events(st, o)
    n = len(st[o]["arrivals"])
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(11, 0.26 * len(lanes) + 3.2), gridspec_kw=dict(height_ratios=[len(lanes), 7]),
                                 sharex=True)
    col = {"G": GOAL, "O": OBJ, "I": INST}
    sty = {"prior": dict(marker="s", s=34, facecolors="white"), "created": dict(marker="o", s=48),
           "refit": dict(marker="o", s=48, facecolors="white"), "used": dict(marker="|", s=120),
           "reused": dict(marker="*", s=110)}
    for j, k in enumerate(lanes):
        ax.axhline(j, color="#dce2e5", lw=0.8, zorder=0)
        for kk, i, free in uses:
            if kk == k and free:
                ax.scatter([i], [j], color=GOOD, zorder=4, **sty["reused"])
            elif kk == k:
                ax.plot([i, i], [j - 0.38, j + 0.38], color=col[k[0]], lw=2.2, alpha=0.55, zorder=2)
        for (kk, i), m in marks.items():
            if kk != k:
                continue
            kw = dict(sty[m])
            c = GOOD if m == "reused" else col[k[0]]
            if kw.get("facecolors") == "white":
                ax.scatter([i], [j], edgecolors=c, linewidths=1.6, zorder=3, **kw)
            else:
                ax.scatter([i], [j], color=c, zorder=3, **kw)
    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels([short(k) for k in lanes], fontsize=8)
    for t, k in zip(ax.get_yticklabels(), lanes):
        t.set_color(col[k[0]])
    ax.invert_yaxis()
    for m, lab in (("prior", "from tasks outside the stream"), ("created", "created by a demonstration"),
                   ("refit", "refit by a later demonstration"), ("used", "serves the arrival"),
                   ("reused", "serves it with no demonstration")):
        kw = dict(sty[m])
        c = GOOD if m == "reused" else INK
        if kw.get("facecolors") == "white":
            ax.scatter([], [], edgecolors=c, linewidths=1.4, label=lab, **kw)
        else:
            ax.scatter([], [], color=c, label=lab, **kw)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0 + 2.4 / len(lanes)), ncol=5, frameon=False, fontsize=8)
    demos, protos = [0], [sum(1 for (k, i), m in marks.items() if i == 0)]
    have = {k for (k, i) in marks if i == 0}
    for i in range(1, n + 1):
        demos.append(demos[-1] + int(story[i - 1]["demo"]))
        have |= {k for (k, ii), m in marks.items() if ii == i and m == "created"}
        protos.append(len(have))
    bx.step(range(n + 1), demos, where="post", color=INK, lw=1.8, label="demonstrations requested")
    bx.step(range(n + 1), protos, where="post", color=OBJ, lw=1.8, label="prototypes stored")
    bx.plot(range(n + 1), range(n + 1), ls=":", color=MUTED, lw=1, label="one per arrival")
    bx.set_xlabel("arrival")
    bx.legend(frameon=False, fontsize=8, loc="upper left")
    for a_ in (ax, bx):
        a_.spines[["top", "right"]].set_visible(False)
    bx.set_xlim(-0.5, n + 0.5)
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", "robot", f"stream_lanes_{o}.png")
    fig.savefig(out, dpi=150)
    shutil.copy(out, os.path.join(R, "site", "media", os.path.basename(out)))
    return story, dict(demos=demos[-1], prototypes=protos[-1])


def main():
    st = json.load(open(os.path.join(R, "w1_state.json")))
    texts = {}
    try:
        import importlib.util
        sp = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
        r3p = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(r3p)
        texts = r3p.libero90_texts()[0]
    except Exception:                                            # noqa: BLE001
        pass
    out = {}
    for o in ("a", "b"):
        story, tot = draw(st, o)
        for s_ in story:
            s_["text"] = texts.get(s_["task"], "")
        out[o] = dict(story=story, totals=tot)
        print(o, tot, flush=True)
    json.dump(out, open(os.path.join(R, "stream_story.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
