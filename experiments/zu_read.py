"""U1 read (exploratory, offline, CPU): unsupervised boundaries and identities from the frozen pi0.5's 40 conditional fields
(docs/UNSUPERVISED.md).  Reads results/robot/u1/*.npz (experiments/zu_score.py).

BOUNDARIES.  C[f, l] = the flow loss at frame f under training instruction l (the demo's own instruction is never a label; for
LIBERO-10 demos, which ARE training demos, their own instruction is removed from the 40).  Two segmenters (segment.py):
    label : exact Viterbi over (frame, label), cost = normalised loss, lam per segment  -> cuts where the best field switches
    level : optimal piecewise-constant partition of log(min_l C[f, l])                -> cuts where the best field's loss jumps
The penalty is chosen on the EVEN template tasks (max F1 of grasp + release recovery at 10 frames = 0.5 s) and scored on the ODD
ones.  Truth = the demo's sustained grasp / release (scoring only).  Controls: the same cost without the gripper dim (dims 0-5),
and random cuts with the same count.

IDENTITIES.  A segment is HELD (a zero-byte pointer to its label's field) when its mean loss under its label is below I6's held
threshold (the held concepts' max, results/robot/i6_novelty.json), else NEW.  New segments are clustered by their LOSS
SIGNATURE: the mean over the segment of log C[f, :], centred -- the policy's own description of a segment by how well each of
its held fields explains it (architecture-agnostic: it needs only the flow loss, which every policy here has).  Number of units
by the best silhouette (average linkage, correlation distance).  The empty-prompt prefix latent is a second feature (a check).

    $PY experiments/zu_read.py [--record]
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood.policy import primitives as PR                       # noqa: E402
from lifelong_ood.policy import segment as S                           # noqa: E402

R = os.path.join(ROOT, "results", "robot")
FIG = os.path.join(ROOT, "figures", "robot")
TOL = 10                                                                 # frames (0.5 s at 20 Hz)
LAMS = [0.5, 1, 2, 4, 8, 16, 32, 64]
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def kind(text):
    steps = PR.steps_of(text)
    if len(steps) > 1:
        return "two-step"
    ok = PR.object_key(text)
    if ok.startswith("put|") and PR.goal_key(text).split("|")[0]:
        return "pick-place"
    v = ok.split("|")[0]
    return {"open": "open", "close": "close", "turn on": "knob", "turn off": "knob", "stack": "stack"}.get(v, "other")


def load_all():
    demos = []
    for p in sorted(glob.glob(os.path.join(R, "u1", "*.npz"))):
        z = np.load(p, allow_pickle=True)
        name = os.path.basename(p)[:-4]
        st, t = name.split("_")[0], int(name.split("_")[1])
        full = ((7 * z["err"].mean(-1) + 25 * z["pad"]) / 32).mean(2)    # [F, 42]: I6's loss
        nogrip = z["err"][..., :6].mean((2, 3))                          # dims 0-5 only
        labels = [l for l in range(40) if not (st in ("l10", "l40") and l == t)]
        g, r = S.gripper_truth(z["actions"])
        demos.append(dict(name=name, set=st, task=t, text=str(z["text"]), kind=kind(str(z["text"])), frames=z["frames"],
                          full=full, nogrip=nogrip, labels=np.asarray(labels), lat=z["lat"], actions=z["actions"],
                          grasp=g, release=r, T=len(z["actions"])))
    return demos


def segs_of(d, method, lam, cost="full"):
    C = d[cost][:, d["labels"]]
    if method == "label":
        segs = S.label_viterbi(S.normalise(C), lam)
        return [(lo, hi, int(d["labels"][l])) for lo, hi, l in segs]
    y = np.log(C.min(1))
    segs = S.level_partition(y, lam * 0.05)
    return [(lo, hi, int(d["labels"][int(np.argmin(C[lo:hi].mean(0)))])) for lo, hi, _ in segs]


def cuts(d, segs):
    return [int(d["frames"][b]) for b in S.boundaries(segs)]


def score_demo(d, b):
    truth = d["grasp"][:1] + d["release"][:1] if d["kind"] == "pick-place" else d["grasp"] + d["release"]
    return S.match(b, truth, TOL), truth


def evaluate(demos, method, lam, cost="full"):
    errs_g, errs_r, prec, rec, npred = [], [], [], [], []
    for d in demos:
        b = cuts(d, segs_of(d, method, lam, cost))
        m, _ = score_demo(d, b)
        if d["grasp"]:
            errs_g.append(m["err"][0])
        if d["release"]:
            errs_r.append(m["err"][1] if len(m["err"]) > 1 else np.inf)
        rec.append(m["recall"])
        if m["precision"] is not None:
            prec.append(m["precision"])
        npred.append(m["n_pred"])
    p, r_ = float(np.mean(prec)) if prec else 0.0, float(np.mean(rec))
    return dict(lam=lam, method=method, cost=cost, n=len(demos), recall=r_, precision=p, f1=2 * p * r_ / (p + r_ + 1e-12),
                grasp_within_tol=float(np.mean(np.asarray(errs_g) <= TOL)), release_within_tol=float(np.mean(np.asarray(errs_r) <= TOL)),
                grasp_err_median=float(np.median(errs_g)), release_err_median=float(np.median(errs_r)),
                cuts_per_demo=float(np.mean(npred)), err_grasp=[float(x) for x in errs_g], err_release=[float(x) for x in errs_r])


def random_control(demos, n_cuts, reps=200, seed=0):
    rng = np.random.default_rng(seed)
    g, r = [], []
    for d in demos:
        for _ in range(reps):
            b = rng.integers(0, d["T"], size=max(1, int(round(n_cuts))))
            m, _ = score_demo(d, b)
            g.append(m["err"][0] <= TOL)
            r.append(len(m["err"]) > 1 and m["err"][1] <= TOL)
    return dict(grasp_within_tol=float(np.mean(g)), release_within_tol=float(np.mean(r)))


def tune(demos, method, cost="full"):
    tr = [d for d in demos if d["task"] % 2 == 0]
    te = [d for d in demos if d["task"] % 2 == 1]
    sweep = [evaluate(tr, method, l, cost) for l in LAMS]
    best = max(sweep, key=lambda x: x["f1"])
    test = evaluate(te, method, best["lam"], cost)
    test["random_same_count"] = random_control(te, test["cuts_per_demo"])
    return dict(sweep=[{k: v for k, v in s.items() if not k.startswith("err_")} for s in sweep], lam=best["lam"], test=test,
                all=evaluate(demos, method, best["lam"], cost))


def sig(d, lo, hi):
    x = np.log(d["full"][lo:hi, :40]).mean(0)
    if d["set"] == "l10":
        x[d["task"]] = np.nan
    x = x - np.nanmean(x)
    return np.nan_to_num(x, nan=0.0)


def cluster(X, kmax=30):
    from scipy.cluster.hierarchy import fcluster, linkage
    from sklearn.metrics import silhouette_score
    Z = linkage(X, "average", metric="correlation")
    best = (-1, None, None)
    for k in range(2, min(kmax, len(X) - 1) + 1):
        lab = fcluster(Z, k, "maxclust")
        if len(set(lab)) < 2:
            continue
        s = silhouette_score(X, lab, metric="correlation")
        if s > best[0]:
            best = (s, k, lab)
    return best


def supervised():
    """task -> (object concept, goal concept) of the supervised library (E3 + E5)."""
    e3 = json.load(open(os.path.join(R, "e3_plan.json")))
    e5 = json.load(open(os.path.join(R, "e5_plan.json")))
    out = {int(t): (v["type_proto"], v["goal_proto"]) for t, v in e3.items()}
    out.update({int(t): (v["type"], v["goal"]) for t, v in e5["tasks"].items()})
    return out


def nmi_ari(a, b):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    return dict(nmi=float(normalized_mutual_info_score(a, b)), ari=float(adjusted_rand_score(a, b)), n=len(a))


def main(record):
    demos = load_all()
    i6 = json.load(open(os.path.join(R, "i6_novelty.json")))
    thr = float(i6["held_max"])
    pp = [d for d in demos if d["set"] == "l90" and d["kind"] == "pick-place" and d["grasp"] and d["release"]]
    res = dict(n_demos=len(demos), n_pickplace=len(pp), tol_frames=TOL, held_threshold=thr,
               by_kind={k: sum(d["kind"] == k for d in demos) for k in sorted({d["kind"] for d in demos})})
    for method in ("label", "level"):
        for cost in ("full", "nogrip"):
            if method == "level" and cost == "nogrip":
                continue
            res[f"{method}_{cost}"] = tune(pp, method, cost)
            t = res[f"{method}_{cost}"]["test"]
            print(f"[u1] {method}/{cost}: lam {res[f'{method}_{cost}']['lam']}, test n={t['n']}: grasp within {TOL} "
                  f"{t['grasp_within_tol']:.2f} (median err {t['grasp_err_median']:.0f}), release {t['release_within_tol']:.2f} "
                  f"(median {t['release_err_median']:.0f}), precision {t['precision']:.2f}, cuts/demo {t['cuts_per_demo']:.1f}; "
                  f"random same count: grasp {t['random_same_count']['grasp_within_tol']:.2f}, release "
                  f"{t['random_same_count']['release_within_tol']:.2f}", flush=True)
    lam = res["label_full"]["lam"]
    # where the boundaries fall relative to the grasp: the policy's best field switches (label), per frame offset
    # -- non-template tasks
    nont = {}
    for d in demos:
        if d["kind"] == "pick-place" and d["set"] == "l90":
            continue
        segs = segs_of(d, "label", lam)
        b = cuts(d, segs)
        truth = d["grasp"] + d["release"]
        m = S.match(b, truth, TOL)
        nont[d["name"]] = dict(kind=d["kind"], text=d["text"], cuts=b, grasps=d["grasp"], releases=d["release"], T=d["T"],
                               recall=m["recall"], precision=m["precision"],
                               labels=[int(l) for _, _, l in segs])
    res["non_template"] = nont
    res["non_template_by_kind"] = {}
    for k in sorted({v["kind"] for v in nont.values()}):
        vs = [v for v in nont.values() if v["kind"] == k]
        res["non_template_by_kind"][k] = dict(n=len(vs), cuts_per_demo=float(np.mean([len(v["cuts"]) for v in vs])),
                                              gripper_events_per_demo=float(np.mean([len(v["grasps"]) + len(v["releases"]) for v in vs])),
                                              recall=float(np.nanmean([v["recall"] if v["recall"] is not None else np.nan for v in vs])),
                                              precision=float(np.nanmean([v["precision"] if v["precision"] is not None else np.nan
                                                                          for v in vs])))
    # identities
    segrows = []
    for d in demos:
        for lo, hi, l in segs_of(d, "label", lam):
            loss = float(d["full"][lo:hi, l].mean())
            f0, f1 = int(d["frames"][lo]), int(d["frames"][hi - 1])
            g = d["grasp"][0] if d["grasp"] else None
            r = d["release"][0] if d["release"] else None
            phase = None
            if g is not None and r is not None:
                mid = (f0 + f1) / 2
                ov_a = max(0, min(f1, g) - f0)
                ov_c = max(0, min(f1, r) - max(f0, g))
                phase = "approach" if ov_a >= ov_c and ov_a > 0 else "carry" if ov_c > 0 else "after"
            segrows.append(dict(demo=d["name"], set=d["set"], task=d["task"], kind=d["kind"], lo=f0, hi=f1, label=l, loss=loss,
                                held=loss <= thr, phase=phase, sig=sig(d, lo, hi), lat=d["lat"][lo:hi, 0].mean(0)))
    res["segments"] = len(segrows)
    res["held_segments"] = int(sum(s["held"] for s in segrows))
    res["held_by_set"] = {st: [int(sum(s["held"] for s in segrows if s["set"] == st)), int(sum(s["set"] == st for s in segrows))]
                          for st in ("l90", "l10")}
    new = [s for s in segrows if not s["held"]]
    X = np.stack([s["sig"] for s in new])
    sil, k, lab = cluster(X)
    Xl = np.stack([s["lat"] for s in new])
    sil_l, k_l, lab_l = cluster(Xl - Xl.mean(0))
    for s, c, cl in zip(new, lab, lab_l):
        s["unit"], s["unit_lat"] = int(c), int(cl)
    tasks_all = sorted({(s["set"], s["task"]) for s in segrows})
    units = {}
    for s in new:
        units.setdefault(s["unit"], set()).add((s["set"], s["task"]))
    labels_used = {}
    for s in segrows:
        labels_used.setdefault(s["label"], set()).add((s["set"], s["task"]))
    res["units"] = dict(feature="loss signature", k=int(k), silhouette=float(sil), tasks=len(tasks_all),
                        tasks_per_unit=float(np.mean([len(v) for v in units.values()])),
                        units_per_task=float(len(units) / len(tasks_all)),
                        sizes={int(u): len(v) for u, v in sorted(units.items())},
                        lat_k=int(k_l), lat_silhouette=float(sil_l),
                        sig_vs_lat=nmi_ari(list(lab), list(lab_l)))
    res["label_identity"] = dict(labels_used=len(labels_used), tasks=len(tasks_all),
                                 tasks_per_label=float(np.mean([len(v) for v in labels_used.values()])),
                                 per_label={int(l): len(v) for l, v in sorted(labels_used.items())})
    # agreement with the supervised concepts
    sup = supervised()
    agree = {}
    for phase, idx in (("approach", 0), ("carry", 1)):
        ss = [s for s in new if s["set"] == "l90" and s["phase"] == phase and s["task"] in sup]
        by_task = {}
        for s in ss:                                                     # the task's longest segment in the phase
            if s["task"] not in by_task or s["hi"] - s["lo"] > by_task[s["task"]]["hi"] - by_task[s["task"]]["lo"]:
                by_task[s["task"]] = s
        t_ = sorted(by_task)
        a = [by_task[t]["unit"] for t in t_]
        b = [sup[t][idx] for t in t_]
        l_ = [by_task[t]["label"] for t in t_]
        groups = {}
        for t in t_:
            groups.setdefault(by_task[t]["unit"], []).append(dict(task=t, concept=sup[t][idx]))
        agree[phase] = dict(unit_vs_concept=nmi_ari(a, b), label_vs_concept=nmi_ari(l_, b),
                            n_concepts=len(set(b)), n_units=len(set(a)), n_labels=len(set(l_)),
                            purity=float(sum(max(np.unique([g["concept"] for g in v], return_counts=True)[1]) for v in groups.values())
                                         / len(t_)),
                            groups={int(u): v for u, v in groups.items()})
    both = [t for t in sup if any(s["task"] == t and s["phase"] == "approach" for s in new)
            and any(s["task"] == t and s["phase"] == "carry" for s in new)]
    same = 0
    for t in both:
        ua = {s["unit"] for s in new if s["task"] == t and s["phase"] == "approach"}
        uc = {s["unit"] for s in new if s["task"] == t and s["phase"] == "carry"}
        same += bool(ua & uc)
    agree["approach_and_carry_share_a_unit"] = [same, len(both)]
    res["agreement"] = agree
    print(json.dumps({k: v for k, v in res.items() if k in ("held_segments", "segments", "held_by_set", "units", "label_identity")},
                     indent=1, default=str)[:3000], flush=True)
    for ph in ("approach", "carry"):
        a = agree[ph]
        print(f"[u1] {ph}: units vs concepts NMI {a['unit_vs_concept']['nmi']:.2f} ARI {a['unit_vs_concept']['ari']:.2f} "
              f"(n={a['unit_vs_concept']['n']}, {a['n_units']} units, {a['n_concepts']} concepts, purity {a['purity']:.2f}); "
              f"labels vs concepts NMI {a['label_vs_concept']['nmi']:.2f}", flush=True)
    json.dump(res, open(os.path.join(R, "z_u1_read.json"), "w", encoding="utf-8"), indent=1, default=str)
    figures(demos, res, segrows, new, lam)
    if record:
        from lifelong_ood import ledger
        slim = {k: v for k, v in res.items() if k != "non_template"}
        slim["non_template"] = {k: {kk: vv for kk, vv in v.items() if kk != "text"} for k, v in res["non_template"].items()}
        print(ledger.record("u1_segment", bench="libero_90", tag="read", seed=0, metrics=slim,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=dict(tol=TOL, lams=LAMS),
                            script="experiments/zu_read.py", command=" ".join(sys.argv)))


def figures(demos, res, segrows, new, lam):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    os.makedirs(FIG, exist_ok=True)
    # 1 timelines
    pick = []
    for want in ("pick-place", "pick-place", "open", "close", "knob", "two-step", "two-step", "stack"):
        cand = [d for d in demos if d["kind"] == want and d["name"] not in [p["name"] for p in pick]]
        if want == "two-step":
            cand = sorted(cand, key=lambda d: d["set"] != ("l10" if sum(p["kind"] == "two-step" for p in pick) else "l90"))
        if cand:
            pick.append(cand[0])
    fig, axes = plt.subplots(len(pick), 1, figsize=(9, 1.45 * len(pick)), sharex=False)
    for ax, d in zip(np.atleast_1d(axes), pick):
        C = np.log(d["full"][:, d["labels"]])
        C = C - C.min(1, keepdims=True)
        x = d["frames"]
        ax.imshow(C.T, aspect="auto", cmap="Greys_r", vmin=0, vmax=3, extent=[x[0], x[-1] + 1, len(d["labels"]), 0])
        segs = segs_of(d, "label", lam)
        labs = sorted({l for _, _, l in segs})
        for lo, hi, l in segs:
            ax.axvspan(x[lo], x[hi - 1] + 1, ymin=0.0, ymax=0.08, color=COL[labs.index(l) % len(COL)], lw=0)
            ax.text((x[lo] + x[hi - 1]) / 2, len(d["labels"]) * 0.85, f"#{l}", ha="center", fontsize=7, color="white")
        for b in cuts(d, segs):
            ax.axvline(b, color="#eb6834", lw=1.5)
        for g in d["grasp"]:
            ax.axvline(g, color="#1baf7a", lw=1.2, ls="--")
        for r in d["release"]:
            ax.axvline(r, color="#2a78d6", lw=1.2, ls=":")
        ax.set_yticks([])
        ax.set_title(f"{d['set'].upper()} {d['task']} ({d['kind']}): {d['text'][:80]}", fontsize=8, loc="left")
    np.atleast_1d(axes)[-1].set_xlabel("frame (20 Hz).  grey: log loss under each of the 40 held instructions minus the best "
                                       "(dark = explains);  orange: discovered cut;  green dashed: grasp;  blue dotted: release")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "u1_timelines.png"), dpi=140)
    plt.close(fig)
    # 2 boundary-error histogram (all template demos, tuned lam)
    fig, axs = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True)
    for ax, key, nm in zip(axs, ("err_grasp", "err_release"), ("grasp", "release")):
        for j, (m, c) in enumerate((("label_full", COL[0]), ("label_nogrip", COL[1]), ("level_full", COL[2]))):
            e = np.minimum(np.asarray(res[m]["all"][key]), 80)
            ax.hist(e, bins=np.arange(0, 84, 4), histtype="step", lw=2, color=c, label=m.replace("_", " / "))
        ax.axvline(TOL, color="#52514e", lw=1, ls="--")
        ax.set_xlabel(f"|nearest cut - {nm}| (frames; 80 = none)")
    axs[0].set_ylabel("pick-place demos")
    axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "u1_boundary_error.png"), dpi=140)
    plt.close(fig)
    # 3 unit-reuse matrix
    tasks = sorted({(s["set"], s["task"]) for s in new}, key=lambda x: (x[0] != "l90", x[1]))
    units = sorted({s["unit"] for s in new})
    M = np.zeros((len(units), len(tasks)))
    for s in new:
        M[units.index(s["unit"]), tasks.index((s["set"], s["task"]))] += 1
    order = np.argsort(-(M > 0).sum(1))
    fig, ax = plt.subplots(figsize=(12, 0.18 * len(units) + 1.4))
    ax.imshow((M[order] > 0), aspect="auto", cmap="Blues", interpolation="nearest")
    ax.set_yticks(range(len(units)))
    ax.set_yticklabels([f"u{units[i]} ({int((M[i] > 0).sum())})" for i in order], fontsize=6)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([f"{'' if s == 'l90' else 'L10-'}{t}" for s, t in tasks], fontsize=5, rotation=90)
    ax.set_title(f"{len(units)} discovered units x {len(tasks)} tasks (new segments; a cell = the task uses the unit)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "u1_reuse.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main("--record" in sys.argv)
