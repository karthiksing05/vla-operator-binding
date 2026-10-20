"""X3 (EgoDex): per-hand grasp / release events from the 3D hand poses, for the unsupervised scheme's contact cuts.

THE DEFINITION (chosen, `contact_events(mode="stops")`): a contact event of a hand is an interior STOP of its wrist, the
classic reach -> grasp -> transport -> release -> retract kinematics.  The wrist speed |dp/dt| (30 Hz, metres/s, frame-free)
is Gaussian-smoothed (sigma = 2 frames); every local minimum whose depth (scipy prominence, relative to the lower of its two
flanking maxima `ref`) leaves it at <= 0.6 * ref, with ref >= 0.2 m/s, is a stop.  Stops alternate grasp, release, grasp, ...
(the hand is taken to start empty).  Each stop also gets the aperture trend across it (thumb tip to the nearer of the index /
middle tips, median of frames +5..+15 minus -15..-5): closing (< 0) agrees with a grasp, opening with a release; the agreement
rate is reported as an independent check of the alternation.  The parameters were chosen on the first 60 / 30 / 20 / 20 test
episodes of basic_pick_place / vertical_pick_place / stack / pick_place_food (the "dev" episodes) and are reported separately
on the remaining ("held") episodes.

THE NEGATIVE VARIANTS (`mode="fixed"/"adaptive"`), kept for the record: hysteresis on the aperture alone.  Fingertip
confidences drop exactly while an object is held (occlusion), and the relaxed hand's aperture (3-8 cm) is the width of most
grasped objects, so aperture thresholds find one grasp/release pair on the used hand in only 13-22% of basic_pick_place
episodes (results/robot/egodex/test_contacts_summary.json).
Validation (eye-free): event counts per task.  Pick-and-place tasks (basic_pick_place, vertical_pick_place, pick_place_food,
stack) should show about one grasp and one release on the hand the metadata names (environment 'hand:right'/'hand:left'), and
finger-only tasks (play_piano, type_keyboard, push_pop_toy) few or none.

    python -I experiments/x3_egodex_contacts.py --root /coc/flash10/ksingara3/robot/data/egodex/test --out results/robot/egodex
"""
from __future__ import annotations

import argparse
import collections
import json
import os

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks

TIPS = ("ThumbTip", "IndexFingerTip", "MiddleFingerTip")
DEV = {"basic_pick_place": 60, "vertical_pick_place": 30, "stack": 20, "pick_place_food": 20}
PICK_PLACE = ("basic_pick_place", "vertical_pick_place", "pick_place_food", "stack")
FINGER_ONLY = ("play_piano", "type_keyboard", "push_pop_toy")


def _fill(x, ok):
    """Linear fill of frames where ok is False (per column); all-missing stays as is."""
    if ok.all() or not ok.any():
        return x
    idx = np.arange(len(x))
    return np.stack([np.interp(idx, idx[ok], x[ok, j]) for j in range(x.shape[1])], 1)


def aperture(f, hand, min_conf=0.3):
    """[T] aperture in metres for hand in {'left','right'}; returns (a, valid_fraction)."""
    P = {t: f[f"transforms/{hand}{t}"][:, :3, 3].astype(np.float64) for t in TIPS}
    ok = np.ones(len(P["ThumbTip"]), bool)
    if "confidences" in f:
        for t in TIPS:
            k = f"confidences/{hand}{t}"
            if k in f:
                ok &= f[k][:] >= min_conf
    for t in TIPS:
        P[t] = _fill(P[t], ok)
    a = np.minimum(np.linalg.norm(P["ThumbTip"] - P["IndexFingerTip"], axis=1),
                   np.linalg.norm(P["ThumbTip"] - P["MiddleFingerTip"], axis=1))
    return median_filter(a, size=5, mode="nearest"), float(ok.mean())


def hysteresis(a, lo, hi, min_frames=6):
    """Boolean closed[T]: closes below lo, opens above hi; then runs shorter than min_frames are flipped."""
    closed = np.zeros(len(a), bool)
    s = bool(a[0] < (lo + hi) / 2)
    for i, v in enumerate(a):
        if s and v > hi:
            s = False
        elif not s and v < lo:
            s = True
        closed[i] = s
    # remove short runs (iteratively, shortest first)
    for _ in range(20):
        edges = np.flatnonzero(np.diff(closed.astype(int))) + 1
        b = np.concatenate([[0], edges, [len(closed)]])
        runs = [(b[i], b[i + 1]) for i in range(len(b) - 1)]
        short = [(e - s0, s0, e) for s0, e in runs if e - s0 < min_frames and not (s0 == 0 or e == len(closed))]
        if not short:
            break
        _, s0, e = min(short)
        closed[s0:e] = ~closed[s0:e]
    return closed


def events(closed):
    d = np.diff(closed.astype(int))
    return dict(grasp=(np.flatnonzero(d == 1) + 1).tolist(), release=(np.flatnonzero(d == -1) + 1).tolist(),
                closed_at_start=bool(closed[0]), closed_at_end=bool(closed[-1]))


def wrist_stops(w, sigma=2.0, ratio=0.6, v_move=0.2):
    """w [T, 3] wrist positions (m) -> (stop frames, smoothed speed [T])."""
    sp = np.r_[0.0, np.linalg.norm(np.diff(w, axis=0), axis=1) * 30.0]
    sp = gaussian_filter1d(sp, sigma)
    v, pr = find_peaks(-sp, prominence=0.0)
    ref = sp[v] + pr["prominences"]
    keep = (ref >= v_move) & (sp[v] <= ratio * ref)
    return v[keep].astype(int).tolist(), sp


def stop_events(f, hand, sigma=2.0, ratio=0.6, v_move=0.2, min_conf=0.0):
    w = f[f"transforms/{hand}Hand"][:, :3, 3].astype(np.float64)
    st, sp = wrist_stops(w, sigma, ratio, v_move)
    a, vf = aperture(f, hand, min_conf)
    trend = []
    for t in st:
        pre, post = a[max(0, t - 15):max(1, t - 4)], a[t + 5:t + 16]
        trend.append(float(np.median(post) - np.median(pre)) if len(pre) and len(post) else 0.0)
    return dict(stops=st, grasp=st[0::2], release=st[1::2], trend=trend, path=float(np.linalg.norm(np.diff(w, axis=0), axis=1).sum()),
                start_moving=bool(sp[:3].mean() > 0.15), end_moving=bool(sp[-3:].mean() > 0.15), valid=vf)


def contact_events(f, mode="adaptive", fixed=(0.045, 0.065), ratio=(0.55, 0.75), min_frames=6, min_conf=0.3):
    """{'left': events, 'right': events} for an open h5py file."""
    if mode == "stops":
        return {h: stop_events(f, h) for h in ("left", "right")}
    out = {}
    for hand in ("left", "right"):
        a, vf = aperture(f, hand, min_conf)
        lo, hi = fixed
        if mode == "adaptive":
            ref = float(np.percentile(a, 95))
            if ref >= 0.05:
                lo, hi = ratio[0] * ref, ratio[1] * ref
        c = hysteresis(a, lo, hi, min_frames)
        out[hand] = dict(events(c), a_ref=float(np.percentile(a, 95)), a_min=float(a.min()), valid=vf, lo=lo, hi=hi)
    return out


def used_hand(env):
    env = (env or "").replace(" ", "")
    if "hand:right" in env:
        return "right"
    if "hand:left" in env:
        return "left"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/coc/flash10/ksingara3/robot/data/egodex/test")
    ap.add_argument("--out", default="results/robot/egodex")
    ap.add_argument("--max-per-task", type=int, default=1000)
    a = ap.parse_args()
    modes = {"stops": dict(mode="stops"), "fixed": dict(mode="fixed"), "adaptive": dict(mode="adaptive"),
             "fixed_tight": dict(mode="fixed", fixed=(0.03, 0.05)), "adaptive_tight": dict(mode="adaptive", ratio=(0.45, 0.65))}
    rows = []
    for task in sorted(os.listdir(a.root)):
        d = os.path.join(a.root, task)
        names = sorted([n for n in os.listdir(d) if n.endswith(".hdf5")], key=lambda n: int(n.split(".")[0]))[: a.max_per_task]
        for n in names:
            try:
                with h5py.File(os.path.join(d, n), "r") as f:
                    T = f["transforms/camera"].shape[0]
                    env = f.attrs.get("environment", "")
                    env = env.decode() if isinstance(env, bytes) else str(env)
                    r = dict(task=task, ep=n[:-5], T=int(T), hand=used_hand(env), llm_type=str(f.attrs.get("llm_type", "")),
                             which=str(f.attrs.get("which_llm_description", "")))
                    for m, kw in modes.items():
                        r[m] = contact_events(f, **kw)
                rows.append(r)
            except Exception as e:
                rows.append(dict(task=task, ep=n, error=repr(e)))
        print(task, len(names), flush=True)
    json.dump(rows, open(os.path.join(a.out, "test_contacts.json"), "w"))

    # summary per task and mode
    summ = {}
    for m in modes:
        per = collections.defaultdict(list)
        for r in rows:
            if "error" in r:
                continue
            e = r[m]
            h = r["hand"]
            g = {k: len(e[k]["grasp"]) for k in ("left", "right")}
            rl = {k: len(e[k]["release"]) for k in ("left", "right")}
            per[r["task"]].append(dict(g=g, r=rl, hand=h,
                                       one_pair_used=(h is not None and g[h] == 1 and rl[h] == 1),
                                       other_idle=(h is not None and g["left" if h == "right" else "right"] == 0)))
        summ[m] = {}
        for t, l in per.items():
            gg = np.array([x["g"]["left"] + x["g"]["right"] for x in l])
            rr = np.array([x["r"]["left"] + x["r"]["right"] for x in l])
            hl = [x for x in l if x["hand"]]
            summ[m][t] = dict(n=len(l), grasps_mean=round(float(gg.mean()), 2), grasps_median=float(np.median(gg)),
                              releases_mean=round(float(rr.mean()), 2),
                              zero_events=round(float((gg + rr == 0).mean()), 2),
                              n_hand_named=len(hl),
                              one_pair_on_used_hand=round(float(np.mean([x["one_pair_used"] for x in hl])), 2) if hl else None,
                              other_hand_idle=round(float(np.mean([x["other_idle"] for x in hl])), 2) if hl else None)
    # the chosen definition: stops on the used hand (named by metadata, else the hand whose wrist travels further)
    st = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if "error" in r:
            continue
        e = r["stops"]
        h = r["hand"] or max(("left", "right"), key=lambda k: e[k]["path"])
        o = "left" if h == "right" else "right"
        split = "dev" if int(r["ep"]) < DEV.get(r["task"], -1) else "held"
        tr = e[h]["trend"]
        st[r["task"]][split].append(dict(n=len(e[h]["stops"]), n_other=len(e[o]["stops"]),
                                         agree=[(t < 0) == (k % 2 == 0) for k, t in enumerate(tr)]))
    summ["stops"] = {}
    for t, sp in st.items():
        summ["stops"][t] = {}
        for split, l in sp.items():
            n = np.array([x["n"] for x in l])
            ag = [y for x in l for y in x["agree"]]
            summ["stops"][t][split] = dict(n=len(l), stops_mean=round(float(n.mean()), 2),
                                            hist={int(k): int(c) for k, c in zip(*np.unique(n, return_counts=True))},
                                            exactly_two=round(float((n == 2).mean()), 2),
                                            other_hand_zero=round(float(np.mean([x["n_other"] == 0 for x in l])), 2),
                                            aperture_agrees=round(float(np.mean(ag)), 2) if ag else None)
    json.dump(summ, open(os.path.join(a.out, "test_contacts_summary.json"), "w"), indent=1)
    print("== stops (chosen)")
    for t in PICK_PLACE + FINGER_ONLY:
        if t in summ["stops"]:
            print(" ", t, summ["stops"][t])
    for m in [k for k in modes if k != "stops"]:
        print("==", m)
        for t in PICK_PLACE + FINGER_ONLY:
            if t in summ[m]:
                print(" ", t, summ[m][t])
        allg = [v["grasps_mean"] for v in summ[m].values()]
        print("  median over tasks of mean grasps/episode:", float(np.median(allg)))


if __name__ == "__main__":
    main()
