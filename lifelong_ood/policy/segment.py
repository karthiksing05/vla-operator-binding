"""UNSUPERVISED SEGMENTATION of a demonstration by the frozen policy's own conditional fields (plan U1, docs/UNSUPERVISED.md).

Input: a cost matrix C [F, L] -- the frozen policy's flow-matching loss on the chunk at frame f under held instruction l (the
policy's conditional field l), computed without the demo's own instruction and without reading the gripper.  Output: segments,
each a frame range with the label of the held field that explains it best.

    label_viterbi(C, lam)        -> segments minimising sum_f C[f, label(f)] + lam per segment (an optimal partition: Viterbi
                                    over labels with a switch cost; same-label breaks never pay, so segments = label runs)
    level_partition(y, lam)      -> optimal partition of a 1-D profile into piecewise-constant pieces (squared error + lam
                                    per piece), for jumps of the best field's loss with no change of label
    boundaries(segs)             -> the interior cut frames
    gripper_truth(actions)       -> the demo's sustained grasps and releases (closed >= RELEASE_RUN frames), for SCORING only
    match(pred, truth, tol)      -> nearest-boundary errors and precision / recall at a tolerance

Pure numpy; no policy import, so it runs anywhere.
"""
from __future__ import annotations

import numpy as np

from .primitives import RELEASE_RUN


def normalise(C):
    """Per-demo scale: divide by the median over frames of the best field's loss, so a penalty lam is in units of
    'frames of typical best-field loss' and one lam serves every demo."""
    C = np.asarray(C, np.float64)
    return C / max(float(np.median(C.min(1))), 1e-12)


def label_viterbi(C, lam):
    """C [F, L] per-frame costs.  -> [(lo, hi, label)] frame-index ranges (hi exclusive) minimising total cost + lam x segments.
    Exact: dynamic programming over (frame, label) where changing label costs lam."""
    C = np.asarray(C, np.float64)
    F, L = C.shape
    D = np.empty((F, L))
    back = np.zeros((F, L), np.int64)
    D[0] = C[0] + lam
    for f in range(1, F):
        j = int(np.argmin(D[f - 1]))
        stay, switch = D[f - 1], D[f - 1, j] + lam
        take = switch < stay
        back[f] = np.where(take, j, np.arange(L))
        D[f] = np.where(take, switch, stay) + C[f]
    lab = np.empty(F, np.int64)
    lab[-1] = int(np.argmin(D[-1]))
    for f in range(F - 1, 0, -1):
        lab[f - 1] = back[f, lab[f]]
    return runs(lab)


def runs(lab):
    out, lo = [], 0
    for f in range(1, len(lab) + 1):
        if f == len(lab) or lab[f] != lab[lo]:
            out.append((lo, f, int(lab[lo])))
            lo = f
    return out


def level_partition(y, lam, min_len=2):
    """Optimal partition of y [F] into pieces of at least min_len, cost = sum of squared deviations from each piece's mean +
    lam per piece.  O(F^2).  -> [(lo, hi, piece_index)]."""
    y = np.asarray(y, np.float64)
    F = len(y)
    s1 = np.concatenate([[0], np.cumsum(y)])
    s2 = np.concatenate([[0], np.cumsum(y * y)])

    def sse(a, b):
        n = b - a
        return s2[b] - s2[a] - (s1[b] - s1[a]) ** 2 / n

    best = np.full(F + 1, np.inf)
    best[0] = 0.0
    arg = np.zeros(F + 1, np.int64)
    for b in range(min_len, F + 1):
        for a in range(0, b - min_len + 1):
            if not np.isfinite(best[a]):
                continue
            c = best[a] + sse(a, b) + lam
            if c < best[b]:
                best[b], arg[b] = c, a
    cuts, b = [], F
    while b > 0:
        cuts.append((int(arg[b]), b))
        b = int(arg[b])
    return [(lo, hi, i) for i, (lo, hi) in enumerate(reversed(cuts))]


def boundaries(segs):
    return [lo for lo, _, _ in segs[1:]]


def gripper_truth(actions, run=RELEASE_RUN):
    """-> (grasps, releases): frames where a closed run of >= run frames starts / ends.  Scoring only."""
    c = np.asarray(actions)[:, 6] > 0
    g, r, n = [], [], 0
    for i, x in enumerate(c):
        if x:
            n += 1
        else:
            if n >= run:
                g.append(i - n)
                r.append(i)
            n = 0
    if n >= run:
        g.append(len(c) - n)
    return g, r


def match(pred, truth, tol):
    """pred, truth: frame lists.  -> dict(err = per truth event the distance to the nearest predicted boundary (inf if none),
    hit = events within tol, precision = predicted boundaries within tol of some event)."""
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    err = [float(np.min(np.abs(pred - t))) if len(pred) else float("inf") for t in truth]
    near = [bool(len(truth) and np.min(np.abs(truth - p)) <= tol) for p in pred]
    return dict(err=err, recall=float(np.mean([e <= tol for e in err])) if len(err) else None,
                precision=float(np.mean(near)) if len(near) else None, n_pred=int(len(pred)), n_true=int(len(truth)))
