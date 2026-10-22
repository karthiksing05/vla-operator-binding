"""THE RELATION R, distilled from the VLA (framework v3, docs/FRAMEWORK.md section 2; plan v4 P3).  numpy only.

On training instructions the policy composes correctly, so R maps its part fields to its whole field at the SAME point x_t:

    eps(x | o, instruction) - eps(x | o, null)  ~  sum_r  W_r[cell] * (eps(x | o, part_r) - eps(x | o, null))

R0  closed-form per-role weights.  A cell is (dimension group pos / rot / grip) x (chunk half) x (phase pre / post grasp) x
    (level u); every (point, action dim) pair of an observation falls in exactly one cell, so the least-squares problem splits
    into one R x R ridge per cell.  Fitting keeps only SUFFICIENT STATISTICS per task (A = sum x x', b = sum x y, yy = sum y^2),
    so leave-task-out refits and held-out errors are exact subtractions: SSE = yy - 2 w'b + w'Aw on the held-out task's stats.
R1  R0 with context-dependent weights (a GLM): W_r[cell] + sum_k beta_rk[cell] z_k, z = standardised context per observation
    (gripper opening, time since the last gripper event); t enters through the level cells.
R2  a stub (a bounded set-transformer merger, plan v4: only if R0 / R1 leave a gap).

Roles are the rows of W (templates over group x half x phase x level).  The readout names the mode of combination per role and
group from the weights: AND (weight ~ 1 in both phases: a product of experts in eps), THEN (~ 1 in one phase, ~ 0 in the other:
a switch at the grasp), OR (fractional weights whose sum over the roles is ~ 1: a convex mix), else graded.

compose() is the affine form (1 - sum_r w_r) eps_null + sum_r w_r eps_r: with a one-hot weight it returns that condition's field
bit for bit.
"""
from __future__ import annotations

import numpy as np

GROUPS = ("pos", "rot", "grip")
PHASES = ("pre", "post")


# ── cells ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def dim_cells(H=10, A=7):
    """-> [H*A] (group, half) index per flat action dim, group-major: gh = group * 2 + half."""
    d = np.arange(H * A)
    a, h = d % A, d // A
    g = np.where(a < 3, 0, np.where(a < 6, 1, 2))
    return g * 2 + (h >= H // 2).astype(int)


def n_cells(L):
    return 3 * 2 * 2 * L


def cell_of(gh, phase, level, L):
    """(group-half, phase, level) -> cell id; layout [group, half, phase, level]."""
    return (gh * 2 + phase) * L + level


# ── features ───────────────────────────────────────────────────────────────────────────────────────────────────────
def r0_features(Xr, ctx=None):
    """Xr [R, L, P, D] role deltas -> features [R, L, P, D] (identity)."""
    return np.asarray(Xr, np.float64)


def r1_features(Xr, ctx):
    """Role deltas times [1, z_1 .. z_K] -> [R * (1 + K), L, P, D] (role-major: r * (1+K) + k)."""
    Xr = np.asarray(Xr, np.float64)
    z = np.concatenate([[1.0], np.asarray(ctx, np.float64)])
    return (Xr[:, None] * z[None, :, None, None, None]).reshape(-1, *Xr.shape[1:])


# ── sufficient statistics ──────────────────────────────────────────────────────────────────────────────────────────
def obs_stats(F, Y, phase, H=10, A=7):
    """One observation.  F [Fd, L, P, D] features, Y [L, P, D] target, phase 0 / 1.
    -> (A [cells, Fd, Fd], b [cells, Fd], yy [cells], cnt [cells])."""
    F = np.asarray(F, np.float64)
    Y = np.asarray(Y, np.float64)
    Fd, L = F.shape[0], F.shape[1]
    gh = dim_cells(H, A)
    nc = n_cells(L)
    Am, bm, yy, cnt = np.zeros((nc, Fd, Fd)), np.zeros((nc, Fd)), np.zeros(nc), np.zeros(nc)
    for l in range(L):
        for j in range(6):
            m = gh == j
            c = cell_of(j, int(phase), l, L)
            Fx = F[:, l][:, :, m].reshape(Fd, -1)
            y = Y[l][:, m].reshape(-1)
            Am[c] += Fx @ Fx.T
            bm[c] += Fx @ y
            yy[c] += y @ y
            cnt[c] += y.size
    return Am, bm, yy, cnt


class Stats:
    """Per-task sums of obs_stats.  tasks: list of ids; S[task] = (A, b, yy, cnt)."""

    def __init__(self):
        self.S = {}

    def add(self, task, st):
        if task in self.S:
            self.S[task] = tuple(x + y for x, y in zip(self.S[task], st))
        else:
            self.S[task] = tuple(np.array(x, copy=True) for x in st)

    def total(self, tasks=None, weights=None):
        ts = list(self.S) if tasks is None else list(tasks)
        w = np.ones(len(ts)) if weights is None else np.asarray(weights, float)
        out = None
        for t, wt in zip(ts, w):
            if wt == 0:
                continue
            s = tuple(wt * x for x in self.S[t])
            out = s if out is None else tuple(x + y for x, y in zip(out, s))
        return out


# ── fitting and scoring ────────────────────────────────────────────────────────────────────────────────────────────
def solve(st, lam=1e-2, prior=None):
    """Ridge per cell: w_c = argmin |y - F'w|^2 + lam_c |w - prior|^2, lam_c = lam * tr(A_c) / Fd.  -> W [cells, Fd]."""
    A, b, _, cnt = st
    nc, Fd = b.shape
    W = np.zeros((nc, Fd))
    p0 = np.zeros(Fd) if prior is None else np.asarray(prior, float)
    for c in range(nc):
        if cnt[c] == 0:
            W[c] = p0
            continue
        lc = lam * max(np.trace(A[c]), 1e-12) / Fd
        W[c] = np.linalg.solve(A[c] + lc * np.eye(Fd), b[c] + lc * p0)
    return W


def sse(W, st):
    """Held-out error from sufficient statistics.  -> (sse [cells], sst [cells])."""
    A, b, yy, _ = st
    s = yy - 2 * np.einsum("cf,cf->c", W, b) + np.einsum("cf,cfg,cg->c", W, A, W)
    return s, yy


def explained(W, st):
    s, t = sse(W, st)
    return float(1.0 - s.sum() / max(t.sum(), 1e-12))


def leave_task_out(stats, lam=1e-2, prior=None):
    """-> dict(task -> (W fitted without the task, sse cells, sst cells)) and the pooled held-out explained fraction."""
    tasks = list(stats.S)
    tot = stats.total()
    per, S, T = {}, 0.0, 0.0
    for t in tasks:
        tr = tuple(x - y for x, y in zip(tot, stats.S[t]))
        W = solve(tr, lam, prior)
        s, y = sse(W, stats.S[t])
        per[t] = (W, s, y)
        S += s.sum()
        T += y.sum()
    return per, float(1.0 - S / max(T, 1e-12))


def choose_lam(stats, grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0), prior=None):
    """Leave-task-out choice of the ridge scale on the given tasks.  -> (best lam, {lam: explained})."""
    sc = {lam: leave_task_out(stats, lam, prior)[1] for lam in grid}
    return max(sc, key=sc.get), sc


def nested_lto(stats, grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0), prior=None):
    """Outer leave-task-out with the ridge scale chosen by an inner leave-task-out on the remaining tasks.
    -> (pooled held-out explained fraction, per task dict(explained, lam, W), cells sse / sst summed)."""
    tasks = list(stats.S)
    per, S, T = {}, None, None
    for t in tasks:
        inner = Stats()
        for u in tasks:
            if u != t:
                inner.S[u] = stats.S[u]
        lam, _ = choose_lam(inner, grid, prior)
        W = solve(inner.total(), lam, prior)
        s, y = sse(W, stats.S[t])
        per[t] = dict(explained=float(1 - s.sum() / max(y.sum(), 1e-12)), lam=lam, W=W, sse=s, sst=y)
        S = s if S is None else S + s
        T = y if T is None else T + y
    return float(1 - S.sum() / max(T.sum(), 1e-12)), per, (S, T)


# ── templates, readout, stability ──────────────────────────────────────────────────────────────────────────────────
def templates(W, roles, L):
    """W [cells, R] -> {role: array [group 3, half 2, phase 2, level L]}."""
    T = np.asarray(W).reshape(3, 2, 2, L, -1)
    return {r: T[..., i] for i, r in enumerate(roles)}


def readout(W, roles, L, tol=0.35):
    """Mode of combination per role and dimension group, from the weights averaged over halves and levels per phase.
    -> {role: {group: dict(pre, post, label)}} with label in AND / THEN / OR-share / off / graded."""
    T = templates(W, roles, L)
    out = {}
    for g, gname in enumerate(GROUPS):
        tot = {p: sum(float(T[r][g, :, p].mean()) for r in roles) for p in range(2)}
        for r in roles:
            pre, post = float(T[r][g, :, 0].mean()), float(T[r][g, :, 1].mean())
            near1 = [abs(w - 1) < tol for w in (pre, post)]
            near0 = [abs(w) < tol for w in (pre, post)]
            if all(near1):
                lab = "AND"
            elif (near1[0] and near0[1]) or (near0[0] and near1[1]):
                lab = "THEN(" + ("pre" if near1[0] else "post") + ")"
            elif all(near0):
                lab = "off"
            elif all(0 < w < 1 for w in (pre, post)) and all(abs(tot[p] - 1) < tol for p in range(2)):
                lab = "OR-share"
            else:
                lab = "graded"
            out.setdefault(r, {})[gname] = dict(pre=pre, post=post, label=lab, sum_pre=tot[0], sum_post=tot[1])
    return out


def bootstrap_stability(stats, lam, B=100, seed=0, prior=None, roles=None, L=None):
    """Resample tasks with replacement, refit, correlate each refit's weight vector with the full fit.
    -> dict(median corr, 5th percentile, mean pairwise corr over the first 30 refits, per-role medians)."""
    rng = np.random.default_rng(seed)
    tasks = list(stats.S)
    W0 = solve(stats.total(), lam, prior)
    Ws, cs = [], []
    for _ in range(int(B)):
        pick = rng.integers(0, len(tasks), len(tasks))
        w = np.bincount(pick, minlength=len(tasks)).astype(float)
        Wb = solve(stats.total(tasks, w), lam, prior)
        Ws.append(Wb)
        cs.append(float(np.corrcoef(Wb.ravel(), W0.ravel())[0, 1]))
    pw = [float(np.corrcoef(Ws[i].ravel(), Ws[j].ravel())[0, 1]) for i in range(min(30, B)) for j in range(i + 1, min(30, B))]
    out = dict(median=float(np.median(cs)), p05=float(np.percentile(cs, 5)), pairwise=float(np.mean(pw)), B=int(B))
    if roles is not None:
        out["per_role"] = {r: float(np.median([np.corrcoef(Wb[:, i], W0[:, i])[0, 1] for Wb in Ws])) for i, r in enumerate(roles)}
    return out


# ── applying R ─────────────────────────────────────────────────────────────────────────────────────────────────────
def weights_at(W, level, phase, L, H=10, A=7):
    """W [cells, R] -> per-dim weights [R, H*A] at one level index and phase."""
    gh = dim_cells(H, A)
    c = cell_of(gh, int(phase), int(level), L)
    return np.asarray(W)[c].T


def compose(w, e_null, e_roles):
    """Affine composition (1 - sum_r w_r) e_null + sum_r w_r e_r.  w [R, D] (or [R]); e_null [n, D]; e_roles [R, n, D].
    A one-hot w returns that role's field exactly."""
    w = np.asarray(w, e_null.dtype)
    if w.ndim == 1:
        w = np.repeat(w[:, None], e_null.shape[-1], 1)
    out = (1.0 - w.sum(0))[None] * e_null
    for r in range(len(w)):
        out = out + w[r][None] * e_roles[r]
    return out


class R2:
    """Stub: a bounded set-transformer merger over frozen fields, initialised at R0, trained once leave-task-out (plan v4).
    Built only if R0 / R1 leave a gap."""

    def fit(self, *a, **k):
        raise NotImplementedError("R2 is a stub: build it only if R0 / R1 leave a gap (docs/PLAN_V4.md P3)")

    def __call__(self, *a, **k):
        raise NotImplementedError("R2 is a stub")
