"""Tests for lifelong_ood/protos (store + relation).  The relation tests are numpy; the Gaussian / score_ops tests use torch on
the CPU, so run the file through SLURM (slurm/jobs/cpu.sbatch), never on the login node."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lifelong_ood.protos import relation as R                                       # noqa: E402
from lifelong_ood.protos import store as S                                          # noqa: E402

H, A, L, P = 10, 7, 3, 4
D = H * A


def _synthetic(W, n_tasks=12, per_task=5, noise=0.0, seed=0, roles=3):
    """Linear superposition: Y = sum_r W_r[cell] X_r (+ noise), X_r random per observation."""
    rng = np.random.default_rng(seed)
    st = R.Stats()
    for t in range(n_tasks):
        for _ in range(per_task):
            X = rng.standard_normal((roles, L, P, D))
            ph = int(rng.integers(0, 2))
            Y = np.zeros((L, P, D))
            for l in range(L):
                w = R.weights_at(W, l, ph, L, H, A)                     # [R, D]
                Y[l] = (w[:, None, :] * X[:, l]).sum(0)
            Y += noise * rng.standard_normal(Y.shape)
            st.add(t, R.obs_stats(R.r0_features(X), Y, ph, H, A))
    return st


def test_r0_recovers_known_weights():
    rng = np.random.default_rng(1)
    W = rng.uniform(-0.5, 1.5, (R.n_cells(L), 3))
    st = _synthetic(W, noise=0.0)
    What = R.solve(st.total(), lam=1e-8)
    assert np.abs(What - W).max() < 1e-5
    # with noise: recovered up to the noise, and the held-out explained fraction is high
    st = _synthetic(W, noise=0.05, seed=2)
    What = R.solve(st.total(), lam=1e-4)
    assert np.abs(What - W).max() < 0.05
    _, ex = R.leave_task_out(st, lam=1e-4)
    assert ex > 0.99


def test_r0_heldout_sse_matches_direct():
    """The sufficient-statistics error equals the error computed on the arrays."""
    rng = np.random.default_rng(3)
    W = rng.normal(1, 0.3, (R.n_cells(L), 2))
    X = rng.standard_normal((2, L, P, D))
    Y = rng.standard_normal((L, P, D))
    st = R.obs_stats(X, Y, 1, H, A)
    s, t = R.sse(W, st)
    pred = np.stack([R.compose(R.weights_at(W, l, 1, L, H, A), np.zeros((P, D)), X[:, l]) for l in range(L)])
    assert np.isclose(s.sum(), ((Y - pred) ** 2).sum())
    assert np.isclose(t.sum(), (Y ** 2).sum())


def test_r1_context_recovered():
    """R1 recovers a weight that depends linearly on a context feature."""
    rng = np.random.default_rng(4)
    st = R.Stats()
    for t in range(10):
        for _ in range(6):
            z = rng.standard_normal(1)
            X = rng.standard_normal((1, L, P, D))
            Y = (0.5 + 0.4 * z[0]) * X[0]
            st.add(t, R.obs_stats(R.r1_features(X, z), Y, 0, H, A))
    W = R.solve(st.total(), lam=1e-8)
    used = st.total()[3] > 0
    assert np.allclose(W[used, 0], 0.5, atol=1e-6) and np.allclose(W[used, 1], 0.4, atol=1e-6)


def test_one_hot_reproduces_single_condition_exactly():
    rng = np.random.default_rng(5)
    e_null = rng.standard_normal((P, D)).astype(np.float32)
    e = rng.standard_normal((3, P, D)).astype(np.float32)
    for r in range(3):
        w = np.zeros((3, D), np.float32)
        w[r] = 1.0
        assert np.array_equal(R.compose(w, e_null, e), e[r])
    assert np.array_equal(R.compose(np.zeros((3, D), np.float32), e_null, e), e_null)


def test_readout_labels():
    roles = ["obj", "goal"]
    W = np.zeros((R.n_cells(L), 2))
    T = W.reshape(3, 2, 2, L, 2)
    T[0, :, 0, :, 0], T[0, :, 1, :, 1] = 1.0, 1.0                       # pos: obj before the grasp, goal after -> THEN
    T[1, :, :, :, 0], T[1, :, :, :, 1] = 1.0, 1.0                       # rot: both on -> AND
    T[2, :, :, :, 0], T[2, :, :, :, 1] = 0.5, 0.5                       # grip: shares summing to 1 -> OR
    out = R.readout(T.reshape(-1, 2), roles, L)
    assert out["obj"]["pos"]["label"] == "THEN(pre)" and out["goal"]["pos"]["label"] == "THEN(post)"
    assert out["obj"]["rot"]["label"] == "AND" and out["goal"]["grip"]["label"] == "OR-share"


def test_bootstrap_stability_high_on_clean_data():
    rng = np.random.default_rng(6)
    W = rng.uniform(0, 1.2, (R.n_cells(L), 3))
    st = _synthetic(W, noise=0.05, seed=7)
    b = R.bootstrap_stability(st, 1e-4, B=20, roles=["a", "b", "c"], L=L)
    assert b["median"] > 0.99


# ── the flow-path Gaussian (torch) ─────────────────────────────────────────────────────────────────────────────────
torch = pytest.importorskip("torch")


def test_flow_gaussian_eps_tweedie_and_score():
    g = torch.Generator().manual_seed(0)
    d = 7
    mu = torch.randn(d, generator=g, dtype=torch.float64)
    var = torch.rand(d, generator=g, dtype=torch.float64) * 0.5 + 0.05
    for t in (0.1, 0.4, 0.7, 0.95):
        x = torch.randn(5, d, generator=g, dtype=torch.float64).requires_grad_(True)
        s = (1 - t) ** 2 * var + t ** 2
        logp = (-0.5 * (x - (1 - t) * mu) ** 2 / s).sum()
        (grad,) = torch.autograd.grad(logp, x)
        e = S.flow_gaussian_eps(x.detach(), t, mu, var)
        assert torch.allclose(-e / t, grad, atol=1e-10)                                  # score = -E[eps|x] / t
        a = S.flow_gaussian_a(x.detach(), t, mu, var)
        assert torch.allclose(a, (x.detach() - t * e) / (1 - t), atol=1e-10)             # Tweedie on the flow path
        assert torch.allclose(S.flow_gaussian_score(x.detach(), t, mu, var), grad, atol=1e-10)


def test_flow_gaussian_eps_monte_carlo():
    """E[eps | x_t] is linear in x_t for a Gaussian: the regression slope of eps on x_t (per dim) is t / s."""
    g = torch.Generator().manual_seed(1)
    n, d, t = 400000, 3, 0.35
    mu = torch.tensor([0.5, -1.0, 0.2], dtype=torch.float64)
    var = torch.tensor([0.1, 0.4, 0.02], dtype=torch.float64)
    a = mu + var.sqrt() * torch.randn(n, d, generator=g, dtype=torch.float64)
    e = torch.randn(n, d, generator=g, dtype=torch.float64)
    x = t * e + (1 - t) * a
    xc = x - x.mean(0)
    slope = (xc * (e - e.mean(0))).sum(0) / (xc * xc).sum(0)
    s = (1 - t) ** 2 * var + t ** 2
    assert torch.allclose(slope, t / s, rtol=0.02)
    pred = S.flow_gaussian_eps(x, t, mu, var)
    resid = e - pred
    assert float(resid.mean(0).abs().max()) < 5e-3                                       # unbiased


class _GaussBackbone:
    """A backbone whose field is the flow-path Gaussian: score_ops must find mu (Tweedie) and var (Laplace)."""
    path = "flow"
    device = "cpu"

    def __init__(self, mu, var):
        self.mu, self.var, self.dim = mu, var, len(mu)

    def eps(self, X, t, **_):
        X = torch.as_tensor(X, dtype=torch.float32)
        return S.flow_gaussian_eps(X, float(t), torch.as_tensor(self.mu), torch.as_tensor(self.var))


def test_score_ops_on_flow_gaussian():
    from lifelong_ood.backbones.score_ops import laplace_diag, mode_ascent
    rng = np.random.default_rng(2)
    mu = rng.normal(0, 0.5, D).astype(np.float32)
    var = rng.uniform(0.02, 0.3, D).astype(np.float32)
    bk = _GaussBackbone(mu, var)
    for u in (0.2, 0.55):
        x0s = rng.normal(0, 1, (4, D)).astype(np.float32)
        xt, x0 = mode_ascent(bk, None, u, iters=400, lr=0.02, x_init=x0s, n_starts=4)
        assert np.abs(x0 - mu[None]).max() < 0.03
        ext = laplace_diag(bk, xt, u, probes=8, var_floor=1e-5, var_cap=4.0)
        assert np.allclose(ext, var[None], rtol=0.05, atol=2e-3)


def test_store_hierarchy_edges():
    conds = [S.Condition("null", "null", ""), S.Condition("instr", "instr", "put the bowl on the plate"),
             S.Condition("obj:named", "obj", "the bowl"), S.Condition("goal:own", "goal", "on the plate")]
    st = S.Store(conds, levels=(0.1, 0.3))
    for c in conds:
        for u in (0.1, 0.3):
            st.add(c.name, u, np.zeros((2, D)), np.ones((2, D)))
    E = st.edges()
    assert (S.Key("instr", 0.1), S.Key("instr", 0.3), "u") in E
    assert (S.Key("instr", 0.1), S.Key("obj:named", 0.1), "c") in E
    assert (S.Key("obj:named", 0.3), S.Key("null", 0.3), "c") in E
    assert not any(k == S.Key("null", 0.1) and kind == "c" for k, _, kind in E)
