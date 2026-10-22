"""Tests for lifelong_ood/compose (numpy only, small sizes; the full measurements are experiments/s0_synth.py)."""
import copy
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.setrecursionlimit(100000)

from lifelong_ood.compose.gcobweb import GCobweb                                   # noqa: E402
from lifelong_ood.compose.generate import context_fn_obs, generate, recombination_pool  # noqa: E402
from lifelong_ood.compose.learn import calibrate_content, calibrate_windows, discover, learn          # noqa: E402
from lifelong_ood.compose.parse import parse                                       # noqa: E402
from lifelong_ood.compose.store import Config, Store, poe, submodular_cover        # noqa: E402
from lifelong_ood.compose.synth import D, KOBS, Domain, Env                        # noqa: E402


# ---- gcobweb -----------------------------------------------------------------------------------------
def _blobs(n=60, seed=0):
    r = np.random.default_rng(seed)
    C = np.array([[0, 0], [5, 5], [-5, 5]], float)
    X = np.concatenate([c + 0.3 * r.standard_normal((n, 2)) for c in C])
    y = np.repeat(np.arange(3), n)
    p = r.permutation(len(X))
    return X[p], y[p]


def test_gcobweb_counts_and_means_are_exact():
    X, _ = _blobs()
    t = GCobweb(2, prior_var=0.01)
    for i, x in enumerate(X):
        t.ifit(x, payload=i)
    assert t.root.count == len(X)
    assert np.allclose(t.root.mean, X.mean(0))
    assert np.allclose(t.root.msq / t.root.count, X.var(0))
    # every node's count equals the number of instances stored below it, and children sum to the parent
    for n in t.nodes():
        assert len(list(n.all_members())) == n.count
        if n.children:
            assert sum(c.count for c in n.children) == n.count
            assert np.allclose(sum(c.count * c.mean for c in n.children) / n.count, n.mean)


def test_gcobweb_recovers_clusters_and_categorizes():
    X, y = _blobs()
    t = GCobweb(2, prior_var=0.01)
    for i, x in enumerate(X):
        t.ifit(x, payload=int(y[i]))
    # some node per cluster holds >= 90% of it with >= 95% purity
    for k in range(3):
        ok = False
        for n in t.nodes():
            m = list(n.all_members())
            if m and m.count(k) >= 0.9 * 60 and m.count(k) >= 0.95 * len(m):
                ok = True
        assert ok, k
    # best-first categorisation of a fresh point lands below the root, in its cluster
    node, post = t.categorize(np.array([5.1, 4.9]))
    assert node is not t.root and post <= 0
    m = list(node.all_members())
    assert np.mean(np.array(m) == 1) > 0.9
    # the basic level is an ancestor on the path
    b = node.basic()
    assert b in node.path()


def test_gcobweb_operators_all_fire():
    """Incorporate (best), create (new), merge and split all occur on clustered data."""
    r = np.random.default_rng(3)
    X = np.concatenate([c + 0.3 * r.standard_normal((80, 2)) for c in r.uniform(-8, 8, (6, 2))])
    X = X[r.permutation(len(X))]
    t = GCobweb(2, prior_var=0.01, seed=1)
    for x in X:
        t.ifit(x)
    for op in ("best", "new", "merge", "split", "fringe"):
        assert t.ops[op] > 0, (op, dict(t.ops))
    assert len(t.redirect) == t.ops["split"]


def test_gcobweb_nominal_bags_and_exact_match():
    t = GCobweb(0, alpha=1e-4)
    for _ in range(5):
        t.ifit(np.zeros(0), {"a": "x", "b": {"p": 0.5, "q": 0.5}})
    for _ in range(5):
        t.ifit(np.zeros(0), {"a": "y", "b": "r"})
    assert t.root.count == 10
    # exact matches accumulate on one leaf per distinct instance
    leaves = [n for n in t.nodes() if not n.children]
    assert sorted(n.count for n in leaves) == [5, 5]
    n1, _ = t.categorize(np.zeros(0), {"a": "x", "b": "p"})
    n2, _ = t.categorize(np.zeros(0), {"a": "y", "b": "r"})
    assert n1 is not n2
    # an unseen value is far less likely than a seen one
    lp_seen = t.root.log_prob(np.zeros(0), t.encode_nom({"a": "x"}, grow=False))
    lp_new = t.root.log_prob(np.zeros(0), t.encode_nom({"a": "zzz"}, grow=False))
    assert lp_seen > lp_new + 5


def test_gcobweb_flat_is_one_level():
    X, _ = _blobs()
    t = GCobweb(2, prior_var=0.01, flat=True)
    for x in X:
        t.ifit(x)
    assert all(not c.children for c in t.root.children)
    assert 2 <= len(t.root.children) <= 40


# ---- PoE / coverage ----------------------------------------------------------------------------------
def test_poe_closed_form_and_submodular_selection():
    M = np.array([[0.0, 0.0, 9.0, 9.0], [9.0, 9.0, 1.0, 1.0], [5.0, 5.0, 5.0, 5.0]])
    V = np.full_like(M, 0.1)
    x = np.array([0.0, 0.0, 1.0, 1.0])            # first half from expert 0, second from expert 1
    S, Ls = submodular_cover(x, M, V, K=3, gain_min=1.0)
    assert {0, 1} <= set(S)
    mean, var, w = poe(Ls, M[S], V[S], tau=0.5)
    assert np.allclose(mean, x, atol=1e-3)
    assert np.all(var <= 0.1 + 1e-9)
    # equal experts with equal weights: PoE equals the expert
    mean2, var2, _ = poe(np.zeros((2, 4)), np.ones((2, 4)), np.full((2, 4), 0.2))
    assert np.allclose(mean2, 1.0) and np.allclose(var2, 0.2)


# ---- end to end on a small synthetic store -----------------------------------------------------------
@pytest.fixture(scope="module")
def small():
    dom = Domain(0)
    st = Store(D, KOBS, Config(seed=0))
    train = [dom.demo(dom.sample_task(dom.train_combos)) for _ in range(10)]
    for d in train:
        discover(st, d.Z, d.O)
    cal = [dom.demo(dom.sample_task(dom.train_combos)) for _ in range(3)]
    st.calibrate_cover(np.concatenate([c.Z for c in cal]))
    calibrate_windows(st, [(c.Z, c.O) for c in cal])
    learned = [learn(st, d.Z, d.O, label_fn=d.label)["parse"] for d in train]
    calibrate_content(st, [(c.Z, c.O) for c in cal])
    return dom, st, train, learned


def test_synth_structure():
    dom = Domain(1)
    d = dom.demo([(0, 1), (2, 3)])
    assert len(d.units) == 8 and d.Z.shape == (len(d.O), D) and d.O.shape[1] == KOBS
    assert all(5 <= e - s <= 15 for _, _, s, e in d.units)
    assert [u[0] for u in d.units[:4]] == ["approach", "grasp", "carry", "release"]
    assert d.unit_seq()[2] == ("carry", (0, 1))
    assert set(dom.heldout).isdisjoint(dom.train_combos)
    assert all(o != dom.new_obj for o, _ in dom.train_combos)


def test_parse_boundaries_in_distribution(small):
    dom, st, _, _ = small
    F = []
    for _ in range(3):
        d = dom.demo(dom.sample_task(dom.train_combos))
        p = parse(st, d.Z, d.O, gated=True)
        true, pred = d.boundaries(), p.boundaries()
        hit = sum(any(abs(b - t) <= 2 for b in pred) for t in true)
        F.append(hit / len(true))
        assert p.new_frac() < 0.1
    assert np.mean(F) > 0.8


def test_grec_uses_stored_elements_and_with(small):
    dom, st, _, _ = small
    s2 = copy.deepcopy(st)
    h = dom.heldout[0]
    d = dom.demo([h])
    p = parse(s2, d.Z, d.O, gated=True)
    assert p.new_frac() < 0.1
    carry = [w for w in p.windows if any(u[0] == "carry" for u in d.label(w.span))]
    assert carry and any(w.kind == "with" for w in carry)
    r = learn(s2, d.Z, d.O, label_fn=d.label)
    assert r["parse"].root is not None
    env = Env(dom, [h])
    out, trace = generate(s2, r["parse"].root, context_fn_obs(env.obs), emit_fn=lambda el: env.step(el.code))
    assert out and all(e.kind in ("prim", "with") for e in out)
    seq = []
    for u in env.executed:
        if not seq or seq[-1] != u:
            seq.append(u)
    assert seq == d.unit_seq()


def test_gnew_detects_creates_reuses(small):
    dom, st, _, _ = small
    s2 = copy.deepcopy(st)
    o = dom.new_obj
    d1 = dom.demo([(o, 0)])
    p1 = parse(s2, d1.Z, d1.O, gated=True)
    assert p1.new_frac() > 0.4
    r = learn(s2, d1.Z, d1.O, label_fn=d1.label)
    assert len(r["created"]) > 0
    d2 = dom.demo([(o, 2)])
    p2 = parse(s2, d2.Z, d2.O, gated=True)
    assert p2.new_frac() < p1.new_frac() / 2


def test_recombination_pool_and_context_callback(small):
    dom, st, _, learned = small
    root = next(p.root for p in learned if p.root is not None)
    pool = recombination_pool(st, root)
    assert root in pool or len(pool) >= 1
    calls = []

    def fn(store, cand, emitted):
        calls.append(cand.key)
        return 0.0

    out, trace = generate(st, root, fn)
    assert calls and out


def test_legacy_compose_reexport():
    import lifelong_ood.compose as C
    assert callable(C.poe_teacher) and callable(C.submodular_select)
