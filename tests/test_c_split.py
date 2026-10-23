"""LIBERO-90-C split checks (experiments/c_split.py, docs/L90C.md): atom coverage, the CD bins, no held-out compound in training,
the part set's atoms absent from training, and the sealed-task guard.  Reads only results/l90c/split.json (never the sealed file).
"""
import collections
import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT = os.path.join(ROOT, "results", "l90c", "split.json")
pytestmark = pytest.mark.skipif(not os.path.exists(SPLIT), reason="split.json not built (python experiments/c_split.py)")


def _cs():
    spec = importlib.util.spec_from_file_location("c_split", os.path.join(ROOT, "experiments", "c_split.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def ctx():
    from lifelong_ood import l90c
    from lifelong_ood.policy import factors as F
    cs = _cs()
    s = l90c.load_split()
    tasks = {t.tid: t for t in F.load(cs.SUITES)}
    return cs, s, tasks


def _dev_test(s):
    return [t for b in s["bins"] for t in s["dev"][b]]


def test_partition(ctx):
    cs, s, tasks = ctx
    train, test, part = set(s["train"]), set(_dev_test(s)), set(s["dev"]["part"])
    assert not train & test and not train & part and not test & part
    assert all(t.startswith("libero_90:") for t in test | part)
    assert set(s["train_l90"]) <= train and set(s["train_suite_additions"]) <= train
    n_l90 = len(s["train_l90"]) + len(test) + len(part) + sum(s["n_sealed"].values()) + len(s["no_demos"])
    assert n_l90 == 90


def test_atom_coverage(ctx):
    """Every atom of every dev test task is in >= 2 LIBERO-90 training tasks."""
    cs, s, tasks = ctx
    cnt = collections.Counter()
    for tid in s["train_l90"]:
        cnt.update(set(cs.atoms(tasks[tid])))
    for tid in _dev_test(s):
        for a in cs.atoms(tasks[tid]):
            assert cnt[a] >= 2, (tid, a, cnt[a])
    assert s["checks"]["min_train_count_of_test_atoms"] >= 2


def test_bins(ctx):
    """Each dev test task's bin is the one its unseen-compound fraction gives against the training cells; recorded CDs agree."""
    cs, s, tasks = ctx
    seen = set()
    for tid in s["train_l90"]:
        seen |= set(cs.compounds(tasks[tid]))
    for b in s["bins"]:
        for tid in s["dev"][b]:
            u = cs.unseen_frac(tasks[tid], seen)
            assert cs.grade(u) == b, (tid, u, b)
            assert abs(s["dev_per_task"][tid]["u"] - u) < 1e-3
    for b in s["bins"]:
        assert s["checks"]["bin_n"][b] == len(s["dev"][b]) + s["n_sealed"][b]
    cds = [s["checks"]["bin_cd"][b] for b in s["bins"]]
    assert cds == sorted(cds), cds                              # the bins are ordered in compound divergence


def test_no_heldout_compound_in_training(ctx):
    """No compound that a test or part task holds out (absent from the LIBERO-90 training cells) is in ANY training task,
    suite additions included; the part atoms are in no training task."""
    cs, s, tasks = ctx
    seen = set()
    for tid in s["train_l90"]:
        seen |= set(cs.compounds(tasks[tid]))
    held = set()
    for tid in _dev_test(s) + s["dev"]["part"]:
        held |= {k for k in cs.compounds(tasks[tid]) if k not in seen}
    assert {str(k) for k in held} <= set(s["held_compounds"])
    for tid in s["train"]:
        assert not set(cs.compounds(tasks[tid])) & held, tid
        assert not cs.has_part(tasks[tid]), tid
    assert not s["checks"]["held_compounds_in_training"] and not s["checks"]["part_atoms_in_training"]


def test_part_set_complete(ctx):
    """Every LIBERO-90 task naming a part (object, target or verb) is in the part set (dev or sealed), never in training."""
    cs, s, tasks = ctx
    n_part = sum(1 for t in tasks.values() if t.suite == "libero_90" and cs.has_part(t))
    assert len(s["dev"]["part"]) + s["n_sealed"]["part"] == n_part


def test_sealed_guard(ctx):
    from lifelong_ood import l90c
    cs, s, tasks = ctx
    sealed = [tid for tid in tasks if tid.startswith("libero_90:") and l90c.is_sealed(tid)]
    assert len(sealed) == sum(s["n_sealed"].values())
    with pytest.raises(l90c.SealedError):
        l90c.check_task(sealed[0])
    with pytest.raises(l90c.SealedError):
        l90c.load_sealed()
    with pytest.raises(l90c.SealedError):
        l90c.load_sealed(prereg="not_a_run_id")
    assert l90c.check_task(s["train"][0]) == s["train"][0]
    for tid in _dev_test(s):
        assert not l90c.is_sealed(tid)
