"""CALVIN-C split (experiments/c_calvin_split.py, lifelong_ood/calvin_c.py): the rules hold for the rebuilt and the frozen
splits, and the sealed guard refuses reads."""
import importlib.util
import json
import pathlib
from collections import Counter

import pytest

from lifelong_ood import calvin_c as C

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("c_calvin_split", ROOT / "experiments" / "c_calvin_split.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def test_cells():
    assert len(C.CELLS) == 21
    assert sum(c[0] == "lift" for c in C.CELLS) == 9
    assert sum(c[0] == "push" for c in C.CELLS) == 6 and sum(c[0] == "rotate" for c in C.CELLS) == 6
    for c in C.CELLS:
        assert C.parse_cell(C.cell_name(c)) == c
    for t in ("stack_block", "place_in_drawer", "push_into_drawer", "open_drawer", "turn_on_led"):
        assert C.parse_cell(t) is None


def _held_rules(s):
    held, train, part = s["held"], s["train"], s["part"]
    assert len(held) in (6, 7)
    assert set(held) | set(train) == set(C.CELLS) and not set(held) & set(train)
    cnt = Counter(a for c in train for a in C.atoms(c))
    for c in C.CELLS:
        for a in C.atoms(c):
            assert cnt[a] >= 2, a                                   # every atom in >= 2 training cells
    seen = {k for c in train for k in C.compounds(c)}
    fully = [(v, c) for v in C.VERBS for c in C.COLOURS if ("vc", v, c) not in seen]
    assert fully == [(part[1], part[0])]                            # exactly the part pair is new as verb-colour
    for c in held:
        g = sum(k not in seen for k in C.compounds(c))
        assert g == s["grades"][c] and g in (0, 1, 2)
    assert {0, 1, 2} <= set(s["grades"].values())                    # the ladder has every grade
    assert len(s["sealed"]) == round(len(held) / 3)
    dev = [c for c in held if c not in s["sealed"]]
    for g in (0, 1, 2):
        assert any(s["grades"][c] == g for c in dev), g            # every grade keeps a dev cell
    assert any(c[0] == part[1] and c[1] == part[0] for c in dev)    # and so does the part set


def test_build_rules():
    sets = S.build()
    assert set(sets) == {0, 1, 2}
    assert sets[0]["part"] == ("pink", "rotate")
    assert {p[1] for p in (s["part"] for s in sets.values())} == set(C.VERBS)
    assert {p[0] for p in (s["part"] for s in sets.values())} == set(C.COLOURS)
    for s in sets.values():
        _held_rules(s)


def test_build_deterministic():
    a, b = S.build(), S.build()
    for k in a:
        assert a[k]["held"] == b[k]["held"] and a[k]["sealed"] == b[k]["sealed"]


@pytest.mark.parametrize("seed", C.SEEDS)
def test_frozen_matches_rebuild(seed):
    p = C.split_path(seed)
    if not p.exists():
        pytest.skip("split not frozen yet")
    s = C.load_split(seed)
    b = S.build()[seed]
    assert s["train"] == b["train"]
    assert sorted(d["cell"] for d in s["dev"]) == sorted(c for c in b["held"] if c not in b["sealed"])
    assert s["n_sealed"] == len(b["sealed"])
    assert "sealed" not in json.loads(p.read_text(encoding="utf-8"))   # the dev file never lists sealed cells
    # held_out_mask covers dev and sealed cells and nothing else
    mask = C.held_out_mask([C.cell_name(c) for c in C.CELLS] + ["stack_block"], seed)
    assert sum(mask) == len(b["held"]) and not mask[-1]
    for c, m in zip(C.CELLS, mask):
        assert m == (c in b["held"])
        assert C.is_sealed(c, seed) == (c in b["sealed"])


@pytest.mark.parametrize("seed", C.SEEDS)
def test_sealed_guard(seed, monkeypatch):
    if not C.split_path(seed).exists():
        pytest.skip("split not frozen yet")
    monkeypatch.delenv("CALVINC_UNSEAL", raising=False)
    with pytest.raises(C.SealedError):
        C.load_sealed(seed)
    with pytest.raises(C.SealedError):
        C.load_sealed(seed, prereg="not-a-row")
    monkeypatch.setenv("CALVINC_UNSEAL", "not-a-row")
    with pytest.raises(C.SealedError):
        C.load_sealed(seed, prereg="not-a-row")                      # env matches but no m3_prereg row
    b = S.build()[seed]
    monkeypatch.delenv("CALVINC_UNSEAL", raising=False)
    with pytest.raises(C.SealedError):
        C.check_cell(b["sealed"][0], seed)
    dev = [c for c in b["held"] if c not in b["sealed"]][0]
    assert C.check_cell(dev, seed) == dev


def test_overlaps():
    import numpy as np
    idx = np.array([[0, 63], [40, 103], [80, 143], [200, 263]])
    best = S.overlaps(idx, np.array([1]))
    assert list(best) == [24, 64, 24, 0]
