"""X4 Meta-World: the graded split (docs/SPLIT.md rule) of the 50 Meta-World v3 tasks against each training set.

Every task is factored into parts read by rule from its env name (Farama Metaworld v3 names, the description string only
where the name has no verb):
  verb      the motion (press, pull, push, open, close, place, ...);
  object    the thing moved or acted on (button, handle, puck, door, ...; None when only the gripper moves);
  target    the place or receiver (peg, bin, hole, shelf, basket, coffee machine, ...; None if none is named);
  mods      approach modifiers that change the motion (wall, side, topdown, back).

Grades of a held-out task T against a training set S (the parts of S's tasks):
  G0 seen      T is in S;
  G1 relation  every part of T (verb, object, target, each modifier) occurs in S, but never in this combination;
  G4 part      some part of T never occurs in S (the unseen parts are listed).
Meta-World has one scene (the Sawyer tabletop), so G2 (a part seen only in another scene) and G3 (unseen scene) are empty by
construction.  Within G1 the record says whether (verb, object) itself is a training pair (then only a modifier / target is
recombined) or not (the verb and the object are each seen, never together).

Training sets:
  mt50        all 50 tasks: lerobot/metaworld_mt50 (2,500 episodes, 50 per task).  Every public Meta-World checkpoint of our
              families was trained on it with no episode filter: lerobot/smolvla_metaworld and jadechoghari/smolvla_metaworld
              (train_config.json: dataset.repo_id lerobot/metaworld_mt50, episodes null), RLinf/RLinf-Pi0-MetaWorld-SFT and
              RLinf/RLinf-Pi05-MetaWorld-SFT (RLinf pi0_metaworld / pi05_metaworld TrainConfig: repo_id lerobot/metaworld_mt50).
  ml45        Meta-World's ML45 train classes (45); test = bin-picking, box-close, hand-insert, door-lock, door-unlock.
  x4_35       our proposed split: ML45-train minus G1_HOLDOUT (35 tasks); test = G1_HOLDOUT (10) + the ML45 test tasks (5).

python experiments/x4_metaworld_split.py [--record] [--out results/x4_metaworld/split.json]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]

# (verb, object, target, mods) per v3 task.  Rule: name tokens give the verb / object / modifiers; where the name is a noun
# only (assembly, basketball, soccer, hammer, shelf-place, sweep-into, ...) the verb is the description's main verb.
F = {
    "assembly":                 ("place",    "nut",      "peg",            ()),            # Pick up a nut and place it onto a peg
    "basketball":               ("place",    "ball",     "basket",         ()),            # Dunk the basketball into the basket
    "bin-picking":              ("place",    "puck",     "bin",            ()),            # Grasp the puck from one bin and place it into another bin
    "box-close":                ("close",    "lid",      "box",            ()),            # Grasp the cover and close the box with it
    "button-press-topdown":     ("press",    "button",   None,             ("topdown",)),
    "button-press-topdown-wall": ("press",   "button",   None,             ("topdown", "wall")),
    "button-press":             ("press",    "button",   None,             ()),
    "button-press-wall":        ("press",    "button",   None,             ("wall",)),
    "coffee-button":            ("press",    "button",   "coffee machine", ()),            # Push a button on the coffee machine
    "coffee-pull":              ("pull",     "mug",      "coffee machine", ()),
    "coffee-push":              ("push",     "mug",      "coffee machine", ()),
    "dial-turn":                ("turn",     "dial",     None,             ()),
    "disassemble":              ("pick-out", "nut",      "peg",            ()),            # Pick a nut out of a peg
    "door-close":               ("close",    "door",     None,             ()),
    "door-lock":                ("lock",     "door",     None,             ()),            # rotate the lock clockwise
    "door-open":                ("open",     "door",     None,             ()),
    "door-unlock":              ("unlock",   "door",     None,             ()),
    "drawer-close":             ("close",    "drawer",   None,             ()),
    "drawer-open":              ("open",     "drawer",   None,             ()),
    "faucet-close":             ("close",    "faucet",   None,             ()),            # Rotate the faucet clockwise
    "faucet-open":              ("open",     "faucet",   None,             ()),
    "hammer":                   ("hammer",   "hammer",   "nail",           ()),            # Hammer a screw on the wall
    "hand-insert":              ("insert",   None,       "hole",           ()),            # Insert the gripper into a hole
    "handle-press-side":        ("press",    "handle",   None,             ("side",)),
    "handle-press":             ("press",    "handle",   None,             ()),
    "handle-pull-side":         ("pull",     "handle",   None,             ("side",)),
    "handle-pull":              ("pull",     "handle",   None,             ()),
    "lever-pull":               ("pull",     "lever",    None,             ()),
    "peg-insert-side":          ("insert",   "peg",      "hole",           ("side",)),
    "peg-unplug-side":          ("unplug",   "peg",      None,             ("side",)),
    "pick-out-of-hole":         ("pick-out", "puck",     "hole",           ()),
    "pick-place":               ("place",    "puck",     "goal",           ()),
    "pick-place-wall":          ("place",    "puck",     "goal",           ("wall",)),
    "plate-slide-back-side":    ("slide",    "plate",    "cabinet",        ("back", "side")),
    "plate-slide-back":         ("slide",    "plate",    "cabinet",        ("back",)),
    "plate-slide-side":         ("slide",    "plate",    "cabinet",        ("side",)),
    "plate-slide":              ("slide",    "plate",    "cabinet",        ()),
    "push-back":                ("push",     "puck",     "goal",           ("back",)),
    "push":                     ("push",     "puck",     "goal",           ()),
    "push-wall":                ("push",     "puck",     "goal",           ("wall",)),
    "reach":                    ("reach",    None,       "goal",           ()),
    "reach-wall":               ("reach",    None,       "goal",           ("wall",)),
    "shelf-place":              ("place",    "puck",     "shelf",          ()),
    "soccer":                   ("push",     "ball",     "goal",           ()),            # Kick a soccer into the goal
    "stick-pull":               ("pull",     "box",      None,             ("stick",)),    # Grasp a stick and pull a box with the stick
    "stick-push":               ("push",     "box",      None,             ("stick",)),
    "sweep-into":               ("sweep",    "puck",     "hole",           ()),
    "sweep":                    ("sweep",    "puck",     None,             ()),            # Sweep a puck off the table
    "window-close":             ("close",    "window",   None,             ()),
    "window-open":              ("open",     "window",   None,             ()),
}
TASKS = sorted(F)
ML45_TEST = ["bin-picking", "box-close", "hand-insert", "door-lock", "door-unlock"]
ML45_TRAIN = [t for t in TASKS if t not in ML45_TEST]
# one task per (verb, object) family whose verb, object, target and modifiers all stay in training (a G1 hold-out)
G1_HOLDOUT = ["door-close", "window-close", "drawer-open", "faucet-open", "coffee-push", "stick-pull",
              "handle-pull-side", "plate-slide-back-side", "button-press-topdown-wall", "push-wall"]
X4_TRAIN = [t for t in ML45_TRAIN if t not in G1_HOLDOUT]
DIFFICULTY = {   # SmolVLA / LeRobot / RLinf grouping (lerobot envs/metaworld_config.json DIFFICULTY_TO_TASKS)
    "easy": ["button-press", "button-press-topdown", "button-press-topdown-wall", "button-press-wall", "coffee-button",
             "dial-turn", "door-close", "door-lock", "door-open", "door-unlock", "drawer-close", "drawer-open",
             "faucet-close", "faucet-open", "handle-press", "handle-press-side", "handle-pull", "handle-pull-side",
             "lever-pull", "plate-slide", "plate-slide-back", "plate-slide-back-side", "plate-slide-side", "reach",
             "reach-wall", "window-close", "window-open", "peg-unplug-side"],
    "medium": ["basketball", "bin-picking", "box-close", "coffee-pull", "coffee-push", "hammer", "peg-insert-side",
               "push-wall", "soccer", "sweep", "sweep-into"],
    "hard": ["assembly", "hand-insert", "pick-out-of-hole", "pick-place", "push", "push-back"],
    "very_hard": ["shelf-place", "disassemble", "stick-pull", "stick-push", "pick-place-wall"],
}
DIFF_OF = {t: d for d, ts in DIFFICULTY.items() for t in ts}


def parts(t):
    v, o, g, m = F[t]
    out = {("verb", v)}
    if o is not None:
        out.add(("object", o))
    if g is not None:
        out.add(("target", g))
    out |= {("mod", x) for x in m}
    return out


def grade(t, train):
    if t in train:
        return dict(grade="G0", unseen=[], pair_seen=True)
    seen = set().union(*(parts(s) for s in train))
    unseen = sorted(f"{k}:{v}" for k, v in parts(t) - seen)
    v, o = F[t][0], F[t][1]
    pair_seen = any(F[s][0] == v and F[s][1] == o for s in train)
    return dict(grade="G4" if unseen else "G1", unseen=unseen, pair_seen=pair_seen)


def check_metaworld():
    """Cross-check names and ML45 against the installed Farama package, when it imports (the X4 venv)."""
    try:
        import metaworld
    except Exception as e:                                    # noqa: BLE001
        return dict(checked=False, why=repr(e))
    names = sorted(n[:-3] for n in metaworld.ALL_V3_ENVIRONMENTS)
    ml45 = metaworld.ML45(seed=0)
    test = sorted(n[:-3] for n in ml45.test_classes)
    train = sorted(n[:-3] for n in ml45.train_classes)
    ok = names == TASKS and test == sorted(ML45_TEST) and train == sorted(ML45_TRAIN)
    return dict(checked=True, ok=ok, version=getattr(metaworld, "__version__", None), n_envs=len(names),
                ml45_test=test, names_match=names == TASKS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results/x4_metaworld/split.json"))
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    assert len(TASKS) == 50 and sorted(sum(DIFFICULTY.values(), [])) == TASKS
    sets = dict(mt50=dict(train=TASKS, test=TASKS), ml45=dict(train=ML45_TRAIN, test=ML45_TEST),
                x4_35=dict(train=X4_TRAIN, test=G1_HOLDOUT + ML45_TEST))
    res = {}
    for name, s in sets.items():
        rows = {t: dict(**grade(t, s["train"]), difficulty=DIFF_OF[t], parts=sorted(f"{k}:{v}" for k, v in parts(t)))
                for t in s["test"]}
        counts = {}
        for r in rows.values():
            counts[r["grade"]] = counts.get(r["grade"], 0) + 1
        res[name] = dict(n_train=len(s["train"]), n_test=len(s["test"]), counts=counts, tasks=rows)
    # every task held out alone against the other 49 (leave-one-out: which tasks could ever be G1)
    loo = {t: grade(t, [s for s in TASKS if s != t]) for t in TASKS}
    res["leave_one_out"] = dict(n_train=49, n_test=50, counts={g: sum(r["grade"] == g for r in loo.values()) for g in ("G1", "G4")},
                                tasks=loo)
    res["metaworld_check"] = check_metaworld()
    p = pathlib.Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    for name in ("mt50", "ml45", "x4_35"):
        print(f"== {name}: train {res[name]['n_train']}, test {res[name]['n_test']}, grades {res[name]['counts']}")
        if name != "mt50":
            for t, r in res[name]["tasks"].items():
                print(f"   {t:28s} {r['grade']}  {r['difficulty']:9s} pair_seen={r['pair_seen']!s:5s} unseen={r['unseen']}")
    print("== leave-one-out:", res["leave_one_out"]["counts"],
          "G4 alone:", sorted(t for t, r in loo.items() if r["grade"] == "G4"))
    print("== metaworld check:", res["metaworld_check"])
    if a.record:
        sys.path.insert(0, str(REPO))
        from lifelong_ood import ledger
        m = {k: dict(counts=v["counts"], n_train=v["n_train"], n_test=v["n_test"],
                     grades={t: r["grade"] for t, r in v["tasks"].items()},
                     unseen={t: r["unseen"] for t, r in v["tasks"].items() if r["unseen"]})
             for k, v in res.items() if k in ("mt50", "ml45", "x4_35", "leave_one_out")}
        m["metaworld_check"] = res["metaworld_check"]
        m["factors"] = {t: list(F[t][:3]) + [list(F[t][3])] for t in TASKS}
        pth = ledger.record("x4_metaworld_split", bench="metaworld_v3", tag="split", seed=0, metrics=m,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(out=a.out, ml45_test=ML45_TEST, g1_holdout=G1_HOLDOUT, x4_train=X4_TRAIN),
                            script="experiments/x4_metaworld_split.py", command=" ".join(sys.argv))
        print("ledger:", pth)


if __name__ == "__main__":
    main()
