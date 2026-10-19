"""The COMPOSITIONAL SPLIT, made explicit and graded (docs/SPLIT.md).  CPU; no simulator.

Training set: pi0.5's 40 tasks (LIBERO-Spatial / Object / Goal / 10), fixed by the frozen checkpoint.  Every evaluated task is read
by rule from its BDDL file (lifelong_ood/policy/factors.py: object = the moved object's type; relation = (predicate, target type,
region); scene; instance qualifiers), and graded by what is UNSEEN in the 40:
    G1  relation   every part seen, each part seen IN THIS SCENE, but object and goal never paired in one training task
    G2  binding    every part and the scene seen, but some part never seen in this scene
    G3  scene      the scene never seen; every part seen in other scenes
    G4  part       the moved object, the goal relation or the predicate never seen anywhere
    G0  seen       the same (object, relation) pair in this scene is a training task
Every result with per-task counts is then stratified by grade.

    $PY experiments/split_grades.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood.policy import factors as F                         # noqa: E402

R = os.path.join(ROOT, "results", "robot")
NAMES = {"G0": "seen", "G1": "relation", "G2": "binding", "G3": "scene", "G4": "part"}


LOC = set(F.QUALIFIERS) | {"top", "bottom", "middle"}                   # regions that name a LOCATION (a drawer, a side)


def norm(task):
    """Read-level normalisation, applied to every task alike: In and On are one placing predicate (written On) (Li's suites write On(basket)
    where LIBERO writes In(basket_contain)), a region counts only when it names a location (Li's 'base' for the stove's 'cook'),
    and Li's OOD suites sit in their base suite's scene (the same table, objects added)."""
    import dataclasses
    sl = [(("On" if x[0] in ("In", "On") else x[0]), x[1], x[2], (x[3] if x[3] in LOC else None)) for x in task.slots]
    sc = task.scene.replace("_ood", "") if task.suite in F.OOD_SUITES else task.scene
    return dataclasses.replace(task, slots=sl, scene=sc)


def grade(task, train):
    task = norm(task)
    train = [norm(t) for t in train]
    c = F.coverage(task, train)
    moved, rels = set(task.moved), {s[:1] + s[2:] for s in task.slots}
    paired = lambda ts: any(moved and moved <= set(t.moved) and rels <= {s[:1] + s[2:] for s in t.slots} for t in ts)   # noqa: E731
    same = [t for t in train if t.scene == task.scene]
    if not (c["obj_any"] or not moved) or not c["rel_any"] or not c["pred_any"]:
        g = "G4"
    elif not c["scene_seen"]:
        g = "G3"
    elif paired(same):
        g = "G0"
    elif (c["obj_same_scene"] or not moved) and c["rel_same_scene"]:
        g = "G1"
    else:
        g = "G2"
    return g, c


def tasks():
    train = F.load(F.TRAIN_SUITES)
    ood = F.load(F.OOD_SUITES)
    l90 = F.load(("libero_90",))
    out = {}
    for t in ood:
        out[f"{t.suite.split('_')[1]}:{t.index}"] = t
    for t in l90:
        out[str(t.index)] = t
    return train, out


def results():
    """-> {result name: {task key: {arm: successes}}} from the per-task reads that exist."""
    res = {}
    for name, f in (("G8 LIBERO-90 set d (pi0.5)", "g8_read.json"), ("Z3 pointers, zero fit (pi0.5)", "z3_read.json"),
                    ("Li OOD G15 (pi0.5)", "g15_read_pi05.json"), ("Li OOD G15 (pi0)", "g15_read_pi0.json"),
                    ("Li OOD G18 (SmolVLA)", "g18_read_smolvla.json"), ("Li OOD G18 (X-VLA)", "g18_read_xvla.json"),
                    ("Li OOD Z1b (FLOWER, zero fit)", "z1b_read.json")):
        p = os.path.join(R, f)
        if os.path.exists(p):
            d = json.load(open(p))
            if d.get("per_task"):
                res[name] = {k: {a: v for a, v in row.items() if isinstance(v, int)} for k, row in d["per_task"].items()}
    p = os.path.join(R, "z4_pilot.json")
    if os.path.exists(p):
        res["Z4 programs, zero fit, pilot (pi0.5)"] = {k: v for k, v in json.load(open(p))["rows"].items()}
    return res


def main():
    train, ts = tasks()
    grades = {}
    for k, t in ts.items():
        g, c = grade(t, train)
        grades[k] = dict(grade=g, language=t.language, scene=t.scene, coverage=c)
    print("grade counts:", {g: sum(v["grade"] == g for v in grades.values()) for g in NAMES}, "of", len(grades))
    table = {}
    for name, rows in results().items():
        by = {}
        for k, arms in rows.items():
            if k not in grades:
                continue
            g = grades[k]["grade"]
            b = by.setdefault(g, dict(tasks=0))
            b["tasks"] += 1
            for a, v in arms.items():
                if v is not None:
                    b[a] = b.get(a, 0) + int(v)
        table[name] = by
        print(f"\n{name}")
        for g in sorted(by):
            b = by[g]
            print(f"  {g} {NAMES[g]:9} tasks {b['tasks']:3}  " + "  ".join(f"{a} {v}" for a, v in b.items() if a != "tasks"))
    json.dump(dict(grades=grades, table=table), open(os.path.join(R, "split_grades.json"), "w"), indent=1, default=str)
    if "--record" in sys.argv:
        from lifelong_ood import ledger
        counts = {g: sum(v["grade"] == g for v in grades.values()) for g in NAMES}
        print(ledger.record("split_grades", bench="libero", tag="read", seed=0, metrics=dict(counts=counts, table=table,
                            grades={k: v["grade"] for k, v in grades.items()}, parser=F.parser_hash()),
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/split_grades.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    main()
