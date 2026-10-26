"""PLAN_SOLVE S1 (docs/PLAN_S1.md): the binder's labels.  CPU (chunks env).  Reads results/bind/s1/frames/{train,dev}/*.npz (names
only) and writes results/bind/s1/labels.json.

TRAINING examples (training cells only; factorised and counterfactual within the scene).  Per training frame, for EVERY entity /
region of the frame that has a phrase (not only the cell's own target):
  object role  every movable instance m:  the TYPE phrase (factors.object_phrases over the TRAINING tasks + PHRASE_FALLBACK) ->
               target = every instance of m's type (an ambiguous phrase is labelled with the union);
               INSTANCE phrases read from the training instructions (the object phrase of each In / On step -> that atom's first
               argument), used on every frame of the same libero_90 scene (the scene's layout is shared) or of the same task (other
               suites, whose scene names do not fix the layout)
  goal role    every region site whose (fixture type, region) has a phrase ATTESTED in a training instruction (GOAL_T; the source
               task is listed), table regions relative to a body ("to the <dir> of the <body phrase>", attested by :37, :69, :34,
               goal:5), and movable goal objects of types that training atoms use as targets ("on the <type phrase>" -> every
               instance of the type); plus each training instruction's own goal phrase -> its atom's target (same scope rule)
DEV units (evaluation only; never trained on): per dev task, the step with an In / On goal; object phrase -> the atom's first
argument, goal phrase -> its target.  Depth = E4's pre-registered code (experiments/bind_e4.code), unchanged.
seen = (role, normalised phrase, scene) occurs among the TRAINING examples; else the unit is a held-out combination.
Phrase split (both sets): "pick up X and place / put it G" -> X / G; else the first goal preposition after the verb, where
"in / on / at the <left|right|front|back|middle>" followed by the end or another preposition is the object's qualifier.

    python experiments/s1_labels.py        (chunks env, slurm/jobs/cpu.sbatch)
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
S1 = os.path.join(ROOT, "results", "bind", "s1")
QUAL = ("left", "right", "front", "back", "middle")
DIRS = ("left", "right", "front", "back")
GOAL_PREPS = ("on top of", "to the right of", "to the left of", "to the front of", "to the back of", "under", "into", "in", "on")
TABLES = ("kitchen_table", "study_table", "living_room_table", "main_table", "table")
# (fixture family, region suffix) -> phrase, each ATTESTED in a training instruction (source in the comment)
GOAL_T = {("cabinet", "top_side"): "on top of the cabinet",                  # libero_90:10 / :25 / :31, goal:2 / :4
          ("cabinet", "top_region"): "in the top drawer of the cabinet",      # libero_90:3 / :4 / :5
          ("cabinet", "middle_region"): "in the middle drawer of the cabinet",  # goal:0 ("the middle drawer of the cabinet") + :26's "in"
          ("cabinet", "bottom_region"): "in the bottom drawer of the cabinet",  # libero_90:26
          ("shelf", "top_side"): "on top of the cabinet",                     # libero_90:41 / :43
          ("shelf", "top_region"): "on the cabinet shelf",                    # libero_90:86 / :88
          ("shelf", "bottom_region"): "under the cabinet shelf",              # libero_90:42 / :89
          ("basket", "contain_region"): "in the basket",                      # libero_90:46 ...
          ("tray", "contain_region"): "in the tray",                          # libero_90:55 ...
          ("stove", "cook_region"): "on the stove",                           # libero_90:18 / :19, goal:1
          ("microwave", "heating_region"): "in the microwave",                # libero_10:9
          ("wine_rack", "top_region"): "on the wine rack"}                    # libero_90:27
FAMILIES = (("wooden_two_layer_shelf", "shelf"), ("cabinet", "cabinet"), ("basket", "basket"), ("tray", "tray"),
            ("flat_stove", "stove"), ("microwave", "microwave"), ("wine_rack", "wine_rack"))


def itype(n):
    return re.sub(r"_\d+$", "", n)


def norm(p):
    p = " ".join(p.lower().split())
    return re.sub(r"^(the|both the) ", "", p)


def split_step(c):
    """-> (object phrase, goal phrase or '')."""
    c = " ".join(c.strip().lower().split())
    m = re.match(r"^pick up (.+?) and (?:place|put) (?:it|them) (.+)$", c)
    if m:
        return m.group(1), m.group(2)
    w = c.split()
    if not w or w[0] not in ("put", "place", "stack", "push"):
        return c, ""
    rest = w[1:]
    preps = [p.split() for p in GOAL_PREPS]
    for i in range(1, len(rest)):
        for p in preps:
            if rest[i:i + len(p)] != p:
                continue
            j = i + len(p)
            if p[0] in ("in", "on", "at") and len(p) == 1 and j + 1 < len(rest) + 1 and rest[j:j + 1] == ["the"] \
                    and j + 1 < len(rest) and rest[j + 1] in QUAL and (j + 2 == len(rest) or any(
                        rest[j + 2:j + 2 + len(q)] == q for q in preps)):
                break                                             # "on the left" / "in the middle": the object's qualifier
            return " ".join(rest[:i]), " ".join(rest[i:])
    return " ".join(rest), ""


def goal_steps(t):
    from lifelong_ood.policy import primitives as P
    out = []
    for c in P.steps_of(t["lang"]):
        o, g = split_step(c)
        if g:
            out.append((o, g))
    atoms = [a for a in t["goal"] if a[0] in ("In", "On") and len(a) == 3]
    return list(zip(out, atoms)) if len(out) == len(atoms) else None


def family(body_type):
    return next((f for k, f in FAMILIES if k in body_type), None)


def main():
    from lifelong_ood.policy import factors as F
    import bind_e4 as E4
    import m1_compose_create as M1
    tasks = json.load(open(os.path.join(S1, "tasks.json"), encoding="utf-8"))
    tt = M1.task_table()
    train_t = [tt[t] for t, x in tasks.items() if x["set"] == "train"]
    ph = F.object_phrases(train_t)
    goal_obj_types = {itype(a[2]) for t, x in tasks.items() if x["set"] == "train" for a in x["goal"]
                      if a[0] in ("In", "On") and len(a) == 3 and a[2] in x["objects"]}
    # instance phrases and own goal phrases from the training instructions
    inst = []                                                     # (scope kind, scope, role, phrase, target)
    skipped = []
    for tid, x in tasks.items():
        if x["set"] != "train":
            continue
        gs = goal_steps(x)
        if gs is None:
            skipped.append(tid)
            continue
        scope = ("scene", x["scene"]) if tid.startswith("libero_90:") else ("task", tid)
        for (o, g), a in gs:
            if not re.search(r"\b(both|them|bowls)\b", o):
                inst.append((*scope, "object", norm(o), a[1]))
            inst.append((*scope, "goal", norm(g), a[2]))
    train, seen = [], set()
    for f in sorted(glob.glob(os.path.join(S1, "frames", "train", "*.npz"))):
        z = np.load(f, allow_pickle=True)
        tid, x = str(z["tid"]), tasks[str(z["tid"])]
        bodies, sites = [str(b) for b in z["body_names"]], [str(s) for s in z["site_names"]]
        movable = [b for b in bodies if b in x["objects"]]
        ex = {}

        def add(role, phrase, targets):
            k = (role, norm(phrase))
            ex.setdefault(k, set()).update(targets)
        for m in movable:
            add("object", F.phrase_of(itype(m), ph), [b for b in movable if itype(b) == itype(m)])
            if itype(m) in goal_obj_types:
                add("goal", "on the " + F.phrase_of(itype(m), ph), [b for b in movable if itype(b) == itype(m)])
        for s in sites:
            body = max([b for b in bodies if s.startswith(b + "_")], key=len, default=None)
            if body is None or "init_region" in s:
                continue
            btype = x["fixtures"].get(body) or x["objects"].get(body) or itype(body)
            if btype in TABLES or body in TABLES:
                core = s[len(body) + 1:-len("_region")] if s.endswith("_region") else ""
                parts = core.split("_")
                if len(parts) >= 2 and parts[-1] in DIRS:
                    ref = "_".join(parts[:-1])
                    reft = next((itype(b) for b in bodies if itype(b) == ref or itype(b).endswith("_" + ref)), ref)
                    add("goal", f"to the {parts[-1]} of the {F.phrase_of(reft, ph)}", [s])
                continue
            fam = family(btype)
            suf = s[len(body) + 1:]
            if fam and (fam, suf) in GOAL_T:
                add("goal", GOAL_T[(fam, suf)], [s])
        iex = {}                                                  # instruction phrases override a template with the same text
        for kind, scope, role, phrase, target in inst:
            if (kind == "scene" and scope == x["scene"]) or (kind == "task" and scope == tid):
                if target in bodies or target in sites:
                    iex.setdefault((role, phrase), set()).add(target)
        ex.update(iex)
        for (role, phrase), tg in sorted(ex.items()):
            tg = sorted(t for t in tg if t in bodies or t in sites)
            if tg:
                train.append(dict(frame=os.path.relpath(f, ROOT), tid=tid, scene=x["scene"], role=role, phrase=phrase, targets=tg))
                seen.add((role, phrase, x["scene"]))
    scn = json.load(open(os.path.join(ROOT, "results", "bind", "e1_scene.json"), encoding="utf-8"))
    dev, dev_phr = [], {}
    for tid, x in tasks.items():
        if x["set"] != "dev":
            continue
        gs = goal_steps(x)
        assert gs, tid
        (o, g), a = gs[0]
        d_obj, d_goal, _ = E4.code(tt[tid], scn[tid])
        dev_phr[tid] = dict(object=norm(o), goal=norm(g), named=a[1], target=a[2], d_obj=d_obj, d_goal=d_goal)
        for f in sorted(glob.glob(os.path.join(S1, "frames", "dev", tid.replace(":", "_") + "_*.npz"))):
            for role, phrase, target, d in (("object", norm(o), a[1], d_obj), ("goal", norm(g), a[2], d_goal)):
                dev.append(dict(frame=os.path.relpath(f, ROOT), tid=tid, scene=x["scene"], role=role, phrase=phrase, targets=[target],
                                depth=int(d), seen=(role, phrase, x["scene"]) in seen,
                                phrase_seen_any=any(r == role and p == phrase for r, p, _ in seen),
                                scene_familiar=any(s == x["scene"] for _, _, s in seen)))
    phrases = sorted({e["phrase"] for e in train} | {e["phrase"] for e in dev})
    out = dict(train=train, dev=dev, phrases=phrases, dev_phrases=dev_phr, skipped_train_tasks=skipped,
               type_phrases=ph, goal_templates={f"{k[0]}|{k[1]}": v for k, v in GOAL_T.items()},
               n=dict(train=len(train), dev=len(dev), train_frames=len({e["frame"] for e in train}), phrases=len(phrases)))
    json.dump(out, open(os.path.join(S1, "labels.json"), "w", encoding="utf-8"), indent=1)
    print(f"[s1 labels] {out['n']}; skipped {skipped}", flush=True)
    from collections import Counter
    print("[s1 labels] train examples per role:", Counter(e["role"] for e in train), flush=True)
    for tid, d in dev_phr.items():
        u = [e for e in dev if e["tid"] == tid]
        print(f"[s1 labels] dev {tid}: obj {d['object']!r} -> {d['named']} (depth {d['d_obj']}, seen "
              f"{u[0]['seen'] if u else None}); goal {d['goal']!r} -> {d['target']} (depth {d['d_goal']}, seen "
              f"{u[1]['seen'] if len(u) > 1 else None})", flush=True)


if __name__ == "__main__":
    main()
