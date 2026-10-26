"""PLAN_BIND E1 (docs/PLAN_E12.md section 1; pre-registration experiments/prereg/pbind_e1.py, ledger m3_prereg tag pbind_e1):
exemplar binding.  CPU only; no policy, no simulator, no new rollout.

Units: every zs binding failure on the 29 dev tasks (results/fm/types, arm zs, fresh_m1, 20 episodes per task):
  wrong_object                 role object, filler = the bound object
  wrong_goal / other_region    role goal,   filler = the BDDL region (else object) the named object ended in (`alt`, canonicalised:
                               the most specific name; >1 distinct after that -> excluded as ambiguous)
  wrong_goal / carried_elsewhere   role goal, filler = the scene candidate site nearest (xy) the final object position within
                               10 cm (D5 rule; results/bind/e1_scene.json); else excluded (counted)
Training lexicon: the 76 training cells of split.json "train" (task_table goal atoms); split_sealed.json is never opened.

Accounts (each predicts the wrong filler; ties count fractionally, 1 / |tied set| if the filler is in it):
  EX  exemplar binding: the filler bound for that role most often among the training cells of the dev task's SCENE, excluding the
      dev task's correct filler (undefined in a scene with no training cell)
  IS  instruction-selected trajectory: the binding for that role of the training cell whose instruction is most similar
      (word Jaccard over lower-cased words, all 76 cells; ties fractional), mapped into the dev scene (same name, else the unique
      candidate of the same type key; else no prediction)
Baseline (PRIMARY, frequency-matched): pick a candidate in proportion to (1 + the global count of its type key for that role across
all training cells); expected hit = P(filler).  Uniform over candidates is descriptive.  Candidates: object role = the dev task's
movable objects minus the named one; goal role = the scene's BDDL goal candidates for
the named object (fm_types.Scene.alternatives) plus the observed fillers, minus the correct filler and the held object.
Type key: objects drop the instance suffix (_<n>); regions = body type + region suffix.

Falsification (PLAN_BIND 2): E1 FAILS if EX's hit rate is not above the baseline: the two-level task-cluster bootstrap 95% CI of
mean(hit_EX - P_base) must have lo > 0 (units in scenes with no training cell are excluded from EX and counted).
Separation (descriptive): IS vs the baseline; EX - IS on the units both define; discordant units (EX-only vs IS-only hits).

    python experiments/bind_e1.py [--record]      (chunks env, via slurm/jobs/cpu.sbatch) -> results/bind/e1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import m1_compose_create as M1                                    # noqa: E402

OUT = os.path.join(ROOT, "results", "bind")
NEAR = 0.10                       # D5 rule: a carried_elsewhere filler = the nearest scene candidate site (xy) within 10 cm


def inst_type(name):
    return re.sub(r"_\d+$", "", name)


def body_of(name, bodies):
    c = [b for b in bodies if name == b or name.startswith(b + "_")]
    return max(c, key=len) if c else None


def tkey(name, t):
    """Type key of a filler in task t's scene: objects -> type; regions of a body -> body type + region suffix."""
    bodies = dict(t.objects, **t.fixtures)
    b = body_of(name, bodies)
    if b is None:
        return inst_type(name)
    typ = bodies[b]
    return typ if name == b else typ + name[len(b):]


def roles(t):
    """(object fillers, goal fillers) of a task's In / On atoms."""
    pl = [a for a in t.goal if a[0] in ("In", "On") and len(a) >= 3]
    return [str(a[1]) for a in pl], [str(a[2]) for a in pl]


def jacc(a, b):
    A, B = set(a.lower().split()), set(b.lower().split())
    return len(A & B) / max(len(A | B), 1)


def canon(alt, t):
    """Most specific names: drop a body name when one of its regions is also listed."""
    alt = sorted(set(alt))
    keep = [a for a in alt if not any(b != a and b.startswith(a + "_") for b in alt)]
    return keep


def frac_hit(pred, filler):
    """pred: Counter-like {name: weight} of the tied argmax set (weights sum 1) -> fractional hit."""
    return float(pred.get(filler, 0.0))


def argmax_set(cnt):
    if not cnt:
        return {}
    m = max(cnt.values())
    tied = [k for k, v in cnt.items() if v == m]
    return {k: 1.0 / len(tied) for k in tied}


def main():
    import d4_read
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--dry", action="store_true", help="build the units; print counts only (no hit, no baseline)")
    a = ap.parse_args()
    from lifelong_ood import l90c
    tt = M1.task_table()
    split = l90c.load_split()
    train = [tt[x] for x in split["train"]]
    dev = [tid for tid, _ in M1.dev_tasks()]
    SCN = json.load(open(os.path.join(OUT, "e1_scene.json"), encoding="utf-8"))
    gcount = {"object": Counter(), "goal": Counter()}               # global type-key counts over training cells
    for t in train:
        o, g = roles(t)
        for x in set(o):
            gcount["object"][tkey(x, t)] += 1
        for x in set(g):
            gcount["goal"][tkey(x, t)] += 1
    units, excl = [], Counter()
    for tid in dev:
        t = tt[tid]
        f = os.path.join(ROOT, "results", "fm", "types", tid.replace(":", "_") + ".json")
        eps = json.load(open(f, encoding="utf-8"))["arms"]["zs"]["episodes"]
        own_o, own_g = roles(t)
        if not own_o:
            continue
        scene_train = [x for x in train if x.scene == t.scene]
        bodies = dict(t.objects, **t.fixtures)
        obs_fill = set()
        for e in eps:
            if e["type"] == "wrong_goal" and e.get("sub") == "other_region":
                obs_fill |= set(canon(e.get("alt") or [], t))
        for e in eps:
            if e["type"] == "wrong_object":
                role, filler, correct = "object", e["bound"], set(own_o)
                lab = "bound"
            elif e["type"] == "wrong_goal":
                role, correct = "goal", set(own_g)
                if e.get("sub") != "other_region":
                    fin = SCN.get(tid, {}).get("carried_final", {}).get(str(e["trial"]))
                    near = None
                    if fin is not None:
                        dn = {n: float(np.linalg.norm(np.asarray(p[:2]) - np.asarray(fin[:2]))) for n, p in SCN[tid]["pos"].items()
                              if n in SCN[tid]["alternatives"]}
                        if dn and min(dn.values()) <= NEAR:
                            near = min(dn, key=dn.get)
                    if near is None:
                        excl["carried_elsewhere_no_site_10cm"] += 1
                        continue
                    filler, lab = near, "near_site"
                else:
                    c = canon(e.get("alt") or [], t)
                    if len(c) != 1:
                        excl["ambiguous_alt"] += 1
                        continue
                    filler, lab = c[0], "bddl"
            else:
                continue
            held = e.get("bound")
            if role == "object":
                cands = sorted(set(t.objects) - correct)
            else:
                cands = sorted((set(SCN[tid]["alternatives"]) | obs_fill) - correct - {held})
            if filler not in cands:
                cands = sorted(set(cands) | {filler})
            w = np.array([1.0 + gcount[role][tkey(c, t)] for c in cands])
            p_freq = float(w[cands.index(filler)] / w.sum())
            p_unif = 1.0 / len(cands)
            # EX: the scene's training bindings for this role
            ex = None
            if scene_train:
                cnt = Counter()
                for x in scene_train:
                    o, g = roles(x)
                    for v in set(o if role == "object" else g):
                        if v not in correct:
                            cnt[v] += 1
                ex = argmax_set(cnt) if cnt else {}
            # IS: the most similar training instruction's binding, mapped into the dev scene
            sims = [jacc(t.language, x.language) for x in train]
            m = max(sims)
            near = [x for x, s in zip(train, sims) if s == m]
            is_pred = {}
            for x in near:
                o, g = roles(x)
                vals = o if role == "object" else g
                mapped = []
                for v in set(vals):
                    if v in cands or v in correct:
                        mapped.append(v)
                    else:
                        same = [c for c in cands + sorted(correct) if tkey(c, t) == tkey(v, x)]
                        if len(same) == 1:
                            mapped.append(same[0])
                for v in mapped:
                    is_pred[v] = is_pred.get(v, 0.0) + 1.0 / (len(near) * len(mapped))
            units.append(dict(tid=tid, scene=t.scene, trial=e["trial"], role=role, filler=filler, label=lab, n_cands=len(cands),
                              p_freq=p_freq, p_unif=p_unif, ex=ex, is_pred=is_pred, is_near=[x.tid for x in near], is_sim=m,
                              hit_ex=(None if ex is None else frac_hit(ex, filler)), hit_is=frac_hit(is_pred, filler)))
    res = dict(n_units=len(units), excluded=dict(excl), by_role=dict(Counter(u["role"] for u in units)),
               by_task=dict(Counter(u["tid"] for u in units)))
    if a.dry:
        print(f"[bind e1 dry] units {res['n_units']}; excluded {res['excluded']}; roles {res['by_role']}; tasks {res['by_task']}; "
              f"EX undefined (no training cell in the scene) {sum(u['hit_ex'] is None for u in units)}; candidates per unit "
              f"median {np.median([u['n_cands'] for u in units]):.0f}", flush=True)
        return
    ex_u = [u for u in units if u["hit_ex"] is not None]
    res["ex_undefined_new_scene"] = len(units) - len(ex_u)

    def rate(us, key):
        per = {}
        for u in us:
            per.setdefault(u["tid"], []).append(float(key(u)))
        return d4_read.rate(per)
    res["EX_hit"] = rate(ex_u, lambda u: u["hit_ex"])
    res["EX_base_freq"] = rate(ex_u, lambda u: u["p_freq"])
    res["EX_minus_freq"] = rate(ex_u, lambda u: u["hit_ex"] - u["p_freq"])
    res["EX_minus_unif"] = rate(ex_u, lambda u: u["hit_ex"] - u["p_unif"])
    res["IS_hit"] = rate(units, lambda u: u["hit_is"])
    res["IS_base_freq"] = rate(units, lambda u: u["p_freq"])
    res["IS_minus_freq"] = rate(units, lambda u: u["hit_is"] - u["p_freq"])
    res["EX_minus_IS"] = rate(ex_u, lambda u: u["hit_ex"] - u["hit_is"])
    disc = [u for u in ex_u if abs(u["hit_ex"] - u["hit_is"]) > 1e-9]
    res["discordant"] = dict(n=len(disc), ex_better=sum(u["hit_ex"] > u["hit_is"] for u in disc),
                             is_better=sum(u["hit_is"] > u["hit_ex"] for u in disc))
    for role in ("object", "goal"):
        ru = [u for u in ex_u if u["role"] == role]
        if ru:
            res[f"EX_minus_freq_{role}"] = rate(ru, lambda u: u["hit_ex"] - u["p_freq"])
            res[f"IS_minus_freq_{role}"] = rate([u for u in units if u["role"] == role], lambda u: u["hit_is"] - u["p_freq"])
    lo = res["EX_minus_freq"]["lo"]
    res["verdict"] = "PASS (not falsified)" if lo is not None and lo > 0 else "FAIL (falsified)"
    res["units"] = units
    os.makedirs(OUT, exist_ok=True)
    tmp = os.path.join(OUT, "e1.json.tmp")
    json.dump(res, open(tmp, "w", encoding="utf-8"), indent=1, default=float)
    os.replace(tmp, os.path.join(OUT, "e1.json"))
    for k in ("n_units", "excluded", "by_role", "by_task", "ex_undefined_new_scene", "EX_hit", "EX_base_freq", "EX_minus_freq",
              "EX_minus_unif", "IS_hit", "IS_minus_freq", "EX_minus_IS", "discordant", "EX_minus_freq_object", "EX_minus_freq_goal",
              "IS_minus_freq_object", "IS_minus_freq_goal", "verdict"):
        if k in res:
            print(f"[bind e1] {k}: {json.dumps(res[k], default=float)}", flush=True)
    per_task = {}
    for u in units:
        per_task.setdefault(u["tid"], []).append(u)
    for tid, us in sorted(per_task.items()):
        u0 = us[0]
        print(f"[bind e1]   {tid} ({u0['scene']}, n {len(us)}): fillers {dict(Counter(u['filler'] for u in us))}; EX {u0['ex']}; "
              f"IS {u0['is_pred']} (near {u0['is_near']}, sim {u0['is_sim']:.2f}); p_freq {np.mean([u['p_freq'] for u in us]):.3f}",
              flush=True)
    if a.record:
        from lifelong_ood import ledger
        print(ledger.record("bind_e1", bench="libero_90_c", tag="read", seed=0, metrics={k: v for k, v in res.items() if k != "units"},
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/bind_e1.py", command="python experiments/bind_e1.py --record"))


if __name__ == "__main__":
    main()
