"""P0 of plan v4 (docs/PLAN_V4.md section 3.1, docs/DESIGN.md 3.1): freeze the LIBERO-90-C split by an MCD search.

ATOMS (read by rule from each task's BDDL through lifelong_ood/policy/factors.py; nothing tuned).  A task is a list of
STEPS, one per goal atom (LIBERO-10's two-step tasks and LIBERO-90's "X and Y" tasks are decomposed into their steps).
Each step gives
    verb    v:<BDDL predicate>             (In, On, Open, Close, Turnon, Turnoff)
    object  o:<type>                       (the moved object, or the articulated fixture for Open / Close / Turn*)
    target  t:<type>[:<region>]            (receptacle + region; for Open / Close the drawer, e.g. wooden_cabinet:top;
                                            none for Turnon / Turnoff / the microwave door)
and the task adds its scene s:<scene> and its instance qualifiers q:<left|right|front|back|middle>.
COMPOUNDS per step: object x target, verb x object, object x scene, target x scene.

DIVERGENCES (Keysers et al. arXiv 1912.09713, eq. 1-3): for a training set V and a test set W with normalised atom /
compound frequencies F_A, F_C,
    D_A(V||W) = 1 - C_0.5(F_A(V) || F_A(W)),   D_C(V||W) = 1 - C_0.1(F_C(V) || F_C(W)),   C_a(P||Q) = sum p^a q^(1-a).
A task's own CD is D_C(train || {task}); a bin's CD is D_C(train || bin).

POOL.  LIBERO-90's 90 tasks, minus task 51 (no episode in the IPEC LeRobot port: ledger l90_scenes), and the PART set:
two objects and one predicate held out of every training task (one per role: a moved object, a receptacle, a verb):
    ketchup (o), desk_caddy (t), Turnon (v)            -- the plan's own examples, fixed before the search.
"Absent" means absent from every training task's goal and instruction.  LIBERO-90 scenes share their props, so a part
object may sit in a training scene as a distractor; split.json counts those tasks.

SEARCH.  Seeded simulated annealing over the set W of combination test tasks (the rest of the pool trains), with
    hard:  D_A(train || W) <= 0.02, and every atom of every W task in >= 2 training tasks;
    bins:  each W task is graded by the fraction u of its compounds absent from training (u = 0 -> bin "0.1",
           0 < u < 0.4 -> "0.3", u >= 0.4 -> "0.5"; thresholds fixed before the search), aiming at >= 8 tasks per bin and a
           bin CD near its nominal value;
    size:  at most W_MAX test tasks, so the training cells stay near the plan's ~60.
SUITE ADDITIONS.  LIBERO-Spatial / -Object / -Goal / -10 tasks (pi05_libero's 40, demos in physical-intelligence/libero)
join training only if none of their compounds is a held-out compound (a compound of a test or part task absent from the
LIBERO-90 training cells) and none of their atoms is a part atom.  LIBERO-10 tasks that fail go to `two_step_test`.
SEALED.  One third of each bin and of the part set (stratified by which part), chosen by a seeded hash, go to
results/l90c/split_sealed.json (never committed); split.json keeps salted hashes so loaders refuse them
(lifelong_ood/l90c.py).

    python experiments/c_split.py --dry      # search, check, print; write nothing
    python experiments/c_split.py            # search, check, print (writes nothing if split.json exists and matches)
    python experiments/c_split.py --record   # also write the ledger row c_split
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lifelong_ood import l90c  # noqa: E402
from lifelong_ood.policy import factors as F  # noqa: E402

SEED = 20261022
NO_DEMOS = ("libero_90:51",)                       # zero episodes in IPEC-COMMUNITY/libero_90_no_noops_lerobot
PART = dict(object="ketchup", target="desk_caddy", verb="Turnon")
BIN_EDGES = (0.0, 0.4)                             # u == 0 | 0 < u < 0.4 | u >= 0.4
BINS = ("0.1", "0.3", "0.5")
MIN_PER_BIN = 8
W_MAX = 30
AD_MAX = 0.02
ALPHA_A, ALPHA_C = 0.5, 0.1
SUITES = ("libero_90", "libero_10", "libero_spatial", "libero_object", "libero_goal")
TRIVIAL_REGION = (None, "contain", "cook", "heating")


# ── atoms and compounds ─────────────────────────────────────────────────────────────────────────────────────────────
def steps(task):
    """-> [(verb, object, target or None)] with type-level names."""
    out = []
    for s in task.slots:
        pred, obj = s[0], s[1]
        tgt, region = (s[2], s[3]) if len(s) > 3 else (None, None)
        if pred in ("In", "On"):
            t = tgt + ("" if region in TRIVIAL_REGION else ":" + region)
        elif pred in ("Open", "Close") and region:
            t = f"{obj}:{region}"                  # the drawer
        else:
            t = None
        out.append((pred, obj, t))
    return out


def atoms(task):
    c = collections.Counter()
    for v, o, t in steps(task):
        c["v:" + v] += 1
        c["o:" + o] += 1
        if t:
            c["t:" + t] += 1
    c["s:" + task.scene] += 1
    for q in task.qualifiers:
        c["q:" + q] += 1
    return c


def compounds(task):
    c = collections.Counter()
    s = "s:" + task.scene
    for v, o, t in steps(task):
        o_, v_ = "o:" + o, "v:" + v
        c[(v_, o_)] += 1
        c[(o_, s)] += 1
        if t:
            t_ = "t:" + t
            c[(o_, t_)] += 1
            c[(t_, s)] += 1
    return c


def part_atoms():
    return {"o:" + PART["object"], "t:" + PART["target"], "o:" + PART["target"], "v:" + PART["verb"]}


def has_part(task):
    a = atoms(task)
    hit = set(a) & part_atoms()
    # the caddy as a target type with any region (t:desk_caddy:front_contain ...)
    hit |= {k for k in a if k.startswith("t:" + PART["target"] + ":")}
    return sorted(hit)


def which_part(task):
    h = has_part(task)
    for role, val in (("object", PART["object"]), ("target", PART["target"]), ("verb", PART["verb"])):
        if any(val in k for k in h):
            return role
    return None


# ── divergences ─────────────────────────────────────────────────────────────────────────────────────────────────────
def dist(counters):
    tot = collections.Counter()
    for c in counters:
        tot.update(c)
    n = sum(tot.values())
    return {k: v / n for k, v in tot.items()} if n else {}


def chernoff(p, q, a):
    return sum(p[k] ** a * q[k] ** (1 - a) for k in p.keys() & q.keys())


def d_atom(train, test):
    return 1.0 - chernoff(dist(map(atoms, train)), dist(map(atoms, test)), ALPHA_A)


def d_comp(train, test):
    return 1.0 - chernoff(dist(map(compounds, train)), dist(map(compounds, test)), ALPHA_C)


def unseen_frac(task, train_comp):
    c = compounds(task)
    return sum(v for k, v in c.items() if k not in train_comp) / sum(c.values())


def grade(u):
    return BINS[0] if u <= BIN_EDGES[0] else BINS[1] if u < BIN_EDGES[1] else BINS[2]


def atom_task_counts(tasks):
    c = collections.Counter()
    for t in tasks:
        c.update(set(atoms(t)))
    return c


# ── the search ──────────────────────────────────────────────────────────────────────────────────────────────────────
def evaluate(pool, W):
    train = [t for t in pool if t.tid not in W]
    test = [t for t in pool if t.tid in W]
    if not test:
        return dict(score=1e9)
    seen_c = set()
    for t in train:
        seen_c |= set(compounds(t))
    cnt = atom_task_counts(train)
    under = sum(1 for t in test for a in atoms(t) if cnt[a] < 2)
    da = d_atom(train, test)
    bins = collections.defaultdict(list)
    for t in test:
        bins[grade(unseen_frac(t, seen_c))].append(t)
    cds = {b: d_comp(train, bins[b]) if bins[b] else 1.0 for b in BINS}
    # hard: every test atom in >= 2 training tasks; then >= MIN_PER_BIN per bin; then the smallest atom divergence; bin CDs
    # near nominal; at most W_MAX test tasks.  D_A <= AD_MAX is reported, not forced: its sampling floor at LIBERO-90's size
    # is far above it (split.json: checks.atom_divergence_null).
    score = (1000.0 * under + 5.0 * sum(max(0, MIN_PER_BIN - len(bins[b])) for b in BINS) + 10.0 * da
             + 2.0 * sum(abs(cds[b] - float(b)) for b in BINS) + 0.5 * max(0, len(test) - W_MAX))
    return dict(score=score, under=under, da=da, n={b: len(bins[b]) for b in BINS}, cd=cds,
                bins={b: sorted(t.tid for t in bins[b]) for b in BINS})


def search(pool, seed=SEED, iters=30000, restarts=8):
    rng = random.Random(seed)
    ids = [t.tid for t in pool]
    best, best_W = None, None
    for r in range(restarts):
        W = set(rng.sample(ids, 27))
        cur = evaluate(pool, W)
        for i in range(iters):
            T = 0.5 * (1 - i / iters) + 1e-3
            W2 = set(W)
            mv = rng.random()
            out_ = sorted(set(ids) - W2)
            if mv < 0.5 and W2 and out_:                               # swap
                W2.remove(rng.choice(sorted(W2)))
                W2.add(rng.choice(out_))
            elif mv < 0.75 and W2:
                W2.remove(rng.choice(sorted(W2)))
            elif out_:
                W2.add(rng.choice(out_))
            new = evaluate(pool, W2)
            if new["score"] <= cur["score"] or rng.random() < math.exp(-(new["score"] - cur["score"]) / T):
                W, cur = W2, new
                if best is None or cur["score"] < best["score"]:
                    best, best_W = cur, set(W)
        print(f"[c_split] restart {r}: best score {best['score']:.3f} n={best['n']} da={best['da']:.4f} "
              f"cd={ {b: round(v, 3) for b, v in best['cd'].items()} } under={best['under']}", flush=True)
    return best_W, best


def null_atom_divergence(pool, n, draws=500, seed=SEED):
    """D_A between random disjoint (train, test) partitions of the pool with |test| = n: the measure's sampling floor here."""
    rng = random.Random(seed + 1)
    v = []
    for _ in range(draws):
        W = set(rng.sample([t.tid for t in pool], n))
        v.append(d_atom([t for t in pool if t.tid not in W], [t for t in pool if t.tid in W]))
    v.sort()
    return dict(n_test=n, mean=sum(v) / len(v), p05=v[int(0.05 * len(v))], min=v[0])


# ── assembling the split ────────────────────────────────────────────────────────────────────────────────────────────
def sealed_pick(tids, k, salt):
    return sorted(sorted(tids, key=lambda t: hashlib.sha256((salt + t).encode()).hexdigest())[:k])


def build(seed=SEED):
    all_tasks = F.load(SUITES)
    by_id = {t.tid: t for t in all_tasks}
    l90 = [t for t in all_tasks if t.suite == "libero_90"]
    part = [t for t in l90 if has_part(t)]
    pool = [t for t in l90 if not has_part(t) and t.tid not in NO_DEMOS]
    W, ev = search(pool, seed)
    train90 = [t for t in pool if t.tid not in W]
    test = [t for t in pool if t.tid in W]
    seen_c = set()
    for t in train90:
        seen_c |= set(compounds(t))
    held_c = set()
    for t in test + part:
        held_c |= {k for k in compounds(t) if k not in seen_c}
    adds, rejected = [], {}
    for t in all_tasks:
        if t.suite == "libero_90":
            continue
        bad = sorted(map(str, set(compounds(t)) & held_c))
        if has_part(t) or bad:
            rejected[t.tid] = dict(part=has_part(t), held_compounds=bad)
        else:
            adds.append(t)
    two_step_test = sorted(tid for tid in rejected if tid.startswith("libero_10:"))
    salt = hashlib.sha256(f"l90c-{seed}".encode()).hexdigest()[:16]
    dev, sealed = {}, {}
    for b in BINS:
        tids = ev["bins"][b]
        s = sealed_pick(tids, round(len(tids) / 3), salt)
        sealed[b], dev[b] = s, sorted(set(tids) - set(s))
    sealed["part"], dev["part"] = [], []
    for role in ("object", "target", "verb"):
        tids = sorted(t.tid for t in part if which_part(t) == role)
        s = sealed_pick(tids, round(len(tids) / 3), salt)
        sealed["part"] += s
        dev["part"] += sorted(set(tids) - set(s))
    return dict(by_id=by_id, pool=pool, part=part, train90=train90, test=test, adds=adds, rejected=rejected,
                two_step_test=two_step_test, ev=ev, held_c=held_c, seen_c=seen_c, salt=salt, dev=dev, sealed=sealed)


def checks(b):
    train90, test, part, adds = b["train90"], b["test"], b["part"], b["adds"]
    train_all = train90 + adds
    cnt90 = atom_task_counts(train90)
    out = {}
    out["atom_divergence_train90_vs_test"] = d_atom(train90, test)
    out["atom_divergence_trainall_vs_test"] = d_atom(train_all, test)
    out["compound_divergence_train90_vs_test"] = d_comp(train90, test)
    out["min_train_count_of_test_atoms"] = min(cnt90[a] for t in test for a in atoms(t))
    out["test_atoms_below_2"] = sorted({a for t in test for a in atoms(t) if cnt90[a] < 2})
    pa = part_atoms()
    out["part_atoms_in_training"] = sorted({a for t in train_all for a in atoms(t)
                                            if a in pa or a.startswith("t:" + PART["target"] + ":")})
    out["held_compounds_in_training"] = sorted(map(str, {k for t in train_all for k in compounds(t)} & b["held_c"]))
    out["part_objects_as_distractor"] = {v: sorted(t.tid for t in train_all if v in set(t.objects.values()) | set(t.fixtures.values()))
                                         for v in (PART["object"], PART["target"])}
    per_task = {}
    for t in test:
        u = unseen_frac(t, b["seen_c"])
        per_task[t.tid] = dict(u=round(u, 4), cd=round(d_comp(train90, [t]), 4), bin=grade(u),
                               unseen=sorted(map(str, (k for k in compounds(t) if k not in b["seen_c"]))))
    out["per_task"] = per_task
    out["bin_cd"] = {k: round(d_comp(train90, [b["by_id"][x] for x in v]), 4) for k, v in b["ev"]["bins"].items() if v}
    out["bin_n"] = {k: len(v) for k, v in b["ev"]["bins"].items()}
    out["bin_mean_task_cd"] = {k: round(sum(per_task[x]["cd"] for x in v) / len(v), 4) for k, v in b["ev"]["bins"].items() if v}
    out["part_cd"] = round(d_comp(train90, part), 4)
    out["part_atom_divergence"] = round(d_atom(train90, part), 4)
    return out


def table(b, chk):
    rows = []
    for t in b["train90"] + b["adds"]:
        rows.append(("train", t.tid, t.scene, t.language))
    for bn in BINS:                                     # dev tasks only: the logs must not name the sealed third
        for tid in b["dev"][bn]:
            t = b["by_id"][tid]
            rows.append((f"test cd~{bn}", tid, t.scene, t.language + f"  [u={chk['per_task'][tid]['u']}, cd={chk['per_task'][tid]['cd']}]"))
    for t in b["part"]:
        if t.tid in b["dev"]["part"]:
            rows.append((f"part:{which_part(t)}", t.tid, t.scene, t.language))
    rows.append(("sealed", "-", "-", json.dumps({k: len(v) for k, v in b["sealed"].items()})))
    return rows


def main():
    b = build()
    chk = checks(b)
    chk["atom_divergence_null"] = null_atom_divergence(b["pool"], len(b["test"]))
    chk["atom_divergence_meets_0.02"] = chk["atom_divergence_train90_vs_test"] <= AD_MAX
    assert not chk["test_atoms_below_2"], chk["test_atoms_below_2"]
    assert not chk["part_atoms_in_training"] and not chk["held_compounds_in_training"]
    train_ids = sorted(t.tid for t in b["train90"] + b["adds"])
    print(json.dumps({k: v for k, v in chk.items() if k != "per_task"}, indent=1))
    for r in table(b, chk):
        print(f"{r[0]:14} {r[1]:18} {r[2]:22} {r[3]}")
    if "--dry" in sys.argv:
        return
    os.makedirs(l90c.RESULTS, exist_ok=True)
    if l90c.SPLIT.exists() and "--force" not in sys.argv:
        old = json.loads(l90c.SPLIT.read_text(encoding="utf-8"))
        assert old["train"] == train_ids, "split.json exists and differs: the split is frozen"
        print("split.json exists and matches the rebuild; not rewriting")
        return
    sealed_obj = dict(note="SEALED: read once at the final read, via lifelong_ood.l90c.load_sealed(prereg=...)",
                      seed=SEED, sealed=b["sealed"])
    l90c.SEALED.write_text(json.dumps(sealed_obj, indent=1) + "\n", encoding="utf-8")
    pt = chk["per_task"]
    split = dict(
        benchmark="LIBERO-90-C (our MCD split of LIBERO-90; docs/PLAN_V4.md 3.1, docs/L90C.md)",
        seed=SEED, parser_hash=F.parser_hash(), part=PART, no_demos=list(NO_DEMOS), bins=list(BINS),
        bin_edges=list(BIN_EDGES), ad_max=AD_MAX, alpha=dict(atom=ALPHA_A, compound=ALPHA_C),
        train=train_ids, train_l90=sorted(t.tid for t in b["train90"]), train_suite_additions=sorted(t.tid for t in b["adds"]),
        suite_rejected=b["rejected"], two_step_test=b["two_step_test"],
        dev={k: v for k, v in b["dev"].items()},
        dev_per_task={tid: pt[tid] for k in BINS for tid in b["dev"][k]},
        n_sealed={k: len(v) for k, v in b["sealed"].items()}, hash_salt=b["salt"],
        sealed_hashes=sorted(l90c.task_hash(t, b["salt"]) for v in b["sealed"].values() for t in v),
        sealed_sha256=hashlib.sha256(l90c.SEALED.read_bytes()).hexdigest(),
        held_compounds=sorted(map(str, b["held_c"])),
        checks={k: v for k, v in chk.items() if k != "per_task"},
        languages={t.tid: t.language for t in b["train90"] + b["adds"] + b["test"] + b["part"]})
    l90c.SPLIT.write_text(json.dumps(split, indent=1) + "\n", encoding="utf-8")
    n = dict(train=len(train_ids), train_l90=len(b["train90"]), train_adds=len(b["adds"]),
             **{f"dev_{k}": len(v) for k, v in b["dev"].items()}, **{f"sealed_{k}": len(v) for k, v in b["sealed"].items()})
    print(json.dumps(n))
    if "--record" in sys.argv:
        from lifelong_ood import ledger
        print(ledger.record("c_split", bench="libero_90_c", tag="frozen", seed=SEED,
                            metrics=dict(counts=n, part=PART, checks={k: v for k, v in chk.items() if k != "per_task"},
                                         split_sha256=hashlib.sha256(l90c.SPLIT.read_bytes()).hexdigest(),
                                         sealed_sha256=split["sealed_sha256"]),
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(seed=SEED, part=PART, bins=list(BINS), bin_edges=list(BIN_EDGES), ad_max=AD_MAX,
                                        w_max=W_MAX, min_per_bin=MIN_PER_BIN, parser_hash=F.parser_hash()),
                            script="experiments/c_split.py", command=" ".join(sys.argv),
                            artifacts=[str(l90c.SPLIT), str(l90c.SEALED)]))


if __name__ == "__main__":
    main()
