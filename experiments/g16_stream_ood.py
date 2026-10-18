"""Plan G, read G16 (iteration 13; ledger m3_prereg tag g16_stream_ood): continual practice over Li's 20 OOD tasks.  A held
concept's practice-chosen source is stored per operator, keyed by (suite, concept words), and reused by later arrivals; only unseen
concepts are practised (experiments/prereg/g16_stream_ood.py).  Episodes are deterministic in (arm, start-state set, trial, seed),
so an arm already run on the same states by G10 / G14 / G15 is read from there; anything else is launched under g16_*.

    $PY experiments/g16_stream_ood.py --run --backbone pi0_libero --order plan|reversed   # sequential, waits on its episodes
    $PY experiments/g16_stream_ood.py --read [--record]
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
R = os.path.join(ROOT, "results", "robot")
SEED, LAM = 13, 24
PLAN = os.path.join(R, "g4_plan.json")
OPS = ("switch", "tli")


def armf(op, a, g):
    return f"pfield@{a}.{g}" if op == "switch" else f"tli@{a}.{g}.{LAM}"


def search(kind, op, sh):
    """Experiments that may already hold an arm's episodes on the right states, then G16's own."""
    old = {("s1", "switch"): "g10_s1", ("s2", "switch"): "g10_s2", ("s1", "tli"): "g14_s1", ("s2", "tli"): "g14_s2",
           ("acc", "switch"): "g10_acc", ("acc", "tli"): "g15_acc", ("main", "switch"): "g15_main", ("main", "tli"): "g15_main"}
    return [f"{old[(kind, op)]}_{sh}", f"g16_{kind}_{sh}"]


def find(exps, v, arm, backbone):
    import g4_ood as G
    for e in exps:
        z = G._ld(e, v, arm, backbone)
        if z is not None:
            return z
    return None


def ensure(kind, op, k, v, arms, trials, fresh, backbone):
    """-> {arm: episodes}; launches the arms no experiment holds yet and waits for them."""
    import g4_ood as G
    sh = k.split(":")[0]
    exps = search(kind, op, sh)
    miss = [a for a in arms if find(exps, v, a, backbone) is None]
    if miss:
        print(f"[g16] {k} {kind} {op}: launching {len(miss)} arms", G._launch(exps[-1], k, v, miss, trials, fresh, backbone), flush=True)
    t0 = time.time()
    while True:
        got = {a: find(exps, v, a, backbone) for a in arms}
        if None not in got.values():
            return got
        if time.time() - t0 > 6 * 3600:
            raise SystemExit(f"[g16] {k} {kind} {op}: arms still missing after 6 h: {[a for a, z in got.items() if z is None]}")
        time.sleep(120)


def order_of(name):
    ks = list(json.load(open(PLAN))["tasks"])
    return ks if name == "plan" else ks[::-1]


def run(backbone, order):
    import g4_ood as G
    pl, pool = json.load(open(PLAN)), G.source_pool()
    short = backbone.split("_")[0]
    store = {op: {"object": {}, "goal": {}} for op in OPS}
    log = []
    for k in order_of(order):
        v = pl["tasks"][k]
        oa, og = G.ours_sources(v, pl)
        ko, kg = f"{v['suite']}|{v['object_concept']}", f"{v['suite']}|{v['goal_concept']}"
        rec = dict(task=k, practice=0, reused=[])
        pair = {}
        for op in OPS:
            if ko in store[op]["object"]:
                a = store[op]["object"][ko]
                rec["reused"].append(f"{op}:object")
            else:
                sc = ensure("s1", op, k, v, [armf(op, c, og) for c in pool], 5, "fresh_ood_a", backbone)
                a = max(pool, key=lambda c: (sum(sc[armf(op, c, og)].values()), c == oa, -c))
                rec["practice"] += 5 * len(pool)
                store[op]["object"][ko] = a
            if kg in store[op]["goal"]:
                g = store[op]["goal"][kg]
                rec["reused"].append(f"{op}:goal")
            else:
                sc = ensure("s2", op, k, v, [armf(op, a, c) for c in pool], 5, "fresh_ood_a", backbone)
                g = max(pool, key=lambda c: (sum(sc[armf(op, a, c)].values()), c == og, -c))
                rec["practice"] += 5 * len(pool)
                store[op]["goal"][kg] = g
            pair[op] = armf(op, a, g)
        acc = {op: sum(ensure("acc", op, k, v, [pair[op]], 10, "fresh_ood_a", backbone)[pair[op]].values()) for op in OPS}
        rec["practice"] += 20
        rec.update(pairs=pair, acceptance=acc, operator="tli" if acc["tli"] > acc["switch"] else "switch")
        rec["served"] = pair[rec["operator"]]
        log.append(rec)
        print(f"[g16 {short} {order}] {k}: served {rec['served']} (acc {acc}), practice {rec['practice']}, reused {rec['reused']}", flush=True)
        json.dump(dict(log=log, store=store), open(os.path.join(R, f"g16_{short}_{order}.json"), "w"), indent=1)
    for rec in log:                                                     # the test on set e (after the stream: no feedback)
        k = rec["task"]
        ensure("main", rec["operator"], k, pl["tasks"][k], [rec["served"]], 20, "fresh_ood_e", backbone)
    print(f"[g16 {short} {order}] G16_DONE", flush=True)


def read(record):
    import e1_read as E1
    import g13_tli_sources as G13
    import g15_operator as G15
    pl = json.load(open(PLAN))
    C = E1.compare
    for backbone in ("pi0_libero", "pi05_libero"):
        short = backbone.split("_")[0]
        g15 = json.load(open(os.path.join(R, f"g15_read_{short}.json")))
        out = dict(backbone=backbone)
        for order in ("plan", "reversed"):
            f = os.path.join(R, f"g16_{short}_{order}.json")
            if not os.path.exists(f):
                continue
            st = json.load(open(f))
            x, missing = {}, []
            for rec in st["log"]:
                k = rec["task"]
                v, sh = pl["tasks"][k], k.split(":")[0]
                a15 = G15.arms(k, v, backbone)
                z = dict(stream=find(search("main", rec["operator"], sh), v, rec["served"], backbone),
                         g15=find([f"g15_main_{sh}"], v, a15[g15["choice"][k]], backbone),
                         zs=find([f"g15_main_{sh}"], v, "zs", backbone), tli_li=find([f"g15_main_{sh}"], v, G13.arms(k, v, backbone)["tli_li"], backbone))
                if None in z.values():
                    missing.append(k)
                    continue
                x[k] = z
            r = dict(missing=missing, tasks=len(x), practice=sum(rec["practice"] for rec in st["log"]),
                     practice_g15=len(pl["tasks"]) * (2 * 2 * 16 * 5 + 20),
                     reused=sum(len(rec["reused"]) for rec in st["log"]),
                     tli_served=sum(rec["operator"] == "tli" for rec in st["log"]),
                     same_as_g15=sum(rec["served"] == G15.arms(rec["task"], pl["tasks"][rec["task"]], backbone)[g15["choice"][rec["task"]]]
                                     for rec in st["log"]),
                     stream_vs_g15=C({k: {"x": z["stream"], "y": z["g15"]} for k, z in x.items()}),
                     stream_vs_zs=C({k: {"x": z["stream"], "y": z["zs"]} for k, z in x.items()}),
                     stream_vs_tli_li=C({k: {"x": z["stream"], "y": z["tli_li"]} for k, z in x.items()}))
            if x:
                r["stream_vs_g15"]["noninferior"] = bool(r["stream_vs_g15"]["diff"]["lo"] > -0.05)
                for n_ in ("stream_vs_g15", "stream_vs_zs", "stream_vs_tli_li"):
                    h = r[n_]
                    print(f"[g16 {short} {order}] {n_:16} {h['x']} vs {h['y']} of {h['n']}, discordant {h['discordant']}, sign p "
                          f"{h['sign_p']:.3g}, diff {h['diff']['diff']:+.3f} [{h['diff']['lo']:+.3f}, {h['diff']['hi']:+.3f}]", flush=True)
            print(f"[g16 {short} {order}] practice {r['practice']} vs {r['practice_g15']} per-task; reused {r['reused']}; TLI served "
                  f"{r['tli_served']}; same arm as G15 {r['same_as_g15']}; missing {missing}", flush=True)
            out[order] = r
        json.dump(out, open(os.path.join(R, f"g16_read_{short}.json"), "w"), indent=1)
        if record and all(o in out and not out[o]["missing"] for o in ("plan", "reversed")):
            from lifelong_ood import ledger
            print(ledger.record("g16_stream_ood", bench="libero_ood", tag=f"read_{short}", seed=SEED, metrics=out,
                                bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=dict(backbone=backbone),
                                script="experiments/g16_stream_ood.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    bb = sys.argv[sys.argv.index("--backbone") + 1] if "--backbone" in sys.argv else "pi0_libero"
    od = sys.argv[sys.argv.index("--order") + 1] if "--order" in sys.argv else "plan"
    if "--run" in sys.argv:
        run(bb, od)
    if "--read" in sys.argv:
        read("--record" in sys.argv)
