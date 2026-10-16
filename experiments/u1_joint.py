"""U1: ONE learner on a joint stream of held-concept and new-concept tasks -- Li's 20 compositional OOD tasks (no demonstration
exists for any) and the 30 LIBERO-90 tasks of the W1 stream -- with both kinds of unit and W1's request rule.  Exact replay of
recorded episodes, no new decision rule.

Why a replay is exact: a pointer unit (the policy's own field under a held concept's training instruction, G4b) carries no
weights and is never refit, so it neither changes nor is changed by the prototypes the stream fits; its practice episodes on the
stream's practice states (stored states 0-9, noise seed 13: i6b_accept; Li's set a: g4_accept) are fixed whatever the library
holds.  Each LIBERO-90 arrival therefore sees W1's candidates plus its pointer candidate (I6b's plan; where its concepts share
words with held ones).  The rule: keep the best of {policy, prototype composition, pointer composition} in practice (ties to the
policy, then the prototypes); request the one demonstration only if the best is below 8 of 10.  The script checks that no
pointer changes a request (then W1's library, refits and later arrivals replay unchanged) and lists the arrivals whose KEPT unit
changes; those are served from the pointer's own test episodes (i6b_main, E3's test states set b).  Li's tasks are served as in
G4b (set c); none can request a demonstration (none exists).

    $PY experiments/u1_joint.py --read [--record]
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
R = os.path.join(ROOT, "results", "robot")


def read(record):
    import e1_read as E1
    import e3_read as E3
    import e4_stream as E4
    import g4_ood as G4
    w1 = json.load(open(os.path.join(R, "w1_state.json")))
    ip = json.load(open(os.path.join(R, "i6b_plan.json")))
    parm = {t: f"pfield@{v['o']}.{v['g']}" for t, v in ip.items()}
    pf_acc = {t: sum(E3.load("i6b_accept", t, a).values()) for t, a in parm.items()}
    g4b = json.load(open(os.path.join(R, "g4b_read.json")))
    pl = json.load(open(G4.PLAN))
    C = E1.compare
    res = {}
    for o, sv in E4.served(w1).items():
        arr = {str(r["task"]): r for r in w1[o]["arrivals"]}
        request_changed, kept_changed, x = [], [], {}
        for t, (fin, at_arrival) in sv.items():
            r = arr[t]
            best = max(r["zs"], r.get("acc1") or 0)
            f = pf_acc.get(t)
            if f is not None and r.get("demo") and f >= 8:
                request_changed.append(t)
            pointer_kept = f is not None and not r.get("demo") and f > best
            zs = E3.load("e3_main", t, "zs")
            if pointer_kept:
                kept_changed.append(t)
                served = E3.load("i6b_main", t, parm[t])
            else:
                served = zs if fin == "zs" else next((z for z in (E4.load(m, t, fin) for m in ("w1_main", "e4_main"))
                                                      if z is not None), None)
            if served is None or zs is None:
                raise SystemExit(f"[u1] {o} task {t}: no test episodes for {fin}")
            x[f"l90:{t}"] = {"x": served, "y": zs}
        for k, v in pl["tasks"].items():
            sh = k.split(":")[0]
            arms = G4.arms_b(v, pl)
            x[f"ood:{k}"] = {"x": G4.load(f"g4b_main_{sh}", v["suite"], v["index"], arms[g4b["kept"][k]]),
                             "y": G4.load(f"g4b_main_{sh}", v["suite"], v["index"], "zs")}
        h = C(x)
        res[o] = dict(H1=h, arrivals=len(x), demos=w1[o]["demos"] if isinstance(w1[o]["demos"], int) else len(w1[o]["demos"]),
                      requests_changed_by_pointers=request_changed, kept_changed_to_pointer=kept_changed,
                      l90=C({k: v for k, v in x.items() if k.startswith("l90")}), ood=C({k: v for k, v in x.items() if k.startswith("ood")}))
        print(f"[u1] order {o}: {len(x)} arrivals, {res[o]['demos']} demonstrations; requests changed by pointers {request_changed}; "
              f"kept unit changed to a pointer on {kept_changed}; served {h['x']} vs policy {h['y']} of {h['n']} (discordant "
              f"{h['discordant']}, sign p {h['sign_p']:.3g}); LIBERO-90 {res[o]['l90']['x']} vs {res[o]['l90']['y']}, OOD "
              f"{res[o]['ood']['x']} vs {res[o]['ood']['y']}", flush=True)
    json.dump(res, open(os.path.join(R, "u1_read.json"), "w"), indent=1)
    if record:
        from lifelong_ood import ledger
        print(ledger.record("u1_joint", bench="libero", tag="read", seed=13, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/u1_joint.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--read" in sys.argv:
        read("--record" in sys.argv)
