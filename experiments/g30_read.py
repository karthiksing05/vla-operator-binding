"""Plan D2's reads (docs/BASELINES.md; pre-registrations ledger m3_prereg tags g30_full_sft, g30_multidemo, g30_cl, g30_demodiffusion,
g30_sdp): every new arm against E3's library with acceptance and the per-task LoRA with acceptance (E3's set b, seed 13), and the
continual baselines against W1's served learner (order a, G11's protocol).  Episodes from experiments/g30_feed.py.

    ~/flash/miniconda3/envs/chunks/bin/python experiments/g30_read.py [--record]
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import e1_read as E1          # noqa: E402
import e3_read as E3          # noqa: E402
import e4_stream as E4        # noqa: E402

R = os.path.join(ROOT, "results", "robot")
SEED = 13
E3_ARMS = {"full": "fullft@_g30", "n5": "r3lora@_g30n5", "nall": "r3lora@_g30nall", "demodiff": "demodiff@g30", "sdp": "sdp@g30"}
CL_ARMS = {"ewc": "bcl@ewc_g30", "packnet": "lora@bcl_packnet_g30", "packnet75": "lora@bcl_packnet75_g30"}


def C(tab):
    if not tab:
        return dict(x=0, y=0, n=0, tasks=0, diff=dict(diff=0.0, lo=0.0, hi=0.0), discordant=[0, 0], sign_p=1.0)
    return E1.compare(tab)


def line(k, h):
    return (f"{k:22} {h['x']:4d} vs {h['y']:4d} of {h['n']} ({h['tasks']} tasks), discordant {h['discordant']}, sign p {h['sign_p']:.2g}, "
            f"diff {h['diff']['diff']:+.3f} [{h['diff']['lo']:+.3f}, {h['diff']['hi']:+.3f}]")


def e3_part():
    plan = json.load(open(os.path.join(R, "e3_plan.json")))
    e3 = json.load(open(os.path.join(R, "e3_read.json")))
    test, acc, kept, missing = {}, {}, {}, {}
    for t, v in plan.items():
        if not v["lora"]:
            continue                                                       # task 51: no demonstration (served by the policy)
        A = E3.arms(v, True)
        zs, za = E3.load("e3_main", t, "zs"), E3.load("e3_accept", t, "zs")
        cur = dict(zs=zs, lib=E3.load("e3_main", t, A[e3["kept"][t]]), lora=E3.load("e3_main", t, A["lora"]))
        cur["lora_acc"] = cur["lora"] if e3["kept_lora"][t] == "lora" else zs
        acc[t], kept[t] = {"zs": int(sum(za.values()))}, {}
        for k, arm in E3_ARMS.items():
            sa, sx = E3.load("g30_accept", t, arm), E3.load("g30_main", t, arm)
            if sa is None or sx is None:
                missing.setdefault(k, []).append(t)
                continue
            acc[t][k] = int(sum(sa.values()))
            kept[t][k] = k if acc[t][k] > acc[t]["zs"] else "zs"
            cur[k], cur[k + "_acc"] = sx, (sx if kept[t][k] == k else zs)
        test[t] = cur
    H = {}

    def cmp(x, y, sub=None):
        return C({t: {"x": z[x], "y": z[y]} for t, z in test.items() if x in z and y in z and (sub is None or t in sub)})
    for k in E3_ARMS:
        if not any(k in z for z in test.values()):
            continue
        H[f"lib_vs_{k}_acc"] = cmp("lib", k + "_acc")
        H[f"{k}_acc_vs_lora_acc"] = cmp(k + "_acc", "lora_acc")
        H[f"{k}_vs_zs"] = cmp(k, "zs")
        H[f"{k}_acc_vs_zs"] = cmp(k + "_acc", "zs")
    if any("n5" in z for z in test.values()):
        H["n5_acc_vs_lora_acc"]["H1_multidemo_passes"] = bool(H["n5_acc_vs_lora_acc"]["x"] > H["n5_acc_vs_lora_acc"]["y"]
                                                               and H["n5_acc_vs_lora_acc"]["sign_p"] < 0.05)
    for k, h in H.items():
        if k.startswith("lib_vs_"):
            h["noninferior"] = bool(h["diff"]["lo"] > -0.05)
            h["superior_sign"] = bool(h["x"] > h["y"] and h["sign_p"] < 0.05)
    by_grade = {}
    for t, z in test.items():
        g = plan[t].get("grade") or E1_grade(t)
        d = by_grade.setdefault(g, {})
        for k, x in z.items():
            d[k] = d.get(k, 0) + int(sum(x.values()))
    return dict(hypotheses=H, acceptance=acc, kept=kept, missing=missing, by_grade=by_grade,
                per_task={t: {k: int(sum(x.values())) for k, x in z.items()} for t, z in test.items()})


GRADES = None


def E1_grade(t):
    global GRADES
    if GRADES is None:
        GRADES = {}
        f = os.path.join(R, "split_grades.json")
        if os.path.exists(f):
            GRADES = {str(k): v["grade"] for k, v in json.load(open(f))["grades"].items()}
    return GRADES.get(str(t), "?")


def cl_part():
    w1 = json.load(open(os.path.join(R, "w1_state.json")))
    sv = E4.served(w1)["a"]
    x, missing = {}, []
    for t, (fin, _) in sv.items():
        zs = E3.load("e3_main", t, "zs")
        served = zs if fin == "zs" else next((z for z in (E4.load(m, t, fin) for m in ("w1_main", "e4_main")) if z is not None), None)
        z = dict(w1=served, zs=zs, seq=E3.load("g11_main", t, "bcl@seq_g11"), er=E3.load("g11_main", t, "bcl@er_g11"),
                 **{k: E3.load("g30_clmain", t, a) for k, a in CL_ARMS.items()})
        if None in z.values():
            missing.append(t)
            continue
        x[t] = z
    H = {}
    for k in CL_ARMS:
        H[f"w1_vs_{k}"] = C({t: {"x": z["w1"], "y": z[k]} for t, z in x.items()})
        H[f"{k}_vs_zs"] = C({t: {"x": z[k], "y": z["zs"]} for t, z in x.items()})
        H[f"{k}_vs_seq"] = C({t: {"x": z[k], "y": z["seq"]} for t, z in x.items()})
        H[f"{k}_vs_er"] = C({t: {"x": z[k], "y": z["er"]} for t, z in x.items()})
    ps = sorted([(H["w1_vs_ewc"]["sign_p"], "w1_vs_ewc"), (H["w1_vs_packnet"]["sign_p"], "w1_vs_packnet")]) if x and "w1_vs_ewc" in H else []
    for i, (p, k) in enumerate(ps):                                        # Holm over H1, H2
        H[k]["holm_p"] = min(1.0, p * (len(ps) - i))
        H[k]["passes"] = bool(H[k]["x"] > H[k]["y"] and all(H[kk]["sign_p"] * (len(ps) - j) < 0.05 for j, (_, kk) in enumerate(ps[:i + 1])))
    order = w1["a"]["order"]
    first, last = order[:10], order[-10:]
    pos = {k: dict(first10=int(sum(sum(x[str(t)][k].values()) for t in first if str(t) in x)),
                   last10=int(sum(sum(x[str(t)][k].values()) for t in last if str(t) in x))) for k in ("w1", "zs", "seq", "er", *CL_ARMS)}
    return dict(hypotheses=H, missing=missing, by_position=pos, per_task={t: {k: int(sum(v.values())) for k, v in z.items()} for t, z in x.items()})


def main(record):
    res = dict(e3=e3_part(), cl=cl_part())
    for part in ("e3", "cl"):
        for k, h in res[part]["hypotheses"].items():
            if h["n"]:
                print(f"[g30] {line(k, h)}" + (f" noninf {h['noninferior']}" if "noninferior" in h else "")
                      + (f" holm {h['holm_p']:.2g} passes {h['passes']}" if "holm_p" in h else ""), flush=True)
    print("[g30] missing e3", {k: len(v) for k, v in res["e3"]["missing"].items()}, "cl", res["cl"]["missing"], flush=True)
    print("[g30] W1 by position", res["cl"]["by_position"], flush=True)
    json.dump(res, open(os.path.join(R, "g30_read.json"), "w", encoding="utf-8"), indent=1)
    if record:
        from lifelong_ood import ledger
        for tag, ok in (("e3_arms", True), ("cl", not res["cl"]["missing"])):
            if ok:
                print(ledger.record("g30_baselines", bench="libero_90", tag=f"read_{tag}", seed=SEED, metrics=res["e3" if tag == "e3_arms" else "cl"],
                                    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                                    script="experiments/g30_read.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    main("--record" in sys.argv)
