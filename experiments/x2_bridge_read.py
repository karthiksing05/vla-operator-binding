"""Plan X read X2 (pre-registered: experiments/prereg/x2_bridge_switch.py, ledger m3_prereg tag x2_bridge_switch).

    PYTHONPATH=. python3 experiments/x2_bridge_read.py record     # rollout JSONs -> ledger x2_bridge_switch (tag pi0, keys task, arm)
    PYTHONPATH=. python3 experiments/x2_bridge_read.py read [--record]   # the paired read (+ ledger x2_bridge_switch tag read)

Arms (experiments/x2_simpler_rollout.py, pi0 = open-pi-zero bridge_beta via OPZBackbone, stock protocol, 24 layouts x seeds 0-9):
  G1 (primary, H1): 'switch' (results/x2_simpler/switch/sw_<task>.json) vs pi0 alone from ledger x2_simpler_zs (its result file);
  G4 (descriptive), the registered 0.15 kg can variant (X2G4*Heavy-v0): 'zs' and 'switch' (results/x2_simpler/switch/{zs,sw}_<task>.json).
Episodes pair by (layout, seed): the driver seeds the flow noise per (seed, layout, query), so both arms draw the same noise at each
query index.  H1: pooled over the 3 G1 tasks' 720 pairs, switch > alone with the exact sign test on discordant pairs (alpha 0.05);
the difference with its task-level (two-level cluster) bootstrap interval, as experiments/e1_read.py's compare().
"""
import glob
import json
import os
import sys

REPO = "/coc/flash10/ksingara3/lifelong-ood-learner"
sys.path.insert(0, REPO)
from lifelong_ood import ledger, stats  # noqa: E402

RES = f"{REPO}/results/x2_simpler"
BYTES0 = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
G1 = ["X2G1GreenBlockOnPlate-v0", "X2G1CarrotOnYellowBlock-v0", "X2G1CarrotInBasket-v0"]
G4 = ["X2G4PepsiOnTowelHeavy-v0", "X2G4RedBullOnPlateHeavy-v0", "X2G4SevenUpOnTowelHeavy-v0"]


def outcomes(path):
    r = json.load(open(path))
    return r, {f"{x['seed']}:{x['episode']}": int(bool(x["success"])) for x in r["rows"]}


def describe(r):
    R = r["rows"]
    n = len(R)
    sw = [x["switched_at"] for x in R if x.get("switched_at") is not None]
    return dict(n=n, successes=sum(x["success"] for x in R),
                grasped=sum(x["stats"].get("is_src_obj_grasped", False) for x in R) / n,
                moved_correct=sum(x["stats"].get("moved_correct_obj", False) for x in R) / n,
                switch_fired=len(sw) / n if r.get("prompt_after") else None,
                switch_query_median=(sorted(sw)[len(sw) // 2] if sw else None),
                success_given_fired=(sum(x["success"] for x in R if x.get("switched_at") is not None) / len(sw) if sw else None))


def record():
    files = [(t, "switch", f"{RES}/switch/sw_{t}.json") for t in G1 + G4] + [(t, "zs", f"{RES}/switch/zs_{t}.json") for t in G4]
    for t, arm, f in files:
        if not os.path.exists(f):
            print("[x2 read] missing", f)
            continue
        r, oc = outcomes(f)
        if r["n"] != 240:
            print("[x2 read] incomplete", f, r["n"])
            continue
        print(ledger.record("x2_bridge_switch", bench="simpler_widowx", tag="pi0", keys=dict(task=t, arm=arm), seed=0,
                            metrics=dict(describe(r), success=r["success_rate"], outcomes=oc),
                            bytes=dict(BYTES0),
                            config=dict(checkpoint="allenzren/open-pi-zero bridge_beta", prompt_a=r["prompt"], prompt_b=r["prompt_after"],
                                        layouts=24, seeds=10, max_steps=r["max_episode_steps"], prereg="m3_prereg x2_bridge_switch",
                                        variant="mass 0.15 kg" if "Heavy" in t else None),
                            script="experiments/x2_simpler_rollout.py", command=r["command"], backbone="open-pi-zero/bridge_beta",
                            seconds=r["seconds"], artifacts=dict(result=os.path.relpath(f, REPO))))


def arm_table():
    tab = {}
    for row in ledger.load("x2_bridge_switch", tag="pi0"):
        tab.setdefault(row["keys"]["task"], {})[row["keys"]["arm"]] = row
    for row in ledger.load("x2_simpler_zs", tag="pi0"):
        t = row["keys"]["task"]
        if t in G1:
            r, oc = outcomes(os.path.join(REPO, row["artifacts"]["result"]))
            tab.setdefault(t, {})["zs"] = dict(metrics=dict(describe(r), outcomes=oc), _path=row["_path"])
    return tab


def compare(tab):
    pairs = stats.paired(tab, "x", "y", list(tab))
    d = stats.bootstrap_diff(pairs, B=10000)
    d.pop("samples", None)
    b, c = stats.discordant(pairs)
    return dict(x=int(sum(sum(v[0]) for v in pairs.values())), y=int(sum(sum(v[1]) for v in pairs.values())),
                n=int(sum(len(v[0]) for v in pairs.values())), tasks=len(pairs), diff=d, discordant=[int(b), int(c)],
                sign_p=float(stats.sign_test(b, c)))


def read(rec=False):
    tab = arm_table()
    out = dict(per_task={}, missing=[])
    for t in G1 + G4:
        a = tab.get(t, {})
        if "switch" not in a or "zs" not in a:
            out["missing"].append(t)
            continue
        x, y = a["switch"]["metrics"]["outcomes"], a["zs"]["metrics"]["outcomes"]
        one = compare({t: {"x": {k: int(v) for k, v in x.items()}, "y": {k: int(v) for k, v in y.items()}}})
        dx, dy = ({k: v for k, v in a[arm]["metrics"].items() if k != "outcomes"} for arm in ("switch", "zs"))
        out["per_task"][t] = dict(switch=one["x"], alone=one["y"], n=one["n"], discordant_switch_only_alone_only=one["discordant"],
                                  sign_p=one["sign_p"], diff=one["diff"]["diff"], switch_desc=dx, alone_desc=dy,
                                  grade="G1" if t in G1 else "G4 (0.15 kg variant)")
    def pooled(ts):
        ts = [t for t in ts if t in out["per_task"]]
        return compare({t: {"x": tab[t]["switch"]["metrics"]["outcomes"], "y": tab[t]["zs"]["metrics"]["outcomes"]} for t in ts}) if ts else None
    h1 = pooled(G1)
    if h1 is not None:
        h1["passes"] = bool(h1["x"] > h1["y"] and h1["sign_p"] < 0.05 and h1["tasks"] == 3 and h1["n"] == 720)
    out["H1_G1_pooled"] = h1
    out["G4_pooled_descriptive"] = pooled(G4)
    for t, v in out["per_task"].items():
        sd, ad = v["switch_desc"], v["alone_desc"]
        print(f"{t:30} {v['grade']:22} switch {v['switch']:3}/{v['n']}  alone {v['alone']:3}/{v['n']}  discordant "
              f"{v['discordant_switch_only_alone_only'][0]:3} vs {v['discordant_switch_only_alone_only'][1]:3}  sign p {v['sign_p']:.3g}  "
              f"| grasped {100 * sd['grasped']:.1f} vs {100 * ad['grasped']:.1f}  switch fired {100 * (sd['switch_fired'] or 0):.1f}% "
              f"(median query {sd['switch_query_median']})")
    for k in ("H1_G1_pooled", "G4_pooled_descriptive"):
        v = out[k]
        if v:
            print(f"{k}: switch {v['x']}/{v['n']} vs alone {v['y']}/{v['n']}  diff {100 * v['diff']['diff']:+.1f} pp "
                  f"[{100 * v['diff']['lo']:+.1f}, {100 * v['diff']['hi']:+.1f}] (task-level bootstrap 95%)  discordant {v['discordant']}  "
                  f"sign p {v['sign_p']:.3g}" + (f"  PASSES {v['passes']}" if "passes" in v else ""))
    if out["missing"]:
        print("missing", out["missing"])
    if rec and not out["missing"]:
        print(ledger.record("x2_bridge_switch", bench="simpler_widowx", tag="read", seed=0, metrics=out, bytes=dict(BYTES0),
                            config=dict(prereg="m3_prereg x2_bridge_switch", alpha=0.05, bootstrap_B=10000),
                            script="experiments/x2_bridge_read.py", command="PYTHONPATH=. python3 experiments/x2_bridge_read.py read --record"))
    return out


if __name__ == "__main__":
    if sys.argv[1] == "record":
        record()
    else:
        read("--record" in sys.argv)
