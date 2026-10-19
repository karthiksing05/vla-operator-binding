"""Plan H3, read G21 (ledger m3_prereg tag g21_operators): two more operators for held concepts as baselines, with the same source
pairs as the grasp switch and TLI, so the operator is the only thing that varies (lifelong_ood/policy/held_ops.py):

  gpc     Cao et al.'s General Policy Composition: w v(.|instruction a) + (1 - w) v(.|instruction g) at every Euler step;
          w in {0.3, 0.5, 0.7} chosen by practice (5 episodes each on start states set a; most successes, ties to 0.5, then the
          lower w).  gpc@a.g.ph (w = 1 before the grasp, 0 from it) is the grasp switch through the composition path: a check,
          10 episodes on set a paired with G10's acceptance episodes of pfield@a.g.
  inject  Grant et al.'s activation injection into the action expert, mirrored on TLI (the source-mean difference at every expert
          layer input; object source before the grasp, goal source from it).

Pairs: G10's practice-chosen pair (results/robot/g10_read_<short>.json "chosen"; sel) and Li et al.'s pair (G4d's li_grasp; li).
Test on set e (20 per task, noise seed 13), paired with G15's files on set e (g15_main: the policy alone, Li's TLI, the grasp switch
with the chosen pair, the practice-chosen operator); the grasp switch with Li's pair and TLI with the chosen pair run here when
G15 has no file for them.

    $PY experiments/g21_operators.py --identify --backbone pi05_libero [--tasks 14,17]   # GPU (openpi venv): expert latents
    $PY experiments/g21_operators.py --run --backbone pi05_libero [--no-identify] [--resume]   # identify, practice, choice, test; waits
    $PY experiments/g21_operators.py --read --backbone pi05_libero [--record]
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
R = os.path.join(ROOT, "results", "robot")
SEED, LAM = 13, 24
WS = (30, 50, 70)
PLAN = os.path.join(R, "g4_plan.json")
EXCLUDE = "hal,tars,t1000,vicki,johnny5,c3po,olivaw"


def _launch(exp, k, v, arms, trials, fresh, backbone):
    """g4_ood._launch, then the packs it submitted are kept off the excluded nodes (launch_pack passes no --exclude)."""
    import re
    import g4_ood as G
    out = subprocess.run([G.PY, "experiments/launch_pack.py", "--tasks", str(v["index"]), "--arms", ",".join(arms), "--experiment",
                          exp, "--first", "0", "--trials", str(trials), "--seed", str(SEED), "--bind", "track", "--partition", "overcap",
                          "--per", "4", "--suite", v["suite"], "--fresh", "--fresh-dir", fresh]
                         + (["--backbone", backbone] if backbone != "pi05_libero" else []),
                         capture_output=True, text=True, cwd=ROOT).stdout.strip()
    for j in re.findall(r"-> job (\d+)", out):
        subprocess.run(["scontrol", "update", f"JobId={j}", f"ExcNodeList={EXCLUDE}"], capture_output=True)
    return out[-60:]


def pairs(k, v, backbone):
    import g4_ood as G
    g10 = json.load(open(os.path.join(R, f"g10_read_{backbone.split('_')[0]}.json")))
    pl = json.load(open(PLAN))
    sp = lambda arm: tuple(int(x) for x in arm.split("@")[1].split(".")[:2])   # noqa: E731
    return dict(sel=sp(g10["chosen"][k]), li=sp(G.li_arms(v, pl)["li_grasp"]))


def needed(backbone):
    pl = json.load(open(PLAN))
    return sorted({t for k, v in pl["tasks"].items() for p in pairs(k, v, backbone).values() for t in p})


def identify(backbone, tasks=None):
    import numpy as np
    from g12_tli import make_backbone
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy import held_ops as HO
    bk = make_backbone(backbone)
    print(f"[g21] {bk.describe()}, {len(HO.expert_layers(bk))} expert layers", flush=True)
    d = D.Demos()
    for t in (tasks or needed(backbone)):
        f = HO.path(backbone, t, ROOT)
        if os.path.exists(f):
            continue
        os.makedirs(os.path.dirname(f), exist_ok=True)
        lat, n = HO.identify(bk, d, t)
        tmp = f"{f}.tmp{os.getpid()}.npy"
        np.save(tmp, lat)
        os.replace(tmp, f)
        print(f"[g21] task {t} '{d.tasks[t]}': {n} frames, latent {lat.shape}, |M| by layer (step 0) "
              f"{[round(float(x), 1) for x in np.linalg.norm(lat[0], axis=-1).mean(-1)[::4]]}", flush=True)
    print("[g21] IDENTIFY_DONE", flush=True)


def sb_identify(backbone, tasks=None):
    script = f"experiments/g21_operators.py --identify --backbone {backbone}" + (f" --tasks {'+'.join(map(str, tasks))}" if tasks else "")   # sbatch --export splits on commas
    out = subprocess.run(["sbatch", "--parsable", f"--exclude={EXCLUDE}", "--partition=overcap",
                          "--account=overcap", "--qos=short", f"--export=ALL,ENV=openpi,SCRIPT={script}", "slurm/jobs/robot_gpu.sbatch"],
                         capture_output=True, text=True, cwd=ROOT)
    return out.stdout.strip() or out.stderr.strip()


def latents_ready(backbone):
    from lifelong_ood.policy import held_ops as HO
    return all(os.path.exists(HO.path(backbone, t, ROOT)) for t in needed(backbone))


def practice_arms(k, v, backbone):
    return list(dict.fromkeys(f"gpc@{a}.{g}.{w}" for a, g in pairs(k, v, backbone).values() for w in WS))


def choose_w(backbone):
    """-> {task: {"sel": w, "li": w}} (None where practice is incomplete)."""
    import g4_ood as G
    pl, out = json.load(open(PLAN)), {}
    for k, v in pl["tasks"].items():
        out[k] = {}
        for p, (a, g) in pairs(k, v, backbone).items():
            sc = {w: G._ld(f"g21_s1_{k.split(':')[0]}", v, f"gpc@{a}.{g}.{w}", backbone) for w in WS}
            out[k][p] = None if None in sc.values() else max(WS, key=lambda w: (sum(sc[w].values()), w == 50, -w))
    return out


def test_arms(k, v, backbone, ws):
    """-> {name: arm} on set e; the new operators and the descriptive comparators."""
    import g15_operator as G15
    p = pairs(k, v, backbone)
    g15 = G15.arms(k, v, backbone)
    (sa, sg), (la, lg) = p["sel"], p["li"]
    return dict(gpc_sel=f"gpc@{sa}.{sg}.{ws['sel']}", gpc_li=f"gpc@{la}.{lg}.{ws['li']}", inj_sel=f"inject@{sa}.{sg}",
                inj_li=f"inject@{la}.{lg}", zs=g15["zs"], tli_li=g15["tli_li"], switch_sel=g15["psel"], switch_li=f"pfield@{la}.{lg}",
                tli_sel=f"tli@{sa}.{sg}.{LAM}")


def ld_e(k, v, arm, backbone):
    """An arm's set-e episodes: G15's file if it has one, else this read's."""
    import g4_ood as G
    sh = k.split(":")[0]
    z = G._ld(f"g15_main_{sh}", v, arm, backbone)
    return z if z is not None else G._ld(f"g21_main_{sh}", v, arm, backbone)


def _wait(cond, what):
    while not cond():
        time.sleep(300)
    print(f"[g21] {what} ready {time.strftime('%F %T')}", flush=True)


def run(backbone):
    import g4_ood as G
    pl = json.load(open(PLAN))
    short = backbone.split("_")[0]
    if not latents_ready(backbone) and "--no-identify" not in sys.argv:     # --no-identify: the identify job is already queued
        print(f"[g21 {short}] identify -> job {sb_identify(backbone)}", flush=True)
    for k, v in pl["tasks"].items():
        if "--resume" in sys.argv:                                     # --resume: practice and the phase check already queued
            break
        sh = k.split(":")[0]
        a, g = pairs(k, v, backbone)["sel"]
        print(k, "practice", _launch(f"g21_s1_{sh}", k, v, practice_arms(k, v, backbone), 5, "fresh_ood_a", backbone), flush=True)
        print(k, "phase check", _launch(f"g21_ph_{sh}", k, v, [f"gpc@{a}.{g}.ph"], 10, "fresh_ood_a", backbone), flush=True)
    _wait(lambda: all(None not in c.values() for c in choose_w(backbone).values()), "practice")
    ws = choose_w(backbone)
    print(f"[g21 {short}] chosen w {json.dumps(ws)}", flush=True)
    _wait(lambda: latents_ready(backbone), "latents")
    for k, v in pl["tasks"].items():
        sh = k.split(":")[0]
        todo = [arm for arm in dict.fromkeys(test_arms(k, v, backbone, ws[k]).values())
                if G._ld(f"g15_main_{sh}", v, arm, backbone) is None]
        print(k, todo, _launch(f"g21_main_{sh}", k, v, todo, 20, "fresh_ood_e", backbone), flush=True)
    print(f"[g21 {short}] G21_TEST_LAUNCHED {time.strftime('%F %T')}", flush=True)

    def done():
        return all(ld_e(k, v, arm, backbone) is not None for k, v in pl["tasks"].items()
                   for arm in test_arms(k, v, backbone, ws[k]).values())
    _wait(done, "test")
    read(backbone, False)
    print(f"[g21 {short}] G21_DONE backbone={backbone} read=results/robot/g21_read_{short}.json {time.strftime('%F %T')}", flush=True)


def read(backbone, record):
    import e1_read as E1
    import g15_operator as G15
    import g4_ood as G
    pl = json.load(open(PLAN))
    short = backbone.split("_")[0]
    ws = choose_w(backbone)
    ch, _ = G15.choice(backbone)
    x, missing, ph = {}, [], {}
    for k, v in pl["tasks"].items():
        sh = k.split(":")[0]
        if None in ws[k].values() or ch[k] is None:
            missing.append(k)
            continue
        ta = test_arms(k, v, backbone, ws[k])
        z = {n: ld_e(k, v, arm, backbone) for n, arm in ta.items()}
        z["chosen"] = ld_e(k, v, G15.arms(k, v, backbone)[ch[k]], backbone)
        a, g = pairs(k, v, backbone)["sel"]
        ph[k] = dict(gpc_ph=G._ld(f"g21_ph_{sh}", v, f"gpc@{a}.{g}.ph", backbone), switch=G._ld(f"g10_acc_{sh}", v, f"pfield@{a}.{g}", backbone))
        if None in z.values():
            missing.append(k)
            continue
        x[k] = z
    C = E1.compare
    cmp = lambda a, b: C({k: {"x": z[a], "y": z[b]} for k, z in x.items()})   # noqa: E731
    res = dict(backbone=backbone, missing=missing, w=ws, choice=ch, tasks=len(x))
    for op in ("gpc", "inj"):
        res[f"H1_{op}_sel_vs_chosen"] = h1 = cmp(f"{op}_sel", "chosen")
        h1["noninferior"] = bool(h1["diff"]["lo"] > -0.05)
        res[f"H2_{op}_sel_vs_zs"] = h2 = cmp(f"{op}_sel", "zs")
        h2["passes"] = bool(h2["x"] > h2["y"] and h2["sign_p"] < 0.05)
        for b in ("switch_sel", "tli_sel"):
            res[f"{op}_sel_vs_{b}"] = cmp(f"{op}_sel", b)
        for b in ("switch_li", "tli_li", "zs"):
            res[f"{op}_li_vs_{b}"] = cmp(f"{op}_li", b)
    pk = {k: d for k, d in ph.items() if None not in d.values()}
    agree = [int(d["gpc_ph"][t] == d["switch"][t]) for d in pk.values() for t in d["gpc_ph"] if t in d["switch"]]
    res["phase_check"] = dict(tasks=len(pk), episodes=len(agree), agree=int(sum(agree)),
                              gpc_ph=int(sum(sum(d["gpc_ph"].values()) for d in pk.values())),
                              switch=int(sum(sum(d["switch"][t] for t in d["gpc_ph"] if t in d["switch"]) for d in pk.values())))
    res["totals"] = {n: int(sum(sum(z[n].values()) for z in x.values())) for n in (next(iter(x.values())) if x else {})}
    res["per_task"] = {k: {n: int(sum(e.values())) for n, e in z.items()} for k, z in x.items()}
    for n_, h in res.items():
        if isinstance(h, dict) and "sign_p" in h:
            print(f"[g21 {short}] {n_:24} {h['x']} vs {h['y']} of {h['n']}, discordant {h['discordant']}, sign p {h['sign_p']:.3g}, "
                  f"diff {h['diff']['diff']:+.3f} [{h['diff']['lo']:+.3f}, {h['diff']['hi']:+.3f}]", flush=True)
    print(f"[g21 {short}] phase check {res['phase_check']} | totals {res['totals']} | missing {missing}", flush=True)
    json.dump(res, open(os.path.join(R, f"g21_read_{short}.json"), "w", encoding="utf-8"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("g21_operators", bench="libero_ood", tag=f"read_{short}", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=dict(backbone=backbone),
                            script="experiments/g21_operators.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    bb = sys.argv[sys.argv.index("--backbone") + 1] if "--backbone" in sys.argv else "pi05_libero"
    if "--identify" in sys.argv:
        identify(bb, [int(x) for x in sys.argv[sys.argv.index("--tasks") + 1].replace("+", ",").split(",")] if "--tasks" in sys.argv else None)
    if "--run" in sys.argv:
        run(bb)
    if "--read" in sys.argv:
        read(bb, "--record" in sys.argv)
