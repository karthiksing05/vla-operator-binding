"""Plan G, read G12: Li et al.'s Text Latent Interpolation (TLI) on our harness (novelty audit, docs/NOVELTY.md section 3.3 item
1: Li report pi0 0.09 -> 0.83 on their 20 compositional OOD tasks, above their own prompt switch).  The text latents are
identified on openpi's PyTorch pi0 by Li's procedure (lifelong_ood/policy/tli.py); the base-task pairs and lambda are Li's
task_hidden_states_mapping (examples/libero/main.py at 587a6cb, below).  Tested on start states set c (20 per task, noise seed
13), paired with G4d's policy alone and pi0^S, and with G10's practice-chosen pair.

    $PY experiments/g12_tli.py --identify [--backbone pi0_libero]     # GPU job: the 9 base tasks' latents -> results/robot/tli/
    $PY experiments/g12_tli.py --launch | --read [--record]
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
R = os.path.join(ROOT, "results", "robot")
PY = os.path.expanduser("~/flash/miniconda3/envs/chunks/bin/python")
SEED = 13
# Li's latent files avg_states_<i>_<i+1>: the training task index i of physical-intelligence/libero (checked against our Demos)
KEY = dict(put_wine=14, put_cheese=13, plate=10, stove=17, cabinet=18, bowl=13, center_bowl=38, next_plate_bowl=36, cookie_bowl=35,
           ramkin_bowl=32)
MAP = dict(                                                             # Li's task_hidden_states_mapping, the 20 OOD tasks
    put_the_cream_cheese_on_the_plate=("put_cheese", "plate", 24), put_the_cream_cheese_on_the_stove=("put_cheese", "stove", 24),
    put_the_cream_cheese_on_top_of_the_cabinet=("put_cheese", "cabinet", 24), put_the_wine_bottle_in_the_bowl=("put_wine", "bowl", 12),
    put_the_wine_bottle_on_the_plate=("put_wine", "plate", 24), put_the_wine_bottle_on_the_stove=("put_wine", "stove", 24),
    put_the_orange_juice_on_the_stove=("put_wine", "stove", 24), put_the_tomato_sauce_on_top_of_the_cabinet=("put_cheese", "cabinet", 24),
    put_the_bbq_source_on_the_plate=("put_cheese", "plate", 24), put_the_cream_cheese_in_the_basket=("put_cheese", "stove", 24),
    put_the_bowl_at_table_center_on_the_stove=("center_bowl", "stove", 24),
    put_the_bowl_at_table_center_on_the_cabinet=("center_bowl", "cabinet", 24),
    put_the_bowl_next_to_the_plate_on_the_stove=("next_plate_bowl", "stove", 30),
    put_the_bowl_next_to_the_plate_on_the_cabinet=("next_plate_bowl", "cabinet", 24),
    put_the_bowl_on_cookie_box_on_the_stove=("cookie_bowl", "stove", 24), put_the_bowl_on_cookie_box_on_the_cabinet=("cookie_bowl", "cabinet", 14),
    put_the_orange_juice_on_the_plate=("put_wine", "ramkin_bowl", 24), put_the_milk_on_the_plate=("put_wine", "ramkin_bowl", 24),
    put_the_chocolate_pudding_on_the_plate=("put_cheese", "cookie_bowl", 24), put_the_butter_on_the_plate=("put_cheese", "cookie_bowl", 24))


def arm(v):
    a, b, lam = MAP[v["language"].replace(" ", "_")]
    return f"tli@{KEY[a]}.{KEY[b]}.{lam}"


def identify(backbone):
    import numpy as np
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy import tli as TL
    from lifelong_ood.policy.backbone import PolicyBackbone
    bk = PolicyBackbone(backbone, device="cuda")
    d = D.Demos()
    os.makedirs(os.path.join(R, "tli"), exist_ok=True)
    need = set(KEY.values())                                            # and G10's practice-chosen sources (G13)
    for b in ("pi0", "pi05"):
        f10 = os.path.join(R, f"g10_read_{b}.json")
        if os.path.exists(f10):
            need |= {int(i) for a in json.load(open(f10))["chosen"].values() for i in a.split("@")[1].split(".")}
    import g4_ood as G4                                                 # G14: every candidate source
    need |= set(G4.source_pool())
    for t in sorted(need):
        f = TL.path(backbone, t, ROOT)
        if os.path.exists(f):
            continue
        lat, n = TL.identify(bk, d, t)
        np.save(f, lat)
        print(f"[g12] task {t} '{d.tasks[t]}': {n} frames, latent {lat.shape}, norms by layer "
              f"{[round(float(x), 1) for x in np.linalg.norm(lat, axis=-1).mean(-1)[::4]]}", flush=True)


def launch(backbone):
    pl = json.load(open(os.path.join(R, "g4_plan.json")))
    for k, v in pl["tasks"].items():
        cmd = [PY, "experiments/launch_pack.py", "--tasks", str(v["index"]), "--arms", arm(v), "--experiment",
               f"g12_main_{k.split(':')[0]}", "--first", "0", "--trials", "20", "--seed", str(SEED), "--bind", "track",
               "--partition", "overcap", "--per", "1", "--suite", v["suite"], "--fresh", "--fresh-dir", "fresh_ood_c"]
        cmd += ["--backbone", backbone] if backbone != "pi05_libero" else []
        print(k, arm(v), subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-90:], flush=True)


def read(backbone, record):
    import e1_read as E1
    import g4_ood as G
    pl = json.load(open(os.path.join(R, "g4_plan.json")))
    short = backbone.split("_")[0]
    g10 = json.load(open(os.path.join(R, f"g10_read_{short}.json")))
    main_zs = "g4c_main" if backbone != "pi05_libero" else "g4b_main"
    x, missing = {}, []
    for k, v in pl["tasks"].items():
        sh = k.split(":")[0]
        la = G.li_arms(v, pl)
        z = dict(tli=G._ld(f"g12_main_{sh}", v, arm(v), backbone), zs=G._ld(f"{main_zs}_{sh}", v, "zs", backbone),
                 li_fixed=G._ld(f"g4d_main_{sh}", v, la["li_fixed"], backbone), li_grasp=G._ld(f"g4d_main_{sh}", v, la["li_grasp"], backbone),
                 sel=G._ld(f"g10_main_{sh}", v, g10["chosen"][k], backbone))
        if None in z.values():
            missing.append(k)
            continue
        x[k] = z
    C = E1.compare
    res = dict(backbone=backbone, missing=missing, tasks=len(x), arms={k: arm(v) for k, v in pl["tasks"].items()},
               tli_vs_zs=C({k: {"x": z["tli"], "y": z["zs"]} for k, z in x.items()}),
               sel_vs_tli=C({k: {"x": z["sel"], "y": z["tli"]} for k, z in x.items()}),
               tli_vs_li_fixed=C({k: {"x": z["tli"], "y": z["li_fixed"]} for k, z in x.items()}),
               tli_vs_li_grasp=C({k: {"x": z["tli"], "y": z["li_grasp"]} for k, z in x.items()}),
               by_suite={s: {a: int(sum(sum(z[a].values()) for k, z in x.items() if pl["tasks"][k]["suite"] == s)) for a in
                             ("tli", "zs", "li_fixed", "li_grasp", "sel")} for s in ("libero_goal_ood", "libero_spatial_ood")},
               per_task={k: {a: int(sum(v.values())) for a, v in z.items()} for k, z in x.items()})
    res["sel_vs_tli"]["noninferior"] = bool(res["sel_vs_tli"]["diff"]["lo"] > -0.05)
    for n_ in ("tli_vs_zs", "sel_vs_tli", "tli_vs_li_fixed", "tli_vs_li_grasp"):
        h = res[n_]
        print(f"[g12 {short}] {n_:16} {h['x']} vs {h['y']} of {h['n']}, discordant {h['discordant']}, sign p {h['sign_p']:.3g}, "
              f"diff {h['diff']['diff']:+.3f} [{h['diff']['lo']:+.3f}, {h['diff']['hi']:+.3f}]", flush=True)
    print(f"[g12 {short}] by suite {res['by_suite']} | missing {missing}", flush=True)
    json.dump(res, open(os.path.join(R, f"g12_read_{short}.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("g12_tli", bench="libero_ood", tag=f"read_{short}", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=dict(backbone=backbone),
                            script="experiments/g12_tli.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    bb = sys.argv[sys.argv.index("--backbone") + 1] if "--backbone" in sys.argv else "pi0_libero"
    if "--identify" in sys.argv:
        identify(bb)
    if "--launch" in sys.argv:
        launch(bb)
    if "--read" in sys.argv:
        read(bb, "--record" in sys.argv)
