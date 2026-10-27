"""X2: pi0-alone (open-pi-zero bridge_beta through OPZBackbone) baselines on the graded tasks (+ the 4 stock tasks through the same
driver), from results/x2_simpler/zs/zs_<task>.json (experiments/x2_simpler_rollout.py) -> table + ledger x2_simpler_zs (tag pi0, key task).
    PYTHONPATH=. python3 experiments/x2_simpler_zs_record.py [--ledger]
"""
import glob
import json
import math
import os
import sys

REPO = "/coc/flash10/ksingara3/vla-operator-binding"
BYTES0 = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
STOCK_PUB = {"widowx_carrot_on_plate": 55.8, "widowx_put_eggplant_in_basket": 85.4, "widowx_spoon_on_towel": 84.6, "widowx_stack_cube": 47.9}


def main():
    sys.path.insert(0, REPO)
    sys.path.insert(0, f"{REPO}/experiments")
    rows = []
    for f in sorted(glob.glob(f"{REPO}/results/x2_simpler/zs/zs_*.json")):
        r = json.load(open(f))
        n, k = r["n"], r["successes"]
        p = k / n
        R = r["rows"]
        grasp = sum(x["stats"].get("is_src_obj_grasped", False) for x in R) / n
        moved = sum(x["stats"].get("moved_correct_obj", False) for x in R) / n
        per_seed = {}
        for x in R:
            per_seed.setdefault(x["seed"], []).append(x["success"])
        rows.append(dict(task=r["task"], n=n, success=100 * p, se=100 * math.sqrt(p * (1 - p) / n), grasped=100 * grasp,
                         moved_correct=100 * moved, seeds=len(per_seed), per_seed=[100 * sum(v) / len(v) for _, v in sorted(per_seed.items())],
                         seconds=r["seconds"], max_steps=r["max_episode_steps"], command=r["command"], file=os.path.relpath(f, REPO)))
    for x in rows:
        pub = STOCK_PUB.get(x["task"])
        print(f"{x['task']:32} n={x['n']:4} success {x['success']:5.1f} +- {x['se']:.1f}  grasped {x['grasped']:5.1f}  moved-correct "
              f"{x['moved_correct']:5.1f}  {x['seconds'] / 60:.0f} min" + (f"  (authors' eval, published {pub})" if pub else ""))
    if "--ledger" not in sys.argv:
        return
    from lifelong_ood import ledger
    meta = {   # experiments/x2_simpler_graded.py GRADED (copied: importing it pulls in SAPIEN, kept off the login node)
        "X2G1GreenBlockOnPlate-v0": dict(grade="G1", pair="green block -> plate", instruction="put the green block on the plate"),
        "X2G1CarrotOnYellowBlock-v0": dict(grade="G1", pair="carrot -> yellow block", instruction="put carrot on the yellow block"),
        "X2G1CarrotInBasket-v0": dict(grade="G1", pair="carrot -> basket", instruction="put carrot into yellow basket"),
        "X2G4PepsiOnTowel-v0": dict(grade="G4", pair="pepsi can -> towel", instruction="put the pepsi can on the towel"),
        "X2G4RedBullOnPlate-v0": dict(grade="G4", pair="redbull can -> plate", instruction="put the red bull can on the plate"),
        "X2G4SevenUpOnTowel-v0": dict(grade="G4", pair="7up can -> towel", instruction="put the 7up can on the towel")}
    for x in rows:
        m = meta.get(x["task"], {})
        ledger.record("x2_simpler_zs", bench="simpler_widowx", tag="pi0", keys=dict(task=x["task"]), seed=0,
                      metrics=dict(success=x["success"], se=x["se"], n=x["n"], grasped=x["grasped"], moved_correct_obj=x["moved_correct"],
                                   success_per_seed=x["per_seed"], published_stock=STOCK_PUB.get(x["task"])),
                      bytes=dict(BYTES0),
                      config=dict(checkpoint="allenzren/open-pi-zero bridge_beta", backbone="lifelong_ood/policy/backbone_opz.py OPZBackbone",
                                  simpler_fork="allenzren/SimplerEnv (stock scenes)", layouts=24, seeds=x["seeds"], max_steps=x["max_steps"],
                                  act_steps=4, dtype="float32", grade=m.get("grade", "stock"), pair=m.get("pair"),
                                  instruction=m.get("instruction")),
                      script="experiments/x2_simpler_rollout.py", command=x["command"], backbone="open-pi-zero/bridge_beta",
                      seconds=x["seconds"], artifacts=dict(result=x["file"]), notes="docs/SIMPLER.md (pi0 zero-shot on the graded tasks)")
    print("ledger rows written")


if __name__ == "__main__":
    main()
