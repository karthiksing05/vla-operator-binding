"""X2 parity summary: read the SimplerEnv WidowX runs (X-VLA client logs, open-pi-zero eval logs), print the tables, write ledger rows.

    python experiments/x2_simpler_summary.py [--ledger]
X-VLA: results/x2_simpler/xvla_WidowX_<tag>_<job>/<task>/widowx_results.txt (one JSON row per episode; the authors' client
breaks on the first success within 1200 steps, so 'steps' also gives success within the stock 60 / 120-step budget).
pi0: slurm/<job>.out of slurm/jobs/x2_pi0_simpler.sbatch ('[x2pi0] ... ckpt= task= nep=' header, 'Success rate:' footer).
"""
import argparse
import glob
import json
import os
import re

REPO = "/coc/flash10/ksingara3/lifelong-ood-learner"
RES = f"{REPO}/results/x2_simpler"
XV_TASKS = {"spoon": ("widowx_spoon_on_towel", 60), "carrot": ("widowx_carrot_on_plate", 60),
            "blocks": ("widowx_stack_cube", 60), "eggplant": ("widowx_put_eggplant_in_basket", 120)}
XV_PUB = {"spoon": 100.0, "carrot": 91.7, "blocks": 95.8, "eggplant": 95.8}
PI0_PUB = {  # open-pi-zero README, visual matching, float32, 240 trials per task
    "bridge_beta": {"widowx_carrot_on_plate": 55.8, "widowx_put_eggplant_in_basket": 85.4, "widowx_spoon_on_towel": 84.6, "widowx_stack_cube": 47.9},
    "bridge_uniform": {"widowx_carrot_on_plate": 58.8, "widowx_put_eggplant_in_basket": 79.2, "widowx_spoon_on_towel": 63.3, "widowx_stack_cube": 21.3},
}
BYTES0 = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)


def xvla_runs():
    out = {}
    for d in sorted(glob.glob(f"{RES}/xvla_WidowX_run*_*")):
        tag = os.path.basename(d)
        r = {}
        for t, (_, lim) in XV_TASKS.items():
            f = f"{d}/{t}/widowx_results.txt"
            if not os.path.exists(f):
                continue
            rows = [json.loads(l) for l in open(f) if l.strip()]
            ok = [x for x in rows if "done" in x]
            if len(ok) < 24:
                continue
            r[t] = dict(n=len(ok), succ=sum(x["done"] for x in ok), succ_in_budget=sum(x["done"] and x["steps"] <= lim for x in ok),
                        sec=sum(x["duration_sec"] for x in ok))
        if len(r) == 4:
            out[tag] = r
    return out


def pi0_runs():
    out = {}
    for f in sorted(glob.glob(f"{REPO}/slurm/*.out")):
        txt = open(f, errors="replace").read()
        m = re.search(r"\[x2pi0\] .* ckpt=(\S+) task=(\S+) nep=(\d+) bf16=(\S+)", txt)
        s = re.search(r"Success rate: ([0-9.]+)", txt)
        n = re.search(r"Number of episodes: (\d+)", txt)
        dt = re.search(r"\[x2pi0\] done in (\d+)s", txt)
        if m and s and n and int(n.group(1)) == int(m.group(3)) and int(m.group(3)) >= 240:
            out[(m.group(1), m.group(2), m.group(4), os.path.basename(f)[:-4])] = dict(rate=float(s.group(1)), n=int(n.group(1)),
                                                                                      sec=int(dt.group(1)) if dt else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", action="store_true")
    a = ap.parse_args()
    xv, p0 = xvla_runs(), pi0_runs()
    print("X-VLA-WidowX (authors' client: 24 episodes, success at any step within 1200; 'budget' = within the stock 60/120 steps)")
    for tag, r in xv.items():
        cells = " ".join(f"{t} {100*v['succ']/v['n']:.1f} ({100*v['succ_in_budget']/v['n']:.1f})" for t, v in r.items())
        avg = sum(100 * v['succ'] / v['n'] for v in r.values()) / 4
        avgb = sum(100 * v['succ_in_budget'] / v['n'] for v in r.values()) / 4
        print(f"  {tag}: {cells} | avg {avg:.1f} (budget {avgb:.1f})")
    print("open-pi-zero (240 trials per task, success at the 60/120-step timeout)")
    for (ck, task, bf, job), v in p0.items():
        print(f"  {ck} bf16={bf} {task}: {100*v['rate']:.1f} (published {PI0_PUB[ck][task]}) n={v['n']} {v['sec']}s job {job}")
    if not a.ledger:
        return
    from lifelong_ood import ledger
    if xv:
        metrics = {}
        for t in XV_TASKS:
            vals = [100 * r[t]["succ"] / r[t]["n"] for r in xv.values()]
            valsb = [100 * r[t]["succ_in_budget"] / r[t]["n"] for r in xv.values()]
            metrics[f"{t}_success"] = sum(vals) / len(vals)
            metrics[f"{t}_success_runs"] = vals
            metrics[f"{t}_success_stock_budget"] = sum(valsb) / len(valsb)
            metrics[f"{t}_published"] = XV_PUB[t]
        metrics["avg_success"] = sum(metrics[f"{t}_success"] for t in XV_TASKS) / 4
        metrics["avg_success_stock_budget"] = sum(metrics[f"{t}_success_stock_budget"] for t in XV_TASKS) / 4
        metrics["avg_published"] = 95.8
        metrics["runs"] = list(xv)
        ledger.record("x2_simpler_parity", bench="simpler_widowx", tag="xvla", seed=0, metrics=metrics, bytes=dict(BYTES0),
                      config=dict(checkpoint="2toINF/X-VLA-WidowX", simpler_fork="255isWhite/SimplerEnv (ManiSkill2_real2sim 54ae2e0, abs EE)",
                                  episodes_per_task=24, max_steps=1200, early_stop_on_success=True, domain_id=0, denoise_steps=10,
                                  visual_matching=True),
                      script="slurm/jobs/x2_xvla_simpler.sbatch", backbone="2toINF/X-VLA-WidowX",
                      command="sbatch --export=ALL,MODEL=2toINF/X-VLA-WidowX,SUITE=WidowX,TASKS=spoon:carrot:blocks:eggplant,TAG=runK slurm/jobs/x2_xvla_simpler.sbatch",
                      notes="docs/SIMPLER.md; eggplant scene differs from stock SimplerEnv in the 255isWhite fork (camera + robot init moved)")
    for ck in PI0_PUB:
        rows = {task: v for (c, task, bf, job), v in p0.items() if c == ck and bf == "False"}
        if len(rows) < 4:
            continue
        metrics = {f"{task}_success": 100 * v["rate"] for task, v in rows.items()}
        metrics.update({f"{task}_published": PI0_PUB[ck][task] for task in rows})
        metrics["avg_success"] = sum(100 * v["rate"] for v in rows.values()) / 4
        metrics["avg_published"] = sum(PI0_PUB[ck].values()) / 4
        ledger.record("x2_simpler_parity", bench="simpler_widowx", tag="pi0", keys=dict(ckpt=ck), seed=0, metrics=metrics, bytes=dict(BYTES0),
                      config=dict(checkpoint=f"allenzren/open-pi-zero {ck}", simpler_fork="allenzren/SimplerEnv (proprio)", trials_per_task=240,
                                  act_steps=4, horizon_steps=4, use_bf16=False, use_torch_compile=True, visual_matching=True,
                                  tokenizer="paligemma tokenizer.model rebuilt from the openpi big_vision copy (sha256-identical)"),
                      script="slurm/jobs/x2_pi0_simpler.sbatch", backbone=f"open-pi-zero/{ck}",
                      command=f"sbatch --export=ALL,CKPT={ck},TASK=<task>,NEP=240,BF16=False slurm/jobs/x2_pi0_simpler.sbatch",
                      notes="docs/SIMPLER.md")
    gdirs = sorted(glob.glob(f"{RES}/xvla_google-VM_run*_*"))
    pub = {"coke_can": 98.3, "move_near": 97.5, "open_close": 74.5, "place_in": 63.8}
    for d in gdirs:
        m = {}
        for t in pub:
            f = f"{d}/{t}/google_results.txt"
            rows = [json.loads(l) for l in open(f) if l.strip()] if os.path.exists(f) else []
            ok = [x for x in rows if "done" in x]
            if ok:
                m[f"{t}_success"] = 100 * sum(x["done"] for x in ok) / len(ok)
                m[f"{t}_episodes"] = len(ok)
                m[f"{t}_published"] = pub[t]
        if len(m) == 12:
            m["avg_success"] = sum(m[f"{t}_success"] for t in pub) / 4
            m["avg_published"] = 83.5
            ledger.record("x2_simpler_parity", bench="simpler_google_vm", tag="xvla", seed=0, metrics=m, bytes=dict(BYTES0),
                          config=dict(checkpoint="2toINF/X-VLA-Google-Robot", simpler_fork="255isWhite/SimplerEnv", setting="visual matching",
                                      max_steps="2x stock", early_stop_on_success=True),
                          script="slurm/jobs/x2_xvla_simpler.sbatch", backbone="2toINF/X-VLA-Google-Robot",
                          command="sbatch --export=ALL,MODEL=2toINF/X-VLA-Google-Robot,SUITE=google-VM,TASKS=coke_can:move_near:open_close:place_in,TAG=run0 slurm/jobs/x2_xvla_simpler.sbatch",
                          notes="docs/SIMPLER.md; " + os.path.basename(d))
    print("ledger rows written")


if __name__ == "__main__":
    main()
