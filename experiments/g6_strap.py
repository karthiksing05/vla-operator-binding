"""PLAN G, READ G6: STRAP (Memmel et al., ICLR 2025, arXiv 2412.15182) against the library, on E3's tasks, start states and seed.

STRAP adapts a policy to a task from its demonstration plus sub-trajectories retrieved from prior data.  Here its prior is
everything the library could draw on and more: the 1,693 demonstrations of pi0.5's 40 training tasks (experiments/b_strap.py's
prior, embedded in results/robot/strap_emb.npz) and the ONE demonstration of every other LIBERO-90 task (the demonstrations the
library's prototypes are consolidated from).  STRAP's own rules (b_strap.py --official: segmentation by end-effector stops, S-DTW
with steps (1,1) and (1,2) over DINOv2-B features, K = 100 split across the sub-trajectories) pick the matches; a rank-16 LoRA on
the action expert is trained 2,000 steps on the demonstration and the matches, every frame under the target instruction.  The
STRAP arm gets the same acceptance test as the per-task LoRA (kept only if it beats the policy on stored states 0-9); its paired
episodes reuse E3's start states (fresh set b) and noise seed 13, so E3's arms need no rerun.

    stage dump      (ENV=openpi, GPU): target demos + LIBERO-90 prior demos, frames at 224  -> results/robot/strap90_frames.npz
    stage embed     (ENV=chunks, GPU): DINOv2-B features                                      -> results/robot/strap90_emb.npz
    stage retrieve  (ENV=chunks, GPU): S-DTW, the top K per task                              -> results/robot/strap90_<t>_retrieved.json
    stage train     (ENV=openpi, GPU): --task t, the LoRA                                     -> results/robot/strap90_<t>_lora.pt
    --launch-fit | --accept | --test | --read [--record]  (driver, login node)
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
R = os.path.join(ROOT, "results", "robot")
PY = os.path.expanduser("~/flash/miniconda3/envs/chunks/bin/python")
SEED, NAME, K = 13, "strap90", 100
E3PLAN = json.load(open(os.path.join(R, "e3_plan.json")))
TASKS = [int(t) for t, v in E3PLAN.items() if v["lora"]]           # the tasks with a demonstration (task 51 has none)


def _resize(imgs):
    from PIL import Image
    return np.stack([np.asarray(Image.fromarray(x).resize((224, 224), Image.BICUBIC)) for x in imgs])


def prior90():
    """-> {LIBERO-90 task: its one demonstration's episode index}, every task with a demonstration."""
    from lifelong_ood.policy import data as D
    d90 = D.Demos90()
    out = {}
    for u in range(90):
        e = d90.episodes_of_task(u)
        if e:
            out[u] = int(e[0])
    return out


def dump():
    from lifelong_ood.policy import data as D
    d90 = D.Demos90()
    out = {}
    for u, e in prior90().items():
        ep = d90.load(e)
        img, _ = d90.frames(e)
        out[f"img_{u}"], out[f"eef_{u}"] = _resize(img), ep["state"][:, :3]
        print(f"[g6] dump task {u} (episode {e}): {len(img)} frames", flush=True)
    np.savez_compressed(os.path.join(R, f"{NAME}_frames.npz"), **out)


def embed():
    import torch
    import timm
    m = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224).cuda().eval().half()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    Z = np.load(os.path.join(R, f"{NAME}_frames.npz"))
    out = {}
    with torch.no_grad():
        for k in Z.files:
            if k.startswith("img_"):
                x = Z[k]
                f = [m(((torch.as_tensor(x[i:i + 256], device="cuda").permute(0, 3, 1, 2).float() / 255.0 - mean) / std).half())
                     .float().cpu().numpy() for i in range(0, len(x), 256)]
                out["f_" + k[4:]] = np.concatenate(f).astype(np.float16)
                out["eef_" + k[4:]] = Z["eef_" + k[4:]]
    np.savez(os.path.join(R, f"{NAME}_emb.npz"), **out)
    print(f"[g6] embedded {len(out) // 2} LIBERO-90 demonstrations", flush=True)


def retrieve():
    import b_strap as S
    Z40, Z90 = np.load(os.path.join(R, "strap_emb.npz")), np.load(os.path.join(R, f"{NAME}_emb.npz"))
    ep40, F40 = Z40["ep"], Z40["f"].astype(np.float32)
    src = [("d40", int(e), F40[ep40 == e]) for e in np.unique(ep40)]
    p90 = prior90()
    for t in TASKS:
        q, eef = Z90[f"f_{t}"].astype(np.float32), Z90[f"eef_{t}"]
        pool = src + [("l90", p90[u], Z90[f"f_{u}"].astype(np.float32)) for u in sorted(p90) if u != t]
        lens = np.array([len(x[2]) for x in pool])
        P = np.zeros((len(pool), lens.max(), q.shape[1]), np.float32)
        for k, x in enumerate(pool):
            P[k, :lens[k]] = x[2]
        segs = S.segments_official(eef)
        per = int(K / len(segs))
        picks = []
        for a, b in segs:
            cost, st, en = S.sdtw21(q[a:b], P, lens)
            for k in np.argsort(cost)[:per]:
                picks.append(dict(segment=[int(a), int(b)], source=pool[k][0], episode=pool[k][1], start=int(st[k]), end=int(en[k]),
                                  cost=float(cost[k])))
        picks = sorted(picks, key=lambda x: x["cost"])
        json.dump(dict(task=t, demo_episode=p90[t], segments=[list(map(int, x)) for x in segs], K=K, picks=picks),
                  open(os.path.join(R, f"{NAME}_{t}_retrieved.json"), "w"), indent=1)
        n90 = sum(p["source"] == "l90" for p in picks)
        print(f"[g6] task {t}: {len(segs)} sub-trajectories, {len(picks)} retrieved ({n90} from LIBERO-90), "
              f"{sum(p['end'] - p['start'] + 1 for p in picks)} frames", flush=True)


def train(t, steps=2000, rank=16, lr=1e-4, batch=32, seed=0):
    sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
    from lifelong_ood import ledger
    from lifelong_ood.experts.lora import add_node
    from lifelong_ood.policy import adapt
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy.backbone import PolicyBackbone
    bk = PolicyBackbone("pi05_libero", device="cuda")
    norm = D.Norm.load()
    d90, dm = D.Demos90(), D.Demos()
    ret = json.load(open(os.path.join(R, f"{NAME}_{t}_retrieved.json")))
    ep = d90.load(ret["demo_episode"])
    T = len(ep["actions"])
    idx, _ = D.chunk_indices(T, bk.H)
    obs = [d90.observation(ep, i) for i in range(T)]
    rows = [D.pad_rows(D.to_rows(ep["actions"][idx], norm), bk.H)]
    picks = sorted(ret["picks"], key=lambda p: (p["source"], p["episode"], p["start"]))   # one decode per prior episode
    nmax = 3000 - T                                                      # host memory: at most 3,000 frames; every demo frame,
    tot = sum(p["end"] - p["start"] + 1 for p in picks)                  # the retrieved ones evenly thinned
    keep = set(np.linspace(0, tot - 1, min(tot, nmax)).round().astype(int).tolist())
    cache, g = {}, 0
    for p in picks:
        key = (p["source"], p["episode"])
        if key not in cache:
            cache.clear()
            cache[key] = dm.load(p["episode"], images=True) if p["source"] == "d40" else d90.load(p["episode"])
        pe = cache[key]
        pidx, _ = D.chunk_indices(len(pe["actions"]), bk.H)
        fr = []
        for i in range(p["start"], p["end"] + 1):
            if g in keep:
                fr.append(i)
            g += 1
        for i in fr:
            o = (dm if p["source"] == "d40" else d90).observation(pe, i)
            o["prompt"] = ep["task"]                                         # STRAP's single-task policy: the target instruction
            obs.append(o)
        if fr:
            rows.append(D.pad_rows(D.to_rows(pe["actions"][pidx[fr]], norm), bk.H))
    full = np.concatenate(rows)
    bank = adapt.PrefixBank(bk, obs, store="cpu")
    print(f"[g6] task {t}: demo {T} frames + retrieved {bank.n - T} frames", flush=True)
    state, banks = adapt.inject(bk, scope="expert")
    key = f"{NAME}_{t}"
    params = add_node(banks, key, rank)
    t1 = time.time()
    hist = adapt.train(bk, state, key, params, bank, full, steps=steps, lr=lr, batch=batch, seed=seed)
    path = os.path.join(R, f"{NAME}_{t}_lora.pt")
    nbytes = adapt.save(banks, key, path, meta=dict(task=t, rank=rank, scope="expert", retrieved=len(picks)))
    m = dict(task=t, demo_frames=int(T), retrieved_frames=int(bank.n - T), retrieved=len(picks),
             from_l90=sum(p["source"] == "l90" for p in picks), final_loss=hist["loss"][-1], first_loss=hist["loss"][0],
             train_secs=time.time() - t1)
    print(ledger.record("g6_strap", bench="libero_90", tag="fit", seed=seed, keys=dict(task=t), metrics=m,
                        bytes=dict(experts=nbytes, tree=0, codes=0, samples=0, teacher_copies=0, total=nbytes),
                        config=dict(steps=steps, rank=rank, lr=lr, batch=batch, K=K), script="experiments/g6_strap.py",
                        command=" ".join(sys.argv), artifacts=[path]))


def sb(script, env="openpi"):
    out = subprocess.run(["sbatch", "--parsable", "--exclude=hal,tars,t1000,vicki,johnny5,c3po", "--partition=overcap",
                          "--account=overcap", "--qos=short", f"--export=ALL,ENV={env},SCRIPT={script}", "slurm/jobs/robot_gpu.sbatch"],
                         capture_output=True, text=True, cwd=ROOT)
    return out.stdout.strip() or out.stderr.strip()


def launch(exp, trials, fresh):
    for t in TASKS:
        cmd = [PY, "experiments/launch_pack.py", "--tasks", str(t), "--arms", f"lora@{NAME}", "--experiment", exp, "--first", "0",
               "--trials", str(trials), "--seed", str(SEED), "--bind", "track", "--partition", "overcap", "--per", "1"]
        cmd += ["--fresh", "--fresh-dir", "fresh_states_b"] if fresh else []
        print(t, subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-120:], flush=True)


def read(record):
    import e1_read as E1
    import e3_read as E3
    e3 = json.load(open(os.path.join(R, "e3_read.json")))
    arm = f"lora@{NAME}"
    acc, test, kept_strap, missing = {}, {}, {}, []
    for t in map(str, TASKS):
        a = E3.load(f"g6_accept", t, arm)
        x = E3.load(f"g6_main", t, arm)
        za, zx = E3.load("e3_accept", t, "zs"), E3.load("e3_main", t, "zs")
        v = E3PLAN[t]
        lib = E3.load("e3_main", t, E3.arms(v, True)[e3["kept"][t]])
        lora = E3.load("e3_main", t, E3.arms(v, True)[e3["kept_lora"][t]])
        if None in (a, x, za, zx, lib, lora):
            missing.append(t)
            continue
        acc[t] = dict(strap=int(sum(a.values())), zs=int(sum(za.values())))
        kept_strap[t] = "strap" if acc[t]["strap"] > acc[t]["zs"] else "zs"
        test[t] = dict(strap=x, zs=zx, lib=lib, lora_acc=lora, strap_acc=x if kept_strap[t] == "strap" else zx)
    C = E1.compare
    h1 = C({t: {"x": z["lib"], "y": z["strap_acc"]} for t, z in test.items()})
    h1["noninferior"] = bool(h1["diff"]["lo"] > -0.05)
    desc = {"strap_vs_zs": C({t: {"x": z["strap"], "y": z["zs"]} for t, z in test.items()}),
            "strap_acc_vs_lora_acc": C({t: {"x": z["strap_acc"], "y": z["lora_acc"]} for t, z in test.items()}),
            "lib_vs_strap": C({t: {"x": z["lib"], "y": z["strap"]} for t, z in test.items()})}
    res = dict(missing=missing, acceptance=acc, kept_strap=kept_strap, H1=h1, descriptive=desc,
               per_task={t: {k: int(sum(z.values())) for k, z in x.items()} for t, x in test.items()})
    print(f"[g6] H1 library vs STRAP+acc: {h1['x']} vs {h1['y']} of {h1['n']}, diff {h1['diff']['diff']:+.3f} "
          f"[{h1['diff']['lo']:+.3f}, {h1['diff']['hi']:+.3f}] -> non-inferior {h1['noninferior']}", flush=True)
    for k, v in desc.items():
        print(f"[g6]   {k}: {v['x']} vs {v['y']} of {v['n']}, diff {v['diff']['diff']:+.3f}", flush=True)
    if missing:
        print("[g6] missing", missing, flush=True)
    json.dump(res, open(os.path.join(R, "g6_read.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("g6_strap", bench="libero_90", tag="read", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=dict(K=K),
                            script="experiments/g6_strap.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", choices=("dump", "embed", "retrieve", "train"))
    ap.add_argument("--task", type=int)
    ap.add_argument("--steps", type=int, default=2000)
    for f in ("--launch-fit", "--accept", "--test", "--read", "--record"):
        ap.add_argument(f, action="store_true")
    a = ap.parse_args()
    if a.stage:
        {"dump": dump, "embed": embed, "retrieve": retrieve, "train": lambda: train(a.task, steps=a.steps)}[a.stage]()
    if a.launch_fit:
        for t in TASKS:
            if not os.path.exists(os.path.join(R, f"{NAME}_{t}_lora.pt")):
                print(t, sb(f"experiments/g6_strap.py train --task {t} --steps {a.steps}"), flush=True)
    if a.accept:
        launch("g6_accept", 10, False)
    if a.test:
        launch("g6_main", 20, True)
    if a.read:
        read(a.record)
