"""Plan D2, read G30g: the adapted Sparse Diffusion Policy (lifelong_ood/policy/sdp.py; the user's decision 4): per task, SDP's
top-2 routers over the frozen per-task one-demonstration LoRAs of every OTHER LIBERO-90 task that has one (launch_pack._lorasc;
the task's own adapter is left out, so the router must compose held experts), trained on the task's one own-scene demonstration
(1,000 steps, lr 1e-3, batch 32, flow loss + 0.01 x SDP's load-balancing term).  Saves results/robot/sdp_<task>_g30_router.pt
(arm sdp@g30); ledger row g30_sdp tag fit, per task.  Read with E3's acceptance and set b (experiments/g30_feed.py --extra sdp).

    $PY experiments/g30_sdp.py --submit [--tasks a+b] [--per-job k]
    python experiments/g30_sdp.py --job --tasks a+b       # inside a GPU job (openpi venv)
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
R = os.path.join(ROOT, "results", "robot")
EXCL = "--exclude=hal,tars,t1000,vicki,johnny5,c3po,olivaw"
STEPS, LR, BATCH, K, AUX = 1000, 1e-3, 32, 2, 0.01


def _mod(name, f):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "experiments", f))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def pool(target):
    lp = _mod("lp", "launch_pack.py")
    out = []
    for t in range(90):
        if t == target:
            continue
        p = lp._lorasc(t)
        if p is not None:
            out.append((t, p))
    return out


def path_of(t):
    return os.path.join(R, f"sdp_{t}_g30_router.pt")


def fit(t, bk, norm, c0, r3p_texts):
    import numpy as np
    import torch
    from lifelong_ood import ledger
    from lifelong_ood.policy import adapt, sdp
    from lifelong_ood.policy import data as D
    experts = pool(t)
    d90 = D.Demos90()
    ep_idx = d90.episodes_of_task(t)[0]
    ep = d90.load(ep_idx)
    T = len(ep["actions"])
    bank = adapt.PrefixBank(bk, [d90.observation(ep, i) for i in range(T)])
    idx, _ = D.chunk_indices(T, bk.H)
    full = D.pad_rows(c0.chunk_rows(ep, idx, range(T), norm), bk.H, bk.a, bk.A)
    Rt, state, banks = sdp.build(bk, [p for _, p in experts], k=K, seed=0)
    params = list(Rt.routers.parameters())
    torch.manual_seed(0)
    gen = torch.Generator(device="cpu").manual_seed(0)
    rng = np.random.default_rng(0)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    Afull = torch.as_tensor(np.asarray(full, np.float32), device=bk.device)
    hist = dict(step=[], loss=[], aux=[])
    t0 = time.time()
    Rt.collect_aux = True
    for s in range(1, STEPS + 1):
        ix = rng.integers(0, bank.n, size=min(BATCH, bank.n))
        pre = bank.take(ix)
        a = Afull[ix]
        tt, eps = adapt._flow_batch(len(ix), bk.H, bk.A, gen, bk.device)
        xt = tt[:, None, None] * eps + (1 - tt[:, None, None]) * a
        u = (eps - a).reshape(len(ix), -1)
        Rt.aux = []
        v = bk.v_full(xt, tt, pre)
        loss = adapt.flow_loss(bk, v, u, xt, tt)
        aux = torch.stack(Rt.aux).mean() if Rt.aux else torch.zeros((), device=bk.device)
        opt.zero_grad(set_to_none=True)
        (loss + AUX * aux).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if s % 100 == 0 or s == 1:
            hist["step"].append(s); hist["loss"].append(float(loss)); hist["aux"].append(float(aux))
            print(f"[sdp] task {t} step {s}/{STEPS}: flow {float(loss):.4f} aux {float(aux):.3f} ({time.time() - t0:.0f}s)", flush=True)
    Rt.collect_aux = False
    meta = dict(task=t, experts=[p for _, p in experts], expert_tasks=[e for e, _ in experts], k=K, steps=STEPS, lr=LR, aux=AUX,
                demo_episode=int(ep_idx))
    tmp = path_of(t) + ".tmp"
    Rt.save(tmp, meta)
    os.replace(tmp, path_of(t))
    nbytes = int(sum(p.numel() for p in params) * 2)
    secs = time.time() - t0
    print(ledger.record("g30_sdp", bench="libero_90", tag="fit", seed=0, keys=dict(task=t),
                        metrics=dict(task=t, n_experts=len(experts), expert_tasks=meta["expert_tasks"], first_loss=hist["loss"][0],
                                     final_loss=hist["loss"][-1], curve=hist, train_secs=secs, n_router_params=int(sum(p.numel() for p in params))),
                        bytes=dict(experts=nbytes, tree=0, codes=0, samples=0, teacher_copies=0, total=nbytes), config=dict(meta),
                        script="experiments/g30_sdp.py", command=" ".join(sys.argv), artifacts=[path_of(t)]), flush=True)
    Rt.detach()
    del Rt, bank
    torch.cuda.empty_cache()


def job(tasks):
    c0 = _mod("c0", "c0_fit.py")
    bk, norm = c0.backbone("pi05_libero")
    for t in tasks:
        if os.path.exists(path_of(t)):
            print(f"[sdp] {t}: exists", flush=True)
            continue
        fit(t, bk, norm, c0, None)


def submit(tasks, per_job):
    todo = [t for t in tasks if not os.path.exists(path_of(t))]
    for i in range(0, len(todo), per_job):
        part = todo[i:i + per_job]
        args = ["sbatch", "--parsable", EXCL, "--partition=overcap", "--account=overcap", "--qos=short", "--time=04:00:00",
                "--job-name=g30sdp", f"--export=ALL,ENV=openpi,SCRIPT=experiments/g30_sdp.py --job --tasks {'+'.join(map(str, part))}",
                "slurm/jobs/robot_gpu.sbatch"]
        r = subprocess.run(args, capture_output=True, text=True, cwd=ROOT)
        print(part, r.stdout.strip() or r.stderr.strip(), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--job", action="store_true")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--per-job", type=int, default=10)
    a = ap.parse_args()
    ts = ([int(x) for x in a.tasks.split("+")] if a.tasks else
          sorted(int(t) for t in json.load(open(os.path.join(R, "e3_plan.json"))) if int(t) != 51))
    if a.job:
        job(ts)
    if a.submit:
        submit(ts, a.per_job)
