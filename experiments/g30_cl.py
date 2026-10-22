"""Plan D2: LIBERO's own lifelong baselines, EWC and PackNet, on the W1 stream, in G11's protocol (experiments/b_continual.py):
ONE rank-16 LoRA on pi0.5's action expert trained through W1's arrival order a, one demonstration per task (its own scene),
500 steps per task, lr 1e-4, batch 32; each demo frame prompted with its task's own instruction.

    ewc      -- online EWC as in LIBERO (libero/lifelong/algos/ewc.py, configs/lifelong/ewc.yaml): after each task the diagonal
                Fisher is the mean over batches of the squared gradient of the batch's flow loss (LIBERO squares the gradient of
                the batch mean NLL), decayed by gamma = 0.9 and summed over tasks; the loss adds e_lambda * sum F (theta -
                theta*)^2 with e_lambda = 50000 and theta* the parameters at the end of the previous task.  The final adapter
                serves every task (bcl_ewc<tag>.pt).
    packnet  -- PackNet as in LIBERO (libero/lifelong/algos/packnet.py): every weight of the adapter is owned by one task; a
                task trains the free weights (the others' gradients zeroed; LoRA A and B are the adapter's Linear weights), then
                prunes the lowest-magnitude fraction of what it trained back to free (zero), then fine-tunes its kept weights for
                --post-steps (LIBERO: post_prune_epochs = n_epochs = 50; it also keeps the best post-prune epoch by rollouts,
                which a one-demonstration protocol cannot, so the last is kept).  At test task t is served by the weights owned
                by tasks up to t (LIBERO's get_eval_algo): bcl_packnet<tag>_<task>_lora.pt, the arm lora@bcl_packnet<tag>.
                --prune equal (default): task i of N prunes 1 - 1/(N - i) of its trained weights, so every task keeps an equal
                share (1/N) of the adapter; --prune 0.75: LIBERO's prune_perc (on 30 tasks the last ones keep almost nothing).

Resumable task by task (results/robot/bcl_<mode><tag>.ckpt.pt).  Ledger row g30_cl.

    sbatch --export=ALL,ENV=openpi,SCRIPT="experiments/g30_cl.py --mode ewc --tasks 43+71+..." slurm/jobs/robot_gpu.sbatch
"""
import argparse
import importlib.util
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
from lifelong_ood import ledger                                        # noqa: E402
from lifelong_ood.experts.lora import add_node, keys_on                # noqa: E402
from lifelong_ood.policy import adapt                                  # noqa: E402
from lifelong_ood.policy import data as D                              # noqa: E402

R = os.path.join(ROOT, "results", "robot")


def fisher(bk, state, key, params, bank, chunks, batches=20, batch=32, seed=0):
    """LIBERO's EWC Fisher: mean over batches of (d mean-batch-loss / d theta)^2, on the task's own frames."""
    import torch
    gen = torch.Generator(device="cpu").manual_seed(10_000 + seed)
    rng = np.random.default_rng(10_000 + seed)
    A = torch.as_tensor(np.asarray(chunks, np.float32), device=bk.device)
    on = keys_on(state, {key})
    F = [torch.zeros_like(p) for p in params]
    for _ in range(batches):
        idx = rng.integers(0, bank.n, size=min(batch, bank.n))
        pre = bank.take(idx)
        a = A[idx]
        t, eps = adapt._flow_batch(len(idx), bk.H, bk.A, gen, bk.device)
        xt = t[:, None, None] * eps + (1 - t[:, None, None]) * a
        u = (eps - a).reshape(len(idx), -1)
        for p in params:
            p.grad = None
        with on():
            v = bk.v_full(xt, t, pre)
        adapt.flow_loss(bk, v, u, xt, t).backward()
        for f, p in zip(F, params):
            if p.grad is not None:
                f += p.grad.detach() ** 2
    for p in params:
        p.grad = None
    return [f / batches for f in F]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("ewc", "packnet"))
    ap.add_argument("--tasks", required=True, help="t1+t2+... in stream order")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--post-steps", type=int, default=500, help="packnet: fine-tuning steps after pruning")
    ap.add_argument("--prune", default="equal", help="packnet: equal | a fraction (LIBERO 0.75)")
    ap.add_argument("--e-lambda", type=float, default=50000.0)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--fisher-batches", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    import torch
    tasks = [int(x) for x in a.tasks.split("+")]
    from lifelong_ood.policy.backbone import PolicyBackbone
    bk = PolicyBackbone("pi05_libero", device="cuda")
    d90 = D.Demos90()
    norm = D.Norm.load()
    skipped = [t for t in tasks if not d90.episodes_of_task(t)]
    tasks = [t for t in tasks if t not in skipped]
    if skipped:
        print(f"[g30cl] skipped (no demonstration in the dataset): {skipped}", flush=True)
    data = {}
    for t in tasks:                                                           # the same rows as b_continual.py
        ep = d90.load(d90.episodes_of_task(t)[0])
        T = len(ep["actions"])
        idx, _ = D.chunk_indices(T, bk.H)
        data[t] = ([d90.observation(ep, i) for i in range(T)], D.pad_rows(D.to_rows(ep["actions"][idx], norm), bk.H))
    state, banks = adapt.inject(bk)
    key = f"bcl_{a.mode}"
    params = add_node(banks, key, a.rank)
    N = len(tasks)
    ckpt = os.path.join(R, f"bcl_{a.mode}{a.tag}.ckpt.pt")
    done, curves = 0, []
    fish = star = None
    owner = [torch.zeros_like(p, dtype=torch.int16) for p in params]           # packnet: 0 = free, k = task k (1-based)
    if os.path.exists(ckpt):
        c = torch.load(ckpt, map_location="cpu")
        with torch.no_grad():
            for p_, w in zip(params, c["params"]):
                p_.copy_(w.to(p_.device))
        done, curves = int(c["done"]), c.get("curves", [])
        if c.get("fish") is not None:
            fish = [f.to(bk.device) for f in c["fish"]]
            star = [s.to(bk.device) for s in c["star"]]
        if c.get("owner") is not None:
            owner = [o.to(bk.device) for o in c["owner"]]
        print(f"[g30cl] resumed {ckpt}: {done} tasks done", flush=True)
    t0 = time.time()
    log = lambda x: print(x, flush=True)                                       # noqa: E731
    for i, t in enumerate(tasks):
        if i < done:
            continue
        obs, rows = data[t]
        tb = adapt.PrefixBank(bk, obs)
        cur = {}
        if a.mode == "ewc":
            pen = None
            if fish is not None:
                def pen():
                    return a.e_lambda * sum((f * (p - s) ** 2).sum() for f, p, s in zip(fish, params, star))
            cur["train"] = adapt.train(bk, state, key, params, tb, rows, steps=a.steps, lr=a.lr, batch=a.batch, seed=i,
                                       penalty=pen, log=log)
            F = fisher(bk, state, key, params, tb, rows, batches=a.fisher_batches, batch=a.batch, seed=i)
            fish = F if fish is None else [a.gamma * f + g for f, g in zip(fish, F)]
            star = [p.detach().clone() for p in params]
        else:
            k = i + 1
            with torch.no_grad():
                for o in owner:
                    o[o == 0] = k                                              # start_task: the free weights become task k's
            mine = [o == k for o in owner]

            def hook():
                for p, m in zip(params, mine):
                    if p.grad is not None:
                        p.grad.mul_(m.to(p.grad.dtype))
            cur["train"] = adapt.train(bk, state, key, params, tb, rows, steps=a.steps, lr=a.lr, batch=a.batch, seed=i,
                                       grad_hook=hook, log=log)
            frac = (1.0 - 1.0 / (N - i)) if a.prune == "equal" else float(a.prune)
            with torch.no_grad():                                               # prune: per weight tensor, as LIBERO per layer
                for p, o in zip(params, owner):
                    m = o == k
                    n_ = int(m.sum())
                    r = round(frac * n_)
                    if r <= 0 or n_ == 0:
                        continue
                    cut = p[m].abs().flatten().float().cpu().kthvalue(r)[0].item()
                    rm = (p.abs() <= cut) & m
                    o[rm] = 0
                    p[rm] = 0.0
            mine = [o == k for o in owner]
            cur["kept"] = int(sum(int(m.sum()) for m in mine))
            cur["free"] = int(sum(int((o == 0).sum()) for o in owner))
            if a.post_steps:
                cur["post"] = adapt.train(bk, state, key, params, tb, rows, steps=a.post_steps, lr=a.lr, batch=a.batch,
                                          seed=1000 + i, grad_hook=hook, log=log)
            with torch.no_grad():                                               # the pruned weights stay exactly zero
                for p, o in zip(params, owner):
                    p[o == 0] = 0.0
        curves.append(cur)
        del tb
        print(f"[g30cl] {a.mode}: task {t} ({i + 1}/{N}) done at {time.time() - t0:.0f}s {({k_: v for k_, v in cur.items() if isinstance(v, int)})}",
              flush=True)
        torch.save(dict(params=[p_.detach().cpu() for p_ in params], done=i + 1, curves=curves,
                        fish=None if fish is None else [f.cpu() for f in fish], star=None if star is None else [s.cpu() for s in star],
                        owner=[o.cpu() for o in owner] if a.mode == "packnet" else None), ckpt)
    paths = []
    meta = dict(mode=a.mode, tasks=tasks, rank=a.rank, scope="expert", prune=a.prune)
    if a.mode == "ewc":
        path = os.path.join(R, f"bcl_ewc{a.tag}.pt")
        nb = adapt.save(banks, key, path, meta=meta)
        paths.append(path)
    else:
        full = [p.detach().clone() for p in params]
        nb = 0
        for i, t in enumerate(tasks):                                           # get_eval_algo: weights of tasks 1..i+1
            with torch.no_grad():
                for p, w, o in zip(params, full, owner):
                    p.copy_(w * ((o >= 1) & (o <= i + 1)).to(w.dtype))
            path = os.path.join(R, f"bcl_packnet{a.tag}_{t}_lora.pt")
            nb = adapt.save(banks, key, path, meta=dict(meta, served_task=t, position=i))
            paths.append(path)
        with torch.no_grad():
            for p, w in zip(params, full):
                p.copy_(w)
        torch.save(dict(owner=[o.cpu() for o in owner]), os.path.join(R, f"bcl_packnet{a.tag}_owner.pt"))
    print(f"[g30cl] saved {len(paths)} adapter(s), {nb / 1e6:.1f} MB each, in {time.time() - t0:.0f}s", flush=True)
    fl = [c["train"]["loss"][-1] for c in curves]
    print(ledger.record("g30_cl", bench="libero_90", tag=f"fit_{a.mode}{a.tag}", seed=0,
                        metrics=dict(mode=a.mode, tasks=tasks, final_losses=fl,
                                     kept=[c.get("kept") for c in curves], free=[c.get("free") for c in curves],
                                     post_final=[c["post"]["loss"][-1] for c in curves if "post" in c], secs=time.time() - t0),
                        bytes=dict(experts=nb, tree=0, codes=0, samples=0, teacher_copies=0, total=nb), config=vars(a),
                        script="experiments/g30_cl.py", command=" ".join(sys.argv), artifacts=paths))


if __name__ == "__main__":
    main()
