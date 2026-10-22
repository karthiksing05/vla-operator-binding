"""B1 stage (e): fine-tuning reference curves on one dev test cell, from ONE demonstration.

Arms (DESIGN section 4, "what the store replaces"):
  lora4, lora16  rank-r LoRA on every transformer Linear + a new task code z; base frozen
  full           full SFT of every parameter + z
  mtsft          the per-task ceiling: the base policy trained further on all training cells PLUS all kept demos of
                 the cell (GPU; tau = 0.8 x its success, DESIGN section 5)
The one-demo arms share step 0: z = the training embedding with the lowest flow loss on the demo (the
"nearest-embedding" zero-shot; the demo is the query, never executed or scored).  Checkpoints at
0/25/50/100/200/400/800 steps; 20 episodes each; results/b1/finetune/<cell>/<arm>.json, resumable per checkpoint.

    sbatch ... --export=ALL,SCRIPT=experiments/b1_finetune.py,ARGS="--cell Jaco_Box_GoalWall_Push --arm lora4" ...
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lifelong_ood import b1, b1_data, b1_eval  # noqa: E402
from lifelong_ood.b1_policy import H, FlowPolicy, n_params  # noqa: E402

STEPS = (0, 25, 50, 100, 200, 400, 800)


def load_base(path, dev="cpu"):
    ck = torch.load(path, map_location=dev)
    meta = ck["meta"]
    m = FlowPolicy(**{k: meta["cfg"][k] for k in ("state_dim", "n_tasks", "cond", "d", "layers", "heads", "emb_dim")})
    m.load_state_dict(ck["ema"])
    return m.to(dev), b1_data.Norm.from_dict(meta["norm"]), meta


def demo_tensors(obs, act, norm, dev="cpu"):
    s, c = b1_data.chunks(norm.state(obs)[None], norm.act(act)[None])
    return torch.tensor(s, dtype=torch.float32, device=dev), torch.tensor(c, dtype=torch.float32, device=dev)


@torch.no_grad()
def nearest_embedding(model, s, c, draws=8, seed=0):
    """Index of the training embedding whose policy gives the demo the lowest flow loss (common noise draws)."""
    E = model.task_emb.weight
    g = torch.Generator(device=s.device).manual_seed(seed)
    losses = torch.zeros(E.shape[0])
    for _ in range(draws):
        x0 = torch.randn(c.shape, device=s.device, generator=g)
        t = torch.rand(c.shape[0], device=s.device, generator=g)
        for k in range(E.shape[0]):
            losses[k] += model.loss(s, c, z=E[k].expand(s.shape[0], -1), x0=x0, t=t).item()
    return int(losses.argmin()), (losses / draws).tolist()


def trainable_state(model, z, arm):
    """What a checkpoint must keep: the adapters and z (LoRA), or every weight in fp16 (full SFT)."""
    if arm.startswith("lora"):
        sd = {k: v for k, v in model.state_dict().items() if k.endswith(".A") or k.endswith(".B")}
    else:
        sd = {k: v.half() for k, v in model.state_dict().items()}
    return dict(z=z.detach().clone(), params=sd)


def read():
    """Aggregate results/b1/finetune/*/<arm>.json -> one ledger row b1_finetune per arm (mean success per grade
    and checkpoint; the bytes block charges the per-task trainable parameters at fp32, z included)."""
    from lifelong_ood import ledger
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((b1.RESULTS / "finetune").glob("*/*.json"))]
    rows = [r for r in rows if r.get("done")]
    for arm in ("lora4", "lora16", "full", "mtsft"):
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        curve = {}
        for r in rs:
            for k, v in r["ckpts"].items():
                curve.setdefault(r["grade"][0], {}).setdefault(k, []).append(v["success"])
        m = {g: {k: dict(n=len(v), mean=float(np.mean(v))) for k, v in sorted(d.items(), key=lambda x: int(x[0]))}
             for g, d in curve.items()}
        cpu_s = float(sum(v["train_s"] + v["eval_s"] for r in rs for v in r["ckpts"].values()))
        per_task = int(4 * rs[0].get("trainable_params", 0))
        print(arm, json.dumps(m))
        print(ledger.record("b1_finetune", bench="composuite", tag="dev", seed=0, keys=dict(arm=arm),
                            metrics=dict(curve=m, n_cells=len(rs), seconds=cpu_s,
                                         tau={r["cell"] and "_".join(r["cell"]): r.get("tau") for r in rs} if arm == "mtsft" else None),
                            bytes=dict(experts=per_task * len(rs), tree=0, codes=0, samples=0, teacher_copies=0,
                                       total=per_task * len(rs)),
                            config=dict(steps=STEPS, episodes=len(next(iter(rs[0]["ckpts"].values()))["episodes"])),
                            script="experiments/b1_finetune.py", command=" ".join(sys.argv),
                            artifacts=[str(b1.RESULTS / "finetune")]))


def run_cell(a, cell_name):
    cell = tuple(cell_name.split("_"))
    split = b1.load_split()
    grade = [g for g, cs in split["dev"].items() if cell in cs]
    assert a.kind != "expert" or grade, f"{cell_name} is not a dev test cell"
    torch.set_num_threads(4)
    dev = "cuda" if (a.arm == "mtsft" and torch.cuda.is_available()) else "cpu"
    out = b1.RESULTS / "finetune" / cell_name
    out.mkdir(parents=True, exist_ok=True)
    resf = out / f"{a.arm}.json"
    res = json.loads(resf.read_text(encoding="utf-8")) if resf.exists() else dict(cell=list(cell), arm=a.arm,
                                                                                 grade=grade, ckpts={})
    if res.get("done"):
        print(cell_name, a.arm, "already done")
        return
    model, norm, meta = load_base(a.base, dev)
    if a.kind == "expert":
        obs, act, suc = b1_data.load_cell(cell)
    else:
        z = np.load(b1.DATA / a.kind / cell_name / "episodes.npz")
        obs, act, suc = b1_data.strip_task_id(z["obs"], cell), z["act"], z["success"]
    demo_i = int(np.flatnonzero(suc)[0]) if suc.any() else 0     # the first successful expert episode
    s, c = demo_tensors(obs[demo_i], act[demo_i], norm, dev)
    k0, l0 = nearest_embedding(model, s, c)
    res.update(demo_index=demo_i, nearest=dict(index=k0, cell=meta["cells"][k0], loss=l0[k0]))
    z = torch.nn.Parameter(model.task_emb.weight[k0].detach().clone()[None])
    t_train = 0.0
    if a.arm == "mtsft":
        from experiments.b1_policy import Batcher, load_train  # noqa: E402
        cells = [tuple(x) for x in meta["cells"]]
        O, A, _ = load_train(cells, meta["args"]["n_eps"], a.kind)
        bt = Batcher(O, A, range(len(cells)), [b1.multi_hot(x) for x in cells], norm, dev)
        ns, nc = b1_data.chunks(norm.state(obs), norm.act(act))   # all kept demos of the cell
        ns, nc = (torch.tensor(x, dtype=torch.float32, device=dev) for x in (ns, nc))
        model.train()
        opt = torch.optim.AdamW(list(model.parameters()) + [z], lr=1e-4)
        res["trainable_params"] = int(n_params(model) + z.numel())
        gen = torch.Generator(device=dev).manual_seed(a.seed)
        t0 = time.time()
        for step in range(a.mt_steps):
            st, ch, tk = bt.sample(3 * a.batch, gen)        # 3/4 training cells, 1/4 the new cell
            j = torch.randint(0, ns.shape[0], (a.batch,), device=dev, generator=gen)
            l_old = model.loss(st, ch, tk)
            l_new = model.loss(ns[j], nc[j], z=z.expand(a.batch, -1))
            loss = 0.75 * l_old + 0.25 * l_new
            opt.zero_grad(set_to_none=True), loss.backward(), opt.step()
        t_train = time.time() - t0
        model = model.cpu().eval()
        r = b1_eval.rollout(model, norm, cell, dict(z=z.detach().cpu()), a.episodes, seed=1)
        res["ckpts"][str(a.mt_steps)] = dict(success=float(np.mean(r["success"])), episodes=r["success"],
                                             train_s=t_train, eval_s=r["seconds"])
        res["tau"] = 0.8 * res["ckpts"][str(a.mt_steps)]["success"]
    else:
        if a.arm.startswith("lora"):
            params = model.add_lora(int(a.arm[4:])) + [z]
            lr = 1e-3
        else:
            params = list(model.parameters()) + [z]
            lr = 1e-4
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        res["trainable_params"] = int(sum(p.numel() for p in params))
        gen = torch.Generator().manual_seed(a.seed)
        env = b1_eval.make_env(cell)
        step = 0
        for target in STEPS:
            t0 = time.time()
            model.train()
            while step < target:
                j = torch.randint(0, s.shape[0], (a.batch,), generator=gen)
                loss = model.loss(s[j], c[j], z=z.expand(a.batch, -1))
                opt.zero_grad(set_to_none=True), loss.backward(), opt.step()
                step += 1
            t_train += time.time() - t0
            if str(target) in res["ckpts"]:
                continue        # resumed: training is deterministic given the seed, so replaying steps is exact
            torch.save(dict(step=target, **trainable_state(model, z, a.arm)), out / f"{a.arm}_step{target}.pt")
            r = b1_eval.rollout(model, norm, cell, dict(z=z.detach()), a.episodes, seed=1, env=env)
            res["ckpts"][str(target)] = dict(success=float(np.mean(r["success"])), episodes=r["success"],
                                             train_s=t_train, eval_s=r["seconds"])
            resf.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
            print(cell_name, a.arm, target, res["ckpts"][str(target)]["success"], flush=True)
    res["done"] = True
    resf.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")


def main():
    if "--read" in sys.argv:
        return read()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, nargs="+")
    ap.add_argument("--arm", required=True, choices=["lora4", "lora16", "full", "mtsft"])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--base", default=str(b1.RESULTS / "policy" / "free" / "latest.pt"))
    ap.add_argument("--kind", default="expert")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--mt-steps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for name in a.cell:
        run_cell(a, name)




if __name__ == "__main__":
    main()
