"""B1 stage (d): train the base flow-matching action-chunk policy on the training cells (GPU, resumable).

    --cond free    a free per-task embedding, no factor structure (the base policy; DESIGN C5)
    --cond oracle  the 16-dim factor multi-hot (upper bound)
Checkpoints every --ckpt-every steps to results/b1/policy/<cond>/ (latest.pt + step_<k>.pt every 20k); a
preempted job resumes from latest.pt.  Recipe: lifelong_ood/b1_policy.py docstring.

    sbatch ... --gres=gpu:a40:1 --export=ALL,SCRIPT=experiments/b1_policy.py,ARGS="--cond free" slurm/jobs/b1_cpu.sbatch
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lifelong_ood import b1, b1_data  # noqa: E402
from lifelong_ood.b1_policy import H, FlowPolicy, n_params  # noqa: E402


def load_train(cells, n_eps, kind):
    O, A, S = [], [], []
    for c in cells:
        if kind == "expert":
            o, a, s = b1_data.load_cell(c, n_eps)
        else:  # smoke data written by b1_env_smoke.py --fake-data
            z = np.load(b1.DATA / kind / b1.cell_name(c) / "episodes.npz")
            o, a, s = b1_data.strip_task_id(z["obs"][:n_eps], c), z["act"][:n_eps], z["success"][:n_eps]
        O.append(o), A.append(a), S.append(s)
    return O, A, S


class Batcher:
    """All episodes on the device; a batch samples (episode, t) uniformly and gathers the H-chunk on the fly."""

    def __init__(self, O, A, task_ids, cond_vecs, norm, dev):
        self.obs = torch.tensor(np.concatenate([norm.state(o) for o in O]), device=dev)
        self.act = torch.tensor(np.concatenate([norm.act(a) for a in A]), device=dev)
        self.task = torch.tensor(np.concatenate([[t] * len(o) for t, o in zip(task_ids, O)]), device=dev)
        self.cvec = torch.tensor(np.asarray(cond_vecs, np.float32), device=dev)
        self.E, self.T = self.act.shape[:2]
        self.ar = torch.arange(H, device=dev)

    def sample(self, B, gen):
        e = torch.randint(0, self.E, (B,), device=self.obs.device, generator=gen)
        t = torch.randint(0, self.T, (B,), device=self.obs.device, generator=gen)
        idx = torch.clamp(t[:, None] + self.ar[None], max=self.T - 1)
        return self.obs[e, t], self.act[e[:, None], idx], self.task[e]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=["free", "oracle"], required=True)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-eps", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--kind", default="expert")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = pathlib.Path(a.out or b1.RESULTS / "policy" / (a.cond if a.kind == "expert" else f"{a.cond}_{a.kind}"))
    out.mkdir(parents=True, exist_ok=True)
    split = b1.load_split()
    cells = split["train"] if a.kind == "expert" else [tuple(c.split("_")) for c in
                                                       sorted(os.listdir(b1.DATA / a.kind))]
    O, A, S = load_train(cells, a.n_eps, a.kind)
    norm = b1_data.Norm.fit(np.concatenate(O), np.concatenate(A))
    torch.manual_seed(a.seed)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    model = FlowPolicy(O[0].shape[-1], len(cells), cond=a.cond).to(dev)
    ema = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 1000) * 0.5 * (1 + np.cos(np.pi * min(s, a.steps) / a.steps)))
    step, gpu_s = 0, 0.0
    lat = out / "latest.pt"
    if lat.exists():
        ck = torch.load(lat, map_location=dev)
        model.load_state_dict(ck["model"]), ema.load_state_dict(ck["ema"]), opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]), gen.set_state(ck["gen"].cpu())
        step, gpu_s = ck["step"], ck["gpu_s"]
        print(f"resumed at step {step}")
    mh = [b1.multi_hot(c) for c in cells]
    bt = Batcher(O, A, range(len(cells)), mh, norm, dev)
    meta = dict(cells=[list(c) for c in cells], cond=a.cond, norm=norm.to_dict(), cfg=model.cfg, args=vars(a),
                n_params=n_params(model), demo_success=[float(s.mean()) for s in S])
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(f"{len(cells)} cells, {bt.E} episodes, {n_params(model) / 1e6:.2f}M params, device {dev}")
    t0, losses = time.time(), []
    while step < a.steps:
        s, ch, tk = bt.sample(a.batch, gen)
        cond = tk if a.cond == "free" else bt.cvec[tk]
        loss = model.loss(s, ch, cond)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(), sched.step()
        with torch.no_grad():
            d = min(0.999, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
        losses.append(loss.item())
        step += 1
        if step % 500 == 0:
            print(f"step {step} loss {np.mean(losses[-500:]):.4f} {time.time() - t0:.0f}s", flush=True)
        if step % a.ckpt_every == 0 or step == a.steps:
            gpu_s += time.time() - t0
            t0 = time.time()
            ck = dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                      gen=gen.get_state(), step=step, gpu_s=gpu_s, meta=meta)
            torch.save(ck, out / "latest.tmp")
            os.replace(out / "latest.tmp", lat)
            if step % 20_000 == 0 or step == a.steps:
                torch.save(dict(ema=ema.state_dict(), step=step, meta=meta), out / f"step_{step}.pt")
    (out / "DONE").write_text(json.dumps(dict(step=step, gpu_s=gpu_s, final_loss=float(np.mean(losses[-500:])) if losses else None)) + "\n",
                              encoding="utf-8")
    print("done", step, f"{gpu_s / 3600:.2f} GPU-h")


if __name__ == "__main__":
    main()
