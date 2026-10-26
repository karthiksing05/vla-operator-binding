"""PLAN_SOLVE S0 (docs/PLAN_S0.md): D5's batched server (experiments/d5_serve.py, unchanged) + the binding adapter (s0_adapter.py,
results/bind/s0/adapter.pt).  New file.

Request fields (absent -> exactly d5_serve.BatchPolicy):
  s0/mobj, s0/mgoal [B, 256]   the role masks over the agentview tokens; the adapter edits the prefix's deep image K/V, the frozen
                               policy samples with the row's own noise (b/seed) as d5_serve does
Parity at start: zero masks reproduce the plain sampler exactly (the edit is zero by construction).

    python experiments/s0_serve.py --port P [--adapter results/bind/s0/adapter.pt]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
import d5_serve as D5S                                               # noqa: E402


class S0Policy(D5S.BatchPolicy):
    def __init__(self, bk, adapter, num_steps=10):
        super().__init__(bk, num_steps=num_steps)
        self.ad = adapter

    def infer(self, obs, *, noise=None):
        import torch
        if "s0/mobj" not in obs:
            return super().infer(obs, noise=noise)
        t0 = time.monotonic()
        I, W = np.asarray(obs["b/image"]), np.asarray(obs["b/wrist"])
        S, sd = np.asarray(obs["b/state"], np.float32), np.asarray(obs["b/seed"]).astype(np.int64)
        MO, MG = np.asarray(obs["s0/mobj"], np.float32), np.asarray(obs["s0/mgoal"], np.float32)
        prompt = str(obs["prompt"])
        B = len(I)
        acts = np.zeros((B, self.bk.H, 7), np.float32)
        with torch.no_grad():
            for g0 in range(0, B, D5S.GROUP):
                idx = list(range(g0, min(g0 + D5S.GROUP, B)))
                ods = [{"observation/image": I[q], "observation/wrist_image": W[q], "observation/state": S[q], "prompt": prompt}
                       for q in idx]
                pre = self.bk.prefix(self.bk.observations(ods))
                pre = self.ad.apply(pre, torch.tensor(MO[idx], device=self.bk.device), torch.tensor(MG[idx], device=self.bk.device))
                nz = np.stack([self.noise(sd[q]) for q in idx])
                rows = self.bk.sample(len(idx), pre, num_steps=self.num_steps, noise=nz)
                acts[idx] = self.bk.unnormalize(rows).reshape(len(idx), self.bk.H, 7)
        self.calls += 1
        self.rows += B
        self.secs += time.monotonic() - t0
        return {"actions": acts}


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--adapter", default=os.path.join(ROOT, "results", "bind", "s0", "adapter.pt"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from lifelong_ood.policy.backbone import PolicyBackbone
    from lifelong_ood.policy.serve import serve, warm
    import s0_adapter as SA
    bk = PolicyBackbone(os.environ.get("BACKBONE_CONFIG") or "pi05_l90c", checkpoint_dir=os.environ.get("BACKBONE_CKPT") or None,
                        device="cuda")
    ad = SA.BindAdapter().to(bk.device)
    ad.load_state_dict(torch.load(a.adapter, map_location=bk.device))
    ad.eval()
    pol = S0Policy(bk, ad)
    from openpi.policies import libero_policy
    ex = libero_policy.make_libero_example()
    ex["observation/state"] = np.asarray(ex["observation/state"], np.float32)
    b = {"b/image": np.stack([ex["observation/image"]] * 2), "b/wrist": np.stack([ex["observation/wrist_image"]] * 2),
         "b/state": np.stack([ex["observation/state"]] * 2), "b/seed": np.array([5, 6]), "prompt": ex["prompt"]}
    plain = pol.infer(b)["actions"]
    zero = pol.infer(dict(b, **{"s0/mobj": np.zeros((2, 256), np.float32), "s0/mgoal": np.zeros((2, 256), np.float32)}))["actions"]
    m = np.zeros((2, 256), np.float32)
    m[:, 100:110] = 1.0
    edit = pol.infer(dict(b, **{"s0/mobj": m, "s0/mgoal": m[:, ::-1].copy()}))["actions"]
    logging.info(f"[s0 serve] parity: zero masks vs plain max |diff| {float(np.abs(zero - plain).max()):.2e}; a mask moves the "
                 f"chunk by {float(np.abs(edit - plain).max()):.3e}; adapter {ad.nbytes() / 1e6:.1f} MB fp16")
    warm(pol)
    serve(pol, a.port)


if __name__ == "__main__":
    main()
