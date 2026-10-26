"""PROBE D5 (docs/PLAN_D5.md): a BATCHED policy server for the counterfactual sandboxes (openpi env).  New file; nothing shared.

    python experiments/d5_serve.py --port P          (BACKBONE_CONFIG / BACKBONE_CKPT as for serve_guided.py)

One websocket request carries B sandbox observations of ONE task (one prompt):
    b/image [B, 224, 224, 3] uint8, b/wrist [B, 224, 224, 3] uint8, b/state [B, 8], b/seed [B] int, prompt str,
    optional b/sig [B] bool: also return the SigLIP prototype readout of those rows (the D3 / R4 vision anchor).
Every row is sampled by the policy's own Euler sampler (10 steps) from the noise torch.Generator(cpu).manual_seed(seed).randn(dim_full),
exactly bk.sample(1, prefix, seed=seed), i.e. the noise a single-observation query with that seed would draw.
-> {"actions": [B, H, 7] dataset actions, "sig_tok": [B, P] argmax agentview token per prototype, "sig_val": [B, P]}.
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
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
GROUP = 16                         # observations per prefix forward


class BatchPolicy:
    def __init__(self, bk, num_steps=10):
        import torch
        self.bk = bk
        self.num_steps = int(num_steps)
        pz = np.load(os.path.join(ROOT, "results", "rb4", "protos.npz"), allow_pickle=True)
        self.PR = torch.as_tensor(pz["sig"], device=bk.device)
        self.calls, self.rows, self.secs = 0, 0, 0.0

    def noise(self, seed):
        import torch
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        return torch.randn((1, self.bk.dim_full), generator=g).numpy()[0]

    def _sig(self, observation):
        import torch
        m = self.bk.model
        images, img_masks, lang_tokens, lang_masks, _ = m._preprocess_observation(observation, train=False)
        embs, _, _ = m.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        S = torch.nn.functional.normalize(embs[:, :256].float(), dim=-1)          # [b, 256, D] agentview tokens
        sim = torch.einsum("pd,btd->bpt", self.PR, S)                              # [b, P, 256]
        v, i = sim.max(-1)
        return i.cpu().numpy(), v.cpu().numpy()

    def infer(self, obs, *, noise=None):
        import torch
        t0 = time.monotonic()
        if "b/image" not in obs:                     # the warm-up example: one plain query
            o = {k: v for k, v in obs.items() if not k.startswith("meta/")}
            ob = self.bk.observation(o)
            pre = self.bk.prefix(ob)
            rows = self.bk.sample(1, pre, seed=0, num_steps=self.num_steps)
            return {"actions": self.bk.unnormalize(rows)[0].astype(np.float32)}
        I, W = np.asarray(obs["b/image"]), np.asarray(obs["b/wrist"])
        S, sd = np.asarray(obs["b/state"], np.float32), np.asarray(obs["b/seed"]).astype(np.int64)
        want = np.asarray(obs["b/sig"]).astype(bool) if "b/sig" in obs else np.zeros(len(I), bool)
        prompt = str(obs["prompt"])
        B = len(I)
        acts = np.zeros((B, self.bk.H, 7), np.float32)
        P = int(self.PR.shape[0])
        stok, sval = np.full((B, P), -1, np.int64), np.full((B, P), np.nan, np.float32)
        with torch.no_grad():
            for g0 in range(0, B, GROUP):
                idx = list(range(g0, min(g0 + GROUP, B)))
                ods = [{"observation/image": I[q], "observation/wrist_image": W[q], "observation/state": S[q], "prompt": prompt}
                       for q in idx]
                ob = self.bk.observations(ods)
                pre = self.bk.prefix(ob)
                nz = np.stack([self.noise(sd[q]) for q in idx])
                rows = self.bk.sample(len(idx), pre, num_steps=self.num_steps, noise=nz)
                acts[idx] = self.bk.unnormalize(rows).reshape(len(idx), self.bk.H, 7)
                if want[idx].any():
                    ti, tv = self._sig(ob)
                    for j, q in enumerate(idx):
                        if want[q]:
                            stok[q], sval[q] = ti[j], tv[j]
        self.calls += 1
        self.rows += B
        self.secs += time.monotonic() - t0
        if self.calls % 50 == 0:
            logging.info(f"[d5 serve] {self.calls} calls, {self.rows} rows, {1000 * self.secs / max(self.rows, 1):.0f} ms per row")
        return {"actions": acts, "sig_tok": stok, "sig_val": sval}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--num-steps", type=int, default=10)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from lifelong_ood.policy.backbone import PolicyBackbone
    from lifelong_ood.policy.serve import serve, warm
    bk = PolicyBackbone(os.environ.get("BACKBONE_CONFIG") or "pi05_l90c", checkpoint_dir=os.environ.get("BACKBONE_CKPT") or None,
                        device="cuda")
    logging.info(bk.describe())
    pol = BatchPolicy(bk, num_steps=a.num_steps)
    # parity: a batched row = the single-observation sampler with the same seed
    from openpi.policies import libero_policy
    ex = libero_policy.make_libero_example()
    ex["observation/state"] = np.asarray(ex["observation/state"], np.float32)
    b = {"b/image": np.stack([ex["observation/image"]] * 3), "b/wrist": np.stack([ex["observation/wrist_image"]] * 3),
         "b/state": np.stack([ex["observation/state"]] * 3), "b/seed": np.array([5, 6, 5]), "prompt": ex["prompt"]}
    out = pol.infer(b)["actions"]
    ob = bk.observation({k: ex[k] for k in ("observation/image", "observation/wrist_image", "observation/state", "prompt")})
    one = bk.unnormalize(bk.sample(1, bk.prefix(ob), seed=5, num_steps=a.num_steps))[0]
    logging.info(f"[d5 serve] parity batched vs single (seed 5): max |diff| {float(np.abs(out[0] - one).max()):.2e}; "
                 f"same seed rows {float(np.abs(out[0] - out[2]).max()):.2e}; different seeds {float(np.abs(out[0] - out[1]).max()):.2e}")
    warm(pol)
    serve(pol, a.port)


if __name__ == "__main__":
    main()
