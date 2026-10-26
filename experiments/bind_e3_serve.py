"""PLAN_BIND E3 (docs/PLAN_E34.md section 2): D5's batched server (experiments/d5_serve.py, unchanged) + an opt-in action-expert
activation INJECTION (2603.19233-style).  New file.

Request fields (absent -> exactly d5_serve.BatchPolicy):
  e3/inj        "16-17" | "16": the action-expert decoder layers whose OUTPUTS are replaced
  e3/src_image, e3/src_wrist [224, 224, 3], e3/src_state [8], e3/src_prompt: the SOURCE observation
For an injected request every row is first sampled on the source observation with the row's own noise (10 Euler steps), recording
the chosen layers' outputs at every denoising step; the target rows are then sampled with the same noise and those outputs replaced,
step by step.  Weights stay frozen.

    python experiments/bind_e3_serve.py --port P
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


class E3Policy(D5S.BatchPolicy):
    def _layers(self):
        return self.bk.expert.model.layers

    def _hooked_sample(self, pre, nz, layers, mode, store):
        """mode 'rec': append each chosen layer's output per call to store[l]; 'inj': replace it by store[l][call]."""
        import torch
        calls = {l: 0 for l in layers}

        def mk(l):
            def f(mod, args, out):
                h = out[0] if isinstance(out, tuple) else out
                if mode == "rec":
                    store.setdefault(l, []).append(h.detach().clone())
                    return None
                src = store[l][calls[l]]
                calls[l] += 1
                assert src.shape == h.shape, (src.shape, h.shape)
                h2 = src.to(h.dtype)
                return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
            return f
        hs = [self._layers()[l].register_forward_hook(mk(l)) for l in layers]
        try:
            with torch.no_grad():
                return self.bk.sample(len(nz), pre, num_steps=self.num_steps, noise=nz)
        finally:
            for h in hs:
                h.remove()

    def infer(self, obs, *, noise=None):
        inj = str(obs.get("e3/inj", "")) if "b/image" in obs else ""
        if not inj:
            return super().infer(obs, noise=noise)
        import torch
        t0 = time.monotonic()
        layers = [16, 17] if inj == "16-17" else [16]
        I, W = np.asarray(obs["b/image"]), np.asarray(obs["b/wrist"])
        S, sd = np.asarray(obs["b/state"], np.float32), np.asarray(obs["b/seed"]).astype(np.int64)
        prompt = str(obs["prompt"])
        sI, sW = np.asarray(obs["e3/src_image"]), np.asarray(obs["e3/src_wrist"])
        sS, sp = np.asarray(obs["e3/src_state"], np.float32), str(obs["e3/src_prompt"])
        B = len(I)
        acts = np.zeros((B, self.bk.H, 7), np.float32)
        src_acts = np.zeros((B, self.bk.H, 7), np.float32)
        with torch.no_grad():
            for g0 in range(0, B, D5S.GROUP):
                idx = list(range(g0, min(g0 + D5S.GROUP, B)))
                nz = np.stack([self.noise(sd[q]) for q in idx])
                ob_s = self.bk.observations([{"observation/image": sI, "observation/wrist_image": sW, "observation/state": sS,
                                              "prompt": sp} for _ in idx])
                store = {}
                rows_s = self._hooked_sample(self.bk.prefix(ob_s), nz, layers, "rec", store)
                src_acts[idx] = self.bk.unnormalize(rows_s).reshape(len(idx), self.bk.H, 7)
                ob = self.bk.observations([{"observation/image": I[q], "observation/wrist_image": W[q], "observation/state": S[q],
                                            "prompt": prompt} for q in idx])
                rows = self._hooked_sample(self.bk.prefix(ob), nz, layers, "inj", store)
                acts[idx] = self.bk.unnormalize(rows).reshape(len(idx), self.bk.H, 7)
        self.calls += 1
        self.rows += B
        self.secs += time.monotonic() - t0
        return {"actions": acts, "src_actions": src_acts, "sig_tok": np.full((B, 1), -1, np.int64),
                "sig_val": np.full((B, 1), np.nan, np.float32)}


def parity(pol):
    """Injecting a request's OWN source (source = target) must reproduce the plain sampler."""
    from openpi.policies import libero_policy
    ex = libero_policy.make_libero_example()
    st = np.asarray(ex["observation/state"], np.float32)
    b = {"b/image": np.stack([ex["observation/image"]] * 2), "b/wrist": np.stack([ex["observation/wrist_image"]] * 2),
         "b/state": np.stack([st] * 2), "b/seed": np.array([5, 6]), "prompt": ex["prompt"]}
    off = pol.infer(dict(b))["actions"]
    on = pol.infer(dict(b, **{"e3/inj": "16-17", "e3/src_image": ex["observation/image"], "e3/src_wrist": ex["observation/wrist_image"],
                              "e3/src_state": st, "e3/src_prompt": ex["prompt"]}))
    logging.info(f"[e3 serve] parity: self-injection vs plain max |diff| {float(np.abs(on['actions'] - off).max()):.2e}; source rows vs "
                 f"plain {float(np.abs(on['src_actions'] - off).max()):.2e}; expert layers {len(pol._layers())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from lifelong_ood.policy.backbone import PolicyBackbone
    from lifelong_ood.policy.serve import serve, warm
    bk = PolicyBackbone(os.environ.get("BACKBONE_CONFIG") or "pi05_l90c", checkpoint_dir=os.environ.get("BACKBONE_CKPT") or None,
                        device="cuda")
    pol = E3Policy(bk)
    parity(pol)
    warm(pol)
    serve(pol, a.port)


if __name__ == "__main__":
    main()
