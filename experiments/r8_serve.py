"""PROBE R8 (docs/PLAN_R8.md): D5's batched policy server (experiments/d5_serve.py, imported unchanged) + opt-in region re-conditioning
arms.  New file; nothing shared is edited.  Weights stay frozen; no demo is used.

Request fields (absent -> exactly d5_serve.BatchPolicy):
  r8/arm     "off" | "c3" | "c2"
  r8/phrase  the region phrase as it appears in the prompt (r8_ops.spec)
  r8/proto   (c3) the prototype key in results/r8/protos.npz
  r8/alts    (c2) the alternative-region prompts, joined by "|"
  r8/w       (c2) the contrast weight
C3: before every encoder layer i >= 1 of the prefix pass, the residual stream at the phrase's tokens is REPLACED by the prototype: the
    mean, over the training carry observations whose instruction names that region (D5's index; images / state from D4's renders), of
    the same layer's input at the same phrase tokens.
C2: Euler with v = v(p) + w [v(p) - mean_a v(p_a)] on the first C2_STEPS of the 10 steps (p_a: the same observation, the prompt with
    the phrase replaced by another region of the same body); plain v(p) afterwards.  The same noise as the plain sampler.

    python experiments/r8_serve.py --port P                 (serve)
    python experiments/r8_serve.py --build-protos           (results/r8/protos.npz; GPU job)
"""
from __future__ import annotations

import argparse
import glob
import json
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

PROTOS = os.path.join(ROOT, "results", "r8", "protos.npz")
C2_STEPS = 5
N_IMG_TOK = 256


class Tok:
    def __init__(self):
        from openpi.models.tokenizer import PaligemmaTokenizer
        self.sp = PaligemmaTokenizer()._tokenizer

    def span(self, prompt, phrase):
        """Token index range [a, b) of `phrase` inside openpi's pi05 tokenization of `prompt` (BOS + 'Task: <cleaned>, State: ...')."""
        text = prompt.strip().replace("_", " ").replace("\n", " ")
        i = text.index(phrase)
        full = self.sp.encode(f"Task: {text}, State: 0;\nAction: ", add_bos=True)
        pa = self.sp.encode("Task: " + text[:i].rstrip(), add_bos=True)
        pb = self.sp.encode("Task: " + text[:i + len(phrase)], add_bos=True)
        assert full[:len(pb)] == pb and pb[:len(pa)] == pa, (prompt, phrase)
        return len(pa), len(pb)


class R8Policy(D5S.BatchPolicy):
    def __init__(self, bk, num_steps=10):
        super().__init__(bk, num_steps)
        self.tok = Tok()
        self.protos = {}
        if os.path.exists(PROTOS):
            z = np.load(PROTOS, allow_pickle=True)
            self.protos = {k: z[k] for k in z.files if not k.startswith("meta")}
            logging.info(f"[r8 serve] prototypes {sorted(self.protos)}")

    def _obs(self, I, W, S, idx, prompt):
        return self.bk.observations([{"observation/image": I[q], "observation/wrist_image": W[q], "observation/state": S[q],
                                      "prompt": prompt} for q in idx])

    def prefix_patched(self, ob, a, b, proto=None, record=None):
        """The prefix with the phrase tokens' residual replaced by proto [L, n, D] (layers >= 1), or recorded into `record`."""
        import torch
        off = N_IMG_TOK * len(ob.images)

        def fn(i, h, span):
            if record is not None:
                record.append(h[:, off + a:off + b].float().cpu().numpy())
                return None
            if proto is None or i == 0:
                return None
            h = h.clone()
            h[:, off + a:off + b] = torch.as_tensor(proto[i], device=h.device, dtype=h.dtype)[None]
            return h
        hs = self.bk.cond_hook(fn)
        try:
            return self.bk.prefix(ob)
        finally:
            for x in hs:
                x.remove()

    def euler_c2(self, pre, pres_alt, noise, w):
        import torch
        bk = self.bk
        n = len(noise)
        K = self.num_steps
        x = torch.as_tensor(noise, device=bk.device).reshape(n, bk.dim_full).float()
        dt = torch.tensor(-1.0 / K, dtype=torch.float32, device=bk.device)
        t = torch.tensor(1.0, dtype=torch.float32, device=bk.device)
        s = 0
        while t >= -dt / 2:
            vt = bk.v_full(x, t.expand(n), pre)
            if s < C2_STEPS and w != 0.0 and pres_alt:
                va = sum(bk.v_full(x, t.expand(n), p) for p in pres_alt) / len(pres_alt)
                vt = vt + w * (vt - va)
            x = x + dt * vt
            t = t + dt
            s += 1
        return x[:, bk.real].float().cpu().numpy()

    def infer(self, obs, *, noise=None):
        import torch
        arm = str(obs.get("r8/arm", "off")) if "b/image" in obs else "off"
        if arm == "off":
            return super().infer(obs, noise=noise)
        t0 = time.monotonic()
        I, W = np.asarray(obs["b/image"]), np.asarray(obs["b/wrist"])
        S, sd = np.asarray(obs["b/state"], np.float32), np.asarray(obs["b/seed"]).astype(np.int64)
        prompt, phrase = str(obs["prompt"]), str(obs["r8/phrase"])
        B = len(I)
        acts = np.zeros((B, self.bk.H, 7), np.float32)
        with torch.no_grad():
            for g0 in range(0, B, D5S.GROUP):
                idx = list(range(g0, min(g0 + D5S.GROUP, B)))
                ob = self._obs(I, W, S, idx, prompt)
                nz = np.stack([self.noise(sd[q]) for q in idx])
                if arm == "c3":
                    a, b = self.tok.span(prompt, phrase)
                    proto = self.protos[str(obs["r8/proto"])]
                    assert proto.shape[1] == b - a, (proto.shape, a, b)
                    pre = self.prefix_patched(ob, a, b, proto=proto)
                    rows = self.bk.sample(len(idx), pre, num_steps=self.num_steps, noise=nz)
                elif arm == "c2":
                    pre = self.bk.prefix(ob)
                    alts = [p for p in str(obs["r8/alts"]).split("|") if p]
                    pres = [self.bk.prefix(self._obs(I, W, S, idx, p)) for p in alts]
                    rows = self.euler_c2(pre, pres, nz, float(obs["r8/w"]))
                else:
                    raise ValueError(arm)
                acts[idx] = self.bk.unnormalize(rows).reshape(len(idx), self.bk.H, 7)
        self.calls += 1
        self.rows += B
        self.secs += time.monotonic() - t0
        return {"actions": acts, "sig_tok": np.full((B, 1), -1, np.int64), "sig_val": np.full((B, 1), np.nan, np.float32)}


def build_protos(bk):
    """Prototype per (phrase, source tasks) used by any R8 unit: results/r8/plan.json lists them."""
    import torch
    pol = R8Policy(bk)
    plan = json.load(open(os.path.join(ROOT, "results", "r8", "plan.json"), encoding="utf-8"))
    d5 = json.load(open(os.path.join(ROOT, "results", "d5", "index.json"), encoding="utf-8"))
    out, meta = {}, {}
    for key, pr in plan["protos"].items():
        obs = [o for o in d5["train"] if o["tid"] in pr["sources"] and pr["phrase"] in o["prompt"]]
        recs = []
        for o in obs:
            f = os.path.join(ROOT, "results", "d4", "render", "train_l90c", o["key"][3:] + ".npz")
            z = np.load(f, allow_pickle=True)
            I, W, S = z["images"][:1], z["wrists"][:1], np.asarray(z["state8"], np.float32)[None]
            ob = pol._obs(I, W, S, [0], o["prompt"])
            a, b = pol.tok.span(o["prompt"], pr["phrase"])
            rec = []
            with torch.no_grad():
                pol.prefix_patched(ob, a, b, record=rec)
            recs.append(np.stack([r[0] for r in rec]))               # [L, n, D]
        assert recs, key
        out[key] = np.mean(recs, 0).astype(np.float32)
        meta[key] = dict(n_obs=len(recs), obs=[o["key"] for o in obs], shape=list(out[key].shape))
        logging.info(f"[r8 protos] {key}: {meta[key]['n_obs']} observations, shape {meta[key]['shape']}")
    np.savez_compressed(PROTOS, meta=np.array(json.dumps(meta)), **out)


def parity(pol):
    """c2 at w 0 must equal the plain sampler; c3 with the prompt's OWN recorded residual must equal the plain prefix."""
    import torch
    from openpi.policies import libero_policy
    ex = libero_policy.make_libero_example()
    ex["prompt"] = "put the black bowl in the top drawer of the cabinet"
    st = np.asarray(ex["observation/state"], np.float32)
    I, W, S = np.stack([ex["observation/image"]] * 2), np.stack([ex["observation/wrist_image"]] * 2), np.stack([st] * 2)
    base = {"b/image": I, "b/wrist": W, "b/state": S, "b/seed": np.array([5, 6]), "prompt": ex["prompt"]}
    off = pol.infer(dict(base))["actions"]
    c2 = pol.infer(dict(base, **{"r8/arm": "c2", "r8/phrase": "in the top drawer of the cabinet",
                                 "r8/alts": "put the black bowl on top of the cabinet", "r8/w": 0.0}))["actions"]
    a, b = pol.tok.span(ex["prompt"], "in the top drawer of the cabinet")
    rec = []
    with torch.no_grad():
        pol.prefix_patched(pol._obs(I, W, S, [0], ex["prompt"]), a, b, record=rec)
    pol.protos["_self"] = np.stack([r[0] for r in rec])
    c3 = pol.infer(dict(base, **{"r8/arm": "c3", "r8/phrase": "in the top drawer of the cabinet", "r8/proto": "_self"}))["actions"]
    del pol.protos["_self"]
    logging.info(f"[r8 serve] parity: c2(w 0) vs off max |diff| {float(np.abs(c2 - off).max()):.2e}; c3(own residual) vs off "
                 f"{float(np.abs(c3 - off).max()):.2e}; phrase tokens {a}-{b}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--build-protos", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from lifelong_ood.policy.backbone import PolicyBackbone
    from lifelong_ood.policy.serve import serve, warm
    bk = PolicyBackbone(os.environ.get("BACKBONE_CONFIG") or "pi05_l90c", checkpoint_dir=os.environ.get("BACKBONE_CKPT") or None,
                        device="cuda")
    if a.build_protos:
        build_protos(bk)
        return
    pol = R8Policy(bk)
    parity(pol)
    warm(pol)
    serve(pol, a.port)


if __name__ == "__main__":
    main()
