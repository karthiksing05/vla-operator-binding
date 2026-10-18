"""Li et al.'s Text Latent Interpolation (TLI; arXiv 2505.03500, code github.com/QuanyiLi/pi0-text-latent at 587a6cb), ported to
openpi's PyTorch pi0 for the reviewer baseline G12.

A task's TEXT LATENT is the residual stream at the input of each of the 18 PaliGemma layers, at the prompt's first 9 token
positions (after the 3 x 256 image tokens), averaged over frames 0-119 of the task's first 20 training demonstrations, with no image
augmentation (Li's scripts/text_latent.py: compute_loss_with_extra, train=False).  At test time the policy reads the test
instruction itself; before every layer l >= 1 the prefix pass adds, on those 9 positions,
    (1 - alpha) (T1 - T2)_l + alpha (T2 - T1)_l,   alpha = linspace(0, 1, lambda + 2)[query],  held at 1 after the ramp,
so task 1's behaviour fades into task 2's (Li's policy.py, layer_to_intervene="all", use_TEI_and_TLI=False: layer 0, the text
embedding, is left alone).  lambda per test task is Li's task_hidden_states_mapping (experiments/g12_tli.py).
"""
import os

import numpy as np
import torch

OFFSET, NTOK, NFRAME, NDEMO = 256 * 3, 9, 120, 20


def _layers(bk):
    return bk.model.paligemma_with_expert.paligemma.language_model.layers


@torch.no_grad()
def identify(bk, demos, task, batch=24):
    """-> [L, 9, D] float32: the mean layer-input residual at the prompt's first 9 positions over frames 0-119 of the task's first
    20 demonstrations (dataset order, as Li's create_dataloader takes them)."""
    L = len(_layers(bk))
    acc, cnt, cur = None, 0, {}

    def pre(i):
        def f(mod, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            cur[i] = h[:, OFFSET:OFFSET + NTOK].float().sum(0)
        return f
    hs = [ly.register_forward_pre_hook(pre(i), with_kwargs=True) for i, ly in enumerate(_layers(bk))]
    try:
        frames = []
        for e in demos.episodes(task)[:NDEMO]:
            ep = demos.load(e, images=True)
            frames += [(ep, i) for i in range(len(ep["frame_index"])) if ep["frame_index"][i] < NFRAME]
        for s in range(0, len(frames), batch):
            obs = bk.observations([demos.observation(ep, i) for ep, i in frames[s:s + batch]])
            bk.prefix(obs)
            x = torch.stack([cur[i] for i in range(L)])
            acc = x if acc is None else acc + x
            cnt += len(frames[s:s + batch])
    finally:
        for h in hs:
            h.remove()
    return (acc / cnt).cpu().numpy().astype(np.float32), cnt


class TLIHooks:
    """Serve hooks: the text-latent shift on every prefix pass, alpha stepped once per query (Li's self.step)."""

    def __init__(self, bk, t1, t2, lam):
        self.alphas = np.linspace(0, 1, int(lam) + 2)
        d = torch.as_tensor(t1 - t2, device=bk.device)
        d[0] = 0.0                                                      # layer 0 (the text embedding) untouched
        self.d = d
        self.nq, self.cur = 0, None
        self._h = [ly.register_forward_pre_hook(self._pre(i), with_kwargs=True) for i, ly in enumerate(_layers(bk))]

    def _pre(self, i):
        def f(mod, args, kwargs):
            if self.cur is None or i == 0:
                return None
            h = args[0] if args else kwargs["hidden_states"]
            if h.shape[1] < OFFSET + NTOK:
                return None
            h = h.clone()
            h[:, OFFSET:OFFSET + NTOK] += self.cur[i].to(h.dtype)
            if i == 1 and self.nq <= 1:
                import logging
                logging.info(f"tli: shift applied at layer 1, query {self.nq}, |shift| {float(self.cur[i].norm()):.1f}, "
                             f"|h| {float(h[:, OFFSET:OFFSET + NTOK].float().norm()):.1f}")
            return ((h,) + tuple(args[1:]), kwargs) if args else (args, {**kwargs, "hidden_states": h})
        return f

    def reset(self, meta):
        self.nq = 0

    def rewrite(self, obs, meta):
        a = float(self.alphas[min(self.nq, len(self.alphas) - 1)])
        self.cur = (1 - 2 * a) * self.d                                 # (1 - a)(T1 - T2) + a (T2 - T1)
        self.nq += 1
        return obs

    def __call__(self, obs, pre, meta=None):                          # no guidance: the policy samples on the shifted prefix
        return None, None


def path(config, task, root):
    return os.path.join(root, "results", "robot", "tli", f"{config}_{task}.npy")
