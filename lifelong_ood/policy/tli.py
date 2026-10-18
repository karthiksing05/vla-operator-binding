"""Li et al.'s Text Latent Interpolation (TLI; arXiv 2505.03500, code github.com/QuanyiLi/pi0-text-latent at 587a6cb), ported to
openpi's PyTorch pi0 for the reviewer baseline G12.

A task's TEXT LATENT is the residual stream at the input of each of the 18 PaliGemma layers, at the prompt's first 9 token
positions (after the 3 x 256 image tokens), averaged over frames 0-119 of the task's first 20 training demonstrations, with no image
augmentation (Li's scripts/text_latent.py: compute_loss_with_extra, train=False).  At test time the policy reads the test
instruction itself; before every layer l >= 1 the prefix pass adds, on those 9 positions,
    (1 - alpha) (T1 - T2)_l + alpha (T2 - T1)_l,   alpha = linspace(0, 1, lambda + 2)[query],  held at 1 after the ramp,
so task 1's behaviour fades into task 2's (Li's policy.py, layer_to_intervene="all", use_TEI_and_TLI=False: layer 0, the text
embedding, is left alone).  lambda per test task is Li's task_hidden_states_mapping (experiments/g12_tli.py).

Architecture-agnostic: the operator reads only the backbone interface text_span / cond_hook (backbone.PolicyBackbone), so the same
code runs on pi0 / pi0.5 (PaliGemma, 18 layers, instruction at 768 = 3 x 256 image tokens), SmolVLA (the SmolVLM2 text model
truncated to 16 layers, instruction at 128 = 2 views x 64 tokens, then a state token) and X-VLA (Florence-2's BART-large encoder,
12 layers, instruction at 50 = the agent view's DaViT tokens; the other views bypass the encoder).  "The first 9 positions" are the first 9 INSTRUCTION tokens: a prompt with fewer tokens
contributes zeros past its end (its latent is 0 there) and receives no shift there (the pads are masked keys in pi0 / SmolVLA, so
this changes nothing for them; X-VLA's encoder attends its pads, and the shift stays on the instruction's own tokens).
"""
import logging
import os

import numpy as np
import torch

OFFSET, NTOK, NFRAME, NDEMO = 256 * 3, 9, 120, 20                     # OFFSET: pi0 / pi0.5's (bk.text_span gives each backbone's)


def _mask(nv, k, device):
    """[B, k] float: position j < the row's number of instruction tokens."""
    return (torch.arange(k, device=device)[None] < nv.to(device)[:, None]).float()


@torch.no_grad()
def identify(bk, demos, task, batch=24):
    """-> ([L, 9, D] float32, frames): the mean layer-input residual at the instruction's first 9 positions over frames 0-119 of
    the task's first 20 demonstrations (dataset order, as Li's create_dataloader takes them); positions past a prompt's end are 0."""
    cur, num = {}, {}

    def f(i, h, span):
        off, nv = span
        x = h[:, off:off + NTOK].float()
        k = x.shape[1]
        m = _mask(nv, k, h.device)
        s = torch.zeros((NTOK, h.shape[-1]), device=h.device)
        s[:k] = (x * m[..., None]).sum(0)
        c = torch.zeros(NTOK, device=h.device)
        c[:k] = m.sum(0)
        cur[i], num[i] = s, c
        return None
    hs = bk.cond_hook(f)
    acc, cnt, n = None, None, 0
    try:
        frames = []
        for e in demos.episodes(task)[:NDEMO]:
            ep = demos.load(e, images=True)
            frames += [(ep, i) for i in range(len(ep["frame_index"])) if ep["frame_index"][i] < NFRAME]
        for s in range(0, len(frames), batch):
            cur.clear()
            num.clear()
            obs = bk.observations([demos.observation(ep, i) for ep, i in frames[s:s + batch]])
            bk.prefix(obs)
            L = sorted(cur)
            assert L == list(range(len(L))), f"tli: layers seen {L}"
            x, c = torch.stack([cur[i] for i in L]), torch.stack([num[i] for i in L])
            acc, cnt = (x, c) if acc is None else (acc + x, cnt + c)
            n += len(frames[s:s + batch])
    finally:
        for h in hs:
            h.remove()
    out = torch.where(cnt[..., None] > 0, acc / cnt.clamp(min=1)[..., None], torch.zeros_like(acc))
    return out.cpu().numpy().astype(np.float32), n


class TLIHooks:
    """Serve hooks: the text-latent shift on every prefix pass, alpha stepped once per query (Li's self.step)."""

    def __init__(self, bk, t1, t2, lam):
        self.alphas = np.linspace(0, 1, int(lam) + 2)
        d = torch.as_tensor(t1 - t2, device=bk.device)
        d[0] = 0.0                                                      # layer 0 (the text embedding) untouched
        self.d = d
        self.nq, self.cur, self._warned = 0, None, False
        self._h = bk.cond_hook(self._f)

    def _f(self, i, h, span):
        if self.cur is None or i == 0:
            return None
        if span is None:                                                # the encoder ran outside bk.prefix: no span, no shift
            if not self._warned:
                logging.warning("tli: conditioning pass outside bk.prefix(); no shift applied")
                self._warned = True
            return None
        off, nv = span
        k = min(NTOK, h.shape[1] - off)
        if k <= 0:
            return None
        m = _mask(nv, k, h.device).to(h.dtype)
        h = h.clone()
        h[:, off:off + k] += m[..., None] * self.cur[i][:k].to(h.dtype)[None]
        if i == 1 and self.nq <= 1:
            logging.info(f"tli: shift applied at layer 1, query {self.nq}, |shift| {float(self.cur[i].norm()):.1f}, "
                         f"|h| {float(h[:, off:off + k].float().norm()):.1f}, offset {off}, instruction tokens {nv.tolist()}")
        return h

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
