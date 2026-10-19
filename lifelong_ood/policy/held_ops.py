"""Two more operators for held concepts (plan H3 baselines; read G21): velocity composition (GPC) and activation injection.

GPC (Cao et al., "General Policy Composition", arXiv 2510.01068, ICLR 2026).  Training-free: at every Euler step the chunk follows
    v(x, t) = w v(x, t | o, instruction_a) + (1 - w) v(x, t | o, instruction_g),
a convex mix of the frozen policy's own fields under the object source's and the goal source's training instructions, both at the
same noisy chunk (conditionals.sample_composed; the two prefixes are one batched VLM pass).  Their test-time search over w is,
here, practice over w in {0.3, 0.5, 0.7} (experiments/g21_operators.py).  `phase=True` sets w = 1 before the grasp and 0 from it
(the grasp tracker is PrimHooks' own, unchanged): the grasp switch (pfield@a.g) computed through the composition path, a check.

ACTIVATION INJECTION (Grant et al., arXiv 2603.19233; code github.com/CWRU-AISM/action-atlas, experiments/hooks.py and
cross_task_injection*.py).  Their site is the output of each decoder layer of pi0.5's action expert (gemma_expert.model.layers,
forward hooks: the residual stream after the layer), every call recorded during a source episode and written back by REPLACEMENT
in call order during the target episode.  Ported, mirroring TLI (lifelong_ood/policy/tli.py) in what is averaged and how it is
written:
  identify  the mean residual at the INPUT of every expert layer (= the previous layer's output), per denoising step k (10) and
            per suffix token (pi0.5: the H action tokens; pi0: the state token and H action tokens), over frames 0-119 of the
            source task's first 20 training demonstrations, with the policy sampling its own chunk under the demo's instruction.
  inject    the test residual at the input of every expert layer l >= 1 (layer 0's input, the noisy chunk's embedding, untouched,
            as TLI leaves the text embedding) is moved by  s (M_a - M_g)[k, l],  s = +1 before the grasp and -1 from the grasp to
            the release: TLI's (1 - 2 alpha)(T1 - T2) with alpha stepped at the grasp, written into the action expert.  The
            residual stream carries every addition forward, so layer l adds only the increment (M_a - M_g)[k, l] - (M_a - M_g)[k, l-1]
            and the residual at layer l stands s (M_a - M_g)[k, l] from the policy's own (to first order), as a per-layer
            replacement relative to the other source would.  (Adding the full difference at every layer, TLI's literal form,
            accumulates: on pi0.5 the summed shift at the last layer was 0.9-2.1x the residual's norm, and the policy moved no
            object in 4 smoke episodes, against 2 of 4 successes alone; fixed 2026-10-19, before pre-registration.)
Deviation from Grant et al.: an additive difference of source means instead of replacement by one recorded episode.  Replacing the
expert residual by a mean over frames erases the noisy chunk x_t from the expert (its output no longer depends on x_t), so the Euler
integral returns x_noise + mean(a - eps): the test noise passes into the action.  Their replay avoids this only because the same
noise is drawn at capture and replay.  `rep=True` is the replacement variant (the residual at layers >= 1 set to the source's
mean), kept only as a smoke check of that argument.

Both run on openpi's pi0 / pi0.5 (PolicyBackbone); expert_layers() names the hook site and raises for backbones whose expert
layers are not called as modules (SmolVLA's interleaved loop calls the submodules directly; X-VLA's sampler is its own).
"""
from __future__ import annotations

import logging
import os

import numpy as np
import torch

NFRAME, NDEMO = 120, 20


def expert_layers(bk):
    if getattr(bk, "kind", "") != "pi0":
        raise NotImplementedError(f"activation injection: no expert-layer hook site for backbone kind {getattr(bk, 'kind', '?')}")
    return list(bk.expert.model.layers)


def path(config, task, root):
    return os.path.join(root, "results", "robot", "inject", f"{config}_{task}.npy")


class _Grasp:
    """PrimHooks' own grasp / release tracker (goal_only): carrying after 3 closed queries, released when the gripper opens."""

    def __init__(self, bk):
        from .primitives import PrimHooks
        self.p = PrimHooks(bk, ["-"], [None], goal_only=True)

    @property
    def carrying(self):
        return self.p.carrying

    def reset(self, meta):
        self.p.reset(meta)

    def observe(self, actions):
        self.p.observe(actions)


class GPCHooks:
    """Serve hooks for GuidedPolicy's composition path (hooks.compose)."""

    def __init__(self, bk, prompt_a, prompt_g, w=0.5, phase=False):
        self.pa, self.pg, self.w, self.phase = prompt_a, prompt_g, float(w), bool(phase)
        self.g = _Grasp(bk)
        self.nq = 0

    def reset(self, meta):
        self.g.reset(meta)
        self.nq = 0

    def observe(self, actions):
        self.g.observe(actions)

    def compose(self, obs, meta):
        from .conditionals import Composition
        w = (0.0 if self.g.carrying else 1.0) if self.phase else self.w
        if self.nq < 2:
            logging.info(f"gpc: query {self.nq}, w {w} on '{self.pa}', {1 - w} on '{self.pg}'")
        self.nq += 1
        return Composition([self.pa, self.pg], np.array([w, 1.0 - w], np.float32)).check()

    def __call__(self, obs, pre, meta=None):
        return None, None


@torch.no_grad()
def identify(bk, demos, task, batch=24, seed=0):
    """-> ([K, L, T, D] float32, frames): the mean expert-layer input residual per denoising step, layer and suffix token over frames
    0-119 of the task's first 20 demonstrations, the policy sampling under the demo's instruction (noise seed `seed` + batch)."""
    layers = expert_layers(bk)
    K = int(bk.num_steps)
    st = dict(k=-1, acc=None)

    def pre(i):
        def f(mod, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            if i == 0:
                st["k"] += 1
            k = min(st["k"], K - 1)
            if st["acc"] is None:
                st["acc"] = torch.zeros((K, len(layers)) + tuple(h.shape[1:]), dtype=torch.float32, device=h.device)
            st["acc"][k, i] += h.float().sum(0)
            return None
        return f
    hs = [ly.register_forward_pre_hook(pre(i), with_kwargs=True) for i, ly in enumerate(layers)]
    n = 0
    try:
        frames = []
        for e in demos.episodes(task)[:NDEMO]:
            ep = demos.load(e, images=True)
            frames += [(ep, i) for i in range(len(ep["frame_index"])) if ep["frame_index"][i] < NFRAME]
        for b, s in enumerate(range(0, len(frames), batch)):
            chunk = frames[s:s + batch]
            obs = bk.observations([demos.observation(ep, i) for ep, i in chunk])
            p = bk.prefix(obs)
            st["k"] = -1
            bk.sample(len(chunk), p, seed=seed + b)
            assert st["k"] == K - 1, f"inject identify: {st['k'] + 1} expert passes, expected {K}"
            n += len(chunk)
    finally:
        for h in hs:
            h.remove()
    return (st["acc"] / n).cpu().numpy().astype(np.float32), n


class InjectHooks:
    """Serve hooks: the residual at every expert layer input l >= 1 moved by the source-mean difference (or, rep, the source mean written
    there), from the object source before the grasp and the goal source from it; k = the denoising step of the query."""

    def __init__(self, bk, m_a, m_g, rep=False):
        self.layers = expert_layers(bk)
        self.K = int(bk.num_steps)
        assert m_a.shape == m_g.shape and m_a.shape[:2] == (self.K, len(self.layers)), (m_a.shape, m_g.shape)
        self.rep = bool(rep)
        dev = bk.device
        self.ma, self.mg = torch.as_tensor(m_a, device=dev), torch.as_tensor(m_g, device=dev)
        self.d = self.ma - self.mg
        self.inc = self.d.clone()                                       # the per-layer increment: the shift does not accumulate
        self.inc[:, 2:] = self.d[:, 2:] - self.d[:, 1:-1]
        self.g = _Grasp(bk)
        self.k, self.nq, self._warned = -1, 0, False
        self._h = [ly.register_forward_pre_hook(self._pre(i), with_kwargs=True) for i, ly in enumerate(self.layers)]

    def _pre(self, i):
        def f(mod, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            if i == 0:
                self.k += 1
                return None
            k = min(max(self.k, 0), self.K - 1)
            if h.shape[1:] != self.d.shape[2:]:
                if not self._warned:
                    logging.warning(f"inject: residual {tuple(h.shape)} vs latent {tuple(self.d.shape)}; no injection")
                    self._warned = True
                return None
            if self.rep:
                src = self.mg if self.g.carrying else self.ma
                out = src[k, i].to(h.dtype)[None].expand_as(h).clone()
            else:
                s = -1.0 if self.g.carrying else 1.0
                out = h + (s * self.inc[k, i]).to(h.dtype)[None]
            if i == 1 and k == 0 and self.nq <= 2:
                logging.info(f"inject: query {self.nq}, {'rep' if self.rep else 'add'}, carrying {self.g.carrying}, "
                             f"|shift| {float(self.d[k, i].norm()):.1f}, |h| {float(h[0].float().norm()):.1f}")
            if args:
                return (out,) + tuple(args[1:]), kwargs
            return args, {**kwargs, "hidden_states": out}
        return f

    def reset(self, meta):
        self.g.reset(meta)
        self.nq = 0

    def rewrite(self, obs, meta):
        self.k = -1
        self.nq += 1
        return obs

    def observe(self, actions):
        self.g.observe(actions)

    def __call__(self, obs, pre, meta=None):
        return None, None
