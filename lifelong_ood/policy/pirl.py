"""Plan D2 (docs/BASELINES.md, ledger m3_prereg tag g30_rl): pi_RL's Flow-Noise PPO (Chen et al., arXiv 2510.25889; RLinf,
rlinf/models/embodiment/openpi/tasks/rl.py) on pi0.5, with the trainable weights restricted to a rank-16 LoRA on the action expert
(the VLM and the expert's base weights frozen), served online inside our LIBERO harness on one A40.  ReinFlow (arXiv 2505.22094) is
the same learned-noise idea; this arm counts for both.

What follows pi_RL / RLinf (pi05 LIBERO PPO config, libero_spatial_ppo_openpi_pi05.yaml; Flow-Noise as in the ManiSkill configs):
  * the K-step Euler chain x_{k+1} = x_k - dt v(x_k, t_k) + sigma_k * N(0, I) is a Gaussian policy; sigma_k comes from an
    ExploreNoiseNet (MLP 1024 -> 128 -> 64 -> 32, tanh, on the expert's final hidden state of each action token; log-variance
    squashed by tanh into [log s_min^2, log s_max^2]); the chunk's log-likelihood is the JOINT one: the per-element Gaussian
    log-probabilities averaged over the K + 1 chain steps (the initial N(0, I) included), on the executed part of the chunk
    (the first 5 actions -- the client replans every 5 steps, RLinf's num_action_chunks 5 -- and the 7 real dims), summed over
    those 35 elements;
  * the critic: ValueHead (2048 -> 1024 -> 512 -> 256 -> 1, ReLU) on the masked mean of the VLM's prefix output
    (value_after_vlm, value_vlm_mode mean_token), the VLM frozen so the critic's input is detached;
  * chunk-level sparse reward (1 on the last chunk of a successful episode), GAE gamma 0.99 lambda 0.95, normalised advantages,
    PPO clip 0.2 / 0.2, dual clip 3.0, value clip 0.2, Adam betas (0.9, 0.95), gradient clip 1.0, value lr 1e-4, one update
    epoch per batch.
What differs (one A40, LoRA, our protocol):
  * the trainable actor is the per-task one-demonstration LoRA (rank 16; initialised from that task's adapter, launch_pack
    _lorasc) plus the noise head, lr 2e-5 (RLinf's full-expert lr is 5e-6; LoRA usually needs a larger one);
  * K = 10 Euler steps (our serving; RLinf's pi0.5 LIBERO PPO uses 3), so sigma's range is RLinf's [0.08, 0.16] scaled by
    sqrt(3 / 10) to keep the chain's injected variance;
  * a PPO batch is 4 episodes (RLinf: 64 environments x 8 rollout epochs), minibatches of 16 chunks;
  * at test the chain is the deterministic ODE (noise head off) through the trained LoRA: the served artefact is an ordinary
    LoRA file (lora:<path>), so acceptance and test run exactly as every other adapter arm.

Budgets: the policy keeps training until `episodes` episodes have reported their outcome; when the run's wall-clock since its
first episode passes `budget_s` (the library's practice per task, matched), the LoRA is saved to `save_matched`, and the final
one to `save`.  A checkpoint after every update (`ckpt`) makes a requeued run resume (episodes already counted are not redone;
any extra client episodes then only serve).
"""
from __future__ import annotations

import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from .serve import GuidedPolicy


class ExploreNoiseNet(nn.Module):
    """RLinf's ExploreNoiseNet (rlinf/models/embodiment/modules/explore_noise_net.py), learnable schedule."""

    def __init__(self, in_dim, out_dim, smin, smax, hidden=(128, 64)):
        super().__init__()
        dims = [in_dim, *hidden, out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        self.mlp = nn.Sequential(*layers)
        self.register_buffer("lo", torch.log(torch.tensor(smin ** 2)))
        self.register_buffer("hi", torch.log(torch.tensor(smax ** 2)))

    def forward(self, h):
        z = torch.tanh(self.mlp(h))
        return torch.exp(0.5 * (self.lo + (self.hi - self.lo) * (z + 1) / 2))


class ValueHead(nn.Module):
    def __init__(self, in_dim, hidden=(1024, 512, 256)):
        super().__init__()
        dims = [in_dim, *hidden]
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def logprob(x, mu, sigma):
    return -torch.log(sigma) - 0.5 * math.log(2 * math.pi) - 0.5 * ((x - mu) / sigma) ** 2


class PiRLPolicy(GuidedPolicy):
    def __init__(self, bk, lora_path, save, save_matched=None, ckpt=None, budget_s=577.0, episodes=100, batch_eps=4,
                 minibatch=16, lr=2e-5, value_lr=1e-4, num_steps=10, seed=0, sigma=(0.08, 0.16), sigma_steps_ref=3,
                 n_exec=5, gamma=0.99, lam=0.95, clip=0.2, dual_clip=3.0, value_clip=0.2, log=logging.info):
        super().__init__(bk, hooks=None, num_steps=num_steps, seed=seed)
        from . import adapt
        self.adapt = adapt
        self.state, self.key, self.banks = adapt.load(bk, lora_path)
        self.state["on"] = frozenset({self.key})
        self.lora = [q for b in self.banks if self.key in b.A for q in (b.A[self.key].weight, b.B[self.key].weight)]
        for q in self.lora:
            q.requires_grad_(True)
        scale = math.sqrt(sigma_steps_ref / num_steps)
        dev = bk.device
        torch.manual_seed(seed)
        self.noise = ExploreNoiseNet(bk.model.action_out_proj.in_features, bk.A, sigma[0] * scale, sigma[1] * scale).to(dev)
        wv = int(getattr(getattr(bk.model.paligemma_with_expert.paligemma.config, "text_config", None), "hidden_size", 2048))
        self.value = ValueHead(wv).to(dev)
        self.opt_a = torch.optim.Adam(self.lora + list(self.noise.parameters()), lr=lr, betas=(0.9, 0.95), eps=1e-8)
        self.opt_v = torch.optim.Adam(self.value.parameters(), lr=value_lr, betas=(0.9, 0.95), eps=1e-8)
        self.gen = torch.Generator(device="cpu").manual_seed(10_000 + seed)
        self.rng = np.random.default_rng(seed)
        self.save, self.save_matched, self.ckpt = save, save_matched, ckpt
        self.budget_s, self.episodes, self.batch_eps, self.minibatch = float(budget_s), int(episodes), int(batch_eps), int(minibatch)
        self.n_exec, self.gamma, self.lam, self.clip, self.dual_clip, self.value_clip = n_exec, gamma, lam, clip, dual_clip, value_clip
        self.log = log
        self.cur, self.buf = [], []
        self.n_eps, self.outcomes, self.updates, self.elapsed, self.matched_done = 0, [], [], 0.0, False
        self.t_first = None
        self._h = None
        bk.model.action_out_proj.register_forward_hook(lambda m, i, o: setattr(self, "_h", i[0]))
        if ckpt and os.path.exists(ckpt):
            self._load(ckpt)

    # ── persistence ──
    def _load(self, path):
        c = torch.load(path, map_location="cpu", weights_only=False)
        with torch.no_grad():
            for q, w in zip(self.lora, c["lora"]):
                q.copy_(w.to(q.device, q.dtype))
        self.noise.load_state_dict(c["noise"]); self.value.load_state_dict(c["value"])
        self.opt_a.load_state_dict(c["opt_a"]); self.opt_v.load_state_dict(c["opt_v"])
        self.gen.set_state(c["gen"]); self.rng.bit_generator.state = c["rng"]
        self.n_eps, self.outcomes, self.updates = c["n_eps"], c["outcomes"], c["updates"]
        self.elapsed, self.matched_done = c["elapsed"], c["matched_done"]
        self.log(f"[pirl] resumed {path}: {self.n_eps} episodes, {len(self.updates)} updates, {self.elapsed:.0f}s")

    def _save_ckpt(self):
        if not self.ckpt:
            return
        torch.save(dict(lora=[q.detach().cpu() for q in self.lora], noise=self.noise.state_dict(), value=self.value.state_dict(),
                        opt_a=self.opt_a.state_dict(), opt_v=self.opt_v.state_dict(), gen=self.gen.get_state(),
                        rng=self.rng.bit_generator.state, n_eps=self.n_eps, outcomes=self.outcomes, updates=self.updates,
                        elapsed=self.elapsed, matched_done=self.matched_done), self.ckpt + ".tmp")
        os.replace(self.ckpt + ".tmp", self.ckpt)

    def _save_lora(self, path, what):
        meta = dict(method="pi_RL Flow-Noise PPO (LoRA on the action expert)", what=what, episodes=self.n_eps,
                    outcomes=list(self.outcomes), updates=len(self.updates), elapsed_s=self.elapsed, scope="expert")
        self.adapt.save(self.banks, self.key, path + ".tmp", meta=meta)
        os.replace(path + ".tmp", path)
        self.log(f"[pirl] saved {what} LoRA {path} after {self.n_eps} episodes, {self.elapsed:.0f}s")

    # ── the model pieces ──
    @torch.no_grad()
    def _prefix(self, observation):
        """bk.prefix, also returning the VLM's prefix output mean-pooled over valid tokens (the critic's input)."""
        bk, m = self.bk, self.bk.model
        images, img_masks, lang_tokens, lang_masks, state = m._preprocess_observation(observation, train=False)
        embs, pad, att = m.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
        a2 = make_att_2d_masks(pad, att)
        pos = torch.cumsum(pad, dim=1) - 1
        m.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        out, kv = m.paligemma_with_expert.forward(attention_mask=m._prepare_attention_masks_4d(a2), position_ids=pos,
                                                  past_key_values=None, inputs_embeds=[embs, None], use_cache=True)
        from .backbone import Prefix
        o = out[0].float()
        w = pad.float()[..., None]
        feat = (o * w).sum(1) / w.sum(1).clamp_min(1.0)
        return Prefix(state=state, pad_masks=pad, kv=kv, n=int(state.shape[0])), feat

    def _step(self, x, k, pre):
        """One Euler step's Gaussian: (mean, sigma) at chain step k (t_k = 1 - k / K)."""
        bk, K = self.bk, self.num_steps
        n = len(x)
        t = torch.full((n,), 1.0 - k / K, device=bk.device)
        v = bk.v_full(x, t, pre).reshape(n, bk.H, bk.A)
        sig = self.noise(self._h.float())                     # [n, H, A], from the expert's final hidden state
        return x - (1.0 / K) * v, sig

    def _chain_logp(self, chain, pre):
        """The joint log-likelihood of a stored chain [n, K + 1, H, A] (with gradient) -> [n]."""
        lp = [logprob(chain[:, 0], torch.zeros_like(chain[:, 0]), torch.ones_like(chain[:, 0]))]
        for k in range(self.num_steps):
            mu, sig = self._step(chain[:, k], k, pre)
            lp.append(logprob(chain[:, k + 1], mu, sig))
        lp = torch.stack(lp, 1).mean(1)[:, :self.n_exec, :self.bk.a]
        return lp.sum((-1, -2))

    # ── serving + collection ──
    def infer(self, obs, *, noise=None):
        t0 = time.monotonic()
        meta = {k[5:]: v for k, v in obs.items() if k.startswith("meta/")}
        obs = {k: v for k, v in obs.items() if not k.startswith("meta/")}
        if meta.get("episode_start"):
            self.ep_q = 0
            self._end_episode(meta)
            if self.t_first is None:
                self.t_first = time.monotonic() - self.elapsed
        bk = self.bk
        observation = bk.observation(obs)
        pre, feat = self._prefix(observation)
        g = torch.Generator(device="cpu").manual_seed(self.seed + self.calls)
        x = torch.randn((1, bk.H, bk.A), generator=g).to(bk.device)
        training = self.n_eps < self.episodes
        chain = [x]
        with torch.no_grad():
            for k in range(self.num_steps):
                mu, sig = self._step(x, k, pre)
                x = mu + (sig * torch.randn(x.shape, generator=self.gen).to(bk.device) if training else 0.0)
                chain.append(x)
            if training:
                ch = torch.stack(chain, 1)
                lp = float(self._chain_logp(ch, pre)[0])
                val = float(self.value(feat)[0])
                self.cur.append(dict(obs=obs, chain=ch[0].cpu(), logp=lp, value=val))
        rows = x.reshape(1, -1)[:, bk.real].float().cpu().numpy()
        actions = bk.unnormalize(rows, state=observation.state[0].float().cpu().numpy())[0]
        self.calls += 1
        self.ep_q += 1
        self.secs += time.monotonic() - t0
        return {"actions": np.asarray(actions, np.float32), "state": np.asarray(obs["observation/state"], np.float32)}

    def _end_episode(self, meta):
        if not self.cur or meta.get("prev_success") is None:
            self.cur = []
            return
        if self.n_eps >= self.episodes:
            self.cur = []
            return
        r = float(bool(meta["prev_success"]))
        self.outcomes.append(r)
        self.buf.append((self.cur, r))
        self.cur = []
        self.n_eps += 1
        self.log(f"[pirl] episode {self.n_eps}: success {r:.0f} (last 10: {np.mean(self.outcomes[-10:]):.2f})")
        if len(self.buf) >= self.batch_eps or self.n_eps >= self.episodes:
            info = self.update()
            self.updates.append(info)
            self.log(f"[pirl] update {len(self.updates)}: {info}")
        if self.t_first is not None:
            self.elapsed = time.monotonic() - self.t_first
        if self.save_matched and not self.matched_done and self.elapsed >= self.budget_s:
            self._save_lora(self.save_matched, f"matched budget ({self.budget_s:.0f}s)")
            self.matched_done = True
        if self.n_eps >= self.episodes:
            if self.save_matched and not self.matched_done:     # the whole run fitted inside the matched budget
                self._save_lora(self.save_matched, "matched budget (run ended first)")
                self.matched_done = True
            self._save_lora(self.save, f"{self.episodes} episodes")
        self._save_ckpt()

    # ── PPO ──
    def update(self):
        bk = self.bk
        S, A_, R_, V_ = [], [], [], []
        adv, ret = [], []
        for steps, r in self.buf:
            T = len(steps)
            vals = [s["value"] for s in steps]
            gae, a_ = 0.0, [0.0] * T
            for i in reversed(range(T)):
                last = i == T - 1
                rew = r if last else 0.0
                nv = 0.0 if last else vals[i + 1]
                d = rew + self.gamma * nv - vals[i]
                gae = d + self.gamma * self.lam * (0.0 if last else gae)
                a_[i] = gae
            for i, s in enumerate(steps):
                S.append(s); adv.append(a_[i]); ret.append(a_[i] + vals[i])
        self.buf = []
        adv = np.asarray(adv, np.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret = np.asarray(ret, np.float32)
        N = len(S)
        order = self.rng.permutation(N)
        stats = dict(pl=[], vl=[], ratio=[], clipfrac=[])
        for i in range(0, N, self.minibatch):
            idx = order[i:i + self.minibatch]
            observation = bk.observations([S[j]["obs"] for j in idx])
            pre, feat = self._prefix(observation)
            ch = torch.stack([S[j]["chain"] for j in idx]).to(bk.device)
            old = torch.as_tensor([S[j]["logp"] for j in idx], device=bk.device)
            a = torch.as_tensor(adv[idx], device=bk.device)
            lp = self._chain_logp(ch, pre)
            ratio = torch.exp((lp - old).clamp(-20, 20))
            p1 = -a * ratio
            p2 = -a * ratio.clamp(1 - self.clip, 1 + self.clip)
            pl = torch.max(p1, p2)
            pl = torch.where(a < 0, torch.min(pl, -a * self.dual_clip), pl).mean()
            self.opt_a.zero_grad(set_to_none=True)
            pl.backward()
            torch.nn.utils.clip_grad_norm_(self.lora + list(self.noise.parameters()), 1.0)
            self.opt_a.step()
            v = self.value(feat)
            vo = torch.as_tensor([S[j]["value"] for j in idx], device=bk.device)
            rt = torch.as_tensor(ret[idx], device=bk.device)
            vc = vo + (v - vo).clamp(-self.value_clip, self.value_clip)
            vl = 0.5 * torch.max((v - rt) ** 2, (vc - rt) ** 2).mean()
            self.opt_v.zero_grad(set_to_none=True)
            vl.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), 1.0)
            self.opt_v.step()
            stats["pl"].append(float(pl)); stats["vl"].append(float(vl)); stats["ratio"].append(float(ratio.mean()))
            stats["clipfrac"].append(float(((ratio - 1).abs() > self.clip).float().mean()))
            del pre
        return dict(chunks=N, **{k: round(float(np.mean(v)), 5) for k, v in stats.items()})
