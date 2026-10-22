"""B1 base policy: a flow-matching action-chunk transformer over CompoSuite state (docs/B1_COMPOSUITE.md section 4).

Recipe (stated, not invented):
  * Action chunking with receding-horizon execution from Diffusion Policy (Chi et al., RSS 2023): predict H = 16
    actions from the current state, execute the first 8, replan.  Actions and states are normalised with dataset
    statistics (actions min-max to [-1, 1], states z-scored), as in Diffusion Policy.
  * The generative objective is conditional flow matching with the linear (rectified-flow) path (Lipman et al.
    ICLR 2023; Liu et al. ICLR 2023), as in pi0 (Black et al. 2024): x_t = (1 - t) x0 + t x1, x0 ~ N(0, I), x1 the
    normalised chunk, target velocity x1 - x0, t ~ U(0, 1); sampling integrates 10 Euler steps from t = 0 to 1.
  * The network is a small transformer encoder over H action tokens plus three condition tokens (state, task,
    flow time), DiT-style but with tokens for conditioning (in the spirit of pi0's action expert).

Conditioning (DESIGN C5): `cond="free"` gives each training task a free learned embedding (nn.Embedding) with no
factor structure; `cond="oracle"` replaces it with a linear map of the 16-dim factor multi-hot (the upper bound).
The 16-dim task indicator inside CompoSuite's observation is ALWAYS removed from the state (`state_slice`), so the
free policy never sees factor labels.  LoRA (Hu et al. 2021) wraps every Linear inside the transformer blocks.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

H, EXEC, ACT_DIM = 16, 8, 8


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.r = 0
        self.A = self.B = None
        self.scale = 1.0

    def add_lora(self, r, alpha=None):
        self.r = r
        self.A = nn.Parameter(torch.randn(r, self.base.in_features, device=self.base.weight.device) / math.sqrt(r))
        self.B = nn.Parameter(torch.zeros(self.base.out_features, r, device=self.base.weight.device))
        self.scale = (alpha or r) / r

    def forward(self, x):
        y = self.base(x)
        if self.r:
            y = y + (x @ self.A.t() @ self.B.t()) * self.scale
        return y


class Block(nn.Module):
    def __init__(self, d, heads, mlp=4):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = LoRALinear(nn.Linear(d, 3 * d))
        self.proj = LoRALinear(nn.Linear(d, d))
        self.fc1 = LoRALinear(nn.Linear(d, mlp * d))
        self.fc2 = LoRALinear(nn.Linear(mlp * d, d))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(a)
        return x + self.fc2(F.gelu(self.fc1(self.n2(x))))


def time_embed(t, d):
    half = d // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None] * 1000 * f[None]
    return torch.cat([a.sin(), a.cos()], -1)


class FlowPolicy(nn.Module):
    def __init__(self, state_dim, n_tasks, cond="free", d=256, layers=4, heads=4, emb_dim=32, mh_dim=16):
        super().__init__()
        self.cfg = dict(state_dim=state_dim, n_tasks=n_tasks, cond=cond, d=d, layers=layers, heads=heads,
                        emb_dim=emb_dim, mh_dim=mh_dim)
        self.cond = cond
        self.state_in = nn.Sequential(nn.Linear(state_dim, d), nn.GELU(), nn.Linear(d, d))
        if cond == "free":
            self.task_emb = nn.Embedding(n_tasks, emb_dim)
            nn.init.normal_(self.task_emb.weight, std=0.5)
            self.task_in = nn.Linear(emb_dim, d)
        else:
            self.task_in = nn.Linear(mh_dim, d)
        self.time_in = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.act_in = nn.Linear(ACT_DIM, d)
        self.pos = nn.Parameter(torch.zeros(1, H + 3, d))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.out_n = nn.LayerNorm(d)
        self.out = nn.Linear(d, ACT_DIM)

    # task conditioning: an index (free) or a multi-hot (oracle) -> (B, emb) vector; `z` overrides the free lookup
    def task_vec(self, task=None, z=None):
        if self.cond == "free":
            return self.task_in(self.task_emb(task) if z is None else z)
        return self.task_in(task)

    def velocity(self, x_t, t, state, task=None, z=None):
        c = torch.stack([self.state_in(state), self.task_vec(task, z), self.time_in(time_embed(t, self.pos.shape[-1]))], 1)
        h = torch.cat([c, self.act_in(x_t)], 1) + self.pos
        for b in self.blocks:
            h = b(h)
        return self.out(self.out_n(h[:, 3:]))

    def loss(self, state, chunk, task=None, z=None, reduce=True, x0=None, t=None):
        x1 = chunk
        x0 = torch.randn_like(x1) if x0 is None else x0
        t = torch.rand(x1.shape[0], device=x1.device) if t is None else t
        xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
        err = (self.velocity(xt, t, state, task, z) - (x1 - x0)).pow(2).mean((1, 2))
        return err.mean() if reduce else err

    @torch.no_grad()
    def sample(self, state, task=None, z=None, steps=10, gen=None):
        x = torch.randn(state.shape[0], H, ACT_DIM, device=state.device, generator=gen)
        for i in range(steps):
            t = torch.full((state.shape[0],), i / steps, device=state.device)
            x = x + self.velocity(x, t, state, task, z) / steps
        return x

    # LoRA: freeze everything, add rank-r adapters on every block Linear
    def add_lora(self, r):
        for p in self.parameters():
            p.requires_grad_(False)
        params = []
        for m in self.modules():
            if isinstance(m, LoRALinear):
                m.add_lora(r)
                params += [m.A, m.B]
        return params


def n_params(m, trainable=False):
    return sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable))
