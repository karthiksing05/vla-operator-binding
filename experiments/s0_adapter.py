"""PLAN_SOLVE S0 (docs/PLAN_S0.md): the BINDING ADAPTER.  A small module trained once per policy (frozen) that writes a role-wise
binding into the deep image tokens where the policy reads it: an additive edit of the agentview image tokens' cached keys and
values at the prefix's layers LAYERS (10-17; R3c's locus: a whole-image deep K/V splice moved the binding +0.44, a region splice
did not).  The edit is SCENE-WIDE: every image position's edit depends on its own (k, v), on the role masks there, and on the
mask-pooled (k, v) of each role.

    delta_l(p) = f_l(h_p, m_obj(p), m_goal(p), d_obj, d_goal) - f_l(h_p, 0, 0, 0, 0)
    h_p = [k_l(p); v_l(p)],  d_role = sum_p m_role(p) h_p / sum_p m_role(p)
f_l = Linear(3 x 512 + 2 -> R) -> GELU -> Linear(R -> 512) (the last layer zero-initialised).  With both masks zero the edit is
exactly zero (the subtraction), so the adapter changes nothing unless a binding is given.
"""
from __future__ import annotations

import torch
from torch import nn

LAYERS = tuple(range(10, 18))
NIMG, HD, R = 256, 256, 256


class BindAdapter(nn.Module):
    def __init__(self, layers=LAYERS, hidden=R):
        super().__init__()
        self.layers = tuple(layers)
        self.ln = nn.ModuleList([nn.LayerNorm(2 * HD) for _ in self.layers])
        self.w1 = nn.ModuleList([nn.Linear(3 * 2 * HD + 2, hidden) for _ in self.layers])
        self.w2 = nn.ModuleList([nn.Linear(hidden, 2 * HD) for _ in self.layers])
        for w in self.w2:
            nn.init.zeros_(w.weight)
            nn.init.zeros_(w.bias)

    def nbytes(self):
        return int(sum(p.numel() for p in self.parameters()) * 2)

    def _f(self, j, h, mo, mg, do, dg):
        x = torch.cat([h, do.expand_as(h), dg.expand_as(h), mo, mg], -1)
        return self.w2[j](nn.functional.gelu(self.w1[j](x)))

    def edit(self, kv, m_obj, m_goal):
        """kv: a transformers DynamicCache of the prefix ([B, 1, T, 256] per layer); m_obj, m_goal: [B, 256] agentview token masks.
        -> a NEW cache whose layers in LAYERS carry the edited image keys / values (others shared)."""
        out = type(kv)()
        mo, mg = m_obj.float()[..., None], m_goal.float()[..., None]                         # [B, 256, 1]
        for l in range(len(kv.key_cache)):
            k, v = kv.key_cache[l], kv.value_cache[l]
            if l in self.layers:
                j = self.layers.index(l)
                h = self.ln[j](torch.cat([k[:, 0, :NIMG], v[:, 0, :NIMG]], -1).float())     # [B, 256, 512]
                do = (mo * h).sum(1, keepdim=True) / mo.sum(1, keepdim=True).clamp(min=1e-6)
                dg = (mg * h).sum(1, keepdim=True) / mg.sum(1, keepdim=True).clamp(min=1e-6)
                z = torch.zeros_like
                d = self._f(j, h, mo, mg, do, dg) - self._f(j, h, z(mo), z(mg), z(do), z(dg))
                dk, dv = d[..., :HD].to(k.dtype), d[..., HD:].to(v.dtype)
                k = torch.cat([k[:, :, :NIMG] + dk[:, None], k[:, :, NIMG:]], 2)
                v = torch.cat([v[:, :, :NIMG] + dv[:, None], v[:, :, NIMG:]], 2)
            out.update(k, v, l)
        return out

    def apply(self, pre, m_obj, m_goal):
        from lifelong_ood.policy.backbone import Prefix
        return Prefix(state=pre.state, pad_masks=pre.pad_masks, kv=self.edit(pre.kv, m_obj, m_goal), n=pre.n)
