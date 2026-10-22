"""Plan D2, the adapted Sparse Diffusion Policy (Wang et al., CoRL 2024, arXiv 2407.01531; the user's decision 4, 2026-10-20).

SDP replaces layers of a diffusion policy with mixtures of experts; each task owns a router that picks the top-k experts per token,
and a new task reuses the frozen experts and trains only its router (SDP's task transfer).  Here the experts are our per-task
one-demonstration LoRAs (rank 16 on every Linear of pi0.5's action expert, launch_pack._lorasc), frozen; every LoRA bank (one per
Linear) gets its own router, a linear map from the token's input to one logit per expert; the top k = 2 logits are softmaxed
(SDP's default top-2 gating), the rest are zero, so a token's delta is  sum_k g_k B_k A_k x.  The router alone is trained on the
new task's one demonstration with openpi's flow loss plus SDP's load-balancing term (switch-transformer style, weight 0.01).

The stacked form is exact and cheap: h = x A_all^T  [.., N, r]; z = g * h; delta = z.reshape(.., N r) B_all^T.
"""
import torch
import torch.nn as nn


class SDPRouters(nn.Module):
    """Top-k routers over N frozen LoRA experts, one router per bank, applied by forward hooks on the KeyedLoRA banks (which stay
    otherwise off: state["on"] empty)."""

    def __init__(self, banks, expert_sds, k=2, init_std=0.01, seed=0):
        super().__init__()
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.banks = [b for b in banks if all(b.name in sd for sd in expert_sds)]
        self.k, self.n = k, len(expert_sds)
        self.routers = nn.ParameterDict()
        self.A, self.B = {}, {}
        for b in self.banks:
            dev = b.base.weight.device
            A = torch.cat([sd[b.name]["A"].float() for sd in expert_sds], 0)               # [N r, in]
            B = torch.cat([sd[b.name]["B"].float() for sd in expert_sds], 1)               # [out, N r]
            self.A[b.name], self.B[b.name] = A.to(dev), B.to(dev)
            key = b.name.replace(".", "__")
            self.routers[key] = nn.Parameter((torch.randn(self.n, A.shape[1], generator=g) * init_std).to(dev))
        self.r = next(iter(self.A.values())).shape[0] // self.n
        self.aux = []
        self.handles = []
        self.collect_aux = False

    def _hook(self, b):
        key = b.name.replace(".", "__")

        def hook(mod, inp, out):
            x = inp[0]
            xf = x.float()
            logits = xf @ self.routers[key].t()                                            # [.., N]
            top, idx = logits.topk(self.k, dim=-1)
            gate = torch.zeros_like(logits).scatter(-1, idx, torch.softmax(top, -1))       # top-k softmax, zeros elsewhere
            h = (xf @ self.A[b.name].t()).reshape(*x.shape[:-1], self.n, self.r)
            z = (h * gate[..., None]).reshape(*x.shape[:-1], self.n * self.r)
            if self.collect_aux:                                                           # SDP's load-balancing term
                p = torch.softmax(logits, -1).reshape(-1, self.n).mean(0)
                f = (gate > 0).float().reshape(-1, self.n).mean(0) / self.k
                self.aux.append(self.n * (p * f).sum())
            return out + (z @ self.B[b.name].t()).to(out.dtype)
        return hook

    def attach(self):
        self.detach()
        self.handles = [b.register_forward_hook(self._hook(b)) for b in self.banks]

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def save(self, path, meta):
        torch.save(dict(routers={k: v.detach().cpu() for k, v in self.routers.items()}, k=self.k, meta=meta), path)

    def load_routers(self, sd):
        with torch.no_grad():
            for k_, v in sd.items():
                self.routers[k_].copy_(v.to(self.routers[k_].device))


def load_expert_sds(paths):
    """-> [{bank name: {"A", "B"}}] from adapt.save files (fp16 on disk)."""
    return [torch.load(p, map_location="cpu")["banks"] for p in paths]


def build(bk, paths, k=2, seed=0):
    """Inject the banks (state off), load the experts, -> (routers attached, state, banks)."""
    from . import adapt
    state, banks = adapt.inject(bk)
    state["on"] = frozenset()
    R = SDPRouters(banks, load_expert_sds(paths), k=k, seed=seed)
    R.attach()
    return R, state, banks
