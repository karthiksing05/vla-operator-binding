"""ADAPT: low-rank experts on pi0.5's ACTION EXPERT, trained by flow matching with the VLM prefix cached.

Knowledge insulation (arXiv 2505.23705) means the action loss never shaped the VLM: an expert that lives in the
action expert reproduces pi0.5's own training regime and leaves the prefix untouched, so the prefix of every
training observation is computed ONCE (no_grad) and each step runs only the 300M expert with the adapters.

    state, banks = inject(bk)                               # KeyedLoRA banks on the expert's Linear layers
    params = add_node(banks, "task35", rank=16)
    bank = PrefixBank(bk, obs_dicts)                        # cached prefixes, gathered by row
    train(bk, state, "task35", params, bank, chunks_full, steps=1000)   # flat LoRA (few-shot baseline)
    train(..., anchor=(anchor_bank, anchor_chunks))         # + the r ~ 0 anchor term (fold)
    save(banks, "task35", path); load(bk, path) -> (state, key)

Loss = openpi's: t ~ Beta(1.5, 1) * 0.999 + 0.001, x_t = t eps + (1 - t) a, u = eps - a, mean (v - u)^2 over the
full padded chunk.  The anchor term = mean (v_on - v_off)^2 on library observations, with the base policy's own
velocity (expert off) as the target: no teacher copy.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ..experts.lora import add_node, inject_keyed_lora, keys_on


SCOPES = {
    # where the expert's adapters go: the whole action expert (every Linear of the 18 layers, 163 banks), its last two
    # layers, or ONLY the final projection from the expert's (image-conditioned) hidden state to the velocity -- the
    # smallest adapter that can still correct actions from what the policy sees
    "expert": ("expert", None),
    "last2": ("expert", ("layers.16.", "layers.17.")),
    "out": ("model", ("action_out_proj",)),
}


def inject(bk, state=None, exclude=("lm_head", "embed_tokens"), scope="expert"):
    state = state if state is not None else {"on": frozenset()}
    where, include = SCOPES[scope]
    module = bk.expert if where == "expert" else bk.model
    banks = inject_keyed_lora(module, state, exclude=exclude, include=include)
    return state, banks


def n_params(params):
    return int(sum(p.numel() for p in params))


class PrefixBank:
    """Prefixes of many observations cached as ONE batched Prefix (built in blocks), gathered by row index.

    store: where the cached keys / values live -- the policy's device (default) or "cpu", which holds them in host memory and
    moves each gathered batch to the GPU: the same values, for banks larger than the GPU (about 26 MB per observation in
    fp32).  Blocks are copied into preallocated tensors, so the peak is the bank plus one block (a concatenation held two
    copies: experiments/r17_nft.py --recovery ran out of memory at 1,033 observations)."""

    def __init__(self, bk, obs_dicts, block=32, store=None):
        from .backbone import Prefix
        self.bk = bk
        n = len(obs_dicts)
        self.state = self.pad = self.k = self.v = None
        for i in range(0, n, block):
            p = bk.prefix(bk.observations(obs_dicts[i:i + block]))
            if self.k is None:
                dev = torch.device(store) if store else p.state.device
                new = lambda t: torch.empty((n,) + tuple(t.shape[1:]), dtype=t.dtype, device=dev)     # noqa: E731
                self.state, self.pad = new(p.state), new(p.pad_masks)
                self.k = [new(t) for t in p.kv.key_cache]
                self.v = [new(t) for t in p.kv.value_cache]
                self._cache_type = type(p.kv)
            j = slice(i, i + len(p.state))
            self.state[j].copy_(p.state)
            self.pad[j].copy_(p.pad_masks)
            for l in range(len(self.k)):
                self.k[l][j].copy_(p.kv.key_cache[l])
                self.v[l][j].copy_(p.kv.value_cache[l])
            del p
        self.n = n
        self._Prefix = Prefix

    def take(self, idx):
        idx = torch.as_tensor(np.asarray(idx), device=self.state.device, dtype=torch.long)
        dev = self.bk.device
        g = lambda t: t.index_select(0, idx).to(dev, non_blocking=True)                       # noqa: E731
        kv = self._cache_type()
        for l in range(len(self.k)):
            kv.update(g(self.k[l]), g(self.v[l]), l)
        return self._Prefix(state=g(self.state), pad_masks=g(self.pad), kv=kv, n=len(idx))

    def bytes(self):
        return int(sum(t.numel() * t.element_size() for t in self.k + self.v))


class LazyPrefixBank:
    """PrefixBank's interface for banks too large to cache (about 26 MB per observation): keeps the observations and computes
    each gathered batch's prefix on the fly with the frozen VLM, under no_grad.  The same values as a cached bank, at one VLM
    forward per training step (baselines that train on thousands of frames: experiments/b_strap.py, b_er.py)."""

    def __init__(self, bk, obs_dicts):
        self.bk = bk
        self.obs = list(obs_dicts)
        self.n = len(self.obs)

    def take(self, idx):
        with torch.no_grad():
            return self.bk.prefix(self.bk.observations([self.obs[int(i)] for i in np.asarray(idx)]))

    def bytes(self):
        return 0


def flow_loss(bk, v, u, xt, t):
    """The flow-matching loss of a batch: mean ||v - u||^2, or the backbone's own objective when it defines one (X-VLA predicts
    the clean chunk, so its loss lives in a_hat space: backbone_xvla.XVLABackbone.flow_loss)."""
    if hasattr(bk, "flow_loss"):
        return bk.flow_loss(v, u, xt, t)
    return ((v - u) ** 2).mean()


def _flow_batch(n, H, A, gen, device):
    b = torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
    t = (b.sample((n,)) * 0.999 + 0.001).to(device)
    eps = torch.randn((n, H, A), generator=gen).to(device)
    return t, eps


def train(bk, state, key, params, bank, chunks_full, steps=1000, lr=1e-4, batch=32, seed=0, anchor=None,
          anchor_weight=1.0, log_every=100, log=print, weight_decay=0.0, groups=None, gate_of=None, eval_fn=None, eval_every=0,
          profile=None):
    """Flow-matching fine-tune of the expert `key` on (bank rows, chunks_full [N, H, A] padded, normalised).
    anchor = (anchor_bank, anchor_chunks_full): the r ~ 0 half, target = the base velocity (expert off).
    groups = [(row indices, rows per batch), ...]: every batch draws that many rows from each group (default: uniform over the
    bank).  gate_of(row indices) -> a gate the batch's velocity runs under (e.g. attention binding per row; attn_bias.biased).
    eval_fn() -> float, called every eval_every steps (plan F1: held-out flow loss; it must draw no random numbers of its own:
    heldout_loss with fixed_draws), recorded in hist["eval_step"] / hist["eval"].  profile: a dict that accumulates the
    synchronised seconds of the batch's prefixes (take: a VLM forward for a lazy bank, a gather for a cached one), the forward
    and backward pass, and the optimiser step, plus their counts (plan F1's fit-time profile).  Both default off: unchanged.
    -> dict of loss curves."""
    torch.manual_seed(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    A = torch.as_tensor(np.asarray(chunks_full, np.float32), device=bk.device)
    on = keys_on(state, {key})
    hist = dict(step=[], loss=[], anchor=[])
    if eval_fn is not None:
        hist["eval_step"], hist["eval"] = [], []
    t0 = time.time()
    tick = _ticker(profile)
    for s in range(1, steps + 1):
        if groups is None:
            idx = rng.integers(0, bank.n, size=min(batch, bank.n))
        else:
            idx = np.concatenate([np.asarray(g)[rng.integers(0, len(g), size=k)] for g, k in groups])
        tick(None)
        pre = bank.take(idx)
        tick("take")
        a = A[idx]
        t, eps = _flow_batch(len(idx), bk.H, bk.A, gen, bk.device)
        xt = t[:, None, None] * eps + (1 - t[:, None, None]) * a
        u = (eps - a).reshape(len(idx), -1)
        with on():
            v = bk.v_full(xt, t, pre, None if gate_of is None else gate_of(idx))
        loss = flow_loss(bk, v, u, xt, t)
        la = torch.zeros((), device=bk.device)
        if anchor is not None:
            abank, achunks = anchor
            j = rng.integers(0, abank.n, size=min(batch, abank.n))
            apre = abank.take(j)
            aa = torch.as_tensor(np.asarray(achunks[j], np.float32), device=bk.device)
            ta, ea = _flow_batch(len(j), bk.H, bk.A, gen, bk.device)
            xa = ta[:, None, None] * ea + (1 - ta[:, None, None]) * aa
            with torch.no_grad():
                v_off = bk.v_full(xa, ta, apre)
            with on():
                v_on = bk.v_full(xa, ta, apre)
            la = ((v_on - v_off) ** 2).mean()
        tot = loss + anchor_weight * la
        opt.zero_grad(set_to_none=True)
        tot.backward()
        tick("fwd_bwd")
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        tick("opt")
        if s % log_every == 0 or s == 1:
            hist["step"].append(s); hist["loss"].append(float(loss)); hist["anchor"].append(float(la))
            log(f"[adapt] {key} step {s}/{steps}: flow {float(loss):.4f} anchor {float(la):.5f} ({time.time() - t0:.0f}s)")
        if eval_fn is not None and eval_every and s % eval_every == 0:
            hist["eval_step"].append(s); hist["eval"].append(float(eval_fn()))
            tick(None)
    return hist


def _ticker(profile):
    """-> tick(name): adds the synchronised seconds since the previous tick to profile[name] (name None: only restarts the clock).
    A no-op when profile is None."""
    if profile is None:
        return lambda name: None
    last = [time.time()]

    def tick(name):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.time()
        if name is not None:
            profile[name] = profile.get(name, 0.0) + now - last[0]
            profile[name + "_n"] = profile.get(name + "_n", 0) + 1
        last[0] = now
    return tick


def fixed_draws(n_rows, H, A, n=256, batch=32, seed=1234):
    """Plan F1: n fixed (row, time, noise) draws for a held-out flow loss, in batches -> [(idx, t, eps)] on the CPU.  Times follow
    openpi's Beta(1.5, 1) * 0.999 + 0.001 by its inverse CDF (u^(2/3)), from their own generator, so a held-out evaluation draws no
    number from the training run's generators."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    rng = np.random.default_rng(seed)
    out = []
    for i in range(0, n, batch):
        k = min(batch, n - i)
        idx = rng.integers(0, n_rows, size=k)
        t = torch.rand((k,), generator=g) ** (2.0 / 3.0) * 0.999 + 0.001
        eps = torch.randn((k, H, A), generator=g)
        out.append((idx, t, eps))
    return out


def heldout_loss(bk, state, keys, bank, chunks_full, draws):
    """Plan F1: the mean flow-matching loss over fixed draws (fixed_draws) with exactly `keys` switched on (an empty set: the
    frozen policy).  No gradient; draws no random numbers."""
    A = torch.as_tensor(np.asarray(chunks_full, np.float32), device=bk.device)
    on = keys_on(state, keys)
    tot, n = 0.0, 0
    with torch.no_grad(), on():
        for idx, t, eps in draws:
            t, eps = t.to(bk.device), eps.to(bk.device)
            x = A[idx]
            xt = t[:, None, None] * eps + (1 - t[:, None, None]) * x
            u = (eps - x).reshape(len(idx), -1)
            v = bk.v_full(xt, t, bank.take(idx))
            tot += float(flow_loss(bk, v, u, xt, t)) * len(idx)
            n += len(idx)
    return tot / max(n, 1)


def save(banks, key, path, meta=None):
    sd = {b.name: dict(A=b.A[key].weight.detach().cpu().half(), B=b.B[key].weight.detach().cpu().half()) for b in banks if key in b.A}
    torch.save(dict(key=key, banks=sd, meta=meta or {}), path)
    return int(sum(v["A"].numel() + v["B"].numel() for v in sd.values()) * 2)


def load(bk, path, state=None):
    """Inject banks (idempotent) and load a saved expert.  -> (state, key, banks)."""
    z = torch.load(path, map_location="cpu")
    key = z["key"]
    state, banks = inject(bk, state, scope=z.get("meta", {}).get("scope", "expert"))
    by_name = {b.name: b for b in banks}
    rank = next(iter(z["banks"].values()))["A"].shape[0]
    add_node(banks, key, rank)
    for name, w in z["banks"].items():
        b = by_name[name]
        b.A[key].weight.data.copy_(w["A"].to(b.A[key].weight.dtype))
        b.B[key].weight.data.copy_(w["B"].to(b.B[key].weight.dtype))
    return state, key, banks


def train_full(bk, bank, chunks_full, steps=1000, lr=2.5e-5, batch=32, seed=0, log_every=100, log=print, weight_decay=1e-10):
    """FULL fine-tuning of the action expert (every parameter) on (bank rows, chunks_full), openpi's flow loss, in mixed
    precision: the model computes in its own dtype (bf16), AdamW updates fp32 master copies (bf16 alone drops the small
    updates of a low learning rate).  -> loss curve."""
    torch.manual_seed(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    rng = np.random.default_rng(seed)
    params = [p for p in bk.expert.parameters()]
    for p in params:
        p.requires_grad_(True)
    master = [p.detach().float().clone().requires_grad_(True) for p in params]
    opt = torch.optim.AdamW(master, lr=lr, weight_decay=weight_decay)
    A = torch.as_tensor(np.asarray(chunks_full, np.float32), device=bk.device)
    hist = dict(step=[], loss=[])
    t0 = time.time()
    for s in range(1, steps + 1):
        idx = rng.integers(0, bank.n, size=min(batch, bank.n))
        pre = bank.take(idx)
        a = A[idx]
        t, eps = _flow_batch(len(idx), bk.H, bk.A, gen, bk.device)
        xt = t[:, None, None] * eps + (1 - t[:, None, None]) * a
        u = (eps - a).reshape(len(idx), -1)
        for p in params:
            p.grad = None
        loss = ((bk.v_full(xt, t, pre) - u) ** 2).mean()
        loss.backward()
        for p, m in zip(params, master):
            m.grad = None if p.grad is None else p.grad.float()
        torch.nn.utils.clip_grad_norm_(master, 1.0)
        opt.step()
        with torch.no_grad():
            for p, m in zip(params, master):
                p.copy_(m.to(p.dtype))
        if s % log_every == 0 or s == 1:
            hist["step"].append(s); hist["loss"].append(float(loss))
            log(f"[full] step {s}/{steps}: flow {float(loss):.4f} ({time.time() - t0:.0f}s)")
    for p in params:
        p.requires_grad_(False)
    return hist


def train_nft(bk, state, key, params, pos_bank, pos_chunks, neg_bank=None, neg_chunks=None, steps=1000, lr=1e-4, batch=32,
              neg_batch=8, pooled=False, beta=1.0, seed=0, log_every=100, log=print):
    """Negative-aware flow matching (DiffusionNFT, arXiv 2509.16117) on the adapter `key`, reference = the frozen prior
    (adapter off).  Positive rows train v+ = (1 - beta) v_ref + beta v_theta toward u = eps - a; negative rows train
    v- = (1 + beta) v_ref - beta v_theta toward their u, which moves v_theta AWAY from them by reflection through the
    prior.  beta = 1: the positive loss is exactly the plain flow-matching loss (the success-only control).
    pooled = sample positives and negatives at their natural frequency; else `neg_batch` of every `batch` are negatives."""
    torch.manual_seed(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(params, lr=lr)
    Ap = torch.as_tensor(np.asarray(pos_chunks, np.float32), device=bk.device)
    An = None if neg_bank is None else torch.as_tensor(np.asarray(neg_chunks, np.float32), device=bk.device)
    nneg = 0 if neg_bank is None else neg_bank.n
    on = keys_on(state, {key})
    hist = dict(step=[], loss=[], neg=[])
    t0 = time.time()

    def part(bank, A, idx, positive):
        pre = bank.take(idx)
        a = A[idx]
        t, eps = _flow_batch(len(idx), bk.H, bk.A, gen, bk.device)
        xt = t[:, None, None] * eps + (1 - t[:, None, None]) * a
        u = (eps - a).reshape(len(idx), -1)
        with on():
            v_th = bk.v_full(xt, t, pre)
        if positive and beta == 1.0:
            v = v_th
        else:
            with torch.no_grad():
                v_ref = bk.v_full(xt, t, pre)
            v = (1 - beta) * v_ref + beta * v_th if positive else (1 + beta) * v_ref - beta * v_th
        return ((v - u) ** 2).mean()

    for s in 1 + np.arange(steps):
        if nneg == 0:
            kn = 0
        elif pooled:
            kn = int(rng.binomial(batch, nneg / (nneg + pos_bank.n)))
        else:
            kn = min(neg_batch, batch - 1)
        kp = batch - kn
        lp = part(pos_bank, Ap, rng.integers(0, pos_bank.n, size=min(kp, pos_bank.n)), True)
        ln = part(neg_bank, An, rng.integers(0, nneg, size=kn), False) if kn else torch.zeros((), device=bk.device)
        tot = (kp * lp + kn * ln) / batch
        opt.zero_grad(set_to_none=True)
        tot.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if s % log_every == 0 or s == 1:
            hist["step"].append(int(s)); hist["loss"].append(float(lp)); hist["neg"].append(float(ln))
            log(f"[nft] {key} step {s}/{steps}: positive {float(lp):.4f} negative {float(ln):.4f} (k-={kn}) ({time.time() - t0:.0f}s)")
    return hist
