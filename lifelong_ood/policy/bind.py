"""WORD-LEVEL BINDING SPLICE (plan Z, track B; docs/PLAN_Z.md): the binding between the policy's language-vision encoder and its
action expert, repaired word by word with the bindings the policy already holds.  Training-free: nothing is fitted.

The policy reads the NEW instruction.  At every layer l >= 1 of the conditioning encoder, at the tokens of each content word
(the object phrase, the goal phrase), the residual is moved to the binding that word has under the TRAINING instruction that holds
it (the object word from the object source, the goal phrase from the goal source):
    delta    h[w] += S_src[l, w_src] - S_new|src[l, w]      S = the mean layer-input residual over the source task's demo frames,
                                                            under the source's own instruction (S_src) or the new one (S_new|src):
                                                            the same images, so the shift isolates what the sentence does to the word
    replace  h[w]  = S_src[l, w_src]                        observation-free
Li et al.'s TLI shifts the whole instruction toward one training task and ramps it in time; this edits single words, holds no
schedule, and composes two sources inside one prompt.

Bindings are means over NDEMO demos x frames 0..NFRAME-1 (tli.identify's recipe, all instruction positions up to NT), cached in
results/robot/bind/<config>_<task>[_as_<hash>].npy, computed on first use.  pi0 / pi0.5 (PaliGemma's sentencepiece) only for now.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re

import numpy as np
import torch

NT, NFRAME, NDEMO = 48, 120, 5


# ── tokens ───────────────────────────────────────────────────────────────────────────────────────────────────────
def _tokenize_prompt(bk):
    """The policy's TokenizePrompt transform (openpi), found in its input transforms."""
    seen = set()

    def walk(x):
        if id(x) in seen:
            return None
        seen.add(id(x))
        if type(x).__name__ == "TokenizePrompt":
            return x
        for ch in (getattr(x, "transforms", None) or []):
            r = walk(ch)
            if r is not None:
                return r
        return None
    t = walk(bk.policy._input_transform)
    assert t is not None, "bind: no TokenizePrompt in the policy's input transforms"
    return t


def positions(bk, prompt, phrase):
    """-> (token positions relative to the instruction slot, their ids) of the tokens overlapping `phrase` in the tokenized
    prompt (the policy's own format, BOS included), or None if the phrase is absent."""
    tp = _tokenize_prompt(bk)
    sp = tp.tokenizer._tokenizer
    clean = prompt.strip().replace("_", " ").replace("\n", " ")
    full = f"Task: {clean}, State: 0;\nAction: " if tp.discrete_state_input else clean
    m = re.search(rf"\b{re.escape(phrase)}\b", full)
    if m is None:
        return None
    lo, hi = m.span()
    pieces = sp.EncodeAsImmutableProto(full).pieces
    pos = [j + 1 for j, p in enumerate(pieces) if p.end > lo and p.begin < hi]          # + 1: BOS
    ids = [pieces[j - 1].id for j in pos]
    return (pos, ids) if pos and max(pos) < NT else None


# ── bindings ─────────────────────────────────────────────────────────────────────────────────────────────────────
def path(config, task, prompt=None, root=None):
    root = root or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sfx = "" if prompt is None else "_as_" + hashlib.sha1(prompt.encode()).hexdigest()[:10]
    return os.path.join(root, "results", "robot", "bind", f"{config}_{task}{sfx}.npy")


@torch.no_grad()
def identify(bk, demos, task, prompt=None, batch=24):
    """-> [L, NT, D] float16: the mean layer-input residual at the first NT instruction positions over frames 0..NFRAME-1 of the
    task's first NDEMO demos, under `prompt` (default: the task's own instruction)."""
    cur, num = {}, {}

    def f(i, h, span):
        off, nv = span
        x = h[:, off:off + NT].float()
        k = x.shape[1]
        msk = (torch.arange(k, device=h.device)[None] < nv.to(h.device)[:, None]).float()
        s = torch.zeros((NT, h.shape[-1]), device=h.device)
        s[:k] = (x * msk[..., None]).sum(0)
        c = torch.zeros(NT, device=h.device)
        c[:k] = msk.sum(0)
        cur[i], num[i] = s, c
        return None
    hs = bk.cond_hook(f)
    acc = cnt = None
    try:
        frames = []
        for e in demos.episodes(task)[:NDEMO]:
            ep = demos.load(e, images=True)
            frames += [(ep, i) for i in range(len(ep["frame_index"])) if ep["frame_index"][i] < NFRAME]
        for s in range(0, len(frames), batch):
            cur.clear()
            num.clear()
            obs = [demos.observation(ep, i) for ep, i in frames[s:s + batch]]
            if prompt is not None:
                obs = [dict(o, prompt=prompt) for o in obs]
            bk.prefix(bk.observations(obs))
            L = sorted(cur)
            x, c = torch.stack([cur[i] for i in L]), torch.stack([num[i] for i in L])
            acc, cnt = (x, c) if acc is None else (acc + x, cnt + c)
    finally:
        for h in hs:
            h.remove()
    out = torch.where(cnt[..., None] > 0, acc / cnt.clamp(min=1)[..., None], torch.zeros_like(acc))
    return out.cpu().numpy().astype(np.float16)


def load_or_identify(bk, config, demos, task, prompt=None):
    p = path(config, task, prompt)
    if os.path.exists(p):
        return np.load(p)
    x = identify(bk, demos, task, prompt)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = f"{p}.{os.getpid()}.tmp.npy"
    np.save(tmp, x)
    os.replace(tmp, p)
    return x


def phrases(text):
    """The content phrases of a single-step instruction: (object phrase, goal phrase), e.g. ("wine bottle", "on the stove")."""
    from . import primitives as P
    o = P.object_key(text).split("|", 1)[1]
    g = " ".join(P.goal_key(text).split("|"))
    return re.sub(r"^the ", "", o), g


def match(bk, new, src, phrase):
    """Token positions of `phrase` in the new prompt and in the source prompt with identical ids; falls back to the phrase's last
    word (its head noun).  -> (new positions, source positions, the phrase used) or None."""
    for ph in (phrase, phrase.split()[-1]):
        a, b = positions(bk, new, ph), positions(bk, src, ph)
        if a and b and a[1] == b[1]:
            return a[0], b[0], ph
    return None


class SpliceHooks:
    """Serve hooks: the new instruction, its object phrase bound as under the object source, its goal phrase as under the goal
    source, at every encoder layer >= 1 on every prefix pass.  words="ph": in time as well -- the object phrase's edit before the
    grasp, the goal phrase's from it (the policy's own gripper command, held_ops._Grasp, as the grasp switch reads it)."""

    def __init__(self, bk, config, demos, texts, src_o, src_g, mode="delta", words="og"):
        self.mode, self.words = mode, words
        self.g = None
        if words == "ph":
            from .held_ops import _Grasp
            self.g = _Grasp(bk)
        self.bk, self.config, self.demos, self.texts = bk, config, demos, texts
        self.src = dict(o=int(src_o), g=int(src_g))
        self.edits, self.prompt = None, None
        self._h = bk.cond_hook(self._f)

    def _build(self, prompt):
        po, pg = phrases(prompt)
        edits = []                                                      # (new positions, [L, k, D] target or shift)
        for w, ph in (("o", po), ("g", pg)):
            if w not in self.words and self.words != "ph":
                continue
            t = self.src[w]
            m = match(self.bk, prompt, self.texts[t], ph)
            if m is None:
                logging.warning(f"bind: '{ph}' not matched in source {t} '{self.texts[t]}'; left unbound")
                continue
            pn, ps, used = m
            S = load_or_identify(self.bk, self.config, self.demos, t)[:, ps]
            if self.mode == "delta":
                N = load_or_identify(self.bk, self.config, self.demos, t, prompt)[:, pn]
                v = S.astype(np.float32) - N.astype(np.float32)
            else:
                v = S.astype(np.float32)
            v[0] = 0.0
            edits.append((pn, torch.as_tensor(v, device=self.bk.device), w))
            logging.info(f"bind: {w} '{used}' at {pn} <- source {t} at {ps} ({self.mode}), |v| {float(np.linalg.norm(v)):.1f}")
        self.edits, self.prompt = edits, prompt

    def _f(self, i, h, span):
        if not self.edits or i == 0 or span is None:
            return None
        off, nv = span
        h = h.clone()
        keep = None if self.g is None else ("g" if self.g.carrying else "o")
        for pn, v, w in self.edits:
            if keep is not None and w != keep:
                continue
            idx = [off + p for p in pn]
            if self.mode == "delta":
                h[:, idx] += v[i].to(h.dtype)[None]
            else:
                h[:, idx] = v[i].to(h.dtype)[None].expand(h.shape[0], -1, -1)
        return h

    def reset(self, meta):
        if self.g is not None:
            self.g.reset(meta)

    def observe(self, actions):
        if self.g is not None:
            self.g.observe(actions)

    def rewrite(self, obs, meta):
        if obs.get("prompt") != self.prompt:
            self._build(obs["prompt"])
        return obs

    def __call__(self, obs, pre, meta=None):
        return None, None
