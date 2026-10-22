"""THE LEXICON AUDIT and CAUSAL TRACING of plan L (docs/PLAN_LANG.md, probes L0a / L0b).

Words.  An instruction's first step is parsed by rule (the grammar of lifelong_ood/policy/primitives.step_key and
bind_hooks.split_conj, re-stated here with no torch import so the parse runs anywhere):
    verb   the leading verb ("put", "pick up", "open", "turn on", "push", ...)
    obj    the moved object's head phrase ("black bowl", "cream cheese"), its locating clause and article left out
    rel    the locating relation inside the object phrase ("next to", "on", "between", "from", "at", "in"), when there is one
    goal   the goal phrase, preposition included ("on the plate", "on top of the cabinet"); absent for fixture tasks
    prep   the goal phrase's preposition alone ("on", "in", "on top of")
    func   the article before the object (control: a swap should move the field by about 0)
    whole  the whole instruction (control: the ceiling), swapped for other training instructions
A swap replaces the word's character span by a same-category alternative from the vocabulary of all audited sentences (objects:
the scene's own objects first).

L0a, word effect.  At an observation, every swapped prompt's E[eps | x_t] is evaluated at the SAME points (draft chunks sampled
under the instruction, noised to flow times t with shared noise), next to the instruction and null (the empty prompt, images
kept):
    G(w -> w') = mean_t [ mean_points ||eps(swap) - eps(instr)|| / mean_points ||eps(instr) - eps(null)|| ]
and the grounding of w is the mean over its alternatives.  Endpoint attribution: chunks sampled with the same noise under the
instruction and each swap (deterministic Euler, so the pair differs only by the prompt), integrated to an end-effector endpoint
(protos.store.interpret); the binding direction of a swap is the cosine between the endpoint's horizontal move and the vector
from the word's referent to the alternative's referent (both must be in the scene; 0 = chance).

L0b, causal tracing (ROME, Meng et al. 2202.05262).  The word's input embeddings (layer-0 input of the VLM, its token positions)
get Gaussian noise of 3x the embedding std; the clean state is then restored at ONE (layer input, position group) of the residual
stream (a prefix re-run), or separately in the prefix K/V cache the action expert reads (no re-run).  Recovery
    R = 1 - D(restored, clean) / D(corrupted, clean),   D = mean over points of ||eps - eps_clean||.
Position groups: each instruction token, the "Task:" prefix, the discretised state (pi0.5), the ";\\nAction:" suffix, the images.

GPU functions import torch lazily; parse / vocabulary / scores are numpy only.
"""
from __future__ import annotations

import dataclasses
import hashlib
import re

import numpy as np

LEVELS = (0.2, 0.4, 0.6, 0.8)
KINDS = ("obj", "goal", "verb", "rel", "prep")
CONTROLS = ("func", "whole")
PREDICT_KINDS = ("obj", "goal", "verb", "rel")           # pre-registered (m3_prereg pl1_predict)
VERB_RE = r"^(pick up|put both|put|place|open|close|turn on|turn off|push|stack)\b"
PREP = r"(on top of|in|on|under|to the right of|to the left of|to the front of|into|onto)"      # primitives.PREP
LOC = r"(next to|between|on|in|from|at)"
VERBS = ("put", "pick up", "push", "open", "turn on", "close")
RELS = ("next to", "on", "in", "between", "from", "at")
PREPS = ("on", "in", "under", "on top of", "next to")
FUNCS = ("a", "that")
CAP = dict(obj=10, goal=8, verb=5, rel=5, prep=4, func=2, whole=3)


@dataclasses.dataclass(frozen=True)
class Word:
    kind: str
    phrase: str
    lo: int
    hi: int


def first_step(text):
    """bind_hooks.split_conj's first conjunct ("<A> and <verb> ..."), the whole text for "pick up X and place it ..."."""
    m = re.match(r"(.+?) and ((?:close|open|put|place|turn|stack|push|pick)\b.*)$", text)
    if m is None or text.startswith("pick up"):
        return text
    return m.group(1)


def parse(text):
    """-> list of Word over the instruction's first step (character spans in `text`)."""
    s = first_step(text)
    out = []
    m = re.match(VERB_RE, s)
    if m is None:
        return out
    v = m.group(1)
    vlo, vhi = 0, (3 if v == "put both" else len(v))
    out.append(Word("verb", text[vlo:vhi], vlo, vhi))
    goal = None
    if v == "pick up":
        m2 = re.match(r"pick up (.+?) and (?:place|put) it (.+)$", s)
        if m2 is None:
            olo, ohi = len("pick up "), len(s)
        else:
            olo, ohi = m2.span(1)
            goal = m2.span(2)
    else:
        start = len(v) + 1
        rest = s[start:]
        m2 = re.search(rf"^(.*)\b{PREP} (the (?:[a-z]+ ){{0,3}}[a-z]+(?: of the [a-z]+)?)$", rest)
        if m2 and m2.group(1).strip():
            goal = (start + m2.start(2), len(s))
            olo, ohi = start, start + len(m2.group(1).rstrip())
        else:
            olo, ohi = start, len(s)
    np_ = s[olo:ohi]
    for art in ("both the ", "both ", "the "):                      # the article / quantifier before the head
        if np_.startswith(art):
            if art.endswith("the ") and len(out) == 1:
                tlo = olo + np_.index("the")
                out.append(Word("func", "the", tlo, tlo + 3))
            olo += len(art)
            np_ = np_[len(art):]
            break
    ml = re.match(rf"^(.+?) {LOC} (.+)$", np_)
    if ml:
        out.append(Word("obj", ml.group(1), olo, olo + len(ml.group(1))))
        rlo = olo + ml.start(2)
        out.append(Word("rel", ml.group(2), rlo, rlo + len(ml.group(2))))
    else:
        head = np_.split(" and the ")[0]
        out.append(Word("obj", head, olo, olo + len(head)))
    if goal is not None:
        glo, ghi = goal
        out.append(Word("goal", text[glo:ghi], glo, ghi))
        mp = re.match(PREP, text[glo:ghi])
        if mp:
            out.append(Word("prep", mp.group(1), glo, glo + len(mp.group(1))))
    return out


def swap(text, w, alt):
    return text[:w.lo] + alt + text[w.hi:]


def vocabulary(texts):
    """kind -> sorted unique phrases over the audited sentences."""
    voc = {k: set() for k in KINDS}
    for t in texts:
        ws = parse(t)
        moves = any(w.kind == "goal" for w in ws)                  # objects only from pick-and-place sentences (not "the stove")
        for w in ws:
            if w.kind in voc and (w.kind != "obj" or moves):
                voc[w.kind].add(w.phrase)
    voc["verb"] |= set(VERBS)
    voc["rel"] |= set(RELS)
    voc["prep"] |= set(PREPS)
    return {k: sorted(v) for k, v in voc.items()}


def _h(s):
    return int(hashlib.sha1(s.encode()).hexdigest()[:8], 16)


def alternatives(text, w, voc, train_texts, scene_objs=(), scene_goals=()):
    """Same-category alternatives of word w (deterministic order, capped): the scene's objects / goals first."""
    if w.kind == "whole":
        others = [t for t in train_texts if t != text]
        h = _h(text)
        return list(dict.fromkeys(others[(h + j * 13) % len(others)] for j in range(CAP["whole"])))
    if w.kind == "func":
        return list(FUNCS)
    first = {"obj": list(scene_objs), "goal": list(scene_goals)}.get(w.kind, [])
    pool = list(VERBS) if w.kind == "verb" else list(voc[w.kind])
    ok = lambda p: p != w.phrase and not (w.kind == "obj" and _same_obj(p, w.phrase))   # noqa: E731
    first = [p for p in dict.fromkeys(first) if ok(p)]
    rest = sorted([p for p in dict.fromkeys(pool) if ok(p) and p not in first], key=lambda p: _h(p + w.phrase))
    if w.kind in ("verb", "rel", "prep"):                          # small closed classes: their fixed order
        rest = [p for p in pool if ok(p) and p not in first]
    return (first + rest)[:CAP[w.kind]]


def _same_obj(a, b):
    return a == b or a.split()[-1] == b.split()[-1] and (a in b or b in a)


def conditions(text, voc, train_texts, scene_objs=(), scene_goals=()):
    """-> list of dict(kind, word, alt, prompt, lo, hi): null, instr, every word swap, the controls."""
    C = [dict(kind="null", word="", alt="", prompt="", lo=-1, hi=-1), dict(kind="instr", word="", alt="", prompt=text, lo=-1, hi=-1)]
    words = parse(text) + [Word("whole", text, 0, len(text))]
    for w in words:
        for a in alternatives(text, w, voc, train_texts, scene_objs, scene_goals):
            p = a if w.kind == "whole" else swap(text, w, a)
            if p == text:
                continue
            C.append(dict(kind=w.kind, word=w.phrase, alt=a, prompt=p, lo=w.lo, hi=w.hi))
    return C


# ── token positions (PaliGemma / sentencepiece, bind.positions' convention) ─────────────────────────────────────────
def span_positions(bk, prompt, lo, hi):
    """Token positions (relative to the instruction slot, BOS = 0) of the pieces overlapping characters [lo, hi) of `prompt`."""
    from ..policy import bind as B
    tp = B._tokenize_prompt(bk)
    sp = tp.tokenizer._tokenizer
    clean = prompt.strip().replace("_", " ").replace("\n", " ")
    shift = len("Task: ") if tp.discrete_state_input else 0
    full = f"Task: {clean}, State: 0;\nAction: " if tp.discrete_state_input else clean
    pieces = sp.EncodeAsImmutableProto(full).pieces
    return [j + 1 for j, p in enumerate(pieces) if p.end > lo + shift and p.begin < hi + shift]


def position_groups(bk, observation, prompt):
    """-> list of (name, [positions relative to the instruction slot]) covering the whole tokenized prompt, and the sentence's
    token positions.  pi0.5: "pre" (BOS + "Task:"), one group per sentence token, "state", "suffix"; pi0: BOS + sentence tokens."""
    from ..policy import bind as B
    tp = B._tokenize_prompt(bk)
    sp = tp.tokenizer._tokenizer
    nv = int(observation.tokenized_prompt_mask[0].sum())
    ids = observation.tokenized_prompt[0, :nv].tolist()
    pieces = [sp.IdToPiece(int(i)) for i in ids]
    clean = prompt.strip().replace("_", " ").replace("\n", " ")
    sent = sorted(set(span_positions(bk, prompt, 0, len(clean))))
    groups = []
    if tp.discrete_state_input:
        groups.append(("pre", [p for p in range(0, sent[0])]))
        groups += [(f"t{p}", [p]) for p in sent]
        st = next((j for j in range(sent[-1] + 1, nv) if "State" in pieces[j]), sent[-1] + 1)
        semi = next((j for j in range(nv - 1, st, -1) if ";" in pieces[j]), nv - 1)
        groups.append(("state", list(range(sent[-1] + 1, semi))))
        groups.append(("suffix", list(range(semi, nv))))
    else:
        groups.append(("pre", [0]))
        groups += [(f"t{p}", [p]) for p in sent if p < nv]
    return [(n, g) for n, g in groups if g], sent, pieces


# ── GPU: fields of many prompts at one observation ─────────────────────────────────────────────────────────────────
def shared_points(drafts, levels=LEVELS, n_noise=2, seed=0):
    """{t: (X [n_draft * n_noise, dim])}: (1 - t) d + t eps with fixed eps per (draft, noise, level)."""
    D = np.asarray(drafts, np.float32)
    out = {}
    for j, t in enumerate(levels):
        g = np.random.default_rng(int(seed) * 1000 + 17 * j + 3)
        E = g.standard_normal((n_noise,) + D.shape).astype(np.float32)
        out[float(t)] = ((1.0 - t) * D[None] + t * E).reshape(-1, D.shape[-1]).astype(np.float32)
    return out


def _eps_rows(bk, pre, X, t):
    """E[eps | x_t] for every prefix row at the same points X [n, dim].  -> numpy [rows, n, dim]."""
    import torch
    n = len(X)
    rep = bk.repeat(pre, n)
    Xt = torch.as_tensor(np.tile(X, (pre.n, 1)), device=bk.device)
    with torch.no_grad():
        e = bk.eps(Xt, float(t), prefix=rep)
    return e.reshape(pre.n, n, -1).float().cpu().numpy()


def field_audit(bk, od, prompts, levels=LEVELS, n_draft=2, n_noise=2, n_sample=4, block=16, seed=0):
    """At one raw observation dict `od`: every prompt's eps at shared points and paired samples.
    -> dict(E [C, L, n, dim] float32, rows [C, S, dim], raw [C, S, H, 7], drafts)."""
    import torch
    with torch.no_grad():
        pre_i = bk.prefix(bk.observation(dict(od, prompt=prompts[1])))
        drafts = bk.sample(n_draft, pre_i, seed=11 + seed)
        P = shared_points(drafts, levels, n_noise, seed)
        g = torch.Generator(device="cpu").manual_seed(21 + seed)
        noise = torch.randn((n_sample, bk.dim_full), generator=g).numpy()
        C = len(prompts)
        E = np.zeros((C, len(levels), n_draft * n_noise, bk.dim), np.float32)
        rows = np.zeros((C, n_sample, bk.dim), np.float32)
        raw = np.zeros((C, n_sample, bk.H, 7), np.float32)
        for b0 in range(0, C, block):
            ps = prompts[b0:b0 + block]
            pre = bk.prefix(bk.observations([dict(od, prompt=p) for p in ps]))
            for j, t in enumerate(levels):
                E[b0:b0 + len(ps), j] = _eps_rows(bk, pre, P[float(t)], t)
            rep = bk.repeat(pre, n_sample)
            r = bk.sample(len(ps) * n_sample, rep, noise=np.tile(noise, (len(ps), 1)))
            rows[b0:b0 + len(ps)] = r.reshape(len(ps), n_sample, -1)
            st = pre.state.float().cpu().numpy()
            for c in range(len(ps)):
                raw[b0 + c] = bk.unnormalize(rows[b0 + c], state=st[c])[:, :, :7]
    return dict(E=E, rows=rows, raw=raw, drafts=drafts)


# ── GPU: causal tracing ────────────────────────────────────────────────────────────────────────────────────────────
def embed_std(bk, ods_prompts):
    """The elementwise std of the VLM's layer-0 input at instruction tokens over (observation, prompt) pairs (ROME's noise unit)."""
    import torch
    vals = []

    def f(i, h, span):
        if i == 0 and span is not None:
            off, nv = span
            for r in range(h.shape[0]):
                vals.append(h[r, off:off + int(nv[r])].float().reshape(-1).cpu())
        return None
    hs = bk.cond_hook(f)
    try:
        with torch.no_grad():
            for od, p in ods_prompts:
                bk.prefix(bk.observation(dict(od, prompt=p)))
    finally:
        for h in hs:
            h.remove()
    return float(torch.cat(vals).std())


class _Hook:
    """One cond_hook doing, per batch row: noise at layer 0 on the word positions; restore of the clean state at (layer, positions);
    capture of the clean states."""

    def __init__(self, bk):
        self.bk = bk
        self.noise = None            # [B, k, D] torch, added at layer 0 on self.wpos (absolute)
        self.wpos = None
        self.restore = None          # per row (layer, [absolute positions]) or None
        self.clean = None            # {layer: [T, D]} clean layer inputs of row 0 (batch 1), for restores
        self.capture = None          # dict to fill with layer inputs (batch 1)
        self.h = bk.cond_hook(self._f)

    def _f(self, i, h, span):
        out = None
        if self.capture is not None:
            self.capture[i] = h[0].detach().clone()
        if self.noise is not None and i == 0:
            out = h.clone()
            out[:, self.wpos] += self.noise.to(h.dtype)
        if self.restore is not None:
            for r, rs in enumerate(self.restore):
                if rs is not None and rs[0] == i:
                    if out is None:
                        out = h.clone()
                    out[r, rs[1]] = self.clean[i][rs[1]].to(h.dtype)
        return out

    def remove(self):
        for x in self.h:
            x.remove()


def _kv_rows(kv, idx):
    """A DynamicCache of the rows `idx` of kv (copies, so they can be patched)."""
    kv2 = type(kv)()
    for layer, (k, v) in enumerate(zip(kv.key_cache, kv.value_cache)):
        kv2.update(k[idx].clone(), v[idx].clone(), layer)
    return kv2


def trace_case(bk, od, prompt, wlo, whi, std, levels=LEVELS, n_noise=3, batch=24, seed=0, sigma=3.0):
    """ROME-style tracing of the word at characters [wlo, whi) of `prompt` at observation `od`.
    -> dict(groups [names], layers L, hid [L, G], kv [L, G] recoveries (mean over noise), effect (corrupted vs clean, over the
    instruction-vs-null gap), word (positions), sent (positions))."""
    import torch
    from ..policy.backbone import Prefix
    hk = _Hook(bk)
    try:
        with torch.no_grad():
            obs1 = bk.observation(dict(od, prompt=prompt))
            off = int(bk.text_span(obs1)[0])
            groups, sent, pieces = position_groups(bk, obs1, prompt)
            wpos = span_positions(bk, prompt, wlo, whi)
            G = [(n, [off + p for p in g]) for n, g in groups] + [("img", list(range(0, off)))]
            hk.capture = {}
            pre_c = bk.prefix(obs1)
            hk.clean, hk.capture = hk.capture, None
            L = len(hk.clean)
            pre_n = bk.prefix(bk.observation(dict(od, prompt="")))
            drafts = bk.sample(2, pre_c, seed=11 + seed)
            P = shared_points(drafts, levels, 2, seed)
            X = np.concatenate([P[float(t)] for t in levels])
            tl = np.repeat(np.asarray(levels, np.float32), len(P[float(levels[0])]))

            def eps_all(pre):
                out = []
                for j, t in enumerate(levels):
                    out.append(_eps_rows(bk, pre, P[float(t)], t))
                return np.concatenate(out, 1)                         # [rows, n_all, dim]
            Ec, En = eps_all(pre_c)[0], eps_all(pre_n)[0]
            gap = float(np.linalg.norm(Ec - En, axis=-1).mean())
            g = torch.Generator(device="cpu").manual_seed(1000 + seed)
            D = hk.clean[0].shape[-1]
            Z = (sigma * std * torch.randn((n_noise, len(wpos), D), generator=g)).to(bk.device)
            hk.wpos = [off + p for p in wpos]
            obsB = bk.observations([dict(od, prompt=prompt)] * n_noise)
            hk.noise = Z
            pre_x = bk.prefix(obsB)
            Ex = eps_all(pre_x)
            dx = np.linalg.norm(Ex - Ec[None], axis=-1).mean(1)        # [n_noise]
            # hidden-state restores: rows (layer, group, noise)
            jobs = [(l, gi, k) for l in range(L) for gi in range(len(G)) for k in range(n_noise)]
            hid = np.zeros((L, len(G), n_noise), np.float32)
            for s in range(0, len(jobs), batch):
                jb = jobs[s:s + batch]
                hk.noise = Z[[k for _, _, k in jb]]
                hk.restore = [(l, G[gi][1]) for l, gi, _ in jb]
                pre = bk.prefix(bk.observations([dict(od, prompt=prompt)] * len(jb)))
                hk.restore = None
                Er = eps_all(pre)
                for r, (l, gi, k) in enumerate(jb):
                    hid[l, gi, k] = 1.0 - float(np.linalg.norm(Er[r] - Ec, axis=-1).mean()) / max(dx[k], 1e-9)
            hk.noise = None
            # K/V restores: the corrupted cache with one (layer, group) of the clean cache
            NL = len(pre_x.kv.key_cache)                                  # row NL: the group restored at EVERY layer
            kvj = [(l, gi, k) for l in range(NL + 1) for gi in range(len(G)) for k in range(n_noise)]
            kvr = np.zeros((NL + 1, len(G), n_noise), np.float32)
            for s in range(0, len(kvj), batch):
                jb = kvj[s:s + batch]
                idx = torch.as_tensor([k for _, _, k in jb], device=bk.device)
                kv2 = _kv_rows(pre_x.kv, idx)
                for r, (l, gi, k) in enumerate(jb):
                    pp = torch.as_tensor(G[gi][1], device=bk.device)
                    for ll in (range(NL) if l == NL else (l,)):
                        kv2.key_cache[ll][r][:, pp] = pre_c.kv.key_cache[ll][0][:, pp]
                        kv2.value_cache[ll][r][:, pp] = pre_c.kv.value_cache[ll][0][:, pp]
                pre = Prefix(state=pre_x.state[idx], pad_masks=pre_x.pad_masks[idx], kv=kv2, n=len(jb))
                Er = eps_all(pre)
                for r, (l, gi, k) in enumerate(jb):
                    kvr[l, gi, k] = 1.0 - float(np.linalg.norm(Er[r] - Ec, axis=-1).mean()) / max(dx[k], 1e-9)
    finally:
        hk.remove()
    return dict(groups=np.array([n for n, _ in G]), hid=hid.mean(-1), kv=kvr.mean(-1), hid_all=hid, kv_all=kvr,
                effect=float(dx.mean() / max(gap, 1e-9)), gap=gap, word=np.array(wpos), sent=np.array(sent),
                pieces=np.array(pieces), t=tl)


# ── numpy scores ──────────────────────────────────────────────────────────────────────────────────────────────────
def effects(E, levels=LEVELS):
    """E [C, L, n, dim] with row 0 = null, row 1 = instr -> G [C] (the normalised swap effect; G[1] = 0, G[0] = 1)."""
    E = np.asarray(E, np.float64)
    gap = np.linalg.norm(E[1] - E[0], axis=-1).mean(-1)            # [L]
    d = np.linalg.norm(E - E[1][None], axis=-1).mean(-1)            # [C, L]
    return (d / np.maximum(gap[None], 1e-12)).mean(-1)


def endpoints(raw, eef0, scale):
    """raw [..., H, 7] -> integrated end-effector endpoints [..., 3] (protos.store.interpret's path)."""
    raw = np.asarray(raw, np.float64)
    return np.asarray(eef0, np.float64) + np.cumsum(raw[..., :3] * np.asarray(scale), axis=-2)[..., -1, :]


def direction(end_i, end_s, p_from, p_to):
    """Cosine (horizontal) between the endpoint's move under a swap and the referent -> alternative vector; and whether the swap
    brings the endpoint closer to the alternative's referent.  end_* [S, 3] (paired samples), p_* [3]."""
    d = (np.asarray(end_s) - np.asarray(end_i)).mean(0)[:2]
    v = (np.asarray(p_to) - np.asarray(p_from))[:2]
    cos = float(d @ v / max(np.linalg.norm(d) * np.linalg.norm(v), 1e-12))
    closer = float(np.linalg.norm(np.asarray(end_s)[:, :2] - p_to[:2], axis=1).mean()
                   < np.linalg.norm(np.asarray(end_i)[:, :2] - p_to[:2], axis=1).mean())
    return cos, closer


def auc(score, label):
    """P(score of a positive > score of a negative), ties 1/2."""
    s, y = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()
    return float(gt / (len(pos) * len(neg)))
