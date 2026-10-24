"""PLAN L, PROBE L2: FAST COMPOSITION FROM LANGUAGE PROTOTYPES, NO GRADIENT (docs/PLAN_LANG.md).

A language prototype is a content word's encoder state in a sentence where the policy grounds it: the mean layer-input residual
of the word's tokens over a TRAINING task's demo frames, under that task's own instruction.  Four operators compose the new
instruction from such prototypes; nothing is fitted, and the only data any of them reads from the new task is its ONE demo:

  (a) word      the new instruction; at each content phrase's tokens (object, goal), every encoder layer in [lo, hi], the
                residual is shifted by  S_src[l, w_src] - S_new|src[l, w]  (policy/bind.py's "delta" splice, generalised: any
                source sentence, any layer span, any alignment).  Candidates: the training sentences that contain the word.
  (b) tmpl      as (a), with the new instruction first RE-EXPRESSED in a training template ("put the {O} {G}" or "pick up the {O}
                and place it {G}"), so the word's prototype lands in the placement it was trained in (Z2's leak).
  (c) tli       Li et al.'s text-latent interpolation (policy/tli.py), the pair and lambda chosen on the demo.
  (d) role      (a) on a per-role schedule: the object phrase's edit while the gripper is not carrying, the goal phrase's while it
                carries (the Z2 77-arm form; online through held_ops._Grasp, offline from the demo's own gripper command).

Selection (experiments/pl2_compose.py) uses only the demo: the flow-matching loss of the composed conditioning on demo 0's
frames (fixed draws, shared by every candidate).  Demo 1 is held out for scoring.  No practice episode, no reward.

Contents
  text      roles(), template_of(), fill(), content_words(), overlaps()
  bindings  identify(): S_src / S_new|src for a source task under several prompts in one pass (cached, results/lang/pl2/bind)
  edits     make_edit(): one phrase's [L, k, D] shift (word-matched by token ids, or right-aligned onto another phrase)
  hooks     Composer (the shift, per-row schedule and TLI ramp, for offline batches) and ComposeHooks (serving, lt:<spec>)
  scoring   DemoScorer: the demo's flow loss under a Composer and prompt
  specs     save_spec() / load_spec(): the served form (json + npz of the materialised shifts)

pi0 / pi0.5 only (PaliGemma's sentencepiece through policy/bind.py's tokenizer access); TLI itself is backbone-agnostic.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "lang", "pl2")
NT = 48                                 # instruction positions kept (policy/bind.py's NT)
NDEMO, NFRAME, FSTRIDE = 3, 120, 3      # binding means: 3 demos x frames 0..119 every 3rd (bind.py: 5 x every frame)
REPLAN = 5                              # the client queries every 5 environment steps (TLI's ramp is per query)
STOP = {"the", "a", "an", "of", "on", "in", "into", "onto", "at", "to", "top", "next", "and", "it", "front", "back", "left",
        "right", "middle", "from", "between", "both"}


# ── text ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def roles(text):
    """-> {"o": object phrase (no leading 'the'), "g": goal phrase ('on the plate')} of a single-step instruction."""
    from lifelong_ood.policy import bind as BD
    o, g = BD.phrases(text)
    return {"o": o.strip(), "g": g.strip()}


def template_of(text):
    """A single-step pick-and-place training instruction -> its template with {O} / {G} slots, or None."""
    from lifelong_ood.policy import primitives as P
    if len(P.steps_of(text)) != 1:
        return None
    r = roles(text)
    o, g = r["o"], r["g"]
    if not o or not g or not re.match(r"^(on|in|into|onto|to)\b", g) or text.count(o) != 1 or text.count(g) != 1:
        return None
    if text.startswith("put both") or not (text.startswith("put the") or text.startswith("pick up the")):
        return None
    t = text.replace(o, "{O}", 1)
    if g not in t:
        return None
    t = t.replace(g, "{G}", 1)
    return t if t.index("{O}") < t.index("{G}") else None


def fill(tmpl, o, g):
    return tmpl.replace("{O}", o).replace("{G}", g)


def content_words(phrase):
    return [w for w in re.findall(r"[a-z]+", phrase.lower()) if w not in STOP]


def overlaps(text, phrase):
    """Does `text` share any content word with `phrase`?"""
    ws = set(re.findall(r"[a-z]+", text.lower()))
    return any(w in ws for w in content_words(phrase))


def contains(text, phrase):
    """-> the matched string (the phrase, else its head noun) if `text` holds it as words, else None."""
    for p in (phrase, phrase.split()[-1]):
        if p and re.search(rf"\b{re.escape(p)}\b", text):
            return p
    return None


# ── bindings ─────────────────────────────────────────────────────────────────────────────────────────────────────
def _bpath(config, task, prompt):
    h = hashlib.sha1(prompt.encode()).hexdigest()[:12]
    return os.path.join(OUT, "bind", f"{config}_{task}_{h}.npy")


def identify(bk, config, demos, task, prompts, batch=24):
    """-> {prompt: [L, NT, D] float16}: the mean layer-input residual at the first NT instruction positions over frames 0..NFRAME-1
    (every FSTRIDE-th) of the source task's first NDEMO demos, under each prompt (the SAME frames for every prompt, so
    S_src - S_new|src isolates what the sentence does).  Cached per (config, task, prompt)."""
    import torch
    out, need = {}, []
    for p in dict.fromkeys(prompts):
        f = _bpath(config, task, p)
        if os.path.exists(f):
            out[p] = np.load(f)
        else:
            need.append(p)
    if not need:
        return out
    cur, num = {}, {}

    def hook(i, h, span):
        off, nv = span
        x = h[:, off:off + NT].float()
        k = x.shape[1]
        msk = (torch.arange(k, device=h.device)[None] < nv.to(h.device)[:, None]).float()
        s = torch.zeros((h.shape[0], NT, h.shape[-1]), device=h.device)
        s[:, :k] = x * msk[..., None]
        c = torch.zeros((h.shape[0], NT), device=h.device)
        c[:, :k] = msk
        cur[i], num[i] = s, c
        return None
    frames = []
    for e in demos.episodes(task)[:NDEMO]:
        ep = demos.load(e, images=True)
        frames += [(ep, i) for i in range(len(ep["frame_index"])) if ep["frame_index"][i] < NFRAME and i % FSTRIDE == 0]
    acc = {p: None for p in need}
    cnt = {p: None for p in need}
    hs = bk.cond_hook(hook)
    try:
        with torch.no_grad():
            for s in range(0, len(frames), max(1, batch // len(need))):
                obs0 = [demos.observation(ep, i) for ep, i in frames[s:s + max(1, batch // len(need))]]
                obs = [dict(o, prompt=p) for p in need for o in obs0]
                cur.clear()
                num.clear()
                bk.prefix(bk.observations(obs))
                L = sorted(cur)
                X = torch.stack([cur[i] for i in L], 1)                 # [B, L, NT, D]
                C = torch.stack([num[i] for i in L], 1)                 # [B, L, NT]
                n = len(obs0)
                for j, p in enumerate(need):
                    x, c = X[j * n:(j + 1) * n].sum(0), C[j * n:(j + 1) * n].sum(0)
                    acc[p] = x if acc[p] is None else acc[p] + x
                    cnt[p] = c if cnt[p] is None else cnt[p] + c
    finally:
        for h in hs:
            h.remove()
    os.makedirs(os.path.join(OUT, "bind"), exist_ok=True)
    for p in need:
        m = torch.where(cnt[p][..., None] > 0, acc[p] / cnt[p].clamp(min=1)[..., None], torch.zeros_like(acc[p]))
        out[p] = m.cpu().numpy().astype(np.float16)
        f = _bpath(config, task, p)
        tmp = f"{f}.{os.getpid()}.tmp.npy"
        np.save(tmp, out[p])
        os.replace(tmp, f)
    return out


# ── edits ────────────────────────────────────────────────────────────────────────────────────────────────────────
def make_edit(bk, config, demos, new_prompt, phrase, src, src_phrase=None, role="o"):
    """One phrase's shift.  Word-matched (src_phrase None): the phrase (else its head noun) has identical token ids in the new
    prompt and in the source instruction (bind.match).  Role-aligned (src_phrase given; the controls): the source's own phrase,
    right-aligned onto the new phrase's tokens (the last token onto the last; extra new tokens take the source's first).
    -> dict(role, src, phrase, new_pos, src_pos, v [L, k, D] float32 = S_src - S_new|src, layer 0 zero) or None."""
    from lifelong_ood.policy import bind as BD
    stext = demos.tasks[int(src)]
    if src_phrase is None:
        m = BD.match(bk, new_prompt, stext, phrase)
        if m is None:
            return None
        pn, ps, used = m
    else:
        a, b = BD.positions(bk, new_prompt, phrase), BD.positions(bk, stext, src_phrase)
        if not a or not b:
            return None
        pn, sp = a[0], b[0]
        ps = [sp[max(0, len(sp) - len(pn) + j)] for j in range(len(pn))]
        used = f"{phrase} <- {src_phrase}"
    B = identify(bk, config, demos, int(src), [stext, new_prompt])
    v = B[stext][:, ps].astype(np.float32) - B[new_prompt][:, pn].astype(np.float32)
    v[0] = 0.0
    return dict(role=role, src=int(src), phrase=used, new_pos=[int(x) for x in pn], src_pos=[int(x) for x in ps], v=v)


# ── hooks ────────────────────────────────────────────────────────────────────────────────────────────────────────
class Composer:
    """The composed conditioning as a cond_hook: word shifts (per role, layer span [lo, hi], schedule "og" or "ph") and/or a TLI
    shift (pair difference d = T1 - T2, Li's ramp alpha = linspace(0, 1, lam + 2)[query]).  Per-row state for offline batches:
    `carry` (bool [B], the role schedule) and `q` (int [B], the query index of each row's frame)."""

    def __init__(self, bk, edits=(), lo=1, hi=None, schedule="og", tli=None):
        import torch
        self.bk = bk
        nl = len(bk.cond_layers())
        self.lo, self.hi = int(lo), int(nl - 1 if hi is None else hi)
        self.schedule = schedule
        self.edits = [(e["new_pos"], torch.as_tensor(e["v"], device=bk.device), e["role"]) for e in edits]
        self.tli = None
        if tli is not None:
            d = torch.as_tensor(np.asarray(tli["d"], np.float32), device=bk.device)
            d[0] = 0.0
            self.tli = (d, np.linspace(0, 1, int(tli["lam"]) + 2))
        self.carry = None
        self.q = None
        self._h = None

    def attach(self):
        if self._h is None:
            self._h = self.bk.cond_hook(self._f)
        return self

    def detach(self):
        for h in self._h or []:
            h.remove()
        self._h = None

    def _f(self, i, h, span):
        import torch
        if i == 0 or span is None or (not self.edits and self.tli is None):
            return None
        off, nv = span
        h = h.clone()
        B = h.shape[0]
        if self.edits and self.lo <= i <= self.hi:
            for pn, v, role in self.edits:
                idx = [off + p for p in pn]
                add = v[i].to(h.dtype)[None].expand(B, -1, -1)
                if self.schedule == "ph":
                    c = self.carry if self.carry is not None else torch.zeros(B, dtype=torch.bool)
                    on = (c if role == "g" else ~c).to(h.device, h.dtype).reshape(B, 1, 1)
                    add = add * on
                h[:, idx] += add
        if self.tli is not None:
            d, al = self.tli
            from lifelong_ood.policy import tli as TL
            k = min(TL.NTOK, h.shape[1] - off)
            q = self.q if self.q is not None else np.zeros(B, int)
            a = np.asarray([al[min(int(x), len(al) - 1)] for x in q], np.float32)
            coef = torch.as_tensor(1.0 - 2.0 * a, device=h.device, dtype=h.dtype).reshape(B, 1, 1)
            m = TL._mask(nv, k, h.device).to(h.dtype)
            h[:, off:off + k] += m[..., None] * coef * d[i][:k].to(h.dtype)[None]
        return h


class ComposeHooks(Composer):
    """Serve hooks (experiments/serve_guided.py mode lt:<spec.json>): the spec's prompt replaces the instruction, its shifts run on
    every prefix pass; schedule "ph" follows the policy's own grasp (held_ops._Grasp); TLI's alpha steps once per query."""

    def __init__(self, bk, spec):
        edits, kw = load_spec(spec) if isinstance(spec, str) else spec
        super().__init__(bk, edits, **{k: kw[k] for k in ("lo", "hi", "schedule", "tli") if k in kw})
        self.prompt = kw.get("prompt")
        self.g = None
        if self.schedule == "ph":
            from lifelong_ood.policy.held_ops import _Grasp
            self.g = _Grasp(bk)
        self.nq = 0
        self.token = None                       # plan M1 (experiments/m1_compose_create.py): an optional KV token appended to the
        if kw.get("token"):                     # prompt's cache (policy/token_fold.py), fitted on the one demo; absent = unchanged
            import torch
            from lifelong_ood.policy.token_fold import KVToken
            d = torch.load(kw["token"], map_location=bk.device)["token"]
            self.token = KVToken(d["k"].float(), d["v"].float()).to(bk.device)
        self.attach()
        logging.info(f"lt: prompt '{self.prompt}', {len(self.edits)} edits "
                     f"({', '.join(e['role'] + ':' + str(e['src']) + ' ' + e['phrase'] for e in edits)}), layers {self.lo}-{self.hi}, "
                     f"schedule {self.schedule}, tli {None if self.tli is None else len(self.tli[1]) - 2}, "
                     f"token {kw.get('token') or None}")

    def transform_prefix(self, pre):
        if self.token is None:
            return pre
        from lifelong_ood.policy.token_fold import attach
        return attach(pre, self.token)

    def reset(self, meta):
        self.nq = 0
        if self.g is not None:
            self.g.reset(meta)

    def observe(self, actions):
        if self.g is not None:
            self.g.observe(actions)

    def rewrite(self, obs, meta):
        import torch
        self.carry = torch.tensor([bool(self.g.carrying)]) if self.g is not None else None
        self.q = np.array([self.nq])
        self.nq += 1
        if self.prompt is not None and obs.get("prompt") != self.prompt:
            if self.nq <= 1:
                logging.info(f"lt: instruction '{obs.get('prompt')}' read as '{self.prompt}'")
            obs = dict(obs, prompt=self.prompt)
        return obs

    def __call__(self, obs, pre, meta=None):
        return None, None


# ── specs ────────────────────────────────────────────────────────────────────────────────────────────────────────
def save_spec(path, edits, prompt, lo=1, hi=None, schedule="og", tli=None, meta=None, token=None):
    """json (everything but arrays) + npz (the shifts, float16; TLI's d) next to it.  token: an optional path to a KV token
    (token_fold format) the served hooks append to the prompt's cache (plan M1); omitted from the json when None."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = {f"v{j}": e["v"].astype(np.float16) for j, e in enumerate(edits)}
    if tli is not None:
        arr["tli_d"] = np.asarray(tli["d"], np.float16)
    np.savez_compressed(path[:-5] + ".npz", **arr)
    js = dict(prompt=prompt, lo=lo, hi=hi, schedule=schedule, tli=None if tli is None else dict(lam=int(tli["lam"]),
              t1=tli.get("t1"), t2=tli.get("t2")), edits=[{k: v for k, v in e.items() if k != "v"} for e in edits], meta=meta or {})
    if token is not None:
        js["token"] = token
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(js, f, indent=1)
    os.replace(tmp, path)


def load_spec(path):
    js = json.load(open(path, encoding="utf-8"))
    z = np.load(path[:-5] + ".npz")
    edits = [dict(e, v=z[f"v{j}"].astype(np.float32)) for j, e in enumerate(js["edits"])]
    tli = None if js.get("tli") is None else dict(js["tli"], d=z["tli_d"].astype(np.float32))
    return edits, dict(prompt=js["prompt"], lo=js["lo"], hi=js["hi"], schedule=js["schedule"], tli=tli, token=js.get("token"))


# ── scoring ──────────────────────────────────────────────────────────────────────────────────────────────────────
class DemoScorer:
    """The ONE demo's flow-matching loss under a composed conditioning.

    frames: raw observation dicts (images, state; the prompt is set per call); chunks: [F, H, 7] raw actions starting at each
    frame (clamped at the end); states: [F, 8] raw states (pi0's delta actions); carry: bool [F] (the role schedule);
    q: int [F] (query index).  K fixed (t, eps) draws per frame from `seed`, the same for every candidate (paired)."""

    def __init__(self, bk, frames, chunks, states, carry, q, K=4, seed=0, batch=16):
        import torch
        from lifelong_ood.policy import data as D
        self.bk, self.frames, self.K, self.batch = bk, frames, int(K), int(batch)
        norm = D.Norm.for_backbone(bk)
        rows = np.stack([norm.normalize(c, state=s) for c, s in zip(chunks, states)])              # [F, H, 7]
        x = np.zeros((len(rows), bk.H, bk.A), np.float32)
        x[:, :, :7] = rows
        self.x = torch.as_tensor(x, device=bk.device)
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        F = len(frames)
        self.t = (torch.rand((F, self.K), generator=g) ** (2.0 / 3.0) * 0.999 + 0.001).to(bk.device)
        self.eps = torch.randn((F, self.K, bk.H, bk.A), generator=g).to(bk.device)
        self.carry = np.asarray(carry, bool)
        self.q = np.asarray(q, int)
        self._obs = {}

    def _observations(self, prompt, s, e):
        key = (prompt, s)
        if key not in self._obs:
            self._obs[key] = self.bk.observations([dict(o, prompt=prompt) for o in self.frames[s:e]])
        return self._obs[key]

    def loss(self, prompt, comp=None, per_frame=False):
        """-> mean squared velocity error over the real action dims (and [F] per frame with per_frame)."""
        import torch
        out = []
        if comp is not None:
            comp.attach()
        try:
            with torch.no_grad():
                for s in range(0, len(self.frames), self.batch):
                    e = min(s + self.batch, len(self.frames))
                    if comp is not None:
                        comp.carry = torch.as_tensor(self.carry[s:e])
                        comp.q = self.q[s:e]
                    pre = self.bk.prefix(self._observations(prompt, s, e))
                    n = e - s
                    t = self.t[s:e].reshape(-1)
                    eps = self.eps[s:e].reshape(n * self.K, self.bk.H, self.bk.A)
                    x = self.x[s:e].repeat_interleave(self.K, 0)
                    xt = t[:, None, None] * eps + (1 - t[:, None, None]) * x
                    u = (eps - x)[:, :, :7]
                    v = self.bk.v_full(xt, t, self.bk.repeat(pre, self.K)).reshape(n * self.K, self.bk.H, self.bk.A)[:, :, :7]
                    out.append(((v - u) ** 2).mean((1, 2)).reshape(n, self.K).mean(1).cpu().numpy())
        finally:
            if comp is not None:
                comp.detach()
                comp.carry = comp.q = None
        pf = np.concatenate(out)
        return (float(pf.mean()), pf) if per_frame else float(pf.mean())
