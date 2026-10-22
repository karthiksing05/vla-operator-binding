"""THE PROTOTYPE STORE (framework v3, docs/FRAMEWORK.md section 2; plan v4 P2).

A prototype is a KEY P = (condition c, level u).  At an observation o the key gives
    a live field     eps(x_t, t | prefix(o, c))           (the backbone's own E[eps | x_t], batched over conditions)
    a mode           mu_P(o)  = backbones.score_ops.mode_ascent at t = u, returned in clean-action space by Tweedie
    an extent        Sigma_P(o) = backbones.score_ops.laplace_diag (4 Hutchinson probes), pulled back to clean space
    a Gaussian       N(mu_P, Sigma_P) whose flow-path noise prediction is analytic (flow_gaussian_eps)
Keys are re-evaluated at every observation; nothing here is fitted.

Conditions.  null (the empty prompt, the images kept), the instruction, object / goal / verb PART prompts and created units.
A part prompt's content words can be spliced (policy/bind.py, "replace" mode, observation-free): at every encoder layer >= 1 the
tokens of the phrase take the mean residual the phrase has under a TRAINING instruction that holds it (the source), so the part
reads as it was trained.  The splice needs the backbone's cond_hook / text_span interface and PaliGemma's tokenizer (pi0 / pi0.5);
on other backbones parts run unspliced (logged).

Hierarchy.  Along u (each key's parent is the same condition at the next coarser level) and along condition granularity
(null < part < instruction < created unit; an instruction's parents are its parts at the same level, a part's parent is null).

Interpretation.  The mode chunk unnormalised by the backbone's own output transform; the integrated end-effector path
(eef_0 + cumsum(scale * dpos), scale calibrated from demonstrations: `displacement_scale`); the object nearest the path's
endpoint and the object the displacement points at (cosine of the horizontal displacement with hand -> object); the gripper
timing (first sign change of the command inside the chunk); the per-dimension precision profile 1 / Sigma by
(position / rotation / gripper) x chunk half.

Backbone contract (PolicyBackbone and its siblings): observations, prefix, repeat, bind, eps, sample, unnormalize, dim, H, a,
path; optional cond_hook / text_span for the splice.  Torch is imported lazily so the numpy parts (conditions, Gaussian,
hierarchy, interpretation) run anywhere.
"""
from __future__ import annotations

import contextlib
import dataclasses
import logging
import re

import numpy as np

LEVELS = (0.1, 0.2, 0.3, 0.4, 0.55, 0.7)
GRAN = {"null": 0, "obj": 1, "goal": 1, "verb": 1, "instr": 2, "created": 3}
GROUPS = {"pos": (0, 1, 2), "rot": (3, 4, 5), "grip": (6,)}


# ── keys and conditions ─────────────────────────────────────────────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class Key:
    cond: str
    level: float


@dataclasses.dataclass
class Condition:
    """One conditioning of the policy at an observation."""
    name: str                       # unique within an observation, e.g. "obj:named", "goal:d1", "verb:own"
    kind: str                       # null / instr / obj / goal / verb / created
    prompt: str
    binds: list = dataclasses.field(default_factory=list)   # [(phrase, source task index, source prompt)] splices
    target: str | None = None       # the object / fixture TYPE the part names (attribution), None for null / verb
    variant: str = "bare"           # bare / plain (unspliced) / tmpl (verb-carrying template)

    def to_json(self):
        return dataclasses.asdict(self)

    @staticmethod
    def from_json(d):
        d = dict(d)
        d["binds"] = [tuple(b) for b in d.get("binds", [])]
        return Condition(**d)


# ── the flow-path Gaussian ──────────────────────────────────────────────────────────────────────────────────────────
def flow_gaussian_eps(x, t, mu, var):
    """E[eps | x_t] for a ~ N(mu, diag var) on the flow path x_t = t eps + (1 - t) a (elementwise; numpy or torch):
        x_t ~ N((1 - t) mu, (1 - t)^2 var + t^2),  Cov(eps, x_t) = t   ->   t (x_t - (1 - t) mu) / ((1 - t)^2 var + t^2)."""
    s = (1.0 - t) ** 2 * var + t ** 2
    return t * (x - (1.0 - t) * mu) / s


def flow_gaussian_a(x, t, mu, var):
    """E[a | x_t] (Tweedie) under the same Gaussian: mu + (1 - t) var (x_t - (1 - t) mu) / ((1 - t)^2 var + t^2)."""
    s = (1.0 - t) ** 2 * var + t ** 2
    return mu + (1.0 - t) * var * (x - (1.0 - t) * mu) / s


def flow_gaussian_score(x, t, mu, var):
    """grad_x log p_t(x) = -(x - (1 - t) mu) / ((1 - t)^2 var + t^2)  (= -E[eps | x_t] / t)."""
    s = (1.0 - t) ** 2 * var + t ** 2
    return -(x - (1.0 - t) * mu) / s


def gaussian_guide(mu, var):
    """-> eps_fn(x_rows, t) for PolicyBackbone.sample(guide=(eps_fn, w_fn)): the key's analytic field on the flow path."""
    def f(x, t):
        import torch
        m = torch.as_tensor(np.asarray(mu, np.float32), device=x.device)
        v = torch.as_tensor(np.asarray(var, np.float32), device=x.device)
        return flow_gaussian_eps(x, float(t), m, v)
    return f


# ── LIBERO part prompts (rule-read from the task's BDDL goal and its instruction; policy/factors.py) ─────────────────
VERB_RE = r"^(pick up|put both|put|place|open|close|turn on|turn off|push|stack)\b"
OTHER_VERBS = ("pick up", "open", "turn on", "push")
GOAL_FALLBACK = (("in the basket", "basket"), ("on the plate", "plate"), ("on the stove", "flat_stove"),
                 ("on top of the cabinet", "wooden_cabinet"))


def verb_of(text):
    m = re.match(VERB_RE, text.strip())
    v = m.group(1) if m else text.split()[0]
    return {"put both": "put", "place": "put"}.get(v, v)


def parts_of(task, phrases):
    """-> dict(verb, obj, obj_type, obj_inst, goal, goal_type) for the task's FIRST goal atom (two-step tasks: their first
    In / On atom; fixture tasks: the fixture is the object and there is no goal)."""
    from ..policy import factors as F
    from ..policy import primitives as P
    keys = [P.step_key(s) for s in P.steps_of(task.language)]
    pick = [i for i, a in enumerate(task.goal) if a[0] in ("In", "On") and len(a) == 3]
    out = dict(verb=verb_of(task.language), obj=None, obj_type=None, obj_inst=None, goal=None, goal_type=None)
    if pick:
        i = pick[0]
        s = task.slots[i]
        out["obj_type"], out["obj_inst"], out["goal_type"] = s[1], task.goal[i][1], s[2]
        own = F.own_phrase(task, phrases)
        out["obj"] = own if own else F.phrase_of(s[1], phrases)
        prep = [k.split("|") for k in keys if k.split("|")[1]]
        if prep:
            out["goal"] = f"{prep[0][1]} {prep[0][2]}"
        else:
            out["goal"] = f"{'in' if s[0] == 'In' else 'on'} the {F.phrase_of(s[2], phrases)}"
    else:
        s = task.slots[0]
        out["obj_type"], out["obj_inst"] = s[1], task.goal[0][1] if len(task.goal[0]) > 1 else None
        tgt = keys[0].split("|")[2]
        out["obj"] = re.sub(r"^the ", "", tgt)
    if out["obj"]:
        out["obj"] = re.sub(r"^the ", "", out["obj"])
    return out


def _source(phrase, train_texts, exclude):
    """The first training task (by index) whose instruction holds `phrase` as whole words, excluding `exclude`; else one holding
    the phrase's head noun.  -> (task index, its prompt) or None."""
    for ph in (phrase, phrase.split()[-1]):
        for k in sorted(train_texts):
            if k in exclude:
                continue
            if re.search(rf"\b{re.escape(ph)}\b", train_texts[k]):
                return k, train_texts[k]
    return None


def conditions_for(task, scene_tasks, phrases, train_texts, exclude=(), kinds="full"):
    """The conditions of one observation of `task` (policy/factors.Task).  train_texts: {training task index: instruction}
    (the splice sources); exclude: training indices never used as a source (leave-task-out: the task's own index).
    kinds="full": null, instruction, 3 object parts (named + 2 distractors), 2 goal parts (own + 1 distractor), 2 verb parts,
    and 4 variants of the named parts (plain = unspliced; tmpl = "pick up the X" / "put it G").  kinds="transfer": null,
    instruction, the named object and own goal parts (bare, spliced) and the object template.  -> list of Condition."""
    from ..policy import factors as F
    pt = parts_of(task, phrases)
    ex = set(exclude)

    def bind(*phrs):
        out = []
        for ph in phrs:
            s = _source(ph, train_texts, ex)
            if s is not None:
                out.append((ph, int(s[0]), s[1]))
        return out

    C = [Condition("null", "null", ""), Condition("instr", "instr", task.language)]
    obj, goal, verb = pt["obj"], pt["goal"], pt["verb"]
    C.append(Condition("obj:named", "obj", f"the {obj}", bind(obj), pt["obj_type"]))
    if goal:
        C.append(Condition("goal:own", "goal", goal, bind(re.sub(r"^\w+( \w+)*? the ", "", goal)), pt["goal_type"]))
    if kinds == "transfer":
        C.append(Condition("obj:named:tmpl", "obj", f"pick up the {obj}", bind(obj), pt["obj_type"], "tmpl"))
        return C
    # distractor objects: other movable types of the scene (not the named one, not a goal target), phrases from the scene
    targets = {s[2] for s in task.slots if len(s) > 2 and s[2]}
    movable = sorted({ty for ty in task.objects.values() if ty != pt["obj_type"] and ty not in targets})
    movable.sort(key=lambda ty: (_source(F.phrase_of(ty, phrases), train_texts, ex) is None, ty))
    for j, ty in enumerate(movable[:2]):
        ph = F.phrase_of(ty, phrases)
        C.append(Condition(f"obj:d{j + 1}", "obj", f"the {ph}", bind(ph), ty))
    # a distractor goal: another goal phrase of the same scene with a different target, else a common one
    if goal:
        cand = []
        for t2 in scene_tasks:
            p2 = parts_of(t2, phrases)
            if p2["goal"] and p2["goal_type"] != pt["goal_type"]:
                cand.append((p2["goal"], p2["goal_type"]))
        cand += [g for g in GOAL_FALLBACK if g[1] != pt["goal_type"]]
        g2, ty2 = cand[0]
        C.append(Condition("goal:d1", "goal", g2, bind(re.sub(r"^\w+( \w+)*? the ", "", g2)), ty2))
    C.append(Condition("verb:own", "verb", verb, bind(verb)))
    v2 = next(v for v in OTHER_VERBS if v != verb and not (verb == "put" and v == "pick up"))
    C.append(Condition("verb:d1", "verb", v2, bind(v2)))
    C.append(Condition("obj:named:plain", "obj", f"the {obj}", [], pt["obj_type"], "plain"))
    C.append(Condition("obj:named:tmpl", "obj", f"pick up the {obj}", bind(obj), pt["obj_type"], "tmpl"))
    if goal:
        C.append(Condition("goal:own:plain", "goal", goal, [], pt["goal_type"], "plain"))
        C.append(Condition("goal:own:tmpl", "goal", f"put it {goal}", bind(re.sub(r"^\w+( \w+)*? the ", "", goal)),
                           pt["goal_type"], "tmpl"))
    return C


# ── the word splice, per row ────────────────────────────────────────────────────────────────────────────────────────
class RowSplice:
    """Per-row replace splices inside one batched prefix pass (bind.SpliceHooks applies one edit set to every row)."""

    def __init__(self, bk, config, demos):
        self.bk, self.config, self.demos = bk, config, demos
        self.ok = hasattr(bk, "cond_hook") and hasattr(bk, "text_span")
        self.log = []

    def edits(self, cond):
        """-> [(token positions in the new prompt, torch [L, k, D] source residuals)] for the condition's binds."""
        if not cond.binds or not self.ok:
            return []
        import torch
        from ..policy import bind as B
        out = []
        for phrase, src, src_prompt in cond.binds:
            m = B.match(self.bk, cond.prompt, src_prompt, phrase)
            if m is None:
                self.log.append(f"{cond.name}: '{phrase}' not matched in source {src}")
                continue
            pn, ps, used = m
            S = B.load_or_identify(self.bk, self.config, self.demos, int(src))[:, ps].astype(np.float32)
            S[0] = 0.0
            out.append((pn, torch.as_tensor(S, device=self.bk.device)))
        return out

    @contextlib.contextmanager
    def active(self, row_edits):
        """row_edits: per batch row a list from edits().  Layer 0 is never edited (bind's convention)."""
        if not any(row_edits) or not self.ok:
            yield
            return

        def f(i, h, span):
            if i == 0 or span is None:
                return None
            off = int(span[0])
            h = h.clone()
            for r, ed in enumerate(row_edits):
                for pn, S in ed:
                    h[r, [off + p for p in pn]] = S[i].to(h.dtype)
            return h
        hs = self.bk.cond_hook(f)
        try:
            yield
        finally:
            for x in hs:
                x.remove()


# ── live fields at one observation ─────────────────────────────────────────────────────────────────────────────────
class Live:
    """The batched live fields of a set of conditions at one observation.  Blocks of at most `block` conditions share one
    batched prefix pass (the VLM sees images and text together, so every condition needs its own row)."""

    def __init__(self, bk, obs, conds, splice=None, block=16):
        import torch
        self.bk, self.conds = bk, list(conds)
        self.names = [c.name for c in self.conds]
        self.blocks = []
        with torch.no_grad():
            for b0 in range(0, len(self.conds), block):
                cs = self.conds[b0:b0 + block]
                ed = [splice.edits(c) if splice is not None else [] for c in cs]
                ctx = splice.active(ed) if splice is not None else contextlib.nullcontext()
                with ctx:
                    pre = bk.prefix(bk.observations([dict(obs, prompt=c.prompt) for c in cs]))
                self.blocks.append((b0, len(cs), pre))

    def eps(self, X, t, rows=512):
        """E[eps | x_t] of every condition at the SAME points X [n, dim] (one t for all).  -> numpy [C, n, dim] float32."""
        import torch
        X = torch.as_tensor(np.asarray(X, np.float32), device=self.bk.device)
        n = len(X)
        out = np.zeros((len(self.conds), n, self.bk.dim), np.float32)
        with torch.no_grad():
            for b0, nb, pre in self.blocks:
                per = max(1, rows // max(n, 1))
                for c0 in range(0, nb, per):
                    c1 = min(nb, c0 + per)
                    sub = pre if (c0 == 0 and c1 == nb) else self._rows(pre, list(range(c0, c1)))
                    rep = self.bk.repeat(sub, n)                       # condition-major: rows c*n .. c*n+n-1
                    Xr = X.repeat(c1 - c0, 1)
                    e = self.bk.eps(Xr, float(t), prefix=rep)
                    out[b0 + c0:b0 + c1] = e.reshape(c1 - c0, n, -1).float().cpu().numpy()
        return out

    def _rows(self, pre, idx):
        """A Prefix restricted to some rows: built by repeating with a 0/1 count (repeat_interleave on the batch axis)."""
        import torch
        k = torch.zeros(pre.n, dtype=torch.long, device=pre.state.device)
        k[idx] = 1
        rep = lambda t: t.repeat_interleave(k.to(t.device), dim=0)      # noqa: E731
        kv = pre.kv
        kv2 = type(kv)()
        for layer, (kk, vv) in enumerate(zip(kv.key_cache, kv.value_cache)):
            kv2.update(rep(kk), rep(vv), layer)
        return dataclasses.replace(pre, state=rep(pre.state), pad_masks=rep(pre.pad_masks), kv=kv2, n=len(idx))

    def modes(self, starts, iters=80, lr=0.02, probes=4, var_cap=4.0, seed=0):
        """Mode ascent of every condition from SHARED starts.  starts: {u: x_t rows [S, dim]}.
        -> {u: dict(xt [C, S, dim], x0 [C, S, dim], ext [C, S, dim])} numpy.  One block only (C <= block)."""
        from ..backbones.score_ops import laplace_diag, mode_ascent
        assert len(self.blocks) == 1, "modes: put the ascended conditions in one block"
        _, C, pre = self.blocks[0]
        out = {}
        for u, S0 in starts.items():
            S0 = np.asarray(S0, np.float32)
            S = len(S0)
            rep = self.bk.repeat(pre, S)
            self.bk.bind(rep)
            xi = np.tile(S0, (C, 1))
            xt, x0 = mode_ascent(self.bk, None, float(u), iters=iters, lr=lr, seed=seed, x_init=xi, n_starts=C * S)
            ext = laplace_diag(self.bk, xt, float(u), probes=probes, seed=seed + 7, var_floor=1e-4, var_cap=var_cap)
            out[u] = dict(xt=xt.reshape(C, S, -1), x0=x0.reshape(C, S, -1), ext=ext.reshape(C, S, -1))
        return out


def shared_starts(drafts, levels=LEVELS, seed=0):
    """x_t starts at each level from draft chunks d_k: (1 - u) d_k + u eps_k, with eps_k fixed per (k, level) -- the same
    starts for every condition, so between-condition differences are the fields', not the starts'.  -> {u: [S, dim]}."""
    D = np.asarray(drafts, np.float32)
    out = {}
    for j, u in enumerate(levels):
        g = np.random.default_rng(int(seed) * 1000 + j)
        out[float(u)] = ((1.0 - u) * D + u * g.standard_normal(D.shape)).astype(np.float32)
    return out


def demo_points(a, levels=LEVELS, n=4, seed=0):
    """Points on the flow path of the demonstration's own chunk a [dim]: x = (1 - u) a + u eps_j.  -> ({u: [n, dim]},
    {u: eps [n, dim]}) -- the cached points where the relation is fitted (eps is the true noise, for flow losses)."""
    a = np.asarray(a, np.float32).reshape(-1)
    X, E = {}, {}
    for j, u in enumerate(levels):
        g = np.random.default_rng(int(seed) * 7919 + 31 * j + 1)
        e = g.standard_normal((n, len(a))).astype(np.float32)
        X[float(u)], E[float(u)] = ((1.0 - u) * a + u * e).astype(np.float32), e
    return X, E


# ── the store: keys, Gaussians, hierarchy ───────────────────────────────────────────────────────────────────────────
class Store:
    """Keys of ONE observation.  add(cond, level, x0 [S, dim], ext [S, dim]) per key; the key's Gaussian is the mean mode over
    starts with the mean Laplace extent."""

    def __init__(self, conds, levels=LEVELS):
        self.conds = {c.name: c for c in conds}
        self.levels = tuple(float(u) for u in levels)
        self.entries = {}

    def add(self, cond, level, x0, ext):
        self.entries[Key(cond, float(level))] = dict(x0=np.asarray(x0, np.float32), ext=np.asarray(ext, np.float32))

    def gaussian(self, key):
        e = self.entries[key]
        return e["x0"].mean(0), e["ext"].mean(0)

    def eps(self, key, x, t):
        mu, var = self.gaussian(key)
        return flow_gaussian_eps(x, t, mu, var)

    def parent_u(self, key):
        """Same condition at the next coarser level (larger u), or None at the coarsest."""
        i = self.levels.index(key.level)
        return Key(key.cond, self.levels[i + 1]) if i + 1 < len(self.levels) else None

    def parents_c(self, key):
        """Along condition granularity at the same level: instruction -> its parts; part -> null; created -> instruction."""
        c = self.conds[key.cond]
        g = GRAN[c.kind]
        if g == 0:
            return []
        if c.kind in ("obj", "goal", "verb"):
            return [Key("null", key.level)] if "null" in self.conds else []
        if c.kind == "instr":
            return [Key(n, key.level) for n, cc in self.conds.items() if cc.kind in ("obj", "goal", "verb") and cc.variant == "bare"
                    and n.endswith(("named", "own"))]
        return [Key(n, key.level) for n, cc in self.conds.items() if cc.kind == "instr"]

    def edges(self):
        out = []
        for k in self.entries:
            p = self.parent_u(k)
            if p is not None and p in self.entries:
                out.append((k, p, "u"))
            out += [(k, q, "c") for q in self.parents_c(k) if q in self.entries]
        return out

    def sym_kl(self, a, b):
        m1, v1 = self.gaussian(a)
        m2, v2 = self.gaussian(b)
        d = m1 - m2
        return float(0.5 * ((v1 / v2) + d * d / v2 - 1 + np.log(v2 / v1)).sum() + 0.5 * ((v2 / v1) + d * d / v1 - 1 + np.log(v1 / v2)).sum())


# ── interpretation ──────────────────────────────────────────────────────────────────────────────────────────────────
def displacement_scale(states, actions):
    """Per-axis metres of end-effector motion per unit of position command, by least squares over demonstration frames:
    eef[f + 1] - eef[f] ~ s * a[f, :3].  states [T, >=3], actions [T, 7] (lists of episodes are concatenated)."""
    num, den = np.zeros(3), np.zeros(3)
    for S, A in zip(states, actions):
        d = np.diff(np.asarray(S)[:, :3], axis=0)
        a = np.asarray(A)[:-1, :3]
        num += (d * a).sum(0)
        den += (a * a).sum(0)
    return num / np.maximum(den, 1e-9)


def precision_profile(ext, H=10, A=7):
    """1 / Sigma by (pos / rot / grip) x chunk half.  -> dict name -> float (mean log10 precision)."""
    P = np.log10(1.0 / np.asarray(ext, np.float64).reshape(-1, H, A))
    out = {}
    for g, dims in GROUPS.items():
        for h, sl in (("early", slice(0, H // 2)), ("late", slice(H // 2, H))):
            out[f"{g}_{h}"] = float(P[:, sl][:, :, list(dims)].mean())
    return out


def interpret(raw, eef0, obj_pos, obj_names, scale):
    """raw [H, 7] unnormalised chunk; eef0 [3]; obj_pos [N, 3]; scale [3].  -> dict(path [H, 3], endpoint, nearest (index),
    pointed (index), cos (to each object), grip_first_change (step or -1), grip_closed_frac)."""
    raw = np.asarray(raw, np.float64)
    eef0 = np.asarray(eef0, np.float64)
    path = eef0[None] + np.cumsum(raw[:, :3] * np.asarray(scale)[None], 0)
    end = path[-1]
    P = np.asarray(obj_pos, np.float64)
    d = np.linalg.norm(P - end[None], axis=1)
    disp = (end - eef0)[:2]
    vec = P[:, :2] - eef0[None, :2]
    cos = (vec @ disp) / np.maximum(np.linalg.norm(vec, axis=1) * np.linalg.norm(disp), 1e-12)
    g = raw[:, 6] > 0
    ch = np.where(g[1:] != g[:-1])[0]
    return dict(path=path, endpoint=end, nearest=int(np.argmin(d)), pointed=int(np.argmax(cos)), cos=cos,
                names=list(obj_names), grip_first_change=int(ch[0] + 1) if len(ch) else -1, grip_closed_frac=float(g.mean()))
