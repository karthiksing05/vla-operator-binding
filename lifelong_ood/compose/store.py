"""TRELLIS's two linked taxonomies over elements whose primitives are continuous codes z in R^d.

Context taxonomy (`Store.ctx`, a Gaussian Cobweb over one continuous vector per element)
    [ self code (d) | left neighbourhood (d) | right neighbourhood (d) | observation features (k) ]
    * self = the element's code (pooled frame codes of its span; for a "with"-composite its PoE mean).
      TRELLIS keeps a word's own identity as a hidden content-ref; a continuous primitive has no symbol,
      so its code is its identity and is a visible attribute (weight `w_self`).
    * left / right = distance-weighted means of up to W surface frames just outside the span (TRELLIS
      context windows span surface tokens with distance weighting, App. B); zeros at a sequence end
      (EMPTYNULL).
    * obs = the observation-feature vector at the span's start boundary (optional; k may be 0).
    Every node's self-block marginal N(mean[:d], var[:d]) is a prototype at its own level of
    abstraction; the prototype pool is every non-root node with count >= n_proto.

Content taxonomy (`Store.cnt`, a nominal Cobweb) holds composites by their parts:
    then:  {rel: then, L: bag(ctx ids of part 1), R: bag(ctx ids of part 2), Lc: tag, Rc: tag}
    with:  {rel: with, P: bag(ctx ids of the selected prototypes)}
    A bag is the `bag_size` deepest nodes on the part's context sort path whose count >= n_ref
    (TRELLIS: "a bag of up to three concept identifiers"), uniform weights.  The tag is the "then"
    nesting depth; a with-composite fills a single span and has tag 0.

Relations
    then = temporal order (a composite over adjacent spans);
    with = concurrent factors inside one span: up to K prototypes chosen by greedy submodular
    per-dimension coverage F(S) = sum_r max_{j in S} l_{j,r}(x) and combined by a closed-form weighted
    product of experts with per-dimension weights w_j(r) = softmax_j(l_{j,r}(x) / tau)
    (Wang et al., ICLR 2027 submission, Eqs. 9-12).

`split=False` (ablation) puts content and context into ONE mixed taxonomy: composites are inserted as
[context vector + content attributes], primitives as the context vector alone.
"""
from dataclasses import dataclass, field
import itertools

import numpy as np

from .gcobweb import GCobweb

_LOG2PI = np.log(2 * np.pi)


@dataclass
class Config:
    W: int = 3                 # neighbourhood width in frames
    w_self: float = 1.0
    w_nb: float = 0.5
    w_obs: float = 0.5
    prior_var: float = 0.01    # Cobweb acuity (added variance)
    alpha_content: float = 1e-4
    n_proto: int = 5           # prototype pool: nodes with count >= n_proto
    K: int = 3                 # max prototypes in a with-composite
    gain_min: float = 8.0      # nats per frame: add a prototype only if the coverage gain exceeds this
    poe_tau: float = 0.5       # per-dim softmax temperature (Wang et al. use 0.5)
    seg_pen: float = 32.0      # nats: max log-likelihood loss for explaining two spans by one cover
    cover_q: float = 0.005     # calibration quantile for the coverage threshold
    cover_margin: float = 0.25
    new_q: float = 0.01        # calibration quantile (frame-weighted) of the learner's own windows
    new_margin: float = 0.1
    n_ref: int = 2             # bag members need count >= n_ref
    bag_size: int = 3
    tau_parse: int = 5         # recognition threshold (count) on a non-root ancestor
    tau_gen: int = 10          # recall threshold for the recombination pool
    content_q: float = 0.01
    content_margin: float = 1.0
    split: bool = True         # content/context split (False = one mixed taxonomy)
    flat: bool = False         # flat stores (one-level partitions) instead of hierarchies
    with_on: bool = True       # allow with-composites (False = K=1)
    seed: int = 0


_eid = itertools.count(1)


@dataclass
class Element:
    span: tuple
    kind: str                       # 'prim' | 'with' | 'then'
    parts: list = field(default_factory=list)
    complexity: int = 0
    code: np.ndarray = None         # mean code (d)
    code_var: np.ndarray = None
    cover: list = field(default_factory=list)   # [(node_id, weight-per-dim)] for prim/with
    cover_score: float = 0.0
    new: bool = False               # unrecognised remainder
    ctx_vec: np.ndarray = None
    ctx_node: object = None
    ctx_post: float = 0.0
    content: dict = None
    cnt_node: object = None
    score: float = 0.0
    recognized: bool = True
    key: int = 0

    def __post_init__(self):
        self.key = next(_eid)

    def leaves(self):
        if self.kind == "then":
            for p in self.parts:
                yield from p.leaves()
        else:
            yield self


def loglik_perdim(x, M, V):
    """l_{j,r}(x) = log N(x_r; M_jr, V_jr), shape (P, d)."""
    return -0.5 * (_LOG2PI + np.log(V) + (x[None, :] - M) ** 2 / V)


def submodular_cover(X, M, V, K=3, gain_min=8.0):
    """Greedy maximisation of the per-dimension coverage F(S) = sum_t sum_r max_{j in S} l_{j,r}(x_t)
    (Wang et al. Eqs. 10-11; a sum over the span's frames x_t of their objective, still monotone
    submodular), starting from the best singleton and adding j while the marginal gain per frame exceeds
    gain_min.
    Returns the selected indices and their per-dimension mean log-likelihoods over the frames (|S|, d)."""
    X = np.atleast_2d(X)
    L = -0.5 * (_LOG2PI + np.log(V)[:, None, :] + (X[None, :, :] - M[:, None, :]) ** 2 / V[:, None, :])
    S = [int(np.argmax(L.sum((1, 2))))]
    best = L[S[0]].copy()
    while len(S) < K:
        gains = np.maximum(L, best[None]).sum((1, 2)) - best.sum()
        gains[S] = -np.inf
        j = int(np.argmax(gains))
        if gains[j] / X.shape[0] <= gain_min:      # gain per frame
            break
        S.append(j)
        best = np.maximum(best, L[j])
    return S, L[S].mean(1)


def poe(Ls, M, V, tau=0.5):
    """Weighted diagonal product of experts with w_j(r) = softmax_j(l_{j,r} / tau) (Wang et al. Eq. 9,12)."""
    a = Ls / tau
    a = a - a.max(0, keepdims=True)
    w = np.exp(a)
    w = w / w.sum(0, keepdims=True)
    prec = (w / V).sum(0)
    mean = (w * M / V).sum(0) / prec
    return mean, 1.0 / prec, w


class Store:
    def __init__(self, d, obs_dim=0, cfg=None):
        self.cfg = cfg or Config()
        c = self.cfg
        self.d, self.k = d, obs_dim
        self.ctx_dim = 3 * d + obs_dim
        w = np.concatenate([np.full(d, c.w_self), np.full(2 * d, c.w_nb), np.full(obs_dim, c.w_obs)])
        self.ctx = GCobweb(self.ctx_dim, weights=w, prior_var=c.prior_var, flat=c.flat, seed=c.seed)
        if c.split:
            self.cnt = GCobweb(0, prior_var=c.prior_var, alpha=c.alpha_content, flat=c.flat, seed=c.seed + 1)
        else:
            self.cnt = self.ctx       # one mixed taxonomy
            self.ctx.alpha = c.alpha_content
        self.theta_cover = -np.inf
        self.theta_new = -np.inf
        self.theta_content = -np.inf
        self.elements = {}            # key -> stored Element (generation pools, diagnostics)
        self.labels = {}              # key -> ground-truth label (diagnostics only)
        self.created = set()          # ctx node ids created by L2 (diagnostics)
        self._pool_ver = -1
        self._pool = None
        self._consolidated = 0

    # ---- context descriptions -------------------------------------------------------------------
    def neighbourhood(self, Z, s, e):
        W, d = self.cfg.W, self.d
        left = np.zeros(d)
        if s > 0:
            idx = np.arange(max(0, s - W), s)
            wt = 1.0 / (s - idx)
            left = (wt[:, None] * Z[idx]).sum(0) / wt.sum()
        right = np.zeros(d)
        if e < len(Z):
            idx = np.arange(e, min(len(Z), e + W))
            wt = 1.0 / (idx - e + 1)
            right = (wt[:, None] * Z[idx]).sum(0) / wt.sum()
        return left, right

    def context_vector(self, Z, O, s, e, code=None):
        self_code = Z[s:e].mean(0) if code is None else code
        left, right = self.neighbourhood(Z, s, e)
        obs = O[s] if (O is not None and self.k) else np.zeros(self.k)
        return np.concatenate([self_code, left, right, obs])

    # ---- prototype pool and coverage ------------------------------------------------------------
    def pool(self):
        if self._pool_ver != self.ctx.version or self._pool is None:
            nodes = [n for n in self.ctx.nodes() if n is not self.ctx.root and n.count >= self.cfg.n_proto]
            if not nodes:
                self._pool = ([], np.zeros((0, self.d)), np.ones((0, self.d)))
            else:
                M = np.stack([n.mean[: self.d] for n in nodes])
                V = np.stack([n.var()[: self.d] for n in nodes])
                self._pool = (nodes, M, V)
            self._pool_ver = self.ctx.version
        return self._pool

    def cover(self, x, K=None, frames=None):
        """Coverage of a span by the prototype pool.  Prototypes are selected for the span's frames by
        submodular per-dimension coverage (Wang's query = the span); the span's score is the mean per-frame,
        per-dimension log-likelihood of its frames under the resulting PoE Gaussian, so a span is one
        primitive only if ONE (possibly composite) Gaussian explains every frame.
        Returns (score, nodes, poe mean, poe var, weights)."""
        nodes, M, V = self.pool()
        if not nodes:
            return -np.inf, [], x.copy(), np.full(self.d, self.cfg.prior_var), None
        K = (self.cfg.K if self.cfg.with_on else 1) if K is None else K
        F = x[None, :] if frames is None else np.asarray(frames)
        S, Ls = submodular_cover(F, M, V, K=K, gain_min=self.cfg.gain_min)
        mean, var, w = poe(Ls, M[S], V[S], tau=self.cfg.poe_tau)
        score = float(np.mean(-0.5 * (_LOG2PI + np.log(var)[None] + (F - mean[None]) ** 2 / var[None])))
        return score, [nodes[j] for j in S], mean, var, w

    def calibrate_cover(self, frames):
        """theta_cover = low quantile of single in-distribution frames' coverage, minus a margin."""
        sc = np.array([self.cover(z)[0] for z in frames])
        self.theta_cover = float(np.quantile(sc, self.cfg.cover_q) - self.cfg.cover_margin)
        return self.theta_cover

    # ---- references, content descriptions ---------------------------------------------------------
    def ref_bag(self, node):
        path = [n for n in node.path()[1:] if n.count >= self.cfg.n_ref]
        sel = path[-self.cfg.bag_size:]
        if not sel:
            return {"ROOT": 1.0}
        return {n.id: 1.0 / len(sel) for n in sel}

    def then_content(self, A, B):
        return {"rel": "then", "L": self.ref_bag(A.ctx_node), "R": self.ref_bag(B.ctx_node),
                "Lc": str(A.complexity), "Rc": str(B.complexity)}

    def with_content(self, cover_nodes):
        bag = {}
        for n in cover_nodes:
            for k, v in self.ref_bag(n).items():
                bag[k] = bag.get(k, 0.0) + v
        return {"rel": "with", "P": bag}

    # ---- sorting -------------------------------------------------------------------------------
    def sort_context(self, vec):
        return self.ctx.categorize(vec)

    def content_input(self, content, ctx_vec):
        """What the content taxonomy sees: content only (split) or [context vector + content] (no split)."""
        if self.cfg.split:
            return np.zeros(0), content
        return ctx_vec, content

    def sort_content(self, content, ctx_vec):
        x, nom = self.content_input(content, ctx_vec)
        return self.cnt.categorize(x, nom)

    def tax_score(self, tree, node, post, x, nom):
        """Ranking score at the halting node: the class posterior log P(C|i) (TRELLIS App. B) plus the
        per-attribute log-likelihood, which separates candidates when posteriors saturate."""
        enc = tree.encode_nom(nom, grow=False)
        return post + node.log_prob(x, enc) / max(tree.n_attrs(enc), 1.0)

    def gate(self, tree, node, x, nom, tau):
        """Walk up from the halting node; the first non-root ancestor with count > tau is the gate node.
        Returns (gate node or None, per-attribute log-likelihood of the instance under it)."""
        enc = tree.encode_nom(nom, grow=False)
        cur = node
        while cur is not None and cur.parent is not None:
            if cur.count > tau:
                return cur, cur.log_prob(x, enc) / max(tree.n_attrs(enc), 1.0)
            cur = cur.parent
        return None, -np.inf

    # ---- consolidation (canonicalise references after context restructuring) ----------------------
    def consolidate(self):
        """Split removes a context node; content counts on its id are moved to its surviving ancestor
        (TRELLIS App. B: references are canonicalised to an ancestor when the context tree restructures)."""
        red = self.ctx.redirect
        items = list(red.items())[self._consolidated:]
        self._consolidated = len(red)
        if not items:
            return 0
        moved = 0
        for a in ("L", "R", "P"):
            voc = self.cnt.vocab.get(a)
            if not voc:
                continue
            pairs = []
            for dead, _ in items:
                if dead in voc:
                    alive = self.ctx.resolve(dead)
                    if alive not in voc:
                        voc[alive] = len(voc)
                    pairs.append((voc[dead], voc[alive]))
            if not pairs:
                continue
            for n in self.cnt.nodes():
                if a not in n.nom:
                    continue
                arr = self.cnt._arr(n.nom, a)
                for i_d, i_a in pairs:
                    arr[i_a] += arr[i_d]
                    arr[i_d] = 0.0
                n.nom[a] = arr
            moved += len(pairs)
        if moved:
            self.cnt.version += 1
        return moved

    # ---- sizes ----------------------------------------------------------------------------------
    def sizes(self):
        nodes, _, _ = self.pool()
        out = {"ctx_nodes": self.ctx.n_nodes(), "prototypes": len(nodes),
               "elements": len(self.elements)}
        if self.cfg.split:
            out["cnt_nodes"] = self.cnt.n_nodes()
            out["rules"] = sum(1 for n in self.cnt.nodes() if n.children == [] and n.count >= 2)
        return out
