"""Bottom-up greedy parsing of a sequence of frame codes (TRELLIS P5-P7, FRAMEWORK P2/P4).

The frontier starts as the demo's frames (primitive windows of length one).  At every step candidates are
proposed over ADJACENT frontier elements, scored, gated, and the best recognised one is committed.

  primitive candidates (both parts primitive spans):  the union is one primitive span iff it costs little
      likelihood to explain it with one cover instead of two (Delta <= seg_pen, see primitive_stage) and
      the prototype
      pool covers it: up to K prototypes are selected for its pooled code by submodular per-dimension
      coverage (a single prototype, or a "with"-composite when the selection needs more than one) and the
      mean per-frame, per-dimension log-likelihood of its frames under their PoE must be >= theta_cover.  Pool members have count >= n_proto on a non-root node, so the
      context-path gate on primitives (TRELLIS App. B) is satisfied by construction.  Rank = -Delta.
  then candidates (adjacent elements once no primitive merge is recognised):  the content description
      (references to the parts' context concepts + complexity tags) is sorted through the content taxonomy
      and the context description through the context taxonomy; recognised iff the walk up from the
      content halting node meets a non-root ancestor with count > tau_parse AND the instance's
      per-attribute log-likelihood under that ancestor clears theta_content (calibrated);
      rank = s_c + s_x (Store.tax_score).

Parsing halts when nothing is recognised or one element remains.  A final primitive span whose coverage is
below theta_new (calibrated on the learner's own windows of in-distribution demos; stricter than the
frame-level theta_cover that only gates merges) is marked NEW (the unrecognised remainder, P4).  With gated=False (the learner's own parse, used to bootstrap the
content taxonomy without supervision) every then-candidate counts as recognised.
"""
from dataclasses import dataclass, field

import numpy as np

from .store import Element


@dataclass
class Parse:
    windows: list                    # primitive spans after the primitive stage (incl. NEW ones)
    frontier: list                   # top-level elements when parsing halted
    commits: list = field(default_factory=list)   # committed then-composites, in order
    Z: np.ndarray = None
    O: np.ndarray = None

    @property
    def root(self):
        return self.frontier[0] if len(self.frontier) == 1 else None

    def boundaries(self):
        return [w.span[0] for w in self.windows[1:]]

    def new_spans(self):
        return [w.span for w in self.windows if w.new]

    def new_frac(self):
        T = len(self.Z)
        return sum(e - s for s, e in self.new_spans()) / max(T, 1)

    def complete(self):
        return self.root is not None and not any(w.new for w in self.windows)


def _window(store, Z, O, s, e):
    x = Z[s:e].mean(0)
    score, nodes, mean, var, w = store.cover(x, frames=Z[s:e])
    el = Element(span=(s, e), kind="with" if len(nodes) > 1 else "prim", complexity=0,
                 code=mean, code_var=var, cover=[n.id for n in nodes], cover_score=score)
    el.new = not (score >= max(store.theta_cover, store.theta_new))
    el._cover_nodes = nodes
    el._w = w
    return el


def primitive_stage(store, Z, O):
    """Greedy agglomerative merging of adjacent primitive spans.  Merging A and B costs
        Delta = LL(A) + LL(B) - LL(A u B),   LL = total frame log-likelihood under the span's own cover,
    the log-likelihood lost by explaining both with one (possibly with-) prototype instead of two.  A merge
    is recognised iff Delta <= seg_pen and the union is still covered (score >= theta_cover); the cheapest
    recognised merge is committed first."""
    T = len(Z)
    wins = [_window(store, Z, O, t, t + 1) for t in range(T)]
    if T == 1:
        return wins
    d = store.d
    pen = store.cfg.seg_pen

    def LL(w):
        return w.cover_score * (w.span[1] - w.span[0]) * d

    def cand(i):
        s, e = wins[i].span[0], wins[i + 1].span[1]
        u = _window(store, Z, O, s, e)
        delta = LL(wins[i]) + LL(wins[i + 1]) - LL(u)
        ok = (delta <= pen) and (u.cover_score >= store.theta_cover)
        return (delta if ok else np.inf), u

    cands = [cand(i) for i in range(len(wins) - 1)]
    while cands:
        i = int(np.argmin([c[0] for c in cands]))
        if not np.isfinite(cands[i][0]):
            break
        wins[i:i + 2] = [cands[i][1]]
        cands.pop(i)
        if i < len(cands):
            cands[i] = cand(i)
        if i > 0:
            cands[i - 1] = cand(i - 1)
    # adjacent NEW spans form one unrecognised remainder (it is segmented only once creation covers it)
    out = []
    for w in wins:
        if out and out[-1].new and w.new:
            s0 = out[-1].span[0]
            m = _window(store, Z, O, s0, w.span[1])
            m.new = True
            out[-1] = m
        else:
            out.append(w)
    return out


def _finish_element(store, Z, O, el):
    """Context description + context sort for an element on the frontier."""
    s, e = el.span
    el.ctx_vec = store.context_vector(Z, O, s, e)      # descriptions use the data (pooled frame codes)
    el.ctx_node, el.ctx_post = store.sort_context(el.ctx_vec)
    return el


def _then_candidate(store, Z, O, A, B, gated):
    s, e = A.span[0], B.span[1]
    el = Element(span=(s, e), kind="then", parts=[A, B], complexity=1 + max(A.complexity, B.complexity))
    el.code = Z[s:e].mean(0)
    el.ctx_vec = store.context_vector(Z, O, s, e)
    el.content = store.then_content(A, B)
    xn, xp = store.sort_context(el.ctx_vec)
    cx, cn = store.content_input(el.content, el.ctx_vec)
    cnode, cpost = store.sort_content(el.content, el.ctx_vec)
    el.ctx_node, el.ctx_post, el.cnt_node = xn, xp, cnode
    s_x = store.tax_score(store.ctx, xn, xp, el.ctx_vec, None)
    s_c = store.tax_score(store.cnt, cnode, cpost, cx, cn)
    el.score = s_c + s_x if store.cfg.split else s_c
    if gated:
        g, ll = store.gate(store.cnt, cnode, cx, cn, store.cfg.tau_parse)
        el.recognized = g is not None and ll >= store.theta_content
        el._gate_ll = ll
    else:
        el.recognized = True
        g, ll = store.gate(store.cnt, cnode, cx, cn, store.cfg.tau_parse)
        el._gate_ll = ll
    return el


def parse(store, Z, O=None, gated=True):
    Z = np.asarray(Z, float)
    wins = primitive_stage(store, Z, O)
    for w in wins:
        _finish_element(store, Z, O, w)
    frontier = list(wins)
    commits = []

    def cand(i):
        A, B = frontier[i], frontier[i + 1]
        if A.new or B.new:
            return None
        return _then_candidate(store, Z, O, A, B, gated)

    cands = [cand(i) for i in range(len(frontier) - 1)]
    while len(frontier) > 1:
        ok = [(c.score, i) for i, c in enumerate(cands) if c is not None and c.recognized]
        if not ok:
            break
        _, i = max(ok)
        el = cands[i]
        frontier[i:i + 2] = [el]
        commits.append(el)
        cands.pop(i)
        if i < len(cands):
            cands[i] = cand(i)
        if i > 0:
            cands[i - 1] = cand(i - 1)
    return Parse(windows=wins, frontier=frontier, commits=commits, Z=Z, O=O)
