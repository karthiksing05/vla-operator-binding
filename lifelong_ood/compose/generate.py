"""Top-down generation from a seed composite (TRELLIS P8-P10, App. B; FRAMEWORK P3).

For a composite on the agenda, the recombination pool is the set of stored decompositions under the
deepest ancestor of its content leaf whose count exceeds tau_gen (the recall threshold).  The pool's
candidates are ranked by a CONTEXT CALLBACK  context_fn(store, candidate, emitted) -> score, so a
closed-loop controller can rank expansions against the current observation; the best candidate's parts
are expanded in order, composites recursively through their own pools, primitives emitted (an emit
callback lets the controller execute them, so the next ranking sees the new observation).  Generation
halts when every element is a primitive.
"""
import numpy as np


def recombination_pool(store, el, kind="then"):
    """Decompositions stored under the recall anchor, restricted to the slot's class: the same complexity
    tag the parent's content expects for this slot (TRELLIS P9: candidates whose class the parent does not
    expect are eliminated).  Falls back to the element itself."""
    node = el.cnt_node
    if node is None:
        return [el]
    anchor = node
    while anchor.parent is not None and not anchor.count > store.cfg.tau_gen:
        anchor = anchor.parent
    pool = []
    for k in anchor.all_members():
        c = store.elements.get(k)
        if c is not None and c.kind == kind and c.complexity == el.complexity:
            pool.append(c)
    return pool or [el]


def context_scale(store):
    """One per-dimension variance for every candidate (the median variance over the prototype pool), so
    candidates are compared on the same scale."""
    nodes = store.pool()[0]
    if not nodes:
        return np.full(store.ctx_dim, store.cfg.prior_var)
    return np.median(np.stack([n.var() for n in nodes]), axis=0)


def context_fn_obs(get_obs=None, scale=None):
    """Default context callback: log-likelihood of [left context, current observation] under the
    candidate's stored context description, with a shared per-dimension variance (context_scale).
    `get_obs()` returns the controller's current observation (or None)."""
    cache = {}

    def fn(store, cand, emitted):
        d, k = store.d, store.k
        if "v" not in cache:
            cache["v"] = context_scale(store) if scale is None else np.full(store.ctx_dim, scale)
        v = cache["v"]
        left = emitted[-1].code if emitted else np.zeros(d)
        c = cand.ctx_vec
        sl = slice(d, 2 * d)
        s = -0.5 * store.cfg.w_nb * np.sum((left - c[sl]) ** 2 / v[sl])
        if k and get_obs is not None:
            o = get_obs()
            so = slice(3 * d, 3 * d + k)
            s += -0.5 * store.cfg.w_obs * np.sum((o - c[so]) ** 2 / v[so])
        return float(s)
    return fn


def generate(store, seed, context_fn, emit_fn=None, max_steps=200):
    out, trace = [], []

    def expand(el):
        if len(out) >= max_steps:
            return
        if el.kind != "then":
            out.append(el)
            if emit_fn is not None:
                emit_fn(el)
            return
        pool = recombination_pool(store, el)
        scores = [context_fn(store, c, out) for c in pool]
        best = pool[int(np.argmax(scores))]
        trace.append({"slot": el.key, "chosen": best.key, "pool": len(pool), "self": best.key == el.key})
        for part in best.parts:
            expand(part)

    expand(seed)
    return out, trace
