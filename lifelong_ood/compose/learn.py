"""Learning = Cobweb's own operators applied to what parsing produces (TRELLIS 3.4, FRAMEWORK L1-L3).

discover     P1 stand-in: the frames of training demonstrations seed the context taxonomy (the
             concept-formation correspondence: a Cobweb tree of Gaussian prototypes over the codes).
learn        L1: parse the demo with the learner's own (ungated) parse; incorporate every primitive
             span into the context taxonomy, every with-composite into the content taxonomy, and every
             committed then-composite into both (composites after their parts, TRELLIS 4.4).
             L2: if the parse leaves a NEW remainder, its frames are incorporated into the context
             taxonomy (Cobweb's create operator makes the new prototype), and the demo is parsed again.
consolidate  canonicalise content references after the context taxonomy restructured (Store.consolidate).
No existing code changes: the old knowledge can only be affected through routing (L3).
"""
import numpy as np

from .parse import parse, primitive_stage


def discover(store, Z, O=None, tag=None):
    for t in range(len(Z)):
        store.ctx.ifit(store.context_vector(Z, O, t, t + 1), payload=("frame", tag, t))


def _runs(spans):
    spans = sorted(spans)
    out = []
    for s, e in spans:
        if out and out[-1][1] == s:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def create(store, Z, O, spans, tag=None):
    """L2: incorporate the frames of an uncovered remainder; returns the ids of new prototypes (pool
    nodes whose members all come from the remainder)."""
    keys = set()
    for s, e in _runs(spans):
        for t in range(s, e):
            key = ("new", tag, t)
            keys.add(key)
            store.ctx.ifit(store.context_vector(Z, O, t, t + 1), payload=key)
    made = []
    for n in store.pool()[0]:
        mem = list(n.all_members())
        if mem and all(m in keys for m in mem):
            made.append(n.id)
    store.created.update(made)
    return made


def incorporate(store, p, label_fn=None):
    Z, O = p.Z, p.O
    for w in p.windows:
        if w.new:
            continue
        w.ctx_node = store.ctx.ifit(w.ctx_vec, payload=w.key)
        store.elements[w.key] = w
        if label_fn is not None:
            store.labels[w.key] = label_fn(w.span)
        if w.kind == "with":
            w.content = store.with_content(w._cover_nodes)
            x, nom = store.content_input(w.content, w.ctx_vec)
            if store.cfg.split:
                w.cnt_node = store.cnt.ifit(x, nom, payload=w.key)
    for c in p.commits:
        A, B = c.parts
        c.content = store.then_content(A, B)
        x, nom = store.content_input(c.content, c.ctx_vec)
        c.cnt_node = store.cnt.ifit(x, nom, payload=c.key)
        if store.cfg.split:
            c.ctx_node = store.ctx.ifit(c.ctx_vec, payload=("ctx", c.key))
        else:
            c.ctx_node = c.cnt_node
        store.elements[c.key] = c
        if label_fn is not None:
            store.labels[c.key] = label_fn(c.span)


def learn(store, Z, O=None, label_fn=None, allow_create=True, tag=None):
    p = parse(store, Z, O, gated=False)
    made = []
    if allow_create and p.new_spans():
        made = create(store, Z, O, p.new_spans(), tag=tag)
        p = parse(store, Z, O, gated=False)
    incorporate(store, p, label_fn)
    store.consolidate()
    return {"parse": p, "created": made, "new_frac_after": p.new_frac()}


def calibrate_content(store, demos):
    """theta_content = low quantile of the per-attribute log-likelihood that in-distribution composites
    get at their gate node, minus a margin."""
    lls = []
    for Z, O in demos:
        p = parse(store, Z, O, gated=False)
        lls += [c._gate_ll for c in p.commits if np.isfinite(c._gate_ll)]
    if lls:
        store.theta_content = float(np.quantile(lls, store.cfg.content_q) - store.cfg.content_margin)
    return store.theta_content


def calibrate_windows(store, demos):
    """theta_new = low frame-weighted quantile of the coverage of the learner's own primitive spans on
    in-distribution demos (no labels), minus a margin."""
    sc, ln = [], []
    for Z, O in demos:
        for w in primitive_stage(store, np.asarray(Z, float), O):
            sc.append(w.cover_score)
            ln.append(w.span[1] - w.span[0])
    if sc:
        store.theta_new = float(np.quantile(np.repeat(sc, ln), store.cfg.new_q) - store.cfg.new_margin)
    return store.theta_new
