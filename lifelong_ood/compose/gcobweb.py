"""A numpy-only mixed Gaussian / nominal Cobweb (incremental concept formation).

The logic follows `cobweb-private/src/cobweb/cobweb_torch.py` (CobwebTorchTree / CobwebTorchNode, the
continuous Cobweb of MacLellan et al.), ported to numpy and extended with nominal attributes:

  * a concept stores a count, a running mean and sum of squared deviations (Welford) over a continuous
    vector, plus weighted value counts for each nominal attribute;
  * variance = meanSq / count + prior_var (the torch version's "acuity adjustment");
  * the score of a child against its parent is the information gain KL(child || parent), summed over
    continuous dimensions (diagonal Gaussians, optionally weighted per dimension; weight 0 = hidden
    attribute, stored but not used to sort) and over nominal attributes (Laplace-smoothed categoricals,
    smoothing alpha, vocabulary = values seen so far for that attribute);
  * partition utility = mean over children of p(child) * KL(child || parent), exactly as in the torch
    code; the four operators (best / new / merge / split), the fringe split at a leaf and the
    exact-match leaf update are the torch code's;
  * categorisation is the torch code's best-first search with score log p(c | x) + log p(x | c);
  * the basic level of a node is the ancestor maximising CU(c) = p(c) KL(c || root) (torch get_basic).

Nominal attribute values may be *bags*: {value: weight} with weights summing to one (TRELLIS Appendix B,
"each content attribute value is a small bag over concept identifiers").  Counts accumulate the weights;
the likelihood of a bag is the mixture  P(bag | C, a) = sum_v w_v P(v | C, a)  (the bag is an uncertain
single value).

`flat=True` turns the taxonomy into a one-level partition (root -> clusters; a cluster never splits),
used as the "flat store" ablation.
"""
import heapq
import itertools
from collections import Counter
import math
import random

import numpy as np

_LOG2PI = math.log(2 * math.pi)
_ids = itertools.count(1)


class Node:
    __slots__ = ("id", "count", "mean", "msq", "nom", "children", "parent", "tree", "members")

    def __init__(self, tree):
        self.id = next(_ids)
        self.tree = tree
        self.count = 0.0
        self.mean = np.zeros(tree.dim)
        self.msq = np.zeros(tree.dim)
        self.nom = {}            # attr -> dense count array (may be shorter than the vocabulary)
        self.children = []
        self.parent = None
        self.members = []        # payloads of instances stored at this leaf

    # ---- statistics -----------------------------------------------------------------------------
    def stats(self):
        return (self.count, self.mean, self.msq, self.nom)

    def var(self):
        return self.tree.var_of(self.msq, self.count)

    def depth(self):
        d, n = 0, self
        while n.parent is not None:
            d, n = d + 1, n.parent
        return d

    def path(self):
        """Root ... self."""
        out, n = [], self
        while n is not None:
            out.append(n)
            n = n.parent
        return out[::-1]

    def is_leaf(self):
        return not self.children

    def subtree(self):
        st = [self]
        while st:
            n = st.pop()
            yield n
            st.extend(n.children)

    def all_members(self):
        for n in self.subtree():
            yield from n.members

    def log_prob(self, x, nom=None):
        return self.tree.log_prob_stats(self.stats(), x, nom)

    def category_utility(self):
        r = self.tree.root
        return (self.count / r.count) * self.tree.kl(self.stats(), r.stats())

    def basic(self):
        """torch get_basic: the ancestor (incl. self) with the largest CU."""
        best, best_cu, cur = self, self.category_utility(), self
        while cur.parent is not None:
            cur = cur.parent
            cu = cur.category_utility()
            if cu > best_cu:
                best, best_cu = cur, cu
        return best


class GCobweb:
    def __init__(self, dim, weights=None, prior_var=0.01, alpha=1e-4, flat=False, seed=0):
        self.dim = int(dim)
        self.w = np.ones(self.dim) if weights is None else np.asarray(weights, float)
        self.prior_var = prior_var
        self.alpha = alpha
        self.flat = flat
        self.vocab = {}          # attr -> {value: index}
        self.rng = random.Random(seed)
        self.version = 0         # bumped on every structural / statistical change
        self.redirect = {}       # dead node id -> surviving node id (split removes a node)
        self.ops = Counter()     # operator usage (diagnostics)
        self.root = Node(self)

    # ---- instance encoding ----------------------------------------------------------------------
    def encode_nom(self, nom, grow=True):
        """{attr: value or {value: w}} -> {attr: (idx array, weight array)}; unseen values get new indices
        when grow=True, else are kept as index -1 (probability alpha-smoothed)."""
        if not nom:
            return {}
        out = {}
        for a, v in nom.items():
            bag = v if isinstance(v, dict) else {v: 1.0}
            tot = float(sum(bag.values()))
            if tot <= 0:
                continue
            voc = self.vocab.setdefault(a, {})
            idx, wt = [], []
            for val, w in bag.items():
                if val not in voc:
                    if grow:
                        voc[val] = len(voc)
                    else:
                        idx.append(-1); wt.append(w / tot); continue
                idx.append(voc[val]); wt.append(w / tot)
            out[a] = (np.array(idx, int), np.array(wt))
        return out

    # ---- primitive statistics --------------------------------------------------------------------
    def var_of(self, msq, n):
        return msq / max(n, 1e-12) + self.prior_var

    def _arr(self, nomd, a):
        V = len(self.vocab.get(a, ()))
        arr = nomd.get(a)
        if arr is None:
            return np.zeros(V)
        if arr.shape[0] < V:
            arr = np.concatenate([arr, np.zeros(V - arr.shape[0])])
        return arr

    def add_stats(self, st, x, enc):
        n, m, s, nomd = st
        n2 = n + 1
        d = x - m
        m2 = m + d / n2
        s2 = s + d * (x - m2)
        nd = dict(nomd)
        for a, (idx, wt) in enc.items():
            arr = self._arr(nomd, a).copy()
            ok = idx >= 0
            np.add.at(arr, idx[ok], wt[ok])
            nd[a] = arr
        return (n2, m2, s2, nd)

    def merge_stats(self, s1, s2):
        n1, m1, q1, d1 = s1
        n2, m2, q2, d2 = s2
        n = n1 + n2
        if n == 0:
            return s1
        dl = m2 - m1
        q = q1 + q2 + dl * dl * (n1 * n2 / n)
        m = (n1 * m1 + n2 * m2) / n
        nd = {}
        for a in set(d1) | set(d2):
            nd[a] = self._arr(d1, a) + self._arr(d2, a)
        return (n, m, q, nd)

    def kl(self, st1, st2):
        """KL(child || parent) summed over weighted continuous dims and nominal attributes."""
        n1, m1, q1, d1 = st1
        n2, m2, q2, d2 = st2
        v1, v2 = self.var_of(q1, n1), self.var_of(q2, n2)
        sc = 0.5 * float(np.sum(self.w * (np.log(v2) - np.log(v1) + (v1 + (m1 - m2) ** 2) / v2 - 1.0)))
        for a in d1:
            c1 = self._arr(d1, a)
            N1 = c1.sum()
            if N1 <= 0:
                continue
            c2 = self._arr(d2, a)
            V = c1.shape[0]
            p1 = (c1 + self.alpha) / (N1 + V * self.alpha)
            p2 = (c2 + self.alpha) / (c2.sum() + V * self.alpha)
            sc += float(np.sum(p1 * (np.log(p1) - np.log(p2))))
        return sc

    def log_prob_stats(self, st, x, enc=None):
        n, m, q, d = st
        v = self.var_of(q, n)
        lp = -0.5 * float(np.sum(self.w * (_LOG2PI + np.log(v) + (x - m) ** 2 / v)))
        if enc:
            for a, (idx, wt) in enc.items():
                c = self._arr(d, a)
                V = max(c.shape[0], 1)
                N = c.sum()
                pv = np.where(idx >= 0, (c[np.clip(idx, 0, None)] if c.shape[0] else 0.0) + self.alpha,
                              self.alpha) / (N + V * self.alpha)
                lp += math.log(max(float(np.dot(wt, pv)), 1e-300))
        return lp

    def n_attrs(self, enc=None):
        """Effective attribute count used to normalise a log-likelihood per attribute."""
        return float(self.w.sum()) + (len(enc) if enc else 0)

    # ---- partition utilities (the torch formulas) --------------------------------------------------
    def _pu(self, parent_st, child_sts):
        if not child_sts:
            return 0.0
        N = parent_st[0]
        return sum((c[0] / N) * self.kl(c, parent_st) for c in child_sts) / len(child_sts)

    def _new_leaf_stats(self, x, enc):
        empty = (0.0, np.zeros(self.dim), np.zeros(self.dim), {})
        return self.add_stats(empty, x, enc)

    # ---- incremental fitting ----------------------------------------------------------------------
    def ifit(self, x, nom=None, payload=None):
        """Incorporate one instance; returns the leaf it was stored at."""
        x = np.asarray(x, float)
        enc = self.encode_nom(nom, grow=True)
        self.version += 1
        if self.flat:
            leaf = self._ifit_flat(x, enc)
        else:
            leaf = self._ifit(x, enc)
        if payload is not None:
            leaf.members.append(payload)
        return leaf

    def _inc(self, node, x, enc):
        node.count, node.mean, node.msq, node.nom = self.add_stats(node.stats(), x, enc)

    def _exact(self, node, x, enc):
        if node.count == 0:
            return False
        if np.any(node.msq / node.count > 1e-12) or not np.allclose(node.mean, x):
            return False
        if set(node.nom) != set(enc):
            return False
        for a, (idx, wt) in enc.items():
            c = self._arr(node.nom, a)
            tgt = np.zeros_like(c)
            np.add.at(tgt, idx[idx >= 0], wt[idx >= 0])
            if not np.allclose(c / node.count, tgt):
                return False
        return True

    def _ifit_flat(self, x, enc):
        r = self.root
        if not r.children:
            if r.count == 0:
                self._inc(r, x, enc)
                c = Node(self); c.parent = r; r.children.append(c)
                self._inc(c, x, enc)
                return c
        parent_ins = self.add_stats(r.stats(), x, enc)
        sts = [c.stats() for c in r.children]
        best_pu, best = -np.inf, None
        for i, c in enumerate(r.children):
            alt = list(sts); alt[i] = self.add_stats(sts[i], x, enc)
            pu = self._pu(parent_ins, alt)
            if pu > best_pu:
                best_pu, best = pu, c
        pu_new = self._pu(parent_ins, sts + [self._new_leaf_stats(x, enc)])
        self._inc(r, x, enc)
        if best is not None and best_pu >= pu_new:
            self._inc(best, x, enc)
            return best
        c = Node(self); c.parent = r; r.children.append(c)
        self._inc(c, x, enc)
        return c

    def _ifit(self, x, enc):
        cur = self.root
        while True:
            if not cur.children and (cur.count == 0 or self._exact(cur, x, enc)):
                self.ops["exact"] += 1
                self._inc(cur, x, enc)
                return cur
            if not cur.children:
                # fringe split: cur becomes a child of a new node that also gets a new leaf
                self.ops["fringe"] += 1
                new = Node(self)
                new.count, new.mean, new.msq = cur.count, cur.mean.copy(), cur.msq.copy()
                new.nom = {a: v.copy() for a, v in cur.nom.items()}
                new.parent = cur.parent
                if cur.parent is not None:
                    cur.parent.children.remove(cur)
                    cur.parent.children.append(new)
                else:
                    self.root = new
                cur.parent = new
                new.children.append(cur)
                self._inc(new, x, enc)
                leaf = Node(self); leaf.parent = new; new.children.append(leaf)
                self._inc(leaf, x, enc)
                return leaf
            parent_ins = self.add_stats(cur.stats(), x, enc)
            sts = [c.stats() for c in cur.children]
            ins = [self.add_stats(s, x, enc) for s in sts]
            # two best children by relative gain (torch two_best_children)
            gains = []
            for i, c in enumerate(cur.children):
                g = (ins[i][0] / parent_ins[0]) * self.kl(ins[i], parent_ins) \
                    - (sts[i][0] / parent_ins[0]) * self.kl(sts[i], parent_ins)
                gains.append((g, c.count, self.rng.random(), i))
            gains.sort(reverse=True)
            i1 = gains[0][3]
            i2 = gains[1][3] if len(gains) > 1 else None
            alt = list(sts); alt[i1] = ins[i1]
            ops = [(self._pu(parent_ins, alt), self.rng.random(), "best")]
            ops.append((self._pu(parent_ins, sts + [self._new_leaf_stats(x, enc)]), self.rng.random(), "new"))
            if len(cur.children) > 2 and i2 is not None:
                mst = self.add_stats(self.merge_stats(sts[i1], sts[i2]), x, enc)
                rest = [s for j, s in enumerate(sts) if j not in (i1, i2)]
                ops.append((self._pu(parent_ins, rest + [mst]), self.rng.random(), "merge"))
            b1 = cur.children[i1]
            if b1.children:
                rest = [s for j, s in enumerate(sts) if j != i1] + [g.stats() for g in b1.children]
                ops.append((self._pu(cur.stats(), rest), self.rng.random(), "split"))
            ops.sort(reverse=True)
            op = ops[0][2]
            self.ops[op] += 1
            if op == "best":
                self._inc(cur, x, enc)
                cur = b1
            elif op == "new":
                self._inc(cur, x, enc)
                leaf = Node(self); leaf.parent = cur; cur.children.append(leaf)
                self._inc(leaf, x, enc)
                return leaf
            elif op == "merge":
                self._inc(cur, x, enc)
                b2 = cur.children[i2]
                m = Node(self)
                m.count, m.mean, m.msq, m.nom = self.merge_stats(b1.stats(), b2.stats())
                m.parent = cur
                for b in (b1, b2):
                    b.parent = m
                    m.children.append(b)
                    cur.children.remove(b)
                cur.children.append(m)
                cur = m
            else:  # split
                cur.children.remove(b1)
                for g in b1.children:
                    g.parent = cur
                    cur.children.append(g)
                self.redirect[b1.id] = cur.id

    # ---- categorisation (no modification) ------------------------------------------------------------
    def categorize(self, x, nom=None, max_nodes=10 ** 9):
        """torch _cobweb_categorize (best-first, use_best=True).  Returns (node, log_post) where log_post
        is the path log-posterior log p(node | x) accumulated from the root."""
        x = np.asarray(x, float)
        enc = self.encode_nom(nom, grow=False)
        return self.categorize_enc(x, enc, max_nodes)

    def categorize_enc(self, x, enc, max_nodes=10 ** 9):
        r = self.root
        if r.count == 0:
            return r, 0.0
        q = [(-r.log_prob(x, enc), 0.0, self.rng.random(), r)]
        best, best_score, best_post, visited = r, -np.inf, 0.0, 0
        while q:
            ns, nll, _, cur = heapq.heappop(q)
            score, cur_ll = -ns, -nll
            visited += 1
            if score > best_score:
                best, best_score, best_post = cur, score, cur_ll
            if best_score > cur_ll:
                break
            if visited >= max_nodes:
                break
            if cur.children:
                lps = np.array([c.log_prob(x, enc) for c in cur.children])
                un = lps + np.log([c.count for c in cur.children]) - math.log(cur.count)
                lz = float(np.logaddexp.reduce(un))
                for i, c in enumerate(cur.children):
                    cll = un[i] - lz + cur_ll
                    heapq.heappush(q, (-(cll + lps[i]), -cll, self.rng.random(), c))
        return best, best_post

    # ---- queries ----------------------------------------------------------------------------------
    def nodes(self):
        return list(self.root.subtree())

    def n_nodes(self):
        return sum(1 for _ in self.root.subtree())

    def resolve(self, nid):
        while nid in self.redirect:
            nid = self.redirect[nid]
        return nid

    def node_index(self):
        return {n.id: n for n in self.root.subtree()}
