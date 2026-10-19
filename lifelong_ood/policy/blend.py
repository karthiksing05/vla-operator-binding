"""SOFT BLEND OF HELD FIELDS, served (plan Z, track I3; docs/INTERPRETATION.md section 4).

The blend experiments/zi_blend.py fitted to a task's demonstration: per contact segment (approach / carry / after the release) a
convex mix of the frozen policy's fields under a few prompts, with one weight per dim group (position, rotation).  The gripper is
categorical (the offline read: no mix beats the best single expert on it), so it follows the segment's single best expert.  The
segment advances at the policy's own gripper events, as program.ProgHooks does.  `ablate`: an expert id dropped from every mix
(the weights renormalised; the gripper's expert re-chosen among the rest), the behavioural counterpart of its Shapley donation.
Sampling is poe.sample_poe over the policy rows only: weights one-hot on one row give that row's policy bit for bit.
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np

from . import poe
from .program import RUN

GROUP = {"position": (0, 1, 2), "rotation": (3, 4, 5)}
SEGS = ("approach", "carry", "after")


def load(name, root):
    rows = json.load(open(os.path.join(root, "results", "robot", "zi_blend.json")))["rows"]
    return next(r for r in rows if r["name"] == name)


class BlendHooks:
    def __init__(self, bk, row, ablate=None, H=10, A=7):
        self.bk, self.H, self.A = bk, H, A
        ids = [int(e) for e in row["expert_ids"]]
        self.prompts = {e: p for e, p in zip(ids, row["experts"])}
        self.ablate = None if ablate is None else int(ablate)
        self.codes = []
        segs = row["segments"]
        order = [s for s in SEGS if s in segs] or ["whole"]
        for s in order:
            seg = segs[s]
            w = {g: {int(k): float(v) for k, v in seg["beta"][g].items()} for g in GROUP}
            grip_loss = {int(k): float(v) for k, v in seg["held"]["gripper"]["single"].items()}
            self.codes.append(dict(seg=s, w=w, grip=grip_loss))
        self.reset(None)
        logging.info(f"blend: {row['name']} '{row['text']}', segments {order}, experts {ids}, ablate {self.ablate}")

    def reset(self, meta):
        self.k, self.run, self.closed, self.nq = 0, 0, False, 0

    def observe(self, actions):
        a = np.asarray(actions)
        closed = bool(a[min(4, len(a) - 1), 6] > 0)
        if closed:
            self.run += 1
            if self.run >= RUN and not self.closed:
                self.closed = True
                self.k = min(self.k + 1, len(self.codes) - 1)
        else:
            if self.closed:
                self.closed = False
                self.k = min(self.k + 1, len(self.codes) - 1)
            self.run = 0

    def _beta(self, code):
        experts = sorted({e for g in GROUP for e in code["w"][g]} | set(code["grip"]))
        experts = [e for e in experts if e != self.ablate]
        B = np.zeros((len(experts), self.H, self.A), np.float32)
        for g, dims in GROUP.items():
            w = np.array([code["w"][g].get(e, 0.0) for e in experts], np.float32)
            if w.sum() <= 0:                                            # every weighted expert ablated: the best remaining one
                w = np.array([-code["grip"].get(e, np.inf) for e in experts], np.float32)
                w = (w == w.max()).astype(np.float32)
            w = w / w.sum()
            B[:, :, list(dims)] = w[:, None, None]
        gi = min(range(len(experts)), key=lambda i: code["grip"].get(experts[i], np.inf))
        B[gi, :, 6] = 1.0
        return experts, B.reshape(len(experts), -1)

    def compose_poe(self, obs, meta):
        code = self.codes[self.k]
        experts, B = self._beta(code)
        rows = [poe.PolicyRow(self.prompts[e], name=str(e)) for e in experts]
        if self.nq < 2 or self.nq % 50 == 0:
            logging.info(f"blend: query {self.nq}, segment {code['seg']}, experts {experts}, "
                         f"position weights {np.round(B[:, 0], 2).tolist()}, gripper from {experts[int(B[:, 6].argmax())]}")
        self.nq += 1
        return poe.policy_prefixes(self.bk, obs, rows), rows, [], B

    def __call__(self, obs, pre, meta=None):
        return None, None
