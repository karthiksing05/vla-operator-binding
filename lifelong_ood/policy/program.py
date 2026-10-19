"""POINTER PROGRAMS (plan Z, track U; docs/PLAN_Z.md): a task as a sequence of held fields, with no instruction grammar.

A demonstration is cut at its own contact events (the gripper closing for a sustained run, then opening: segment.gripper_truth),
and each segment is labelled by the frozen policy's likelihood: the training instruction with the lowest mean flow loss on the
segment's frames (experiments/u1_score.py).  The program is the list of those instructions.  Online, the policy's own commanded
gripper advances the program at the same events: closed for 3 queries -> next segment; open after a closed run -> next segment.
A pick-and-place demo gives three segments (approach, carry, after the release); the grasp switch of plan G is the special case
whose segments are named by the grammar.  Nothing is fitted; a program is a few task indices.
"""
from __future__ import annotations

import logging

import numpy as np

from . import segment as S

RUN = 3                                                                  # queries closed before a grasp counts (PrimHooks' rule)


def segments(actions):
    """-> [(lo, hi)] frame ranges of a demo cut at its sustained grasps and releases."""
    g, r = S.gripper_truth(actions)
    cuts = sorted(set(g) | set(r))
    T = len(actions)
    b = [0] + [c for c in cuts if 0 < c < T] + [T]
    return [(lo, hi) for lo, hi in zip(b[:-1], b[1:]) if hi > lo]


def label(L, frames, segs, labels):
    """L [F, P] per-frame log flow loss over prompts, frames [F] demo frame of each row, segs from `segments`, labels = candidate
    prompt columns.  -> per segment the label with the lowest mean (a segment with no scored frame takes its neighbour's)."""
    out = []
    for lo, hi in segs:
        m = (frames >= lo) & (frames < hi)
        out.append(int(labels[int(np.argmin(L[m][:, labels].mean(0)))]) if m.any() else None)
    for i in range(len(out)):
        if out[i] is None:
            out[i] = next((x for x in out[i + 1:] + out[:i][::-1] if x is not None), int(labels[0]))
    return out


class ProgHooks:
    """Serve hooks: the prompt is the current segment's instruction; the policy's own gripper advances the segment."""

    def __init__(self, prompts):
        self.prompts = list(prompts)
        self.reset(None)

    def reset(self, meta):
        self.seg, self.run, self.closed_seg, self.log = 0, 0, False, []

    def rewrite(self, obs, meta):
        obs = dict(obs)
        obs["prompt"] = self.prompts[min(self.seg, len(self.prompts) - 1)]
        return obs

    def observe(self, actions):
        a = np.asarray(actions)
        closed = bool(a[min(4, len(a) - 1), 6] > 0)
        if closed:
            self.run += 1
            if self.run >= RUN and not self.closed_seg:
                self.closed_seg = True
                self._next("grasp")
        else:
            if self.closed_seg:
                self.closed_seg = False
                self._next("release")
            self.run = 0

    def _next(self, ev):
        if self.seg < len(self.prompts) - 1:
            self.seg += 1
            self.log.append((ev, self.seg))
            logging.info(f"prog: {ev} -> segment {self.seg}: '{self.prompts[self.seg]}'")

    def __call__(self, obs, pre, meta=None):
        return None, None
