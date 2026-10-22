"""B1 data: converted CompoSuite expert episodes -> normalised (state, action-chunk) samples.

Each cell's file (experiments/b1_fetch.py) holds obs (E, 500, 93), act (E, 500, 8), rew, success.  The 93-dim obs
is CompoSuite's GymWrapper order with use_task_id_obs=True: object-state, obstacle-state, goal-state, then the
16-dim task indicator (object_id, robot_id, obstacle_id, subtask_id), then robot0_proprio-state.  `TASK_ID_SLICE`
is read from results/b1/env_smoke_*.json (the env's own observation_positions); `load_cell` ASSERTS that those
16 dims equal the cell's indicator before dropping them, so a layout mistake cannot leak labels into the state.
"""
from __future__ import annotations

import glob
import json

import numpy as np

from . import b1
from .b1_policy import H

ID_KEYS = ("object_id", "robot_id", "obstacle_id", "subtask_id")


def task_id_slice():
    fs = sorted(glob.glob(str(b1.RESULTS / "env_smoke_*.json")))
    lay = json.loads(open(fs[0], encoding="utf-8").read())
    lays = [t["layout"] for t in lay["tasks"].values()]
    spans = [[l[k] for k in ID_KEYS] for l in lays]
    assert all(s == spans[0] for s in spans), "task-id layout differs between tasks"
    lo, hi = spans[0][0][0], spans[0][-1][1]
    assert hi - lo == 16 and all(spans[0][i][1] == spans[0][i + 1][0] for i in range(3))
    return lo, hi


def indicator_in_obs_order(cell):
    """The env's indicator, as composuite/env/compositional_env.py builds it (read 2026-10-21): GymWrapper keys in the
    order object_id, robot_id, obstacle_id, subtask_id; within each, the env's own map order (obj_class_map,
    robot_list, obstacle_class_map with None first, subtask_id 0..3)."""
    order = (("Box", "Dumbbell", "Plate", "Hollowbox"), ("IIWA", "Jaco", "Panda", "Kinova3"),
             ("None", "ObjectWall", "ObjectDoor", "GoalWall"), ("PickPlace", "Push", "Shelf", "Trashcan"))
    r, o, b, j = cell
    v = []
    for vals, x in zip(order, (o, r, b, j)):
        v += [float(x == y) for y in vals]
    return np.array(v, np.float32)


def strip_task_id(obs, cell=None, check=True):
    lo, hi = task_id_slice()
    if check and cell is not None:
        ind = obs[..., lo:hi].reshape(-1, 16)
        want = indicator_in_obs_order(cell)
        assert np.allclose(ind, ind[:1]), "task indicator varies inside an episode"
        g = ind[0].reshape(4, 4)
        assert np.allclose(g.sum(1), 1) and set(np.unique(g)) <= {0.0, 1.0}, f"not a 4x one-hot: {g}"
        if not np.allclose(ind[0], want):
            raise AssertionError(f"indicator {ind[0]} != expected {want} for {cell}: check the layout")
    return np.concatenate([obs[..., :lo], obs[..., hi:]], -1)


def load_cell(cell, n_eps=None, prereg=None):
    d = b1.task_dir(cell, prereg=prereg)
    z = np.load(d / "episodes.npz")
    obs, act, suc = z["obs"], z["act"], z["success"]
    if n_eps is not None:
        obs, act, suc = obs[:n_eps], act[:n_eps], suc[:n_eps]
    return strip_task_id(obs, cell), act, suc


def chunks(obs, act):
    """(E, T, S), (E, T, A) -> states (E*T, S), chunks (E*T, H, A); the chunk is padded with the last action."""
    E, T, _ = act.shape
    idx = np.minimum(np.arange(T)[:, None] + np.arange(H)[None], T - 1)
    return obs.reshape(E * T, -1), act[:, idx].reshape(E * T, H, -1)


class Norm:
    def __init__(self, s_mean, s_std, a_min, a_max):
        self.s_mean, self.s_std, self.a_min, self.a_max = (np.asarray(x, np.float32) for x in (s_mean, s_std, a_min, a_max))

    @classmethod
    def fit(cls, obs, act):
        S, A = obs.reshape(-1, obs.shape[-1]), act.reshape(-1, act.shape[-1])
        return cls(S.mean(0), S.std(0) + 1e-3, A.min(0), A.max(0))

    def state(self, s):
        return (s - self.s_mean) / self.s_std

    def act(self, a):
        return 2 * (a - self.a_min) / np.maximum(self.a_max - self.a_min, 1e-6) - 1

    def unact(self, a):
        return (a + 1) / 2 * np.maximum(self.a_max - self.a_min, 1e-6) + self.a_min

    def to_dict(self):
        return {k: getattr(self, k).tolist() for k in ("s_mean", "s_std", "a_min", "a_max")}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
