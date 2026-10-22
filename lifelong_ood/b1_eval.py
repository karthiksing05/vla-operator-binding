"""B1 closed-loop evaluation on CompoSuite (CPU, MuJoCo).

Success follows the dataset's definition (Dryad README; offline-compositional-rl-datasets/evaluate.py): an episode of
at most 500 control steps succeeds if any state had the task's success flag (reward 1).  We read the env's own
`_check_success()` and stop the episode at the first success, which gives the same verdict.  Episode e of a cell is
reset after np.random.seed(seed * 100003 + e), so every arm meets the same initial states.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from . import b1_data
from .b1_policy import EXEC

EP_LEN = 500


def make_env(cell):
    import composuite
    r, o, b, j = cell
    return composuite.make(r, o, b, j, use_task_id_obs=True, has_renderer=False, has_offscreen_renderer=False,
                           reward_shaping=True, use_camera_obs=False, env_horizon=EP_LEN)


def _obs(o):
    return o[0] if isinstance(o, tuple) else o


@torch.no_grad()
def rollout(model, norm, cell, cond, n_episodes, seed=0, env=None, flow_steps=10, max_steps=EP_LEN):
    """cond: dict(task=LongTensor[1]) | dict(task=FloatTensor[1,16]) | dict(z=FloatTensor[1,emb]).
    -> dict(success=[bool]*n, steps=[int]*n, seconds=float)."""
    env = env or make_env(cell)
    model.eval()
    t0 = time.time()
    succ, lens = [], []
    gen = torch.Generator().manual_seed(seed)
    for e in range(n_episodes):
        np.random.seed(seed * 100003 + e)
        obs = _obs(env.reset())
        ok, t, plan = False, 0, []
        while t < max_steps:
            if not plan:
                s = b1_data.strip_task_id(np.asarray(obs, np.float32)[None], cell, check=(t == 0))
                st = torch.tensor(norm.state(s), dtype=torch.float32)
                ch = model.sample(st, steps=flow_steps, gen=gen, **cond)[0].numpy()
                plan = list(norm.unact(ch)[:EXEC])
            a = np.clip(plan.pop(0), env.action_space.low, env.action_space.high)
            obs = _obs(env.step(a)[0])
            t += 1
            if env.env._check_success():
                ok = True
                break
        succ.append(bool(ok)), lens.append(t)
    return dict(success=succ, steps=lens, seconds=time.time() - t0)
