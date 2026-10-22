"""B1 stage (a): CompoSuite stepping, observation layout and rendering on a compute node.

    $VENV -I experiments/b1_env_smoke.py [--render]
Writes results/b1/env_smoke.json: per task the obs layout (key -> index range), action bounds, step rate and a
render check.  The obs layout is what the converter uses to drop the 16-dim task indicator from the state (C5).
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = [("IIWA", "Box", "None", "PickPlace"), ("Jaco", "Dumbbell", "GoalWall", "Push"),
         ("Kinova3", "Plate", "ObjectDoor", "Shelf"), ("Panda", "Hollowbox", "ObjectWall", "Trashcan")]


def main():
    import composuite
    import robosuite
    import mujoco
    out = dict(robosuite=robosuite.__version__, mujoco=mujoco.__version__, host=os.uname().nodename,
               mujoco_gl=os.environ.get("MUJOCO_GL"), tasks={})
    for t in TASKS:
        env = composuite.make(*t, use_task_id_obs=True, has_renderer=False, has_offscreen_renderer=False,
                              reward_shaping=True, use_camera_obs=False, env_horizon=500)
        obs = env.reset()
        obs = obs[0] if isinstance(obs, tuple) else obs
        lo, hi = env.action_space.low, env.action_space.high
        rng = np.random.default_rng(0)
        t0 = time.time()
        n, rmax = 0, 0.0
        for _ in range(200):
            r = env.step(rng.uniform(lo, hi))
            n += 1
            rmax = max(rmax, float(r[1]))
        dt = time.time() - t0
        out["tasks"]["_".join(t)] = dict(
            obs_dim=int(np.asarray(obs).size), act_dim=int(lo.size), act_low=lo.tolist(), act_high=hi.tolist(),
            layout={k: [int(v[0]), int(v[-1]) + 1] for k, v in env.observation_positions.items()},
            keys=list(env.keys), steps_per_s=n / dt, max_reward_random=rmax,
            control_freq=getattr(env.env, "control_freq", None), horizon=env.env.horizon,
            step_return_len=len(r))
        print(t, out["tasks"]["_".join(t)]["obs_dim"], f"{n / dt:.0f} steps/s", out["tasks"]["_".join(t)]["layout"])
        env.close()
    if "--render" in sys.argv:
        try:
            env = composuite.make(*TASKS[0], has_offscreen_renderer=True, use_camera_obs=True,
                                  camera_names="agentview", camera_heights=128, camera_widths=128)
            o = env.env.reset()
            img = o["agentview_image"]
            out["render"] = dict(ok=True, shape=list(img.shape), mean=float(img.mean()))
        except Exception as e:  # noqa: BLE001
            out["render"] = dict(ok=False, error=repr(e)[:400])
        print("render", out["render"])
    os.makedirs(os.path.join(ROOT, "results", "b1"), exist_ok=True)
    p = os.path.join(ROOT, "results", "b1", f"env_smoke_{os.environ.get('MUJOCO_GL', 'none')}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(p)


if __name__ == "__main__":
    main()
