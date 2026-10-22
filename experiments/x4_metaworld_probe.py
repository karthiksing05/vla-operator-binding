"""X4 preflight on a GPU node: Meta-World v3 renders off-screen through MuJoCo EGL, and the installed package matches the
split's task names and ML45 partition.  Saves one frame per probed task (flipped as RLinf / LeRobot flip camera 2).

MUJOCO_GL=egl python -I experiments/x4_metaworld_probe.py --out results/x4_metaworld/probe
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
REPO = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results/x4_metaworld/probe"))
    ap.add_argument("--tasks", default="reach-v3:door-lock-v3:bin-picking-v3")
    a = ap.parse_args()
    import gymnasium as gym
    import imageio
    import metaworld
    import mujoco
    import numpy as np
    sys.path.insert(0, str(REPO / "experiments"))
    from x4_metaworld_split import check_metaworld
    metaworld.register_mw_envs()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = dict(mujoco=mujoco.__version__, gymnasium=gym.__version__, MUJOCO_GL=os.environ.get("MUJOCO_GL"),
               check=check_metaworld(), frames={})
    for name in a.tasks.split(":"):
        t0 = time.time()
        env = gym.make("Meta-World/MT1", env_name=name, render_mode="rgb_array", camera_id=2, disable_env_checker=True)
        env.env.env.env.env.env.env.model.cam_pos[2] = [0.75, 0.075, 0.7]
        obs, info = env.reset(seed=0)
        img = env.render()[::-1, ::-1]
        imageio.imwrite(out / f"{name}.png", np.ascontiguousarray(img))
        rep["frames"][name] = dict(shape=list(img.shape), mean=float(img.mean()), std=float(img.std()),
                                   obs_dim=int(np.asarray(obs).shape[0]), max_path_length=int(env.unwrapped.max_path_length),
                                   seconds=time.time() - t0)
        env.close()
    (out / "probe.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=1))
    ok = rep["check"].get("ok") and all(f["std"] > 1 for f in rep["frames"].values())
    print("[x4probe]", "OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
