"""X4 parity: RLinf's pi0 / pi0.5 Meta-World SFT checkpoints under the authors' standalone evaluation.

The episode loop is RLinf's toolkits/standalone_eval_scripts/openpi/metaworld_eval.py main() (commit in docs/METAWORLD.md),
copied step for step: gym.make("Meta-World/MT1", env_name, render_mode="rgb_array", camera_id=2), camera 2 moved to
(0.75, 0.075, 0.7), env.reset(), 15 zero-action settle steps, then up to max_steps steps; each step renders, flips the image
(image[::-1, ::-1]), reads state = obs[:4], and when the action queue is empty calls policy.infer({observation/image,
observation/state, prompt}) and queues the first action_chunk actions.  A trial succeeds if info["success"] was ever 1
(their loop does not break, and `success` is only overwritten when success / terminated / truncated fires).  The policy is
their setup_policy(args) (RLinf's TrainConfig pi0_metaworld / pi05_metaworld, norm stats from the checkpoint dir).

Deviations (bookkeeping only): one JSON per task (resumable: tasks with a JSON are skipped), tasks run as a subset given
by --tasks, no videos, and the global numpy / torch / random seeds are set per task to seed + task index (their script
sets none), so a shard's result does not depend on which other tasks share its process.

cwd must be the RLinf root (their prompt JSON path is relative).
python experiments/x4_metaworld_pi0_eval.py --pretrained_path <snapshot> --config_name pi0_metaworld --tasks a-v3:b-v3 --out <dir>
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import random
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rlinf_root", default=os.getcwd())
    ap.add_argument("--config_name", default="pi0_metaworld")
    ap.add_argument("--pretrained_path", required=True)
    ap.add_argument("--tasks", default="all", help="colon-separated v3 env names, or 'all'")
    ap.add_argument("--num_trials_per_task", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=160)
    ap.add_argument("--action_chunk", type=int, default=5)
    ap.add_argument("--num_steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.rlinf_root)
    import numpy as np
    import torch
    # their module: registers the Meta-World envs, reads the prompts / difficulty groups, provides setup_policy
    from toolkits.standalone_eval_scripts.openpi import metaworld_eval as ME
    import gymnasium as gym

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env_list = ME.env_list if a.tasks == "all" else a.tasks.split(":")
    todo = [e for e in env_list if not (out / f"{e}.json").exists()]
    print(f"[x4eval] {len(env_list)} tasks, {len(todo)} to run, out={out}", flush=True)
    if not todo:
        return
    t0 = time.time()
    policy = ME.setup_policy(argparse.Namespace(config_name=a.config_name, pretrained_path=a.pretrained_path,
                                                num_steps=a.num_steps))
    print(f"[x4eval] policy ready in {time.time() - t0:.0f}s", flush=True)
    for env_name in todo:
        idx = ME.env_list.index(env_name)
        random.seed(a.seed + idx); np.random.seed(a.seed + idx); torch.manual_seed(a.seed + idx)
        prompt = ME.task_description_dict[env_name]
        env = gym.make("Meta-World/MT1", env_name=env_name, render_mode="rgb_array", camera_id=2,
                       disable_env_checker=True)
        env.env.env.env.env.env.env.model.cam_pos[2] = [0.75, 0.075, 0.7]
        trials, t1 = [], time.time()
        for trial_id in range(a.num_trials_per_task):
            observation, info = env.reset()
            dummy_action = [0.0] * 4
            for _ in range(15):
                observation, _, _, _, _ = env.step(dummy_action)
            success, first = 0, None
            action_plan = collections.deque()
            n_infer = 0
            for step in range(a.max_steps):
                image = env.render()
                image = image[::-1, ::-1]
                state = observation[:4]
                batch = {"observation/image": image, "observation/state": state, "prompt": prompt}
                if not action_plan:
                    chunk = policy.infer(batch)["actions"]
                    n_infer += 1
                    assert len(chunk) >= a.action_chunk
                    action_plan.extend(chunk[: a.action_chunk])
                action = action_plan.popleft()
                observation, reward, terminated, truncated, info = env.step(action)
                if info.get("success", 0) or terminated or truncated:
                    success = int(info.get("success", 0))
                    if success and first is None:
                        first = step
            trials.append(dict(trial=trial_id, success=success, first_success_step=first, n_infer=n_infer))
        env.close()
        k = sum(t["success"] for t in trials)
        row = dict(task=env_name, prompt=prompt, successes=k, trials=len(trials), rate=k / len(trials),
                   difficulty=next((d for d, ts in ME.difficulty_to_tasks.items() if env_name in ts), None),
                   per_trial=trials, seconds=time.time() - t1,
                   config=dict(config_name=a.config_name, pretrained_path=a.pretrained_path, max_steps=a.max_steps,
                               action_chunk=a.action_chunk, num_steps=a.num_steps, seed=a.seed + idx))
        (out / f"{env_name}.json").write_text(json.dumps(row, indent=1) + "\n", encoding="utf-8")
        print(f"[x4eval] {env_name}: {k}/{len(trials)}  ({time.time() - t1:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
