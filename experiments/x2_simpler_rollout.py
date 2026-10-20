"""X2: closed-loop SimplerEnv rollouts of a PolicyBackbone (open-pi-zero pi0 through lifelong_ood/policy/backbone_opz.py), stock protocol,
with an optional GRASP-SWITCH prompt pair (the supervised scheme's grasp switch; LIBERO's pfield arm / program.ProgHooks rule):
prompt A until the policy's own gripper command has been 'close' for 3 consecutive queries, prompt B from then on.

Stock protocol (open-pi-zero's EvalAgent / SimplerEnv's evaluators): reset with obj_init_options.episode_id (the 24 layouts), query
the policy on the 3rd-view image + proprio, execute all H = 4 actions of the chunk, stop only at the env's time limit (60 steps; 120
for the sink scene), success = the env's success flag at that last step.  The policy's flow noise is seeded per query
(seed, episode, query), so a (task, episode, seed) rollout is reproducible.

    python -I experiments/x2_simpler_rollout.py --task X2G1GreenBlockOnPlate-v0 --episodes 0-23 --seeds 0:1:2 --out OUT.json
        [--prompt "..."] [--prompt_after "..."] [--ckpt bridge_beta] [--video 2]
--task: a stock SimplerEnv name (widowx_carrot_on_plate, ...) or a graded X2 id (experiments/x2_simpler_graded.py).
--prompt: the instruction (default: the env's own); --prompt_after: the grasp-switch prompt B (omit for the policy alone).
Run in /coc/flash10/ksingara3/robot/simpler/.venv_pi0 on a GPU node.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

RUN = 3          # consecutive 'close' queries before the grasp counts (lifelong_ood/policy/program.py RUN)


def parse_eps(s):
    out = []
    for part in s.split(":"):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def make_env(task):
    import simpler_env
    import x2_simpler_graded as G          # registers the X2 graded envs
    if task in simpler_env.ENVIRONMENTS:
        return simpler_env.make(task)
    return G.make(task)


class GraspSwitch:
    """Prompt A until the chunk's gripper command is 'close' (< 0 in SimplerEnv's convention) for RUN consecutive queries, then B."""
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.run, self.switched_at, self.q = 0, None, 0

    def prompt(self):
        return self.b if (self.b is not None and self.switched_at is not None) else self.a

    def observe(self, env_actions):
        closed = bool(env_actions[min(4, len(env_actions) - 1), 6] < 0)
        self.run = self.run + 1 if closed else 0
        if self.run >= RUN and self.switched_at is None:
            self.switched_at = self.q + 1          # the first query that reads prompt B
        self.q += 1


def rollout(bk, env, episode, seed, prompt_a, prompt_b, video=False):
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    obs, info = env.reset(options={"obj_init_options": {"episode_id": int(episode)}})
    instr = env.unwrapped.get_language_instruction()
    sw = GraspSwitch(prompt_a or instr, prompt_b)
    frames, prompts, steps, success, truncated, t_policy = [], [], 0, False, False, 0.0
    while True:
        img = get_image_from_maniskill2_obs_dict(env, obs)
        if video:
            frames.append(img)
        p = sw.prompt()
        prompts.append(p)
        t0 = time.time()
        o = bk.observation(bk.raw_observation(img, obs["agent"]["eef_pos"], p))
        pre = bk.prefix(o)
        rows = bk.sample(1, pre, seed=int(seed) * 1_000_003 + int(episode) * 1_009 + sw.q)
        acts = bk.unnormalize(rows)[0]
        t_policy += time.time() - t0
        sw.observe(acts)
        for a in acts:                       # act_steps = H = 4
            obs, reward, success, truncated, info = env.step(a)
            steps += 1
            if truncated:
                break
        if truncated:
            break
    st = info.get("episode_stats", {})
    return dict(episode=int(episode), seed=int(seed), success=bool(success), steps=steps, queries=sw.q, switched_at=sw.switched_at,
                prompt_a=sw.a, prompt_b=prompt_b, env_instruction=instr,
                stats={k: bool(v) for k, v in st.items()}, policy_sec=round(t_policy, 2)), frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--episodes", default="0-23")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--prompt_after", default=None)
    ap.add_argument("--ckpt", default="bridge_beta")
    ap.add_argument("--out", required=True)
    ap.add_argument("--video", type=int, default=0, help="save the first N episodes as mp4")
    a = ap.parse_args()
    import torch
    from lifelong_ood.policy.backbone_opz import OPZBackbone
    bk = OPZBackbone(a.ckpt)
    print(bk.describe(), flush=True)
    env = make_env(a.task)
    print(f"[rollout] {a.task} max_episode_steps {env.spec.max_episode_steps} prompt_a={a.prompt!r} prompt_b={a.prompt_after!r}",
          flush=True)
    rows, t0 = [], time.time()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    k = 0
    for seed in [int(s) for s in a.seeds.split(":")]:
        for ep in parse_eps(a.episodes):
            te = time.time()
            r, frames = rollout(bk, env, ep, seed, a.prompt, a.prompt_after, video=k < a.video)
            r["sec"] = round(time.time() - te, 2)
            rows.append(r)
            if frames:
                import imageio
                imageio.mimwrite(a.out.replace(".json", f"_ep{ep}_s{seed}_{int(r['success'])}.mp4"), frames, fps=5)
            print("[rollout]", json.dumps(r), flush=True)
            k += 1
    res = dict(task=a.task, ckpt=a.ckpt, backbone=bk.describe(), prompt=a.prompt, prompt_after=a.prompt_after,
               max_episode_steps=env.spec.max_episode_steps, n=len(rows), successes=sum(r["success"] for r in rows),
               success_rate=sum(r["success"] for r in rows) / max(1, len(rows)), seconds=round(time.time() - t0, 1),
               command=" ".join(sys.argv), rows=rows, torch=torch.__version__)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[rollout] {a.task}: {res['successes']}/{res['n']} = {100 * res['success_rate']:.1f}% in {res['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
