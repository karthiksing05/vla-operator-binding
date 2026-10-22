"""X4 wrapper parity: Pi0MetaWorldBackbone (lifelong_ood/policy/backbone_pi0_metaworld.py) against the authors' policy
(RLinf toolkits/standalone_eval_scripts/openpi setup_policy, the object their Meta-World evaluation calls).

On real Meta-World frames (each task: reset, the 15 settle steps, then the observation at the start and after each of
--chunks executed chunks of the authors' actions) it checks:
  1. weights: every tensor of the two models is identical (and the wrapper's strict=False load misses nothing);
  2. inputs: the transformed inputs (image tensors, tokens, masks, normalised state) are identical;
  3. actions, same noise: the wrapper's observation -> prefix -> sample -> unnormalize equals the authors'
     Policy.infer(noise=...) run with the eager sampler, bit for bit; and its distance to the authors' compiled sampler
     (openpi wraps sample_actions in torch.compile max-autotune; their evaluation runs that) is reported.

cwd = RLinf root.  PYTHONPATH=<RLinf>:<repo> python experiments/x4_metaworld_parity.py --pretrained_path <snapshot> [--record]
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
    ap.add_argument("--rlinf_root", default=os.getcwd())
    ap.add_argument("--pretrained_path", required=True)
    ap.add_argument("--config_name", default="pi0_metaworld")
    ap.add_argument("--tasks", default="reach-v3:door-open-v3:pick-place-v3:button-press-topdown-v3")
    ap.add_argument("--chunks", type=int, default=2)
    ap.add_argument("--num_steps", type=int, default=5)
    ap.add_argument("--compiled", type=int, default=1, help="also compare with the authors' compiled sampler")
    ap.add_argument("--out", default=str(REPO / "results/x4_metaworld/parity.json"))
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    sys.path[:0] = [a.rlinf_root, str(REPO)]
    import numpy as np
    import torch
    import gymnasium as gym
    from toolkits.standalone_eval_scripts.openpi import metaworld_eval as ME
    from lifelong_ood.policy.backbone_pi0_metaworld import Pi0MetaWorldBackbone

    t0 = time.time()
    pa = ME.setup_policy(argparse.Namespace(config_name=a.config_name, pretrained_path=a.pretrained_path,
                                            num_steps=a.num_steps))
    B = Pi0MetaWorldBackbone(a.pretrained_path, pi05="pi05" in a.config_name, num_steps=a.num_steps)
    print(B.describe(), f"(both loaded in {time.time() - t0:.0f}s)", flush=True)
    ma, mb = pa._model, B.model
    sa, sb = ma.state_dict(), mb.state_dict()
    w_diff = [k for k in sa if k not in sb or sa[k].dtype != sb[k].dtype or not torch.equal(sa[k], sb[k])]
    weights = dict(n_tensors=len(sa), same_keys=sorted(sa) == sorted(sb), differing=w_diff[:20], n_differing=len(w_diff),
                   missing_keys=B.missing_keys, unexpected_keys=B.unexpected_keys)
    print("weights:", {k: v for k, v in weights.items() if k != "differing"}, flush=True)

    compiled_fn = pa._sample_actions                                   # the authors' (compiled) sampler
    eager_fn = lambda dev, obs, **kw: type(ma).sample_actions(ma, dev, obs, **kw)   # noqa: E731

    def tree_equal(x, y):
        if isinstance(x, dict):
            return set(x) == set(y) and all(tree_equal(x[k], y[k]) for k in x)
        xa, ya = np.asarray(x), np.asarray(y)
        return xa.shape == ya.shape and xa.dtype == ya.dtype and np.array_equal(xa, ya)

    rng = np.random.default_rng(0)
    frames = []
    for name in a.tasks.split(":"):
        env = gym.make("Meta-World/MT1", env_name=name, render_mode="rgb_array", camera_id=2, disable_env_checker=True)
        env.env.env.env.env.env.env.model.cam_pos[2] = [0.75, 0.075, 0.7]
        obs, _ = env.reset(seed=0)
        for _ in range(15):
            obs, _, _, _, _ = env.step([0.0] * 4)
        prompt = ME.task_description_dict[name]
        for c in range(a.chunks + 1):
            img = env.render()[::-1, ::-1]
            batch = {"observation/image": img, "observation/state": obs[:4], "prompt": prompt}
            ia = pa._input_transform(dict(batch))
            ib = B.policy._input_transform(dict(batch))
            same_inputs = tree_equal(ia, ib)
            noise = rng.standard_normal((B.H, B.A)).astype(np.float32)
            pa._sample_actions = eager_fn
            act_eager = pa.infer(dict(batch), noise=noise.copy())["actions"]
            ob = B.observation(dict(batch))
            pre = B.prefix(ob)
            rows = B.sample(1, pre, noise=noise.reshape(1, -1))
            act_ours = B.unnormalize(rows, state=ob.state[0].cpu().numpy())[0]
            row = dict(task=name, chunk=c, same_inputs=bool(same_inputs),
                       bit_identical_eager=bool(np.array_equal(act_ours, act_eager)),
                       max_abs_eager=float(np.abs(act_ours - act_eager).max()), shape=list(act_ours.shape))
            if a.compiled:
                pa._sample_actions = compiled_fn
                act_comp = pa.infer(dict(batch), noise=noise.copy())["actions"]
                row["max_abs_compiled"] = float(np.abs(act_ours - act_comp).max())
                row["compiled_vs_eager"] = float(np.abs(act_comp - act_eager).max())
            frames.append(row)
            print(row, flush=True)
            for k in range(5):                                         # execute the authors' chunk (as their loop does)
                obs, _, _, _, _ = env.step(act_eager[k])
        env.close()
    res = dict(weights=weights, frames=frames, n_frames=len(frames),
               all_inputs_identical=all(f["same_inputs"] for f in frames),
               all_bit_identical_eager=all(f["bit_identical_eager"] for f in frames),
               max_abs_compiled=max((f.get("max_abs_compiled", 0.0) for f in frames), default=None),
               seconds=time.time() - t0)
    p = pathlib.Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "frames"}, indent=1))
    if a.record:
        from lifelong_ood import ledger
        pth = ledger.record("x4_metaworld_parity", bench="metaworld_mt50", tag="pi0_metaworld_backbone", seed=0,
                            metrics={k: v for k, v in res.items()}, keys={"check": "wrapper_vs_authors"},
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(pretrained_path=a.pretrained_path, config_name=a.config_name, tasks=a.tasks,
                                        chunks=a.chunks, num_steps=a.num_steps),
                            script="experiments/x4_metaworld_parity.py", command=" ".join(sys.argv))
        print("ledger:", pth)


if __name__ == "__main__":
    main()
