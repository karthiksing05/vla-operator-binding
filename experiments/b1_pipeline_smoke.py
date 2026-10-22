"""B1 pipeline smoke test (no expert data needed): random-action episodes from the real env in the dataset's format
-> converter-format npz (data/composuite/smoke/) -> the training script on CPU for a few hundred steps -> closed-loop
rollouts with the policy in the loop, timed.  Checks the task-indicator layout assertion on real observations and
measures the rollout cost per episode that budgets stage (e).  Nothing here is a result.

    sbatch ... --export=ALL,SCRIPT=experiments/b1_pipeline_smoke.py slurm/jobs/b1_cpu.sbatch
"""
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lifelong_ood import b1, b1_data, b1_eval  # noqa: E402
from lifelong_ood.b1_policy import FlowPolicy  # noqa: E402

CELLS = [("IIWA", "Box", "None", "PickPlace"), ("Jaco", "Plate", "GoalWall", "Shelf"),
         ("Panda", "Dumbbell", "ObjectDoor", "Push")]
EPS, T = 3, 500


def fake(cell):
    env = b1_eval.make_env(cell)
    rng = np.random.default_rng(0)
    O, A, R = [], [], []
    for e in range(EPS):
        np.random.seed(e)
        o = b1_eval._obs(env.reset())
        obs, act, rew = [], [], []
        for t in range(T):
            a = rng.uniform(env.action_space.low, env.action_space.high)
            obs.append(o), act.append(a)
            o, r = env.step(a)[:2]
            o = b1_eval._obs(o)
            rew.append(r)
        O.append(obs), A.append(act), R.append(rew)
    O, A, R = (np.asarray(x, np.float32) for x in (O, A, R))
    d = b1.DATA / "smoke" / b1.cell_name(cell)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "episodes.npz", obs=O, act=A, rew=R, success=(R >= 1).any(1))
    b1_data.strip_task_id(O, cell)          # the layout assertion, on real observations
    return O.shape


def main():
    out = {}
    t0 = time.time()
    for c in CELLS:
        out[b1.cell_name(c)] = fake(c)
    out["fake_s"] = time.time() - t0
    r = subprocess.run([sys.executable, "-I", str(ROOT / "experiments/b1_policy.py"), "--cond", "free", "--kind", "smoke",
                        "--steps", "300", "--batch", "64", "--n-eps", str(EPS), "--ckpt-every", "100"],
                       capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-3000:])
    ck = torch.load(b1.RESULTS / "policy" / "free_smoke" / "latest.pt", map_location="cpu")
    meta = ck["meta"]
    m = FlowPolicy(**{k: meta["cfg"][k] for k in ("state_dim", "n_tasks", "cond", "d", "layers", "heads", "emb_dim")})
    m.load_state_dict(ck["ema"])
    norm = b1_data.Norm.from_dict(meta["norm"])
    torch.set_num_threads(1)
    res = b1_eval.rollout(m, norm, CELLS[0], dict(task=torch.tensor([0])), n_episodes=2, seed=0)
    out["rollout"] = res
    out["s_per_episode"] = res["seconds"] / 2
    out["steps_per_s"] = sum(res["steps"]) / res["seconds"]
    print(json.dumps(out, indent=1, default=str))
    (b1.RESULTS / "pipeline_smoke.json").write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
