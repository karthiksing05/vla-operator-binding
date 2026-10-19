"""FRESH init states: LIBERO's own placement sampler (the BDDL regions) run under fixed seeds, saved once per task so every arm is
scored from the same never-before-used start states (all 50 stored init states of LIBERO-90 have been used by earlier reads).
Run in the LIBERO client's venv (experiments/fresh_states.sbatch).  Output: results/robot/fresh_states/libero_90_<task>.npy
[n, state_dim], in the stored init states' format (MjSimState flattened), checked against it.

    sbatch --export=ALL,TASKS=10+15,N=20,BASE=100000 experiments/fresh_states.sbatch
"""
import os
import pathlib
import sys

import numpy as np

SUITE = os.environ.get("SUITE", "libero_90")
if SUITE.endswith("_ood"):                                      # plan G4: Li's OOD suites, in their modified LIBERO (libero_client.py)
    sys.path.insert(0, "/coc/flash10/ksingara3/ext_dl/pi0tl/pi0-text-latent/third_party/modified_libero")
    os.environ["LIBERO_CONFIG_PATH"] = "/coc/flash10/ksingara3/robot/libero_ood_config"
if SUITE in ("libero_goal_task", "libero_spatial_task", "libero_object_task", "libero_10_task"):   # plan H4 / G22: LIBERO-PRO's task
    sys.path.insert(0, "/coc/flash10/ksingara3/robot/ext/LIBERO-PRO")                                 # split, in its LIBERO package
    os.environ["LIBERO_CONFIG_PATH"] = "/coc/flash10/ksingara3/robot/libero_pro_config"               # (libero_client.py)
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

OUT = os.path.join("/coc/flash10/ksingara3/lifelong-ood-learner/results/robot",
                   os.environ.get("FRESH_DIR", "fresh_states"))   # a second set (fresh_states_b, BASE 200000) for E3


def main():
    tasks = [int(x) for x in os.environ["TASKS"].split("+")]
    n, base = int(os.environ.get("N", 20)), int(os.environ.get("BASE", 100000))
    os.makedirs(OUT, exist_ok=True)
    suite = benchmark.get_benchmark_dict()[SUITE]()
    for tid in tasks:
        if os.path.exists(os.path.join(OUT, f"{SUITE}_{tid}.npy")):
            continue                                            # done by an earlier (interrupted) job: the seeds fix the states
        task = suite.get_task(tid)
        stored = None if SUITE.endswith("_ood") else suite.get_task_init_states(tid)   # the OOD suites store none
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=64, camera_widths=64)
        states = []
        for k in range(n):
            np.random.seed(base + k)
            env.seed(base + k)
            env.reset()                                         # a new placement from the task's regions
            s = env.sim.get_state().flatten()
            assert stored is None or s.shape == stored[0].shape, (s.shape, stored[0].shape)
            states.append(s)
        states = np.stack(states)
        d = np.min(np.linalg.norm(states[:, None, :] - stored[None, :, :], axis=-1)) if stored is not None else float("nan")
        np.save(os.path.join(OUT, f"{SUITE}_{tid}.npy"), states)
        print(f"[fresh] task {tid}: {n} states (seed base {base}), min distance to a stored init state {d:.3f}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
