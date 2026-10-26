"""PLAN_SOLVE S1 (docs/PLAN_S1.md): render the binder's frames.  LIBERO env (python 3.8), CPU; no policy, no demo.

  train  every l90c TRAINING cell (lifelong_ood.l90c.load_split()["train"]): N_TRAIN fresh sampler states (env.seed(TRAIN_BASE + k);
         env.reset()), the client's 10 settle steps (dummy action), then the agentview / wrist images (client orientation, 224)
  dev    the 24 dev tasks of results/bind/e1_scene.json on fresh_m1 states 0 .. N_DEV-1 (E2 / E2b's states), same settle

Per frame -> results/bind/s1/frames/<set>/<tid>_<k>.npz:
  img, wimg [224,224,3] uint8; state8 [8]
  body_names, body_cov [N,256] f16   per movable object / fixture: segmentation coverage of each of the 16 x 16 agentview tokens
  site_names, site_cov [S,256] f16   per BDDL region site (name contains "region" or ends in "_side", e.g. top_side): coverage of its projected box
                                    (lifelong_ood/policy/where.box_map; the box's bounding rectangle, occlusion ignored)
  pos {body: xyz}, site_pos [S,3]

    PYTHONPATH=.../third_party/libero .../examples/libero/.venv/bin/python experiments/s1_render.py --set train --shard 0/4
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
OUT = os.path.join(ROOT, "results", "bind", "s1", "frames")
N_TRAIN, TRAIN_BASE, N_DEV, SETTLE = 8, 9_300_000, 5, 10
DUMMY = [0.0] * 6 + [-1.0]


def _where():
    spec = importlib.util.spec_from_file_location("where", os.path.join(ROOT, "lifelong_ood", "policy", "where.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def tasks(set_name):
    if set_name == "train":
        split = json.load(open(os.path.join(ROOT, "results", "l90c", "split.json"), encoding="utf-8"))
        return list(split["train"])
    return list(json.load(open(os.path.join(ROOT, "results", "bind", "e1_scene.json"), encoding="utf-8")))


def frame(env, sim, W, LC):
    import f0b_firstframes_seg as F0
    from openpi_client import image_tools
    obs = None
    for _ in range(SETTLE):
        obs, _, _, _ = env.step(DUMMY)
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wimg = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    tables = LC.seg_tables(env)
    cov = LC.seg_maps(env, tables, "agentview")
    m = sim.model
    K = F0.CU.get_camera_intrinsic_matrix(sim, "agentview", 256, 256)
    E = F0.CU.get_camera_extrinsic_matrix(sim, "agentview")
    sids = [i for i in range(m.nsite) if "region" in (m.site_id2name(i) or "") or (m.site_id2name(i) or "").endswith("_side")]
    scov = [W.box_map(sim.data.site_xpos[i], sim.data.site_xmat[i], m.site_size[i], K, E).reshape(-1) for i in sids]
    inner = env.env
    pos = {n: np.asarray(sim.data.body_xpos[inner.obj_body_id[n]], float).tolist() for n in tables[0] if n in inner.obj_body_id}
    st8 = np.concatenate((obs["robot0_eef_pos"], F0.quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])).astype(np.float32)
    return dict(img=image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224)),
                wimg=image_tools.convert_to_uint8(image_tools.resize_with_pad(wimg, 224, 224)), state8=st8,
                body_names=np.array(tables[0]), body_cov=np.asarray(cov, np.float16),
                site_names=np.array([m.site_id2name(i) for i in sids]),
                site_cov=np.asarray(np.stack(scov) if scov else np.zeros((0, 256)), np.float16),
                site_pos=np.array([sim.data.site_xpos[i] for i in sids], np.float64).reshape(-1, 3), pos=json.dumps(pos))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=("train", "dev"), required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--max-tasks", type=int, default=0)
    a = ap.parse_args()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import libero_client as LC
    W = _where()
    i, n = map(int, a.shard.split("/"))
    T = tasks(a.set)[i::n]
    if a.max_tasks:
        T = T[:a.max_tasks]
    od = os.path.join(OUT, a.set)
    os.makedirs(od, exist_ok=True)
    for tid in T:
        suite, idx = tid.split(":")
        t0 = time.time()
        task = benchmark.get_benchmark_dict()[suite]().get_task(int(idx))
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        ks = range(N_TRAIN if a.set == "train" else N_DEV)
        fresh = None
        if a.set == "dev":
            fresh = np.load(os.path.join(ROOT, "results", "robot", "fresh_m1", f"{suite}_{idx}.npy"))
        for k in ks:
            p = os.path.join(od, f"{tid.replace(':', '_')}_{k}.npz")
            if os.path.exists(p):
                continue
            if a.set == "train":
                env.seed(TRAIN_BASE + k)
                env.reset()
            else:
                env.seed(0)
                env.reset()
                env.set_init_state(fresh[k])
            fr = frame(env, env.env.sim, W, LC)
            np.savez_compressed(p + ".tmp.npz", tid=tid, k=k, language=task.language, **fr)
            os.replace(p + ".tmp.npz", p)
        env.close()
        print(f"[s1 render] {a.set} {tid}: {len(ks)} frames, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
