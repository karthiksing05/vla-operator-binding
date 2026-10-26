"""PLAN_SOLVE S0 (docs/PLAN_S0.md): the binding adapter's training data.  LIBERO env (python 3.8).  No demo is EXECUTED: each
training cell's matched demo episode (its per-frame simulator states, results/protos/match or results/robot/p0_demo, the same
files D1 / D4 / D5 used) is SET in the simulator frame by frame, rendered as the policy sees it, and its recorded actions are the
flow-matching targets (fitted, never executed).

Per training cell with an In / On goal and a match file, frames 0, STRIDE, 2 STRIDE, ... ->
results/bind/s0/data/<tid>.npz: frames, img / wimg [n,224,224,3], state8 [n,8], chunk [n,H,7] raw actions (clamped at the
episode's end), body_names / body_cov [n,N,256], site_names / site_cov [n,S,256] (S1's masks), named, target, prompt.

    PYTHONPATH=.../third_party/libero .../examples/libero/.venv/bin/python experiments/s0_data.py --shard i/n
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
OUT = os.path.join(ROOT, "results", "bind", "s0", "data")
STRIDE, H = 3, 10


def cells():
    """-> [(tid, match path, episode, src)] for training cells with an In / On goal and a match file (s0_inventory's rule)."""
    tasks = json.load(open(os.path.join(ROOT, "results", "bind", "s1", "tasks.json"), encoding="utf-8"))
    split = json.load(open(os.path.join(ROOT, "results", "l90c", "split.json"), encoding="utf-8"))
    pp = json.load(open(os.path.join(ROOT, "results", "protos", "index_l90c.json"), encoding="utf-8"))
    eps = {}
    for o in pp["obs"]:
        if o["set"] == "train":
            eps.setdefault(f"{o['suite']}:{o['index']}", (o.get("src"), o.get("hf_task"), o.get("src_hf"), o.get("l90_task"), o["episode"]))
    out = []
    for tid in split["train"]:
        x = tasks[tid]
        if tid not in eps or not any(a[0] in ("In", "On") and len(a) == 3 for a in x["goal"]):
            continue
        src, hf, shf, l90t, ep = eps[tid]
        mp = (os.path.join(ROOT, "results", "robot", "p0_demo", f"{l90t}_match.npz") if src == "ipec"
              else os.path.join(ROOT, "results", "protos", "match", f"tr{(shf if shf is not None else hf):02d}.npz"))
        if os.path.exists(mp):
            out.append(dict(tid=tid, match=mp, episode=int(ep), ipec=src == "ipec", lang=x["lang"],
                            atom=next(a for a in x["goal"] if a[0] in ("In", "On") and len(a) == 3)))
    return out


def frame_no_settle(env, sim, W, LC):
    """S1's frame (s1_render.frame) at the SET state, without the settle steps (a demo frame is not a reset)."""
    import f0b_firstframes_seg as F0
    from openpi_client import image_tools
    obs = env.env._get_observations(force_update=True)
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wimg = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    tables = LC.seg_tables(env)
    cov = LC.seg_maps(env, tables, "agentview")
    m = sim.model
    K = F0.CU.get_camera_intrinsic_matrix(sim, "agentview", 256, 256)
    E = F0.CU.get_camera_extrinsic_matrix(sim, "agentview")
    sids = [i for i in range(m.nsite) if "region" in (m.site_id2name(i) or "") or (m.site_id2name(i) or "").endswith("_side")]
    scov = [W.box_map(sim.data.site_xpos[i], sim.data.site_xmat[i], m.site_size[i], K, E).reshape(-1) for i in sids]
    st8 = np.concatenate((obs["robot0_eef_pos"], F0.quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])).astype(np.float32)
    return dict(img=image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224)),
                wimg=image_tools.convert_to_uint8(image_tools.resize_with_pad(wimg, 224, 224)), state8=st8,
                body_names=np.array(tables[0]), body_cov=np.asarray(cov, np.float16),
                site_names=np.array([m.site_id2name(i) for i in sids]),
                site_cov=np.asarray(np.stack(scov) if scov else np.zeros((0, 256)), np.float16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    C = cells()
    if a.list:
        for c in C:
            print(f"[s0 data] {c['tid']}: {c['atom']} {os.path.basename(c['match'])} ep {c['episode']}", flush=True)
        print(f"[s0 data] {len(C)} cells", flush=True)
        return
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import libero_client as LC
    import s1_render as SR
    W = SR._where()
    i, n = map(int, a.shard.split("/"))
    os.makedirs(OUT, exist_ok=True)
    for c in C[i::n]:
        p = os.path.join(OUT, c["tid"].replace(":", "_") + ".npz")
        if os.path.exists(p):
            continue
        t0 = time.time()
        suite, idx = c["tid"].split(":")
        task = benchmark.get_benchmark_dict()[suite]().get_task(int(idx))
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        env.seed(0)
        env.reset()
        states = np.load(c["match"], allow_pickle=True)["states"]
        acts = np.load(os.path.join(ROOT, "results", "bind", "s0", "acts", c["tid"].replace(":", "_") + ".npy"))   # s0_acts.py
        T = min(len(states), len(acts))
        rec = {k: [] for k in ("frames", "img", "wimg", "state8", "chunk", "body_cov", "site_cov")}
        names = None
        for f in range(0, T, STRIDE):
            env.set_init_state(states[f])
            fr = frame_no_settle(env, env.env.sim, W, LC)
            if names is None:
                names = (fr["body_names"], fr["site_names"])
            ix = np.minimum(np.arange(f, f + H), T - 1)
            rec["frames"].append(f)
            rec["chunk"].append(acts[ix])
            for k in ("img", "wimg", "state8", "body_cov", "site_cov"):
                rec[k].append(fr[k])
        env.close()
        np.savez_compressed(p + ".tmp.npz", tid=c["tid"], prompt=c["lang"], named=c["atom"][1], target=c["atom"][2],
                            body_names=names[0], site_names=names[1], **{k: np.stack(v) for k, v in rec.items()})
        os.replace(p + ".tmp.npz", p)
        print(f"[s0 data] {c['tid']}: {len(rec['frames'])} frames of {T}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
