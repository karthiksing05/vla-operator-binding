"""X3 (EgoDex): inventory of the unzipped test set (header reads only; no video decoding).

Per episode: frame count (transforms/camera), duration at 30 Hz, the language attributes (llm_description, llm_description2,
which_llm_description).  Per task: episode count, total / mean duration, the distinct descriptions.  Also the joint names and
dataset shapes of one file.  Run on a compute node (SLURM):
    python -I experiments/x3_egodex_inventory.py --root /coc/flash10/ksingara3/robot/data/egodex/test --out results/robot/egodex
"""
from __future__ import annotations

import argparse
import collections
import json
import os

import h5py

FPS = 30.0


def _s(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return v if isinstance(v, (str, int, float)) or v is None else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/coc/flash10/ksingara3/robot/data/egodex/test")
    ap.add_argument("--out", default="results/robot/egodex")
    a = ap.parse_args()
    tasks, schema = {}, None
    for task in sorted(os.listdir(a.root)):
        d = os.path.join(a.root, task)
        if not os.path.isdir(d):
            continue
        eps = []
        names = sorted((n for n in os.listdir(d) if n.endswith(".hdf5")), key=lambda n: int(n.split(".")[0]) if n.split(".")[0].isdigit() else 1e9)
        for n in names:
            p = os.path.join(d, n)
            try:
                with h5py.File(p, "r") as f:
                    T = f["transforms/camera"].shape[0]
                    at = {k: _s(f.attrs[k]) for k in f.attrs.keys()}
                    if schema is None:
                        schema = dict(file=p, attrs=sorted(f.attrs.keys()),
                                      transforms=sorted(f["transforms"].keys()),
                                      confidences=sorted(f["confidences"].keys()) if "confidences" in f else [],
                                      top=sorted(f.keys()),
                                      shapes={k: list(f["transforms"][k].shape) for k in ["camera", "leftHand", "rightHand"]},
                                      other={k: [list(f[k].shape), str(f[k].dtype)] for k in f.keys() if isinstance(f[k], h5py.Dataset)},
                                      camera_attrs={k: _s(f["camera"].attrs[k]) for k in f["camera"].attrs} if "camera" in f else {})
                        if "camera" in f and isinstance(f["camera"], h5py.Group):
                            schema["camera_group"] = {k: list(f["camera"][k].shape) for k in f["camera"].keys()}
                eps.append(dict(ep=n.split(".")[0], frames=int(T), sec=round(T / FPS, 2),
                                mp4=os.path.exists(p[:-5] + ".mp4"), **{k: at.get(k) for k in at}))
            except Exception as e:
                eps.append(dict(ep=n, error=repr(e)))
        good = [e for e in eps if "frames" in e]
        descs = collections.Counter(e.get("llm_description") for e in good)
        tasks[task] = dict(episodes=len(good), errors=len(eps) - len(good),
                           hours=round(sum(e["sec"] for e in good) / 3600, 3),
                           mean_sec=round(sum(e["sec"] for e in good) / max(1, len(good)), 1),
                           min_sec=min((e["sec"] for e in good), default=0), max_sec=max((e["sec"] for e in good), default=0),
                           n_desc=len(descs), descs=[d for d, _ in descs.most_common(8)],
                           desc2=[d for d, _ in collections.Counter(e.get("llm_description2") for e in good).most_common(4)],
                           which=dict(collections.Counter(str(e.get("which_llm_description")) for e in good)),
                           episodes_list=eps)
        print(f"{task}: {len(good)} eps, {tasks[task]['hours']} h, {len(descs)} descs", flush=True)
    json.dump(dict(schema=schema, tasks=tasks), open(os.path.join(a.out, "test_inventory.json"), "w"), indent=1)
    print("TOTAL tasks", len(tasks), "episodes", sum(t["episodes"] for t in tasks.values()),
          "hours", round(sum(t["hours"] for t in tasks.values()), 2))


if __name__ == "__main__":
    main()
