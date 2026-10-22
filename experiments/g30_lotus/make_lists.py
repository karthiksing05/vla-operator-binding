"""Plan D2 / G30f (LOTUS): the stream lists and E3's set-b start states in LOTUS's init-file layout.  Run in the LOTUS venv (CPU).
    list_stream.txt   -- pi0.5's 40 training tasks (LIBERO-SPATIAL, -OBJECT, -GOAL, -10 in benchmark order: LOTUS's base stage), then
                         W1's order a (results/robot/w1_state.json) without task 51 (no demonstration in our data, as EWC/PackNet)
    init_setb/libero_90/<task>.pruned_init -- results/robot/fresh_states_b/libero_90_<t>.npy (20 states, sampler seeds 200000-200019)"""
import importlib.util
import json
import os

import numpy as np
import torch

REPO = "/coc/flash10/ksingara3/lifelong-ood-learner"
OUT = os.path.join(REPO, "results/robot/g30_lotus")
spec = importlib.util.spec_from_file_location("m", "/coc/flash10/ksingara3/robot/work/lotus/lotus/libero/benchmark/libero_suite_task_map.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
T = m.libero_task_map
order = json.load(open(os.path.join(REPO, "results/robot/w1_state.json")))["a"]["order"]
base = [f"{s}/{t}_demo" for s in ("libero_spatial", "libero_object", "libero_goal", "libero_10") for t in T[s]]
stream = [f"libero_90/{T['libero_90'][t]}_demo" for t in order if t != 51]
open(os.path.join(OUT, "list_stream.txt"), "w", encoding="utf-8").write("\n".join(base + stream) + "\n")
json.dump(dict(order=order, ids=[t for t in order if t != 51], names={t: T["libero_90"][t] for t in order}),
          open(os.path.join(OUT, "stream_ids.json"), "w", encoding="utf-8"), indent=1)
d = os.path.join(OUT, "init_setb", "libero_90")
os.makedirs(d, exist_ok=True)
for t in order:
    x = np.load(os.path.join(REPO, f"results/robot/fresh_states_b/libero_90_{t}.npy"))
    torch.save(x, os.path.join(d, T["libero_90"][t] + ".pruned_init"))
print(len(base), len(stream), "states", len(os.listdir(d)))
