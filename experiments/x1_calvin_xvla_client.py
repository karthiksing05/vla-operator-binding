"""X1 parity gate: X-VLA (2toINF/X-VLA-Calvin-ABC_D) on CALVIN ABC->D with the authors' own client.

Imports github.com/2toinf/X-VLA evaluation/calvin/calvin_client.py (path passed in) and runs what its main() runs --
seed_everything(0), get_env(Path("ABC_D/validation")), ClientModel(host, port), evaluate_policy(eval_start, eval_end) -- against
the authors' server (`python -m deploy --model_path ...`, the repo root).  Substitutions, all outside the decision path:
  * cwd = --work_dir, which holds ABC_D/validation (the client resolves that relative path); it is the same 6 files the repo
    ships (scene-D env config, task oracle, one annotation per subtask), copied out of the download dir;
  * --no_video makes save_video a no-op (the client writes one mp4 per subtask; it never feeds back into the rollout);
  * numpy's removed np.float / np.int / np.bool aliases are restored as the builtins (urdfpy 0.0.22, imported by the tactile
    camera the client's env config instantiates, still uses them);
  * one warm-up request per server before the evaluation (the client's 10 s timeout, no retry, killed shards on a cold start);
  * the older client (blob e57bc00, before commit ecbef0c "update eval for calvin" of 2026-01-05) has no eval_start/end, so
    its get_sequences is sliced to [start, end) of the same seeded 1,000-chain list;
  * --ep_len overrides the module's EP_LEN.  The authors' client uses 720 steps per subtask; CALVIN's standard protocol
    (calvin_agent/evaluation/evaluate_policy.py, FLOWER) uses 360.  Default 720 = the authors' code.
Per-chain results -> <out>/results_<start>_<end>.json (the client itself only writes cumulative rates to log.txt).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", required=True, help="path to X-VLA evaluation/calvin/calvin_client.py")
    ap.add_argument("--work_dir", required=True, help="dir containing ABC_D/validation")
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=100)
    ap.add_argument("--ep_len", type=int, default=720)
    ap.add_argument("--no_video", action="store_true")
    args = ap.parse_args()

    import numpy as np
    for k, v in (("float", float), ("int", int), ("bool", bool)):   # urdfpy 0.0.22 (tacto's tactile camera) uses the aliases
        if k not in np.__dict__:                                     # numpy >= 1.24 removed; restore them as the builtins
            setattr(np, k, v)
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("xvla_calvin_client", args.client)
    cc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cc)
    cc.EP_LEN = args.ep_len
    if args.no_video:
        cc.save_video = lambda *a, **k: None

    os.chdir(args.work_dir)
    from pathlib import Path
    cc.seed_everything(0, workers=True)
    env = cc.get_env(Path("ABC_D/validation"), show_gui=False)
    model = cc.ClientModel(args.host, args.port)
    # warm-up: the client's requests carry timeout = 10 s and no retry, and a server's FIRST call (CUDA init, 5 servers starting
    # on one GPU) can exceed it, which killed whole shards.  One request on the initial observation, with a long timeout, before
    # the evaluation; the server is stateless (/act decodes the payload and samples), so nothing carries over.
    o = env.get_obs()
    model.reset(o)
    payload = {"language_instruction": "warm up", "proprio": cc.json_numpy.dumps(model.proprio.tolist()),
               "image0": cc.json_numpy.dumps(o["rgb_obs"]["rgb_static"]), "image1": cc.json_numpy.dumps(o["rgb_obs"]["rgb_gripper"]),
               "domain_id": 2, "steps": 10}
    for k in range(5):
        try:
            cc.requests.post(model.url, json=payload, timeout=600).raise_for_status()
            break
        except Exception as e:                       # noqa: BLE001
            print("warm-up failed, retrying:", e, flush=True)
            time.sleep(10)
    model.action_plan.clear()
    t0 = time.time()
    import inspect
    if "eval_start" in inspect.signature(cc.evaluate_policy).parameters:
        res = cc.evaluate_policy(model, env, str(out), debug=False, eval_start=args.start, eval_end=args.end)
    else:                                            # the pre-2026-01-05 client (e57bc00) evaluates all NUM_SEQUENCES
        full = list(cc.get_sequences(cc.NUM_SEQUENCES))
        cc.get_sequences = lambda n: full[args.start:args.end]
        res = cc.evaluate_policy(model, env, str(out), debug=False)
    n = len(res)
    rec = dict(start=args.start, end=args.end, results=[int(r) for r in res], seconds=time.time() - t0,
               avg_len=sum(res) / max(n, 1), chain_sr=[sum(r >= i for r in res) / max(n, 1) for i in range(1, 6)],
               ep_len=args.ep_len)
    (out / f"results_{args.start}_{args.end}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print("RESULT", json.dumps({k: rec[k] for k in ("start", "end", "avg_len", "chain_sr", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
