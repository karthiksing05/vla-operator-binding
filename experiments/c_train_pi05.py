"""P1: train pi05_l90c from pi05_base on the LIBERO-90-C training cells (docs/L90C.md, docs/PLAN_V4.md 3.1).

openpi's own JAX trainer (scripts/train.py: FSDP over the job's GPUs, AdamW, EMA, orbax checkpoints) with the TrainConfig of
lifelong_ood/policy/l90c_config.py (pi05_libero's recipe; data lol/libero_l90c from experiments/c_data.py).  Resumable: a run
whose checkpoint dir already holds a step resumes from it (openpi --resume), so the same command is resubmitted every 4 h.

    --exp NAME --steps N --batch B [--lora] [--save-interval K] [--init PARAMS_DIR]
    --init       initialise a NEW run from an existing JAX checkpoint's params (e.g. .../jax/pi05_l90c/main/5000/params: the
                 EMA weights) instead of the config's weight loader (pi05_base); fresh optimizer, EMA and step counter.  Only
                 used when the run starts (a resume restores its own state).  Default: unset (unchanged behaviour)
    --timing     a timing run: N steps, then report seconds per step (excluding compilation) and the peak memory
    --export     convert the newest (or --step) JAX checkpoint to PyTorch for PolicyBackbone (l90c_config.pytorch_dir)

Run on a 4-GPU node (slurm/jobs/l90c_train.sbatch).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OPENPI = "/coc/flash10/ksingara3/robot/openpi"
LOG = os.path.join(ROOT, "results", "l90c", "train_log.jsonl")


def config(a):
    from lifelong_ood.policy import l90c_config as LC
    name = "pi05_l90c_lora" if a.lora else "pi05_l90c_fac" if a.fac else "pi05_l90c"
    import jax
    n = jax.device_count()
    cfg = LC.train_config(name, exp_name=a.exp, batch_size=a.batch, num_train_steps=a.steps,
                          save_interval=a.save_interval, keep_period=a.keep_period,
                          fsdp_devices=1 if a.lora else n)
    if a.init:                                             # additive: a continued fine-tune from an existing checkpoint
        from openpi.training import weight_loaders as W
        assert os.path.isdir(a.init), a.init
        cfg = dataclasses.replace(cfg, weight_loader=W.CheckpointWeightLoader(a.init))
    ck = cfg.checkpoint_dir
    steps = [int(p) for p in os.listdir(ck) if p.isdigit()] if os.path.isdir(ck) else []
    cfg = dataclasses.replace(cfg, resume=bool(steps), overwrite=False)
    if os.path.isdir(ck) and not steps:                    # a run that died before its first checkpoint: start clean
        shutil.rmtree(ck)
    return cfg, steps


def log(**kw):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), job=os.environ.get("SLURM_JOB_ID"), **kw)) + "\n")


def train(a):
    sys.path.insert(0, OPENPI + "/src")
    sys.path.insert(0, OPENPI + "/scripts")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.92")
    from lifelong_ood.policy import l90c_config as LC
    LC.register()
    import jax
    import train as T                                       # openpi/scripts/train.py
    cfg, steps = config(a)
    print(f"[c_train] {cfg.name}/{cfg.exp_name}: {jax.device_count()} x {jax.devices()[0].device_kind}, batch {cfg.batch_size}, "
          f"steps {cfg.num_train_steps}, fsdp {cfg.fsdp_devices}, resume {cfg.resume} (have {sorted(steps)[-3:]}), "
          f"init {a.init or 'config weight loader'}", flush=True)
    log(event="start", exp=a.exp, name=cfg.name, batch=cfg.batch_size, steps=cfg.num_train_steps, resume=cfg.resume,
        have=sorted(steps), devices=jax.device_count(), **({"init": a.init} if a.init else {}))
    t0 = time.time()
    stamps = []                                             # wall time at each training step (openpi's tqdm loop)
    base_bar = T.tqdm.tqdm

    def bar(it, *args, **kw):
        b = base_bar(it, *args, **kw)

        class _It:
            def __iter__(self_):
                for x in b:
                    stamps.append(time.time())
                    if len(stamps) % 50 == 0:
                        d = stamps[-1] - stamps[-50]
                        print(f"[c_train] step {x}: {d / 49:.3f} s/step (last 50), {time.time() - t0:.0f}s since start", flush=True)
                    yield x

            def write(self_, *aa, **kk):
                return b.write(*aa, **kk)
        return _It()
    T.tqdm.tqdm = bar
    try:
        T.main(cfg)
    finally:
        dt = time.time() - t0
        import numpy as np
        d = np.diff(stamps) if len(stamps) > 2 else np.array([float("nan")])
        steady = d[10:] if len(d) > 20 else d
        stats = dict(steps_run=len(stamps), first_step_at=(stamps[0] - t0) if stamps else None,
                     sec_per_step_median=float(np.median(steady)), sec_per_step_mean=float(np.mean(steady)),
                     sec_per_step_max=float(np.max(steady)))
        try:
            stats["peak_gb"] = [round(dv.memory_stats().get("peak_bytes_in_use", 0) / 1e9, 2) for dv in jax.local_devices()]
        except Exception:                                   # pragma: no cover
            pass
        log(event="end", exp=a.exp, seconds=dt, **stats)
        print(f"[c_train] done in {dt:.0f}s: " + json.dumps(stats), flush=True)


def export(a):
    from lifelong_ood.policy import l90c_config as LC
    name = "pi05_l90c_lora" if a.lora else "pi05_l90c_fac" if a.fac else "pi05_l90c"
    ck = os.path.join(LC.L90C, "jax", name, a.exp)
    steps = sorted(int(p) for p in os.listdir(ck) if p.isdigit())
    step = a.step if a.step is not None else steps[-1]
    src = os.path.join(ck, str(step))
    dst = LC.pytorch_dir(name) + ("" if a.step is None and not a.tag else f"_{a.tag or step}")
    cmd = [sys.executable, OPENPI + "/examples/convert_jax_model_to_pytorch.py", "--checkpoint_dir", src,
           "--config_name", "pi05_libero", "--output_path", dst]
    print("[c_train]", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=OPENPI)
    if rc == 0 and not os.path.exists(os.path.join(dst, "assets")):
        shutil.copytree(os.path.join(src, "assets"), os.path.join(dst, "assets"))
    log(event="export", exp=a.exp, step=step, dst=dst, rc=rc)
    sys.exit(rc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--fac", action="store_true", help="pi05_l90c_fac: factored conditioning (docs/L90C.md 3.1)")
    ap.add_argument("--save-interval", type=int, default=1000)
    ap.add_argument("--keep-period", type=int, default=5000)
    ap.add_argument("--init", default=None, help="a NEW run starts from this JAX params dir (default: the config's loader)")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    export(a) if a.export else train(a)
