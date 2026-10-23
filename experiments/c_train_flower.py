"""CALVIN-C P1: fine-tune FLOWER from flower_vla_pret with the authors' CALVIN ABC recipe on the training cells of a split seed
(docs/PLAN_V4.md 3.2; docs/CALVIN_C.md).  Resumable; runs inside one SLURM job with 4 GPUs (slurm/jobs/c_train_flower.sbatch).

RECIPE = the published checkpoint's own training config (mbreuss/flower_calvin_abc config.yaml, the run that produced 4.54;
its training.log shows 35 epochs x 1000 steps): FLOWERVLA (Florence-2-large, 18-layer DiT, rectified flow), pretrained init
from mbreuss/flower_vla_pret 360000_model_weights.pt (the file its training.log names), AdamW lr 2e-5 with the TriStage
schedule, EMA 0.999, bf16-mixed, DDP over 4 GPUs, batch 8 per GPU, limit_train_batches 1000, max_epochs 35 (= 35,000
optimizer steps; the README says "40k steps", the config and the log say 35k), the language-annotated windows of
task_ABC_D/training via HulcDataModule + ExtendedDiskDataset with the extracted rel_actions, seed 242.
DEVIATIONS (all outside the optimisation, except the data filter, which is the point):
  1. DATA FILTER (CALVIN-C): the lang dataset is lifelong_ood.calvin_c_data.CalvinCDiskDataset, the authors' class restricted to
     the windows experiments/c_calvin_split.py keeps for the split seed (held-out cells and windows overlapping them by
     > 16 frames dropped).  `--full` (the control) keeps every window.
  2. NO IN-TRAINING ROLLOUTS.  The authors evaluate 1,000 scene-D chains every 5 epochs from epoch 9 and keep the best by that
     score (model selection on the test scene).  Here training runs the fixed 35k steps and the final EMA weights are
     evaluated once, afterwards (experiments/x1_calvin_flower_eval.py).  The published checkpoint is the last epoch's.
  3. RESUME.  The authors' `get_last_checkpoint` reloads weights only (optimizer, EMA, step restart).  Here a full Lightning
     checkpoint (weights, optimizer, scheduler, EMA, loop state) is written every --ckpt_every steps and before the wall-clock
     budget, to <run>/ckpt/last.ckpt (temporary name + rename), and `trainer.fit(ckpt_path=...)` resumes from it.
  4. The <Flow> prompt embedding is a plain tensor (not a parameter, never saved): FLOWERVLA draws it from the freshly resized
     token row at construction.  It is saved at the first launch (<run>/prompt_embeds.pt) and restored on every resume, and
     the export writes it into the <Flow> row of the token table so the evaluation's load fix (x1 `--fix_weights`) reads
     exactly the training-time value.
  5. Logging: CSVLogger (no wandb); HulcDataModule.prepare_data (a glob over 1.8M files that only checks the directory is
     non-empty) is skipped; Lightning's own environment launcher (one SLURM task starting 4 local ranks; the authors use srun -n
     4); num_workers from --workers (the authors' 12 per GPU).

    python experiments/c_train_flower.py train --split_seed 0 --run results/calvin_c/flower_c_s0 [--max_steps N] [--full]
    python experiments/c_train_flower.py export --run results/calvin_c/flower_c_s0 [--ckpt <path>] [--out <dir>]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
FLOWER = pathlib.Path("/coc/flash10/ksingara3/robot/ext/flower_vla_calvin")
sys.path.insert(0, str(ROOT))
DATA = "/coc/flash10/ksingara3/robot/data/calvin_abc/task_ABC_D"
HF = "/coc/flash10/ksingara3/robot/data/hf/home/hub"
PUB_CFG = f"{HF}/models--mbreuss--flower_calvin_abc/snapshots/6424ee4c064eb6654a39280314793620b149dd97/config.yaml"
PRET = f"{HF}/models--mbreuss--flower_vla_pret/snapshots/ffd4134a2167a2be32bd7ad93455ed78dc2dcda2/360000_model_weights.pt"


def build_cfg(a):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(PUB_CFG)
    cfg.root_data_dir = DATA
    cfg.model.pretrained_model_path = PRET
    cfg.model.load_pretrained = True
    cfg.num_workers = a.workers
    if a.batch:
        cfg.batch_size = a.batch                      # per-GPU batch; the recipe's global batch is 4 x 8 = 32
    cfg.devices = a.devices
    cfg.callbacks.pop("rollout_lh")
    cfg.callbacks.pop("checkpoint")
    if not a.full:
        keep = ROOT / "results" / "calvin_c" / f"windows_s{a.split_seed}.npz"
        assert keep.exists(), keep
        cfg.datamodule.datasets.lang_dataset._target_ = "lifelong_ood.calvin_c_data.CalvinCDiskDataset"
        cfg.datamodule.datasets.lang_dataset.keep_file = str(keep)
    return cfg


def train(a):
    sys.path.insert(0, str(FLOWER))
    import calvin_env  # noqa: F401  (installed one first; the clone's empty calvin_env/ would shadow it)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "True")
    import hydra
    import torch
    from omegaconf import OmegaConf
    from pytorch_lightning import Callback, Trainer, seed_everything
    from pytorch_lightning.callbacks import LearningRateMonitor
    from pytorch_lightning.loggers import CSVLogger
    from pytorch_lightning.plugins.environments import LightningEnvironment

    run = pathlib.Path(a.run).resolve()
    (run / "ckpt").mkdir(parents=True, exist_ok=True)
    cfg = build_cfg(a)
    seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("medium")      # the authors also set the legacy allow_tf32 flags; torch 2.10 refuses
                                                      # the mix, and "medium" already gives TF32 matmuls (cudnn TF32 is default)
    rank = int(os.environ.get("LOCAL_RANK", "0"))

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data = lambda *x, **k: None
    model = hydra.utils.instantiate(cfg.model)
    pe = run / "prompt_embeds.pt"
    if pe.exists():
        saved = torch.load(pe, map_location="cpu")
        model.prompt_embeds = saved.to(model.prompt_embeds.dtype)
    elif rank == 0:
        torch.save(model.prompt_embeds.detach().cpu().clone(), pe)
    last = run / "ckpt" / "last.ckpt"
    if rank == 0:
        OmegaConf.save(cfg, run / "config.yaml")
        meta = dict(split_seed=None if a.full else a.split_seed, full=a.full, pret=PRET, recipe=PUB_CFG,
                    flow_token=int(model.tokenizer.convert_tokens_to_ids("<Flow>")), devices=a.devices,
                    workers=a.workers, max_steps=a.max_steps, argv=sys.argv, resumed_from=str(last) if last.exists() else None)
        with open(run / f"launch_{int(time.time())}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)

    class Clock(Callback):
        """Full checkpoint every `every` steps and once the wall-clock budget is spent (then stop); step timing log."""

        def __init__(self, every, budget_s):
            self.every, self.budget, self.t0, self.tl, self.sl = every, budget_s, time.time(), None, None
            self.saved = None

        def _save(self, trainer):
            if self.saved == int(trainer.global_step):
                return
            # torch.save streams small writes (~30 MB/s on this NFS): save to node-local disk, then one sequential copy
            loc = pathlib.Path(f"/tmp/ccalvin_{os.environ.get('SLURM_JOB_ID', 'x')}")
            loc.mkdir(parents=True, exist_ok=True)
            tmp, tnfs = loc / "last.ckpt", run / "ckpt" / "last.tmp.ckpt"
            ts = time.time()
            trainer.save_checkpoint(str(tmp))
            if trainer.is_global_zero:
                import shutil
                t1 = time.time()
                shutil.copyfile(tmp, tnfs)
                os.replace(tnfs, last)
                os.remove(tmp)
                print(f"[c_train] checkpoint at step {trainer.global_step}: local save {t1 - ts:.0f}s, copy "
                      f"{time.time() - t1:.0f}s", flush=True)
                with open(run / "progress.json", "w", encoding="utf-8") as f:
                    json.dump(dict(global_step=int(trainer.global_step), epoch=int(trainer.current_epoch),
                                   wall=time.time() - self.t0, time=time.strftime("%Y-%m-%d %H:%M:%S")), f)
            self.saved = int(trainer.global_step)
            trainer.strategy.barrier()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            gs = int(trainer.global_step)
            if self.tl is None:
                self.tl, self.sl = time.time(), gs
            if trainer.is_global_zero and gs % a.log_every == 0 and gs > self.sl:
                dt = (time.time() - self.tl) / (gs - self.sl)
                print(f"[c_train] step {gs} epoch {trainer.current_epoch} {dt:.3f} s/step "
                      f"wall {time.time() - self.t0:.0f}s loss {float(outputs['loss']) if isinstance(outputs, dict) else float(outputs):.4f}",
                      flush=True)
                self.tl, self.sl = time.time(), gs
            if gs % 50 == 0:
                over = trainer.strategy.broadcast(time.time() - self.t0 > self.budget, src=0)
                if over:
                    print(f"[c_train] wall budget reached at step {gs}; checkpointing and stopping", flush=True)
                    self._save(trainer)
                    trainer.should_stop = True
                    return
            if gs % self.every == 0 and gs > 0:
                self._save(trainer)

        def on_train_end(self, trainer, pl_module):
            self._save(trainer)

    callbacks = [hydra.utils.instantiate(cfg.callbacks.ema), LearningRateMonitor(logging_interval="step"),
                 Clock(a.ckpt_every, a.budget_min * 60)]
    targs = dict(OmegaConf.to_container(cfg.trainer, resolve=True))
    targs.update(logger=CSVLogger(str(run), name="logs"), callbacks=callbacks, benchmark=False,
                 strategy="ddp_find_unused_parameters_true" if a.find_unused else "ddp", accelerator="gpu", devices=a.devices,
                 use_distributed_sampler=True, default_root_dir=str(run), sync_batchnorm=True,
                 enable_checkpointing=False, plugins=[LightningEnvironment()], log_every_n_steps=50)
    # the run is defined by optimizer steps (the recipe's 35 epochs x 1000 batches): a mid-epoch resume restarts Lightning's
    # epoch counter, so an epoch budget would overshoot; max_epochs is lifted and max_steps fixed
    total = a.max_steps or int(cfg.max_epochs) * int(cfg.trainer.limit_train_batches)
    targs.update(max_steps=total, max_epochs=-1)
    trainer = Trainer(**targs)
    trainer.fit(model, datamodule=datamodule, ckpt_path=str(last) if last.exists() else None)
    if trainer.is_global_zero:
        done = int(trainer.global_step) >= total
        with open(run / "progress.json", "r", encoding="utf-8") as f:
            p = json.load(f)
        p["finished"] = bool(done)
        with open(run / "progress.json", "w", encoding="utf-8") as f:
            json.dump(p, f)
        print(f"[c_train] fit returned at step {trainer.global_step}; finished={done}", flush=True)


def export(a):
    """last.ckpt -> <out>/{config.yaml, model.safetensors} with the EMA weights (the authors' evaluation weights:
    evaluate_ema_weights_instead), the training-time <Flow> embedding written into the <Flow> token row."""
    import torch
    from omegaconf import OmegaConf
    from safetensors.torch import save_file
    run = pathlib.Path(a.run).resolve()
    ck = pathlib.Path(a.ckpt) if a.ckpt else run / "ckpt" / "last.ckpt"
    out = pathlib.Path(a.out) if a.out else run / f"export_step{a.tag}"
    c = torch.load(ck, map_location="cpu", weights_only=False)
    keys = list(c["state_dict"].keys())
    ek = [k for k in c["callbacks"] if "EMA" in str(k)]
    assert len(ek) == 1, list(c["callbacks"])
    ema = c["callbacks"][ek[0]]["ema_weights"]
    assert len(ema) == len(keys), (len(ema), len(keys))
    sd = {}
    for k, w in zip(keys, ema):
        assert tuple(w.shape) == tuple(c["state_dict"][k].shape), k
        sd[k] = w.detach().to("cpu").float().clone().contiguous() if w.is_floating_point() else w.detach().cpu().clone()
    launches = sorted(run.glob("launch_*.json"))
    tid = json.loads(launches[0].read_text(encoding="utf-8"))["flow_token"]
    pe = torch.load(run / "prompt_embeds.pt", map_location="cpu").float().reshape(-1)
    rows = [k for k in sd if k.endswith("shared.weight") or k.endswith("embed_tokens.weight")]
    for k in rows:
        sd[k][tid] = pe
    out.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(out / "model.safetensors"))
    cfg = OmegaConf.load(run / "config.yaml")
    OmegaConf.save(cfg, out / "config.yaml")
    info = dict(ckpt=str(ck), global_step=int(c.get("global_step", -1)), epoch=int(c.get("epoch", -1)),
                n_tensors=len(sd), flow_token=tid, flow_rows=rows, ema_key=str(ek[0]))
    (out / "export.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--split_seed", type=int, default=0)
    t.add_argument("--run", required=True)
    t.add_argument("--full", action="store_true", help="control: every annotated window (the published recipe as is)")
    t.add_argument("--devices", type=int, default=4)
    t.add_argument("--workers", type=int, default=10)
    t.add_argument("--max_steps", type=int, default=0, help="timing runs only; 0 = the recipe's 35 epochs x 1000 steps")
    t.add_argument("--ckpt_every", type=int, default=1000)
    t.add_argument("--log_every", type=int, default=50)
    t.add_argument("--find_unused", type=int, default=1, help="DDP find_unused_parameters (the authors': true)")
    t.add_argument("--batch", type=int, default=0, help="per-GPU batch override (0 = the recipe's 8)")
    t.add_argument("--budget_min", type=float, default=215.0, help="wall minutes before checkpoint-and-stop")
    e = sub.add_parser("export")
    e.add_argument("--run", required=True)
    e.add_argument("--ckpt", default=None)
    e.add_argument("--out", default=None)
    e.add_argument("--tag", default="final")
    a = ap.parse_args()
    train(a) if a.cmd == "train" else export(a)


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ.setdefault("WANDB_MODE", "disabled")
    main()
