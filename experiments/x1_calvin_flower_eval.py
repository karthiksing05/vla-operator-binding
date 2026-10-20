"""X1 parity gate: FLOWER (mbreuss/flower_calvin_abc) on CALVIN ABC->D with the authors' own evaluation code.

Runs flower_vla_calvin's flower/evaluation/flower_evaluate.py `main()` unchanged (seed_everything(0), model loading with
load_mode_from_safetensor, num_sampling_steps = 4, multistep = 10, ep_len = 360, evaluate_policy / rollout / print_and_save),
with three thin substitutions, all outside the decision path:

  1. get_default_mode_and_env: the authors' version builds a HulcDataModule over the full task_ABC_D dataset only to obtain
     (a) the validation dir for calvin_env's get_env (it reads <dir>/.hydra/merged_config.yaml), (b) the val image transforms
     and (c) the robot_obs normalisation from statistics.yaml.  We do not download the ~500 GB dataset, so a stand-in exposes
     the same attributes: (a) task_ABC_D/validation/.hydra (the scene-D env config; copy of the one X-VLA's repo ships),
     (b) the checkpoint config's datamodule.transforms.val instantiated exactly as HulcDataModule.setup does, and (c) no
     robot_obs transform -- the checkpoint has use_proprio = false, so the state never reaches the network.  The env is the
     authors' HulcWrapper, the model the authors' load_mode_from_safetensor.
  2. get_sequences: returns the [start, end) slice of the standard get_sequences(1000) list, so 1,000 chains can be sharded
     over processes; each chain keeps its own initial state and instructions (the list is seeded, deterministic).
  3. wandb.log is a no-op (print_and_save calls it unconditionally; log_wandb = False).

Per-chain results are written to <out>/results_<start>_<end>.json; x1_calvin_merge.py merges shards.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time


RENAMED = {"vlm.language_shared.weight": "vlm.language_model.model.shared.weight",
           "vlm.language_final_logits_bias": "vlm.language_model.final_logits_bias"}


def _fix_weights(model, path):
    """backbone_flower.FlowerBackbone._fix_weights, on the authors' loaded model."""
    import torch
    from safetensors import safe_open
    sd = model.state_dict()
    done = {}
    with torch.no_grad(), safe_open(str(path), framework="pt") as f:
        for old, new in RENAMED.items():
            if old in f.keys() and new in sd:
                w = f.get_tensor(old)
                assert tuple(w.shape) == tuple(sd[new].shape), (old, tuple(w.shape), tuple(sd[new].shape))
                sd[new].copy_(w)
                done[new] = tuple(w.shape)
        tid = model.tokenizer.convert_tokens_to_ids("<Flow>")
        emb = model.vlm.get_input_embeddings().weight
        model.prompt_embeds = emb[tid].detach().clone()[None, None]
    return dict(loaded=done, flow_token=int(tid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flower_root", required=True, help="clone of github.com/intuitive-robots/flower_vla_calvin")
    ap.add_argument("--ckpt_dir", required=True, help="HF snapshot dir of mbreuss/flower_calvin_abc")
    ap.add_argument("--val_dir", required=True, help="dir with .hydra/merged_config.yaml for scene D (task_ABC_D/validation)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=1000)
    ap.add_argument("--ep_len", type=int, default=360)
    ap.add_argument("--fix_weights", action="store_true",
                    help="DEVIATION (off = the authors' loader): load the two tensors the HF export renamed and set the <Flow> "
                         "prompt embedding to the checkpoint's token row, as lifelong_ood/policy/backbone_flower.py does")
    args = ap.parse_args()

    flower_root = pathlib.Path(args.flower_root).resolve()
    # the clone holds an EMPTY calvin_env/ submodule dir at its root; with the root first on sys.path (flower_evaluate.py puts
    # it there too) `import calvin_env` would resolve to that namespace package (calvin_env.__file__ = None, the README's
    # "Common Issues").  Import the installed calvin_env (mees/calvin_env 797142c, FLOWER's pinned commit) first.
    import calvin_env
    assert calvin_env.__file__ is not None, calvin_env
    sys.path.insert(0, flower_root.as_posix())
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    import hydra
    import torch
    import torchvision
    import wandb
    from omegaconf import OmegaConf

    wandb.log = lambda *a, **k: None
    # calvin_env's EGL probe compiles egl_check/EGL_options.o, which needs X11 headers this cluster lacks; HulcWrapper's own
    # fallback for a failed probe is EGL device 0 (one GPU per job here), so take that branch directly
    import flower.wrappers.hulc_wrapper as hw
    from calvin_env.utils.utils import EglDeviceNotFoundError

    def _no_probe(cuda_id):
        raise EglDeviceNotFoundError()
    hw.get_egl_device_id = _no_probe
    import flower.evaluation.flower_evaluate as fe
    import flower.evaluation.utils as fu
    from flower.evaluation.multistep_sequences import get_sequences as _get_sequences

    all_seqs = _get_sequences(1000)
    fe.get_sequences = lambda n: all_seqs[args.start:args.end]

    class _ValDataset:                                   # the attributes HulcWrapper reads from the val lang dataset
        def __init__(self, cfg):
            self.abs_datasets_dir = pathlib.Path(args.val_dir).resolve()
            self.observation_space = cfg.datamodule.observation_space
            self.proprio_state = cfg.datamodule.proprioception_dims
            tv = cfg.datamodule.transforms.val
            self.transforms = {cam: torchvision.transforms.Compose([hydra.utils.instantiate(t) for t in tv[cam]])
                               for cam in tv}

    def get_default_mode_and_env(train_folder, dataset_path, checkpoint, env=None, lang_embeddings=None,
                                 prep_dm_and_deps=True, device_id=0, eval_cfg_overwrite={}):
        cfg = OmegaConf.merge(OmegaConf.load(pathlib.Path(train_folder) / "config.yaml"),
                              OmegaConf.create(eval_cfg_overwrite))
        device = torch.device(f"cuda:{device_id}")
        dataset = _ValDataset(cfg)
        rollout_cfg = OmegaConf.load(flower_root / "conf/callbacks/rollout_lh/calvin.yaml")
        env = hydra.utils.instantiate(rollout_cfg.env_cfg, dataset, device, show_gui=False)
        model = fu.load_mode_from_safetensor(pathlib.Path(checkpoint), overwrite_cfg=eval_cfg_overwrite.get("model", {}))
        if args.fix_weights:
            print("[x1] fix_weights:", _fix_weights(model, pathlib.Path(checkpoint)), flush=True)
        model.freeze()
        model = model.cuda(device)
        return model, env, None, None

    fe.get_default_mode_and_env = get_default_mode_and_env

    t0 = time.time()
    _eval = fe.evaluate_policy

    def evaluate_policy(*a, **k):
        res, plans = _eval(*a, **k)
        n = len(res)
        rec = dict(start=args.start, end=args.end, results=[int(r) for r in res], seconds=time.time() - t0,
                   avg_len=sum(res) / max(n, 1),
                   chain_sr=[sum(r >= i for r in res) / max(n, 1) for i in range(1, 6)],
                   ep_len=args.ep_len, ckpt=str(args.ckpt_dir))
        (out / f"results_{args.start}_{args.end}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        return res, plans

    fe.evaluate_policy = evaluate_policy

    # print_and_save names the epoch by checkpoint.stem.split("=")[1] (their files are "...avg_seq_len=4.72.ckpt"); the HF file
    # is model.safetensors, so give the key a name with an "=" (printing / results.json only)
    _pas = fe.print_and_save
    fe.print_and_save = lambda total, plans, cfg, log_dir=None: _pas(
        {pathlib.Path("flower_calvin_abc=hf"): v for v in total.values()}, plans, cfg, log_dir=log_dir)

    ckpt = pathlib.Path(args.ckpt_dir).resolve()
    os.chdir(out)                                         # hydra run dir / log_dir land under the output dir
    sys.argv = [sys.argv[0], f"--config-path={flower_root / 'conf'}", "--config-name=eval_calvin",
                f"train_folder={ckpt}", f"checkpoint={ckpt / 'model.safetensors'}", f"dataset_path={args.val_dir}",
                f"log_dir={out / 'flower_logs'}", "log_wandb=false", "num_videos=0", "device=0",
                f"num_sequences={args.end - args.start}", f"ep_len={args.ep_len}",
                f"hydra.run.dir={out / 'hydra'}"]
    fe.main()


if __name__ == "__main__":
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    main()
