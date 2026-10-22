"""Pi0MetaWorldBackbone: RLinf's Meta-World SFT pi0 / pi0.5 (RLinf/RLinf-Pi0-MetaWorld-SFT, RLinf/RLinf-Pi05-MetaWorld-SFT)
behind the PolicyBackbone interface (lifelong_ood/policy/backbone.py), for plan X4 (docs/METAWORLD.md).

The network is openpi's PyTorch PI0Pytorch, so the field, the prefix cache, the sampler and the identities of
PolicyBackbone carry over unchanged.  What differs from the LIBERO checkpoints:
  - the TrainConfig is rebuilt here (RLinf's pi0_metaworld / pi05_metaworld, rlinf/models/embodiment/openpi/dataconfig):
    Pi0Config(action_horizon=5) (pi0.5: pi05=True, discrete_state_input=False), max_token_len 48 / 200;
  - inputs (RLinf MetaworldInputs): one camera, the 480 x 480 corner2 view flipped on both axes by the caller (as RLinf's
    and LeRobot's loops do), as base_0_rgb; the two wrist slots are zero images with mask False; state = obs[:4]
    (end-effector xyz + gripper distance); prompt = the task description; then Normalize (mean / std for pi0, quantiles
    for pi0.5: openpi's rule use_quantile_norm = model_type != PI0) with the checkpoint's lerobot/metaworld_mt50 stats,
    ResizeImages(224), TokenizePrompt, PadStatesAndActions(32);
  - actions: H = 5 chunks of 4 real dims (end-effector delta xyz + gripper), Unnormalize, then the first 4 dims (RLinf
    MetaworldOutputs); no delta / absolute transform (extra_delta_transform False);
  - num_steps 5 (RLinf's evaluation setting);
  - the weights load with strict=False, as RLinf's loader does; the missing / unexpected keys are kept on the object.
The sampler runs eager.  openpi's PI0Pytorch wraps sample_actions in torch.compile(max-autotune), so the authors'
evaluation runs the compiled sampler; the parity check (experiments/x4_metaworld_parity.py) compares against both.

The installed openpi (the X4 venv's rlinf-openpi 0.1.1) is used; nothing from robot/openpi/src is put on sys.path.
"""
from __future__ import annotations

import dataclasses
import os
import pathlib
import time

import numpy as np
import torch

from lifelong_ood.policy.backbone import PolicyBackbone

CKPTS = {"pi0": "RLinf/RLinf-Pi0-MetaWorld-SFT", "pi05": "RLinf/RLinf-Pi05-MetaWorld-SFT"}
ASSET_ID = "lerobot/metaworld_mt50"


def _inputs_cls():
    from openpi import transforms

    @dataclasses.dataclass(frozen=True)
    class MetaworldInputs(transforms.DataTransformFn):
        """RLinf rlinf/models/embodiment/openpi/policies/metaworld_policy.py MetaworldInputs (same fields, same order)."""

        def __call__(self, data: dict) -> dict:
            img = np.asarray(data["observation/image"])
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            if img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            out = {"state": data["observation/state"],
                   "image": {"base_0_rgb": img, "left_wrist_0_rgb": np.zeros_like(img),
                             "right_wrist_0_rgb": np.zeros_like(img)},
                   "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.False_,
                                  "right_wrist_0_rgb": np.False_}}
            if "actions" in data:
                out["actions"] = data["actions"]
            if "prompt" in data:
                out["prompt"] = data["prompt"]
            return out

    @dataclasses.dataclass(frozen=True)
    class MetaworldOutputs(transforms.DataTransformFn):
        def __call__(self, data: dict) -> dict:
            return {"actions": np.asarray(data["actions"][:, :4])}

    return MetaworldInputs, MetaworldOutputs


def resolve(checkpoint):
    if checkpoint in CKPTS:
        checkpoint = CKPTS[checkpoint]
    if os.path.isdir(str(checkpoint)):
        return str(checkpoint)
    from huggingface_hub import snapshot_download
    return snapshot_download(checkpoint)


class Pi0MetaWorldBackbone(PolicyBackbone):
    kind = "pi0_metaworld"

    def __init__(self, checkpoint="pi0", pi05=None, device="cuda", num_steps=5):
        import openpi.models.pi0_config as pi0_config
        import openpi.transforms as T
        from openpi.models import model as _model
        from openpi.models import tokenizer as _tokenizer
        from openpi.models_pytorch import pi0_pytorch
        from openpi.policies import policy as _policy
        from openpi.shared import normalize as _normalize
        import safetensors.torch
        t0 = time.time()
        if pi05 is None:
            pi05 = checkpoint == "pi05" or "Pi05" in str(checkpoint)
        self.checkpoint_dir = resolve(checkpoint)
        cfg = (pi0_config.Pi0Config(pi05=True, action_horizon=5, discrete_state_input=False) if pi05
               else pi0_config.Pi0Config(action_horizon=5))
        self.model_config = cfg
        self.train_config = dataclasses.make_dataclass("Cfg", [("name", str), ("model", object)])(
            name="pi05_metaworld" if pi05 else "pi0_metaworld", model=cfg)
        model = pi0_pytorch.PI0Pytorch(config=cfg)
        missing, unexpected = safetensors.torch.load_model(model, os.path.join(self.checkpoint_dir, "model.safetensors"),
                                                           strict=False)
        self.missing_keys, self.unexpected_keys = sorted(missing), sorted(unexpected)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        norm = _normalize.load(pathlib.Path(self.checkpoint_dir) / ASSET_ID)
        q = cfg.model_type != _model.ModelType.PI0
        Inp, Out = _inputs_cls()
        tok = _tokenizer.PaligemmaTokenizer(cfg.max_token_len)
        if pi05:
            model_in = [T.InjectDefaultPrompt(None), T.ResizeImages(224, 224),
                        T.TokenizePrompt(tok, discrete_state_input=cfg.discrete_state_input),
                        T.PadStatesAndActions(cfg.action_dim)]
        else:
            model_in = [T.InjectDefaultPrompt(None), T.ResizeImages(224, 224), T.TokenizePrompt(tok),
                        T.PadStatesAndActions(cfg.action_dim)]
        self.policy = _policy.Policy(model, transforms=[T.InjectDefaultPrompt(None), Inp(), T.Normalize(norm, use_quantiles=q),
                                                        *model_in],
                                     output_transforms=[T.Unnormalize(norm, use_quantiles=q), Out()],
                                     sample_kwargs={"num_steps": int(num_steps)}, is_pytorch=True, pytorch_device=device)
        self.model = self.policy._model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.H = int(cfg.action_horizon)            # 5
        self.A = int(cfg.action_dim)                # 32, padded
        self.a = 4                                  # real action dims: delta xyz + gripper
        self.dim = self.H * self.a                  # 20
        self.dim_full = self.H * self.A
        self.real = np.concatenate([np.arange(self.a) + h * self.A for h in range(self.H)])
        self.num_steps = int(num_steps)
        self.clip = (-1e9, 1e9)
        self.layout = "chunk"
        self.img = None
        self.n_params = int(sum(p.numel() for p in self.model.parameters()))
        self.expert = self.model.paligemma_with_expert.gemma_expert
        self.load_secs = time.time() - t0
        self._Observation = _model.Observation
        self.use_quantiles = q

    def eager_sample_actions(self, observation, noise, num_steps=None):
        """openpi's sample_actions without the torch.compile wrapper (the reference for bit parity)."""
        fn = type(self.model).sample_actions
        return fn(self.model, self.device, observation, noise=noise, num_steps=int(num_steps or self.num_steps))

    @staticmethod
    def raw_observation(render_rgb, obs, prompt):
        """A Meta-World step -> the raw dict the policy reads: env.render() flipped on both axes, obs[:4], the prompt."""
        return {"observation/image": np.asarray(render_rgb)[::-1, ::-1], "observation/state": np.asarray(obs)[:4],
                "prompt": prompt}

    def describe(self):
        return (f"Pi0MetaWorldBackbone({self.train_config.name}: H={self.H}, A={self.A}, real={self.a}, dim={self.dim}, "
                f"n_params={self.n_params:,}, missing={len(self.missing_keys)}, unexpected={len(self.unexpected_keys)}, "
                f"quantiles={self.use_quantiles}, device={self.device}, load {self.load_secs:.0f}s)")
