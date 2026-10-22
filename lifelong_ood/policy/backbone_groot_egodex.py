"""GrootEgoDexBackbone: GR00T N1.5 fine-tuned on EgoDex human hand motion (plan X3; docs/EGODEX.md), as the same score field over action
chunks as PolicyBackbone / GrootBackbone.

The model.  nvidia/GR00T-N1.5-3B (Eagle-2.5: SigLIP-2 + 12 Qwen3-1.7B layers; the 16-layer cross-attention DiT action head), with
ONE architectural change: the action width 32 -> 64 (EgoDex's 48-D hand frame does not fit 32).  The two layers whose shape
depends on it, action_encoder.W1 (64 -> 1536) and action_decoder.layer2 (1024 -> 64), are re-initialised exactly as GR00T
initialises CategorySpecificLinear (W ~ 0.02 N(0, 1), b = 0); every other tensor is the base checkpoint's.  Embodiment
"new_embodiment" (id 31).  Fine-tuning (experiments/x3_egodex_train.py) tunes the action head (projector, vlln + VL
self-attention, DiT) and freezes Eagle (LLM + vision, kept in bf16).  A fine-tuned checkpoint directory holds
head.safetensors (the action head's state dict), norm.json (the quantile normaliser) and egodex_config.json.

Observation (one dict per row): "image" uint8 [224, 224, 3] (the egocentric frame), "state" float [48] (the current hand frame
in the current camera frame, x3_egodex_data.Episode.frame48 layout), "prompt" str.  State and actions are normalised by
EgoDexNorm (per-dimension 1st / 99th percentiles -> [-1, 1], clipped; the 6D rotation dims pass through), padded with zeros to 64.
Actions: H = 16 steps at stride 2 (frames t + 2, ..., t + 32 at 30 Hz), A = 64 padded, a = 48 real.

Everything below the observation (prefix, v_full, flow_loss, text_span, cond_layers, cond_hook, sample) is GrootBackbone's: the
flow convention (tau = 1 - t, v = -pred, 1000 buckets, Euler K = 4) is the base model's, unchanged by the fine-tune.
Run in the GR00T venv (/coc/flash10/ksingara3/robot/groot/.venv; h5py is not needed here).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from .backbone_groot import GObservation, GrootBackbone, _sdpa_for_flash

BASE = "nvidia/GR00T-N1.5-3B"
EMBODIMENT = 31                    # "new_embodiment" in GR00T's EMBODIMENT_TAG_MAPPING
A_FULL, A_REAL, H, STRIDE = 64, 48, 16, 2
ROT = np.array([o + j for o in (3, 27) for j in range(6)])          # the 6D rotation dims of the 48-D frame
POS = np.setdiff1d(np.arange(A_REAL), ROT)


def _env():
    os.environ.setdefault("HF_HOME", "/coc/flash10/ksingara3/robot/data/hf/home")
    os.environ.setdefault("HF_LEROBOT_HOME", "/coc/flash10/ksingara3/robot/data/hf/lerobot")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


class EgoDexNorm:
    """The 48-D frame <-> [-1, 1]: per position dim, 2 (x - q01) / (q99 - q01) - 1 clipped to [-1.5, 1.5]; rotation dims unchanged.
    (Same interface as backbone_groot.GrootActionCodec: normalize / unnormalize over the last axis.)"""
    quantile = True
    delta = 0
    per_step_state = False

    def __init__(self, q01, q99, clip=1.5):
        self.q01, self.q99 = np.asarray(q01, np.float32), np.asarray(q99, np.float32)
        self.clip = float(clip)
        self.den = np.where(self.q99 - self.q01 > 1e-6, self.q99 - self.q01, 1.0).astype(np.float32)

    def normalize(self, x, state=None):
        x = np.array(x, np.float32, copy=True)
        x[..., POS] = np.clip(2.0 * (x[..., POS] - self.q01[POS]) / self.den[POS] - 1.0, -self.clip, self.clip)
        return x

    def unnormalize(self, x):
        x = np.array(x, np.float32, copy=True)
        x[..., POS] = (x[..., POS] + 1.0) * 0.5 * self.den[POS] + self.q01[POS]
        return x

    def to_json(self):
        return dict(q01=self.q01.tolist(), q99=self.q99.tolist(), clip=self.clip, pos_dims=POS.tolist(), rot_dims=ROT.tolist())

    @classmethod
    def from_json(cls, d):
        return cls(d["q01"], d["q99"], d.get("clip", 1.5))


def eagle_processor():
    _env()
    from lerobot.policies.groot.processor_groot import _build_eagle_processor
    return _build_eagle_processor()


def eagle_content(proc, image, instruction):
    """One row's Eagle content exactly as LeRobot's GrootEagleEncodeStep builds it (one view, the instruction as str([text]))."""
    from PIL import Image
    conv = [{"role": "user", "content": [{"type": "image", "image": Image.fromarray(np.asarray(image, np.uint8))},
                                         {"type": "text", "text": str([str(instruction)])}]}]
    text_list = [proc.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)]
    img_inputs, _ = proc.process_vision_info(conv)
    return dict(text_list=text_list, image_inputs=img_inputs)


def eagle_batch(proc, contents):
    """GR00T's collate: eagle_* tensors for a list of contents (input ids left-padded)."""
    text, imgs = [], []
    for c in contents:
        text += c["text_list"]
        imgs += c["image_inputs"]
    ins = proc(text=text, images=imgs, images_kwargs={"min_dynamic_tiles": 1, "max_dynamic_tiles": 1, "use_thumbnail": False},
               return_tensors="pt", padding=True)
    return {"eagle_" + k: v for k, v in ins.items()}


def pad_state(s48):
    """[B, 48] normalised -> state [B, 1, 64], state_mask [B, 1, 64]."""
    s = torch.as_tensor(np.asarray(s48, np.float32))
    B = s.shape[0]
    st = torch.zeros(B, 1, A_FULL)
    st[:, 0, :A_REAL] = s
    m = torch.zeros(B, 1, A_FULL, dtype=torch.bool)
    m[:, :, :A_REAL] = True
    return st, m


def build_model(head=None, base=BASE, device="cuda", trainable=False):
    """GR00T N1.5 base with the 64-wide action layers (re-initialised), Eagle frozen in bf16; `head`: a head.safetensors to load.
    -> (GR00TN15, info)."""
    _env()
    from huggingface_hub import snapshot_download
    from lerobot.policies.groot.action_head.flow_matching_action_head import CategorySpecificLinear
    from lerobot.policies.groot.groot_n1 import GR00TN15
    path = base if os.path.isdir(base) else snapshot_download(base)
    with _sdpa_for_flash():
        gm = GR00TN15.from_pretrained(path, tune_llm=False, tune_visual=False, tune_projector=True, tune_diffusion_model=True)
    h = gm.action_head
    E = h.config.max_num_embodiments
    torch.manual_seed(0)
    h.action_encoder.W1 = CategorySpecificLinear(E, A_FULL, h.input_embedding_dim)
    h.action_decoder.layer2 = CategorySpecificLinear(E, h.hidden_size, A_FULL)
    h.action_dim = A_FULL
    h.config.action_dim = A_FULL
    h.config.max_action_dim = A_FULL
    gm.action_dim = A_FULL
    gm.config.action_dim = A_FULL
    gm.config.action_head_cfg["action_dim"] = A_FULL
    gm.compute_dtype = "bfloat16"
    gm.config.compute_dtype = "bfloat16"
    gm.backbone.to(torch.bfloat16)
    for p in gm.backbone.parameters():
        p.requires_grad_(False)
    info = dict(base=path, reinit=["action_head.action_encoder.W1", "action_head.action_decoder.layer2"])
    if head is not None:
        from safetensors.torch import load_file
        sd = load_file(head)
        h.load_state_dict(sd, strict=True)
        info.update(head=head, head_tensors=len(sd))
    h.set_trainable_parameters(tune_projector=True, tune_diffusion_model=True)
    if not trainable:
        for p in gm.parameters():
            p.requires_grad_(False)
    gm = gm.to(device)
    gm.backbone.eval()
    return gm, info


class GrootEgoDexBackbone(GrootBackbone):
    kind = "groot_egodex"

    def __init__(self, checkpoint, device="cuda", num_steps=None):
        t0 = time.time()
        self.checkpoint_dir = checkpoint
        self.ego_cfg = json.load(open(os.path.join(checkpoint, "egodex_config.json")))
        self.gm, self.build_info = build_model(os.path.join(checkpoint, "head.safetensors"), base=self.ego_cfg.get("base", BASE),
                                               device=device)
        self.gm.eval()
        self.model = self.gm
        self.head = self.gm.action_head
        self.policy = None
        self.device = device
        self.H = int(self.head.action_horizon)
        self.A = int(self.head.action_dim)
        assert (self.H, self.A) == (H, A_FULL), (self.H, self.A)
        self.a = A_REAL
        self.dim = self.H * self.a
        self.dim_full = self.H * self.A
        self.real = np.concatenate([np.arange(self.a) + h * self.A for h in range(self.H)])
        self.num_steps = int(num_steps or self.head.num_inference_timesteps)
        self.clip = (-1e9, 1e9)
        self.layout = "chunk"
        self.img = None
        self.bf16 = True
        self.n_params = int(sum(p.numel() for p in self.gm.parameters()))
        self.expert = self.head.model
        self._codec = EgoDexNorm.from_json(json.load(open(os.path.join(checkpoint, "norm.json"))))
        self._proc = eagle_processor()
        self._tok = self._proc.tokenizer
        self.weights_check = self.build_info
        self.load_secs = time.time() - t0

    def action_codec(self):
        return self._codec

    @torch.no_grad()
    def raw_inputs(self, obs_dicts):
        """The model's own input dict (eagle_*, state, state_mask, embodiment_id) for a list of observation dicts."""
        eagle = eagle_batch(self._proc, [eagle_content(self._proc, o["image"], o["prompt"]) for o in obs_dicts])
        st, sm = pad_state(np.stack([self._codec.normalize(np.asarray(o["state"], np.float32)) for o in obs_dicts]))
        return dict(eagle, state=st, state_mask=sm, embodiment_id=torch.full((len(obs_dicts),), EMBODIMENT, dtype=torch.long))

    @torch.no_grad()
    def observations(self, obs_dicts):
        r = self.raw_inputs(obs_dicts)
        eagle = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in r.items() if k.startswith("eagle_")}
        starts, counts = [], []
        for b, o in enumerate(obs_dicts):
            idx = self._instr(eagle["eagle_input_ids"][b].tolist(), str(o["prompt"]))
            starts.append(idx[0])
            counts.append(len(idx))
        span = (starts[0] if len(set(starts)) == 1 else None, torch.as_tensor(counts, dtype=torch.long))
        return GObservation(eagle=eagle, state=r["state"].to(self.device), embodiment_id=r["embodiment_id"].to(self.device),
                            span=(span, starts))

    def text_span(self, observation):
        """GrootBackbone's, except that a batch whose rows place the instruction at different offsets (mixed instructions, e.g. an
        evaluation batch) gets None instead of an error; cond_hook functions then receive span=None."""
        (off, nv), starts = observation.span
        return None if off is None else (off, nv)

    def unnormalize(self, rows, state=None):
        """real rows [n, H * 48] -> [n, H, 48] hand frames in the observation's camera frame (metres; 6D rotations)."""
        return self._codec.unnormalize(np.asarray(rows, np.float32).reshape(-1, self.H, self.a))

    def describe(self):
        return (f"GrootEgoDexBackbone({self.checkpoint_dir}: H={self.H} at stride {STRIDE}, A={self.A}, real={self.a}, "
                f"steps={self.num_steps}, n_params={self.n_params:,}, expert={sum(p.numel() for p in self.expert.parameters()):,}, "
                f"load {self.load_secs:.1f}s)")
