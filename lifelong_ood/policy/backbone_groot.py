"""GrootBackbone: NVIDIA GR00T N1.5 (LeRobot 0.4.4's `groot` policy) as the same score field over action chunks as PolicyBackbone.

The checkpoint.  aractingi/lerobot-groot-ft-libero: GR00T-N1.5-3B (nvidia/GR00T-N1.5-3B) fine-tuned by LeRobot on HuggingFaceVLA/libero
(1,693 episodes, 40 tasks: the same four standard suites, 8-D state, 7-D OSC delta actions and 180-degree flipped 256 images as
physical-intelligence/libero, which pi0 / pi0.5 / SmolVLA saw), 50k steps, batch 32, projector + DiT tuned, LLM and vision frozen
(train_config.json).  So it matches our protocol as SmolVLA does; the client's observations need no conversion beyond names.
LeRobot's GR00T N1.5 page reports, for its 30k-step LIBERO fine-tune, Spatial 82 / Object 99 / Long 82 % (the checkpoint is not
named there).  LeRobot >= 0.5.2 supports only N1.7, whose public LIBERO checkpoints (nvidia/gr00t17-lerobot-libero_*) are one per
suite (10 tasks each), so they do not match the 40-task protocol.

Conditioning: Eagle-2.5 (SigLIP-2 + Qwen3-1.7B truncated to select_layer = 12 layers) over [both views' image tokens, the chat-
templated instruction "['<instruction>']"], then the action head's vlln + 4-layer self-attention (process_backbone_output) and its
state encoder (min-max normalised 8-D state, padded to 64; embodiment "new_embodiment" = 31): all cached by `prefix`.  The action
expert is the 16-layer cross-attention DiT (action_head.model).

The flow.  GR00T's time runs the other way: tau = 1 - t (tau = 0 noise), x_tau = (1 - tau) eps + tau a, and the DiT predicts
a - eps at tau discretised into 1000 buckets (int(tau * 1000)); so v (the repo's eps - a at t = 1 - tau) = -pred.  get_action is
Euler from tau = 0 in num_inference_timesteps = 4 steps -- exactly PolicyBackbone.sample with K = 4 (t = 1, 0.75, 0.5, 0.25), so
the base sampler and every identity hold unchanged.  flow_loss: GR00T's MSE on the velocity over the real action dims (its
action_mask); the generic loss also averages over the 25 padded dims, which GR00T never trains.

Actions: H = 16 (the model's action horizon; LeRobot's chunk_size 50 is truncated to 16), A = 32 padded, a = 7 real; min-max
normalised to [-1, 1] with the dataset's min / max (GrootPackInputsStep), so `unnormalize` / `action_codec` invert that.
Attention: LeRobot's Eagle asks for flash_attention_2 (SigLIP and Qwen); flash-attn is not installed, so the request is mapped to
PyTorch SDPA during construction (the same attention, another kernel).  Batched observations are encoded row by row through the
checkpoint's own pre-processor (its pack step keeps only the batch's FIRST instruction) and left-padded together, as its collate does.
Run in its own venv (/coc/flash10/ksingara3/robot/groot/.venv: the SmolVLA venv's pins + peft 0.17.1, which the vendored Eagle imports).
"""
from __future__ import annotations

import contextlib
import dataclasses
import glob
import os
import time

import numpy as np
import torch

from .backbone import PolicyBackbone, Prefix, _tt

CKPT = "aractingi/lerobot-groot-ft-libero"
BUCKETS = 1000


@contextlib.contextmanager
def _sdpa_for_flash():
    """While building the model: an attn_implementation of flash_attention_* falls back to sdpa when flash-attn is missing."""
    from transformers import modeling_utils as mu
    from transformers.utils import is_flash_attn_2_available
    orig = mu.PreTrainedModel._check_and_adjust_attn_implementation

    def check(self, attn_implementation, is_init_check=False):
        if attn_implementation is not None and attn_implementation.startswith("flash_attention") and not is_flash_attn_2_available():
            attn_implementation = "sdpa"
        return orig(self, attn_implementation, is_init_check)
    mu.PreTrainedModel._check_and_adjust_attn_implementation = check
    try:
        yield
    finally:
        mu.PreTrainedModel._check_and_adjust_attn_implementation = orig


class GrootActionCodec:
    """data.Norm's interface for GR00T: the dataset's 7-D actions <-> min-max [-1, 1] (GrootPackInputsStep / ActionUnpackUnnormalize)."""
    quantile = False
    delta = 0
    per_step_state = False

    def __init__(self, lo, hi):
        self.lo, self.hi = np.asarray(lo, np.float32), np.asarray(hi, np.float32)
        self.den = np.where(self.hi - self.lo != 0, self.hi - self.lo, 1.0).astype(np.float32)

    def normalize(self, x, state=None):
        x = np.asarray(x, np.float32)
        return np.where(self.hi != self.lo, 2.0 * (x - self.lo) / self.den - 1.0, 0.0).astype(np.float32)

    def unnormalize(self, x):
        x = np.asarray(x, np.float32)
        return np.where(self.hi != self.lo, (x + 1.0) * 0.5 * self.den + self.lo, self.lo).astype(np.float32)


@dataclasses.dataclass
class GObservation:
    eagle: dict                          # eagle_* tensors (input_ids / attention_mask left-padded), on the device
    state: torch.Tensor                  # [B, 1, 64] min-max normalised, padded
    embodiment_id: torch.Tensor          # [B]
    span: tuple                          # (offset of the instruction's first token, instruction tokens per row [B])


class GrootBackbone(PolicyBackbone):
    kind = "groot"

    def __init__(self, checkpoint=None, device="cuda", num_steps=None):
        os.environ.setdefault("HF_HOME", "/coc/flash10/ksingara3/robot/data/hf/home")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from huggingface_hub import snapshot_download
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.groot.modeling_groot import GrootPolicy
        t0 = time.time()
        path = checkpoint or CKPT
        self.checkpoint_dir = path if os.path.isdir(path) else snapshot_download(path)
        with _sdpa_for_flash():
            # strict=False: safetensors' load_model drops one name of each tied pair (the Qwen3 embed_tokens / lm_head), so the
            # checkpoint's embed_tokens counts as "unexpected"; _verify_weights then copies such tensors and checks every one
            self.policy = GrootPolicy.from_pretrained(self.checkpoint_dir, strict=False).to(device).eval()
        self.weights_check = self._verify_weights(os.path.join(self.checkpoint_dir, "model.safetensors"))
        self.cfg = self.policy.config
        self.pre, _ = make_pre_post_processors(self.cfg, pretrained_path=self.checkpoint_dir,
                                               preprocessor_overrides={"device_processor": {"device": device}})
        self.gm = self.policy._groot_model
        self.model = self.gm
        self.head = self.gm.action_head
        for q in self.gm.parameters():
            q.requires_grad_(False)
        self.device = device
        self.H = int(self.head.action_horizon)                     # 16
        self.A = int(self.head.action_dim)                         # 32 (padded)
        self.a = int(self.cfg.output_features["action"].shape[0])  # 7
        self.dim = self.H * self.a
        self.dim_full = self.H * self.A
        self.real = np.concatenate([np.arange(self.a) + h * self.A for h in range(self.H)])
        self.num_steps = int(num_steps or self.head.num_inference_timesteps)   # 4
        self.clip = (-1e9, 1e9)
        self.layout = "chunk"
        self.img = None
        self.bf16 = bool(self.cfg.use_bf16)
        self.n_params = int(sum(p.numel() for p in self.gm.parameters()))
        self.expert = self.head.model                              # the cross-attention DiT (LoRA goes here)
        from safetensors.numpy import load_file
        st = load_file(glob.glob(os.path.join(self.checkpoint_dir, "policy_postprocessor_step_*_groot_action_unpack_unnormalize*.safetensors"))[0])
        self.action_stats = {k: np.asarray(st[f"action.{k}"], np.float32) for k in ("min", "max", "mean", "std", "q01", "q99")}
        self._codec = GrootActionCodec(self.action_stats["min"], self.action_stats["max"])
        self._tok = next(s for s in self.pre.steps if type(s).__name__ == "GrootEagleEncodeStep").proc.tokenizer
        self.load_secs = time.time() - t0

    def action_codec(self):
        return self._codec

    @torch.no_grad()
    def _verify_weights(self, path):
        """Every checkpoint tensor equals the model's (copying those the tied-name dedup skipped).  -> dict of counts."""
        from safetensors import safe_open
        sd = self.policy.state_dict()
        copied, absent, bad = [], [], []
        with safe_open(path, framework="pt") as f:
            keys = list(f.keys())
            for k in keys:
                if k not in sd:
                    absent.append(k)
                    continue
                w = f.get_tensor(k).to(sd[k].device, sd[k].dtype)
                if not torch.equal(sd[k], w):
                    sd[k].copy_(w)
                    copied.append(k)
                    if not torch.equal(sd[k], w):
                        bad.append(k)
        assert not absent and not bad, f"groot weights: absent {absent[:5]} mismatched {bad[:5]}"
        not_in_file = [k for k in sd if k not in set(keys)]
        return dict(file_tensors=len(keys), copied=copied, model_tensors_not_in_file=len(not_in_file), not_in_file_head=not_in_file[:5])

    def _ac(self):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16)

    # ── conditioning ────────────────────────────────────────────────────────
    def _one(self, o):
        img = lambda k: torch.from_numpy(np.ascontiguousarray(o[k])).permute(2, 0, 1)[None].float() / 255.0   # noqa: E731
        return self.pre({"observation.images.image": img("observation/image"), "observation.images.image2": img("observation/wrist_image"),
                         "observation.state": torch.as_tensor(np.asarray(o["observation/state"], np.float32))[None],
                         "task": [str(o["prompt"])]})

    def _instr(self, ids, text):
        """Token indices of `text` inside one row of input ids (per-token decoding; the prompts are ASCII)."""
        pieces = [self._tok.decode([int(i)]) for i in ids]
        pos, c = [], 0
        for p in pieces:
            pos.append((c, c + len(p)))
            c += len(p)
        full = "".join(pieces)
        c0 = full.rfind(text)
        assert c0 >= 0, f"groot text_span: instruction not found in the prompt: {text!r}"
        c1 = c0 + len(text)
        return [j for j, (a, b) in enumerate(pos) if b > c0 and a < c1 and b > a]

    @torch.no_grad()
    def observations(self, obs_dicts):
        """A list of raw observation dicts (the openpi client's keys; both views flipped as the dataset's; 8-D state; prompt) -> one
        batched input: each row through the checkpoint's own pre-processor, input ids left-padded together (its collate's padding)."""
        rows = [self._one(o) for o in obs_dicts]
        pad = self._tok.pad_token_id if self._tok.pad_token_id is not None else 0
        L = max(int(r["eagle_input_ids"].shape[1]) for r in rows)
        eagle = {}
        for k in rows[0]:
            if not k.startswith("eagle_"):
                continue
            vals = [r[k] for r in rows]
            if k in ("eagle_input_ids", "eagle_attention_mask"):
                fill = pad if k == "eagle_input_ids" else 0
                vals = [torch.cat([torch.full((v.shape[0], L - v.shape[1]), fill, dtype=v.dtype, device=v.device), v], 1) for v in vals]
            eagle[k] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else sum((list(v) for v in vals), [])
        starts, counts = [], []
        for b, o in enumerate(obs_dicts):
            idx = self._instr(eagle["eagle_input_ids"][b].tolist(), str(o["prompt"]))
            starts.append(idx[0])
            counts.append(len(idx))
        span = (starts[0] if len(set(starts)) == 1 else None, torch.as_tensor(counts, dtype=torch.long))
        return GObservation(eagle=eagle, state=torch.cat([r["state"] for r in rows], 0).to(self.device),
                            embodiment_id=torch.cat([r["embodiment_id"] for r in rows], 0).to(self.device), span=(span, starts))

    @torch.no_grad()
    def observation(self, obs_dict):
        return self.observations([obs_dict])

    @torch.no_grad()
    def prefix(self, observation):
        """Eagle (12 Qwen3 layers) + the head's vlln / self-attention + the state encoder, once per observation.
        kv = dict(vl, mask, sf, emb)."""
        o = observation
        self._span = self.text_span(o)
        dt = torch.bfloat16 if self.bf16 else torch.float32
        ins = {k: (v.to(self.device, dtype=dt) if torch.is_floating_point(v) else v.to(self.device)) if torch.is_tensor(v) else v
               for k, v in o.eagle.items()}
        try:
            with self._ac():
                bo = self.gm.backbone(self.gm.backbone.prepare_input(dict(ins)))
                bo = self.head.process_backbone_output(bo)
                sf = self.head.state_encoder(o.state.to(dt), o.embodiment_id)
        finally:
            self._span = None
        n = int(o.state.shape[0])
        kv = dict(vl=bo["backbone_features"], mask=bo["backbone_attention_mask"], sf=sf, emb=o.embodiment_id)
        return Prefix(state=o.state, pad_masks=torch.ones((n, 1), dtype=torch.bool, device=self.device), kv=kv, n=n)

    # ── the conditioning encoder's residual stream (backbone.PolicyBackbone.text_span / cond_hook) ─────────────
    # Eagle's Qwen3 reads [chat template, the two views' image tokens, "['<instruction>']", the assistant header], left-padded
    # (the image tokens of both views replace <IMG_CONTEXT> placeholders); hidden_states[12] is the conditioning, so the 12
    # retained Qwen3 layers are the encoder.  The instruction's offset is per row under left padding: text_span requires one
    # offset per batch (rows of one instruction; TLI's identify and serving satisfy it).  cond_hook is the base class's.
    def text_span(self, observation):
        (off, nv), starts = observation.span
        assert off is not None, f"groot text_span: rows place the instruction at different offsets {starts}"
        return off, nv

    def cond_layers(self):
        return self.gm.backbone.eagle_model.language_model.model.layers

    def _expand(self, prefix, n):
        if prefix.n == n:
            return prefix
        if prefix.n != 1:
            raise ValueError("prefix batch must be 1 or n")
        e = lambda t: t.expand(n, *t.shape[1:])                    # noqa: E731
        return Prefix(state=e(prefix.state), pad_masks=e(prefix.pad_masks), kv={k: e(v) for k, v in prefix.kv.items()}, n=n)

    def repeat(self, prefix, k):
        r = lambda t: t.repeat_interleave(int(k), dim=0)           # noqa: E731
        return Prefix(state=r(prefix.state), pad_masks=r(prefix.pad_masks), kv={kk: r(v) for kk, v in prefix.kv.items()},
                      n=prefix.n * int(k))

    # ── the field ───────────────────────────────────────────────────────────
    def v_full(self, x_full, t, prefix, gate=None):
        """v(x_t, t) = -pred(x, tau = 1 - t) on FULL [n, H, A] inputs (FlowmatchingActionHead.get_action's step).  -> [n, H * A]."""
        x = torch.as_tensor(x_full, dtype=torch.float32, device=self.device).reshape(-1, self.H, self.A)
        n = len(x)
        pre = self._expand(prefix, n)
        tt = _tt(t, self.device).reshape(-1)
        tt = tt.expand(n) if tt.numel() == 1 else tt
        td = torch.floor((1.0 - tt) * BUCKETS + 1e-3).clamp(0, BUCKETS - 1).long()
        h = self.head
        kv = pre.kv
        with self._ac(), (gate() if gate is not None else contextlib.nullcontext()):
            af = h.action_encoder(x.to(kv["vl"].dtype), td, kv["emb"])
            if h.config.add_pos_embed:
                af = af + h.position_embedding(torch.arange(af.shape[1], dtype=torch.long, device=x.device)).unsqueeze(0)
            fut = h.future_tokens.weight.unsqueeze(0).expand(n, -1, -1)
            sa = torch.cat((kv["sf"], fut, af), dim=1)
            out = h.model(hidden_states=sa, encoder_hidden_states=kv["vl"], timestep=td)   # the DiT ignores the VL mask (as in training)
            pred = h.action_decoder(out, kv["emb"])[:, -self.H:]
        return (-pred.float()).reshape(n, self.dim_full)

    def flow_loss(self, v, u, xt, t):
        """GR00T's objective: MSE on the velocity over the real action dims only (its action_mask)."""
        n = len(v)
        d = (v.reshape(n, self.H, self.A) - u.reshape(n, self.H, self.A))[..., : self.a]
        return (d ** 2).mean()

    # ── normalisation ───────────────────────────────────────────────────────
    def unnormalize(self, rows, state=None):
        """real rows [n, dim] -> LIBERO delta actions [n, H, 7]: the checkpoint's inverse min-max (GrootActionUnpackUnnormalizeStep)."""
        return self._codec.unnormalize(np.asarray(rows, np.float32).reshape(-1, self.H, self.a))

    def describe(self):
        return (f"GrootBackbone({self.checkpoint_dir}: H={self.H}, A={self.A}, real={self.a}, dim={self.dim}, steps={self.num_steps}, "
                f"n_params={self.n_params:,}, expert={sum(p.numel() for p in self.expert.parameters()):,}, load {self.load_secs:.1f}s)")
