"""SmolVLABackbone: LeRobot's SmolVLA (lerobot/smolvla_libero) as the same score field over action chunks as PolicyBackbone.

SmolVLA's action expert is trained by flow matching on the same path as openpi's pi0 (x_t = t * eps + (1 - t) * a, t = 1 noise,
target u = eps - a, Euler x <- x + dt * v with dt = -1/num_steps, a prefix KV cache of images + language + state), so every
identity in backbone.py holds and PolicyBackbone's sampler (sample, the guide, pad, eps, tweedie) runs unchanged; this class
supplies the three model-specific pieces: the conditioning (LeRobot's own pre-processor: tokenizer, newline, MEAN_STD state
normalisation), the prefix cache and velocity (VLAFlowMatching.embed_prefix / denoise_step), and the action normalisation
(MEAN_STD over the 7 LIBERO action dims, no state-relative block).

The LIBERO checkpoint was trained on the same 40 tasks (lerobot/libero, 1,693 episodes) as openpi's pi0 / pi0.5 LIBERO
checkpoints, with the same 8-D state, the same 7-D OSC delta actions and the same 180-degree image flip, so the client's
observations need no conversion beyond names and layout.  Run in its own venv (/coc/flash10/ksingara3/robot/smolvla/.venv:
lerobot 0.4.4, transformers 4.57), since openpi patches transformers.
"""
from __future__ import annotations

import dataclasses
import glob
import os
import time

import numpy as np
import torch

from .backbone import PolicyBackbone, Prefix

CKPT = "lerobot/smolvla_libero"


@dataclasses.dataclass
class SmolObservation:
    images: list
    img_masks: list
    lang_tokens: torch.Tensor
    lang_masks: torch.Tensor
    state: torch.Tensor                  # normalised and padded to max_state_dim, as the model sees it


class SmolVLABackbone(PolicyBackbone):
    kind = "smolvla"

    def __init__(self, checkpoint=None, device="cuda", num_steps=None):
        os.environ.setdefault("HF_HOME", "/coc/flash10/ksingara3/robot/data/hf/home")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from huggingface_hub import snapshot_download
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
        t0 = time.time()
        path = checkpoint or CKPT
        self.checkpoint_dir = path if os.path.isdir(path) else snapshot_download(path)
        self.policy = SmolVLAPolicy.from_pretrained(self.checkpoint_dir).to(device).eval()
        self.cfg = self.policy.config
        self.pre, _ = make_pre_post_processors(self.cfg, pretrained_path=self.checkpoint_dir,
                                               preprocessor_overrides={"device_processor": {"device": device}})
        self.model = self.policy.model
        for q in self.model.parameters():
            q.requires_grad_(False)
        self.device = device
        self.H = int(self.cfg.chunk_size)                          # 50
        self.A = int(self.cfg.max_action_dim)                      # 32, padded with zeros in training
        self.a = int(self.cfg.output_features["action"].shape[0])  # 7
        self.dim = self.H * self.a
        self.dim_full = self.H * self.A
        self.real = np.concatenate([np.arange(self.a) + h * self.A for h in range(self.H)])
        self.num_steps = int(num_steps or self.cfg.num_steps)
        self.clip = (-1e9, 1e9)
        self.layout = "chunk"
        self.img = None
        self.n_params = int(sum(p.numel() for p in self.model.parameters()))
        self.expert = self.model.vlm_with_expert.lm_expert
        from safetensors.numpy import load_file
        st = load_file(glob.glob(os.path.join(self.checkpoint_dir, "policy_postprocessor_step_*_unnormalizer_processor.safetensors"))[0])
        self.action_stats = {k: np.asarray(st[f"action.{k}"], np.float32) for k in ("mean", "std", "q01", "q99")}
        self.eps = 1e-8                                            # the checkpoint's normalizer eps
        self.load_secs = time.time() - t0

    # ── conditioning ────────────────────────────────────────────────────────
    @torch.no_grad()
    def observation(self, obs_dict):
        """A raw LIBERO observation dict (the openpi client's keys: observation/image and observation/wrist_image, uint8 HWC, already
        flipped 180 degrees; observation/state, 8-D; prompt) -> the model's inputs, through the checkpoint's own pre-processor."""
        img = lambda k: torch.from_numpy(np.ascontiguousarray(obs_dict[k])).permute(2, 0, 1)[None].float() / 255.0   # noqa: E731
        batch = {"observation.images.image": img("observation/image"), "observation.images.image2": img("observation/wrist_image"),
                 "observation.state": torch.as_tensor(np.asarray(obs_dict["observation/state"], np.float32))[None],
                 "task": [str(obs_dict["prompt"])]}
        b = self.pre(batch)
        images, img_masks = self.policy.prepare_images(b)
        return SmolObservation(images=images, img_masks=img_masks, lang_tokens=b["observation.language.tokens"],
                               lang_masks=b["observation.language.attention_mask"], state=self.policy.prepare_state(b))

    @torch.no_grad()
    def observations(self, obs_dicts):
        """A list of raw observation dicts -> one batched input (per-row conditioning), through the same pre-processor."""
        img = lambda k: torch.stack([torch.from_numpy(np.ascontiguousarray(o[k])).permute(2, 0, 1) for o in obs_dicts]).float() / 255.0  # noqa: E731
        batch = {"observation.images.image": img("observation/image"), "observation.images.image2": img("observation/wrist_image"),
                 "observation.state": torch.as_tensor(np.stack([np.asarray(o["observation/state"], np.float32) for o in obs_dicts])),
                 "task": [str(o["prompt"]) for o in obs_dicts]}
        b = self.pre(batch)
        images, img_masks = self.policy.prepare_images(b)
        return SmolObservation(images=images, img_masks=img_masks, lang_tokens=b["observation.language.tokens"],
                               lang_masks=b["observation.language.attention_mask"], state=self.policy.prepare_state(b))

    @torch.no_grad()
    def prefix(self, observation):
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
        m = self.model
        o = observation
        embs, pad, att = m.embed_prefix(o.images, o.img_masks, o.lang_tokens, o.lang_masks, state=o.state)
        att2d = make_att_2d_masks(pad, att)
        pos = torch.cumsum(pad, dim=1) - 1
        self._span = self.text_span(o)
        assert embs.shape[1] == self._span[0] + o.lang_tokens.shape[1] + 1, "text_span: the language slot is not where expected"
        try:
            _, kv = m.vlm_with_expert.forward(attention_mask=att2d, position_ids=pos, past_key_values=None, inputs_embeds=[embs, None],
                                              use_cache=self.cfg.use_cache, fill_kv_cache=True)
        finally:
            self._span = None
        return Prefix(state=o.state, pad_masks=pad, kv=kv, n=int(o.state.shape[0]))

    # ── the conditioning encoder's residual stream (backbone.PolicyBackbone.text_span / cond_hook) ─────────────
    _img_tok = None

    @torch.no_grad()
    def text_span(self, observation):
        """embed_prefix's sequence = per image ([start token] + SigLIP tokens after pixel shuffle + [end token]), the language tokens
        (right-padded to tokenizer_max_length), one state token."""
        m = self.model
        if self._img_tok is None:
            self._img_tok = int(m.vlm_with_expert.embed_image(observation.images[0][:1]).shape[1])
        sp = (m.global_image_start_token.numel() + m.image_end_token.numel()) if m.add_image_special_tokens else 0
        return len(observation.images) * (self._img_tok + sp), observation.lang_masks.sum(1).long()

    def cond_layers(self):
        return list(self.model.vlm_with_expert.get_vlm_model().text_model.layers)

    def cond_hook(self, fn):
        """SmolVLMWithExpertModel.forward computes each layer inline (it reads the layer's weights, never calls the module), so the
        hook wraps forward_attn_layer, which every layer of the prefix pass (fill_kv_cache, the VLM stream present) runs first; the
        new hidden replaces the stream's list entry, which the same layer's residual adds read too."""
        vwe = self.model.vlm_with_expert
        if "forward_attn_layer" not in vars(vwe):
            fns = self._cond_fns = []
            orig = vwe.forward_attn_layer

            def wrapped(model_layers, inputs_embeds, layer_idx, *a, **k):
                if fns and inputs_embeds[0] is not None and k.get("fill_kv_cache", True):
                    for f in list(fns):
                        out = f(int(layer_idx), inputs_embeds[0], self._span)
                        if out is not None:
                            inputs_embeds[0] = out
                return orig(model_layers, inputs_embeds, layer_idx, *a, **k)
            vwe.forward_attn_layer = wrapped
        fns = self._cond_fns
        fns.append(fn)

        class _Handle:
            def remove(self_):
                if fn in fns:
                    fns.remove(fn)
        return [_Handle()]

    def _expand(self, prefix, n):
        if prefix.n == n:
            return prefix
        if prefix.n != 1:
            raise ValueError("prefix batch must be 1 or n")
        kv = {i: {k: t.expand(n, *t.shape[1:]) for k, t in layer.items()} for i, layer in prefix.kv.items()}
        return Prefix(state=prefix.state.expand(n, *prefix.state.shape[1:]), pad_masks=prefix.pad_masks.expand(n, *prefix.pad_masks.shape[1:]),
                      kv=kv, n=n)

    # ── the field ───────────────────────────────────────────────────────────
    def v_full(self, x_full, t, prefix, gate=None):
        from .backbone import _tt
        x = torch.as_tensor(x_full, dtype=torch.float32, device=self.device).reshape(-1, self.H, self.A)
        n = len(x)
        pre = self._expand(prefix, n)
        tt = _tt(t, self.device).reshape(-1)
        tt = tt.expand(n) if tt.numel() == 1 else tt
        import contextlib
        with (gate() if gate is not None else contextlib.nullcontext()):
            vt = self.model.denoise_step(prefix_pad_masks=pre.pad_masks, past_key_values=pre.kv, x_t=x, timestep=tt)
        return vt.reshape(n, self.dim_full).float()

    # ── normalisation ───────────────────────────────────────────────────────
    def unnormalize(self, rows, state=None):
        """real rows [n, dim] -> LIBERO actions [n, H, 7]: the checkpoint's MEAN_STD unnormaliser (x * (std + eps) + mean)."""
        R = np.asarray(rows, np.float32).reshape(-1, self.H, self.a)
        return R * (self.action_stats["std"] + self.eps) + self.action_stats["mean"]

    def describe(self):
        return (f"SmolVLABackbone({self.checkpoint_dir}: H={self.H}, A={self.A}, real={self.a}, dim={self.dim}, steps={self.num_steps}, "
                f"n_params={self.n_params:,}, load {self.load_secs:.1f}s)")
