"""FlowerBackbone: FLOWER (Reuss et al., arXiv 2509.04996, CoRL 2025; github.com/intuitive-robots/flower_vla_calvin at acb32ee) as the
same score field over action chunks as PolicyBackbone.

The checkpoints.  mbreuss/flower_libero_{spatial,object,goal,10} are each trained on ONE suite's 10 tasks (config: libero_benchmark;
the original LIBERO hdf5 data), mbreuss/flower_libero_90 on LIBERO-90.  No public FLOWER checkpoint was trained on the 40 tasks of
the four standard suites (pi0.5's training set), so a FLOWER arm sees a smaller library: flower_libero_goal knows LIBERO-Goal's
10 tasks (training-task indices 10-19 of physical-intelligence/libero), and a composition whose source tasks fall outside its
suite is outside its training set.  BACKBONE_CKPT picks the checkpoint (default flower_libero_goal).

The authors' LIBERO conventions (flower/rollout/libero_rollout.py, conf/datamodule/transforms/libero_transforms.yaml):
  * images: agentview_image and robot0_eye_in_hand_image AS RENDERED (no flip), env render 224, Resize(112, antialias), / 255,
    CLIP mean / std; no proprioception (use_proprio = False);
  * prompt: "Agent Type: 1-arm Franka Panda, Action Space: Delta End-Effector,  Task Instruction: <instruction>" (models/utils.py
    generate_policy_prompt, style "minimal"), BART-tokenized (max 77, padded to the batch's longest);
  * Florence-2-large's encoder (decoder deleted) reads [agent-view DaViT tokens, wrist-view DaViT tokens, the <Flow> token, the
    prompt tokens] with an all-ones mask; the DiT (18 FlowBlocks, cross-attention on those features) is the action expert;
  * actions: 7-D OSC DELTA actions in [-1, 1] (raw LIBERO units, no normaliser), chunk 10, executed whole (multistep 10);
  * 5 dummy settle steps, max 520 steps.
The repo's client sends openpi's format (both views flipped 180 degrees, resized to 224), so `observations` un-flips both views
and resizes them to 112; it replans every 5 actions (the client's protocol for every backbone) and settles 10 steps.

The flow.  FLOWER's rectified flow is the repo's path: z_t = (1 - t) a + t eps (t = 1 noise), target v = eps - a, Euler
z <- z - dt v from t = 1 in num_sampling_steps = 4 steps, then clamp(-1, 1) (sample_actions); so PolicyBackbone's sampler and
identities hold unchanged with K = 4 and `unnormalize` clamps.  flow_loss: the generic ||v - u||^2 (rf_loss).

Two checkpoint fixes (`_fix_weights`): the HF export renamed vlm.language_model.model.shared / final_logits_bias, which the
authors' loader does not map back (Florence-2's base token embeddings would replace the fine-tuned ones), so they are loaded here;
and the <Flow> token's prompt embedding is not a parameter (`_create_prompt_embed` reads the freshly resized, random row at
construction, so every construction draws another one), so it is set to the checkpoint's row of the token table.
Run in its own venv (/coc/flash10/ksingara3/robot/flower/.venv: torch 2.10, transformers 4.46.3 for Florence-2's remote code,
pytorch-lightning 2.4, hydra 1.3, timm 1.0.11, einops_exts); the FLOWER code is read from robot/ext/flower_vla_calvin.
"""
from __future__ import annotations

import dataclasses
import os
import sys
import time

import numpy as np
import torch

from .backbone import PolicyBackbone, Prefix, _tt

FLOWER_SRC = "/coc/flash10/ksingara3/robot/ext/flower_vla_calvin"
CKPT = "mbreuss/flower_libero_goal"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMG = 112


class FlowerActionCodec:
    """data.Norm's interface: FLOWER's actions are the raw LIBERO delta actions in [-1, 1] (no normaliser)."""
    quantile = False
    delta = 0
    per_step_state = False

    def normalize(self, x, state=None):
        return np.clip(np.asarray(x, np.float32), -1.0, 1.0)

    def unnormalize(self, x):
        return np.clip(np.asarray(x, np.float32), -1.0, 1.0)


@dataclasses.dataclass
class FObservation:
    static: torch.Tensor                 # [B, 1, 3, 112, 112], CLIP-normalised, the env's orientation
    gripper: torch.Tensor
    texts: list
    state: torch.Tensor                  # the client's 8-D state (unused by the model; the interface's state slot)


class FlowerBackbone(PolicyBackbone):
    kind = "flower"

    def __init__(self, checkpoint=None, device="cuda", num_steps=None):
        os.environ.setdefault("HF_HOME", "/coc/flash10/ksingara3/robot/data/hf/home")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("WANDB_MODE", "disabled")
        if FLOWER_SRC not in sys.path:
            sys.path.insert(0, FLOWER_SRC)
        from huggingface_hub import snapshot_download
        from omegaconf import OmegaConf
        t0 = time.time()
        path = checkpoint or CKPT
        self.checkpoint_dir = path if os.path.isdir(path) else snapshot_download(path)
        cfg_file = next(os.path.join(self.checkpoint_dir, f) for f in ("config.yaml", "config.json")
                        if os.path.exists(os.path.join(self.checkpoint_dir, f)))
        cfg = OmegaConf.load(cfg_file)                     # the repo's HF format: the training run's hydra config (YAML)
        mc = OmegaConf.to_container(cfg.model, resolve=True)
        for k in ("_target_", "_recursive_"):
            mc.pop(k, None)
        mc.update(load_pretrained=True, pretrained_model_path=os.path.join(self.checkpoint_dir, "model.safetensors"),
                  optimizer=None, num_sampling_steps=int(num_steps or 4))   # eval_libero.yaml: num_sampling_steps 4
        self.suite = str(cfg.get("libero_benchmark", ""))
        from flower.models.flower import FLOWERVLA
        torch.manual_seed(0)                               # the <Flow> token's embedding row (see the module docstring)
        self.model = FLOWERVLA(**mc)
        self.weights_fix = self._fix_weights(mc["pretrained_model_path"])
        self.model = self.model.to(device).eval()
        for q in self.model.parameters():
            q.requires_grad_(False)
        self.policy = self.model
        self.device = device
        self.H = int(self.model.act_window_size)           # 10
        self.A = int(self.model.action_dim)                # 7: no padded dims
        self.a = self.A
        self.dim = self.H * self.a
        self.dim_full = self.H * self.A
        self.real = np.arange(self.dim)
        self.num_steps = int(self.model.num_sampling_steps)
        self.clip = (-1e9, 1e9)
        self.layout = "chunk"
        self.img = None
        self.n_params = int(sum(p.numel() for p in self.model.parameters()))
        self.expert = self.model.dit                       # the 18 FlowBlocks: the action expert (LoRA goes here)
        import torchvision.transforms as T
        self._resize = T.Resize(IMG, antialias=True)
        self.load_secs = time.time() - t0

    def action_codec(self):
        return FlowerActionCodec()

    # the HF export renamed vlm.language_model.model.* -> vlm.language_*; FLOWERVLA._load_pretrained_weights maps the encoder back
    # (vlm.language_encoder.*) but not these two, so they would stay at Florence-2-large's base values (the fine-tuned token
    # embeddings would be lost)
    _RENAMED = {"vlm.language_shared.weight": "vlm.language_model.model.shared.weight",
                "vlm.language_final_logits_bias": "vlm.language_model.final_logits_bias"}

    @torch.no_grad()
    def _fix_weights(self, path):
        """Load the two renamed tensors; the <Flow> prompt embedding then reads the checkpoint's row of the token table (a fixed
        value in place of a fresh random draw).  -> dict for the log."""
        from safetensors import safe_open
        sd = self.model.state_dict()
        done = {}
        with safe_open(path, framework="pt") as f:
            for old, new in self._RENAMED.items():
                if old in f.keys() and new in sd:
                    w = f.get_tensor(old)
                    assert tuple(w.shape) == tuple(sd[new].shape), (old, tuple(w.shape), tuple(sd[new].shape))
                    sd[new].copy_(w)
                    done[new] = tuple(w.shape)
        tid = self.model.tokenizer.convert_tokens_to_ids("<Flow>")
        emb = self.model.vlm.get_input_embeddings().weight
        self.model.prompt_embeds = emb[tid].detach().clone()[None, None]
        return dict(loaded=done, flow_token=int(tid), tied=bool(emb.data_ptr() == self.model.vlm.get_encoder().embed_tokens.weight.data_ptr()))

    # ── conditioning ────────────────────────────────────────────────────────
    def _views(self, obs_dicts, key):
        ims = [torch.from_numpy(np.ascontiguousarray(np.asarray(o[key])[::-1, ::-1])).permute(2, 0, 1) for o in obs_dicts]
        x = self._resize(torch.stack(ims)).float() / 255.0           # uint8 Resize (as the authors' transform order), then scale
        m = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        s = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
        return ((x - m) / s)[:, None].to(self.device)

    @torch.no_grad()
    def observations(self, obs_dicts):
        """A list of raw observation dicts (the openpi client's keys: both views uint8 HWC flipped 180 degrees; observation/state;
        prompt) -> one batched input in FLOWER's conventions (views un-flipped, 112, CLIP-normalised)."""
        st = torch.as_tensor(np.stack([np.asarray(o["observation/state"], np.float32) for o in obs_dicts]), device=self.device)
        return FObservation(static=self._views(obs_dicts, "observation/image"), gripper=self._views(obs_dicts, "observation/wrist_image"),
                            texts=[str(o["prompt"]) for o in obs_dicts], state=st)

    @torch.no_grad()
    def observation(self, obs_dict):
        return self.observations([obs_dict])

    @torch.no_grad()
    def prefix(self, observation):
        """FLOWERVLA.encode_observations once per observation (Florence-2 encoder over both views, <Flow> and the prompt); the DiT
        reads its features at every (x_t, t).  kv = dict(features, frequency_embeds, action_type)."""
        o = observation
        self._span = span = self.text_span(o)
        try:
            c = self.model.encode_observations({"rgb_obs": {"rgb_static": o.static, "rgb_gripper": o.gripper}, "lang_text": o.texts})
        finally:
            self._span = None
        assert c["features"].shape[1] == 2 * self._img_tok + 1 + self._prompt_len, "text_span: the language slot is not where expected"
        n = len(o.texts)
        kv = dict(features=c["features"], frequency_embeds=c["frequency_embeds"].to(self.device),
                  action_type=c["action_type"].to(self.device))
        return Prefix(state=o.state, pad_masks=torch.ones((n, 1), dtype=torch.bool, device=self.device), kv=kv, n=n)

    # ── the conditioning encoder's residual stream (backbone.PolicyBackbone.text_span / cond_hook) ─────────────
    # The Florence-2 (BART-large) encoder reads [agent DaViT tokens, wrist DaViT tokens, <Flow>, the prompt's BART tokens] with an
    # all-ones mask (pads attended); the prompt starts with a constant robot description, so the INSTRUCTION's first token sits a
    # fixed number of tokens into the prompt.  cond_hook is the base class's (pre-hooks on the encoder layers, which only
    # encode_observations runs).
    _img_tok = None

    @torch.no_grad()
    def text_span(self, observation):
        """-> (offset of the instruction's first token, instruction tokens per row [B]).  Records the padded prompt length for
        prefix()'s check."""
        m = self.model
        if self._img_tok is None:
            self._img_tok = int(m.vlm._encode_image(observation.static[:1, 0].to(next(m.parameters()).dtype)).shape[1])
        prompts = m.construct_prompts({"lang_text": observation.texts})
        tok = m.tokenizer(prompts, padding=True, truncation=True, max_length=77, return_offsets_mapping=True)
        starts, counts = [], []
        for p, txt, offs in zip(prompts, observation.texts, tok["offset_mapping"]):
            c0 = p.rfind(txt)
            c1 = c0 + len(txt)
            idx = [j for j, (a, b) in enumerate(offs) if b > a and b > c0 and a < c1]
            starts.append(idx[0])
            counts.append(len(idx))
        assert len(set(starts)) == 1, f"text_span: instruction offsets differ across rows {starts}"
        p0 = 2 * self._img_tok + 1
        self._prompt_len = len(tok["input_ids"][0])
        return p0 + starts[0], torch.as_tensor(counts, dtype=torch.long)

    def cond_layers(self):
        return self.model.vlm.get_encoder().layers

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
        """FLOWERVLA.dit_forward(z_t, t, cond): the velocity eps - a on [n, H, A] inputs.  -> torch [n, H * A]."""
        import contextlib
        x = torch.as_tensor(x_full, dtype=torch.float32, device=self.device).reshape(-1, self.H, self.A)
        n = len(x)
        pre = self._expand(prefix, n)
        tt = _tt(t, self.device).reshape(-1)
        tt = tt.expand(n) if tt.numel() == 1 else tt
        with (gate() if gate is not None else contextlib.nullcontext()):
            vt = self.model.dit_forward(x, tt, dict(pre.kv, proprio=None))
        return vt.reshape(n, self.dim_full).float()

    # ── normalisation ───────────────────────────────────────────────────────
    def unnormalize(self, rows, state=None):
        """real rows [n, dim] -> LIBERO delta actions [n, H, 7]: sample_actions' clamp(-1, 1) (the actions are raw units)."""
        return FlowerActionCodec().unnormalize(np.asarray(rows, np.float32).reshape(-1, self.H, self.a))

    def describe(self):
        return (f"FlowerBackbone({self.checkpoint_dir}: suite {self.suite}, H={self.H}, A={self.A}, dim={self.dim}, steps={self.num_steps}, "
                f"n_params={self.n_params:,}, expert={sum(p.numel() for p in self.expert.parameters()):,}, load {self.load_secs:.1f}s)")

