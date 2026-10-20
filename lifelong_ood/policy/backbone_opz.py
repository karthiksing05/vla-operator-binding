"""OPZBackbone: open-pi-zero's pi0 (allenzren/open-pi-zero, Bridge checkpoints) as the same score field over action chunks as
PolicyBackbone.  docs/SIMPLER.md ("The backbone interface on these checkpoints") has the mapping; in short:

The checkpoint.  A PyTorch re-implementation of pi0 (not openpi): PaliGemma-3B (SigLIP 256 image tokens + Gemma) as the "vlm"
mixture, a 1-token "proprio" mixture and a 4-token "action" mixture (MoT; block-causal mask: vlm <-> vlm, proprio -> vlm + proprio,
action -> all).  Bridge checkpoints: bridge_beta / bridge_uniform (time sampling in training), float32 inference (the published
numbers' dtype), the eval config's architecture (config/eval/bridge.yaml: time_max_period 1e4, rope theta 1e4).

Conventions (the authors' BridgeSimplerAdapter, src/agent/env_adapter/simpler.py, which `observations` reproduces):
  * ONE image: SimplerEnv's 3rd-view camera (get_image_from_maniskill2_obs_dict, 640 x 480 uint8), cv2 Lanczos-resized to 224,
    [0, 255] -> [-1, 1]; the prompt as  <image> x 256, <bos>, instruction, "\n"  right-padded to 276 tokens (max 20 text tokens);
  * proprio 7-D = [ee pos (3), roll/pitch/yaw of the ee rotation in Bridge's top-down frame (R @ default_rot.T, sxyz), gripper
    openness]; read from obs["agent"]["eef_pos"] (8-D: pos + wxyz quat + openness; the allenzren SimplerEnv fork adds it), bound-
    normalised to [-1, 1] by the dataset's p01 / p99 (config/bridge_statistics.json);
  * actions: H = 4 steps x 7 = [delta pos 3, delta roll/pitch/yaw 3, gripper in [0, 1]], dims 0-5 bound-normalised (p01 / p99),
    the gripper raw; the sampler clamps the chunk to [-1, 1].  `unnormalize` -> SimplerEnv's 7-D actions [delta pos, axis-angle,
    gripper +1 open / -1 close] (postprocess: denormalise, euler2axangle, gripper > 0.5 -> +1).  All 4 actions are executed (5 Hz).

The flow.  open-pi-zero's time runs the OTHER way from openpi's: s = 0 noise, s = 1 data,
    x_s = (1 - (1 - sigma) s) eps + s a,   target u = dx/ds = a - (1 - sigma) eps,   sigma = flow_sig_min = 1e-3,
and the sampler is Euler from s = 0 with ds = +1/K (K = 10).  With the repo's time t = 1 - s:
    x_t = (sigma + (1 - sigma) t) eps + (1 - t) a,     v(x, t) := dx/dt = -u(x, 1 - t),
so PolicyBackbone's Euler (t = 1 .. 1/K, x <- x - v / K) IS the authors' sampler step for step (the same noise gives the same
chunk; `sample` adds only the final clamp), and the identities hold up to sigma: x - t v = a + sigma eps (Tweedie),
x + (1 - t) v = (1 - sigma) eps (= E[eps | x_t] times 1 - sigma).  flow_loss: the generic ||v - u||^2 equals open-pi-zero's MSE
(both signs flip); its exact target in repo terms is u = (1 - sigma) eps - a (adapt.py passes eps - a: an O(sigma) difference).

No padding: A = a = 7, dim = dim_full = 28, real = all.
Run in /coc/flash10/ksingara3/robot/simpler/.venv_pi0 (torch 2.5.0, transformers 4.47.1), with TRANSFORMERS_CACHE pointing at the
folder holding paligemma-3b-pt-224/ (the tokenizer; docs/SIMPLER.md).
"""
from __future__ import annotations

import contextlib
import dataclasses
import os
import sys
import time

import numpy as np
import torch

from .backbone import PolicyBackbone, Prefix, _tt

OPZ = "/coc/flash10/ksingara3/robot/ext/openpizero_src/open-pi-zero"
CKPTS = {"bridge_beta": "/coc/flash10/ksingara3/robot/data/openpizero/bridge_beta_step19296_2024-12-26_22-30_42.pt",
         "bridge_uniform": "/coc/flash10/ksingara3/robot/data/openpizero/bridge_uniform_step19296_2024-12-26_22-31_42.pt"}
TOKENIZER_ROOT = "/coc/flash10/ksingara3/robot/data/paligemma_tok"


def _import_opz():
    if OPZ not in sys.path:
        sys.path.insert(0, OPZ)
    os.environ.setdefault("TRANSFORMERS_CACHE", TOKENIZER_ROOT)
    os.environ.setdefault("VLA_LOG_DIR", "/tmp")
    from omegaconf import OmegaConf
    from src.model.vla.pizero import PiZeroInference
    from src.model.vla import joint_model as JM
    from src.agent.env_adapter.simpler import BridgeSimplerAdapter
    from src.model.kv_cache import KVCache
    return OmegaConf, PiZeroInference, JM, BridgeSimplerAdapter, KVCache


@dataclasses.dataclass
class OObservation:
    input_ids: torch.Tensor          # [B, 276]
    pixel_values: torch.Tensor       # [B, 3, 224, 224] in [-1, 1]
    attention_mask: torch.Tensor     # [B, 276] (1 on image + text tokens)
    proprios: torch.Tensor           # [B, 1, 7] normalised
    prompts: list


class _Handle:
    def __init__(self, lst, fn):
        self.lst, self.fn = lst, fn

    def remove(self):
        if self.fn in self.lst:
            self.lst.remove(self.fn)


class OPZBackbone(PolicyBackbone):
    kind = "opz"

    def __init__(self, checkpoint="bridge_beta", device="cuda", num_steps=None, dtype=torch.float32):
        OmegaConf, PiZeroInference, JM, Adapter, KVCache = _import_opz()
        self._JM, self._KVCache = JM, KVCache
        t0 = time.time()
        cfg = OmegaConf.load(f"{OPZ}/config/eval/bridge.yaml")
        cfg.env.adapter.dataset_statistics_path = f"{OPZ}/config/bridge_statistics.json"
        cfg.env.adapter.pretrained_model_path = f"{os.environ['TRANSFORMERS_CACHE']}/paligemma-3b-pt-224"
        cfg.use_torch_compile = False
        cfg.use_bf16 = dtype == torch.bfloat16
        if num_steps is not None:
            cfg.num_inference_steps = int(num_steps)
        self.cfg = cfg
        path = CKPTS.get(checkpoint, checkpoint)
        self.checkpoint_dir = path
        model = PiZeroInference(cfg, use_ddp=False)
        data = torch.load(path, weights_only=True, map_location="cpu")         # EvalAgent.load_checkpoint
        model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in data["model"].items()}, strict=True)
        del data
        model.freeze_all_weights()
        self.model = model.to(dtype).to(device).eval()
        self.dtype = dtype
        a = cfg.env.adapter
        self.adapter = Adapter(dataset_statistics_path=a.dataset_statistics_path, pretrained_model_path=a.pretrained_model_path,
                               tokenizer_padding=a.tokenizer_padding, num_image_tokens=a.num_image_tokens,
                               image_size=list(a.image_size), max_seq_len=a.max_seq_len)
        self.device = device
        self.H = int(cfg.horizon_steps)                 # 4
        self.A = int(cfg.action_dim)                    # 7
        self.a = self.A
        self.dim = self.dim_full = self.H * self.A      # 28
        self.real = np.arange(self.dim)
        self.num_steps = int(cfg.num_inference_steps)   # 10
        self.sigma = float(self.model.flow_sig_min)     # 1e-3
        self.clip = (-float(cfg.final_action_clip_value), float(cfg.final_action_clip_value))
        self.layout = "chunk"
        self.img = None
        self.n_img = int(a.num_image_tokens)            # 256
        self.n_params = int(sum(p.numel() for p in self.model.parameters()))
        self.expert = self.model.joint_model.mixtures["action"]   # the action expert (proprio is a separate mixture in this ckpt)
        self._cond_fns = []
        self.load_secs = time.time() - t0

    # ── conditioning ────────────────────────────────────────────────────────
    def raw_observation(self, image, eef_pos, prompt):
        """SimplerEnv pieces -> the raw dict `observations` takes: image = get_image_from_maniskill2_obs_dict(env, obs) (uint8
        HWC), eef_pos = obs['agent']['eef_pos'] (8-D), prompt = the instruction."""
        return {"image": np.asarray(image), "eef_pos": np.asarray(eef_pos, np.float64), "prompt": str(prompt)}

    @torch.no_grad()
    def observations(self, obs_dicts):
        """A list of raw dicts (image, eef_pos, prompt) -> one batched input, exactly BridgeSimplerAdapter.preprocess per row."""
        import cv2
        ad = self.adapter
        ims, props = [], []
        for o in obs_dicts:
            im = cv2.resize(np.asarray(o["image"]), ad.image_size, interpolation=cv2.INTER_LANCZOS4)
            ims.append(torch.as_tensor(im, dtype=torch.uint8).permute(2, 0, 1))
            raw = ad.preprocess_proprio({"agent": {"eef_pos": np.asarray(o["eef_pos"])}})
            st = ad.dataset_statistics["proprio"]
            props.append(ad.normalize_bound(raw, np.array(st["p01"]), np.array(st["p99"]), clip_min=-1, clip_max=1))
        prompts = [str(o["prompt"]) for o in obs_dicts]
        mi = ad.processor(text=prompts, images=torch.stack(ims))
        dev = self.device
        return OObservation(input_ids=mi["input_ids"].to(dev), pixel_values=mi["pixel_values"].to(dev, self.dtype),
                            attention_mask=mi["attention_mask"].to(dev),
                            proprios=torch.as_tensor(np.stack(props), dtype=self.dtype, device=dev)[:, None], prompts=prompts)

    @torch.no_grad()
    def observation(self, obs_dict):
        return self.observations([obs_dict])

    def _masks(self, attention_mask):
        m = self.model
        cm, vpos, ppos, apos = m.build_causal_mask_and_position_ids(attention_mask, dtype=self.dtype)
        itp, am = m.split_full_mask_into_submasks(cm)
        d = self.device
        return itp.to(d), am.to(d), vpos.to(d), ppos.to(d), apos.to(d)

    @torch.no_grad()
    def prefix(self, observation):
        """The VLM + proprio pass once per observation (PiZero.infer_action's first half), caching their keys / values; the
        action expert reads them at every (x_t, t).  kv = dict(caches, action_mask, action_pos)."""
        m, o = self.model, observation
        itp, am, vpos, ppos, apos = self._masks(o.attention_mask)
        caches = m.joint_model.build_mixture_caches()
        emb = {"vlm": m._forward_siglip_and_text_embedding(o.input_ids, o.pixel_values), "proprio": m.proprio_encoder(o.proprios)}
        self._span = self.text_span(o)
        try:
            self._joint(emb, itp, {"vlm": vpos, "proprio": ppos}, caches)
        finally:
            self._span = None
        n = int(o.input_ids.shape[0])
        return Prefix(state=o.proprios, pad_masks=o.attention_mask.bool(), kv=dict(caches=caches, action_mask=am, action_pos=apos), n=n)

    def _joint(self, embeds_all, mask, pos, caches):
        """JointModel.forward for the prefix pass (return_caches), unrolled so cond_hook can edit the vlm stream before each layer."""
        jm, JM = self.model.joint_model, self._JM
        for name in embeds_all:
            embeds_all[name] = embeds_all[name] * torch.tensor(embeds_all[name].shape[-1] ** 0.5, dtype=embeds_all[name].dtype,
                                                               device=embeds_all[name].device)
        L = jm.num_hidden_layers
        for i in range(L):
            for fn in list(self._cond_fns):
                out = fn(i, embeds_all["vlm"], self._span)
                if out is not None:
                    embeds_all = dict(embeds_all, vlm=out)
            embeds_all = JM.forward_mixture_layers(jm.mixtures, mask, pos, embeds_all, layer_idx=i, time_cond=None, kv_caches=caches,
                                                   cache_mode="append_non_active",
                                                   post_attn_skip_names=("vlm", "proprio") if i == L - 1 else [])
        return caches

    # ── the conditioning encoder's residual stream ────────────────────────
    def text_span(self, observation):
        """[<image> x 256][<bos> instruction \\n][pad]: offset 256, n_valid = the non-image valid tokens (bos + text + newline)."""
        n = observation.attention_mask.sum(1).long() - self.n_img
        return self.n_img, n

    def cond_layers(self):
        return self.model.joint_model.mixtures["vlm"].layers

    def cond_hook(self, fn):
        """fn(layer_index, hidden [B, 276, 2048], span) -> new hidden or None, before every Gemma layer of the prefix pass.  The
        joint model calls its layers' sub-functions (attn_func / layer_func), never the layer modules, so torch forward hooks
        would not fire: the prefix pass is unrolled in `_joint` and calls fn itself.  Returns one handle (.remove())."""
        self._cond_fns.append(fn)
        return [_Handle(self._cond_fns, fn)]

    # ── the field ───────────────────────────────────────────────────────────
    def _kv(self, caches, n, how):
        out = {}
        for name, c in caches.items():
            c2 = self._KVCache()
            for k, v in zip(c.key_cache, c.value_cache):
                c2.key_cache.append(how(k))
                c2.value_cache.append(how(v))
            out[name] = c2
        return out

    def _expand(self, prefix, n):
        if prefix.n == n:
            return prefix
        if prefix.n != 1:
            raise ValueError("prefix batch must be 1 or n")
        e = lambda t: t.expand(n, *t.shape[1:])                    # noqa: E731
        kv = prefix.kv
        return Prefix(state=e(prefix.state), pad_masks=e(prefix.pad_masks),
                      kv=dict(caches=self._kv(kv["caches"], n, e), action_mask=e(kv["action_mask"]), action_pos=e(kv["action_pos"])), n=n)

    def repeat(self, prefix, k):
        r = lambda t: t.repeat_interleave(int(k), dim=0)           # noqa: E731
        kv = prefix.kv
        return Prefix(state=r(prefix.state), pad_masks=r(prefix.pad_masks),
                      kv=dict(caches=self._kv(kv["caches"], 0, r), action_mask=r(kv["action_mask"]), action_pos=r(kv["action_pos"])),
                      n=prefix.n * int(k))

    def pad(self, X, t, seed=0):
        return torch.as_tensor(X, dtype=torch.float32, device=self.device).reshape(-1, self.H, self.A)

    def u_opz(self, x_full, s, prefix, gate=None):
        """open-pi-zero's own velocity u(x, s) (s = 0 noise, s = 1 data): one action-expert pass on the cached prefix."""
        m = self.model
        x = torch.as_tensor(x_full, device=self.device).reshape(-1, self.H, self.A).to(self.dtype)
        n = len(x)
        pre = self._expand(prefix, n)
        ss = _tt(s, self.device).reshape(-1)
        ss = (ss.expand(n) if ss.numel() == 1 else ss).to(self.dtype)
        tc = m.time_embedding(ss)
        ae = m.action_encoder(x, tc) if not m.action_expert_adaptive_mode else m.action_encoder(x)
        kv = pre.kv
        with (gate() if gate is not None else contextlib.nullcontext()):
            h = m.joint_model(attention_mask=kv["action_mask"], position_ids_all={"action": kv["action_pos"]}, embeds_all={"action": ae},
                              time_cond=tc, kv_caches=kv["caches"], cache_mode="append_non_active")["action"]
        return m.action_decoder(h).reshape(n, self.dim_full).float()

    def v_full(self, x_full, t, prefix, gate=None):
        """v(x, t) = dx/dt = -u(x, 1 - t) on [n, H, A] (= [n, 28]) inputs.  -> torch [n, 28]."""
        tt = _tt(t, self.device).reshape(-1)
        return -self.u_opz(x_full, 1.0 - tt, prefix, gate)

    def flow_loss(self, v, u, xt, t):
        """open-pi-zero's objective, MSE(u_pred, a - (1 - sigma) eps), in the repo's signs: mean ||v - u||^2."""
        return ((v.reshape(len(v), -1) - u.reshape(len(u), -1)) ** 2).mean()

    @torch.no_grad()
    def sample(self, n, prefix, seed=0, num_steps=None, guide=None, step_gate=None, noise=None, full=False, start=None, clamp=True):
        """PolicyBackbone's Euler (t = 1 .. 1/K) = the authors' Euler (s = 0 .. 1 - 1/K), then their final clamp to [-1, 1]."""
        out = super().sample(n, prefix, seed=seed, num_steps=num_steps, guide=guide, step_gate=step_gate, noise=noise, full=full,
                             start=start)
        return np.clip(out, *self.clip) if clamp else out

    @torch.no_grad()
    def reference_sample(self, observation, seed):
        """The authors' sampler (PiZeroInference.forward = infer_action) on this observation with torch.manual_seed(seed); also
        returns the noise it drew (its first random draw on the device), for a same-noise parity check.  -> (chunk [H, A], noise)."""
        m, o = self.model, observation
        itp, am, vpos, ppos, apos = self._masks(o.attention_mask)
        torch.manual_seed(int(seed))
        noise = torch.randn((o.input_ids.shape[0], self.H, self.A), device=self.device, dtype=self.dtype)
        torch.manual_seed(int(seed))
        ref = m(input_ids=o.input_ids, pixel_values=o.pixel_values, image_text_proprio_mask=itp, action_mask=am, vlm_position_ids=vpos,
                proprio_position_ids=ppos, action_position_ids=apos, proprios=o.proprios)
        return ref.float().cpu().numpy(), noise.float().cpu().numpy()

    # ── normalisation ───────────────────────────────────────────────────────
    def unnormalize(self, rows, state=None):
        """rows [n, 28] (normalised chunk) -> SimplerEnv actions [n, H, 7] (BridgeSimplerAdapter.postprocess); `state` unused."""
        R = np.asarray(rows, np.float32).reshape(-1, self.H, self.A)
        return np.stack([self.adapter.postprocess(r.astype(np.float64)) for r in R])

    def describe(self):
        return (f"OPZBackbone({os.path.basename(self.checkpoint_dir)}: H={self.H}, A={self.A}, dim={self.dim}, steps={self.num_steps}, "
                f"sigma={self.sigma}, dtype={self.dtype}, n_params={self.n_params:,}, "
                f"expert={sum(p.numel() for p in self.expert.parameters()):,}, load {self.load_secs:.0f}s)")
