"""XVLABackbone: LeRobot's X-VLA (lerobot/xvla-libero) as the same score field over action chunks as PolicyBackbone.

The checkpoint.  lerobot/xvla-libero holds exactly the weights of 2toINF/X-VLA-Libero (all 902 shared tensors bit-identical; the
tied BART embedding is stored once under another name), so the conventions are the authors' LIBERO client's
(github.com/2toINF/X-VLA evaluation/libero/libero_client.py), which LeRobot's make_xvla_libero_pre_post_processors reproduces:
  * ABSOLUTE end-effector control (the env's OSC controller with use_delta = False after the 10 settle steps); actions are
    20-D "ee6d" = [target eef pos 3, target rotation 6D (first two columns of the rotation matrix) 6, gripper logit 1] + 10 zeros
    (a second arm); the gripper is a BCE logit (sigmoid > 0.5 -> close = +1, else -1), the rest raw units (IDENTITY norm);
  * proprio = [eef pos (controller.ee_pos), rot6d(controller.ee_ori_mat), 0] + 10 zeros; the gripper channels (9, 19) of the
    proprio and of the noisy action are zeroed before the transformer (EE6DActionSpace.preprocess);
  * images: agentview flipped 180 degrees, the WRIST VIEW UNFLIPPED, a third (empty, masked) view; [0, 1] then ImageNet
    mean / std; resized to 224; BART tokens padded to 50; domain_id 3; chunk 30; 10 denoising steps.
The repo's client sends openpi's format (both views flipped, 224, 8-D state = eef pos + axis-angle of the hand BODY quaternion +
finger qpos), so `observation` un-flips the wrist view and rebuilds the proprio: the grip site's frame (the controller's
ee_ori_mat) is the hand body's frame times a constant rotation C (measured in the simulator, below).  The client switches the
controller to absolute mode when the server's metadata says {"control": "absolute"} (experiments/serve_guided.py).

The flow.  X-VLA trains x_t = t * eps + (1 - t) * a (t = 1 noise, t = 0 data) -- the repo's path, no conversion -- but its
transformer predicts the CLEAN chunk a_hat(x_t, t) (x-prediction, MSE x 500 on positions, x 10 on rotations, BCE on the gripper),
so the velocity is v = (x_t - a_hat) / t, and every identity in backbone.py holds (Tweedie: x_t - t v = a_hat).  Its sampler is
NOT Euler: at t = i / K it re-noises the prediction with the INITIAL noise, x_{t'} = t' * x1 + (1 - t') * a_hat (Euler would use
eps_hat = (x_t - (1 - t) a_hat) / t in place of x1).  `sample` here is that rule ("native", the default: it reproduces the
policy's chunk for the same noise) with PolicyBackbone.sample's hooks (guide, step_gate, noise, start); `sampler = "euler"` falls
back to the base Euler sampler on the same field.  `flow_loss` is X-VLA's own objective in a_hat space (adapt.train uses it): the
generic ||v - u||^2 equals ||a_hat - a||^2 / t^2, which explodes at small t.

Row space: real dims a = 10 per step (pos, rot6d, gripper probability), A = 20, H = 30.  `unnormalize` -> ABSOLUTE 7-D LIBERO
actions [pos, axis-angle, gripper +-1]; `action_codec()` (data.Norm.for_backbone) maps the dataset's 7-D DELTA actions to these
rows with each frame's own state (target = pose + 0.05 * delta, R = Rot(0.5 * delta_rot) @ R_site: the OSC controller's own
delta -> goal map, which is how the authors made their absolute LIBERO actions, evaluation/libero/rel2abs.py).
Run in the SmolVLA venv (/coc/flash10/ksingara3/robot/smolvla/.venv: lerobot 0.4.4).
"""
from __future__ import annotations

import contextlib
import dataclasses
import os
import time

import numpy as np
import torch

from .backbone import PolicyBackbone, Prefix, _tt

CKPT = "lerobot/xvla-libero"
DOMAIN_ID = 3                                   # LIBERO's soft prompt (the authors' client, LeRobot's preprocessor)
# R_site = R_body @ C: the grip site (controller.ee_ori_mat) relative to the hand body (robot0_eef_quat), Panda in LIBERO
# (robosuite 1.4.1): -90 degrees about the hand's z; measured in the simulator over 40 steps of random motion (libero_spatial
# task 0), max deviation 2e-3 (the observation and the controller read the pose one physics substep apart)
C_SITE = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], np.float64)
POS_SCALE, ROT_SCALE = 0.05, 0.5                # the OSC controller's output_max (delta action in [-1, 1] -> metres / radians)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ── geometry (numpy; robosuite's conventions) ───────────────────────────────────────────────────────────────────
def aa2mat(aa):
    """axis-angle [..., 3] -> rotation matrices [..., 3, 3] (Rodrigues; robosuite's quat2mat(axisangle2quat(aa)))."""
    aa = np.asarray(aa, np.float64)
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = np.where(th > 1e-12, aa / np.maximum(th, 1e-12), 0.0)
    K = np.zeros(aa.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = k[..., 2], -k[..., 1], k[..., 0]
    s, c = np.sin(th)[..., None], np.cos(th)[..., None]
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def mat2r6d(R):
    """rotation matrices [..., 3, 3] -> X-VLA's 6D rotation [..., 6] (first column, then second column)."""
    R = np.asarray(R)
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def r6d2aa(r6d):
    """[N, 6] -> axis-angle [N, 3], the authors' Rotate6D_to_AxisAngle (Gram-Schmidt, robosuite mat2quat / quat2axisangle)."""
    from lerobot.policies.xvla.utils import rotate6d_to_axis_angle
    return np.asarray(rotate6d_to_axis_angle(np.asarray(r6d, np.float64).reshape(-1, 6)), np.float32)


def state_to_proprio(st8):
    """the client's 8-D state [..., 8] (eef pos, axis-angle of the hand body, finger qpos) -> X-VLA's 20-D proprio."""
    st = np.asarray(st8, np.float64)
    R = aa2mat(st[..., 3:6]) @ C_SITE
    head = np.concatenate([st[..., :3], mat2r6d(R), np.zeros(st.shape[:-1] + (1,))], axis=-1)
    return np.concatenate([head, np.zeros_like(head)], axis=-1).astype(np.float32)


class XVLAActionCodec:
    """data.Norm's interface for X-VLA: the dataset's 7-D DELTA actions (with each frame's state) <-> the model's 10 real dims.
    normalize(x [..., H, 7], state [H, 8] (each step's own frame) or [8]) -> [..., H, 10]; unnormalize(rows [..., H, 10]) ->
    ABSOLUTE 7-D LIBERO actions [pos, axis-angle, gripper +-1] (the controller in absolute mode)."""
    quantile = False
    delta = 0
    per_step_state = True                       # c0_fit.chunk_rows passes each chunk step's own state

    def normalize(self, x, state=None):
        assert state is not None, "X-VLA's actions are absolute targets: pass the state of every step"
        x = np.clip(np.asarray(x, np.float64), -1.0, 1.0)
        st = np.broadcast_to(np.asarray(state, np.float64), x.shape[:-1] + (8,))
        pos = st[..., :3] + POS_SCALE * x[..., :3]
        R = aa2mat(ROT_SCALE * x[..., 3:6]) @ (aa2mat(st[..., 3:6]) @ C_SITE)
        grip = (x[..., 6:7] > 0).astype(np.float64)
        return np.concatenate([pos, mat2r6d(R), grip], axis=-1).astype(np.float32)

    def unnormalize(self, x):
        x = np.asarray(x, np.float32)
        lead = x.shape[:-1]
        f = x.reshape(-1, x.shape[-1])
        out = np.concatenate([f[:, :3], r6d2aa(f[:, 3:9]), np.where(f[:, 9:10] > 0.5, 1.0, -1.0)], axis=-1)
        return out.reshape(lead + (7,)).astype(np.float32)


@dataclasses.dataclass
class XObservation:
    input_ids: torch.Tensor
    image_input: torch.Tensor
    image_mask: torch.Tensor
    domain_id: torch.Tensor
    state: torch.Tensor                     # the 20-D proprio, as the model sees it (before the gripper channels are zeroed)


class XVLABackbone(PolicyBackbone):
    kind = "xvla"

    def __init__(self, checkpoint=None, device="cuda", num_steps=None, sampler="native"):
        os.environ.setdefault("HF_HOME", "/coc/flash10/ksingara3/robot/data/hf/home")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from huggingface_hub import snapshot_download
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        t0 = time.time()
        path = checkpoint or CKPT
        self.checkpoint_dir = path if os.path.isdir(path) else snapshot_download(path)
        self.policy = XVLAPolicy.from_pretrained(self.checkpoint_dir).to(device).eval()
        self.cfg = self.policy.config
        # the saved pre-processor: rename, batch, BART tokenizer (max 50), domain_id 3, device, IDENTITY normaliser.  The image
        # steps (flip, [0, 1], ImageNet) are the LIBERO env processor's, done in `observation`
        self.pre, _ = make_pre_post_processors(self.cfg, pretrained_path=self.checkpoint_dir,
                                               preprocessor_overrides={"device_processor": {"device": device}})
        self.model = self.policy.model
        for q in self.model.parameters():
            q.requires_grad_(False)
        self.device = device
        self.dtype = next(self.model.transformer.parameters()).dtype
        self.H = int(self.cfg.chunk_size)                          # 30
        self.A = int(self.model.dim_action)                        # 20 (ee6d: two arms)
        self.a = 10                                                # pos 3, rot6d 6, gripper 1 (the second arm is zeros)
        self.dim = self.H * self.a
        self.dim_full = self.H * self.A
        self.real = np.concatenate([np.arange(self.a) + h * self.A for h in range(self.H)])
        self.num_steps = int(num_steps or self.cfg.num_denoising_steps)   # 10
        self.sampler = sampler
        self.clip = (-1e9, 1e9)
        self.layout = "chunk"
        self.img = None
        self.n_params = int(sum(p.numel() for p in self.model.parameters()))
        self.expert = self.model.transformer                       # the SoftPromptedTransformer: the denoiser (LoRA goes here)
        sp = self.model.action_space
        self.grip_idx = list(sp.gripper_idx)                       # (9, 19)
        self.load_secs = time.time() - t0

    def action_codec(self):
        return XVLAActionCodec()

    # ── conditioning ────────────────────────────────────────────────────────
    def _batch(self, obs_dicts):
        def img(k, flip):
            ims = []
            for o in obs_dicts:
                x = np.asarray(o[k])
                ims.append(torch.from_numpy(np.array(x[::-1, ::-1] if flip else x)).permute(2, 0, 1))
            x = torch.stack(ims).float() / 255.0
            m = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
            s = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
            return (x - m) / s
        st = np.stack([np.asarray(o["observation/state"], np.float32) for o in obs_dicts])
        return {"observation.images.image": img("observation/image", False),        # flipped by the client, as X-VLA saw it
                "observation.images.image2": img("observation/wrist_image", True),   # un-flipped: X-VLA's wrist view is raw
                "observation.state": torch.as_tensor(state_to_proprio(st)),
                "task": [str(o["prompt"]) for o in obs_dicts]}

    @torch.no_grad()
    def observations(self, obs_dicts):
        """A list of raw observation dicts (the openpi client's keys; observation/image and observation/wrist_image uint8 HWC, both
        flipped 180 degrees; observation/state 8-D; prompt) -> one batched input, through the checkpoint's own pre-processor."""
        b = self.pre(self._batch(obs_dicts))
        n = len(obs_dicts)
        if "domain_id" not in b:
            b["domain_id"] = torch.full((n,), DOMAIN_ID, dtype=torch.long)
        ins = self.policy._build_model_inputs(b)
        return XObservation(input_ids=ins["input_ids"], image_input=ins["image_input"], image_mask=ins["image_mask"],
                            domain_id=ins["domain_id"].to(self.device), state=ins["proprio"])

    @torch.no_grad()
    def observation(self, obs_dict):
        return self.observations([obs_dict])

    @torch.no_grad()
    def prefix(self, observation):
        """The Florence-2 encoding (language + agent view, the other views' tokens) once per observation; the denoiser reads it
        at every (x_t, t).  kv = dict(vlm, aux, domain)."""
        o = observation
        self._span = span = self.text_span(o)
        try:
            enc = self.model.forward_vlm(o.input_ids, o.image_input.to(self.dtype), o.image_mask)
        finally:
            self._span = None
        assert enc["vlm_features"].shape[1] == span[0] + o.input_ids.shape[1], "text_span: the language slot is not where expected"
        n = int(o.state.shape[0])
        kv = dict(vlm=enc["vlm_features"], aux=enc["aux_visual_inputs"], domain=o.domain_id)
        return Prefix(state=o.state, pad_masks=torch.ones((n, 1), dtype=torch.bool, device=o.state.device), kv=kv, n=n)

    # ── the conditioning encoder's residual stream (backbone.PolicyBackbone.text_span / cond_hook) ─────────────
    # forward_vlm: the Florence-2 (BART-large) encoder reads [agent-view DaViT tokens, BART tokens of the instruction] with an
    # all-ones attention mask (the pads ARE attended); the other views bypass it.  cond_hook is the base class's (forward
    # pre-hooks on the encoder's layer modules, which only forward_vlm runs).
    _img_tok = None

    @torch.no_grad()
    def text_span(self, observation):
        if self._img_tok is None:
            self._img_tok = int(self.model.vlm._encode_image(observation.image_input[:1, 0].to(self.dtype)).shape[1])
        pad = getattr(self.model.vlm.config, "pad_token_id", None)
        pad = 1 if pad is None else int(pad)                       # BART's <pad>
        return self._img_tok, (observation.input_ids != pad).sum(1).long()

    def cond_layers(self):
        return self.model.vlm.language_model.model.encoder.layers

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
    def x0_full(self, x_full, t, prefix, gate=None):
        """The denoiser's clean-chunk prediction a_hat(x_t, t) on FULL [n, H, A] inputs (grippers as probabilities).  -> [n, H * A]."""
        x = torch.as_tensor(x_full, dtype=torch.float32, device=self.device).reshape(-1, self.H, self.A)
        n = len(x)
        pre = self._expand(prefix, n)
        tt = _tt(t, self.device).reshape(-1)
        tt = tt.expand(n) if tt.numel() == 1 else tt
        proprio_m, x_m = self.model.action_space.preprocess(pre.state.to(self.dtype), x.to(self.dtype))
        with (gate() if gate is not None else contextlib.nullcontext()):
            out = self.model.transformer(domain_id=pre.kv["domain"], vlm_features=pre.kv["vlm"], aux_visual_inputs=pre.kv["aux"],
                                         action_with_noise=x_m, proprio=proprio_m, t=tt.to(self.dtype))
        out = out.float()
        g = torch.zeros(self.A, dtype=torch.bool, device=out.device)
        g[self.grip_idx] = True
        out = torch.where(g, torch.sigmoid(out), out)              # EE6DActionSpace.postprocess, out of place
        return out.reshape(n, self.dim_full)

    def v_full(self, x_full, t, prefix, gate=None):
        """v = (x_t - a_hat) / t on FULL [n, H, A] inputs.  -> torch [n, H * A]."""
        x = torch.as_tensor(x_full, dtype=torch.float32, device=self.device).reshape(-1, self.dim_full)
        tt = _tt(t, self.device).reshape(-1, 1)
        return (x - self.x0_full(x, t, prefix, gate)) / tt

    def flow_loss(self, v, u, xt, t):
        """X-VLA's training objective (EE6DActionSpace.compute_loss) in a_hat space: a_hat = x_t - t v, a = x_t - t u."""
        n = len(v)
        tt = _tt(t, self.device).reshape(-1, 1)
        x = torch.as_tensor(xt, dtype=torch.float32, device=self.device).reshape(n, -1)
        ah = (x - tt * v.reshape(n, -1)).reshape(n, self.H, self.A)
        a = (x - tt * u.reshape(n, -1)).reshape(n, self.H, self.A)
        sp = self.model.action_space
        mse = lambda idx: ((ah[..., list(idx)] - a[..., list(idx)]) ** 2).mean()   # noqa: E731
        p = ah[..., self.grip_idx].clamp(1e-6, 1 - 1e-6)
        y = a[..., self.grip_idx]
        bce = -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()
        return (mse(sp.POS_IDX_1) + mse(sp.POS_IDX_2)) * sp.XYZ_SCALE + (mse(sp.ROT_IDX_1) + mse(sp.ROT_IDX_2)) * sp.ROT_SCALE \
            + bce * sp.GRIPPER_SCALE

    # ── sampling: X-VLA's own rule, with PolicyBackbone.sample's hooks ─────
    @torch.no_grad()
    def sample(self, n, prefix, seed=0, num_steps=None, guide=None, step_gate=None, noise=None, full=False, start=None):
        """X-VLA's sampler: at t = i / K (i = K .. 1) predict a_hat, then x <- t' * x1 + (1 - t') * a_hat with t' = (i - 1) / K and
        x1 the INITIAL noise; the last step returns a_hat.  Same hooks as PolicyBackbone.sample (a guide mixes an analytic eps
        into the prediction and a_hat = x_t - t v of the mixed velocity; start = (t0, a0) re-noises a0 with x1 at t0)."""
        if self.sampler == "euler":
            return super().sample(n, prefix, seed=seed, num_steps=num_steps, guide=guide, step_gate=step_gate, noise=noise,
                                  full=full, start=start)
        K = int(num_steps or self.num_steps)
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        x1 = (torch.randn((n, self.dim_full), generator=g).to(self.device) if noise is None
              else torch.as_tensor(noise, device=self.device).reshape(n, self.dim_full).float())
        x = x1.clone()
        i0 = K
        if start is not None:
            t0, a0 = start
            i0 = int(round(float(t0) * K))
            a_full = torch.zeros((n, self.dim_full), dtype=torch.float32, device=self.device)
            a_full[:, self.real] = torch.as_tensor(np.asarray(a0, np.float32).reshape(n, -1), device=self.device)
            x = (i0 / K) * x1 + (1.0 - i0 / K) * a_full
        for i in range(i0, 0, -1):
            tf = i / K
            t = torch.tensor(tf, dtype=torch.float32, device=self.device)
            gate = None
            if step_gate is not None:
                sg = step_gate(x[:, self.real], tf)
                gate = sg[0] if isinstance(sg, tuple) else sg
            vt = self.v_full(x, t.expand(n), prefix, gate)
            if guide is not None and tf < 1.0 - 1e-6:
                eps_fn, w_fn = guide
                xr = x[:, self.real]
                e0 = x + (1.0 - t) * vt
                et = torch.as_tensor(eps_fn(xr, tf), dtype=torch.float32, device=self.device)
                w = w_fn(xr, tf)
                if isinstance(w, np.ndarray):
                    w = torch.as_tensor(w, dtype=e0.dtype, device=e0.device)
                if torch.is_tensor(w):
                    w = w.to(e0.dtype).reshape(1, -1) if w.numel() == self.dim else w.to(e0.dtype).reshape(-1, 1)
                else:
                    w = float(w)
                e = e0.clone()
                e[:, self.real] = e0[:, self.real] + w * (et - e0[:, self.real])
                vt = (e - x) / (1.0 - t)
            a_hat = x - t * vt
            tn = (i - 1) / K
            x = tn * x1 + (1.0 - tn) * a_hat if i > 1 else a_hat
        out = x if full else x[:, self.real]
        return out.float().cpu().numpy()

    # ── normalisation ───────────────────────────────────────────────────────
    def unnormalize(self, rows, state=None):
        """real rows [n, dim] -> ABSOLUTE LIBERO actions [n, H, 7] (pos, axis-angle, gripper +-1), the authors' client's
        post-processing; `state` is unused (the targets are absolute)."""
        R = np.asarray(rows, np.float32).reshape(-1, self.H, self.a)
        return XVLAActionCodec().unnormalize(R)

    def describe(self):
        return (f"XVLABackbone({self.checkpoint_dir}: H={self.H}, A={self.A}, real={self.a}, dim={self.dim}, steps={self.num_steps}, "
                f"sampler={self.sampler}, n_params={self.n_params:,}, expert={sum(p.numel() for p in self.expert.parameters()):,}, "
                f"load {self.load_secs:.1f}s)")
