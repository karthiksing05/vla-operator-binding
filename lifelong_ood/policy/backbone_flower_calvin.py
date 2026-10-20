"""FlowerCalvinBackbone: FLOWER's CALVIN ABC -> D checkpoint (mbreuss/flower_calvin_abc) as the same score field over action
chunks as PolicyBackbone.  A subclass of FlowerBackbone (backbone_flower.py, whose LIBERO behaviour is unchanged): the network
(FLOWERVLA), the flow (z_t = (1 - t) a + t eps, Euler from t = 1 in 4 steps, clamp), prefix / text_span / cond_hook / v_full /
flow_loss and the load fix (`_fix_weights`: the two tensors the HF export renamed, and the <Flow> prompt embedding set to the
checkpoint's token row) are inherited.  What differs is CALVIN's observation convention (docs/CALVIN.md, "Mapping the backbone
interface"), read from the checkpoint's config and the authors' HulcWrapper:
  * images: calvin_env's raw `rgb_static` (200 x 200) and `rgb_gripper` (84 x 84), uint8 HWC, AS RENDERED (no flip);
    datamodule.transforms.val = Resize(224, antialias) on uint8, / 255, CLIP mean / std;
  * prompt style `default`: the instruction alone (the LIBERO checkpoints use `minimal`, a robot description first);
  * no proprio (use_proprio = false); the state slot carries calvin_env's 15-D robot_obs, unused by the model;
  * actions: 7-D RELATIVE end-effector actions in [-1, 1] (pos 3, euler 3, gripper), chunk 10 executed whole (multistep 10);
    HulcWrapper binarises the gripper (> 0 -> +1, else -1) before env.step.
Observation dicts: {"rgb_static": uint8 [200, 200, 3], "rgb_gripper": uint8 [84, 84, 3], "robot_obs": [15], "prompt": str}
(env.get_obs()'s rgb_obs entries and robot_obs, plus the instruction).
Run in /coc/flash10/ksingara3/robot/calvin/.venv (torch 2.10, transformers 4.46.3, the FLOWER code from robot/ext/flower_vla_calvin).
"""
from __future__ import annotations

import numpy as np
import torch

from .backbone_flower import CLIP_MEAN, CLIP_STD, FlowerBackbone, FObservation

CKPT = "mbreuss/flower_calvin_abc"
IMG = 224


class FlowerCalvinBackbone(FlowerBackbone):
    kind = "flower_calvin"

    def __init__(self, checkpoint=None, device="cuda", num_steps=None):
        super().__init__(checkpoint=checkpoint or CKPT, device=device, num_steps=num_steps)
        import torchvision.transforms as T
        self._resize = T.Resize(IMG, antialias=True)
        assert str(self.model.vlm_prompt_style) == "default", self.model.vlm_prompt_style
        assert not self.model.use_proprio

    def _views(self, obs_dicts, key):
        ims = [torch.from_numpy(np.ascontiguousarray(np.asarray(o[key], np.uint8))).permute(2, 0, 1) for o in obs_dicts]
        # Resize on uint8 ON THE DEVICE (HulcWrapper moves the uint8 frame to the GPU before the val transforms; the CPU kernel
        # rounds differently by one grey level), then ScaleImageTensor and Normalize, as the val transforms
        x = self._resize(torch.stack(ims).to(self.device)).float() / 255.0
        m = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        s = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)
        return ((x - m) / s)[:, None]

    @torch.no_grad()
    def observations(self, obs_dicts):
        """A list of CALVIN observation dicts (keys above) -> one batched FObservation in FLOWER-CALVIN's conventions."""
        st = torch.as_tensor(np.stack([np.asarray(o["robot_obs"], np.float32) for o in obs_dicts]), device=self.device)
        return FObservation(static=self._views(obs_dicts, "rgb_static"), gripper=self._views(obs_dicts, "rgb_gripper"),
                            texts=[str(o["prompt"]) for o in obs_dicts], state=st)

    def unnormalize(self, rows, state=None):
        """real rows [n, dim] -> CALVIN relative actions [n, H, 7] (clamp(-1, 1), as sample_actions); the env wrapper binarises
        the gripper."""
        return super().unnormalize(rows, state)
