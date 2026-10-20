"""X2 step 5 feasibility (no policy): recombined SimplerEnv Bridge tasks built by subclassing the stock put-on envs.

Registers unseen object x target pairs in the two Bridge scenes, without touching the downloaded SimplerEnv code:
  table scene (bridge_table_1_v1, overlay bridge_real_eval_1.png): carrot->towel, spoon->plate, eggplant->plate,
      green cube->plate, carrot->yellow cube (a hard 'stack' relation on a novel source);
  sink scene (bridge_table_1_v2, overlay bridge_sink.png): carrot->basket (the basket is the overlay photo + an invisible
      target plane, exactly as in the stock eggplant task).
For each: reset over the 24 (or 24) layouts, render the first frame, and check the stock success checker by TELEPORTING the
source object (a) at its initial pose -> success must be False, (b) resting on the target -> success must be True.

    python -I experiments/x2_simpler_recombine.py OUT_DIR      (on a GPU node: SAPIEN renders through Vulkan)
"""
import json
import os
import sys

import numpy as np
import simpler_env  # first: initialises mani_skill2_real2sim.envs (registration imports it circularly)
import sapien.core as sapien
from transforms3d.euler import euler2quat
from mani_skill2_real2sim.utils.registration import register_env
from mani_skill2_real2sim.envs.custom_scenes.put_on_in_scene import (PutOnBridgeInSceneEnv, PutEggplantInBasketScene)
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict


def table_grid():
    """The stock carrot/spoon layout: 4 grid corners around (-0.16, 0), every ordered (source, target) pair -> 12 xy configs."""
    c, h = np.array([-0.16, 0.00]), np.array([0.075, 0.075])
    g = (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1) * h[None] + c[None]
    return [np.array([a, b]) for i, a in enumerate(g) for j, b in enumerate(g) if i != j]


TWO_YAWS = [np.array([[1, 0, 0, 0], [1, 0, 0, 0]]), np.array([euler2quat(0, 0, np.pi / 2), [1, 0, 0, 0]])]


def make_table_env(name, src, tgt, instr, require_completely_on=True):
    class _Env(PutOnBridgeInSceneEnv):
        def __init__(self, **kw):
            super().__init__(source_obj_name=src, target_obj_name=tgt, xy_configs=table_grid(), quat_configs=TWO_YAWS, **kw)

        def evaluate(self, **kw):
            return super().evaluate(success_require_src_completely_on_target=require_completely_on, **kw)

        def get_language_instruction(self, **kw):
            return instr
    _Env.__name__ = name.replace("-", "_")
    register_env(name, max_episode_steps=60)(_Env)
    return name


class PutCarrotInBasketScene(PutEggplantInBasketScene):
    """The stock sink scene (basket = overlay photo + dummy target plane, success with z offset 0.06), carrot as the source."""
    def reset(self, seed=None, options=None):
        # the stock __init__ hardcodes the eggplant; reset reads _source_obj_name into model_ids, and a changed model id makes
        # _set_model reconfigure the scene
        self._source_obj_name = "bridge_carrot_generated_modified"
        return super().reset(seed=seed, options=options)

    def get_language_instruction(self, **kw):
        return "put carrot into yellow basket"


register_env("X2PutCarrotInBasketScene-v0", max_episode_steps=120)(PutCarrotInBasketScene)

TASKS = [
    make_table_env("X2PutCarrotOnTowel-v0", "bridge_carrot_generated_modified", "table_cloth_generated_shorter", "put carrot on towel", False),
    make_table_env("X2PutSpoonOnPlate-v0", "bridge_spoon_generated_modified", "bridge_plate_objaverse_larger", "put the spoon on the plate"),
    make_table_env("X2PutEggplantOnPlate-v0", "eggplant", "bridge_plate_objaverse_larger", "put eggplant on plate"),
    make_table_env("X2PutGreenCubeOnPlate-v0", "green_cube_3cm", "bridge_plate_objaverse_larger", "put the green block on the plate"),
    make_table_env("X2PutCarrotOnYellowCube-v0", "bridge_carrot_generated_modified", "yellow_cube_3cm", "put carrot on the yellow block"),
    "X2PutCarrotInBasketScene-v0",
]


def teleport_check(env):
    u = env.unwrapped
    src, tgt = u.episode_source_obj, u.episode_target_obj
    before = bool(u.evaluate()["success"])
    # rest the source just above the target's top face (bounding boxes in the world frame at t = 0), then let physics settle
    zt = u.episode_target_obj_bbox_world[2] / 2 + u.episode_source_obj_bbox_world[2] / 2
    src.set_pose(sapien.Pose(tgt.pose.p + np.array([0, 0, zt + 0.005]), src.pose.q))
    src.set_velocity(np.zeros(3)); src.set_angular_velocity(np.zeros(3))
    for _ in range(100):                      # 0.2 s of physics at 500 Hz, robot holding still
        u._scene.step()
    ev = u.evaluate()
    return before, bool(ev["success"]), {k: bool(v) for k, v in ev.items() if k != "episode_stats"}


def main(out):
    os.makedirs(out, exist_ok=True)
    rows = {}
    for name in TASKS:
        import gymnasium as gym                  # (the allenzren fork's simpler_env.make takes no kwargs)
        env = gym.make(name, obs_mode="rgbd", prepackaged_config=True)
        n = len(env.unwrapped._xy_configs) * len(env.unwrapped._quat_configs)
        res = []
        for ep in range(n):
            obs, info = env.reset(options={"obj_init_options": {"episode_id": ep}})
            if ep == 0:
                from PIL import Image
                Image.fromarray(get_image_from_maniskill2_obs_dict(env, obs)).save(f"{out}/{name}.png")
            res.append(teleport_check(env))
        rows[name] = dict(instruction=env.unwrapped.get_language_instruction(), layouts=n,
                          success_at_init=sum(r[0] for r in res), success_after_teleport=sum(r[1] for r in res),
                          example=res[0][2], control_mode=env.unwrapped._control_mode if hasattr(env.unwrapped, "_control_mode") else None)
        print("[recombine]", name, json.dumps(rows[name]), flush=True)
        env.close()
    json.dump(rows, open(f"{out}/recombine_check.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
