"""X2 graded Bridge tasks (docs/SIMPLER.md, 'The graded split against all of Bridge V2'): six SimplerEnv tasks, stock protocol.

    G1 relation (both parts named in Bridge V2, never together):
        X2G1GreenBlockOnPlate-v0   green block -> plate        table scene, 60 steps
        X2G1CarrotOnYellowBlock-v0 carrot -> yellow block      table scene, 60 steps
        X2G1CarrotInBasket-v0      carrot -> basket            sink scene (basket = overlay photo + invisible plane), 120 steps
    G4 part (object never named in Bridge V2; Google-robot assets on the Bridge table, standing upright):
        X2G4PepsiOnTowel-v0        Pepsi can -> towel          table scene, 60 steps
        X2G4RedBullOnPlate-v0      Red Bull can -> plate       table scene, 60 steps
        X2G4SevenUpOnTowel-v0      7up can -> towel            table scene, 60 steps

Every task subclasses the stock put-on env (no downloaded file changes), keeps the stock success checker and the stock 24-layout
grid of the carrot/spoon tasks (4 corners x ordered pairs x 2 yaws; the sink task keeps the eggplant's 8 x 3 layouts), and
the stock overlay / camera / controller of whichever SimplerEnv fork is installed (use the allenzren fork = stock scenes).
The cans are the 'opened_*' Google-robot models (density 50, as in the stock coke-can tasks; the unopened models are 1000
kg/m^3, about 0.4 kg, above the WidowX 250's payload), stood upright (Google's 'upright' quat) with the yaw on top.

Import this module (after `import simpler_env`) to register the envs; GRADED maps env id -> metadata.
Validation (no policy; GPU node, SAPIEN renders through Vulkan):
    python -I experiments/x2_simpler_graded.py OUT_DIR
resets every layout, saves the first frame, records each object's settled pose (can uprightness), and checks the stock
success checker: False at reset, True after the source is placed resting on the target (then 0.4 s of physics).
"""
import json
import os
import sys

import numpy as np
import simpler_env  # noqa: F401  (first: initialises mani_skill2_real2sim.envs; registration imports it circularly)
import sapien.core as sapien
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult, quat2mat
from mani_skill2_real2sim import ASSET_DIR
from mani_skill2_real2sim.utils.registration import register_env
from mani_skill2_real2sim.utils.io_utils import load_json
from mani_skill2_real2sim.envs.custom_scenes.put_on_in_scene import PutOnBridgeInSceneEnv, PutEggplantInBasketScene

PICK_DB = load_json(ASSET_DIR / "custom/info_pick_custom_v0.json")   # Google-robot model metadata (bbox, density)
CANS = ("opened_pepsi_can", "opened_redbull_can", "opened_7up_can")
UPRIGHT = euler2quat(np.pi / 2, 0, 0)                                   # GraspSingleInSceneEnv.orientations_dict["upright"]


def table_grid():
    """The stock carrot/spoon layout: 4 grid corners around (-0.16, 0), every ordered (source, target) pair -> 12 xy configs."""
    c, h = np.array([-0.16, 0.00]), np.array([0.075, 0.075])
    g = (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1) * h[None] + c[None]
    return [np.array([a, b]) for i, a in enumerate(g) for j, b in enumerate(g) if i != j]


def two_yaws(src):
    """The stock 2 source yaws (0, pi/2); a can is stood upright first (yaw applied about the world z after)."""
    base = UPRIGHT if src in CANS else np.array([1.0, 0, 0, 0])
    return [np.array([qmult(euler2quat(0, 0, y), base), [1, 0, 0, 0]]) for y in (0.0, np.pi / 2)]


GRADED = {}


def make_table_env(name, src, tgt, instr, grade, pair, completely_on=True):
    override = {src: PICK_DB[src]} if src in CANS else {}

    class _Env(PutOnBridgeInSceneEnv):
        def __init__(self, **kw):
            kw.setdefault("model_db_override", override)
            super().__init__(source_obj_name=src, target_obj_name=tgt, xy_configs=table_grid(), quat_configs=two_yaws(src), **kw)

        def evaluate(self, **kw):
            return super().evaluate(success_require_src_completely_on_target=completely_on, **kw)

        def get_language_instruction(self, **kw):
            return instr
    _Env.__name__ = name.replace("-", "_")
    register_env(name, max_episode_steps=60)(_Env)
    GRADED[name] = dict(grade=grade, pair=pair, source=src, target=tgt, instruction=instr, max_steps=60, scene="table")


class X2G1CarrotInBasket(PutEggplantInBasketScene):
    """The stock sink scene (basket = overlay photo + dummy target plane; success with z offset 0.06), the carrot as the source."""
    def reset(self, seed=None, options=None):
        # the stock __init__ hardcodes the eggplant; reset reads _source_obj_name into model_ids, and a changed model id makes
        # _set_model reconfigure the scene
        self._source_obj_name = "bridge_carrot_generated_modified"
        return super().reset(seed=seed, options=options)

    def get_language_instruction(self, **kw):
        return "put carrot into yellow basket"


make_table_env("X2G1GreenBlockOnPlate-v0", "green_cube_3cm", "bridge_plate_objaverse_larger", "put the green block on the plate",
               "G1", "green block -> plate")
make_table_env("X2G1CarrotOnYellowBlock-v0", "bridge_carrot_generated_modified", "yellow_cube_3cm", "put carrot on the yellow block",
               "G1", "carrot -> yellow block")
register_env("X2G1CarrotInBasket-v0", max_episode_steps=120)(X2G1CarrotInBasket)
GRADED["X2G1CarrotInBasket-v0"] = dict(grade="G1", pair="carrot -> basket", source="bridge_carrot_generated_modified",
                                       target="dummy_sink_target_plane", instruction="put carrot into yellow basket", max_steps=120,
                                       scene="sink")
make_table_env("X2G4PepsiOnTowel-v0", "opened_pepsi_can", "table_cloth_generated_shorter", "put the pepsi can on the towel",
               "G4", "pepsi can -> towel")
make_table_env("X2G4RedBullOnPlate-v0", "opened_redbull_can", "bridge_plate_objaverse_larger", "put the red bull can on the plate",
               "G4", "redbull can -> plate")
make_table_env("X2G4SevenUpOnTowel-v0", "opened_7up_can", "table_cloth_generated_shorter", "put the 7up can on the towel",
               "G4", "7up can -> towel")


def make(name):
    """The env as the stock evaluators build it (prepackaged visual-matching config, rgbd obs)."""
    import gymnasium as gym                      # the allenzren fork's simpler_env.make takes no kwargs
    return gym.make(name, obs_mode="rgbd", prepackaged_config=True)


# ── validation ────────────────────────────────────────────────────────────────────────────────────────────────────────
def place_check(env):
    u = env.unwrapped
    src, tgt = u.episode_source_obj, u.episode_target_obj
    up = quat2mat(src.pose.q)[:, 1] if u.episode_source_obj.name in CANS else None     # a can's long axis is its model y
    before = bool(u.evaluate()["success"])
    settled = dict(src_z=float(src.pose.p[2]), can_up_cos=None if up is None else float(abs(up[2])))
    zt = abs(u.episode_target_obj_bbox_world[2]) / 2 + abs(u.episode_source_obj_bbox_world[2]) / 2
    src.set_pose(sapien.Pose(tgt.pose.p + np.array([0, 0, zt + 0.005]), src.pose.q))
    src.set_velocity(np.zeros(3))
    src.set_angular_velocity(np.zeros(3))
    for _ in range(200):                      # 0.4 s of physics at 500 Hz, robot holding still
        u._scene.step()
    ev = u.evaluate()
    return before, bool(ev["success"]), settled


def main(out):
    from PIL import Image
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    os.makedirs(out, exist_ok=True)
    rows = {}
    for name, meta in GRADED.items():
        env = make(name)
        n = len(env.unwrapped._xy_configs) * len(env.unwrapped._quat_configs)
        res = []
        for ep in range(n):
            obs, info = env.reset(options={"obj_init_options": {"episode_id": ep}})
            if ep in (0, 1):
                Image.fromarray(get_image_from_maniskill2_obs_dict(env, obs)).save(f"{out}/{name}_ep{ep}.png")
            res.append(place_check(env))
        ups = [r[2]["can_up_cos"] for r in res if r[2]["can_up_cos"] is not None]
        rows[name] = dict(meta, layouts=n, max_episode_steps=env.spec.max_episode_steps,
                          success_at_reset=sum(r[0] for r in res), success_after_placement=sum(r[1] for r in res),
                          src_z_settled=[round(min(r[2]["src_z"] for r in res), 4), round(max(r[2]["src_z"] for r in res), 4)],
                          can_upright_min_cos=round(min(ups), 4) if ups else None,
                          control_mode=getattr(env.unwrapped, "_control_mode", None))
        print("[graded]", name, json.dumps(rows[name]), flush=True)
        env.close()
    json.dump(rows, open(f"{out}/graded_check.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
