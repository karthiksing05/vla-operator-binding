"""X2 preflight: build one SimplerEnv WidowX env, reset, render, step once (checks SAPIEN + Vulkan on the node)."""
import sys, time
import numpy as np
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

task = sys.argv[1] if len(sys.argv) > 1 else "widowx_spoon_on_towel"
t = time.time()
env = simpler_env.make(task)
obs, info = env.reset(options={"obj_init_options": {"episode_id": 0}})
img = get_image_from_maniskill2_obs_dict(env, obs)
print("[probe]", task, "instruction:", env.get_language_instruction(), "image", img.shape, img.dtype, img.mean())
print("[probe] control mode:", env.unwrapped._control_mode if hasattr(env.unwrapped, "_control_mode") else env.unwrapped.control_mode)
print("[probe] tcp_pose", obs["extra"]["tcp_pose"], "base_pose", obs["agent"]["base_pose"])
print("[probe] max_episode_steps", env.spec.max_episode_steps, "reset+render s", round(time.time() - t, 2))
if len(sys.argv) > 2:                      # optional: save the first frame (used to compare the two SimplerEnv forks' scenes)
    from PIL import Image
    Image.fromarray(img).save(sys.argv[2])
    print("[probe] saved", sys.argv[2])
