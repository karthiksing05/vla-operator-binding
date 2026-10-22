"""Plan L, L0a input (docs/PLAN_LANG.md): first observations of Li's 20 OOD tasks for the lexicon audit, in Li's modified LIBERO.

Start states: practice set a (results/robot/fresh_ood_a, 10 per task; never a scoring set).  Each state as the client starts an
episode (experiments/libero_client.py): reset, set_init_state, 10 settle steps of the DUMMY action; images flipped and padded to
224 (experiments/f0b_firstframes_seg.py's convention), the 8-d state, every object's and fixture's position.
-> results/lang/frames/<suite>_<task>.npz (img, wimg, state8, pos, names, kinds, language).  Resumable.

    sbatch --export=ALL,ENV=libero,SCRIPT=experiments/pl0_render.py slurm/jobs/robot_gpu.sbatch       (py3.8, the LIBERO venv)
"""
import os
import pathlib
import sys

sys.path.insert(0, "/coc/flash10/ksingara3/ext_dl/pi0tl/pi0-text-latent/third_party/modified_libero")
os.environ["LIBERO_CONFIG_PATH"] = "/coc/flash10/ksingara3/robot/libero_ood_config"
import numpy as np                                                                       # noqa: E402
from libero.libero import benchmark, get_libero_path                                     # noqa: E402
from libero.libero.envs import OffScreenRenderEnv                                        # noqa: E402
from openpi_client import image_tools                                                    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "lang", "frames")
FRESH = os.path.join(ROOT, "results", "robot", "fresh_ood_a")
DUMMY = [0.0] * 6 + [-1.0]


def quat2axisangle(quat):
    q = np.asarray(quat, np.float64).copy()
    q[3] = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(1.0 - q[3] * q[3])
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * np.arccos(q[3])) / den


def main():
    os.makedirs(OUT, exist_ok=True)
    for sname in ("libero_goal_ood", "libero_spatial_ood"):
        suite = benchmark.get_benchmark_dict()[sname]()
        for tid in range(suite.n_tasks):
            out = os.path.join(OUT, "%s_%d.npz" % (sname, tid))
            if os.path.exists(out):
                continue
            task = suite.get_task(tid)
            states = np.load(os.path.join(FRESH, "%s_%d.npy" % (sname, tid)))
            bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
            env.seed(0)
            inner = env.env
            names = list(inner.objects_dict.keys()) + list(inner.fixtures_dict.keys())
            kinds = ["object"] * len(inner.objects_dict) + ["fixture"] * len(inner.fixtures_dict)
            rec = {k: [] for k in ("img", "wimg", "state8", "pos")}
            for s in states:
                env.reset()
                obs = env.set_init_state(s)
                for _ in range(10):
                    obs, _, _, _ = env.step(DUMMY)
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wimg = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                rec["img"].append(image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224)))
                rec["wimg"].append(image_tools.convert_to_uint8(image_tools.resize_with_pad(wimg, 224, 224)))
                rec["state8"].append(np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                                     obs["robot0_gripper_qpos"])).astype(np.float32))
                sim = inner.sim
                rec["pos"].append(np.asarray([np.asarray(obs[n + "_pos"], np.float64) if (n + "_pos") in obs
                                              else np.array(sim.data.body_xpos[inner.obj_body_id[n]]) for n in names], np.float32))
            tmp = out + ".%d.tmp.npz" % os.getpid()
            np.savez_compressed(tmp, names=np.array(names), kinds=np.array(kinds), language=task.language,
                                **{k: np.stack(v) for k, v in rec.items()})
            os.replace(tmp, out)
            env.close()
            print("[pl0_render] %s %d '%s': %d states, %d bodies" % (sname, tid, task.language, len(states), len(names)), flush=True)


if __name__ == "__main__":
    main()
