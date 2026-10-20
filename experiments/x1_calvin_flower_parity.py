"""X1: FlowerCalvinBackbone parity with FLOWER's own CALVIN pipeline (SLURM GPU job).

Reference = the authors' model loaded with their load_mode_from_safetensor (+ the load fix, as the FIX=1 parity runs) and their
HulcWrapper observation transform; candidate = lifelong_ood.policy.backbone_flower_calvin.FlowerCalvinBackbone fed the RAW
calvin_env observation.  On frames from the first chains' initial states and after rolling the reference policy a few chunks,
with the same noise z: reference sample_actions(z, encode_observations(batch)) vs candidate sample(noise=z) -> unnormalize.
Also the same CUDA seed through the reference's own forward() (torch.randn inside) vs the candidate with randn drawn the same way.

python experiments/x1_calvin_flower_parity.py --flower_root <flower_vla_calvin> --ckpt_dir <snapshot> --val_dir <task_ABC_D/validation>
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flower_root", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--chunks", type=int, default=3, help="frames per chain: the initial one + after each executed chunk")
    ap.add_argument("--out", default=str(ROOT / "results/x1_calvin/flower_calvin_parity.json"))
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()

    import calvin_env  # noqa: F401  (the installed one, before the clone's empty calvin_env/ is on sys.path)
    sys.path.insert(0, a.flower_root)
    import hydra
    import numpy as np
    import torch
    import torchvision
    from omegaconf import OmegaConf
    from pytorch_lightning import seed_everything

    import flower.evaluation.utils as fu
    import flower.wrappers.hulc_wrapper as hw
    from calvin_env.utils.utils import EglDeviceNotFoundError
    from flower.evaluation.multistep_sequences import get_sequences
    from x1_calvin_flower_eval import _fix_weights

    def _no_probe(cuda_id):
        raise EglDeviceNotFoundError()
    hw.get_egl_device_id = _no_probe

    ckpt = pathlib.Path(a.ckpt_dir)
    cfg = OmegaConf.load(ckpt / "config.yaml")
    dev = torch.device("cuda:0")

    class _Val:
        abs_datasets_dir = pathlib.Path(a.val_dir)
        observation_space = cfg.datamodule.observation_space
        proprio_state = cfg.datamodule.proprioception_dims
        transforms = {c: torchvision.transforms.Compose([hydra.utils.instantiate(t) for t in cfg.datamodule.transforms.val[c]])
                      for c in cfg.datamodule.transforms.val}

    env = hw.HulcWrapper(_Val(), dev, show_gui=False)
    seed_everything(0, workers=True)
    ref = fu.load_mode_from_safetensor(ckpt / "model.safetensors", overwrite_cfg={"num_sampling_steps": 4})
    fixinfo = _fix_weights(ref, ckpt / "model.safetensors")
    ref.freeze()
    ref = ref.to(dev).eval()

    from lifelong_ood.policy.backbone_flower_calvin import FlowerCalvinBackbone
    wb = FlowerCalvinBackbone(checkpoint=str(ckpt), device="cuda")

    sd_r, sd_w = ref.state_dict(), wb.model.state_dict()
    wdiff = max(float((sd_r[k].float().cpu() - sd_w[k].float().cpu()).abs().max()) for k in sd_r)
    pdiff = float((ref.prompt_embeds.float().cpu() - wb.model.prompt_embeds.float().cpu()).abs().max())

    ann = OmegaConf.load(pathlib.Path(a.flower_root) / "conf/annotations/new_playtable_validation.yaml")
    seqs = get_sequences(a.chains)
    rows = []
    for ci, (init, seq) in enumerate(seqs):
        robot_obs, scene_obs = fu.get_env_state_for_initial_condition(init)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        text = ann[seq[0]][0]
        for c in range(a.chunks):
            obs = env.get_obs()
            raw = env.env.get_obs()
            d = dict(rgb_static=raw["rgb_obs"]["rgb_static"], rgb_gripper=raw["rgb_obs"]["rgb_gripper"],
                     robot_obs=raw["robot_obs"], prompt=text)
            o = wb.observation(d)
            img_diff = max(float((o.static[:, 0] - obs["rgb_obs"]["rgb_static"]).abs().max()),
                           float((o.gripper[:, 0] - obs["rgb_obs"]["rgb_gripper"]).abs().max()))
            g = torch.Generator().manual_seed(1000 * ci + c)
            z = torch.randn(1, 10, 7, generator=g).to(dev)
            with torch.no_grad():
                feats = ref.encode_observations({"rgb_obs": {"rgb_static": obs["rgb_obs"]["rgb_static"],
                                                             "rgb_gripper": obs["rgb_obs"]["rgb_gripper"]},
                                                 "lang_text": [text]})
                a_ref = ref.sample_actions(z.clone(), feats, inference=True).float().cpu().numpy().reshape(10, 7)
                a_wb = wb.unnormalize(wb.sample(1, wb.prefix(o), noise=z.reshape(1, 70).cpu().numpy())).reshape(10, 7)
                torch.manual_seed(77 + ci * 10 + c)
                a_fwd = ref(obs, {"lang_text": text}).float().cpu().numpy().reshape(10, 7)
                torch.manual_seed(77 + ci * 10 + c)
                z2 = torch.randn(1, 10, 7, device=dev)
                a_wb2 = wb.unnormalize(wb.sample(1, wb.prefix(o), noise=z2.reshape(1, 70).cpu().numpy())).reshape(10, 7)
            rows.append(dict(chain=ci, chunk=c, subtask=seq[0], img_maxdiff=img_diff,
                             same_noise_maxdiff=float(np.abs(a_ref - a_wb).max()),
                             same_seed_forward_maxdiff=float(np.abs(a_fwd - a_wb2).max()),
                             action_scale=float(np.abs(a_ref).max())))
            print(json.dumps(rows[-1]), flush=True)
            # advance the env with the reference's chunk (authors' step convention: the whole chunk, gripper binarised)
            for t in range(10):
                env.step(torch.as_tensor(a_ref[t]))
    out = dict(n_frames=len(rows), weights_maxdiff=wdiff, prompt_embed_maxdiff=pdiff, fix=fixinfo,
               max_same_noise=max(r["same_noise_maxdiff"] for r in rows),
               max_same_seed_forward=max(r["same_seed_forward_maxdiff"] for r in rows),
               max_img=max(r["img_maxdiff"] for r in rows), rows=rows, describe=wb.describe())
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, default=str), flush=True)
    if a.record:
        from lifelong_ood import ledger
        p = ledger.record("x1_calvin_parity", bench="calvin_abc_d", tag="flower_calvin_backbone", seed=0,
                          metrics={k: v for k, v in out.items() if k != "describe"},
                          bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                          config=dict(ckpt=str(ckpt), chains=a.chains, chunks=a.chunks),
                          script="experiments/x1_calvin_flower_parity.py", command=" ".join(sys.argv),
                          keys=dict(check="wrapper_vs_authors"))
        print("ledger:", p)


if __name__ == "__main__":
    main()
