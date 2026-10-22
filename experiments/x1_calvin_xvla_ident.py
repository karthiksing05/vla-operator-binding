"""X1: the identity read of x1_calvin_unsup.py with X-VLA-CALVIN (2toINF/X-VLA-Calvin-ABC_D), on the SAME scene-D sequences.

A light adapter, no PolicyBackbone wrapper: the authors' own model class (X-VLA repo models.modeling_xvla.XVLA) and processor
(XVLAProcessor), in their server venv (/coc/flash10/ksingara3/robot/calvin/.venv-xvla).  CALVIN conventions are the authors'
client's (evaluation/calvin/calvin_client.py): image0 = rgb_static, image1 = rgb_gripper (unflipped), domain_id 2; actions are
ABSOLUTE end-effector targets [pos 3, rot6d 6 (euler_xyz_to_rotate6D: R[:, :2] row-major), gripper 1] + 10 zeros, H = 30; the
gripper target is the close probability (the client closes, -1, when it is >= 0.8), so target = (1 - a_gripper) / 2 from
CALVIN's absolute `actions` (+1 open, -1 close); proprio = the previous frame's absolute action in the same layout (the client
feeds back the last executed target).  Loss: X-VLA's training objective per sample (EE6DActionSpace.compute_loss: MSE x 500 on
positions, x 10 on rot6d, BCE-with-logits on the gripper channels) on a_hat = transformer(x_t), x_t = t eps + (1 - t) a, with
K stratified t (the repo's forward) and shared eps per frame.  Frames: every `stride`-th frame f of a segment with f + 29 inside
the segment where possible (else the segment's first frame), so the 30-step target mostly stays in its own subtask.

python experiments/x1_calvin_xvla_ident.py --xvla_root <X-VLA repo> --ckpt_dir <snapshot> --data <task_ABC_D> --seqs <unsup_sceneD.json>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import Counter

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def rot6d(euler):
    from scipy.spatial.transform import Rotation as R
    e = np.asarray(euler, np.float64)
    return R.from_euler("xyz", e, degrees=False).as_matrix()[..., :, :2].reshape(e.shape[:-1] + (6,))


def ee20(act):
    """CALVIN absolute actions [n, 7] (pos, euler, gripper +-1) -> X-VLA rows [n, 20]."""
    act = np.asarray(act, np.float64)
    r = np.concatenate([act[:, :3], rot6d(act[:, 3:6]), (1.0 - act[:, 6:7]) / 2.0], axis=1)
    return np.concatenate([r, np.zeros_like(r)], axis=1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xvla_root", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--seqs", required=True, help="an x1_calvin_unsup.py output json (its `sequences`)")
    ap.add_argument("--annotations", required=True, help="new_playtable_validation.yaml")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first n sequences only")
    ap.add_argument("--out", default=str(ROOT / "results/x1_calvin/xvla_ident_sceneD.json"))
    a = ap.parse_args()

    import torch
    import yaml
    from PIL import Image
    sys.path.insert(0, a.xvla_root)
    from models.modeling_xvla import XVLA
    from models.processing_xvla import XVLAProcessor

    proc = XVLAProcessor.from_pretrained(a.ckpt_dir)
    model = XVLA.from_pretrained(a.ckpt_dir, trust_remote_code=True, torch_dtype=torch.float32).to("cuda").to(torch.float32).eval()
    sp = model.action_space
    ann = yaml.safe_load(open(a.annotations, encoding="utf-8"))
    names = sorted(ann)
    prompts = [str(ann[t][0]).split("\n")[0].replace("’", "'") for t in names]
    P, K, H = len(prompts), a.K, int(model.num_actions)
    seqs = json.load(open(a.seqs, encoding="utf-8"))["sequences"]
    if a.limit:
        seqs = seqs[:a.limit]
    tr = pathlib.Path(a.data) / a.split
    print(f"[xvla-id] {len(seqs)} sequences, {P} prompts, H={H}, K={K}", flush=True)

    def per_sample_loss(pred, target):
        bce = torch.nn.functional.binary_cross_entropy_with_logits
        g = sum(bce(pred[:, :, gi], target[:, :, gi], reduction="none").mean(1) for gi in sp.gripper_idx) / len(sp.gripper_idx)
        pos = sum(((pred[:, :, list(ix)] - target[:, :, list(ix)]) ** 2).mean((1, 2)) for ix in (sp.POS_IDX_1, sp.POS_IDX_2))
        rot = sum(((pred[:, :, list(ix)] - target[:, :, list(ix)]) ** 2).mean((1, 2)) for ix in (sp.ROT_IDX_1, sp.ROT_IDX_2))
        return pos * sp.XYZ_SCALE + rot * sp.ROT_SCALE + g * sp.GRIPPER_SCALE

    segs_out, t0 = [], time.time()
    for si, s in enumerate(seqs):
        for sg, true in ((s["seg1"], s["tasks"][0]), (s["seg2"], s["tasks"][1])):
            lo, hi = sg
            last = max(lo, hi - (H - 1))
            frames = list(range(lo, last + 1, a.stride)) or [lo]
            L = np.zeros((len(frames), P, K), np.float32)
            for q, f in enumerate(frames):
                try:
                    ep = [np.load(tr / f"episode_{g:07d}.npz") for g in range(f - 1, f + H)]
                except FileNotFoundError:
                    L[q] = np.nan
                    continue
                acts = np.stack([e["actions"] for e in ep])                       # [H + 1, 7]
                target = torch.as_tensor(ee20(acts[1:]), device="cuda")[None]     # [1, H, 20]
                proprio = torch.as_tensor(ee20(acts[:1])[0], device="cuda")[None]
                imgs = [Image.fromarray(ep[1]["rgb_static"]), Image.fromarray(ep[1]["rgb_gripper"])]
                ins = [proc(imgs, p) for p in prompts]
                input_ids = torch.cat([x["input_ids"] for x in ins]).cuda()
                image_input = torch.cat([x["image_input"] for x in ins]).cuda().float()
                image_mask = torch.cat([x["image_mask"] for x in ins]).cuda()
                gen = torch.Generator().manual_seed(7919 * si + f)
                tt = ((torch.rand(1, generator=gen) + torch.arange(K) / K) % (1 - 1e-5)).cuda()
                eps = torch.randn((K, H, 20), generator=gen).cuda()
                xt = eps * tt[:, None, None] + target * (1 - tt[:, None, None])         # [K, H, 20]
                with torch.no_grad():
                    enc = model.forward_vlm(input_ids, image_input, image_mask)
                    rep = lambda z: z.repeat_interleave(K, dim=0)                         # noqa: E731  row p*K + k
                    pm, xm = sp.preprocess(proprio.expand(P * K, -1), xt.repeat(P, 1, 1))
                    pred = model.transformer(domain_id=torch.full((P * K,), 2, dtype=torch.long, device="cuda"),
                                             action_with_noise=xm, proprio=pm, t=tt.repeat(P),
                                             vlm_features=rep(enc["vlm_features"]), aux_visual_inputs=rep(enc["aux_visual_inputs"]))
                    L[q] = per_sample_loss(pred.float(), target.expand(P * K, -1, -1)).reshape(P, K).cpu().numpy()
            m = np.nanmean(L, axis=(0, 2))
            ti = names.index(true)
            segs_out.append(dict(seq=si, true=true, pred=names[int(m.argmin())], top1=int(m.argmin() == ti),
                                 rank=int((m < m[ti]).sum()), n_frames=len(frames),
                                 frame_top1=float(np.mean(np.nanmean(L, 2).argmin(1) == ti))))
        if si % 20 == 0:
            print(f"[xvla-id] {si + 1}/{len(seqs)} {time.time() - t0:.0f}s top1 {np.mean([x['top1'] for x in segs_out]):.3f}",
                  flush=True)
    res = dict(n_segments=len(segs_out), top1=float(np.mean([x["top1"] for x in segs_out])), chance=1 / P,
               median_rank=float(np.median([x["rank"] for x in segs_out])), mean_rank=float(np.mean([x["rank"] for x in segs_out])),
               frame_top1=float(np.mean([x["frame_top1"] for x in segs_out])),
               by_task={t: dict(n=sum(x["true"] == t for x in segs_out), top1=float(np.mean([x["top1"] for x in segs_out if x["true"] == t])))
                        for t in sorted({x["true"] for x in segs_out})},
               confusions=Counter(f"{x['true']} -> {x['pred']}" for x in segs_out if not x["top1"]).most_common(15))
    pathlib.Path(a.out).write_text(json.dumps(dict(res, segs=segs_out, args=vars(a)), indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1), flush=True)
    if a.record:
        sys.path.insert(0, str(ROOT))
        from lifelong_ood import ledger
        p = ledger.record("x1_calvin_unsup", bench=f"calvin_{a.split}", tag="xvla_calvin", seed=0, metrics=dict(identity=res),
                          keys=dict(scene="D"), bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                          config=vars(a), script="experiments/x1_calvin_xvla_ident.py", command=" ".join(sys.argv),
                          seconds=time.time() - t0)
        print("ledger:", p)


if __name__ == "__main__":
    main()
