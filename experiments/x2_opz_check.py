"""X2: OPZBackbone (lifelong_ood/policy/backbone_opz.py) against open-pi-zero's own pipeline, as h4_backbone_check.py did for GR00T/FLOWER.

On SimplerEnv frames (the 4 stock WidowX tasks, 2 layouts each, at reset and after 8 steps of the policy):
  * inputs: our `observations` vs the authors' BridgeSimplerAdapter.preprocess(env, obs, instruction) (ids, mask, pixels, proprio);
  * parity: PiZeroInference.forward (infer_action, seeded) vs our PolicyBackbone-Euler `sample` on the SAME noise -> max |diff| in
    normalised units and in SimplerEnv action units (after the same postprocess);
  * text_span decodes to <bos> instruction \\n; cond_hook fires once per Gemma layer per prefix; an identity hook leaves the chunk
    unchanged; Tweedie at t = 1/K vs the final chunk; the flow loss at the sampled chunk; the time per query (prefix + 10 steps).
    python -I experiments/x2_opz_check.py OUT.json [ckpt]      (.venv_pi0, GPU node)
"""
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    out_path = sys.argv[1]
    ck = sys.argv[2] if len(sys.argv) > 2 else "bridge_beta"
    from lifelong_ood.policy.backbone_opz import OPZBackbone
    import simpler_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    bk = OPZBackbone(ck)
    print(bk.describe(), flush=True)
    res = dict(backbone=bk.describe(), frames=[])
    for task in ["widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube", "widowx_put_eggplant_in_basket"]:
        env = simpler_env.make(task)
        for ep in (0, 13):
            obs, _ = env.reset(options={"obj_init_options": {"episode_id": ep}})
            instr = env.unwrapped.get_language_instruction()
            for phase in ("reset", "step8"):
                if phase == "step8":                       # advance 8 env steps with the policy (2 chunks)
                    for q in range(2):
                        img = get_image_from_maniskill2_obs_dict(env, obs)
                        o = bk.observation(bk.raw_observation(img, obs["agent"]["eef_pos"], instr))
                        for a in bk.unnormalize(bk.sample(1, bk.prefix(o), seed=q))[0]:
                            obs, *_ = env.step(a)
                img = get_image_from_maniskill2_obs_dict(env, obs)
                o = bk.observation(bk.raw_observation(img, obs["agent"]["eef_pos"], instr))
                theirs = bk.adapter.preprocess(env, obs, instr)
                rec = dict(task=task, episode=ep, phase=phase, instruction=instr,
                           ids_equal=bool(torch.equal(o.input_ids.cpu(), theirs["input_ids"])),
                           mask_equal=bool(torch.equal(o.attention_mask.cpu(), theirs["attention_mask"])),
                           pixel_maxdiff=float((o.pixel_values.cpu().float() - theirs["pixel_values"].float()).abs().max()),
                           proprio_maxdiff=float((o.proprios.cpu().float() - theirs["proprios"].float()).abs().max()))
                # parity on the same noise
                seed = 1000 + ep
                ref, noise = bk.reference_sample(o, seed)
                t0 = time.time()
                pre = bk.prefix(o)
                ours = bk.sample(1, pre, noise=noise.reshape(1, -1))
                torch.cuda.synchronize()
                rec["query_sec"] = round(time.time() - t0, 3)
                ours = ours.reshape(ref.shape)
                rec["parity_maxdiff_norm"] = float(np.abs(ours - ref).max())
                ea, eb = bk.unnormalize(ref.reshape(1, -1))[0], bk.unnormalize(ours.reshape(1, -1))[0]
                rec["parity_maxdiff_env"] = float(np.abs(ea - eb).max())
                rec["chunk_env"] = np.round(eb, 4).tolist()
                # text span
                off, n = bk.text_span(o)
                toks = o.input_ids[0, off:off + int(n[0])].tolist()
                rec["text_span_decoded"] = bk.adapter.tokenizer.decode(toks)
                # cond_hook: identity hook fires once per layer and changes nothing
                calls = []
                hs = bk.cond_hook(lambda i, h, span: calls.append((i, tuple(h.shape), None if span is None else int(span[0]))) or None)
                pre2 = bk.prefix(o)
                for hd in hs:
                    hd.remove()
                ours2 = bk.sample(1, pre2, noise=noise.reshape(1, -1)).reshape(ref.shape)
                rec["cond_hook_calls"] = len(calls)
                rec["cond_hook_layers"] = sorted(set(c[0] for c in calls)) == list(range(len(bk.cond_layers())))
                rec["identity_hook_maxdiff"] = float(np.abs(ours2 - ours).max())
                # identities: Tweedie at the last Euler step's input; flow loss at the sampled chunk
                x = torch.as_tensor(noise.reshape(1, -1), device=bk.device)
                xs = x.clone()
                K = bk.num_steps
                for k in range(K - 1):                      # x at t = 1/K
                    t = 1.0 - k / K
                    xs = xs - (1.0 / K) * bk.v_full(xs, torch.tensor([t], device=bk.device), pre)
                tw = (xs - (1.0 / K) * bk.v_full(xs, torch.tensor([1.0 / K], device=bk.device), pre)).cpu().numpy()
                rec["tweedie_last_vs_chunk_maxdiff"] = float(np.abs(np.clip(tw, -1, 1).reshape(ref.shape) - ours).max())
                a0 = torch.as_tensor(ours.reshape(1, -1), device=bk.device)
                g = torch.Generator(device="cpu").manual_seed(7)
                eps = torch.randn((16, bk.dim), generator=g).to(bk.device)
                tt = torch.linspace(0.05, 0.95, 16, device=bk.device)
                xt = tt[:, None] * eps + (1 - tt[:, None]) * a0
                v = bk.v_full(xt, tt, bk.repeat(pre, 16) if pre.n == 1 else pre)
                rec["flow_loss_at_chunk"] = float(bk.flow_loss(v, eps - a0, xt, tt))
                rec["flow_loss_at_shuffled_chunk"] = float(bk.flow_loss(
                    v, eps - a0[:, torch.randperm(bk.dim, generator=g).to(bk.device)], xt, tt))
                print("[check]", json.dumps({k: v for k, v in rec.items() if k != "chunk_env"}), flush=True)
                res["frames"].append(rec)
        env.close()
    F = res["frames"]
    res["summary"] = dict(n_frames=len(F), parity_maxdiff_norm=max(f["parity_maxdiff_norm"] for f in F),
                          parity_maxdiff_env=max(f["parity_maxdiff_env"] for f in F),
                          inputs_identical=all(f["ids_equal"] and f["mask_equal"] and f["pixel_maxdiff"] == 0 and f["proprio_maxdiff"] == 0
                                               for f in F),
                          identity_hook_maxdiff=max(f["identity_hook_maxdiff"] for f in F),
                          cond_hook_calls=sorted(set(f["cond_hook_calls"] for f in F)),
                          tweedie_last_vs_chunk_maxdiff=max(f["tweedie_last_vs_chunk_maxdiff"] for f in F),
                          flow_loss_at_chunk_mean=float(np.mean([f["flow_loss_at_chunk"] for f in F])),
                          flow_loss_at_shuffled_mean=float(np.mean([f["flow_loss_at_shuffled_chunk"] for f in F])),
                          query_sec_median=float(np.median([f["query_sec"] for f in F])))
    print("[check] summary", json.dumps(res["summary"]), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
