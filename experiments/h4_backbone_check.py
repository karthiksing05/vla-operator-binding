"""Plan H4: a new backbone (GR00T N1.5 / FLOWER) against its own reference sampler, and the interface pieces our system calls.

    python experiments/h4_backbone_check.py groot|flower [checkpoint]

For each of 3 demonstration frames (physical-intelligence/libero, the backbone's training tasks):
  * parity: the policy's own sampler (GR00T: GR00TN15.get_action; FLOWER: FLOWERVLA.forward) vs our PolicyBackbone.sample on the same
    initial noise, in dataset action units -> max |difference| (bf16 accumulates differences for GR00T);
  * the chunk's distance to the demonstration's next H actions (a sanity check that the decoding is in the right units);
  * text_span: the tokens at the instruction slot decode to the instruction; cond_hook sees every encoder layer once per prefix;
  * a TLI shift of 0 leaves the prefix unchanged; the flow loss at the demonstration chunk; the time per query (prefix + sample).
Writes results/robot/h4_check_<name>.json.
"""
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")


def main():
    name = sys.argv[1]
    ck = sys.argv[2] if len(sys.argv) > 2 else None
    from lifelong_ood.policy import data as D
    if name == "groot":
        from lifelong_ood.policy.backbone_groot import GrootBackbone
        bk = GrootBackbone(ck)
        tasks = [10, 20, 30]                      # one goal, one object, one spatial task (all in its 40)
    else:
        from lifelong_ood.policy.backbone_flower import FlowerBackbone
        bk = FlowerBackbone(ck)
        tasks = [10, 14, 17]                      # LIBERO-Goal tasks (flower_libero_goal's own suite)
    print(bk.describe(), flush=True)
    demos = D.Demos()
    norm = D.Norm.for_backbone(bk)
    out = dict(backbone=bk.describe(), weights=getattr(bk, "weights_check", None) or getattr(bk, "weights_fix", None), frames=[])
    print("weights", out["weights"], flush=True)
    for task in tasks:
        e = demos.episodes(task)[0]
        ep = demos.load(e, images=True)
        i = 20
        o = demos.observation(ep, i)
        ob = bk.observation(o)
        rec = dict(task=task, episode=int(e), frame=i, prompt=o["prompt"])
        # the reference sampler on its own noise draw, then ours on the same noise
        torch.manual_seed(123)
        if name == "groot":
            noise = torch.randn((1, bk.H, bk.A), dtype=torch.bfloat16, device=bk.device)
            torch.manual_seed(123)
            batch = bk._one(o)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                ref = bk.gm.get_action({k: v for k, v in batch.items() if k in ("state", "state_mask", "embodiment_id") or k.startswith("eagle_")})
            ref = ref["action_pred"].float()[0, :, : bk.a].cpu().numpy()
            ref = bk.action_codec().unnormalize(ref)
        else:
            noise = torch.randn((1, bk.H, bk.A), device=bk.device)
            torch.manual_seed(123)
            with torch.no_grad():
                ref = bk.model({"rgb_obs": {"rgb_static": ob.static, "rgb_gripper": ob.gripper}}, {"lang_text": o["prompt"]})
            ref = ref[0].float().cpu().numpy()
        t0 = time.time()
        pre = bk.prefix(ob)
        rows = bk.sample(1, pre, noise=noise.float().reshape(1, -1).cpu().numpy())
        torch.cuda.synchronize()
        rec["secs_per_query"] = time.time() - t0
        ours = bk.unnormalize(rows)[0]
        rec["parity_max_abs"] = float(np.abs(ours - ref).max())
        rec["parity_mean_abs"] = float(np.abs(ours - ref).mean())
        demo = np.asarray(ep["actions"][i:i + bk.H], np.float32)
        k = len(demo)
        rec["demo_mae_ours"] = float(np.abs(ours[:k] - demo).mean())
        rec["demo_mae_zero"] = float(np.abs(demo).mean())
        rec["ours_first"] = np.round(ours[0], 3).tolist()
        rec["demo_first"] = np.round(demo[0], 3).tolist()
        # text_span: which tokens sit in the instruction slot
        off, nv = bk.text_span(ob)
        seen = {}

        def f(li, h, span):
            seen[li] = (h.shape[1], span is not None)
            return None
        hs = bk.cond_hook(f)
        try:
            pre2 = bk.prefix(ob)
        finally:
            for h in hs:
                h.remove()
        rec["cond_layers"] = len(bk.cond_layers())
        rec["hook_layers_seen"] = len(seen)
        rec["hook_seq_len"] = sorted({v[0] for v in seen.values()})
        rec["span"] = [int(off), nv.tolist()]
        if name == "groot":
            ids = ob.eagle["eagle_input_ids"][0].tolist()
            rec["span_text"] = bk._tok.decode(ids[off:off + int(nv[0])])
        else:
            m = bk.model
            p = m.construct_prompts({"lang_text": [o["prompt"]]})[0]
            tid = m.tokenizer([p], padding=True, truncation=True, max_length=77)["input_ids"][0]
            j0 = off - (2 * bk._img_tok + 1)
            rec["span_text"] = m.tokenizer.decode(tid[j0:j0 + int(nv[0])])
            rec["image_tokens_per_view"] = int(bk._img_tok)
        # a zero shift leaves the conditioning unchanged (the hook path itself is inert)
        hs = bk.cond_hook(lambda li, h, span: h.clone())
        try:
            pre3 = bk.prefix(ob)
        finally:
            for h in hs:
                h.remove()
        key = "vl" if name == "groot" else "features"
        rec["zero_shift_max_abs"] = float((pre3.kv[key].float() - pre2.kv[key].float()).abs().max())
        # the flow loss at the demonstration chunk (normalised rows), t = 0.5
        chunk = demo if k == bk.H else np.concatenate([demo, np.repeat(demo[-1:], bk.H - k, 0)])
        a = np.zeros((1, bk.H, bk.A), np.float32)
        a[0, :, : bk.a] = norm.normalize(chunk)
        a = torch.as_tensor(a, device=bk.device)
        g = torch.Generator(device="cpu").manual_seed(0)
        eps = torch.randn(a.shape, generator=g).to(bk.device)
        tt = torch.full((1,), 0.5, device=bk.device)
        xt = 0.5 * eps + 0.5 * a
        with torch.no_grad():
            v = bk.v_full(xt, tt, pre)
        from lifelong_ood.policy.adapt import flow_loss
        rec["flow_loss_t05"] = float(flow_loss(bk, v, (eps - a).reshape(1, -1), xt, tt))
        rec["expert_linear_layers"] = sum(isinstance(mm, torch.nn.Linear) for mm in bk.expert.modules())
        print(json.dumps(rec), flush=True)
        out["frames"].append(rec)
    json.dump(out, open(os.path.join(ROOT, "results", "robot", f"h4_check_{name}.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
