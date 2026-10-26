"""PLAN_SOLVE S2 smoke (docs/PLAN_S2.md): s0_serve.py's adapter server + a DEPLOYABLE binder inside it.  New file.

Request fields (absent -> exactly s0_serve.S0Policy / d5_serve.BatchPolicy):
  s2/obj_phrase, s2/goal_phrase   the parse's role phrases (S1's split); for every row and query the binder turns them into the
                                  role masks the adapter takes (instead of the oracle's s0/mobj, s0/mgoal)
Binder (BINDER = s1c | s1b; the one S2 pre-registers): GroundingDINO boxes for the role's BODY phrase on the row's agentview image
(s1b_detect's settings) -> the trained head picks a candidate and a crop mask (s1c_train / s1b_train) on the frozen policy's own
image tokens.  Mask rule (pre-registered): a role with an EMPTY relation -> every token inside the chosen box; otherwise the tokens
of the crop cells with sigmoid >= 0.5, plus the peak token.  No box -> a zero mask (the adapter then does nothing for that role).

    python experiments/s2_serve.py --port P --binder s1c
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
import s0_serve as S0S                                               # noqa: E402

LMAX = 12


class Binder:
    def __init__(self, bk, kind):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        import s1b_detect as SD
        import s1b_train as SB
        self.bk, self.kind, self.SD, self.SB, self.torch = bk, kind, SD, SB, torch
        self.proc = AutoProcessor.from_pretrained(SD.MODEL)
        self.det = AutoModelForZeroShotObjectDetection.from_pretrained(SD.MODEL).to(bk.device).eval()
        self.lex = json.load(open(os.path.join(ROOT, "results", "bind", "s1b", "det.json")))["lexicon"]
        Head = SB.build(torch)
        d = os.path.join(ROOT, "results", "bind", kind)
        self.heads = {}
        for role in ("object", "goal"):
            h = Head().to(bk.device)
            h.load_state_dict(torch.load(os.path.join(d, f"{role}.pt"), map_location=bk.device))
            self.heads[role] = h.eval()
        from openpi.models import tokenizer as _tok
        self.sp = _tok.PaligemmaTokenizer(200)._tokenizer
        self._emb = {}

    def emb(self, text):
        torch = self.torch
        if text not in self._emb:
            ids = self.sp.encode(text, add_bos=False)[:LMAX] if text else []
            e = torch.zeros((1, LMAX, 2048), device=self.bk.device)
            if ids:
                with torch.no_grad():
                    e[0, :len(ids)] = self.bk.model.paligemma_with_expert.embed_language_tokens(
                        torch.tensor([ids], device=self.bk.device))[0].float()
            self._emb[text] = (e, torch.tensor([len(ids)], device=self.bk.device))
        return self._emb[text]

    def boxes(self, img, body):
        from PIL import Image
        torch, SD = self.torch, self.SD
        im = Image.fromarray(img)
        inp = self.proc(images=im, text=body + ".", return_tensors="pt").to(self.bk.device)
        with torch.no_grad():
            o = self.det(**inp)
        r = self.proc.post_process_grounded_object_detection(o, inp.input_ids, threshold=SD.THR, text_threshold=SD.THR,
                                                             target_sizes=[im.size[::-1]])[0]
        bx = sorted(([*map(float, bb), float(s)] for bb, s in zip(r["boxes"], r["scores"])), key=lambda x: -x[4])
        keep = []
        for x in bx:
            if all(SD.iou(x, k) < SD.NMS for k in keep):
                keep.append(x)
            if len(keep) == SD.K_MAX:
                break
        return keep[:SD.K_MAX]

    def masks(self, img, tokens, role, phrase):
        """img [224,224,3] uint8, tokens [256, 2048] (the policy's agentview tokens), role phrase -> [256] float mask."""
        torch, SB, SD = self.torch, self.SB, self.SD
        body, rel = SD.split(phrase, self.lex)
        bx = self.boxes(img, body)
        out = np.zeros(256, np.float32)
        if not bx:
            return out, dict(body=body, rel=rel, n=0)
        cl = np.stack([SB.cells(b) for b in bx])
        K = len(bx)
        fm = tokens.float().T.reshape(1, 2048, 16, 16)
        grid = torch.tensor(cl / 224.0 * 2 - 1, device=self.bk.device, dtype=torch.float32).reshape(1, K * 64, 1, 2)
        crop = torch.nn.functional.grid_sample(fm, grid, align_corners=False).squeeze(-1).transpose(1, 2).reshape(1, K, 64, 2048)
        gm = torch.tensor(SB.geom(bx), device=self.bk.device)[None]
        e, ln = self.emb(phrase if self.kind == "s1c" else rel)
        with torch.no_grad():
            il, ml = self.heads[role](crop, gm, e, ln)
        c = int(il[0].argmax())
        mm = ml[0, c].cpu().numpy()
        if not rel or rel in ("on the", "in the"):
            x0, y0, x1, y1 = bx[c][:4]
            cc = (np.arange(16) + 0.5) * 14
            inside = ((cc[None, :] >= x0) & (cc[None, :] <= x1) & (cc[:, None] >= y0) & (cc[:, None] <= y1)).reshape(-1)
            out[inside] = 1.0
        else:
            out[SB.tok_of(cl[c][1 / (1 + np.exp(-mm)) >= 0.5])] = 1.0
            out[int(SB.tok_of(cl[c][int(mm.argmax())]))] = 1.0
        return out, dict(body=body, rel=rel, n=K, cand=c)


class S2Policy(S0S.S0Policy):
    def __init__(self, bk, adapter, binder, num_steps=10):
        super().__init__(bk, adapter, num_steps=num_steps)
        self.binder = binder

    def infer(self, obs, *, noise=None):
        if "s2/obj_phrase" not in obs:
            return super().infer(obs, noise=noise)
        import torch
        I = np.asarray(obs["b/image"])
        ods = [{"observation/image": I[q], "observation/wrist_image": np.asarray(obs["b/wrist"])[q],
                "observation/state": np.asarray(obs["b/state"], np.float32)[q], "prompt": str(obs["prompt"])} for q in range(len(I))]
        m = self.bk.model
        MO, MG = [], []
        for g0 in range(0, len(I), 16):
            with torch.no_grad():
                images, img_masks, lt, lm, _ = m._preprocess_observation(self.bk.observations(ods[g0:g0 + 16]), train=False)
                embs, _, _ = m.embed_prefix(images, img_masks, lt, lm)
            for j in range(embs.shape[0]):
                q = g0 + j
                mo, _ = self.binder.masks(I[q], embs[j, :256], "object", str(obs["s2/obj_phrase"]))
                mg, _ = self.binder.masks(I[q], embs[j, :256], "goal", str(obs["s2/goal_phrase"]))
                MO.append(mo)
                MG.append(mg)
        o2 = {k: v for k, v in obs.items() if not k.startswith("s2/")}
        o2["s0/mobj"], o2["s0/mgoal"] = np.stack(MO), np.stack(MG)
        out = super().infer(o2, noise=noise)
        out["s2/mobj"], out["s2/mgoal"] = o2["s0/mobj"], o2["s0/mgoal"]
        return out


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--binder", default=os.environ.get("S2_BINDER", "s1c"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from lifelong_ood.policy.backbone import PolicyBackbone
    from lifelong_ood.policy.serve import serve, warm
    import s0_adapter as SA
    bk = PolicyBackbone(os.environ.get("BACKBONE_CONFIG") or "pi05_l90c", checkpoint_dir=os.environ.get("BACKBONE_CKPT") or None,
                        device="cuda")
    ad = SA.BindAdapter().to(bk.device)
    ad.load_state_dict(torch.load(os.path.join(ROOT, "results", "bind", "s0", "adapter.pt"), map_location=bk.device))
    ad.eval()
    pol = S2Policy(bk, ad, Binder(bk, a.binder))
    from openpi.policies import libero_policy
    ex = libero_policy.make_libero_example()
    ex["observation/state"] = np.asarray(ex["observation/state"], np.float32)
    b = {"b/image": np.stack([ex["observation/image"]] * 2), "b/wrist": np.stack([ex["observation/wrist_image"]] * 2),
         "b/state": np.stack([ex["observation/state"]] * 2), "b/seed": np.array([5, 6]), "prompt": ex["prompt"]}
    t0 = time.time()
    o = pol.infer(dict(b, **{"s2/obj_phrase": "black bowl", "s2/goal_phrase": "on the plate"}))
    logging.info(f"[s2 serve] binder {a.binder}: smoke masks obj {int(o['s2/mobj'].sum())} goal {int(o['s2/mgoal'].sum())} tokens, "
                 f"{time.time() - t0:.1f}s for 2 rows")
    warm(pol)
    serve(pol, a.port)


if __name__ == "__main__":
    main()
