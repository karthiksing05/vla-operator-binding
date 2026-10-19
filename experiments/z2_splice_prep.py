"""Plan Z, track B (docs/PLAN_Z.md): prepare and check the word splice (lifelong_ood/policy/bind.py) on the 20 OOD tasks.

For each task and its practice-chosen sources (G10 stages 1 / 2 on this backbone): the phrase match in the new and source prompts,
the bindings (source own, new-on-source), cached; the shift's norm against the residual's at those tokens.  Checks: a delta splice
whose source prompt is the new prompt is exactly zero; positions decode to the phrase.

    sbatch ... SCRIPT="experiments/z2_splice_prep.py [--backbone pi05_libero]"
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")


def main():
    cfg = sys.argv[sys.argv.index("--backbone") + 1] if "--backbone" in sys.argv else "pi05_libero"
    import g4_ood as G
    from lifelong_ood.policy import bind as BD
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy.backbone import PolicyBackbone
    bk = PolicyBackbone(cfg, device="cuda")
    dm = D.Demos()
    tx = dm.tasks
    pl = json.load(open(G.PLAN))
    s1, s2 = G.g10_choose(1, cfg), G.g10_choose(2, cfg)
    out = {}
    for k, v in pl["tasks"].items():
        new, o, g = v["language"], s1[k], s2[k]
        po, pg = BD.phrases(new)
        row = dict(new=new, src_o=[o, tx[o]], src_g=[g, tx[g]])
        for w, t, ph in (("o", o, po), ("g", g, pg)):
            m = BD.match(bk, new, tx[t], ph)
            if m is None:
                row[w] = None
                continue
            pn, ps, used = m
            S = BD.load_or_identify(bk, cfg, dm, t)[:, ps].astype(np.float32)
            N = BD.load_or_identify(bk, cfg, dm, t, new)[:, pn].astype(np.float32)
            row[w] = dict(phrase=used, pos_new=pn, pos_src=ps, shift=float(np.linalg.norm(S[1:] - N[1:])),
                          resid=float(np.linalg.norm(N[1:])), ratio=float(np.linalg.norm(S[1:] - N[1:]) / np.linalg.norm(N[1:])))
        out[k] = row
        print(k, new, "|", {w: (row[w] or {}).get("phrase") for w in "og"},
              {w: round((row[w] or {}).get("ratio", float("nan")), 3) for w in "og"}, flush=True)
    t = s1["goal:4"]
    z = BD.identify(bk, dm, t, tx[t])
    own = BD.load_or_identify(bk, cfg, dm, t)
    print("[check] own vs re-identified under own prompt: max |diff|", float(np.abs(z.astype(np.float32) - own.astype(np.float32)).max()))
    json.dump(out, open(os.path.join(ROOT, "results", "robot", f"z2_prep_{cfg}.json"), "w"), indent=1)
    print("[z2] done", flush=True)


if __name__ == "__main__":
    main()
