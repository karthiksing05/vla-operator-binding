"""Plan Z, track I (docs/PLAN_Z.md): the experts' velocity fields on a demonstration, for the soft blend and its donation analysis
(experiments/i1_blend.py).  Offline, GPU.

Per demo (results/robot/u1/<set>_<task>_d0.npz gives its U1 scores): the EXPERTS are the frozen pi0.5 under
    the top --topk training instructions by mean flow loss over each contact segment (lifelong_ood/policy/program.py's cuts; the
    union over segments; the demo's own instruction excluded for training demos), the new instruction itself ("own"; for a
    training demo this is the held-out answer, kept as a reference and never an expert of the blend), and the empty prompt.
At every --stride-th frame, --K shared draws (t ~ Beta(1.5, 1), noise), the same for every expert: the velocity v_k(x_t, t) on the
real action dims and the target u = eps - a.  Saves results/robot/i1/<set>_<task>.npz:
    v [F, E, K, H, 7], u [F, K, H, 7], t [F, K], experts (prompt texts), expert_ids (training index, -1 own, -2 empty),
    frames, segments, actions, state(ee pose), text

    sbatch ... SCRIPT="experiments/i1_velocities.py --set l40 --tasks 10-19"
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
from lifelong_ood.policy import data as D                              # noqa: E402
from lifelong_ood.policy import program as PG                          # noqa: E402

OUT = os.path.join(ROOT, "results", "robot", "i1")


def experts_of(z, topk, own_excluded):
    L = np.log(((7 * z["err"].mean(-1) + 25 * z["pad"]) / 32).mean(2))            # [F, 42]
    labels = [l for l in range(40) if l != own_excluded]
    segs = PG.segments(z["actions"])
    fr = z["frames"]
    ids = []
    for lo, hi in segs:
        m = (fr >= lo) & (fr < hi)
        if not m.any():
            continue
        order = [labels[i] for i in np.argsort(L[m][:, labels].mean(0))]
        ids += [i for i in order[:topk] if i not in ids]
    return ids, segs


def main():
    import u1_score as U
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="l40", choices=["l90", "l40"])
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--topk", type=int, default=2)
    a = ap.parse_args()
    import torch
    from lifelong_ood.policy.backbone import PolicyBackbone
    os.makedirs(OUT, exist_ok=True)
    bk = PolicyBackbone("pi05_libero", device="cuda")
    norm = D.Norm.load()
    d40 = D.Demos()
    tx = d40.tasks
    if a.set == "l90":
        import importlib.util
        spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
        r3p = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(r3p)
        texts, _ = r3p.libero90_texts()
        src = D.Demos90()
        eps_of = src.episodes_of_task
    else:
        texts, src, eps_of = tx, d40, d40.episodes
    H, A = bk.H, bk.A
    for t in U.tasks_arg(a.tasks):
        path = os.path.join(OUT, f"{a.set}_{t}.npz")
        if os.path.exists(path):
            continue
        z = np.load(os.path.join(ROOT, "results", "robot", "u1", f"{a.set}_{t}_d0.npz"))
        ids, segs = experts_of(z, a.topk, t if a.set == "l40" else -1)
        prompts = [tx[i] for i in ids] + [texts[t], ""]
        eids = ids + [-1, -2]
        t0 = time.time()
        ep = src.load(eps_of(t)[0])
        T = len(ep["actions"])
        idx, _ = D.chunk_indices(T, H)
        frames = list(range(0, T, a.stride))
        E = len(prompts)
        V = np.zeros((len(frames), E, a.K, H, 7), np.float32)
        Uu = np.zeros((len(frames), a.K, H, 7), np.float32)
        TT = np.zeros((len(frames), a.K), np.float32)
        for j, f in enumerate(frames):
            o = src.observation(ep, f)
            with torch.no_grad():
                pre = bk.prefix(bk.observations([dict(o, prompt=p) for p in prompts]))
                x = torch.as_tensor(D.pad_rows(D.to_rows(ep["actions"][idx[f]][None], norm), H), device=bk.device)[0]
                gen = torch.Generator(device="cpu").manual_seed(7919 * t + f)
                tt = (torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0)).sample((a.K,)) * 0.999 + 0.001).to(bk.device)
                e = torch.randn((a.K, H, A), generator=gen).to(bk.device)
                xt = tt[:, None, None] * e + (1 - tt[:, None, None]) * x[None]
                v = bk.v_full(xt.repeat(E, 1, 1), tt.repeat(E), bk.repeat(pre, a.K)).reshape(E, a.K, H, A)
                V[j] = v[..., :7].cpu().numpy()
                Uu[j] = (e - x[None])[..., :7].cpu().numpy()
                TT[j] = tt.cpu().numpy()
        np.savez_compressed(path, v=V, u=Uu, t=TT, experts=np.asarray(prompts), expert_ids=np.asarray(eids), frames=np.asarray(frames),
                            segments=np.asarray(segs), actions=ep["actions"], state=ep["state"], text=texts[t], task=t)
        print(f"[i1] {a.set} {t} '{texts[t]}': {E} experts {eids}, {len(frames)} frames in {time.time() - t0:.0f}s", flush=True)
    print("[i1] done", flush=True)


if __name__ == "__main__":
    main()
