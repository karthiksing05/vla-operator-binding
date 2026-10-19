"""U1 (exploratory, offline): the frozen pi0.5's flow loss on every frame of a demonstration under each of its 40 training
instructions (docs/UNSUPERVISED.md).  No access to the demo's own instruction is needed by the segmenter (the own instruction is
scored as a 41st column for reference only) and the gripper is never read here.

Per demo, at every --stride-th frame: 42 prompts (the 40 training instructions, the demo's own instruction, the empty prompt) x
--K shared draws of (flow time ~ Beta(1.5, 1), noise), the same draws for every prompt at that frame (paired comparison).
Saves results/robot/u1/<set>_<task>_d<demo>.npz:
    err [F, 42, K, 7]  squared velocity error per real action dim, averaged over the chunk's H steps
    pad [F, 42, K]     the same averaged over the padded dims (so the I6 full-dim loss = (7 err.mean(-1) + 25 pad) / 32)
    lat [F, 2, 256]    the empty-prompt prefix's last-layer values, mean over the two camera images' tokens / over all tokens
    frames, actions, state, task, text, episode

    sbatch ... SCRIPT="experiments/u1_score.py --set l90 --tasks 0-22"     (--set l10: pi0.5's LIBERO-10 demos)
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
from lifelong_ood.policy import data as D                              # noqa: E402

OUT = os.path.join(ROOT, "results", "robot", "u1")


def tasks_arg(s):
    out = []
    for p in s.replace("+", ",").split(","):
        if "-" in p:
            a, b = p.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="l90", choices=["l90", "l10", "l40"])
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--demo", type=int, default=0)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    import torch
    from lifelong_ood.policy.backbone import PolicyBackbone
    os.makedirs(OUT, exist_ok=True)
    bk = PolicyBackbone("pi05_libero", device="cuda")
    norm = D.Norm.load()
    d40 = D.Demos()
    train_texts = [d40.tasks[i] for i in range(40)]
    if a.set == "l90":
        import importlib.util
        spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
        r3p = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(r3p)
        texts, _ = r3p.libero90_texts()
        src = D.Demos90()
        eps_of = lambda t: src.episodes_of_task(t)                       # noqa: E731
    else:
        texts = {i: d40.tasks[i] for i in range(10 if a.set == "l10" else 40)}
        src = d40
        eps_of = lambda t: src.episodes(t)                               # noqa: E731
    H, A, a7 = bk.H, bk.A, 7
    for t in tasks_arg(a.tasks):
        path = os.path.join(OUT, f"{a.set}_{t}_d{a.demo}.npz")
        if os.path.exists(path) and not a.overwrite:
            print(f"[u1] {path} exists; skipped", flush=True)
            continue
        eps = eps_of(t)
        if len(eps) <= a.demo:
            print(f"[u1] {a.set} task {t}: no demo {a.demo}; skipped", flush=True)
            continue
        t0 = time.time()
        ep = src.load(eps[a.demo])
        T = len(ep["actions"])
        idx, _ = D.chunk_indices(T, H)
        prompts = train_texts + [texts[t], ""]
        P = len(prompts)
        frames = list(range(0, T, a.stride))
        err = np.zeros((len(frames), P, a.K, a7), np.float32)
        pad = np.zeros((len(frames), P, a.K), np.float32)
        lat = np.zeros((len(frames), 2, 256), np.float32)
        for j, f in enumerate(frames):
            o = src.observation(ep, f)
            obs = [dict(o, prompt=p) for p in prompts]
            with torch.no_grad():
                pre = bk.prefix(bk.observations(obs))
                v_last = pre.kv.value_cache[-1][-1]                    # empty prompt: [heads, seq, dim]
                lat[j, 0] = v_last[:, :512].float().mean((0, 1)).cpu().numpy()[:256]
                m = pre.pad_masks[-1].bool()
                lat[j, 1] = v_last[:, m].float().mean((0, 1)).cpu().numpy()[:256]
                x = torch.as_tensor(D.pad_rows(D.to_rows(ep["actions"][idx[f]][None], norm), H), device=bk.device)[0]
                gen = torch.Generator(device="cpu").manual_seed(1000003 * t + 7919 * a.demo + f)
                b = torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
                tt = (b.sample((a.K,), ) * 0.999 + 0.001).to(bk.device)
                e = torch.randn((a.K, H, A), generator=gen).to(bk.device)
                xt = tt[:, None, None] * e + (1 - tt[:, None, None]) * x[None]
                u = e - x[None]
                rep = bk.repeat(pre, a.K)                              # row p*K + k = prompt p, draw k
                v = bk.v_full(xt.repeat(P, 1, 1), tt.repeat(P), rep).reshape(P, a.K, H, A)
                se = (v - u[None]) ** 2
                err[j] = se[..., :a7].mean(2).cpu().numpy()
                pad[j] = se[..., a7:].mean((2, 3)).cpu().numpy()
                del pre, rep, v, se
        np.savez_compressed(path, err=err, pad=pad, lat=lat, frames=np.asarray(frames), actions=ep["actions"], state=ep["state"],
                            task=t, text=texts[t], episode=int(eps[a.demo]), prompts=np.asarray(prompts), stride=a.stride, K=a.K)
        full = (7 * err.mean(-1) + 25 * pad) / 32
        best = full[:, :40].mean(-1).argmin(1)
        print(f"[u1] {a.set} task {t} ep {eps[a.demo]} T={T}: {len(frames)} frames in {time.time() - t0:.0f}s; own loss "
              f"{full[:, 40].mean():.4f}, best-of-40 {full[:, :40].mean(-1).min(1).mean():.4f}, empty {full[:, 41].mean():.4f}; "
              f"best labels {np.bincount(best, minlength=40).argsort()[::-1][:3].tolist()}", flush=True)
    print("[u1] done", flush=True)


if __name__ == "__main__":
    main()
