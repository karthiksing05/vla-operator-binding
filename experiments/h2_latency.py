"""Plan H, H2 (cost): TEST-TIME LATENCY of every operator, the way OpenVLA-OFT reports it -- batch 1, one A40, 100 queries
after a warm-up, per-query wall time and Hz, plus peak GPU memory.  One GPU job per backbone; the backbone and hook classes are
used directly (no server, no simulator), on observations from a recorded demonstration (decoded and resized to 224 x 224 before
timing, as the LIBERO client sends them).  A query is GuidedPolicy.infer: preprocessing, the prefix pass, 10 flow steps and the
unnormalisation -- one action chunk.

Arms (each skipped where the backbone has no such unit on disk):
  policy      the frozen policy alone
  switch      the grasp switch: PrimHooks with Li's source pair as phase prompts (the prefix is re-encoded every query anyway;
              the switch only changes the instruction) -- timed once with the approach prompt and once forced to the carry prompt
  tli         Li's TLI: TLIHooks' cond_hook shift on every conditioning layer
  prototype   the object and goal prototypes (rank-4 LoRAs on the action expert), PrimHooks as the library serves them
  lora16      the per-task rank-16 LoRA (the leaf / the per-task baseline)
  lora4       the per-task rank-4 LoRA (G9; pi0.5 only)
  dsrl        DSRL's noise actor (authors' configuration) with its ResNet-18 features, on the frozen policy (pi0.5 only)
  policy_wrapped  the policy with every adapter's banks injected and none on (the wrappers' own cost)
Arms run in the order policy, switch, tli, dsrl, prototype, lora16, lora4, policy_wrapped: the LoRA banks are injected into
the model only from `prototype` on, so the earlier arms see the untouched policy.

    sbatch ... --export=ALL,ENV=openpi,SCRIPT="experiments/h2_latency.py --backbone pi05_libero" slurm/jobs/robot_gpu.sbatch
    sbatch ... --export=ALL,ENV=smolvla,SCRIPT="experiments/h2_latency.py --backbone xvla" slurm/jobs/robot_gpu.sbatch
    $PY experiments/h2_latency.py --submit        # the four jobs
-> results/robot/h2_latency_<backbone>.json; ledger h2_latency tag <backbone>.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
R = os.path.join(ROOT, "results", "robot")
N_WARM, N_TIME = 10, 100
CKPT = {"smolvla": "lerobot/smolvla_libero", "xvla": "lerobot/xvla-libero"}
PROTO = {"pi05_libero": ("Osc1", "Gsccab"), "pi0_libero": ("POsc1", "PGsccab"), "smolvla": ("VOsc1", "VGsccab"),
         "xvla": ("YOsc1", "YGsccab")}
LORA = {"pi05_libero": "", "pi0_libero": "_p0", "smolvla": "_sv", "xvla": "_xv"}
LORA_TASK = 10                       # 'put the black bowl on top of the cabinet' (Osc1 + Gsccab serve it)
# Li's OOD task goal:0 'put the cream cheese in the basket': Li's TLI pair (13 -> 17, lambda 24) and G10's pair (11, 10)
OOD_TEXT, TLI_ARM, SWITCH_PAIR = "put the cream cheese in the basket", (13, 17, 24), (11, 10)
EXCLUDE = "hal,tars,t1000,vicki,johnny5,c3po,olivaw"


def make_backbone(name):
    if name == "smolvla":
        from lifelong_ood.policy.backbone_smolvla import SmolVLABackbone
        return SmolVLABackbone(CKPT[name], device="cuda")
    if name == "xvla":
        from lifelong_ood.policy.backbone_xvla import XVLABackbone
        return XVLABackbone(CKPT[name], device="cuda")
    from lifelong_ood.policy.backbone import PolicyBackbone
    return PolicyBackbone(name, device="cuda")


def observations(n, prompt):
    """n observations from the first demonstration of training task 13 (frames spread over the episode), 224 x 224."""
    from PIL import Image
    from lifelong_ood.policy import data as D
    d = D.Demos()
    ep = d.load(d.episodes(13)[0], images=True)
    idx = np.linspace(0, len(ep["frame_index"]) - 1, n).astype(int)
    out = []
    for i in idx:
        o = d.observation(ep, int(i))
        for k in ("observation/image", "observation/wrist_image"):
            o[k] = np.asarray(Image.fromarray(o[k]).resize((224, 224), Image.BILINEAR), np.uint8)
        o["prompt"] = prompt
        out.append(o)
    return out, d


def time_arm(policy, obs, before=None):
    import torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for j in range(N_WARM):
        o = dict(obs[j % len(obs)])
        if j == 0:
            o["meta/episode_start"] = True
        policy.infer(o)
        if before is not None:
            before()
    ms = []
    for j in range(N_TIME):
        o = obs[j % len(obs)]
        if before is not None:
            before()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = policy.infer(o)
        torch.cuda.synchronize()
        ms.append(1000 * (time.perf_counter() - t0))
        assert np.isfinite(out["actions"]).all()
    ms = np.asarray(ms)
    return dict(ms_mean=float(ms.mean()), ms_median=float(np.median(ms)), ms_p90=float(np.percentile(ms, 90)),
                ms_sd=float(ms.std(ddof=1)), hz=float(1000.0 / ms.mean()), queries=int(N_TIME), warmup=int(N_WARM),
                chunk=int(policy.bk.H), peak_alloc_gb=float(torch.cuda.max_memory_allocated() / 1e9),
                peak_reserved_gb=float(torch.cuda.max_memory_reserved() / 1e9))


def run(name, record):
    import torch
    from lifelong_ood.policy import adapt
    from lifelong_ood.policy import primitives as PR
    from lifelong_ood.policy import tli as TL
    from lifelong_ood.policy.serve import GuidedPolicy
    t_start = time.time()
    bk = make_backbone(name)
    torch.cuda.synchronize()
    base_gb = torch.cuda.memory_allocated() / 1e9
    obs, demos = observations(25, OOD_TEXT)
    res = dict(backbone=name, gpu=torch.cuda.get_device_name(0), num_steps=10, chunk=int(bk.H), base_alloc_gb=float(base_gb),
               n_params=int(bk.n_params), arms={})

    def go(arm, hooks, before=None, note=""):
        r = time_arm(GuidedPolicy(bk, hooks=hooks, num_steps=10, seed=0, log_every=0), obs, before)
        r["note"] = note
        res["arms"][arm] = r
        print(f"[h2 {name}] {arm:16} {r['ms_mean']:7.1f} ms (median {r['ms_median']:.1f}, p90 {r['ms_p90']:.1f}) = "
              f"{r['hz']:5.2f} Hz; peak {r['peak_alloc_gb']:.2f} GB  {note}", flush=True)

    go("policy", None)
    tx = demos.tasks
    a_, g_ = SWITCH_PAIR
    sw = PR.PrimHooks(bk, [OOD_TEXT], [None], goal_only=True, approach_paths=[None],
                      phase_prompts=[(tx[a_], tx[g_])])
    go("switch", sw, note=f"approach prompt: {tx[a_]!r}")
    sw2 = PR.PrimHooks(bk, [OOD_TEXT], [None], goal_only=True, approach_paths=[None], phase_prompts=[(tx[a_], tx[g_])])
    sw2.observe = lambda actions: None                                  # held in the carry phase: the switched instruction

    def carry():
        sw2.carrying = True
    go("switch_carry", sw2, before=carry, note=f"carry prompt: {tx[g_]!r}")
    t1, t2, lam = TLI_ARM
    lat = [TL.path(name, t, ROOT) for t in (t1, t2)]
    if all(os.path.exists(p) for p in lat):
        th = TL.TLIHooks(bk, np.load(lat[0]), np.load(lat[1]), lam)
        go("tli", th, note=f"tli {t1}->{t2} lambda {lam}")
        for h in th._h:
            h.remove()
    if name == "pi05_libero":
        ap = os.path.join(R, f"dsrlz_{LORA_TASK}_actor.pt")
        if os.path.exists(ap):
            from lifelong_ood.policy.baselines import VinnPolicy
            from lifelong_ood.policy.dsrl import DSRLHooks
            enc = VinnPolicy(np.zeros((1, 7), np.float32), np.zeros((1, 224, 224, 3), np.uint8), np.zeros((1, 224, 224, 3), np.uint8),
                             device="cuda")

            def feat(o):
                f_ = enc._embed(np.asarray(o["observation/image"])[None], np.asarray(o["observation/wrist_image"])[None])
                return np.concatenate([f_.float().cpu().numpy()[0], np.asarray(o["observation/state"], np.float32)])
            dh = DSRLHooks(feat, dim=bk.H * bk.A, train=False, load=ap, official=True, H=bk.H, A=bk.A, log=print)
            go("dsrl", dh, note=os.path.basename(ap) + " (frozen policy, authors' configuration)")
    o_, g_p = PROTO[name]
    pp = [os.path.join(R, f"prim_{n}_0.pt") for n in (o_, g_p)]
    if all(os.path.exists(p) for p in pp):
        ph = PR.PrimHooks(bk, [OOD_TEXT], [pp[1]], goal_only=True, approach_paths=[pp[0]])
        go("prototype", ph, note=f"object {o_} before the grasp (goal {g_p} after)")
        def carry2():
            if not ph.carrying:
                ph.carrying = True
                ph._apply()
        ph.observe = lambda actions: None                                # held in the carry phase: the goal prototype on
        go("prototype_goal", ph, before=carry2, note=f"goal {g_p} on")
    for arm, tag in (("lora16", LORA[name]), ("lora4", "_r4" if name == "pi05_libero" else None)):
        if tag is None:
            continue
        lp = os.path.join(R, f"r3_lora_{LORA_TASK}{tag}.pt")
        if not os.path.exists(lp):
            continue
        state, key, banks = adapt.load(bk, lp, state=getattr(bk.expert, "_keyed_state", None))

        class On:                                                        # switch only this adapter on (others stay injected, off)
            def reset(self, meta, _s=state, _k=key):
                _s["on"] = frozenset({_k})

            def __call__(self, obs_, pre, meta=None):
                return None, None
        hk = On()
        state["on"] = frozenset({key})
        go(arm, hk, note=os.path.basename(lp))
        state["on"] = frozenset()
    if getattr(bk.expert, "_keyed_state", None) is not None:                # the wrappers alone: every adapter injected, none on
        bk.expert._keyed_state["on"] = frozenset()
        go("policy_wrapped", None, note="LoRA banks injected, all off")
    res["secs"] = time.time() - t_start
    json.dump(res, open(os.path.join(R, f"h2_latency_{name}.json"), "w"), indent=1)
    if record:
        from lifelong_ood import ledger
        print(ledger.record("h2_latency", bench="libero", tag=name, seed=0, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(backbone=name, warmup=N_WARM, queries=N_TIME, batch=1, num_steps=10, prompt=OOD_TEXT,
                                        tli=TLI_ARM, switch=SWITCH_PAIR, prototypes=PROTO[name], lora_task=LORA_TASK),
                            script="experiments/h2_latency.py", command=" ".join(sys.argv), seconds=res["secs"]), flush=True)


def submit():
    for name in ("pi05_libero", "pi0_libero", "smolvla", "xvla"):
        env = "smolvla" if name in ("smolvla", "xvla") else "openpi"
        out = subprocess.run(["sbatch", "--parsable", "--partition=overcap", "--account=overcap", "--qos=short", f"--exclude={EXCLUDE}",
                              "--time=01:00:00", f"--export=ALL,ENV={env},SCRIPT=experiments/h2_latency.py --backbone {name} --record",
                              "slurm/jobs/robot_gpu.sbatch"], capture_output=True, text=True, cwd=ROOT)
        print(name, out.stdout.strip() or out.stderr.strip(), flush=True)


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--backbone", default="pi05_libero")
    ap_.add_argument("--record", action="store_true")
    ap_.add_argument("--submit", action="store_true")
    a = ap_.parse_args()
    if a.submit:
        submit()
    else:
        run(a.backbone, a.record)
