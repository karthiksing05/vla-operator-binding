"""Plan F1, step 1: where a prototype fit's 38.9 A40-minutes go (docs/FIT_TIME.md).  GPU; ledger f1_profile.

One prototype of E3's library, cut exactly as its c0_fit run cut it (c0_fit.concept_frames; the frame count is checked against
the fit's ledger row).  Timed, with CUDA synchronised at each boundary:
  * start-up: imports and loading pi0.5;
  * data: loading the demonstrations' episodes (parquet + mp4 decode) and building the observation dicts of the phase frames;
  * the fit as c0_fit.py runs it (LazyPrefixBank): per step, the frozen VLM prefix of the batch (take), the action expert's forward
    and backward pass, the optimiser step; --lazy-steps steps, extrapolated to the fit's 1,000;
  * the same fit with every frame's prefix cached once (PrefixBank): the caching pass, then per step the gather, forward and backward,
    optimiser; --cached-steps steps;
  * saving the prototype (adapt.save; to a scratch file, removed).
The prefix does not depend on the adapter (it lives in the action expert), so the two fits see the same values: the first
--lazy-steps losses of both, at the same seed, are compared.  The c0_fit job does no evaluation (acceptance runs elsewhere).

    sbatch ... SCRIPT="experiments/f1_profile.py --name Gscplate" slurm/jobs/robot_gpu.sbatch
    --tmp: run against the not-yet-installed edits (experiments/_c0_fit_f1_tmp.py, lifelong_ood/policy/_adapt_f1_tmp.py): their
           import and function test
"""
import time

T0 = time.time()
import argparse                                                     # noqa: E402
import os                                                           # noqa: E402
import sys                                                          # noqa: E402

import numpy as np                                                  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f1_common as F                                               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="Gscplate")
    ap.add_argument("--lazy-steps", type=int, default=30)
    ap.add_argument("--cached-steps", type=int, default=200)
    ap.add_argument("--tmp", action="store_true")
    ap.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    c0 = F.c0(tmp=False)
    import torch
    from lifelong_ood import ledger
    from lifelong_ood.experts.lora import add_node
    from lifelong_ood.policy import adapt
    from lifelong_ood.policy import data as D
    assert hasattr(adapt, "fixed_draws"), "the edited adapt module is not the one loaded"
    t_import = time.time()
    bk, norm = c0.backbone("pi05_libero")
    torch.cuda.synchronize()
    t_load = time.time()
    fa = F.fit_args(a.name)
    row = F.fit_row(a.name)
    timer = {}
    obs, full, used = c0.concept_frames(fa, bk, norm, fa.demo, fa.ndemo, timer=timer)
    t_data = time.time()
    assert len(obs) == row["metrics"]["frames"], (len(obs), row["metrics"]["frames"])
    print(f"[f1p] {a.name}: {len(obs)} frames from {len(used)} demos (as its fit); load {timer['load']:.1f}s, observations "
          f"{timer['observation']:.1f}s", flush=True)
    state, banks = adapt.inject(bk, scope="expert")

    def run(bank, key, steps):
        params = add_node(banks, key, fa.rank)
        prof = {}
        t = time.time()
        hist = adapt.train(bk, state, key, params, bank, full, steps=steps, lr=fa.lr, batch=fa.batch, log_every=1,
                           log=lambda x: None, profile=prof)
        torch.cuda.synchronize()
        return hist, prof, time.time() - t

    lazy = adapt.LazyPrefixBank(bk, obs)
    h_l, p_l, s_l = run(lazy, "f1p_lazy", a.lazy_steps)
    print(f"[f1p] lazy: {a.lazy_steps} steps in {s_l:.1f}s: " + ", ".join(f"{k} {v:.2f}s" for k, v in p_l.items() if not k.endswith("_n")),
          flush=True)
    del lazy
    t = time.time()
    cached = adapt.PrefixBank(bk, obs)
    torch.cuda.synchronize()
    t_cache = time.time() - t
    h_c, p_c, s_c = run(cached, "f1p_cached", a.cached_steps)
    print(f"[f1p] cached: prefixes of {cached.n} frames in {t_cache:.1f}s ({cached.bytes() / 1e9:.2f} GB); {a.cached_steps} steps in "
          f"{s_c:.1f}s: " + ", ".join(f"{k} {v:.2f}s" for k, v in p_c.items() if not k.endswith("_n")), flush=True)
    if a.tmp:                                                       # function test of the new helpers
        fa.heldout, fa.eval_frames = 1, 64
        ev = c0.heldout_setup(fa, bk, norm, state)
        on = adapt.heldout_loss(bk, state, {"f1p_cached"}, ev["bank"], ev["full"], ev["draws"])
        again = adapt.heldout_loss(bk, state, {"f1p_cached"}, ev["bank"], ev["full"], ev["draws"])
        ch = c0.init_best(argparse.Namespace(init_best="Osc1+Gsccab+Gscbasket"), bk, state, cached, full)
        print(f"[f1p] tmp test: held-out frozen {ev['frozen']:.4f}, after {a.cached_steps} cached steps {on:.4f} (repeat {again:.4f}); "
              f"init_best {ch['chosen']} {ch['losses']}", flush=True)
        assert on == again
    n = min(a.lazy_steps, a.cached_steps)
    ll, lc = np.array(h_l["loss"][:n]), np.array(h_c["loss"][:n])
    os.makedirs(F.F1, exist_ok=True)
    tmp_path = os.path.join(F.F1, f"_profile_{a.name}.pt")
    t = time.time()
    adapt.save(banks, "f1p_cached", tmp_path)
    t_save = time.time() - t
    os.remove(tmp_path)
    # the draws do not depend on the bank, so step k of both runs sees the same rows, times and noise
    eq = dict(steps=n, max_abs=float(np.abs(ll - lc).max()), max_rel=float((np.abs(ll - lc) / np.abs(ll)).max()),
              lazy=ll.tolist(), cached=lc.tolist())
    per = lambda p, k, steps: p.get(k, 0.0) / steps                                         # noqa: E731
    S = fa.steps
    lazy_step = {k: per(p_l, k, a.lazy_steps) for k in ("take", "fwd_bwd", "opt")}
    cached_step = {k: per(p_c, k, a.cached_steps) for k in ("take", "fwd_bwd", "opt")}
    startup = dict(imports=t_import - T0, model_load=t_load - t_import)
    data = dict(episode_load=timer["load"], observations=timer["observation"], other=t_data - t_load - timer["load"] - timer["observation"])
    lazy_fit = dict(startup | data, **{f"{k}_x{S}": v * S for k, v in lazy_step.items()}, save=t_save)
    cached_fit = dict(startup | data, cache=t_cache, **{f"{k}_x{S}": v * S for k, v in cached_step.items()}, save=t_save)
    m = dict(name=a.name, frames=len(obs), demos=len(used), steps_of_fit=S, batch=fa.batch, rank=fa.rank,
             gpu=torch.cuda.get_device_name(0), startup=startup, data=data, lazy_step=lazy_step, cached_step=cached_step,
             cache_secs=t_cache, cache_gb=cached.bytes() / 1e9, save_secs=t_save, equivalence=eq,
             lazy_fit_secs=lazy_fit, lazy_fit_total_min=sum(lazy_fit.values()) / 60,
             cached_fit_secs=cached_fit, cached_fit_total_min=sum(cached_fit.values()) / 60,
             recorded_fit=dict(row=row["run_id"], first_loss=row["metrics"]["curve"]["loss"][0]),
             peak_gb=torch.cuda.max_memory_allocated() / 1e9)
    for k in ("startup", "data", "lazy_step", "cached_step"):
        print(f"[f1p] {k}: " + ", ".join(f"{x} {v:.3f}" for x, v in m[k].items()), flush=True)
    print(f"[f1p] lazy fit of {S} steps ~ {m['lazy_fit_total_min']:.1f} min; cached ~ {m['cached_fit_total_min']:.1f} min; equivalence "
          f"max |dloss| {eq['max_abs']:.2e} (rel {eq['max_rel']:.2e}) over {n} steps; recorded first loss "
          f"{row['metrics']['curve']['loss'][0]:.4f} vs {ll[0]:.4f}", flush=True)
    if not a.no_record:
        print(ledger.record("f1_profile", bench="libero_90", tag=a.name + ("_tmp" if a.tmp else ""), seed=0, metrics=m,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=vars(a),
                            script="experiments/f1_profile.py", command=" ".join(sys.argv)), flush=True)
    print("[f1p] done", flush=True)


if __name__ == "__main__":
    main()
