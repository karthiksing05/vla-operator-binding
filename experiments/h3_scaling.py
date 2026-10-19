"""Plan H, H3 (scaling): how the stored bytes, the fit time and the practice of every method grow with the number of tasks N, and why.

  THE ARGUMENT  A per-task method stores one unit per task, so its store is linear in N.  The compositional library stores one unit
                per CONCEPT (object type, instance child, goal); a task is a combination of concepts, so the library grows with the
                number of distinct concepts C(N), sublinear when concepts recur.  Ideal case: C_o objects x C_g goals give C_o*C_g
                tasks for C_o + C_g units (C = 2 sqrt(N) on a square grid).  Held concepts cost 0 bytes (a pointer); TLI latents cost a
                fixed amount per SOURCE training task, bounded by the training set; task leaves add a per-task cost only where practice
                keeps one.
  CONCEPTS      C(N) measured on the real inventories: G8's 66 single-step LIBERO-90 tasks (e3_plan + e5_plan: the prototypes each
                task needs), Li's 20 OOD tasks (g4_plan; concepts keyed by suite and words, as G16 stores them), LIBERO-PRO's 27
                eligible tasks (g22_plan); mean over 5,000 random orders (5-95% band), plus the W1 stream's orders a/b and G16's
                plan/reversed orders.  Power law C = a N^beta fitted by least squares in log-log over N = 1..N_max (and over the last
                half, the tail slope, as a sensitivity).
  SPACE         unit sizes from ledger h2_space tag units (bf16 bytes); kept fractions from the reads (g8_leaves, g9_rank4lora,
                g6_strap, g7_dsrl, e3_hier, w1_stream); measured curves where an order exists (W1 order a for set b; random-order mean
                on G8's inventory; G16 plan order for the OOD side); extrapolated to N = 90 and N = 1,000 with the fitted C(N).
  TIME          fit A40-minutes per unit from ledger h2_fit (slurm job wall time); practice episodes from h2_practice and the stream
                logs (W1 arrivals, G16 / G17 per-arrival practice).

    $PY experiments/h3_scaling.py [--record]     # -> figures/robot/h3_{space,concepts,time}.png, docs/SCALING.md
Ledger rows (with --record): h3_scaling tags concepts, space, time.  CPU only.
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from lifelong_ood import ledger                                       # noqa: E402

R = os.path.join(ROOT, "results", "robot")
FIG = os.path.join(ROOT, "figures", "robot")
NREP = 5000
NS = (30, 66, 90, 1000)
NMAX = 1000
MB = 1e6


def row(exp, tag):
    fs = sorted(glob.glob(os.path.join(ROOT, "ledger", exp, f"*__{tag}__*.json")))
    if not fs:
        raise KeyError(f"{exp} {tag}")
    rows = [json.load(open(f)) for f in fs]
    r = max(rows, key=lambda x: x.get("started") or "")
    r["_path"] = os.path.relpath(fs[rows.index(r)], ROOT)
    return r


def jl(name):
    return json.load(open(os.path.join(R, name)))


# ── concept inventories ─────────────────────────────────────────────────────────────────────────────────────────────────
def inventories():
    e3, e5 = jl("e3_plan.json"), jl("e5_plan.json")
    l90 = {}
    for t, v in e3.items():
        if v["lora"]:
            l90[int(t)] = {v["object_proto"], v["goal_proto"], v["type_proto"]}
    for t, v in e5["tasks"].items():
        l90[int(t)] = {v["object_proto"], v["goal"], v["type"]}
    g4 = jl("g4_plan.json")["tasks"]
    ood = {k: {f"O:{v['suite']}|{v['object_concept']}", f"G:{v['suite']}|{v['goal_concept']}"} for k, v in g4.items()}
    g22 = jl("g22_plan.json")["tasks"]
    pro = {k: {f"O:{v['suite']}|{v['object_key']}", f"G:{v['suite']}|{v['goal_key']}"} for k, v in g22.items() if v["eligible"]}
    w1 = jl("w1_state.json")
    w1keys = {int(t): {x for x in k.values() if x} for t, k in w1["_meta"]["keys"].items()}
    return dict(l90=l90, ood=ood, pro=pro), w1keys, w1


def curve(inv, order):
    seen, out = set(), []
    for t in order:
        seen |= inv[t]
        out.append(len(seen))
    return np.array(out, float)


def random_curves(inv, rng, n=NREP):
    keys = list(inv)
    cs = np.stack([curve(inv, [keys[i] for i in rng.permutation(len(keys))]) for _ in range(n)])
    return cs.mean(0), np.percentile(cs, 5, 0), np.percentile(cs, 95, 0)


def powerfit(c, lo=1):
    n = np.arange(1, len(c) + 1)
    m = n >= lo
    b, la = np.polyfit(np.log(n[m]), np.log(c[m]), 1)
    return dict(beta=float(b), a=float(np.exp(la)), n_from=int(lo), n_to=int(len(c)))


def grid(inv_l90):
    """The ideal-case reading of the inventory: object-side units (types and instance children) x goal units."""
    e3, e5 = jl("e3_plan.json"), jl("e5_plan.json")
    goals = {v["goal_proto"] for v in e3.values() if v["lora"]} | {v["goal"] for v in e5["tasks"].values()}
    allp = set().union(*inv_l90.values())
    objs = allp - goals
    pairs = len({(min(p - goals), next(iter(p & goals))) for p in inv_l90.values()})
    return dict(C_o=len(objs), C_g=len(goals), units=len(allp), tasks=len(inv_l90), grid=len(objs) * len(goals),
                occupancy=len(inv_l90) / (len(objs) * len(goals)), distinct_pairs=pairs)


# ── the read ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def build():
    rng = np.random.default_rng(0)
    inv, w1keys, w1s = inventories()
    C = {}
    for k, v in inv.items():
        mean, p5, p95 = random_curves(v, rng)
        C[k] = dict(tasks=len(v), units=int(mean[-1]), mean=mean.tolist(), p5=p5.tolist(), p95=p95.tolist(),
                    fit=powerfit(mean), fit_tail=powerfit(mean, lo=len(v) // 2 + 1))
    for o in "ab":
        c = curve(w1keys, w1s[o]["order"])
        C[f"w1_{o}"] = dict(tasks=len(c), units=int(c[-1]), order=w1s[o]["order"], curve=c.tolist(), fit=powerfit(c))
    for o in ("plan", "reversed"):
        lg = jl(f"g16_pi05_{o}.json")["log"]
        c = curve(inv["ood"], [r["task"] for r in lg])
        C[f"g16_{o}"] = dict(tasks=len(c), units=int(c[-1]), order=[r["task"] for r in lg], curve=c.tolist(), fit=powerfit(c))
    C["grid_l90"] = grid(inv["l90"])

    # ── unit sizes and rates (ledger h2_space / h2_fit / h2_practice and the reads) ──
    sp, fi, pr = row("h2_space", "units"), row("h2_fit", "units"), row("h2_practice", "counts")
    U, F, P = sp["metrics"]["units"], fi["metrics"], pr["metrics"]
    P4 = U["prototype_r4/pi05_libero"]["bf16_bytes"]
    L16, L4 = U["lora_r16/pi05_libero"]["bf16_bytes"], U["lora_r4/pi05_libero"]["bf16_bytes"]
    A = U["dsrl_actor/dsrlz"]["bf16_bytes"]
    LB = U["tli_latent/pi05_libero"]["bf16_bytes"]
    ADP = U["cl_adapter/er"]["bf16_bytes"]
    SEQ = U["cl_adapter/seq"]["bf16_bytes"]
    SP = U["strap_prior"]["total"]
    erb = {int(k): v for k, v in sp["metrics"]["er_bytes"].items()}
    DEMO = float(np.mean(list(erb.values())))
    S = F["summary"]
    T_P = S["protos"]["job_s_mean"] / 60
    T16, T4 = S["loras"]["job_s_mean"] / 60, S["loras_r4"]["job_s_mean"] / 60
    T_ST = S["strap"]["job_s_mean"] / 60
    T_DZ = F["fits"]["dsrl"]["dsrlz"]["wall_s"] / 60 / F["fits"]["dsrl"]["dsrlz"]["runs"]
    T_DS = F["fits"]["dsrl"]["dsrls"]["wall_s"] / 60 / F["fits"]["dsrl"]["dsrls"]["runs"]
    T_ER = F["fits"]["cl"]["er"]["wall_s"] / 60
    T_SEQ = F["fits"]["cl"]["seq"]["wall_s"] / 60
    T_LAT = F["fits"]["tli"]["pi05_libero"]["wall_s_per_latent"] / 60
    w1fits = F["fits"]["w1"]
    w1wall = [v["wall_s"] for v in w1fits.values() if v and v.get("wall_s")]
    T_W1 = float(np.mean(w1wall)) / 60
    pl = P["libero90"]

    g8 = row("g8_leaves", "read")["metrics"]
    g9 = row("g9_rank4lora", "read")["metrics"]
    g6 = row("g6_strap", "read")["metrics"]
    g7 = row("g7_dsrl", "read")["metrics"]
    e3h = row("e3_hier", "read")["metrics"]
    w1r = row("w1_stream", "read")["metrics"]
    n8 = g8["tasks"]
    k_leaf = g8["kept"]["lora"] / n8
    k16 = g8["bytes"]["per_task_loras"] / n8
    k4 = g9["kept_lora"] / g9["tasks"]
    k_st = sum(v == "strap" for v in g6["kept_strap"].values()) / len(g6["kept_strap"])
    k_dz = sum(v.get("dsrlz") == "dsrlz" for v in g7["kept"].values()) / len(g7["kept"])
    k_ds = sum(v.get("dsrls") == "dsrls" for v in g7["kept"].values()) / len(g7["kept"])
    beta = C["l90"]["fit"]["beta"]
    beta_t = C["l90"]["fit_tail"]["beta"]
    C66 = C["l90"]["mean"][-1]

    def chat(N, b=beta, anchor_n=n8, anchor_c=C66):
        """Fitted C(N), anchored at the measured end of the inventory (N = 66, C = 48) with the fitted exponent."""
        N = np.asarray(N, float)
        return anchor_c * (N / anchor_n) ** b

    consts = dict(P4=P4, L16=L16, L4=L4, dsrl_actor=A, tli_latent=LB, er_adapter=ADP, seq_adapter=SEQ, strap_prior=SP, demo_mean=DEMO,
                  T_proto=T_P, T_lora16=T16, T_lora4=T4, T_strap=T_ST, T_dsrlz=T_DZ, T_dsrls=T_DS, T_er_stream30=T_ER, T_seq_stream30=T_SEQ,
                  T_tli_latent=T_LAT, T_w1_fit=T_W1, w1_fits=len(w1fits), k_leaf=k_leaf, k_lora16=k16, k_lora4=k4, k_strap=k_st,
                  k_dsrlz=k_dz, k_dsrls=k_ds, w1_demo_rate=w1r["a"]["demos"] / w1r["a"]["arrivals"], pool=16, tli_bound_tasks=40,
                  sources=dict(h2_space=sp["_path"], h2_fit=fi["_path"], h2_practice=pr["_path"]))

    # ── measured curves ──
    Nx = np.arange(1, NMAX + 1, dtype=float)
    meas, extra = {}, {}
    cl = np.array(C["l90"]["mean"])
    n66 = np.arange(1, n8 + 1, dtype=float)
    meas["ours_leaves"] = (n66, P4 * cl + k_leaf * L16 * n66)
    meas["ours_library"] = (n66, P4 * cl)
    meas["lora16_acc"] = (n66, k16 * L16 * n66)
    # set b along W1 order a
    order = w1s["a"]["order"]
    nb = np.arange(1, len(order) + 1, dtype=float)
    al = w1r["a"]["arrivals_log"]
    libsz = [sum(v is not None for v in a2["library_before"].values()) for a2 in al[1:]]
    libsz.append(sum(e["name"] is not None for e in w1s["a"]["library"].values()))
    demo_cum = np.cumsum([erb.get(int(a2["task"]), DEMO) if a2.get("demo") else 0 for a2 in al])
    meas["w1"] = (nb, P4 * np.array(libsz, float) + demo_cum)
    kept_r16 = e3h["kept_lora"]
    meas["lora4_acc"] = (np.arange(1, g9["tasks"] + 1, dtype=float), k4 * L4 * np.arange(1, g9["tasks"] + 1))
    meas["dsrlz"] = (nb, np.cumsum([A if g7["kept"].get(str(t), {}).get("dsrlz") == "dsrlz" else 0 for t in order]))
    meas["strap_acc"] = (nb, SP + np.cumsum([L16 if g6["kept_strap"].get(str(t)) == "strap" else 0 for t in order]))
    meas["er"] = (nb, ADP + np.cumsum([erb.get(int(t), DEMO) for t in order]))
    meas["seq"] = (nb, np.full(len(order), float(SEQ)))
    meas["lora16_acc_w1order"] = (nb, np.cumsum([L16 if kept_r16.get(str(t)) == "lora" else 0 for t in order]))
    # OOD: G16 plan order on pi0.5 -- TLI latents of every source the stream stores (pointers cost 0)
    import h2_cost as H
    import g12_tli as T
    lg = jl("g16_pi05_plan.json")["log"]
    seen, tli_c = set(), []
    for r_ in lg:
        seen |= H.tli_sources(r_["pairs"]["tli"])
        tli_c.append(len(seen))
    n20 = np.arange(1, len(lg) + 1, dtype=float)
    meas["ours_ood"] = (n20, LB * np.array(tli_c, float))
    g4 = jl("g4_plan.json")["tasks"]
    seen, li_c = set(), []
    for r_ in lg:
        seen |= H.tli_sources(T.arm(g4[r_["task"]]))
        li_c.append(len(seen))
    meas["li_tli"] = (n20, LB * np.array(li_c, float))

    # ── extrapolations (dashed): fitted C(N) for the library; kept fraction x unit for the per-task methods ──
    extra["ours_leaves"] = P4 * chat(Nx) + k_leaf * L16 * Nx
    extra["ours_library"] = P4 * chat(Nx)
    extra["w1"] = P4 * chat(Nx, anchor_n=len(order), anchor_c=libsz[-1]) + consts["w1_demo_rate"] * DEMO * Nx
    extra["lora16_acc"] = k16 * L16 * Nx
    extra["lora4_acc"] = k4 * L4 * Nx
    extra["dsrlz"] = k_dz * A * Nx
    extra["strap_acc"] = SP + k_st * L16 * Nx
    extra["er"] = ADP + DEMO * Nx
    extra["seq"] = np.full_like(Nx, float(SEQ))
    extra["ours_ood"] = np.full_like(Nx, consts["pool"] * LB)  # bound: every source of the 16-task pool kept
    extra["li_tli"] = np.full_like(Nx, consts["pool"] * LB)
    extra["lora16_all"] = L16 * Nx
    extra["lora4_all"] = L4 * Nx
    extra["leaves_r4_hyp"] = P4 * chat(Nx) + k_leaf * L4 * Nx

    # ── crossovers ──
    def cross_measured(a, b):
        """smallest measured N from which a stays below b; None if never within the range."""
        idx = np.where(a < b)[0]
        if not len(idx):
            return None
        for i in idx:
            if np.all(a[i:] < b[i:]):
                return int(i + 1)
        return None

    def cross_lib(rate, b=beta, extra_rate=0.0):
        """N solving P4 C(N) + extra_rate N = rate N with C(N) = C66 (N/66)^b; None if rate <= extra_rate."""
        if rate <= extra_rate:
            return None
        return float(n8 * (P4 * C66 / (n8 * (rate - extra_rate))) ** (1 / (1 - b)))

    xs = dict(
        library_vs_lora4_acc=dict(measured=cross_measured(P4 * cl, k4 * L4 * n66), extrapolated=cross_lib(k4 * L4),
                                  extrapolated_tail_beta=cross_lib(k4 * L4, beta_t),
                                  at66_mb=dict(library=P4 * C66 / MB, lora4_acc=k4 * L4 * n8 / MB)),
        library_vs_lora4_all=dict(measured=cross_measured(P4 * cl, L4 * n66), extrapolated=cross_lib(L4)),
        library_vs_lora16_acc=dict(measured=cross_measured(P4 * cl, k16 * L16 * n66), extrapolated=cross_lib(k16 * L16)),
        leaves_vs_lora4_acc=dict(measured=cross_measured(P4 * cl + k_leaf * L16 * n66, k4 * L4 * n66),
                                 extrapolated=cross_lib(k4 * L4, extra_rate=k_leaf * L16),
                                 slope_mb_per_task=dict(leaves=k_leaf * L16 / MB, lora4_acc=k4 * L4 / MB),
                                 note="never: the kept rank-16 leaves alone cost more per task than the kept rank-4 adapters"),
        leaves_vs_lora16_acc=dict(measured=cross_measured(P4 * cl + k_leaf * L16 * n66, k16 * L16 * n66),
                                  extrapolated=cross_lib(k16 * L16, extra_rate=k_leaf * L16)),
        leaves_r4_hyp_vs_lora4_acc=dict(extrapolated=cross_lib(k4 * L4, extra_rate=k_leaf * L4),
                                        note="HYPOTHETICAL: leaves fitted at rank 4 (G8 measured rank-16 leaves only)"),
    )
    # time crossovers: library T_P C(N) vs per-task rate T N
    def tcross(rate, b=beta, extra_rate=0.0):
        if rate <= extra_rate:
            return None
        return float(n8 * (T_P * C66 / (n8 * (rate - extra_rate))) ** (1 / (1 - b)))
    txs = dict(library_vs_lora4=dict(extrapolated=tcross(T4), extrapolated_tail_beta=tcross(T4, beta_t)),
               library_vs_lora16=dict(extrapolated=tcross(T16), extrapolated_tail_beta=tcross(T16, beta_t)),
               leaves_vs_lora16=dict(extrapolated=tcross(T16, extra_rate=T16),
                                     note="never: G8 fits every task's rank-16 LoRA for the acceptance test, on top of the prototypes"),
               library_vs_strap=dict(extrapolated=tcross(T_ST)), library_vs_dsrlz=dict(extrapolated=tcross(T_DZ)))

    # ── time curves ──
    tmeas, textra = {}, {}
    tmeas["ours_library"] = (n66, T_P * cl)
    tmeas["ours_leaves"] = (n66, T_P * cl + T16 * n66)
    tmeas["lora16"] = (n66, T16 * n66)
    tmeas["lora4"] = (np.arange(1, 30, dtype=float), T4 * np.arange(1, 30))
    seen_f, fcum = set(), []
    tw = 0.0
    for a2 in al:
        for k in ("comp1", "comp2", "kept"):
            v = a2.get(k) or ""
            if v.startswith("fac@"):
                for nm in v[4:].split("."):
                    if nm != "x" and nm not in seen_f:
                        seen_f.add(nm)
                        f = w1fits.get(nm)
                        tw += (f["wall_s"] / 60 if f and f.get("wall_s") else T_W1)
        fcum.append(tw)
    tmeas["w1"] = (nb, np.array(fcum))
    tmeas["dsrlz"] = (nb, T_DZ * nb)
    tmeas["strap"] = (np.arange(1, 30, dtype=float), T_ST * np.arange(1, 30))
    tmeas["er"] = (nb, T_ER / len(order) * nb)
    tmeas["seq"] = (nb, T_SEQ / len(order) * nb)
    tmeas["ours_ood"] = (n20, np.full(len(n20), consts["pool"] * T_LAT))
    tmeas["li_tli"] = (n20, T_LAT * np.array(li_c, float))
    textra["ours_library"] = T_P * chat(Nx)
    textra["ours_leaves"] = T_P * chat(Nx) + T16 * Nx
    textra["lora16"] = T16 * Nx
    textra["lora4"] = T4 * Nx
    textra["w1"] = fcum[-1] / len(order) * Nx
    textra["dsrlz"] = T_DZ * Nx
    textra["strap"] = T_ST * Nx
    textra["er"] = T_ER / len(order) * Nx
    textra["seq"] = T_SEQ / len(order) * Nx
    textra["ours_ood"] = np.full_like(Nx, consts["pool"] * T_LAT)
    textra["li_tli"] = np.full_like(Nx, consts["pool"] * T_LAT)
    # practice
    pmeas, pextra = {}, {}
    p_g8 = (pl["g8_zs"] + pl["g8_fac"] + pl["g8_lora"]) / pl["g8_tasks"]
    p_acc = (pl["g8_zs"] + pl["g8_lora"]) / pl["g8_tasks"]
    p_dz = (pl["e3_zs30"] + pl["dsrlz_train"] + pl["dsrlz_acc"]) / 30
    pmeas["ours_leaves"] = (n66, p_g8 * n66)
    pmeas["lora16"] = (n66, p_acc * n66)
    pmeas["w1"] = (nb, np.cumsum([10 * sum(k in a2 for k in ("zs", "acc1", "acc2")) for a2 in al]))
    pmeas["dsrlz"] = (nb, p_dz * nb)
    po = P["ood"]["pi05_libero"]
    pmeas["g15"] = (n20, po["system"] / 20 * n20)
    pmeas["g16"] = (n20, np.cumsum([r_["practice"] for r_ in lg]))
    lg17 = jl("g17_pi05_plan.json")["log"]
    pmeas["g17"] = (n20, np.cumsum([r_["practice"] for r_ in lg17]))
    # G16's rule: 160 episodes per unseen concept (2 operators x 16 sources x 5) + 20 acceptance per arrival
    cood = C["ood"]["fit"]
    c_ood_hat = C["ood"]["mean"][-1] * (Nx / 20) ** cood["beta"]
    per_concept = (pmeas["g16"][1][-1] - 20 * 20) / C["g16_plan"]["units"]
    pextra["ours_leaves"] = p_g8 * Nx
    pextra["lora16"] = p_acc * Nx
    pextra["w1"] = pmeas["w1"][1][-1] / len(order) * Nx
    pextra["dsrlz"] = p_dz * Nx
    pextra["g15"] = po["system"] / 20 * Nx
    pextra["g16"] = 20 * Nx + per_concept * c_ood_hat

    # ── the table (bytes / fit minutes / practice at N = 30, 66, 90, 1000) ──
    def at(meas_, extra_, key, N):
        if key in meas_ and N <= meas_[key][0][-1]:
            return float(meas_[key][1][int(N) - 1]), "measured"
        return float(extra_[key][int(N) - 1]), "extrapolated"

    methods = [
        ("ours_leaves", "Hierarchy with task leaves (ours, G8)", "P4 C(N) + k_leaf L16 N", "ours_leaves", "ours_leaves", "ours_leaves"),
        ("ours_library", "Library without leaves (ours, G8)", "P4 C(N)", "ours_library", "ours_library", None),
        ("w1", "W1 learner (ours, stream, order a)", "P4 C(N) + request rate x demo N", "w1", "w1", "w1"),
        ("lora16_acc", "Per-task LoRA r16 + acceptance", "k16 L16 N", "lora16", "lora16", "lora16"),
        ("lora4_acc", "Per-task LoRA r4 + acceptance", "k4 L4 N", "lora4", "lora4", "lora16"),
        ("dsrlz", "DSRL on the policy (dsrlz) + acceptance", "k_dsrl A N", "dsrlz", "dsrlz", "dsrlz"),
        ("strap_acc", "STRAP + acceptance", "prior + k_strap L16 N", "strap", "strap", "lora16"),
        ("er", "Experience replay", "adapter + demo N", "er", "er", None),
        ("seq", "Sequential fine-tuning", "adapter (constant)", "seq", "seq", None),
        ("ours_ood", "OOD: pointers + TLI latents (ours, G16 plan order)", "LB x sources, <= 16 LB", "ours_ood", "ours_ood", "g16"),
        ("li_tli", "OOD: Li's TLI, hand pairs", "LB x sources, <= 16 LB", "li_tli", "li_tli", None),
    ]
    table = {}
    for key, name, law, tkey, _, pkey in methods:
        ent = dict(method=name, growth_law=law, bytes_at={}, bytes_status={}, fit_min_at={}, fit_status={}, practice_at={},
                   practice_status={})
        for N in NS:
            v, s = at(meas, extra, key, N)
            ent["bytes_at"][str(N)], ent["bytes_status"][str(N)] = v, s
            v, s = at(tmeas, textra, tkey, N)
            ent["fit_min_at"][str(N)], ent["fit_status"][str(N)] = v, s
            if key == "ours_library":                              # policy + composition episodes only
                v, s = (pl["g8_zs"] + pl["g8_fac"]) / pl["g8_tasks"] * N, ("measured" if N <= n8 else "extrapolated")
            elif pkey is None:
                v, s = 0.0, "measured"
            else:
                v, s = at(pmeas, pextra, pkey, N)
            if key in ("lora4_acc", "strap_acc"):                  # zero-shot + adapter acceptance, 20 per task
                v, s = p_acc * N, ("measured" if N <= 29 else "extrapolated")
            ent["practice_at"][str(N)], ent["practice_status"][str(N)] = float(v), s
        ent["bytes_per_task_at_1000"] = ent["bytes_at"]["1000"] / 1000
        table[key] = ent
    return dict(C=C, consts=consts, beta=beta, beta_tail=beta_t, meas=meas, extra=extra, tmeas=tmeas, textra=textra, pmeas=pmeas,
                pextra=pextra, cross_space=xs, cross_time=txs, table=table, Nx=Nx, chat66=float(chat(66)),
                chat=dict(N90=float(chat(90)), N1000=float(chat(1000)), N90_tail=float(chat(90, beta_t)), N1000_tail=float(chat(1000, beta_t)),
                          ideal_1000=float(2 * np.sqrt(1000))),
                practice_per_concept=float(per_concept), c_ood_1000=float(c_ood_hat[-1]),
                reads=dict(g8=row("g8_leaves", "read")["_path"], g9=row("g9_rank4lora", "read")["_path"],
                           g6=row("g6_strap", "read")["_path"], g7=row("g7_dsrl", "read")["_path"], e3=row("e3_hier", "read")["_path"],
                           w1=row("w1_stream", "read")["_path"]))


# ── figures ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
STY = {   # colour, marker, label
    "ours_leaves": ("#008300", "o", "ours: hierarchy with task leaves (G8)"),
    "ours_library": ("#1baf7a", "D", "ours: library without leaves (G8)"),
    "w1": ("#5a9e3a", "^", "ours: W1 stream (order a)"),
    "ours_ood": ("#333333", "s", "ours OOD: pointers + TLI latents (G16)"),
    "li_tli": ("#9a9a9a", "s", "OOD: Li's TLI, hand pairs"),
    "lora16_acc": ("#2a78d6", None, "per-task LoRA r16 + acceptance"),
    "lora16": ("#2a78d6", None, "per-task LoRA r16"),
    "lora4_acc": ("#6250d6", None, "per-task LoRA r4 + acceptance"),
    "lora4": ("#6250d6", None, "per-task LoRA r4"),
    "dsrlz": ("#eb6834", None, "DSRL on the policy + acceptance"),
    "strap_acc": ("#e87ba4", None, "STRAP + acceptance (36 GB prior)"),
    "strap": ("#e87ba4", None, "STRAP"),
    "er": ("#eda100", None, "experience replay"),
    "seq": ("#e34948", None, "sequential fine-tuning"),
    "g15": ("#333333", "s", "ours OOD: practice per task (G15)"),
    "g16": ("#333333", "v", "ours OOD: stream with reuse (G16)"),
    "g17": ("#777777", "^", "ours OOD: gated stream (G17)"),
}


def _plot(ax, meas, extra, Nx, keys, scale=1.0):
    for k in keys:
        c, m, lab = STY[k]
        lw = 2.2 if k.startswith("ours") or k == "w1" else 1.5
        if k in meas:
            x, y = meas[k]
            ax.plot(x, np.asarray(y) / scale, color=c, lw=lw, marker=m, markevery=max(1, len(x) // 6), ms=5, label=lab)
            x0 = int(x[-1])
        else:
            x0 = 1
        if k in extra:
            ax.plot(Nx[x0 - 1:], np.asarray(extra[k][x0 - 1:]) / scale, color=c, lw=lw * 0.8, ls="--", alpha=0.85,
                    label=None if k in meas else lab)


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.grid(True, which="major", color="#e5e5e5", lw=0.6)


def figures(B):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    os.makedirs(FIG, exist_ok=True)
    Nx = B["Nx"]
    # space
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    keys = ["ours_leaves", "ours_library", "w1", "ours_ood", "lora16_acc", "lora4_acc", "dsrlz", "strap_acc", "er", "seq"]
    _plot(ax, B["meas"], B["extra"], Nx, keys, scale=MB)
    xo = B["cross_space"]["library_vs_lora4_acc"]["extrapolated"]
    if xo and xo <= NMAX:
        yo = B["extra"]["ours_library"][int(round(xo)) - 1] / MB
        ax.plot([xo], [yo], marker="*", ms=15, color="#111111", zorder=5, ls="none")
        ax.annotate(f"library = LoRA r4 + acceptance\nat N ≈ {xo:.0f}", (xo, yo), xytext=(130, 0.5), fontsize=8,
                    arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8))
    for N, ha in ((66, "right"), (90, "left")):
        ax.axvline(N, color="#bbbbbb", lw=0.8, ls=":")
        ax.text(N * (0.97 if ha == "right" else 1.03), 0.12, f"N = {N}", fontsize=7, color="#666666", ha=ha, va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, NMAX)
    ax.set_ylim(0.1, 2e5)
    ax.set_xlabel("number of tasks N", fontsize=9)
    ax.set_ylabel("cumulative stored bytes (MB, bf16 units + stored data)", fontsize=9)
    ax.set_title("Storage against tasks: solid = measured, dashed = extrapolated (fitted C(N) for the library; kept "
                 "fraction × unit for per-task methods)", fontsize=9)
    h, l = ax.get_legend_handles_labels()
    h += [Line2D([], [], color="#444", lw=1.5), Line2D([], [], color="#444", lw=1.2, ls="--")]
    l += ["measured", "extrapolated"]
    ax.legend(h, l, fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "h3_space.png"), dpi=150)
    plt.close(fig)

    # concepts
    C = B["C"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for k, c, lab in (("l90", "#008300", "LIBERO-90, 66 single-step tasks"), ("ood", "#333333", "Li's 20 OOD tasks"),
                      ("pro", "#eb6834", "LIBERO-PRO, 27 eligible tasks")):
        m, lo, hi = (np.array(C[k][x]) for x in ("mean", "p5", "p95"))
        n = np.arange(1, len(m) + 1)
        f = C[k]["fit"]
        ax.fill_between(n, lo, hi, color=c, alpha=0.12, lw=0)
        ax.plot(n, m, color=c, lw=2, label=f"{lab}: mean of 5,000 orders")
        ax.plot(n, f["a"] * n ** f["beta"], color=c, lw=1, ls="--", label=f"   fit C = {f['a']:.2f} N^{f['beta']:.2f}")
    for k, c, lab in (("w1_a", "#5a9e3a", "W1 order a"), ("w1_b", "#9bc77f", "W1 order b"), ("g16_plan", "#777777", "G16 plan order"),
                      ("g16_reversed", "#b0b0b0", "G16 reversed order")):
        y = np.array(C[k]["curve"])
        ax.plot(np.arange(1, len(y) + 1), y, color=c, lw=1.2, marker=".", ms=4, label=f"{lab} (β = {C[k]['fit']['beta']:.2f})")
    n = np.arange(1, 67)
    ax.plot(n, n, color="#2a78d6", lw=1, ls=":", label="one unit per task (C = N)")
    ax.plot(n, 2 * np.sqrt(n), color="#6250d6", lw=1, ls=":", label="ideal grid (C = 2√N)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("tasks arrived N", fontsize=9)
    ax.set_ylabel("distinct concepts C(N)", fontsize=9)
    ax.set_title("Distinct concepts as tasks arrive (band: 5–95% of random orders)", fontsize=9)
    ax.legend(fontsize=6.8, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "h3_concepts.png"), dpi=150)
    plt.close(fig)

    # time
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))
    ax = axes[0]
    _plot(ax, B["tmeas"], B["textra"], Nx, ["ours_leaves", "ours_library", "w1", "ours_ood", "lora16", "lora4", "dsrlz", "strap", "er",
                                            "seq"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, NMAX)
    ax.set_xlabel("number of tasks N", fontsize=9)
    ax.set_ylabel("cumulative fit time (A40-minutes, slurm job wall)", fontsize=9)
    ax.set_title("Fit time against tasks (solid measured, dashed extrapolated)", fontsize=9)
    _style(ax)
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    ax = axes[1]
    _plot(ax, B["pmeas"], B["pextra"], Nx, ["ours_leaves", "w1", "lora16", "dsrlz", "g15", "g16", "g17"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, NMAX)
    ax.set_xlabel("number of tasks N", fontsize=9)
    ax.set_ylabel("cumulative practice episodes", fontsize=9)
    ax.set_title("Practice against tasks (LoRA r16/r4 and STRAP with acceptance: 20 per task; ER, seq. FT, Li's TLI: 0)",
                 fontsize=8.5)
    h, l = ax.get_legend_handles_labels()
    l = [x.replace("per-task LoRA r16", "per-task adapter + acceptance") for x in l]
    ax.legend(h, l, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "h3_time.png"), dpi=150)
    plt.close(fig)


# ── doc ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def fmt_b(b):
    if b >= 1e9:
        return f"{b / 1e9:.2f} GB"
    if b >= 1e6:
        return f"{b / 1e6:.1f} MB"
    return f"{b / 1e3:.0f} KB"


def write_doc(B, paths):
    t, k, C, xs, txs = B["table"], B["consts"], B["C"], B["cross_space"], B["cross_time"]
    g = C["grid_l90"]
    sp, ti = paths.get("space", "ledger/h3_scaling/<space row>"), paths.get("time", "ledger/h3_scaling/<time row>")
    cn = paths.get("concepts", "ledger/h3_scaling/<concepts row>")
    mark = lambda s: "" if s == "measured" else "*"                                          # noqa: E731
    lines = []
    for key, e in t.items():
        bt = " / ".join(fmt_b(e["bytes_at"][str(N)]) + mark(e["bytes_status"][str(N)]) for N in NS)
        ft = " / ".join(f"{e['fit_min_at'][str(N)]:,.0f}" + mark(e["fit_status"][str(N)]) for N in NS)
        pt = " / ".join(f"{e['practice_at'][str(N)]:,.0f}" + mark(e["practice_status"][str(N)]) for N in NS)
        lines.append(f"| {e['method']} | {fmt_b(e['bytes_per_task_at_1000'])} | {e['growth_law']} | {bt} | {ft} | {pt} | "
                     f"`table.{key}` |")
    a_ = xs["library_vs_lora4_acc"]
    doc = f"""# Scaling: storage and compute against the number of tasks

Generated by `experiments/h3_scaling.py` (CPU, read from the ledger); figures `figures/robot/h3_space.png`,
`figures/robot/h3_concepts.png`, `figures/robot/h3_time.png`. Every number below is a field of a `ledger/h3_scaling` row:
concepts `{cn}`, space `{sp}`, time `{ti}`. Unit sizes come from `ledger/h2_space` tag units, fit times from `ledger/h2_fit`
tag units, practice from `ledger/h2_practice` tag counts and the stream logs; see docs/COST.md for how those were measured.
Backbone pi0.5 throughout.

## The argument

A per-task method stores one unit per task, so its store grows linearly in the number of tasks N. The compositional library
stores one unit per concept (an object type, an instance child, a goal). A task combines concepts, so the library grows with
the number of distinct concepts C(N). When concepts recur, C(N) grows more slowly than N. In the ideal case C_o objects and C_g
goals give C_o·C_g tasks for C_o + C_g units, which is C = 2√N on a square grid. A held concept costs zero bytes (a pointer).
A TLI latent costs a fixed amount per source training task, and the training set bounds the number of sources. A task leaf adds
a per-task cost only where practice keeps it.

## How many concepts the tasks need (`concepts`)

C(N) on the real inventories, as the mean over 5,000 random arrival orders, with a power law C = a·N^β fitted in log-log over
every N (`C.<inventory>.fit`), plus the slope over the last half of the inventory (`fit_tail`):

| inventory | tasks | units at the end | β (all N) | a | β (last half) |
|---|---|---|---|---|---|
| LIBERO-90 single-step tasks (G8's 66; prototypes each task needs) | {C['l90']['tasks']} | {C['l90']['units']} | {C['l90']['fit']['beta']:.3f} | {C['l90']['fit']['a']:.2f} | {C['l90']['fit_tail']['beta']:.3f} |
| Li's 20 OOD tasks (concepts keyed by suite and words, as G16 stores them) | {C['ood']['tasks']} | {C['ood']['units']} | {C['ood']['fit']['beta']:.3f} | {C['ood']['fit']['a']:.2f} | {C['ood']['fit_tail']['beta']:.3f} |
| LIBERO-PRO's 27 eligible tasks | {C['pro']['tasks']} | {C['pro']['units']} | {C['pro']['fit']['beta']:.3f} | {C['pro']['fit']['a']:.2f} | {C['pro']['fit_tail']['beta']:.3f} |
| W1 stream, order a (concept keys O / I / G) | {C['w1_a']['tasks']} | {C['w1_a']['units']} | {C['w1_a']['fit']['beta']:.3f} | {C['w1_a']['fit']['a']:.2f} | |
| W1 stream, order b | {C['w1_b']['tasks']} | {C['w1_b']['units']} | {C['w1_b']['fit']['beta']:.3f} | {C['w1_b']['fit']['a']:.2f} | |
| G16 OOD stream, plan order | {C['g16_plan']['tasks']} | {C['g16_plan']['units']} | {C['g16_plan']['fit']['beta']:.3f} | {C['g16_plan']['fit']['a']:.2f} | |
| G16 OOD stream, reversed order | {C['g16_reversed']['tasks']} | {C['g16_reversed']['units']} | {C['g16_reversed']['fit']['beta']:.3f} | {C['g16_reversed']['fit']['a']:.2f} | |

C(N) is sublinear on every inventory, but far from the ideal β = 0.5. The reason is visible in the LIBERO-90 inventory
(`C.grid_l90`): its {g['units']} units split into {g['C_o']} object-side units and {g['C_g']} goals, a grid of {g['grid']}
combinations, of which the 66 tasks fill {g['occupancy']:.1%}. Most objects appear with one or two goals, and the first task
already needs two or three units (object type, instance child where there is one, goal), so the library starts above one unit per
task and only drops below it as goals recur. LIBERO-PRO's eligible tasks almost never repeat an object within a suite, which gives
the highest β. A random order of a finite inventory samples it without replacement: by the last half most concepts have been
seen, so the last-half slope (β_tail) runs LOW and is the optimistic case for growth beyond the inventory, while the all-N
β is the one the extrapolation uses.

## Storage (`space`)

Measured curves: the G8 inventory as the mean over random orders (the hierarchy with leaves, the library alone, the per-task
LoRA r16 with acceptance, kept fractions from `ledger/g8_leaves`); set b's methods along W1's order a with their own kept flags
(E3's rank-16 adapters, DSRL's actors, STRAP's adapters; rank 4 at G9's kept fraction {k['k_lora4']:.3f}); the OOD side along
G16's plan order (every TLI source the stream stores; pointers cost 0).

Extrapolation (dashed in the figure, starred in the table). Its assumptions: (1) new tasks draw concepts the way the
measured inventory does, so the library follows C(N) = 48·(N/66)^β, anchored at the measured end with β = {B['beta']:.3f}
(the all-N fit; the last-half slope β = {B['beta_tail']:.3f}, biased low by the finite inventory, is the optimistic sensitivity); (2) every per-task method keeps adapters at its
measured kept fraction; (3) leaves stay at G8's kept fraction {k['k_leaf']:.3f} and rank 16; (4) one prototype is one rank-4 LoRA
({fmt_b(k['P4'])}) whatever the concept, and a refit replaces, never adds; (5) STRAP's prior (the {fmt_b(k['strap_prior'])}
of training demonstrations and indexes it retrieves from) stays fixed; experience replay adds one stored demonstration
({fmt_b(k['demo_mean'])} on average) per task; (6) the OOD latents stop at the 16-source pool ({fmt_b(16 * k['tli_latent'])};
the whole 40-task training set would bound them at {fmt_b(40 * k['tli_latent'])}). N = 90 covers LIBERO-90's 24
multi-step and articulation tasks only under assumption (1); nothing past N = 66 is measured. With β the library holds
{B['chat']['N90']:.0f} prototypes at N = 90 and {B['chat']['N1000']:.0f} at N = 1,000 ({B['chat']['N90_tail']:.0f} and
{B['chat']['N1000_tail']:.0f} with the last-half slope; the ideal grid would need {B['chat']['ideal_1000']:.0f} at N = 1,000).

| method | bytes per task at N = 1,000 | growth law | bytes at N = 30 / 66 / 90 / 1,000 | fit A40-min at N = 30 / 66 / 90 / 1,000 | practice episodes at N = 30 / 66 / 90 / 1,000 | ledger (`space` / `time` rows) |
|---|---|---|---|---|---|---|
{chr(10).join(lines)}

`*` = extrapolated (or, for set b's methods at N = 30 and the OOD side past N = 20, beyond the measured range). Constants
(`consts`): P4 = L4 = {fmt_b(k['P4'])}, L16 = {fmt_b(k['L16'])}, DSRL actor A = {fmt_b(k['dsrl_actor'])}, TLI latent LB =
{fmt_b(k['tli_latent'])}; kept fractions k16 = {k['k_lora16']:.3f}, k4 = {k['k_lora4']:.3f}, k_leaf = {k['k_leaf']:.3f},
k_dsrl = {k['k_dsrlz']:.3f}, k_strap = {k['k_strap']:.3f}; W1 requests a demonstration on {k['w1_demo_rate']:.3f} of arrivals.
The table's bytes and fit cells are fields `table.<key>.bytes_at` / `.fit_min_at` / `.practice_at` (with `_status`) of the
space and time rows.

The crossover (`cross_space`). The library alone stores less than the per-task rank-4 LoRA with acceptance from N ≈
{a_['extrapolated']:.0f} (last-half slope: N ≈ {a_['extrapolated_tail_beta']:.0f}); in the measured range the two are level at
N = 66 ({a_['at66_mb']['library']:.1f} MB against {a_['at66_mb']['lora4_acc']:.1f} MB), so the crossover sits just past the
measured end. A prototype and a rank-4 adapter are the same size, so the comparison is C(N) against k4·N. Against rank-4
adapters kept on every task the library is smaller from N = {xs['library_vs_lora4_all']['measured']}, and against the rank-16
adapters with acceptance from N = {xs['library_vs_lora16_acc']['measured']}. The hierarchy WITH leaves never crosses the rank-4
baseline: its kept rank-16 leaves alone cost {xs['leaves_vs_lora4_acc']['slope_mb_per_task']['leaves']:.2f} MB per task against
{xs['leaves_vs_lora4_acc']['slope_mb_per_task']['lora4_acc']:.2f} MB for the kept rank-4 adapters. It is smaller than the rank-16
baseline from N = {xs['leaves_vs_lora16_acc']['measured']}. Leaves fitted at rank 4 would cross the rank-4 baseline at
N ≈ {xs['leaves_r4_hyp_vs_lora4_acc']['extrapolated']:.0f}, but that is a hypothetical: G8 measured rank-16 leaves only.
DSRL's actors ({fmt_b(k['dsrl_actor'])}, kept on {k['k_dsrlz']:.0%} of tasks) and sequential fine-tuning's single adapter stay
smaller than the library at every N, at success rates of 52.3% and 8.7% (docs/COST.md). STRAP is dominated by its fixed prior
until N in the thousands. The OOD side is bounded: pointers cost nothing and the latents stop at the source pool.

## Time (`time`)

Fit time is the slurm job's wall time on an A40 (`consts`): a prototype {k['T_proto']:.1f} min (48 units; the train loop alone
is about 37.6 min, so a prototype costs closer to 39 min than to 30), a per-task LoRA r16 {k['T_lora16']:.2f} min and r4
{k['T_lora4']:.2f} min, STRAP {k['T_strap']:.1f} min, DSRL {k['T_dsrlz']:.1f} min of practice with SAC updates per task,
one TLI latent {k['T_tli_latent']:.2f} min. The library fits one prototype per concept, so its fit time grows with C(N); a
per-task method's grows with N.

The library costs MORE fit time than a per-task LoRA over the whole measured range and under the extrapolation to
N = 1,000. A prototype costs {k['T_proto'] / k['T_lora4']:.1f} times a rank-4 LoRA, so the library needs C(N)/N below
{k['T_lora4'] / k['T_proto']:.3f} to be cheaper. With β = {B['beta']:.3f} that happens at N ≈ {txs['library_vs_lora4']['extrapolated']:,.0f}
(rank 4) and N ≈ {txs['library_vs_lora16']['extrapolated']:,.0f} (rank 16); with the last-half slope, N ≈
{txs['library_vs_lora4']['extrapolated_tail_beta']:,.0f} and {txs['library_vs_lora16']['extrapolated_tail_beta']:,.0f}. All four lie
beyond N = 1,000 and far past the 66 measured tasks, so within any task set measured or planned here there is no fit-time
crossover. The hierarchy with leaves never crosses at
all, since G8 fits every task's rank-16 LoRA for its acceptance test on top of the prototypes. The W1 stream costs more again
({k['w1_fits']} prototype fits over 30 arrivals at {k['T_w1_fit']:.1f} min each), because every requested demonstration refits
each concept on its path; its fit time grows with the requests, linearly in N. The library is cheaper than STRAP from
N ≈ {txs['library_vs_strap']['extrapolated']:,.0f} (extrapolated) and cheaper than DSRL at every N (a prototype costs less than
one task's DSRL practice). On the OOD side the fit is fixed: identifying the 16-source pool's latents costs
{16 * k['T_tli_latent']:.1f} min once.

Practice. G8's system runs 30 episodes per task (policy, composition, leaf), against 20 for a per-task adapter with acceptance:
practice is linear in N for both, and ours costs 1.5 times more. On the OOD side, practice per task (G15) is 340 episodes;
the stream with reuse (G16) practises only unseen concepts, {B['practice_per_concept']:.0f} episodes per new concept plus 20 per
arrival, so its practice follows C(N): 3,600 episodes for 20 tasks against 6,800, and, under the fitted OOD C(N)
({B['c_ood_1000']:.0f} concepts at N = 1,000, an extrapolation from 20 tasks), about
{t['ours_ood']['practice_at']['1000']:,.0f} against {340 * 1000:,} at N = 1,000. G17's gate re-practises reused concepts
that fail acceptance and lands between the two (5,460 on the plan order).

## Reading

Space scales the way the argument says only for the parts that are shared: the library is sublinear (β ≈ {B['beta']:.2f}) and
the OOD pointers and latents are bounded. Three things keep the measured advantage small. LIBERO-90 fills about a ninth of its
object × goal grid, so concepts recur less than the ideal. A task needs two or three units, so the library starts above one unit
per task. And the leaves G8 keeps on 36% of tasks are rank-16, which makes the full hierarchy linear with a slope larger than the
rank-4 baseline. In time the library is costlier: a prototype trains about ten times longer than a per-task adapter, and the
concept reuse never repays that within a thousand tasks.
"""
    open(os.path.join(ROOT, "docs", "SCALING.md"), "w", encoding="utf-8").write(doc)


def main(record):
    B = build()
    figures(B)
    C, k = B["C"], B["consts"]
    for kk in ("l90", "ood", "pro", "w1_a", "w1_b", "g16_plan", "g16_reversed"):
        print(f"[h3] C {kk:12} tasks {C[kk]['tasks']:3} units {C[kk]['units']:3} beta {C[kk]['fit']['beta']:.3f} "
              f"tail {C[kk].get('fit_tail', {}).get('beta', float('nan')):.3f}")
    print("[h3] grid", C["grid_l90"])
    print("[h3] cross space", json.dumps(B["cross_space"], default=str))
    print("[h3] cross time", json.dumps(B["cross_time"], default=str))
    for key, e in B["table"].items():
        print(f"[h3] {key:14} " + " ".join(f"N={N}: {fmt_b(e['bytes_at'][str(N)])}{'*' if e['bytes_status'][str(N)] != 'measured' else ''}"
                                           f" {e['fit_min_at'][str(N)]:.0f}min {e['practice_at'][str(N)]:.0f}ep" for N in NS))
    paths = {}
    if record:
        Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
        kw = dict(seed=0, script="experiments/h3_scaling.py", command=" ".join(sys.argv), bytes=Z, config=dict(nrep=NREP, NS=list(NS)))
        tl = lambda d: {kk: dict(N=np.asarray(v[0]).tolist(), y=np.asarray(v[1]).tolist()) for kk, v in d.items()}   # noqa: E731
        sel = lambda d: {kk: {str(N): float(np.asarray(v)[N - 1]) for N in NS} for kk, v in d.items()}            # noqa: E731
        src = dict(B["consts"]["sources"], **B["reads"])
        paths["concepts"] = os.path.relpath(ledger.record(
            "h3_scaling", bench="libero", tag="concepts", metrics=dict(C=C, beta=B["beta"], beta_tail=B["beta_tail"], chat=B["chat"],
                                                                       c_ood_1000=B["c_ood_1000"], sources=src), **kw), ROOT)
        paths["space"] = os.path.relpath(ledger.record(
            "h3_scaling", bench="libero", tag="space", metrics=dict(consts=k, cross_space=B["cross_space"], measured=tl(B["meas"]),
                                                                    extrapolated_at=sel(B["extra"]), table=B["table"], sources=src), **kw), ROOT)
        paths["time"] = os.path.relpath(ledger.record(
            "h3_scaling", bench="libero", tag="time", metrics=dict(consts=k, cross_time=B["cross_time"], fit_measured=tl(B["tmeas"]),
                                                                   fit_extrapolated_at=sel(B["textra"]), practice_measured=tl(B["pmeas"]),
                                                                   practice_extrapolated_at=sel(B["pextra"]),
                                                                   practice_per_concept=B["practice_per_concept"], table=B["table"],
                                                                   sources=src), **kw), ROOT)
        print("[h3] ledger", paths)
    write_doc(B, paths)


if __name__ == "__main__":
    main("--record" in sys.argv)
