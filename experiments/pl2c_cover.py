"""PLAN L, PROBE L2c (docs/PLAN_LANG.md; pre-registration experiments/prereg/pl2c_cover.py, ledger m3_prereg tag pl2c_cover):
can a SYMBOLIC, demo-free rule recover Li's hand-chosen TLI pairs?  This is the framework's composition step: parse the new
instruction into role-filler parts, retrieve the stored prototypes (training sentences) that cover those parts IN THE SAME ROLE,
and combine them by text-latent interpolation (lifelong_ood/policy/tli.py).  No demo, no reward and no success number chooses the
pair; only the secondary arm's lambda reads demo 0's flow loss (pl2's scorer, unchanged).

THE RULE (fixed before any L2c number; nothing here is tuned):
  parse     roles of a single-step instruction by the existing reader (lang/transplant.roles = policy/bind.phrases, i.e.
            primitives.step_key / object_key / goal_key): verb V (step_key's verb; "pick up X and place it" reads as "put"), object
            phrase, goal phrase.  The object phrase splits at its first relation word (REL) into the HEAD (the moved thing) and
            its LANDMARK ("bowl" | "cookie box" in "bowl on cookie box").  Role-filler pairs: (V, verb), (O, w) for the head's
            content words, (L, w) for the landmark's, (G, w) for the goal phrase's (transplant.content_words: STOP words out).
  pool      the single-step pick-and-place training prompts (transplant.template_of not None) of the 40 training tasks -- the
            pool pl2's TLI used; both policies were trained on the same 40 tasks, so the pool and the picks are policy-free.
  cover     greedy submodular coverage (Wang et al. 2605.07078's selection, on role-filler pairs): a source covers (r, w) iff w is
            one of ITS role-r fillers (a filler in another role covers nothing: L2's pi0 goal:1 / goal:4 failure).  Pick the
            source with the most still-uncovered pairs; ties by (i) the most of the new sentence's pairs covered in total,
            (ii) the same frame (template with {O} / {G} slots), (iii) word-set Jaccard of the two sentences, (iv) the lower
            task index.  Stop at 2 sources (Li's TLI is pairwise).
  slots     TLI's ramp gives t1 the early (approach) phase and t2 the late (carry) phase, so t1 = the pick whose coverage
            leans most to the object roles: argmax (#O + #L pairs covered) - (#G pairs covered); ties: the first pick is t1.
  lambda    auto_li: Li's lambda for the task, exactly as tli_li uses it (experiments/g12_tli.MAP); auto_dl: lambda in
            {24, 12, 36} (pl2's grid and order) with the lowest demo-0 flow loss for the auto pair (pl2's DemoScorer, K 4, seed 1).

    sbatch -W --wrap "$PY experiments/pl2c_cover.py --picks"              agreement with Li's pairs -> results/lang/pl2c/picks.json
    sbatch ... robot_gpu.sbatch SCRIPT="experiments/pl2c_cover.py --demo-lam --backbone pi0_libero"   (GPU)
    $PY experiments/pl2c_cover.py --fresh-launch | --fresh-check          new start states fresh_ood_pl2c (BASE 5100000)
    $PY experiments/pl2c_cover.py --launch [--dry] | --read [--record]   closed loop: 20 tasks x 20 episodes x {zs, tli_li,
                                                                          auto_li, auto_dl} x {pi0.5, pi0}
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
R = os.path.join(ROOT, "results", "robot")
OUT = os.path.join(ROOT, "results", "lang", "pl2c")
SEED = 13
BBS = ("pi05_libero", "pi0_libero")
TEST_DIR, BASE, N_STATES = "fresh_ood_pl2c", 5100000, 20
USED_BASES = (100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1300000, 3300000, 3400000)   # slurm logs
LAMS = (24, 12, 36)                                                     # pl2's grid, in pl2's order (24 first)
REL = r"\b(next to|on top of|on|in|into|onto|at|from|between|under|near|beside|inside|by)\b"
ARMS = ("zs", "tli_li", "auto_li", "auto_dl")
EXCLUDE = "sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw"


def short(bb):
    return bb.split("_")[0]


# ── the rule ─────────────────────────────────────────────────────────────────────────────────────────────────────
def parse(text):
    """-> dict(pairs {(role, filler)}, frame, words, parts): the instruction's role-filler pairs (V, O, L, G)."""
    import lifelong_ood.lang.transplant as LT
    from lifelong_ood.policy import primitives as P
    r = LT.roles(text)
    verb = P.step_key(text).split("|")[0]
    o = r["o"]
    m = re.search(REL, o)
    head, land = (o[:m.start()].strip(), o[m.end():].strip()) if m else (o, "")
    pairs = {("V", verb)} | {("O", w) for w in LT.content_words(head)} | {("L", w) for w in LT.content_words(land)} \
        | {("G", w) for w in LT.content_words(r["g"])}
    fr = LT.template_of(text)
    if fr is None and o and r["g"] and o in text and r["g"] in text:
        fr = text.replace(o, "{O}", 1).replace(r["g"], "{G}", 1)
    return dict(pairs=pairs, frame=fr, words=set(re.findall(r"[a-z]+", text.lower())),
                parts=dict(verb=verb, head=head, landmark=land, goal=r["g"]))


def pool(tx):
    import lifelong_ood.lang.transplant as LT
    return [t for t in range(40) if LT.template_of(tx[t]) is not None]


def cover(text, tx, k=2):
    """-> dict(picks [t ...] in greedy order, t1, t2, trace): the rule above; k sources (k > 2 only for the 3-source report)."""
    N = parse(text)
    S = {t: parse(tx[t]) for t in pool(tx) if tx[t] != text}
    picks, covered, trace = [], set(), []

    def key(t):
        c = N["pairs"] & S[t]["pairs"]
        jac = len(N["words"] & S[t]["words"]) / len(N["words"] | S[t]["words"])
        return (len(c - covered), len(c), S[t]["frame"] == N["frame"], round(jac, 6), -t)
    for _ in range(k):
        cand = [t for t in S if t not in picks]
        ks = sorted(cand, key=key, reverse=True)
        best = ks[0]
        trace.append(dict(pick=best, key=list(key(best)), runner_up=ks[1] if len(ks) > 1 else None,
                          runner_up_key=list(key(ks[1])) if len(ks) > 1 else None,
                          tie_on=next((n for n, a, b in zip(("gain", "total", "frame", "jaccard", "index"), key(best), key(ks[1]))
                                       if a != b), None) if len(ks) > 1 else None))
        picks.append(best)
        covered |= N["pairs"] & S[best]["pairs"]

    def lean(t):
        c = N["pairs"] & S[t]["pairs"]
        return sum(r in ("O", "L") for r, _ in c) - sum(r == "G" for r, _ in c)
    a, b = picks[0], picks[1]
    t1, t2 = (b, a) if lean(b) > lean(a) else (a, b)
    return dict(picks=picks, t1=int(t1), t2=int(t2), trace=trace, pairs=sorted(map(list, N["pairs"])), parts=N["parts"],
                frame=N["frame"], uncovered=sorted(map(list, N["pairs"] - covered)),
                covered_by={str(t): sorted(map(list, N["pairs"] & S[t]["pairs"])) for t in picks})


def full_cover(text, tx, kmax=4):
    """How many greedy sources until every pair is covered (or no source adds any): the 3-source question."""
    N = parse(text)
    S = {t: parse(tx[t]) for t in pool(tx) if tx[t] != text}
    covered, picks = set(), []
    while covered != N["pairs"] and len(picks) < kmax:
        t = max((t for t in S if t not in picks), key=lambda t: (len((N["pairs"] & S[t]["pairs"]) - covered), -t))
        if not (N["pairs"] & S[t]["pairs"]) - covered:
            break
        picks.append(t)
        covered |= N["pairs"] & S[t]["pairs"]
    return dict(n_sources=len(picks), picks=picks, coverable=covered == N["pairs"],
                never_covered=sorted(map(list, N["pairs"] - covered)))


def picks():
    import g12_tli as T
    from lifelong_ood.policy import data as D
    tx = D.Demos().tasks
    pl = json.load(open(os.path.join(R, "g4_plan.json")))
    out, agree = {}, dict(exact=0, same_set=0, t1=0, t2=0, n=0)
    for k, v in sorted(pl["tasks"].items()):
        a, b, lam = T.MAP[v["language"].replace(" ", "_")]
        li = (T.KEY[a], T.KEY[b], int(lam))
        c = cover(v["language"], tx)
        c3 = cover(v["language"], tx, k=3)
        fc = full_cover(v["language"], tx)
        ex = (c["t1"], c["t2"]) == li[:2]
        r = dict(text=v["language"], suite=v["suite"], index=v["index"], li=dict(t1=li[0], t2=li[1], lam=li[2], t1_text=tx[li[0]],
                 t2_text=tx[li[1]]), auto=dict(t1=c["t1"], t2=c["t2"], lam_li=li[2], t1_text=tx[c["t1"]], t2_text=tx[c["t2"]]),
                 exact=ex, same_set={c["t1"], c["t2"]} == set(li[:2]), t1_agree=c["t1"] == li[0], t2_agree=c["t2"] == li[1],
                 rule=c, third=dict(pick=c3["picks"][2], gain=c3["trace"][2]["key"][0], text=tx[c3["picks"][2]]), full_cover=fc,
                 li_covers=dict(t1=sorted(map(list, parse(v["language"])["pairs"] & parse(tx[li[0]])["pairs"])),
                                t2=sorted(map(list, parse(v["language"])["pairs"] & parse(tx[li[1]])["pairs"]))),
                 arm_li=T.arm(v), arm_auto_li=f"tli@{c['t1']}.{c['t2']}.{li[2]}")
        out[k] = r
        agree["n"] += 1
        for q in ("exact", "same_set"):
            agree[q] += int(r[q])
        agree["t1"] += int(r["t1_agree"])
        agree["t2"] += int(r["t2_agree"])
        print(f"{k:10} '{v['language']}': auto ({c['t1']}, {c['t2']}) li ({li[0]}, {li[1]}) lam {li[2]} "
              f"{'EXACT' if ex else 'same set' if r['same_set'] else 't1' if r['t1_agree'] else 't2' if r['t2_agree'] else 'differ'}"
              f" | uncovered {c['uncovered']} | full cover needs {fc['n_sources']} (coverable {fc['coverable']})", flush=True)
    os.makedirs(OUT, exist_ok=True)
    res = dict(agreement=agree, pool=pool(tx), per_task=out, base_pool_parse={t: sorted(map(list, parse(tx[t])["pairs"])) for t in pool(tx)})
    json.dump(res, open(os.path.join(OUT, "picks.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(agree))


# ── secondary arm: lambda on demo 0 ─────────────────────────────────────────────────────────────────────────────
def demo_lam(bb):
    """Demo-0 flow loss of the auto pair at each lambda in LAMS (pl2's scorer: load_demo, DemoScorer K 4 seed 1, Composer tli),
    and of tli_li (a check against pl2's stored l0['tli_li']).  -> results/lang/pl2c/demo_lam_<short>.json"""
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy.backbone import PolicyBackbone
    import lifelong_ood.lang.transplant as LT
    import pl2_compose as PC
    pk = json.load(open(os.path.join(OUT, "picks.json")))["per_task"]
    idx = json.load(open(os.path.join(PC.OUT, "demos.json")))[bb]
    bk = PolicyBackbone(bb, device="cuda")
    demos = D.Demos()
    f = os.path.join(OUT, f"demo_lam_{short(bb)}.json")
    res = json.load(open(f)) if os.path.exists(f) else {}
    for k, r in sorted(pk.items()):
        if k in res:
            continue
        d0 = PC.load_demo(idx[k]["d0"], bk.H)
        s0 = LT.DemoScorer(bk, d0["frames"], d0["chunks"], d0["states"], d0["carry"], d0["q"], K=PC.K_DRAWS, seed=1)
        lat = {t: PC._tli_lat(bk, bb, demos, t) for t in {r["auto"]["t1"], r["auto"]["t2"], r["li"]["t1"], r["li"]["t2"]}}

        def L(t1, t2, lam):
            return s0.loss(r["text"], LT.Composer(bk, (), tli=dict(d=lat[t1] - lat[t2], lam=lam)))
        lams = {str(lam): L(r["auto"]["t1"], r["auto"]["t2"], lam) for lam in LAMS}
        best = min(LAMS, key=lambda lam: lams[str(lam)])
        res[k] = dict(pair=[r["auto"]["t1"], r["auto"]["t2"]], lams=lams, lam=int(best), zs=s0.loss(r["text"]),
                      tli_li=L(r["li"]["t1"], r["li"]["t2"], r["li"]["lam"]), demo_d0=idx[k]["d0"])
        old = json.load(open(os.path.join(PC.OUT, "offline", f"{short(bb)}_{PC.sh(k)}.json")))
        res[k]["pl2_check"] = dict(zs=old["l0"]["zs"], tli_li=old["l0"].get("tli_li"),
                                   pair24=old["sel"]["tli_sel"]["pairs"].get(f"{r['auto']['t1']}.{r['auto']['t2']}"))
        print(f"[pl2c] {short(bb)} {k}: auto {res[k]['pair']} lams {lams} -> {best}; zs {res[k]['zs']:.5f} (pl2 {old['l0']['zs']:.5f}); "
              f"tli_li {res[k]['tli_li']:.5f} (pl2 {old['l0'].get('tli_li')})", flush=True)
        tmp = f + f".{os.getpid()}.tmp"
        json.dump(res, open(tmp, "w", encoding="utf-8"), indent=1)
        os.replace(tmp, f)


# ── start states ─────────────────────────────────────────────────────────────────────────────────────────────────
def fresh_launch(dry):
    assert not any(b <= BASE < b + 1000 or BASE <= b < BASE + N_STATES for b in USED_BASES), BASE
    assert not os.path.exists(os.path.join(R, TEST_DIR)) or not os.listdir(os.path.join(R, TEST_DIR)), "set exists"
    for suite in ("libero_goal_ood", "libero_spatial_ood"):
        cmd = ["sbatch", "--parsable", "-p", "overcap", "-A", "overcap", "--requeue", f"--exclude={EXCLUDE}",
               f"--export=ALL,SUITE={suite},TASKS={'+'.join(map(str, range(10)))},N={N_STATES},BASE={BASE},FRESH_DIR={TEST_DIR}",
               "experiments/fresh_states.sbatch"]
        print(" ".join(cmd))
        if not dry:
            print(subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=ROOT).stdout.strip())


def fresh_check():
    """Assert the new set is unused: 20 states per task, and no state within 1e-6 of any state of any earlier OOD set."""
    out = {}
    for suite in ("libero_goal_ood", "libero_spatial_ood"):
        for t in range(10):
            x = np.load(os.path.join(R, TEST_DIR, f"{suite}_{t}.npy"))
            assert len(x) == N_STATES, (suite, t, x.shape)
            dmin, sets = np.inf, []
            for f in sorted(glob.glob(os.path.join(R, "fresh_ood_*", f"{suite}_{t}.npy"))):
                if os.path.basename(os.path.dirname(f)) == TEST_DIR:
                    continue
                y = np.load(f)
                if y.shape[1] != x.shape[1]:
                    continue
                dmin = min(dmin, float(np.min(np.linalg.norm(x[:, None] - y[None], axis=-1))))
                sets.append(os.path.basename(os.path.dirname(f)))
            assert dmin > 1e-6, (suite, t, dmin)
            out[f"{suite}_{t}"] = dict(min_dist=dmin, compared=sets)
            print(suite, t, f"min distance to earlier sets {dmin:.4f} over {len(sets)} sets", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "fresh_check.json"), "w", encoding="utf-8"), indent=1)


# ── closed loop ──────────────────────────────────────────────────────────────────────────────────────────────────
def arms_of(bb, k):
    pk = json.load(open(os.path.join(OUT, "picks.json")))["per_task"][k]
    dl = json.load(open(os.path.join(OUT, f"demo_lam_{short(bb)}.json")))[k]
    a = pk["auto"]
    return dict(zs="zs", tli_li=pk["arm_li"], auto_li=pk["arm_auto_li"], auto_dl=f"tli@{a['t1']}.{a['t2']}.{dl['lam']}")


def launch(dry, lanes=6):
    import pl2_compose as PC
    pl = json.load(open(os.path.join(R, "g4_plan.json")))
    assert os.path.exists(os.path.join(OUT, "fresh_check.json")), "run --fresh-check first"
    specs = []
    for bb in BBS:
        for k, v in sorted(pl["tasks"].items()):
            for arm in dict.fromkeys(arms_of(bb, k).values()):
                specs.append(PC._spec(v["index"], arm, v["suite"], bb, f"pl2c_{k.split(':')[0]}", N_STATES, TEST_DIR))
    print(len([s for s in specs if s is not None]), "instances;", sum(s is None for s in specs), "unbuildable")
    PC.submit(specs, lanes=lanes, dry=dry)


def read(record):
    import e1_read as E1
    import h1_factor as HF
    import pl2_compose as PC
    pl = json.load(open(os.path.join(R, "g4_plan.json")))
    named = HF.named_objects()
    pk = json.load(open(os.path.join(OUT, "picks.json")))
    res = {}
    for bb in BBS:
        x, missing, per, parts = {}, [], {}, {}
        for k, v in sorted(pl["tasks"].items()):
            am = arms_of(bb, k)
            z = {n: PC._eps(f"pl2c_{k.split(':')[0]}", v, a, bb) for n, a in am.items()}
            if None in z.values() or any(len(e) != N_STATES for e in z.values()):
                missing.append(k)
                continue
            x[k] = {n: {t: int(bool(e["success"])) for t, e in ep.items()} for n, ep in z.items()}
            nm = named.get((v["suite"], v["index"]))
            per[k] = {n: int(sum(s.values())) for n, s in x[k].items()} | dict(arms=am, exact=pk["per_task"][k]["exact"])
            parts[k] = {n: dict(g=int(sum(e.get("bound") == nm for e in ep.values())),
                                pg=int(sum(bool(e["success"]) for e in ep.values() if e.get("bound") == nm))) for n, ep in z.items()}
        C = E1.compare
        r = dict(backbone=bb, tasks=len(x), missing=missing, per_task=per, parts=parts)
        if x:
            r["totals"] = {n: int(sum(per[k][n] for k in per)) for n in ARMS}
            r["n"] = N_STATES * len(x)
            for arm in ("auto_li", "auto_dl"):
                h1 = C({k: {"x": z[arm], "y": z["tli_li"]} for k, z in x.items()})
                h1["passes"] = bool(h1["diff"]["lo"] >= -0.05)
                h2 = C({k: {"x": z[arm], "y": z["zs"]} for k, z in x.items()})
                h2["passes"] = bool(h2["x"] > h2["y"] and h2["sign_p"] < 0.05)
                r[f"H1_{arm}"], r[f"H2_{arm}"] = h1, h2
            r["tli_li_vs_zs"] = C({k: {"x": z["tli_li"], "y": z["zs"]} for k, z in x.items()})
            dif = [k for k in x if not pk["per_task"][k]["exact"]]
            if dif:
                r["differing_auto_li_vs_tli_li"] = C({k: {"x": x[k]["auto_li"], "y": x[k]["tli_li"]} for k in dif})
            print(f"\n[pl2c] {short(bb)}: {len(x)} tasks, missing {missing}; totals {r['totals']} of {r['n']}")
            for n in ("H1_auto_li", "H2_auto_li", "H1_auto_dl", "H2_auto_dl", "tli_li_vs_zs", "differing_auto_li_vs_tli_li"):
                if n in r:
                    h = r[n]
                    print(f"  {n:28} {h['x']} vs {h['y']} of {h['n']}, discordant {h['discordant']}, sign p {h['sign_p']:.3g}, "
                          f"diff {h['diff']['diff']:+.3f} [{h['diff']['lo']:+.3f}, {h['diff']['hi']:+.3f}]"
                          + (f" passes {h['passes']}" if "passes" in h else ""))
            for k in per:
                print(f"  {k:10} {'EXACT ' if per[k]['exact'] else 'differ'} " + " ".join(f"{n} {per[k][n]:2d}" for n in ARMS)
                      + f"  {per[k]['arms']['auto_li']} vs {per[k]['arms']['tli_li']}, dl {per[k]['arms']['auto_dl']}")
        res[short(bb)] = r
    json.dump(res, open(os.path.join(OUT, "read.json"), "w", encoding="utf-8"), indent=1)
    if record:
        from lifelong_ood import ledger
        print(ledger.record("pl2c_read", bench="libero_ood", tag="read", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(states=TEST_DIR, base=BASE, episodes=N_STATES, arms=ARMS, backbones=list(BBS),
                                        prereg="m3_prereg tag pl2c_cover"),
                            script="experiments/pl2c_cover.py", command=" ".join(sys.argv)))


def record_picks():
    from lifelong_ood import ledger
    pk = json.load(open(os.path.join(OUT, "picks.json")))
    dl = {short(bb): json.load(open(os.path.join(OUT, f"demo_lam_{short(bb)}.json"))) for bb in BBS}
    m = dict(agreement=pk["agreement"], picks={k: dict(auto=[r["auto"]["t1"], r["auto"]["t2"]], li=[r["li"]["t1"], r["li"]["t2"]],
                                                       lam_li=r["li"]["lam"], exact=r["exact"], full_cover=r["full_cover"]["n_sources"],
                                                       uncovered=r["rule"]["uncovered"])
                                              for k, r in pk["per_task"].items()},
             demo_lam={b: {k: dict(lam=v["lam"], lams=v["lams"]) for k, v in d.items()} for b, d in dl.items()})
    print(ledger.record("pl2c_picks", bench="libero_ood", tag="offline", seed=SEED, metrics=m,
                        bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=dict(lams=LAMS, rel=REL),
                        script="experiments/pl2c_cover.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in ("picks", "demo-lam", "fresh-launch", "fresh-check", "launch", "read", "record-picks", "dry", "record"):
        ap.add_argument("--" + f, action="store_true")
    ap.add_argument("--backbone", default="pi05_libero")
    a = ap.parse_args()
    if a.picks:
        picks()
    if a.demo_lam:
        demo_lam(a.backbone)
    if a.fresh_launch:
        fresh_launch(a.dry)
    if a.fresh_check:
        fresh_check()
    if a.record_picks:
        record_picks()
    if a.launch:
        launch(a.dry)
    if a.read:
        read(a.record)
