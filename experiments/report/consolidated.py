"""The page's opening sections: how the method differs from Wang et al. (ICLR 2027), the experimental design, and consolidated
tables of every method's score per benchmark (best in bold, second best underlined).  Every number is read from ledger/."""
from .base import esc, row, table


def _mark(vals, higher=True):
    """-> per value: 'b' (best), 'u' (second best) or '' (ties share a mark; None ignored)."""
    xs = sorted({v for v in vals if v is not None}, reverse=higher)
    best, second = (xs[0] if xs else None), (xs[1] if len(xs) > 1 else None)
    return ["b" if v is not None and v == best else "u" if v is not None and v == second else "" for v in vals]


def _cell(v, mark, fmt=str):
    if v is None:
        return "–"
    s = fmt(v)
    return f"<strong>{s}</strong>" if mark == "b" else f"<u>{s}</u>" if mark == "u" else s


def _cost(side):
    r = row("h2_cost", side)
    return r["rows"] if r else []


def iclr_diff():
    rows = [
        ["Domain", "image generation (ColorMNIST, CelebA)", "robot control (LIBERO; five frozen flow-matching VLAs)"],
        ["What a prototype is", "a mode of a noisy marginal p_t(x_t), with a local Gaussian; the noise level t sets its abstraction",
         "a concept of a task phase: a pointer to the policy's own velocity field under a training instruction (held concept), or a "
         "rank-4 LoRA change to that field fitted on the concept's demonstration frames (new concept)"],
        ["How prototypes are found", "unsupervised: score ascent from the query at several noise levels; no labels, no training",
         "supervised by the instruction: object, instance and goal concepts are read from it; the policy's own flow loss decides "
         "held (point) or new (fit)"],
        ["Evidence used", "one OOD query image, used at test time", "no demonstration (held concepts) or one demonstration per new "
         "concept, plus simulator practice"],
        ["Selection", "greedy submodular coverage of the query, per dimension", "practice: episodes choose each concept's source "
         "instruction and, per task, the operator; acceptance keeps a unit only if it beats the policy"],
        ["Composition operator", "a closed-form product of Gaussian experts, applied at every denoising step through the composable "
         "diffusion sampler (superposition)", "a switch in time at the policy's own grasp, one concept per phase; or text-latent "
         "interpolation at the instruction encoder. Superposed operators (mixed fields, both units always on) fail here"],
        ["Abstraction levels", "noise levels form a hierarchy of prototypes", "concept hierarchy: object type, instance child, task "
         "leaf"],
        ["Over time", "one query at a time; no memory", "a continual library: units stored, reused by later tasks, demonstrations "
         "requested only when practice fails; nothing earlier is damaged"],
        ["New primitives", "not addressed (stated as future work)", "detected by the policy's likelihood (AUC 1.0), fitted, stored "
         "and reused"],
        ["Evaluation", "attribute correctness and diversity of samples for held-out classes", "task success on new start states, "
         "pre-registered, paired, with exact sign tests and task-level bootstrap"],
    ]
    out = ("<p>This work starts from Wang et al.'s program (ICLR 2027 submission, <i>Test-Time Compositional Generalization in "
           "Diffusion Models via Prototype Discovery</i>): a pretrained diffusion model already holds reusable prototypes, and an "
           "out-of-distribution query can be written as a composition of them. We carry the program from image generation to a "
           "frozen robot policy and change four things: what a prototype is (a phase of behaviour, held or fitted), how it is chosen "
           "(by practice), how prototypes combine (in time, at the grasp), and how they persist (a continual library). The present "
           "system is supervised by the instruction's object-and-goal grammar; Wang et al.'s discovery is unsupervised, and closing "
           "that gap is the end goal of this line of work.</p>")
    out += table(["", "Wang et al. (ICLR 2027)", "This work"], rows, "Methodology, side by side.", num_from=9)
    return out


def design():
    out = ("<p><b>Policies.</b> Five frozen flow-matching VLAs: pi0.5 and pi0 (openpi, PyTorch), SmolVLA and X-VLA (LeRobot), and "
           "GR00T N1.5; each was fine-tuned on LIBERO's 40 training tasks and is never updated. FLOWER is wrapped but trained on one "
           "10-task suite per checkpoint, which does not match the protocol.</p>"
           "<p><b>Benchmarks.</b> (1) LIBERO-90 tasks the policies never saw, one demonstration per new concept (66 single-step "
           "tasks; 29 on the other policies). (2) Li et al.'s compositional OOD suites (20 tasks, each a training task's object "
           "paired with another's target; no demonstration of any test task). (3) LIBERO-PRO's task split (a training scene with a "
           "new instruction and success test; the 27 of 40 tasks with one object, one goal and one grasp).</p>"
           "<p><b>Start states.</b> LIBERO's placement sampler under fixed seeds, saved once per task, so every arm is scored from "
           "the same never-used states: practice on set a, design decisions on earlier sets, each confirmatory test on a set no "
           "decision saw (sets b, c, d, e, f). LIBERO-PRO is also scored on its own published start states.</p>"
           "<p><b>Protocol.</b> Every confirmatory read is pre-registered (ledger m3_prereg) before its first episode: hypotheses, "
           "arms, start states, noise seed. Episodes are paired across arms (same start state, same seed). Tests: exact sign test "
           "on discordant pairs; task-level bootstrap 95% interval of the success difference; non-inferiority at a margin of 0.05; "
           "Holm correction where hypotheses share a family. 20 test episodes per task.</p>"
           "<p><b>Metrics.</b> Success on new start states (accuracy); bytes stored per task and cumulatively (space); fit "
           "A40-minutes, practice episodes and demonstrations per task, and latency per action chunk (time).</p>"
           "<p><b>Baselines.</b> Per-task LoRA (rank 16 and 4, with and without the same acceptance test); retrieval (STRAP); "
           "noise-space RL on the frozen policy (DSRL); sequential fine-tuning and experience replay through the same stream; "
           "Li et al.'s prompt switching and text-latent interpolation with hand-chosen pairs; velocity-field mixing (GPC); "
           "activation injection. Every baseline runs on the same frozen policy, tasks, start states and seed.</p>")
    return out


def _tbl(head, rows_, caption, score_cols, lower=()):
    """rows_: lists of raw values; score_cols: column indices marked best (higher) / second; lower: marked by lowest."""
    base_ = [str(r[0]).lower().startswith(("policy alone", "the policy alone", "pi0.5 given", "published")) for r in rows_]
    marks = {c: _mark([None if (c in lower and b) else r[c] for r, b in zip(rows_, base_)], higher=c not in lower) for c in score_cols}
    body = []
    for i, r in enumerate(rows_):
        body.append([esc(str(r[0]))] + [_cell(v, marks[c][i] if c in marks else "", (lambda x: f"{x:,}" if isinstance(x, int) else
                                                                                     ("0" if isinstance(x, float) and x == 0 else f"{x:.2f}" if isinstance(x, float) and abs(x) < 1 else f"{x:.1f}" if isinstance(x, float) else str(x))))
                                        for c, v in enumerate(r) if c > 0])
    return table(head, body, caption, num_from=1)


def results():
    out = ("<p>Every method's score, by benchmark. Success counts are out of the episodes shown; within a column the best is in "
           "<strong>bold</strong> and the second best <u>underlined</u> (for bytes, fit time and practice, the lowest). All arms "
           "share tasks, start states and seed within a table.</p>")
    lib = _cost("libero90")
    for set_, cap in (("d", "LIBERO-90, pi0.5, one demonstration per new concept: 66 tasks, start states set d (1,320 episodes)."),
                      ("b", "LIBERO-90, pi0.5: the 29 library tasks, start states set b (580 episodes), against every adapter, "
                            "retrieval and RL baseline."),
                      ("b30", "LIBERO-90, pi0.5, continual learning: the 30-task stream, start states set b (600 episodes).")):
        rs = [r for r in lib if r.get("set") == set_]
        if not rs:
            continue
        rows_ = [[r["method"], r["k"], round(r["bytes_task"] / 1e6, 1), round(r["fit_job_min_task"], 1), int(round(r["practice_task"])),
                  round(r["demos_task"], 2)] for r in rs]
        out += _tbl(["method", "successes", "MB / task", "fit A40-min / task", "practice / task", "demos / task"], rows_, cap,
                    score_cols=(1, 2, 3, 4), lower=(2, 3, 4))
    pol = _cost("libero90_policies")
    if pol:
        meths = list(dict.fromkeys(r["method"].split(" (")[0] for r in pol))
        pols = list(dict.fromkeys(r["policy"] for r in pol))
        rows_ = [[m] + [next((r["k"] for r in pol if r["method"].startswith(m) and r["policy"] == p), None) for p in pols] for m in meths]
        out += _tbl(["method"] + [f"{p} (of 580)" for p in pols], rows_,
                    "LIBERO-90 on three more policies: 29 tasks, start states set d (G19).", score_cols=tuple(range(1, len(pols) + 1)))
    ood = _cost("ood")
    g21 = {b: row("g21_operators", f"read_{b}") for b in ("pi05", "pi0")}
    if ood:
        pols = list(dict.fromkeys(r["policy"] for r in ood))
        nm = lambda m: "Policy alone" if m.startswith("Policy alone") else m                      # noqa: E731
        meths = list(dict.fromkeys(nm(r["method"]) for r in ood if "Stream" not in r["method"]))
        rows_ = [[m] + [next((r["k"] for r in ood if nm(r["method"]) == m and r["policy"] == p), None) for p in pols] for m in meths]
        key = {"pi0.5": "pi05", "pi0": "pi0"}
        for lab, k in (("Velocity mixing after GPC, practice-chosen pair (G21)", "gpc_sel"),
                       ("Activation injection, practice-chosen pair (G21)", "inj_sel")):
            rows_.append([lab] + [(g21[key[p]]["totals"][k] if p in key and g21.get(key[p]) else None) for p in pols])
        rows_.sort(key=lambda r: r[0] == "Policy alone")
        out += _tbl(["method"] + [f"{p} (of 400)" for p in pols], rows_,
                    "Li et al.'s compositional OOD suites, no demonstration of any test task: 20 tasks, start states set e. GR00T "
                    "N1.5 (G23) is running.", score_cols=tuple(range(1, len(pols) + 1)))
    pro = row("g22_libero_pro", "read_pi05")
    if pro:
        ps = pro["per_suite"]
        suites = ("goal", "spatial", "object", "ten")
        rows_ = [["pi0.5 given the old instruction (published protocol)"] + [ps[s]["pro_origprompt"] for s in suites],
                 ["pi0.5 given the new instruction"] + [ps[s]["pro_zs"] for s in suites],
                 ["the system (ours)"] + [ps[s]["pro_system"] for s in suites],
                 ["published (LIBERO-PRO, task split)"] + [ps[s]["published_task"] for s in suites]]
        out += _tbl(["arm", "Goal", "Spatial", "Object", "LIBERO-10"], rows_,
                    f"LIBERO-PRO's task split, pi0.5, success rate on LIBERO-PRO's own start states over the tasks of the system's "
                    f"form (G22; {sum(ps[s]['eligible'] for s in suites)} of 40 tasks).", score_cols=(1, 2, 3, 4))
    ab = row("g20_ablation", "read")
    if ab:
        s, c, o = ab["libero_90"]["system"], ab["libero_90"]["composition"], ab["ood"]["system"]
        full_l, full_c, full_o = s["policy"]["x"], c["obj_only"]["x"], o["zs"]["x"]
        rows_ = [["full system", full_l, "", full_o],
                 ["− task leaves", s["no_leaves"]["y"], "", None], ["− practice (composition always served)", s["no_practice"]["y"], "", None],
                 ["− composition (per-task LoRA + acceptance)", s["no_composition"]["y"], "", None],
                 ["− the goal unit (object unit only)", None, c["obj_only"]["y"], None], ["− the object unit (goal unit only)", None, c["goal_only"]["y"], None],
                 ["− the switch at the grasp (both units always on)", None, c["ungated"]["y"], None],
                 ["− operator choice: grasp switch only", None, "", o["psel"]["y"]], ["− operator choice: TLI only", None, "", o["tsel"]["y"]],
                 ["− source practice (word-matched sources)", None, "", o["word"]["y"]], ["− the grasp trigger (Li's fixed step)", None, "", o["fixed"]["y"]],
                 ["− the goal source (object source only)", None, "", o["obj_src_only"]["y"]], ["− the object source (goal source only)", None, "", o["goal_src_only"]["y"]],
                 ["the policy alone", s["policy"]["y"], "", o["zs"]["y"]]]
        rows_[0][2] = full_c
        for r in rows_:
            r[2] = None if r[2] == "" else r[2]
        out += _tbl(["removed", "LIBERO-90 system (of 1,320)", "LIBERO-90 composition, always on (of 1,320)", "Li's OOD (of 400)"], rows_,
                    "Ablation on pi0.5 (G20): each component removed, paired with the full system on the same start states (LIBERO-90 "
                    "set d; Li's suites set e).", score_cols=(1, 2, 3))
    return out
