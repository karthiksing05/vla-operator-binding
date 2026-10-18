"""Part I of the results page: the current program -- a factored, hierarchical prototype library for a frozen policy (plans D,
E, W; docs/METHOD.md).  Each function returns one section's HTML from ledger rows; a read not yet recorded renders as Running."""
from . import diagrams
from .base import L, esc, fig, pending, row, table, video


def _ci(d):
    return f"{d['diff']:+.3f}, 95% interval {d['lo']:+.3f} to {d['hi']:+.3f}"


# ── the reads at a glance ─────────────────────────────────────────────────────────────────────────────────────────────
def _v_d1(m):
    h1, h2 = m["hypotheses"]["H1"], m["hypotheses"]["H2"]
    return f"goal primitives: {h1['x']} vs {h1['y']} zero-shot; vs per-task LoRA + acc. {h2['x']} vs {h2['y']} (non-inferiority fails)"


def _v_e1(m):
    h1, h2 = m["hypotheses"]["H1"], m["hypotheses"]["H2"]
    return f"object + goal library: {h1['x']} vs {h1['y']}; vs LoRA + acc. {h2['x']} vs {h2['y']} ({_ci(h2['diff'])})"


def _v_e3(m):
    h1, h2 = m["hypotheses"]["H1"], m["hypotheses"]["H2"]
    return (f"hierarchical library: {h1['x']} vs {h1['y']}; vs LoRA + acc. {h2['x']} vs {h2['y']} ({_ci(h2['diff'])}); "
            f"{m['bytes']['prototypes']} prototypes, {m['bytes']['library'] / 1e6:.1f} MB")


def _v_stream(m):
    return "; ".join(f"order {o}: {m[o]['demos']} demos, served {m[o]['H1']['x']} vs {m[o]['H1']['y']}, no damage "
                     f"{'holds' if (m[o].get('damage') or {}).get('noninferior') else 'not established'}" for o in ("a", "b"))


def _v_e5(m):
    h = m["hypotheses"]
    return (f"{h['H1']['x']} vs {h['H1']['y']} zero-shot; vs LoRA + acc. {_ci(h['H2']['diff'])}; pooled with E3 "
            f"{_ci(h['H3']['diff'])}")


READS = [  # (label, experiment, tag, pre-registered, section, verdict)
    ("D1 factored goal primitives", "d1_factored", "read", True, "c-factored", _v_d1),
    ("E0 object primitives (screen)", "e0_object", "screen", False, "c-factored",
     lambda m: f"library {m['totals']['all']} vs zero-shot {m['totals']['zs']} of 300; LoRA {m['totals']['lora']} of 290"),
    ("data defect: shared instructions", "l90_scenes", "map", False, "c-factored",
     lambda m: f"{len(m['wrong_scene_tasks'])} tasks had another scene's demonstration; fixed"),
    ("E1 factored library", "e1_factored", "read", True, "c-factored", _v_e1),
    ("E2 instance children (screen)", "e2_instance", "screen", False, "c-hier",
     lambda m: f"children {m['totals']['scratch']} vs type {m['totals']['type']} of 40; LoRA {m['totals']['lora']}"),
    ("E3 hierarchical library", "e3_hier", "read", True, "c-hier", _v_e3),
    ("i4 the factors in weight space", "i4_factors", "read", False, "c-hier",
     lambda m: f"object prototypes nearly orthogonal (cos {m['similarity']['object_object']:.3f}); stable without one demo "
               f"(cos {m['stability']['same_object']:.2f})"),
    ("E5 replication on 37 new tasks", "e5_replicate", "read", True, "c-repl", _v_e5),
    ("G0 recombination, held out", "g0_recombine", "read", True, "c-gen",
     lambda m: f"concepts from other tasks alone {m['hypotheses']['H1']['x']} vs {m['hypotheses']['H1']['y']} zero-shot (fails); "
               f"with practice {m['hypotheses']['H1b']['x']}; vs monolithic same-data adapter {m['hypotheses']['H2']['y']} (passes)"),
    ("G2 adapter composition baselines", "g2_baselines", "read", True, "c-gen",
     lambda m: f"task arithmetic {m['comparisons']['gfac_vs_sum']['y']}, averaging {m['comparisons']['gfac_vs_avg']['y']}, "
               f"composed concepts {m['comparisons']['gfac_vs_sum']['x']}"),
    ("G5a factor-level selection", "g5_factors", "read", True, "c-gen",
     lambda m: f"per-factor choice {m['H1']['x']} vs both-only {m['H1']['y']} (n.s.); object alone {m['alone_vs_zs']['obj']['x']}, "
               f"goal alone {m['alone_vs_zs']['goal']['x']}, policy {m['alone_vs_zs']['obj']['y']}"),
    ("G5b the switch at the grasp", "g5b_gating", "read", True, "c-gen",
     lambda m: f"gated {m['H1']['x']} vs both always on {m['H1']['y']} (sign p {m['H1']['sign_p']:.1g}); always on vs policy "
               f"{m['ungated_vs_zs']['y']}"),
    ("i5 the library in weight space", "i5_library", "read", False, "c-library",
     lambda m: f"{m['n_prototypes']} prototypes; instance to own type cos {m['instance_to_own_type']:.3f} vs {m['instance_to_other_types']:.3f}"),
    ("G3 a second policy (SmolVLA)", "g3_crossmodel", "read_smolvla", True, "c-gen", lambda m: _v_e3(m) if "bytes" in m else
     f"{m['hypotheses']['H1']['x']} vs {m['hypotheses']['H1']['y']} zero-shot; vs LoRA + acc. {_ci(m['hypotheses']['H2']['diff'])}"),
    ("G3 a third policy (pi0)", "g3_crossmodel", "read_pi0", True, "c-gen",
     lambda m: f"{m['hypotheses']['H1']['x']} vs {m['hypotheses']['H1']['y']} zero-shot; vs LoRA + acc. {_ci(m['hypotheses']['H2']['diff'])}"),
    ("G7 against DSRL", "g7_dsrl", "read", True, "c-gen",
     lambda m: f"vs DSRL on the frozen policy {m['hypotheses']['lib_vs_dsrlz_acc']['x']} vs {m['hypotheses']['lib_vs_dsrlz_acc']['y']}; "
               f"vs DSRL on the per-task LoRA {_ci(m['hypotheses']['lib_vs_dsrls_acc']['diff'])}"),
    ("G7b DSRL steering the library", "g7_dsrl", "read_library", True, "c-gen",
     lambda m: f"vs DSRL on the LoRA {_ci(m['H1']['diff'])}; vs the library {m['H2']['x']} vs {m['H2']['y']}: both fail"),
    ("G6 against STRAP", "g6_strap", "read", True, "c-gen",
     lambda m: f"{m['H1']['x']} vs STRAP + acc. {m['H1']['y']} of {m['H1']['n']}"),
    ("G4 Li's compositional OOD suites", "g4_ood", "read", True, "c-limits",
     lambda m: f"kept {m['H1']['x']} vs policy {m['H1']['y']} of {m['H1']['n']}: fails (concepts the policy already holds)"),
    ("G4b composing the policy's own fields", "g4_ood", "read_field", True, "c-limits",
     lambda m: f"kept {m['H1']['x']} vs policy {m['H1']['y']} of {m['H1']['n']} (sign p {m['H1']['sign_p']:.1g}): passes"),
    ("G4c the policy's own fields on pi0", "g4_ood", "read_field_pi0", True, "c-limits",
     lambda m: f"kept {m['H1']['x']} vs policy {m['H1']['y']} of {m['H1']['n']}: passes"),
    ("G4c the policy's own fields on SmolVLA", "g4_ood", "read_field_smolvla", True, "c-limits",
     lambda m: f"kept {m['H1']['x']} vs policy {m['H1']['y']} of {m['H1']['n']}: passes"),
    ("i6 likelihood detects a new concept", "i6_novelty", "read", False, "c-limits",
     lambda m: f"AUC {m['auc']:.2f}: held at most {m['held_max']:.2g}, new at least {m['new_min']:.2g}"),
    ("U1 one learner, both kinds of task", "u1_joint", "read", False, "c-stream",
     lambda m: f"50 arrivals: demos {m['a']['demos']}/{m['b']['demos']}; served {m['a']['H1']['x']}/{m['b']['H1']['x']} vs {m['a']['H1']['y']} of {m['a']['H1']['n']}"),
    ("G5c prototype rank 8 / 16", "g5c_rank", "read", True, "c-gen",
     lambda m: f"rank 16 vs LoRA + acc. {_ci(m['r16']['vs_lora_acc']['diff'])}; vs rank 4 {m['r16']['vs_rank4']['x']} vs {m['r16']['vs_rank4']['y']}"),
    ("i7 the anatomy of a composed step", "i7_anatomy", "read", False, "c-library",
     lambda m: f"named object first: object unit {m['metrics']['obj']['named_first']}/{m['metrics']['obj']['episodes']} vs policy {m['metrics']['zs']['named_first']}"),
    ("G8 task leaves", "g8_leaves", "read", True, "c-gen",
     lambda m: f"{m['H1']['x']} vs per-task LoRA + acc. {m['H1']['y']} of {m['H1']['n']} ({_ci(m['H1']['diff'])}): superior"),
    ("G9 rank-matched per-task LoRA", "g9_rank4lora", "read", True, "c-gen",
     lambda m: f"library {m['H1']['x']} vs rank-4 LoRA + acc. {m['H1']['y']} of {m['H1']['n']} ({_ci(m['H1']['diff'])}); {m['bytes']['library']/1e6:.1f} vs {m['bytes']['r4_total']/1e6:.1f} MB"),
    ("G4d Li's prompt switching on our harness (pi0)", "g4_ood", "read_li_pi0", True, "c-limits",
     lambda m: f"Li's sources {m['li_fixed_vs_zs']['x']} (fixed) / {m['li_grasp_vs_zs']['x']} (grasp); ours {m['ours_fixed_vs_zs']['x']} / {m['ours_grasp_vs_zs']['x']}; policy {m['li_fixed_vs_zs']['y']} of {m['li_fixed_vs_zs']['n']}"),
    ("G3 X-VLA (fourth policy)", "g3_crossmodel", "read_xvla", True, "c-gen",
     lambda m: f"{m['hypotheses']['H1']['x']} vs {m['hypotheses']['H1']['y']} zero-shot (n.s.); vs LoRA + acc. {_ci(m['hypotheses']['H2']['diff'])}"),
    ("G4c the policy's own fields on X-VLA", "g4_ood", "read_field_xvla", True, "c-limits",
     lambda m: f"kept {m['H1']['x']} vs policy {m['H1']['y']} of {m['H1']['n']}: passes"),
    ("G10 practice chooses the sources (pi0.5)", "g4_ood", "read_g10_pi05", True, "c-limits",
     lambda m: f"{m['sel_vs_li_grasp']['x']} vs Li's hand-chosen {m['sel_vs_li_grasp']['y']} of {m['sel_vs_li_grasp']['n']} ({_ci(m['sel_vs_li_grasp']['diff'])}): better"),
    ("G10 practice chooses the sources (pi0)", "g4_ood", "read_g10_pi0", True, "c-limits",
     lambda m: f"{m['sel_vs_li_grasp']['x']} vs Li's hand-chosen {m['sel_vs_li_grasp']['y']} of {m['sel_vs_li_grasp']['n']} ({_ci(m['sel_vs_li_grasp']['diff'])})"),
    ("G12 Li's text latent interpolation (pi0)", "g12_tli", "read_pi0", True, "c-limits",
     lambda m: f"TLI {m['tli_vs_zs']['x']} vs policy {m['tli_vs_zs']['y']}; practice-chosen pointer {m['sel_vs_tli']['x']} vs TLI ({_ci(m['sel_vs_tli']['diff'])}): trails"),
    ("G13 TLI with practice-chosen pairs (pi0)", "g13_tli_sources", "read_pi0", True, "c-limits",
     lambda m: f"TLI chosen {m['sel_vs_li']['x']} vs Li's TLI {m['sel_vs_li']['y']}; grasp switch {m['sel_vs_psel']['y']} ({_ci(m['psel_vs_li']['diff'])} vs Li's TLI): fails"),
    ("G13 TLI with practice-chosen pairs (pi05)", "g13_tli_sources", "read_pi05", True, "c-limits",
     lambda m: f"TLI chosen {m['sel_vs_li']['x']} vs Li's TLI {m['sel_vs_li']['y']}; grasp switch {m['sel_vs_psel']['y']} ({_ci(m['psel_vs_li']['diff'])} vs Li's TLI): fails"),
    ("G14 practice chooses pairs under TLI (pi0)", "g14_tli_practice", "read_pi0", True, "c-limits",
     lambda m: f"{m['sel_vs_tli_li']['x']} vs Li's TLI {m['sel_vs_tli_li']['y']} ({_ci(m['sel_vs_tli_li']['diff'])}); NI not established"),
    ("G14 practice chooses pairs under TLI (pi05)", "g14_tli_practice", "read_pi05", True, "c-limits",
     lambda m: f"{m['sel_vs_tli_li']['x']} vs Li's TLI {m['sel_vs_tli_li']['y']} ({_ci(m['sel_vs_tli_li']['diff'])}); NI not established"),
    ("E4 stream, first rule", "e4_stream", "read", True, "c-stream", _v_stream),
    ("W1 stream, the thesis's rule", "w1_stream", "read", True, "c-stream", _v_stream),
    ("W2 stream with re-checks", "w2_stream", "read", True, "c-stream", _v_stream),
    ("W0 Wang et al.'s query selection", "w0_select", "probe_sample", False, "c-limits",
     lambda m: f"right concept on top for {m['summary']['object']['top1']} of {m['summary']['object']['queries']} queries: fails"),
    ("T0 a new object without a demo", "t0_transfer", "screen", False, "c-limits",
     lambda m: f"borrowed prototypes help {m['transfers_where_policy_fails']} of {m['policy_below_theta']} failing tasks: fails"),
]


def reads_index():
    rows = []
    for label, exp, tag, pre, sec, verdict in READS:
        m = row(exp, tag)
        kind = '<span class="tag pre">pre-registered</span>' if pre else '<span class="tag">exploratory</span>'
        res = esc(verdict(m)) if m else '<span class="tag run">running</span>'
        rows.append([f'<a href="#{sec}">{esc(label)}</a>', kind, res, f"<code>{esc(exp)}</code>"])
    return ("<p>Every read of the current program, in the order it ran; each links to its section, and each verdict is computed from "
            "its ledger row. The earlier program's reads are in Part II and, with every number, in <code>docs/RESULTS.md</code>.</p>"
            + table(["read", "kind", "result", "ledger"], rows, "The reads at a glance.", num_from=9, cls="reads"))


# ── sections ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def overview():
    e3, w1, e5 = row("e3_hier", "read"), row("w1_stream", "read"), row("e5_replicate", "read")
    out = ("<p>A vision-language-action policy fine-tuned on forty tasks fails most new tasks in the same rooms. A new task arrives "
           "with its instruction, its scene and at most one demonstration. The frozen policy (pi0.5) takes it on by composing small "
           "prototypes learned from earlier tasks: an OBJECT prototype for the approach (how to reach and grasp this object, or the "
           "named instance of it) and a GOAL prototype for the carry (where to put it), each a rank-4 adapter consolidated over every "
           "demonstration of its concept. Practice decides between the composition and the policy alone, and the one demonstration "
           "is requested only when neither passes. This follows Wang et al.'s program (prototypes of a frozen generative model, "
           "selected and composed per out-of-distribution task) and adds what they lack: new primitives detected by practice, "
           "consolidated into concepts and reused.</p>")
    if e3:
        h1, h2, b = e3["hypotheses"]["H1"], e3["hypotheses"]["H2"], e3["bytes"]
        out += (f"<p>On start states no design decision saw, the kept library succeeds {h1['x']} of {h1['n']} times against "
                f"{h1['y']} for the policy alone, and {h2['x']} of {h2['n']} against {h2['y']} for a separate one-demonstration "
                f"adapter per task with the same acceptance test (difference {_ci(h2['diff'])}), storing {b['prototypes']} "
                f"prototypes ({b['library'] / 1e6:.1f} MB) in place of {b['loras']} adapters ({b['per_task_lora'] / 1e6:.1f} MB); "
                f"non-inferiority at the pre-registered margin of 0.05 is not established.")
    if w1:
        out += (f" Learned as a stream, it asks for {w1['a']['demos']} and {w1['b']['demos']} demonstrations over 30 arrivals in "
                f"two orders and damages no earlier arrival.")
    if e5:
        out += f" A pre-registered replication on new tasks: {_v_e5(e5)}."
    out += ("</p>" if e3 else "")
    g8, g10 = row("g8_leaves", "read"), row("g4_ood", "read_g10_pi05")
    if g8 and g10:
        out += (f"<p>The system is built in iterations, each answering a weakness the previous version showed. With task leaves at the "
                f"bottom of the hierarchy it beats the per-task adapter ({g8['H1']['x']} against {g8['H1']['y']} of {g8['H1']['n']}); "
                f"on a published compositional OOD benchmark with no demonstrations, composing the policy's own concepts with "
                f"practice-chosen sources reaches {g10['sel_vs_li_grasp']['x']} of {g10['sel_vs_li_grasp']['n']} against "
                f"{g10['sel_vs_li_grasp']['y']} for a prior method's hand-chosen sources.</p>")
        out += fig('<img src="media/iterations.png" alt="success per iteration against the per-task adapter on LIBERO-90, and on Li\'s OOD suites" style="width:100%">',
                   "<b>The iterations.</b> Left: LIBERO-90, the system kept by practice against the per-task LoRA with the same "
                   "acceptance test, at iterations 1-3 (30 tasks) and 8 (66 tasks, new start states). Right: Li's compositional OOD "
                   "suites on pi0.5, all on the same start states: the policy alone, its own fields with word-matched, hand-chosen "
                   "(Li) and practice-chosen sources.", wide=True)
    return out


def method():
    return ("<p>A step of an instruction is split at the policy's own grasp. Before the grasp the step's OBJECT prototype acts, keyed "
            "by the verb and the object (“put the black bowl”); when the instruction names an instance (“the middle "
            "black bowl”) its INSTANCE child acts instead. From the grasp to the release the GOAL prototype acts, keyed by the "
            "preposition and the target (“in the basket”). Each prototype is a rank-4 LoRA on the action expert, fitted by "
            "flow matching on the frames of its phase in every demonstration of its concept, each demonstration from the task's own "
            "scene. A missing factor is left to the policy. Ten practice episodes from stored start states decide between the "
            "composition and the policy (ties go to the policy). As a stream, an arrival requests its one demonstration only when "
            "neither reaches 8 of 10 in practice; the demonstration refits every concept on its path (instance, type, goal). Reads are "
            "pre-registered, scored on fresh start states from LIBERO's own sampler, paired by start state and noise, and tested with "
            "the exact sign test and a two-level bootstrap. The method is written out in <code>docs/METHOD.md</code>.</p>"
            + fig(diagrams.episode(), "<b>One composed step.</b> The object prototype is switched on from the start until the policy's own "
                  "gripper has stayed closed for three queries; the goal prototype then acts until the gripper opens. The frozen policy "
                  "generates every action throughout; a prototype only adds its low-rank update while it is on.", wide=True))


def lineage():
    q13, f1, d1, e1, e3, e5 = (row("q13_main", "read"), row("f1_fresh", "read"), row("d1_factored", "read"), row("e1_factored", "read"),
                               row("e3_hier", "read"), row("e5_replicate", "read"))
    rows = []
    if q13:
        h = q13["hypotheses"]["H1"]
        rows.append(("binding + folded token", [(0, 1, "the policy, attention bound; a token per task", "tok")],
                     f"{h['x']} vs {h['control']} of {h['n']} zero-shot"))
    if f1:
        h, l_ = f1["hypotheses"]["H1"], f1["descriptive"]["c2_vs_lora_acc"]
        rows.append(("step primitive", [(0, 1, "one LoRA per step key", "step")], f"{h['x']} vs {h['y']}; vs per-task LoRA {l_['x']} vs {l_['y']} of {l_['n']}"))
    for name, m, segs in (("goal primitive", d1, [(0, 0.5, "policy", "pol"), (0.5, 1, "goal", "goal")]),
                          ("object + goal", e1, [(0, 0.5, "object", "obj"), (0.5, 1, "goal", "goal")]),
                          ("+ instance children", e3, [(0, 0.5, "instance or type", "inst"), (0.5, 1, "goal", "goal")])):
        if m:
            h1, h2 = m["hypotheses"]["H1"], m["hypotheses"]["H2"]
            rows.append((name, segs, f"{h1['x']} vs {h1['y']}; vs per-task LoRA {h2['x']} vs {h2['y']} of {h2['n']}"))
    if e5:
        h = e5["hypotheses"]["H3"]
        rows.append(("+ 37 new tasks (pooled)", [(0, 0.5, "instance or type", "inst"), (0.5, 1, "goal", "goal")],
                     f"vs per-task LoRA {h['x']} vs {h['y']} of {h['n']:,}"))
    return ("<p>Each approach stored a different unit per task; the factorisation got finer as each measured failure pointed at the "
            "phase it lived in. Every row's numbers come from its pre-registered read (kept arms against the policy alone, then against "
            "the per-task LoRA with the same acceptance test, successes on fresh start states).</p>"
            + fig(diagrams.lineage(rows), "<b>From one unit per task to concepts per phase.</b> What acts in each phase of an episode, "
                  "approach by approach, and its result. Dashed: the policy alone.", wide=True))


def library_view():
    out = ("<p>The library is a small hierarchy of concepts. In weight space each prototype is its own direction; the dendrogram "
           "clusters the full library's prototypes by the cosine of their updates, labelled with the number of demonstrations each "
           "consolidates (exploratory).</p>")
    i5 = row("i5_library", "read")
    if i5:
        out += fig('<img src="media/i5_dendrogram.png" alt="dendrograms of the object and goal prototypes in weight space" style="width:100%">',
                   f"<b>The full library in weight space</b> ({i5['n_prototypes']} prototypes over the 67 single-step tasks). Average "
                   f"linkage on one minus the cosine of the updates; blue object types, violet instance children, brown goals. An "
                   f"instance child's cosine to its own type is {i5['instance_to_own_type']:.2f} against "
                   f"{i5['instance_to_other_types']:.2f} to other types.", wide=True)
    i7 = row("i7_anatomy", "read")
    if i7:
        m = i7["metrics"]
        out += ("<h3>What each unit does to the motion</h3><p>Recorded end-effector and object trajectories (ten tasks, five start "
                "states each, exploratory) show the division of labour. ")
        out += (f"The object unit decides which object is taken: the first grasp takes the named object in {m['obj']['named_first']} "
                f"of {m['obj']['episodes']} episodes with the object unit alone and {m['both']['named_first']} with both units, "
                f"against {m['zs']['named_first']} for the policy alone and {m['goal']['named_first']} with the goal unit alone. The "
                f"transport separates the arms little: the policy already carries, and fails in the approach. Each instance child "
                f"takes its own bowl; the type prototype mostly takes the back bowl whatever the instruction names.</p>")
        out += fig('<img src="media/i7_paths.png" alt="top-down end-effector paths per unit from one start state on four tasks" style="width:100%">',
                   "<b>Paths from one start state.</b> Top-down end-effector paths, approach solid until the policy's own grasp "
                   "(dot) and transport dashed; the named object starred, the goal square, other objects grey.", wide=True)
        out += fig('<img src="media/i7_centred.png" alt="distance to the named object over the approach and to the goal over the transport, per unit" style="width:100%">',
                   "<b>Concept-centred view.</b> Left: distance from the end effector to the named object, time normalised to the "
                   "grasp; right: the grasped object's distance to the goal. Median and quartiles over all tasks and episodes.", wide=True)
        out += fig('<img src="media/i7_instances.png" alt="which bowl each unit grasps first on the three black-bowl tasks" style="width:100%">',
                   "<b>The hierarchy at work.</b> On the three black-bowl tasks, the bowl grasped first: the instance child takes the "
                   "named bowl in every episode; the type prototype carries the bowl its demonstrations mostly showed.", wide=True)
    story = None
    try:
        import json as _j
        import os as _o
        from .base import R as _R
        story = _j.load(open(_o.path.join(_R, "stream_story.json")))
    except (OSError, ValueError):
        pass
    if story:
        out += ("<h3>A continual-learning walk-through</h3><p>Tasks arrive one at a time (the W1 stream, order a). Each lane is a concept; "
                "a square is a concept the library held from tasks outside the stream, a filled circle a concept a demonstration "
                "created, a ring a concept a later demonstration refit, a tick an arrival the concept serves, a star an arrival served "
                "with no demonstration. Demonstrations and stored prototypes grow far slower than one per arrival.</p>")
        out += fig('<img src="media/stream_lanes_a.png" alt="concept swimlanes over the arrivals of the stream" style="width:100%">',
                   "<b>The library as it learns.</b> Concepts (rows) over arrivals (columns) in the W1 stream, order a; below, "
                   "demonstrations requested and prototypes stored.", wide=True)
        rows = []
        for s_ in story["a"]["story"][:12]:
            dec = ("the policy holds it" if not s_["demo"] and s_["kept"] == "policy" else
                   "composed, no demonstration" if not s_["demo"] else
                   f"demonstration; refits {', '.join(s_['refit']) or '(none)'}")
            rows.append([s_["arrival"], esc(s_["text"]), s_["policy"], "&ndash;" if s_["composition"] is None else s_["composition"],
                         esc(dec), "&ndash;" if s_.get("after") is None else s_["after"], s_["kept"]])
        out += table(["#", "arrival", "policy", "composed", "decision", "after refit", "kept"], rows,
                     "The first twelve arrivals of the W1 stream (order a): practice successes of 10 for the policy alone and for the "
                     "composition from the library as it stood, the decision, and the practice after the refit.", num_from=2)
    return out


def factored():
    out = ""
    d0, d1, dg = row("d0_goal", "screen"), row("d1_factored", "read"), row("d1_factored", "diagnosis")
    if d0 and d1 and dg:
        t0, h1, h2, g = d0["totals"], d1["hypotheses"]["H1"], d1["hypotheses"]["H2"], dg["totals"]
        kept = {k: sum(v == k for v in d1["kept"].values()) for k in ("zs", "goal", "goal_bind")}
        out += (f"<p>A step primitive fuses two factors: which object the arm reaches for, and where it carries it. Plan D splits each "
                f"step at the policy's own grasp and gives the carry a GOAL primitive, keyed by the goal with the object left out. In a "
                f"screen on {t0['tasks']} held-out tasks it succeeds {t0['goal']} of {10 * t0['tasks']} times against {t0['zs']} "
                f"zero-shot (exploratory). Pre-registered on the stream's {h1['tasks']} tasks (m3_prereg tag d1_factored), the "
                f"acceptance test kept the policy on {kept['zs']} tasks, the goal primitive on {kept['goal']} and the goal primitive "
                f"with binding on {kept['goal_bind']}; the kept arms succeed {h1['x']} of {h1['n']} against {h1['y']} zero-shot, "
                f"but against the per-task one-demo LoRA with acceptance only {h2['x']} of {h2['n']} against {h2['y']} (difference "
                f"{_ci(h2['diff'])}). The first grasp shows why: on the {len(dg['tasks'])} tasks only the LoRA solves, the policy "
                f"grasps the wrong object first in {g['zs']['wrong']} of {g['zs']['n']} episodes (the milk for the orange juice), and "
                f"binding turns the arm toward the named object but grasps nothing in {g['goal_bind']['none']}. The approach needs its "
                f"own prototype.</p>")
    e0, mp, sn = row("e0_object", "screen"), row("l90_scenes", "map"), row("d1_factored", "sensitivity")
    if e0 and dg:
        t, gap = e0["totals"], list(dg["tasks"])
        out += (f"<p>Plan E gives the approach an OBJECT prototype, fitted only on frames before the first grasp and consolidated over "
                f"every demonstration of that object; the stream's {e0['tasks']['all']} tasks need {e0['library']['object_keys']} "
                f"object and {e0['library']['goal_keys']} goal prototypes. In a screen (exploratory) the composed library succeeds "
                f"{t['all']} of {10 * e0['tasks']['all']} times against {t['zs']} zero-shot, the per-task LoRA {t['lora']} of "
                f"{10 * e0['tasks']['lora']}; on the {len(gap)} wrong-object tasks {sum(e0['per_task'][x]['all'] for x in gap)} of "
                f"{10 * len(gap)} (LoRA {sum(e0['per_task'][x]['lora'] for x in gap)}).</p>")
    if mp and sn:
        out += (f"<p>That screen exposed a data defect. The IPEC port of LIBERO-90 merges tasks by instruction ({mp['shared_texts']} "
                f"instructions shared by {mp['shared_tasks']} tasks in different scenes), and every earlier read had taken an "
                f"instruction's first episode as the task's demonstration: {len(mp['wrong_scene_tasks'])} tasks had another scene's "
                f"demonstration, the per-task LoRA's included. Each task now takes its demonstration from its own scene's block, matched "
                f"to its rendered first frames. Without the {len(sn['excluded'])} affected stream tasks D1 keeps its verdicts.</p>")
    e1 = row("e1_factored", "read")
    if e1:
        h1, h2 = e1["hypotheses"]["H1"], e1["hypotheses"]["H2"]
        out += (f"<p>Pre-registered on the corrected data (m3_prereg tag e1_factored), the acceptance test kept the library on "
                f"{sum(v == 'fac' for v in e1['kept'].values())} of {h1['tasks']} tasks; the kept arms succeed {h1['x']} of {h1['n']} "
                f"against {h1['y']} zero-shot and {h2['x']} of {h2['n']} against {h2['y']} for the per-task LoRA with acceptance "
                f"(difference {_ci(h2['diff'])}). Most of the remaining gap sits on four tasks that name an instance of the black "
                f"bowl.</p>")
    out += fig(video("fac_53_0"), "<b>The approach is where the policy fails.</b> Pick up the orange juice and put it in the basket, on a "
               "fresh start state: the policy alone takes the milk; the composed object and goal prototypes take the orange juice and "
               "place it; the per-task LoRA, fitted on this task's demonstration, succeeds too.", wide=True)
    out += fig(video("fac_71_0"), "<b>Put the red mug on the plate.</b> The policy alone takes the porcelain mug; the red mug's object "
               "prototype takes the named mug.", wide=True)
    out += fig(video("fac_41_1"), "<b>Put the frying pan on top of the cabinet.</b> The policy alone reaches for the white bowl; the "
               "frying pan's object prototype, consolidated over six demonstrations of the pan, and the cabinet's goal prototype "
               "complete the task.", wide=True)
    return out


def hierarchy():
    e2, e3, i4 = row("e2_instance", "screen"), row("e3_hier", "read"), row("i4_factors", "read")
    out = ""
    if e2 and e3:
        h1, h2, de, b = e3["hypotheses"]["H1"], e3["hypotheses"]["H2"], e3["descriptive"], e3["bytes"]
        out += (f"<p>A prototype pooled over 14 black bowls averages their approaches and does not pick the named one. Cobweb's "
                f"hierarchy answers it: a type concept has instance children, and a step that names an instance uses the child, "
                f"consolidated over the demonstrations with that object and that qualifier. In a screen the children succeed "
                f"{e2['totals']['scratch']} of 40 times on the four instance tasks against {e2['totals']['type']} for the type "
                f"prototype (exploratory). The confirmatory read (m3_prereg tag e3_hier) tested the hierarchical library, "
                f"{b['prototypes']} prototypes for {h1['tasks']} tasks, on a second set of start states no design decision had seen: "
                f"the kept arms succeed {h1['x']} of {h1['n']} times against {h1['y']} zero-shot (sign p {h1['sign_p']:.2g}) and "
                f"{h2['x']} of {h2['n']} against {h2['y']} for the per-task LoRA with acceptance (difference {_ci(h2['diff'])}). "
                f"The interval holds zero, but its lower bound misses the margin of 0.05, as the registration's power projection "
                f"expected. Without acceptance the library and the LoRA are level ({de['hfac_vs_lora']['x']} against "
                f"{de['hfac_vs_lora']['y']}); the instance children succeed {de['instance_child_vs_type']['x']} of "
                f"{de['instance_child_vs_type']['n']} against {de['instance_child_vs_type']['y']} for the type prototype. The "
                f"library stores {b['library'] / 1e6:.1f} MB where the per-task LoRAs store {b['per_task_lora'] / 1e6:.1f} MB.</p>")
    out += fig(diagrams.hierarchy(), "<b>The library as concepts.</b> Object types with their instance children (approach) and goal "
               "concepts (transport), each a prototype consolidated over the demonstrations shown; a task names one object node and one "
               "goal node and composes them.", wide=True)
    out += fig(video("inst_14_0"), "<b>The middle black bowl.</b> The type prototype, pooled over every black bowl, reaches for the wrong "
               "one; the instance child takes the middle bowl; on this start state the per-task LoRA fails.", wide=True)
    out += fig(video("inst_13_0"), "<b>The black bowl at the front.</b> The policy and the type prototype take the bowl at the back; the "
               "instance child takes the front one.", wide=True)
    if i4:
        sm, st = i4["similarity"], i4["stability"]
        out += (f"<p>In weight space (exploratory) each object prototype points its own way: the mean cosine between object "
                f"prototypes is {sm['object_object']:.3f}, between object and goal prototypes {sm['object_goal']:.3f}. Both factors "
                f"act in the same layers and modules, so they differ in direction, not in location. A prototype summarises its "
                f"demonstrations: refit without one task's demonstration it keeps a cosine of {st['same_object']:.2f} with the full "
                f"one, against {st['other_objects']:.3f} with other objects' prototypes.</p>")
    return out


def replication():
    e5 = row("e5_replicate", "read")
    if not e5:
        return pending("E5: the library on 37 new LIBERO-90 tasks",
                       "Pre-registered (m3_prereg tag e5_replicate): the hierarchical library, with prototypes for new objects and "
                       "goals (books, moka pots, drawers, the stove, the caddy), against the per-task LoRA with acceptance on the 37 "
                       "single-step pick-and-place tasks outside the stream; non-inferiority on them and pooled with E3.")
    h = e5["hypotheses"]
    return (f"<p>Pre-registered (m3_prereg tag e5_replicate) on the 37 single-step pick-and-place LIBERO-90 tasks outside the "
            f"stream, none used by any design decision: the kept arms succeed {h['H1']['x']} of {h['H1']['n']} times against "
            f"{h['H1']['y']} zero-shot (sign p {h['H1']['sign_p']:.2g}); against the per-task LoRA with acceptance {h['H2']['x']} "
            f"against {h['H2']['y']} (difference {_ci(h['H2']['diff'])}; non-inferiority "
            f"{'holds' if h['H2']['noninferior'] else 'is not established'}). Pooled with E3's tasks ({h['H3']['tasks']} tasks): "
            f"difference {_ci(h['H3']['diff'])}; non-inferiority {'holds' if h['H3']['noninferior'] else 'is not established'}.</p>")


def generalise():
    g0, g2 = row("g0_recombine", "read"), row("g2_baselines", "read")
    if not g0:
        return pending("G0: held-out recombination", "Pre-registered (m3_prereg tag g0_recombine).")
    h, d, x = g0["hypotheses"], g0["descriptive"], g0["exploratory_by_policy"]
    out = (f"<p>The library above holds each task's own demonstration in its concepts. The decisive test of composition removes it "
           f"(m3_prereg tag g0_recombine): the 67 single-step tasks are split into 5 folds, and every concept a fold needs is refit "
           f"with no demonstration of that fold's tasks, so each of the {h['H1']['tasks']} eligible tasks can be solved only by "
           f"recombining an object and a goal learned from other tasks. Alone, the recombined concepts succeed {h['H1']['x']} of "
           f"{h['H1']['n']} times against {h['H1']['y']} for the policy (sign p {h['H1']['sign_p']:.2g}): the pre-registered test "
           f"fails. With practice choosing between them and the policy, {h['H1b']['x']} (sign p {h['H1b']['sign_p']:.2g}). The "
           f"factorisation matters: a monolithic adapter trained on the same demonstrations succeeds {h['H2']['y']} times, below the "
           f"policy (sign p {h['H2']['sign_p']:.2g} for the recombined concepts against it).</p>")
    rows = [[f"the policy {'fails' if k == 'policy_fails' else 'succeeds'} ({v['tasks']} tasks)",
             v["zs"], v["gfac"], v["mono"], f"{v['gfac_better']} / {v['gfac_worse']}"] for k, v in
            sorted(x.items(), key=lambda kv: kv[0] != "policy_fails")]
    out += table(["held-out tasks where", "policy alone", "recombined concepts", "monolithic adapter", "better / worse tasks"], rows,
                 "Recombination of concepts learned only from other tasks, split by whether the policy alone succeeds on at least half "
                 "of a task's 20 test episodes (exploratory, not pre-registered). Successes on fresh start states.", num_from=1)
    out += ("<p>Recombination is a large gain exactly where the policy lacks the skill and interferes where it has it; the two cancel "
            "in the pooled test, and practice tells them apart.</p>")
    if g2:
        c = g2["comparisons"]
        out += (f"<p>Composing adapters the usual way does not do this (m3_prereg tag g2_baselines). On the same held-out tasks, the "
                f"per-task adapters of an object donor and a goal donor, both always on: summed (task arithmetic) they succeed "
                f"{c['gfac_vs_sum']['y']} times, averaged {c['gfac_vs_avg']['y']}, against {c['gfac_vs_sum']['x']} for the recombined "
                f"concepts; with practice choosing, {c['kept_gfac_vs_kept_best_baseline']['x']} against "
                f"{c['kept_gfac_vs_kept_best_baseline']['y']} for the better baseline. Averaging is level with the recombined concepts "
                f"without practice (sign p {c['gfac_vs_avg']['sign_p']:.2g}).</p>")
    g3 = row("g3_crossmodel", "read_smolvla")
    if g3:
        h1, h2 = g3["hypotheses"]["H1"], g3["hypotheses"]["H2"]
        out += (f"<p>On a second policy (m3_prereg tag g3_crossmodel): the same concepts refit on SmolVLA, at the same tasks, start states "
                f"and seed. SmolVLA alone succeeds {h1['y']} of {h1['n']} times; the kept library {h1['x']} (sign p {h1['sign_p']:.2g}). "
                f"Against SmolVLA's own per-task LoRA with acceptance, {h2['x']} against {h2['y']} of {h2['n']} (difference "
                f"{_ci(h2['diff'])}): the gain over the frozen policy carries over, the parity with a per-task adapter does not.</p>")
    g3p = row("g3_crossmodel", "read_pi0")
    if g3p:
        h1, h2 = g3p["hypotheses"]["H1"], g3p["hypotheses"]["H2"]
        out += (f"<p>On a third flow-matching policy, pi0 (m3_prereg tag g3_crossmodel_pi0), pi0 alone succeeds {h1['y']} of {h1['n']} "
                f"times and the kept library {h1['x']} (sign p {h1['sign_p']:.2g}); against pi0's per-task LoRA with acceptance, "
                f"{h2['x']} against {h2['y']} of {h2['n']} (difference {_ci(h2['diff'])}). The scope is flow-matching and diffusion "
                f"policies: a prototype is a low-rank change to the policy's velocity field, switched on for one phase.</p>")
    g7, g6 = row("g7_dsrl", "read"), row("g6_strap", "read")
    if g7:
        z, l = g7["hypotheses"]["lib_vs_dsrlz_acc"], g7["hypotheses"]["lib_vs_dsrls_acc"]
        out += (f"<p>Against DSRL (m3_prereg tag g7_dsrl), which trains a noise-space policy from 100 practice episodes per task: "
                f"steering the frozen policy without a demonstration it reaches {z['y']} of {z['n']} with acceptance, against "
                f"{z['x']} for the library (difference {_ci(z['diff'])}); steering each task's one-demo LoRA it is the strongest arm "
                f"run here, {l['y']} of {l['n']} against {l['x']} (difference {_ci(l['diff'])}), at ten times the practice and a "
                f"per-task adapter and actor.</p>")
    if g6:
        h = g6["H1"]
        out += (f"<p>Against STRAP (m3_prereg tag g6_strap), retrieving sub-trajectories from the library's own data and pi0.5's "
                f"training data: {h['y']} of {h['n']} with acceptance, against {h['x']} for the library (difference "
                f"{_ci(h['diff'])}); STRAP alone falls below the policy.</p>")
    g8 = row("g8_leaves", "read")
    if g8:
        h, b = g8["H1"], g8["bytes"]
        out += (f"<p><b>Task leaves.</b> The hierarchy gains a lowest level: a task the shared concepts cannot serve keeps its own "
                f"one-demonstration adapter as a leaf, kept only where practice prefers it (m3_prereg tag g8_leaves, new start "
                f"states). Over {g8['tasks']} tasks it succeeds {h['x']} of {h['n']} times against {h['y']} for the per-task LoRA with "
                f"acceptance (difference {_ci(h['diff'])}): superior, with leaves on {g8['kept']['lora']} tasks where the baseline "
                f"keeps {b['per_task_loras']} adapters, {b['hierarchy_mb']:.1f} MB against {b['lora_acc_mb']:.1f} MB.</p>")
    g5 = row("g5_factors", "read")
    if g5:
        a = g5["alone_vs_zs"]
        out += (f"<p>Where does the interference come from? Letting practice choose per factor (m3_prereg tag g5_factors) gives "
                f"{g5['H1']['x']} against {g5['H1']['y']} for choosing only between the policy and both concepts (sign p "
                f"{g5['H1']['sign_p']:.2g}, not significant). Alone, the object concept succeeds {a['obj']['x']} times, below the policy's "
                f"{a['obj']['y']}, and the goal concept {a['goal']['x']}: the object factor, which changes which object the arm reaches "
                f"for, is the one that interferes when the policy already knows the task.</p>")
    g5b = row("g5b_gating", "read")
    if g5b:
        h, u = g5b["H1"], g5b["ungated_vs_zs"]
        out += (f"<p>Does the switch at the grasp matter (m3_prereg tag g5b_gating)? With both prototypes on for the whole episode the "
                f"library succeeds {h['y']} of {h['n']} times, level with the policy's {u['y']} (sign p {u['sign_p']:.2g}); switched at "
                f"the policy's own grasp it succeeds {h['x']} (discordant pairs {h['discordant'][0]} against {h['discordant'][1]}). "
                f"The same two prototypes, summed without the phase, give nothing: the gain is carried by the gating.</p>")
    return out


def streams():
    e4, w1, w2 = row("e4_stream", "read"), row("w1_stream", "read"), row("w2_stream", "read")
    if not e4:
        return ""
    flow = fig(diagrams.stream(), "<b>The learner on one arrival.</b> A demonstration is requested only when neither the policy nor the "
               "composition reaches 8 of 10 practice episodes (W1; E4 asked only whether the composition beat the policy). The "
               "demonstration is written into every concept on its path, and later arrivals read the updated library.", wide=True)
    out = flow + ("<p>Learned as a stream, the library starts from the demonstrations of the LIBERO-90 tasks outside the stream, and the "
           "30 tasks arrive one at a time in two orders, each composed from the current library; a demonstration, when requested, "
           "updates every concept on its path, as Cobweb updates every node an instance passes. Each task is served at the end by the "
           "final library where it kept the library (pre-registered, m3_prereg tags e4_stream, w1_stream, w2_stream).</p>")
    rows = []
    for name, r in (("E4", e4), ("W1", w1), ("W2", w2)):
        for o in ("a", "b") if r else ():
            x, dm = r[o], r[o].get("damage") or {}
            rows.append([name, o, x["demos"], len(x["kept_without_demo"]), f"{x['H1']['x']} / {x['H1']['y']}",
                         f"{x['vs_lora_acc']['x']} / {x['vs_lora_acc']['y']}", f"{dm['x']} / {dm['y']}" if dm else "&ndash;"])
    out += table(["rule", "order", "demos", "no demo", "vs zero-shot", "vs LoRA + acc.", "final / arrival"], rows,
                 "The library learned as a stream, per rule and order: demonstrations requested (of the 29 that exist); arrivals kept "
                 "without one; served successes against zero-shot (of 600) and against the per-task LoRA with acceptance (of 580); on "
                 "the arrivals whose composition changed after they arrived, the final composition against the arrival-time one. "
                 "Fresh start states.", num_from=2)
    a_, b_ = e4["a"], e4["b"]
    out += (f"<p>E4 requests a demonstration whenever the composition does not beat the policy in practice. Both orders beat the policy "
            f"by far ({a_['H1']['x']} and {b_['H1']['x']} of {a_['H1']['n']} against {a_['H1']['y']}), but in order b one arrival kept "
            f"at 9 practice successes against the policy's 8 falls to 0 of 20 under the final library, and the losses sit on arrivals "
            f"kept on a thin practice margin (2 against 1, 3 against 0).")
    if w1:
        out += (f" W1 uses the thesis's rule: no demonstration only when the better of the policy and the composition reaches 8 of 10. "
                f"It asks for fewer demonstrations ({w1['a']['demos']} and {w1['b']['demos']} against {a_['demos']} and "
                f"{b_['demos']}) and damages no earlier arrival in either order; in order b it comes within "
                f"{_ci(w1['b']['vs_lora_acc']['diff'])} of the per-task LoRA with acceptance.")
    if w2:
        n_re = sum(len(a.get("rechecks", [])) for o in ("a", "b") for a in w2[o]["arrivals_log"])
        out += (f" W2 adds a re-check after every refit ({n_re} re-checks): it does not improve on W1, because ten practice episodes "
                f"are too noisy a guard to tell a damaged unit from chance.")
    out += "</p>"
    cap = ("<b>Demonstrations requested as the stream arrives.</b> E4 requests one whenever the composition does not beat the policy"
           + ("; W1 only when neither the policy nor the composition reaches 8 of 10 practice episodes" if w1 else "")
           + ". Hollow markers: arrivals kept without a demonstration; dotted: one demonstration per arrival.")
    u1 = row("u1_joint", "read")
    if u1:
        a_, b_ = u1["a"], u1["b"]
        out += (f"<p><b>One learner, both kinds of task.</b> A joint stream of {a_['arrivals']} arrivals adds Li's 20 compositional "
                f"OOD tasks, which have no demonstrations, to W1's 30 LIBERO-90 tasks; held concepts become pointers to the "
                f"policy's own fields, new ones prototypes, and practice keeps what beats the policy (an exact replay: pointers carry "
                f"no weights, so they change no request). It requests {a_['demos']} and {b_['demos']} demonstrations, all for "
                f"LIBERO-90 concepts, and serves {a_['H1']['x']} and {b_['H1']['x']} of {a_['H1']['n']} against {a_['H1']['y']} "
                f"for the policy alone.</p>")
    return out + fig('<img src="media/streams_demos.png" alt="cumulative demonstrations requested against arrivals" style="width:100%">', cap)


def limits_of_method():
    w0, t0 = row("w0_select", "probe_sample"), row("t0_transfer", "screen")
    out = ""
    if w0:
        out += (f"<p><b>Selection by a query's likelihood.</b> Wang et al. select a model's prototypes by how well they cover one query. "
                f"Given a held-out demonstration as the query, neither the policy's flow-matching loss nor the error of each "
                f"prototype's own samples picks the instruction's concept above chance (object concept on top for "
                f"{w0['summary']['object']['top1']} of {w0['summary']['object']['queries']} queries; exploratory): one trajectory's "
                f"motion varies more between placements than between concepts. The learner selects by the instruction's words and "
                f"keeps by practice.</p>")
    if t0:
        out += (f"<p><b>A new object without a demonstration.</b> Sorting a new object under the known objects its name is closest to "
                f"(the policy's own language features) and letting practice choose helps on {t0['transfers_where_policy_fails']} of the "
                f"{t0['policy_below_theta']} tasks the policy fails (exploratory): object prototypes do not transfer between objects, "
                f"and a new object needs its one example. Reuse comes from objects and goals that recur across tasks.</p>")
    g4, dg = row("g4_ood", "read"), row("g4_ood", "diagnosis")
    if g4 and dg:
        h = g4["H1"]
        out += (f"<p><b>Concepts the policy already holds.</b> On Li's compositional OOD suites (20 tasks, each pairing one training "
                f"task's object with another's target; m3_prereg tag g4_ood), every concept was fitted on pi0.5's own training "
                f"demonstrations. The kept arms succeed {h['x']} of {h['n']} times against {h['y']} for the policy: the test fails. "
                f"The prototypes learned nothing: their flow loss starts at {dg['g4']['first']:.2g} and ends at {dg['g4']['last']:.2g}, "
                f"where E3's fall from {dg['e3']['first']:.2g} to {dg['e3']['last']:.2g}. The library supplies a concept the policy "
                f"lacks; it does not undo the policy's binding of names to the layouts it was trained on.</p>")
    g4b = row("g4_ood", "read_field")
    if g4b:
        h, pt = g4b["H1"], g4b["per_task"]
        out += (f"<p><b>The remedy: compose the policy's own fields.</b> A concept the policy already holds is its own field under the "
                f"instruction of a training task that taught it; the step runs under the object concept's instruction until the "
                f"grasp and the goal concept's after it, and practice chooses (m3_prereg tag g4b_field, a new set of start states). "
                f"The kept arms succeed {h['x']} of {h['n']} times against {h['y']} (discordant pairs {h['discordant'][0]} to "
                f"{h['discordant'][1]}; task-level difference {_ci(h['diff'])}), in both suites; the orange juice onto the stove "
                f"{pt['goal:1']['pfield']} against {pt['goal:1']['zs']}, the cream cheese onto the plate {pt['goal:7']['pfield']} "
                f"against {pt['goal:7']['zs']}, the bowl at table center onto the cabinet {pt['spatial:8']['pfield']} against "
                f"{pt['spatial:8']['zs']}. Unfiltered the field composition gives {g4b['pfield_vs_zs']['x']}: it helps where the "
                f"source concepts are scene-matched, so practice must choose. It is score composition in the sense of our ICLR-2027 "
                f"study, applied to a policy's phases.</p>")
    cp, cs = row("g4_ood", "read_field_pi0"), row("g4_ood", "read_field_smolvla")
    if cp and cs:
        out += (f"<p>The same composition, unchanged, on the two other flow-matching policies (m3_prereg tag g4c_field): pi0 "
                f"{cp['H1']['x']} against {cp['H1']['y']} of {cp['H1']['n']} (discordant pairs {cp['H1']['discordant'][0]} to "
                f"{cp['H1']['discordant'][1]}; Goal-OOD {cp['goal']['x']} against {cp['goal']['y']}), SmolVLA {cs['H1']['x']} "
                f"against {cs['H1']['y']}. It carries across the three policies.</p>")
    d0 = row("g4_ood", "read_li_pi0")
    if d0:
        out += (f"<p><b>Against Li's prompt switching.</b> The pointer operator is Li's, and it reproduces on our harness (m3_prereg tag "
                f"g4d_li): with Li's hand-chosen source pairs pi0 succeeds {d0['li_fixed_vs_zs']['x']} of {d0['li_fixed_vs_zs']['n']} "
                f"times at their fixed switching step and {d0['li_grasp_vs_zs']['x']} at its own grasp, against "
                f"{d0['ours_fixed_vs_zs']['x']} and {d0['ours_grasp_vs_zs']['x']} with our automatically chosen sources. The gap is the "
                f"choice of sources, not the trigger: Li's pairs pick training tasks whose motion reaches where the new object lies, "
                f"ours match the instruction's words.</p>")
    g10a, g10b = row("g4_ood", "read_g10_pi05"), row("g4_ood", "read_g10_pi0")
    if g10a and g10b:
        out += (f"<p><b>Practice chooses the sources.</b> With no hand labels, practice picks each held concept's source among 16 "
                f"training instructions (m3_prereg tag g10_sources). On pi0.5 the chosen pairs succeed {g10a['sel_vs_li_grasp']['x']} of "
                f"{g10a['sel_vs_li_grasp']['n']} times against {g10a['sel_vs_li_grasp']['y']} for Li's hand-chosen pairs (difference "
                f"{_ci(g10a['sel_vs_li_grasp']['diff'])}); on pi0 {g10b['sel_vs_li_grasp']['x']} against {g10b['sel_vs_li_grasp']['y']} "
                f"(difference {_ci(g10b['sel_vs_li_grasp']['diff'])}), level with Li's fixed-step operator.</p>")
    g12 = row("g12_tli", "read_pi0")
    if g12:
        out += (f"<p><b>Li's text latent interpolation is the stronger operator on pi0.</b> Ported to the PyTorch pi0 with its "
                f"latents identified by Li's procedure (m3_prereg tag g12_tli), TLI succeeds {g12['tli_vs_zs']['x']} of "
                f"{g12['tli_vs_zs']['n']} times against {g12['tli_vs_zs']['y']} for the policy alone, and the practice-chosen "
                f"pairs switched at the grasp trail it ({g12['sel_vs_tli']['x']}; difference {_ci(g12['sel_vs_tli']['diff'])}). "
                f"TLI uses hand-chosen pairs; the next iteration gives it practice-chosen ones.</p>")
    g13a, g13b = row("g13_tli_sources", "read_pi0"), row("g13_tli_sources", "read_pi05")
    if g13a and g13b:
        out += (f"<p><b>Which operator wins depends on the policy.</b> On new start states (set d; m3_prereg tag g13_tli_sources) "
                f"pi0 succeeds {g13a['sel_vs_li']['y']} of 400 times with Li's TLI and {g13a['sel_vs_psel']['y']} with the "
                f"practice-chosen grasp switch; pi0.5 {g13b['sel_vs_li']['y']} and {g13b['sel_vs_psel']['y']}. TLI fed the pairs "
                f"chosen for the grasp switch falls behind both ({g13a['sel_vs_li']['x']} and {g13b['sel_vs_li']['x']}): pairs "
                f"chosen under one operator do not carry over to the other. Practice now chooses the pairs under TLI (G14).</p>")
    g14a, g14b = row("g14_tli_practice", "read_pi0"), row("g14_tli_practice", "read_pi05")
    if g14a and g14b:
        out += (f"<p><b>Practice chooses the pairs under TLI.</b> With TLI as the operator during practice (m3_prereg tag "
                f"g14_tli_practice), pi0 succeeds {g14a['sel_vs_tli_li']['x']} of 400 times against {g14a['sel_vs_tli_li']['y']} "
                f"for Li's hand-chosen pairs ({_ci(g14a['sel_vs_tli_li']['diff'])}) and pi0.5 {g14b['sel_vs_tli_li']['x']} "
                f"against {g14b['sel_vs_tli_li']['y']} ({_ci(g14b['sel_vs_tli_li']['diff'])}): most of the gap closes without "
                f"hand labels; non-inferiority is not established.</p>")
    i6 = row("i6_novelty", "read")
    if i6:
        out += (f"<p><b>Which kind of unit a concept gets.</b> The frozen policy's own flow-matching loss on a concept's frames "
                f"separates the {i6['n_held']} concepts of its training tasks from the {i6['n_new']} LIBERO-90 concepts it never saw "
                f"(AUC {i6['auc']:.2f}; held at most {i6['held_max']:.2g}, new at least {i6['new_min']:.2g}; exploratory). This is the "
                f"detector of new primitives Wang et al. lack: a held concept is a pointer to the policy's own field, a new one is "
                f"fitted as a prototype. The same words are new in new scenes, so held means held under its instruction and scene. "
                f"Caveat: held concepts are scored on demonstrations the policy was trained on.</p>")
    i6b = row("i6_novelty", "field_on_new")
    if i6b:
        h, g = i6b["pfield_vs_zs"], i6b["hfac_vs_pfield"]
        out += (f"<p>The routing is right in both directions. On {i6b['tasks']} LIBERO-90 tasks whose concepts share words with held "
                f"ones, the field composition equals the policy ({h['x']} against {h['y']} of {h['n']}; exploratory), while the "
                f"prototypes the router assigns to these new concepts reach {g['x']}.</p>")
    return out


def limitations():
    return ("<ul><li>Non-inferiority to a separate adapter per task is not established: on 30 tasks the tasks differ too much among "
            "themselves; the replication adds 37.</li>"
            "<li>A new object needs its one demonstration; the policy's language features do not find a known object to stand in.</li>"
            "<li>Ten practice episodes are a noisy guard: re-checking after every refit does not improve on the stream without it.</li>"
            "<li>Single-step pick-and-place only; two-step tasks and articulated objects (drawers, knobs) are not composed.</li>"
            "<li>Simulation only (LIBERO-90), one policy (pi0.5).</li></ul>")


def verification():
    tags = sorted({r["tag"] for r in L.load("m3_prereg")})
    return (f"<p>Every read quoted as a test was pre-registered in <code>ledger/m3_prereg</code> before its first episode "
            f"({len(tags)} registrations: {esc(', '.join(tags))}); reads marked exploratory were not. Episodes are paired by task, "
            f"start state and noise; intervals come from a two-level bootstrap over tasks and episodes, tests from the exact sign test "
            f"on discordant pairs, with Holm correction within a read. The full log of reads, each with its ledger tags, is "
            f"<code>docs/RESULTS.md</code>; <code>experiments/status.py</code> shows what is running.</p>")


SECTIONS = [("c-overview", "", "In brief", overview, False), ("c-reads", "", "The reads at a glance", reads_index, False),
            ("c-method", "1", "The method", method, False), ("c-lineage", "", "How the unit was factored", lineage, False),
            ("c-library", "", "The library, and how it grows", library_view, False), ("c-factored", "2", "Factored steps: object and goal prototypes", factored, False),
            ("c-hier", "3", "Instance children: a hierarchy of prototypes", hierarchy, False),
            ("c-repl", "4", "Replication on new tasks", replication, False),
            ("c-gen", "5", "Does composition generalise?", generalise, False),
            ("c-stream", "6", "The library learned as a stream", streams, False),
            ("c-limits", "7", "What does not work", limits_of_method, False), ("c-limitations", "8", "Limitations", limitations, False),
            ("c-verify", "", "Verification", verification, False)]
