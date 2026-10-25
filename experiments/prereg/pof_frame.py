"""PRE-REGISTRATION of probe OF (docs/PLAN_OF.md; code experiments/of_frame.py), written 2026-10-25 AFTER the index, the poses and
the frame selection (poses only) and a 2-frame GPU smoke per model (results/of/scratch, never scored), and BEFORE any D1, D2 or D3
statistic is computed (asserted below: no results/of/obs_* or dev_* files, no read.json, no of_read row).

D1  object-frame transfer of approach behaviour, pre-grasp demo frames of each model's training tasks, object types in >= 2 scenes.
    d = endpoint - hand (xy), predicted leave-scene-out by d = lam (c - h) + R b in frame (c, R): world (0, I), object (oracle pos,
    yaw), control (another non-named movable object); T = mean over types of P2's corr2 (x / y Pearson averaged) pooled per type,
    each frame at its best level of {0.1, 0.3, 0.55}; task-cluster bootstrap B 2000.
    H1 l90c part (obj:named), H2 l90c instr, H3 libero part, H4 libero instr: T_obj >= 0.5, T_obj - T_world >= 0.3, T_obj > T_ctrl and
    p = max(P*(T_obj - T_world <= 0), P*(T_obj - T_ctrl <= 0)) passes Holm over H1-H4 at 0.05.
    route 1 alive = H1 or H3; instruction-level only = only H2 / H4; dead otherwise.
D2  word -> agentview attention (heads, role word rows, layers {0,1,2}), argmax in the oracle region; l90c by zs episode type,
    libero by its own-sample plurality proxy.  Estimate gate: hit(wrong) >= hit(success) - 0.10 and >= 2 x chance (object role;
    goal role the same).
D3  SigLIP region-pooled identity: leave-scene-out nearest-prototype accuracy >= 0.8 and >= 3 x chance; stable if |far - near|
    accuracy <= 0.10; dissociation = both.  Vision anchor (argmax per-token cosine to the named type's prototype): the D2 gate.
"""
import glob
import hashlib
import json
import os
import sys
ROOT = "/coc/flash10/ksingara3/lifelong-ood-learner"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger

OUT = os.path.join(ROOT, "results", "of")
for m in ("l90c", "libero"):
    assert not glob.glob(os.path.join(OUT, f"obs_{m}", "*.npz")), f"a D1 frame file of {m} already exists"
    assert not glob.glob(os.path.join(OUT, f"dev_{m}", "*.npz")), f"a dev file of {m} already exists"
    assert not os.path.exists(os.path.join(OUT, f"protos_{m}.npz")), "prototypes already exist"
assert not os.path.exists(os.path.join(OUT, "read.json")) and not os.path.exists(os.path.join(OUT, "sanity.json"))
assert not ledger.load("of_read"), "an of_read row already exists"
files = {n: os.path.join(ROOT, p) for n, p in (("script", "experiments/of_frame.py"), ("doc", "docs/PLAN_OF.md"),
                                               ("index_l90c", "results/of/index_l90c.json"), ("index_libero", "results/of/index_libero.json"),
                                               ("sel_l90c", "results/of/sel_l90c.json"), ("sel_libero", "results/of/sel_libero.json"))}
sha = {n: hashlib.sha256(open(f, "rb").read()).hexdigest() for n, f in files.items()}
sel = {m: json.load(open(files[f"sel_{m}"], encoding="utf-8")) for m in ("l90c", "libero")}
counts = {m: dict(frames=len(s["obs"]), tasks=len({o["tid"] for o in s["obs"]}), multi_scene_types={c: len(v) for c, v in
                                                                                                   s["multi_scene_cats"].items()})
          for m, s in sel.items()}
print(json.dumps(dict(sha=sha, counts=counts), indent=1))
DISCLOSURE = ("seen before this row: P2/P3, R2a, R3b, R3c, L0/L0b/L1 published results (docs/ATLAS.md); the index / poses / select "
              "logs (task lists, candidate and kept frame counts, object types per scene; no model output); a 2-frame GPU smoke per "
              "model (results/of/scratch: runtime, conditions, image-vs-sim-render alignment correlations; no endpoint, transfer, "
              "attention or feature statistic computed).  The index was rerun once before the poses were used, adding a filter that "
              "the instruction's FIRST step is put / pick up / stack (two-step tasks starting with turn on / open approach another "
              "body before the named object).  D3 was added by the user (via the coordinator) before this row was written and is "
              "pre-registered in it.  P2's transfer definition was read from experiments/pp2_dev.py; the D1 statistic generalises it "
              "(docs/PLAN_OF.md section 0).  Hypothesis family: Holm over H1-H4 (D1 only); D2 / D3 are estimate gates.")
print(ledger.record(
    "m3_prereg", bench="libero_90_c", tag="pof_frame", seed=0,
    metrics={"H1": "D1 l90c part (obj:named spliced mode): T_obj >= 0.5, T_obj - T_world >= 0.3, T_obj > T_ctrl, Holm p",
             "H2": "D1 l90c instr mode: as H1", "H3": "D1 libero part: as H1", "H4": "D1 libero instr: as H1",
             "family": "Holm over H1-H4 at 0.05; p = max of the two one-sided task-cluster bootstrap p (obj <= world, obj <= ctrl)",
             "d1_statistic": "leave-scene-out 3-parameter approach model per object type, P2 corr2 pooled per type, mean over types "
                             "(>= 5 frames, >= 2 scenes); each frame at its own best level of {0.1, 0.3, 0.55} as P2; own samples, "
                             "part variants, alt instructions, null, object_trans and P2-centre descriptive",
             "route1": "alive if H1 or H3; instruction-level only if only H2/H4; dead otherwise",
             "d2_gate": "word->agentview attention, mean heads x role word rows x layers {0,1,2}, argmax in the oracle region; "
                        "hit(wrong) >= hit(success) - 0.10 and >= 2 x chance; l90c zs episode types (episode means, R2a ci_rate), "
                        "libero own-sample plurality proxy (observation unit); object role primary, goal role reported likewise",
             "d3_gates": "SigLIP region-pooled leave-scene-out nearest-prototype accuracy >= 0.8 and >= 3 x chance; stable if "
                         "|acc far - acc near| <= 0.10 (median centroid shift); dissociation = both; vision anchor = argmax "
                         "per-token cosine to the named type prototype, gate as D2",
             "n": counts, "disclosure": DISCLOSURE, "files_sha256": sha,
             "written_before": "any D1 / D2 / D3 statistic, any D1 frame or dev observation run beyond the 2-frame smoke"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/pof_frame.py",
    command="pre-registration: probe OF (D1 object-frame transfer H1-H4, D2 attention anchor gate, D3 vision what/where + anchor)"))
