"""CALVIN-C Stage A (headroom): zero-shot single-subtask success of FLOWER-C vs the full-ABC control on the DEV held-out
cells of split seed 0 (pre-registration: experiments/prereg/cc_headroom.py, ledger m3_prereg tag cc_headroom).

Question.  FLOWER-C (held-out cells' windows removed) and the control flower_abc_full (every window, same recipe) tie on the
chains without held-out cells (4.527 vs 4.528), and the flow-loss audit shows only a +0.030 held-out gap.  Is there any
zero-shot success gap on the held-out cells themselves?  If the control does not beat FLOWER-C by a clear margin, the split
leaves no headroom for a composition method.

Episodes (identical for both models).
  * Cells: the 5 dev held-out cells of split_s0.json (each passed through calvin_c.check_cell; sealed cells are never named
    or read), plus 3 TRAINING sibling cells (same verb and colour or direction family) as a harness check only.
  * Start states: every initial condition of the authors' evaluation sequence generator (flower/evaluation/
    multistep_sequences.get_sequences: led, lightbulb, slider, drawer, block places, grasped 0, 1-2 blocks on the table) in
    which the cell's precondition holds (the cell's block on the table, nothing grasped).  The scene is built by the authors'
    get_env_state_for_initial_condition (block table slot and yaw seeded per condition).  These are scene-D states: the
    harness has only the scene-D env config (task_ABC_D/validation/.hydra); no ABC env config exists here.
  * Instruction: CALVIN's validation sentence for the cell (new_playtable_validation, val_annotations[cell][0]), the same
    sentence the chain evaluation uses.
  * Rollout: the authors' rollout() logic (ep_len 360, num_sampling_steps 4, multistep 10, success = the task oracle detects
    the cell's subtask), from a fresh env.reset to the start state.  torch / numpy / random are reseeded per episode from
    sha256(cell|k), so both models see the same starts and the same seeds.

    python experiments/cc_headroom.py roll --ckpt <export dir> --model <name> --part i --nparts n   (calvin venv, GPU)
    python experiments/cc_headroom.py read [--record]                                                  (CPU)
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lifelong_ood import calvin_c as C  # noqa: E402

SPLIT_SEED = 0
OUT = C.RESULTS / "headroom"
MODELS = dict(flower_c_s0=C.RESULTS / "flower_c_s0" / "export_stepfinal",
              flower_abc_full=C.RESULTS / "flower_abc_full" / "export_stepfinal")
SIBLINGS = ("push_blue_block_left", "rotate_red_block_right", "push_pink_block_right")   # training cells, harness check
EP_LEN = 360
BOOT = 10000
BAR = 0.10

# the authors' initial-condition space (multistep_sequences.get_sequences), copied so this module needs no FLOWER import;
# roll() asserts the copy against the authors' `tasks` preconditions
POSSIBLE = {"led": [0, 1], "lightbulb": [0, 1], "slider": ["right", "left"], "drawer": ["closed", "open"],
            "red_block": ["table", "slider_right", "slider_left"], "blue_block": ["table", "slider_right", "slider_left"],
            "pink_block": ["table", "slider_right", "slider_left"], "grasped": [0]}


def conditions():
    f = lambda l: l.count("table") in [1, 2] and l.count("slider_right") < 2 and l.count("slider_left") < 2  # noqa: E731
    return [dict(zip(POSSIBLE.keys(), v)) for v in filter(f, itertools.product(*POSSIBLE.values()))]


def cells():
    """-> [(name, role, grade)] : the dev held-out cells (role 'heldout') and the training siblings (role 'train')."""
    s = C.load_split(SPLIT_SEED)
    out = []
    for d in s["dev"]:
        C.check_cell(d["cell"], SPLIT_SEED)
        out.append((C.cell_name(d["cell"]), "heldout", int(d["grade"])))
    train = {C.cell_name(c) for c in s["train"]}
    for n in SIBLINGS:
        assert n in train, n
        out.append((n, "train", None))
    return out


def episodes():
    """-> [dict(cell, role, grade, k, cond, seed)] in a fixed order."""
    eps = []
    for name, role, grade in cells():
        v, c, _ = C.parse_cell(name)
        assert v in ("push", "rotate"), name               # precondition below is the push / rotate one
        ok = [x for x in conditions() if x[f"{c}_block"] == "table" and x["grasped"] == 0]
        for k, cond in enumerate(ok):
            seed = int(hashlib.sha256(f"{name}|{k}".encode()).hexdigest()[:8], 16)
            eps.append(dict(cell=name, role=role, grade=grade, k=k, cond=cond, seed=seed))
    return eps


def episodes_sha():
    return hashlib.sha256(json.dumps(episodes(), sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------------------------ roll
def roll(a):
    flower_root = pathlib.Path(a.flower_root).resolve()
    import calvin_env
    assert calvin_env.__file__ is not None, calvin_env
    sys.path.insert(0, flower_root.as_posix())
    sys.path.insert(0, str(ROOT / "experiments"))
    import random

    import hydra
    import numpy as np
    import torch
    import wandb
    from omegaconf import OmegaConf  # noqa: F401
    wandb.log = lambda *x, **k: None
    import flower.wrappers.hulc_wrapper as hw
    from calvin_env.utils.utils import EglDeviceNotFoundError

    def _no_probe(cuda_id):
        raise EglDeviceNotFoundError()
    hw.get_egl_device_id = _no_probe
    import flower.evaluation.flower_evaluate as fe
    from flower.evaluation import multistep_sequences as ms
    from flower.evaluation.utils import get_env_state_for_initial_condition
    import x1_calvin_flower_eval as X1

    eps = episodes()
    for e in eps:                                        # the copied precondition equals the authors' one
        assert len(ms.valid_task(e["cond"], ms.tasks[e["cell"]])) == 1, e
    mine = eps[a.part::a.nparts]
    out = pathlib.Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = out / f"episodes_{a.part}of{a.nparts}.jsonl"
    done = set()
    if log.exists():
        for ln in log.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                done.add((r["cell"], r["k"]))
    todo = [e for e in mine if (e["cell"], e["k"]) not in done]
    print(f"[cch] model={a.model} part {a.part}/{a.nparts}: {len(mine)} episodes, {len(todo)} to run", flush=True)
    if not todo:
        return

    def evaluate_policy(model, env, lang_embeddings, cfg, num_videos=0, save_dir=None):
        task_oracle = hydra.utils.instantiate(cfg.tasks)
        ann = cfg.annotations
        t0 = time.time()
        for i, e in enumerate(todo):
            robot_obs, scene_obs = get_env_state_for_initial_condition(e["cond"])
            env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
            torch.manual_seed(e["seed"]); np.random.seed(e["seed"] % (2 ** 32)); random.seed(e["seed"])
            torch.cuda.manual_seed_all(e["seed"])
            obs = env.get_obs()
            lang = ann[e["cell"]][0]
            goal = {"lang_text": lang}
            model.reset()
            start_info = env.get_info()
            success, steps, info = False, cfg.ep_len, None
            for step in range(cfg.ep_len):
                action = model.step(obs, goal)
                obs, _, _, info = env.step(action)
                if len(task_oracle.get_task_info_for_set(start_info, info, {e["cell"]})) > 0:
                    success, steps = True, step + 1
                    break
            solved = sorted(task_oracle.get_task_info(start_info, info)) if info is not None else []
            rec = dict(cell=e["cell"], role=e["role"], grade=e["grade"], k=e["k"], seed=e["seed"], success=bool(success),
                       steps=int(steps), solved=solved, lang=lang, model=a.model)
            with open(log, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[cch] {i + 1}/{len(todo)} {e['cell']} k={e['k']} success={success} steps={steps} solved={solved} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        return [], {}

    fe.evaluate_policy = evaluate_policy
    fe.print_and_save = lambda *x, **k: None
    # X1's stand-ins (the scene-D env, the val transforms, the load fix), restated with this run's arguments
    ckpt = pathlib.Path(a.ckpt).resolve()

    def get_default_mode_and_env(train_folder, dataset_path, checkpoint, env=None, lang_embeddings=None,
                                 prep_dm_and_deps=True, device_id=0, eval_cfg_overwrite={}):
        import torchvision
        from omegaconf import OmegaConf as OC
        import flower.evaluation.utils as fu
        cfg = OC.merge(OC.load(pathlib.Path(train_folder) / "config.yaml"), OC.create(eval_cfg_overwrite))
        device = torch.device(f"cuda:{device_id}")

        class _ValDataset:
            def __init__(self, cfg):
                self.abs_datasets_dir = pathlib.Path(a.val_dir).resolve()
                self.observation_space = cfg.datamodule.observation_space
                self.proprio_state = cfg.datamodule.proprioception_dims
                tv = cfg.datamodule.transforms.val
                self.transforms = {cam: torchvision.transforms.Compose([hydra.utils.instantiate(t) for t in tv[cam]])
                                   for cam in tv}
        rollout_cfg = OC.load(flower_root / "conf/callbacks/rollout_lh/calvin.yaml")
        env = hydra.utils.instantiate(rollout_cfg.env_cfg, _ValDataset(cfg), device, show_gui=False)
        model = fu.load_mode_from_safetensor(pathlib.Path(checkpoint), overwrite_cfg=eval_cfg_overwrite.get("model", {}))
        print("[cch] fix_weights:", X1._fix_weights(model, pathlib.Path(checkpoint)), flush=True)
        model.freeze()
        return model.cuda(device), env, None, None

    fe.get_default_mode_and_env = get_default_mode_and_env
    hyd = out / f"hydra_{a.part}"
    os.chdir(out)
    sys.argv = [sys.argv[0], f"--config-path={flower_root / 'conf'}", "--config-name=eval_calvin",
                f"train_folder={ckpt}", f"checkpoint={ckpt / 'model.safetensors'}", f"dataset_path={a.val_dir}",
                f"log_dir={out / f'flower_logs_{a.part}'}","log_wandb=false", "num_videos=0", "device=0",
                "num_sequences=1", f"ep_len={EP_LEN}", f"hydra.run.dir={hyd}"]
    fe.main()


# ------------------------------------------------------------------------------------------------------------------ read
def load_runs():
    res = {}
    for m in MODELS:
        rows = {}
        for p in sorted((OUT / m).glob("episodes_*.jsonl")):
            for ln in p.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    r = json.loads(ln)
                    rows[(r["cell"], r["k"])] = r
        res[m] = rows
    return res


def read(a):
    import numpy as np
    eps = episodes()
    runs = load_runs()
    keys = [(e["cell"], e["k"]) for e in eps]
    missing = {m: sum(k not in runs[m] for k in keys) for m in MODELS}
    if any(missing.values()) and not a.partial:
        raise SystemExit(f"missing episodes: {missing}")
    table, gaps = [], {}
    cellinfo = {}
    for e in eps:
        cellinfo.setdefault(e["cell"], dict(role=e["role"], grade=e["grade"], ks=[]))["ks"].append(e["k"])
    succ = {}
    for cell, ci in cellinfo.items():
        ks = [k for k in ci["ks"] if all((cell, k) in runs[m] for m in MODELS)]
        fc = np.array([runs["flower_c_s0"][(cell, k)]["success"] for k in ks], float)
        ct = np.array([runs["flower_abc_full"][(cell, k)]["success"] for k in ks], float)
        succ[cell] = (fc, ct)
        both = int(((fc == 1) & (ct == 1)).sum()); only_ct = int(((fc == 0) & (ct == 1)).sum())
        only_fc = int(((fc == 1) & (ct == 0)).sum())
        fails = {}
        for k in ks:                                      # what FLOWER-C did instead on its failures
            r = runs["flower_c_s0"][(cell, k)]
            if not r["success"]:
                key = "+".join(r["solved"]) or "none"
                fails[key] = fails.get(key, 0) + 1
        row = dict(cell=cell, role=ci["role"], grade=ci["grade"], n=len(ks), flower_c=float(fc.mean()),
                   control=float(ct.mean()), gap=float(ct.mean() - fc.mean()), both=both, only_control=only_ct,
                   only_flower_c=only_fc, flower_c_failures=dict(sorted(fails.items(), key=lambda kv: -kv[1])),
                   lang=runs["flower_c_s0"][(cell, ks[0])]["lang"] if ks else None)
        table.append(row)
        gaps[cell] = row["gap"]
    held = [r["cell"] for r in table if r["role"] == "heldout"]
    rng = np.random.default_rng(0)
    pooled = float(np.mean([gaps[c] for c in held]))
    boots = np.empty(BOOT)
    for b in range(BOOT):                                 # two-stage cluster bootstrap: cells, then paired episodes
        cs = rng.choice(len(held), len(held), replace=True)
        g = []
        for i in cs:
            fc, ct = succ[held[i]]
            j = rng.integers(0, len(fc), len(fc))
            g.append(ct[j].mean() - fc[j].mean())
        boots[b] = np.mean(g)
    lo, hi = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
    sib = [r for r in table if r["role"] == "train"]
    sib_gap = float(np.mean([r["gap"] for r in sib])) if sib else None
    by_grade = {}
    for r in table:
        if r["role"] == "heldout":
            by_grade.setdefault(r["grade"], []).append(r)
    grade_rows = {g: dict(n_cells=len(rs), flower_c=float(np.mean([r["flower_c"] for r in rs])),
                          control=float(np.mean([r["control"] for r in rs])), gap=float(np.mean([r["gap"] for r in rs])))
                  for g, rs in sorted(by_grade.items())}
    verdict = "headroom" if lo > BAR else "no_headroom"
    out = dict(split_seed=SPLIT_SEED, episodes_sha256=episodes_sha(), n_episodes_per_cell={r["cell"]: r["n"] for r in table},
               cells=table, pooled_gap_heldout=pooled, ci95=[lo, hi], bar=BAR, verdict=verdict,
               by_grade=grade_rows, sibling_gap_mean=sib_gap, missing=missing,
               flower_c_heldout_mean=float(np.mean([r["flower_c"] for r in table if r["role"] == "heldout"])),
               control_heldout_mean=float(np.mean([r["control"] for r in table if r["role"] == "heldout"])),
               scene="D (task_ABC_D/validation env config; no ABC env config here)")
    p = OUT / "headroom_s0.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("| cell | role | grade | n | FLOWER-C | control | gap (control - FLOWER-C) | only control / only FLOWER-C |")
    print("|---|---|---|---|---|---|---|---|")
    for r in table:
        print(f"| {r['cell']} | {r['role']} | {'' if r['grade'] is None else r['grade']} | {r['n']} | {r['flower_c']:.3f} | "
              f"{r['control']:.3f} | {r['gap']:+.3f} | {r['only_control']} / {r['only_flower_c']} |")
    print(f"pooled held-out gap {pooled:+.3f} [{lo:+.3f}, {hi:+.3f}] bar {BAR}: {verdict}; sibling gap {sib_gap}")
    print(json.dumps(grade_rows))
    for r in table:
        print(r["cell"], "FLOWER-C failures:", r["flower_c_failures"])
    print("wrote", p)
    if a.record:
        from lifelong_ood import ledger
        print(ledger.record("cc_headroom", bench=C.BENCH, tag="stage_a", seed=SPLIT_SEED, metrics=out,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(models={k: str(v) for k, v in MODELS.items()}, ep_len=EP_LEN, boot=BOOT,
                                        siblings=SIBLINGS, prereg="m3_prereg tag cc_headroom"),
                            script="experiments/cc_headroom.py", command=" ".join(sys.argv), artifacts=[str(p)]))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("roll")
    r.add_argument("--ckpt", required=True)
    r.add_argument("--model", required=True, choices=list(MODELS))
    r.add_argument("--part", type=int, default=0)
    r.add_argument("--nparts", type=int, default=1)
    r.add_argument("--flower_root", default="/coc/flash10/ksingara3/robot/ext/flower_vla_calvin")
    r.add_argument("--val_dir", default="/coc/flash10/ksingara3/robot/calvin/task_ABC_D/validation")
    r.add_argument("--out", default=None)
    d = sp.add_parser("read")
    d.add_argument("--record", action="store_true")
    d.add_argument("--partial", action="store_true")
    sp.add_parser("list")
    a = ap.parse_args()
    if a.cmd == "roll":
        a.out = a.out or str(OUT / a.model)
        roll(a)
    elif a.cmd == "read":
        read(a)
    else:
        eps = episodes()
        from collections import Counter
        print(Counter((e["cell"], e["role"], e["grade"]) for e in eps), episodes_sha())


if __name__ == "__main__":
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    main()
