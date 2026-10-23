"""CALVIN-C flow-loss contamination audit (docs/PLAN_V4.md 3.2: "unannotated play still contains the motions; state this and
run the flow-loss audit"; the I6 method).  SLURM GPU job, one A40, CALVIN venv.

QUESTION.  Does a checkpoint already know the held-out cells?  For each model, the rectified-flow training loss
    L = || (eps - a) - v(z_t, t | o, l) ||^2,  z_t = (1 - t) a + t eps     (FLOWERVLA.rf_loss, per window frame, mean over 10x7)
on ABC training windows of the split's DEV held-out cells against windows of its training block cells, with the same
K (eps, t) draws per frame for every model and instruction (seeded by window, frame and draw).  Sealed cells are never read.
Instructions: `ann` = the window's own training annotation; `canon` = CALVIN's evaluation sentence for the window's subtask;
`swap_colour` / `swap_slot` = the evaluation sentence of the same family with the colour (or the place / direction) changed.
Delta_colour = L(swap_colour) - L(canon) measures how much the model's field depends on the named colour; a model that never
saw a cell should show a smaller Delta there than on its training cells.  The comparison of interest is held-out vs training
cells WITHIN a model; across models only the pattern is compared.
MODELS.  `pret` = FLOWERVLA built by the CALVIN recipe from flower_vla_pret (step 0 of the fine-tune: the pretrained
Florence-2 + DiT weights that load, the rest at init; missing / unexpected keys are recorded); `abc` = mbreuss/flower_calvin_abc
(trained on every cell: the "seen" reference); any exported FLOWER-C dir (config.yaml + model.safetensors).  HF-format dirs
load with the authors' load_mode_from_safetensor plus the load fix (x1_calvin_flower_eval._fix_weights).
Statistics: mean per window; held-out minus training difference with a cell-cluster bootstrap 95% CI (2,000 draws).

    python experiments/c_calvin_audit.py --split_seed 0 --models pret abc [name=<dir> ...] [--record]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
FLOWER = pathlib.Path("/coc/flash10/ksingara3/robot/ext/flower_vla_calvin")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from c_train_flower import DATA, PRET, PUB_CFG  # noqa: E402
from lifelong_ood import calvin_c as C  # noqa: E402

ABC = os.path.dirname(PUB_CFG)
N_FRAMES = 6                                            # frames per window, evenly spread over its sample starts


def swap(cell, what):
    v, c, s = cell
    if what == "colour":
        return (v, C.COLOURS[(C.COLOURS.index(c) + 1) % 3], s)
    sl = C.SLOTS[v]
    return (v, c, sl[(sl.index(s) + 1) % len(sl)])


def seeded(*key):
    return int(hashlib.sha256("-".join(map(str, key)).encode()).hexdigest()[:8], 16)


def load_model(name, path, prompt_embeds=None):
    import hydra
    import torch
    from omegaconf import OmegaConf
    info = dict(name=name, path=path)
    if name == "pret":
        cfg = OmegaConf.load(PUB_CFG)
        mc = cfg.model
        mc.pretrained_model_path = PRET
        mc.load_pretrained = True
        torch.manual_seed(242)
        model = hydra.utils.instantiate(mc)
        miss, unexp = model._load_pretrained_weights(PRET)
        n_all = len(model.state_dict())
        info.update(n_keys=n_all, n_missing=len(miss), n_unexpected=len(unexp),
                    missing_prefixes=sorted({".".join(k.split(".")[:2]) for k in miss}),
                    unexpected_prefixes=sorted({".".join(k.split(".")[:2]) for k in unexp}))
        if prompt_embeds and os.path.exists(prompt_embeds):
            model.prompt_embeds = torch.load(prompt_embeds, map_location="cpu")
            info["prompt_embeds"] = prompt_embeds
    else:
        import flower.evaluation.utils as fu
        from x1_calvin_flower_eval import _fix_weights
        torch.manual_seed(0)
        model = fu.load_mode_from_safetensor(pathlib.Path(path) / "model.safetensors", overwrite_cfg={})
        info["fix"] = _fix_weights(model, pathlib.Path(path) / "model.safetensors")
    model = model.cuda().eval()
    for q in model.parameters():
        q.requires_grad_(False)
    return model, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--models", nargs="+", default=["pret", "abc"])
    ap.add_argument("--windows", type=int, default=24, help="windows per cell")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--prompt_embeds", default=str(ROOT / "results/calvin_c/flower_c_s0/prompt_embeds.pt"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, str(FLOWER))
    import calvin_env  # noqa: F401
    import hydra
    import numpy as np
    import torch
    import torchvision
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader, Subset

    t0 = time.time()
    split = C.load_split(a.split_seed)
    dev = [tuple(d["cell"]) for d in split["dev"]]
    train_cells = [tuple(c) for c in split["train"]]
    canon = OmegaConf.load(FLOWER / "conf/annotations/new_playtable_validation.yaml")
    canon = {k: list(v)[0] for k, v in canon.items()}

    cfg = OmegaConf.load(PUB_CFG)
    tv = cfg.datamodule.transforms.val
    tf = {cam: torchvision.transforms.Compose([hydra.utils.instantiate(t) for t in tv[cam]]) for cam in tv}
    dcfg = OmegaConf.to_container(cfg.datamodule.datasets.lang_dataset, resolve=False)
    for k in ("batch_size", "num_workers"):
        dcfg.pop(k, None)
    from flower.datasets.disk_dataset import ExtendedDiskDataset
    ds = ExtendedDiskDataset(datasets_dir=pathlib.Path(DATA) / "training", transforms=tf, key="lang",
                             save_format="npz", batch_size=1, min_window_size=10, max_window_size=10,
                             proprio_state=cfg.datamodule.proprioception_dims, obs_space=cfg.datamodule.observation_space,
                             skip_frames=1, pad=False, lang_folder=cfg.lang_folder, aux_lang_loss_window=8, num_workers=0,
                             action_seq_len=10, obs_seq_len=1, future_range=1, use_extracted_rel_actions=True)
    lang = np.load(pathlib.Path(DATA) / "training" / cfg.lang_folder / "auto_lang_ann.npy", allow_pickle=True).item()
    tasks = [str(t) for t in lang["language"]["task"]]
    keep = np.load(C.RESULTS / f"windows_s{a.split_seed}.npz")["keep"]
    ll = np.asarray(ds.lang_lookup)
    first, count = {}, {}
    for pos, w in enumerate(ll):
        first.setdefault(int(w), pos)
        count[int(w)] = count.get(int(w), 0) + 1

    rows, picks = [], []
    for cell in dev + train_cells:
        C.check_cell(cell, a.split_seed)
        name = C.cell_name(cell)
        role = "held_out" if cell in dev else "train"
        cand = [i for i, t in enumerate(tasks) if t == name and (role == "held_out" or keep[i])]
        rng = np.random.default_rng(seeded("calvin_c_audit", a.split_seed, name))
        for w in sorted(rng.choice(cand, size=min(a.windows, len(cand)), replace=False).tolist()):
            n_w = count[w]                                   # windows vary in length: spread the frames over its starts
            for o in sorted({int(round(x)) for x in np.linspace(0, n_w - 1, N_FRAMES)}):
                assert ll[first[w] + o] == w, (w, o)
                picks.append((first[w] + o, w, o, name, role))
    print(f"[audit] {len(dev)} dev cells, {len(train_cells)} training cells, {len(picks)} frames", flush=True)

    texts = {}
    for _, w, _, name, _ in picks:
        cell = C.parse_cell(name)
        texts[w] = dict(ann=None, canon=canon[name], swap_colour=canon[C.cell_name(swap(cell, "colour"))],
                        swap_slot=canon[C.cell_name(swap(cell, "slot"))])
    conds = ("ann", "canon", "swap_colour", "swap_slot")
    results = {}
    infos = []
    for spec in a.models:
        name, path = (spec.split("=", 1) + [None])[:2] if "=" in spec else (spec, ABC if spec == "abc" else None)
        model, info = load_model(name, path, a.prompt_embeds)
        infos.append(info)
        print(f"[audit] model {name}: {json.dumps(info, default=str)[:600]}", flush=True)
        loss = {c: np.zeros(len(picks)) for c in conds}
        dl = DataLoader(Subset(ds, [p[0] for p in picks]), batch_size=a.batch, num_workers=a.workers, shuffle=False)
        n = 0
        for batch in dl:
            b = len(batch["actions"])
            meta = picks[n:n + b]
            assert [int(x) for x in batch["idx"]] == [p[0] for p in meta]
            acts = batch["actions"].cuda().float()
            eps = torch.stack([torch.stack([torch.from_numpy(np.random.default_rng(seeded(p[1], p[2], k)).standard_normal((10, 7))).float()
                                            for p in meta]) for k in range(a.K)]).cuda()
            tt = torch.tensor([[np.random.default_rng(seeded("t", p[1], p[2], k)).uniform(1e-5, 1 - 1e-5) for p in meta]
                               for k in range(a.K)], dtype=torch.float32).cuda()
            for cnd in conds:
                lt = list(batch["lang_text"]) if cnd == "ann" else [texts[p[1]][cnd] for p in meta]
                bb = {"rgb_obs": {k: v.cuda() for k, v in batch["rgb_obs"].items()}, "lang_text": lt}
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    cond = model.encode_observations(bb)
                    tot = torch.zeros(b, device="cuda")
                    for k in range(a.K):
                        te = tt[k].view(b, 1, 1)
                        zt = (1 - te) * acts + te * eps[k]
                        v = model.dit_forward(zt, tt[k], cond).float()
                        tot += (((eps[k] - acts) - v) ** 2).mean(dim=(1, 2))
                loss[cnd][n:n + b] = (tot / a.K).cpu().numpy()
            n += b
        results[name] = loss
        del model
        torch.cuda.empty_cache()
        print(f"[audit] {name} done {time.time() - t0:.0f}s", flush=True)

    roles = np.array([p[4] for p in picks])
    cells = np.array([p[3] for p in picks])
    wins = np.array([p[1] for p in picks])

    def per_window(x):
        u = sorted(set(wins.tolist()))
        return {w: float(x[wins == w].mean()) for w in u}

    def summary(x):
        pw = per_window(x)
        wc = {w: cells[wins == w][0] for w in pw}
        wr = {w: roles[wins == w][0] for w in pw}
        by_cell = {}
        for w, v in pw.items():
            by_cell.setdefault(wc[w], []).append(v)
        cm = {c: float(np.mean(v)) for c, v in by_cell.items()}
        h = [cm[c] for c in cm if wr[next(w for w in pw if wc[w] == c)] == "held_out"]
        t = [cm[c] for c in cm if wr[next(w for w in pw if wc[w] == c)] == "train"]
        rng = np.random.default_rng(0)
        boots = [np.mean(rng.choice(h, len(h))) - np.mean(rng.choice(t, len(t))) for _ in range(2000)]
        return dict(held_out=float(np.mean(h)), train=float(np.mean(t)), diff=float(np.mean(h) - np.mean(t)),
                    ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], by_cell=cm)

    out = dict(split_seed=a.split_seed, dev_cells=[C.cell_name(c) for c in dev], n_train_cells=len(train_cells),
               windows_per_cell=a.windows, frames_per_window=N_FRAMES, K=a.K, models=infos, seconds=time.time() - t0)
    for name, loss in results.items():
        m = {c: summary(loss[c]) for c in conds}
        m["delta_colour"] = summary(loss["swap_colour"] - loss["canon"])
        m["delta_slot"] = summary(loss["swap_slot"] - loss["canon"])
        out[name] = m
        print(f"[audit] {name}: " + " | ".join(f"{k} H {v['held_out']:.4f} T {v['train']:.4f} d {v['diff']:+.4f} "
                                              f"[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]" for k, v in m.items()), flush=True)
    p = pathlib.Path(a.out or C.RESULTS / f"audit_s{a.split_seed}_{'_'.join(results)}.json")
    p.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    np.savez(str(p.with_suffix(".npz")), picks=np.array([p_[:3] for p_ in picks]), cells=cells, roles=roles,
             **{f"{n}__{c}": v for n, l in results.items() for c, v in l.items()})
    print("[audit] wrote", p, flush=True)
    if a.record:
        from lifelong_ood import ledger
        print(ledger.record("c_calvin_audit", bench=C.BENCH, tag="_".join(results), seed=a.split_seed,
                            metrics={k: v for k, v in out.items() if k not in ("models",)},
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(models=a.models, windows=a.windows, K=a.K, n_frames=N_FRAMES),
                            script="experiments/c_calvin_audit.py", command=" ".join(sys.argv), artifacts=[str(p)]))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
