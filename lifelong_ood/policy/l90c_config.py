"""openpi TrainConfigs for LIBERO-90-C (docs/L90C.md), registered at runtime so openpi's own config.py is never edited.

  pi05_l90c        pi05_libero's recipe (same model: pi0.5, action horizon 10, continuous state dropped; same LIBERO transforms;
                   AdamW, clip 1.0, EMA 0.999, cosine schedule) from pi05_base, on lol/libero_l90c (the frozen split's
                   training tasks only, experiments/c_data.py), with its own norm stats.  FSDP over 4 GPUs.  Batch size and
                   steps are set by experiments/c_train_pi05.py (the 200-step timing job decides them) and recorded there.
  pi05_l90c_lora   the same with openpi's LoRA variants (gemma_2b_lora + gemma_300m_lora), no EMA (openpi's low-mem recipe).
  pi05_l90c_fac    pi05_l90c plus FACTORED CONDITIONING (docs/L90C.md 3.1): per sample, with probability 0.3 the instruction is
                   replaced by one of its part prompts (object-only / goal-only / verb-only, the plain variant of
                   lifelong_ood/protos/store.py's rule), with probability 0.1 by null (""); the action target is unchanged
                   (Factored Diffusion Policies 2605.22596's per-factor dropout; CRAFT 2610.00524's counterfactual alignment).
  pi05_base_libero pi05_base itself under the LIBERO transforms and pi05_libero's norm stats: the contamination audit's handle.

`register()` puts them into openpi's registry (openpi.training.config._CONFIGS_DICT) so get_config / create_trained_policy /
PolicyBackbone find them by name.
"""
from __future__ import annotations

import dataclasses
import os
import sys

OPENPI = "/coc/flash10/ksingara3/robot/openpi"
CACHE = "/coc/flash10/ksingara3/robot/openpi_cache/openpi-assets/checkpoints"
L90C = "/coc/flash10/ksingara3/robot/checkpoints_l90c"          # our checkpoints, assets and converted PyTorch weights
REPO_ID = "lol/libero_l90c"
BASE_PARAMS = f"{CACHE}/pi05_base/params"
NAMES = ("pi05_l90c", "pi05_l90c_lora", "pi05_base_libero", "pi05_l90c_fac")
FAC_PROMPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "l90c",
                           "fac_prompts.json")
P_PART, P_NULL = 0.3, 0.1


def _openpi():
    if OPENPI + "/src" not in sys.path:
        sys.path.insert(0, OPENPI + "/src")
    import openpi.training.config as C
    return C


def train_config(name, **over):
    C = _openpi()
    from openpi.models import pi0_config
    from openpi.training import optimizer as O
    from openpi.training import weight_loaders as W
    lib = C.get_config("pi05_libero")
    if name == "pi05_base_libero":
        cfg = dataclasses.replace(lib, name=name, weight_loader=W.CheckpointWeightLoader(BASE_PARAMS),
                                  data=dataclasses.replace(lib.data, assets=C.AssetsConfig(
                                      assets_dir=f"{CACHE}/pi05_libero/assets", asset_id="physical-intelligence/libero")))
        return dataclasses.replace(cfg, **over)
    if name == "pi05_l90c_fac":                                 # same data, steps and norm stats as pi05_l90c
        base = train_config("pi05_l90c")
        cfg = dataclasses.replace(base, name=name, data=_fac_data(C, base.data))
        return dataclasses.replace(cfg, **over)
    lora = name == "pi05_l90c_lora"
    model = (pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False,
                                  paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
             if lora else pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False))
    cfg = C.TrainConfig(
        name=name, model=model,
        data=C.LeRobotLiberoDataConfig(repo_id=REPO_ID, base_config=C.DataConfig(prompt_from_task=True),
                                       extra_delta_transform=False),
        batch_size=32,
        lr_schedule=O.CosineDecaySchedule(warmup_steps=1_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5),   # pi05_libero's (constant
        # 5e-5 after warmup); warmup 1k instead of 10k for the smaller batch
        optimizer=O.AdamW(clip_gradient_norm=1.0),
        ema_decay=None if lora else 0.999,
        freeze_filter=model.get_freeze_filter() if lora else C.TrainConfig.__dataclass_fields__["freeze_filter"].default_factory(),
        weight_loader=W.CheckpointWeightLoader(BASE_PARAMS),
        num_train_steps=30_000, fsdp_devices=1 if lora else 4,
        assets_base_dir=os.path.join(L90C, "assets"), checkpoint_base_dir=os.path.join(L90C, "jax"),
        save_interval=1000, keep_period=5000, log_interval=50, num_workers=8, wandb_enabled=False)
    return dataclasses.replace(cfg, **over)


def fac_prompts(rebuild=False):
    """instruction -> [object-only, goal-only, verb-only] plain part prompts of every training task (store.parts_of's rule; a
    fixture task has no goal part).  Cached in results/l90c/fac_prompts.json."""
    import json
    if os.path.exists(FAC_PROMPTS) and not rebuild:
        return json.load(open(FAC_PROMPTS, encoding="utf-8"))
    from .. import l90c
    from ..protos import store as S
    from . import factors as F
    tasks = {t.tid: t for t in F.load(("libero_90", "libero_10", "libero_spatial", "libero_object", "libero_goal"))}
    phrases = F.object_phrases(list(tasks.values()))
    out = {}
    for tid in l90c.load_split()["train"]:
        t = tasks[tid]
        pt = S.parts_of(t, phrases)
        parts = [f"the {pt['obj']}"] if pt["obj"] else []
        parts += [pt["goal"]] if pt["goal"] else []
        parts += [pt["verb"]] if pt["verb"] else []
        out[t.language] = parts
    os.makedirs(os.path.dirname(FAC_PROMPTS), exist_ok=True)
    with open(FAC_PROMPTS, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


class FactoredPrompt:
    """openpi data transform: the factored-conditioning prompt dropout (training only; inference never sees it).
    Its RNG is created lazily per process, so forked loader workers do not share a stream."""

    def __init__(self, table, p_part=P_PART, p_null=P_NULL, seed=0):
        self.table, self.p_part, self.p_null, self.seed = dict(table), p_part, p_null, seed
        self._pid, self._rng = None, None

    def __call__(self, data):
        import numpy as np
        if self._pid != os.getpid():
            self._pid, self._rng = os.getpid(), np.random.default_rng([self.seed, os.getpid()])
        p = data.get("prompt")
        text = p.decode() if isinstance(p, bytes) else str(np.asarray(p).item() if not isinstance(p, str) else p)
        u = self._rng.random()
        parts = self.table.get(text, [])
        if u < self.p_null:
            text = ""
        elif u < self.p_null + self.p_part and parts:
            text = parts[int(self._rng.integers(len(parts)))]
        return {**data, "prompt": text}


def _fac_data(C, base):
    """A LeRobotLiberoDataConfig whose data transforms start with FactoredPrompt (after the repack, before tokenisation)."""
    import openpi.transforms as T

    @dataclasses.dataclass(frozen=True)
    class FactoredLiberoDataConfig(C.LeRobotLiberoDataConfig):
        def create(self, assets_dirs, model_config):
            dc = super().create(assets_dirs, model_config)
            inputs = [FactoredPrompt(fac_prompts())] + list(dc.data_transforms.inputs)
            return dataclasses.replace(dc, data_transforms=T.Group(inputs=inputs, outputs=dc.data_transforms.outputs))
    return FactoredLiberoDataConfig(repo_id=base.repo_id, base_config=base.base_config,
                                    extra_delta_transform=base.extra_delta_transform,
                                    assets=C.AssetsConfig(assets_dir=os.path.join(L90C, "assets", "pi05_l90c"), asset_id=REPO_ID))


def register(names=NAMES):
    C = _openpi()
    for n in names:
        if n not in C._CONFIGS_DICT:
            C._CONFIGS_DICT[n] = train_config(n)
    return C


def pytorch_dir(name):
    """Where the converted PyTorch weights of a config live (PolicyBackbone's checkpoint_dir)."""
    return f"{CACHE}/pi05_base_pytorch" if name == "pi05_base_libero" else os.path.join(L90C, "pytorch", name)
