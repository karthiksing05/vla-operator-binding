"""Symbolic composition over continuous prototypes (docs/FRAMEWORK.md, docs/S0_SYNTH.md).

    gcobweb   numpy mixed Gaussian / nominal Cobweb (incorporate / create / merge / split, best-first
              categorisation, basic level, counts)
    store     TRELLIS's linked context and content taxonomies over elements with codes z in R^d;
              "then" and "with" relations (submodular coverage + closed-form PoE)
    parse     bottom-up greedy parsing with the recognition threshold; NEW remainder
    generate  top-down expansion from a recombination pool, ranked by a context callback
    learn     discover / incorporate (L1) / create (L2) / consolidate
    synth     a synthetic compositional manipulation domain

Compatibility: this package shadows the older module file lifelong_ood/compose.py (the PoE teacher used
by lifelong_ood.neural and lifelong_ood.policy).  That file is loaded here unchanged and its names are
re-exported lazily, so `from lifelong_ood.compose import poe_teacher` (etc.) keeps working.
"""
import importlib.util as _ilu
import os as _os
import sys as _sys

_LEGACY = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "compose.py")


def _legacy():
    name = "lifelong_ood._compose_legacy"
    mod = _sys.modules.get(name)
    if mod is None:
        spec = _ilu.spec_from_file_location(name, _LEGACY)
        mod = _ilu.module_from_spec(spec)
        mod.__package__ = "lifelong_ood"
        _sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


def __getattr__(name):
    """Lazy re-export of the legacy module's names (loaded only when one is asked for, so importing the
    numpy-only composition machinery never pulls in the legacy module's dependencies)."""
    if name.startswith("__") or not _os.path.exists(_LEGACY):
        raise AttributeError(name)
    mod = _legacy()
    if hasattr(mod, name):
        return getattr(mod, name)
    raise AttributeError(f"module 'lifelong_ood.compose' has no attribute {name!r}")
