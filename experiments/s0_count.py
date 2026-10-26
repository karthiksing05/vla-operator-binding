"""PLAN_SOLVE S0: count finished rollout units per arm (no outcome is computed)."""
import glob, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in sorted(glob.glob(os.path.join(ROOT, "results/bind/s0/roll/*"))):
    print(os.path.basename(d), len(glob.glob(os.path.join(d, "*.npz"))), flush=True)
