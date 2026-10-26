"""PLAN_SOLVE S0: dump each S0 training cell's demo actions (aligned to its match file's states) -> results/bind/s0/acts/<tid>.npy.
Chunks env, CPU.  The actions are flow-matching targets only (never executed)."""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
import s0_data as S
from lifelong_ood.policy import data as D
od = os.path.join(ROOT, "results", "bind", "s0", "acts")
os.makedirs(od, exist_ok=True)
demos, d90 = D.Demos(), D.Demos90()
for c in S.cells():
    mz = np.load(c["match"], allow_pickle=True)
    n = len(mz["states"])
    if c["ipec"] and "actions" in mz.files:
        a, how = np.asarray(mz["actions"], np.float32), "match"
    else:
        a = np.asarray((d90 if c["ipec"] else demos).load(c["episode"])["actions"], np.float32)
        kept = np.asarray(mz["kept"]).astype(int)
        how = f"loader T {len(a)}"
        if len(a) != n:
            a, how = a[kept], how + " [kept]"
    assert len(a) == n, (c["tid"], len(a), n)
    np.save(os.path.join(od, c["tid"].replace(":", "_") + ".npy"), a)
    print(f"[s0 acts] {c['tid']}: {n} frames ({how})", flush=True)
