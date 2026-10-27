"""S2 smoke check: print each smoke unit's binder diagnostic (first-call binder masks vs live oracle masks).  CPU."""
import glob, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "bind", "s2", "smoke")
for f in sorted(glob.glob(os.path.join(d, "*", "*.npz"))):
    m = json.loads(str(np.load(f, allow_pickle=True)["meta"]))
    print("[s2 check]", m["unit"], m["named"], m["own"], json.dumps(m.get("binder_diag")), m["steps"], flush=True)
