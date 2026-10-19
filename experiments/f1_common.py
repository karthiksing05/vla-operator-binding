"""Plan F1 (docs/FIT_TIME.md): shared helpers of the fit-time study.  No torch at import time.

    c0()                      the c0_fit module (or, with tmp=True, the not-yet-installed experiments/_c0_fit_f1_tmp.py, which also
                              installs lifelong_ood/policy/_adapt_f1_tmp.py as lifelong_ood.policy.adapt: the import test of an edit)
    fit_row(name)             the latest ledger c0_fit row of a prototype
    fit_args(name)            that row's command parsed into the fields concept_frames needs (a Namespace)
    kind(name)                approach (object / instance unit) or goal
    pool(kind_, exclude)      the warm-start candidates of a kind: E3's library and G4's held concepts (pi0.5's own demonstrations)
    E3_NAMES, E5_NAMES        the prototypes E3's and E5's plans name (E5: the 26 units E5 added)
"""
import argparse
import glob
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results", "robot")
F1 = os.path.join(R, "f1")
PY = os.path.expanduser("~/flash/miniconda3/envs/chunks/bin/python")
EXCLUDE = "hal,tars,t1000,vicki,johnny5,c3po,olivaw"

E3PLAN = json.load(open(os.path.join(R, "e3_plan.json")))
E5PLAN = json.load(open(os.path.join(R, "e5_plan.json")))["tasks"]
E3_NAMES = sorted({v["object_proto"] for v in E3PLAN.values()} | {v["goal_proto"] for v in E3PLAN.values()})
E5_NAMES = sorted(({v["object_proto"] for v in E5PLAN.values()} | {v["goal"] for v in E5PLAN.values()}) - set(E3_NAMES))
PARENT = {v["object_proto"]: v["type_proto"] for v in E3PLAN.values() if v["instance_key"]}   # instance child -> its type
PARENT.update({v["object_proto"]: v["type"] for v in E5PLAN.values() if v["instance_key"]})


def c0(tmp=False):
    sys.path.insert(0, ROOT)
    sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
    if tmp:
        import lifelong_ood.policy as pol
        spec = importlib.util.spec_from_file_location("lifelong_ood.policy.adapt",
                                                      os.path.join(ROOT, "lifelong_ood", "policy", "_adapt_f1_tmp.py"))
        A = importlib.util.module_from_spec(spec)
        sys.modules["lifelong_ood.policy.adapt"] = A
        spec.loader.exec_module(A)
        pol.adapt = A
    f = "_c0_fit_f1_tmp.py" if tmp else "c0_fit.py"
    spec = importlib.util.spec_from_file_location("c0", os.path.join(ROOT, "experiments", f))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fit_row(name):
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(ROOT, "ledger", "c0_fit", f"*joint_{name}__*.json"))]
    rows = [r for r in rows if "--score" not in r["command"]]
    return max(rows, key=lambda r: r["started"])


def fit_args(name):
    """The prototype's c0_fit command -> Namespace(joint, key, goal, approach, source, ndemo, demo, rank, name, steps, lr, batch)."""
    toks = fit_row(name)["command"].split("c0_fit.py", 1)[1].split()
    a = argparse.Namespace(joint="", key="", goal=False, approach=False, source="l90", ndemo=1, demo=0, rank=16, name=name,
                           steps=1000, lr=1e-4, batch=32, init="", bind=False, backbone="pi05_libero")
    i = 0
    while i < len(toks):
        k = toks[i].lstrip("-").replace("-", "_")
        if k in ("goal", "approach", "bind"):
            setattr(a, k, True)
            i += 1
            continue
        v = toks[i + 1]
        cur = getattr(a, k, None)
        setattr(a, k, type(cur)(v) if isinstance(cur, (int, float)) and not isinstance(cur, bool) else v)
        i += 2
    a.key = a.key.replace("_", " ")
    return a


def kind(name):
    return "goal" if fit_args(name).goal else "approach"


def held_names():
    """G4's prototypes of concepts the policy holds (fitted on pi0.5's own 40 training tasks, --source d40)."""
    out = set()
    for f in glob.glob(os.path.join(ROOT, "ledger", "c0_fit", "*joint_Q*__*.json")):
        r = json.load(open(f))
        if "--source d40" in r["command"] and "--score" not in r["command"]:
            out.add(r["command"].split("--name ")[1].split()[0])
    return sorted(n for n in out if os.path.exists(os.path.join(R, f"prim_{n}_0.pt")))


def pool(kind_, exclude=()):
    names = [n for n in E3_NAMES + held_names() if n not in set(exclude)]
    return [n for n in names if kind(n) == kind_ and os.path.exists(os.path.join(R, f"prim_{n}_0.pt"))]


def sbatch(script, dep=""):
    import subprocess
    args = ["sbatch", "--parsable", f"--exclude={EXCLUDE}", "--partition=overcap", "--account=overcap", "--qos=short"]
    args += [f"--dependency=afterok:{dep}"] if dep else []
    args += [f"--export=ALL,ENV=openpi,SCRIPT={script}", "slurm/jobs/robot_gpu.sbatch"]
    out = subprocess.run(args, capture_output=True, text=True, cwd=ROOT)
    return out.stdout.strip() or out.stderr.strip()
