"""Plan F1: a rank-4 task leaf, fitted by experiments/r3_fewshot.py's own code path (unchanged), with its ledger row written as
f1_leaf_fit instead of r3_fewshot (plan F1 writes only f1_* rows).  Saves results/robot/r3_lora_<task>_f1r4.pt (launch_pack arm
lorat@_f1r4).

    sbatch ... SCRIPT="experiments/f1_leaf_fit.py --task 2" slurm/jobs/robot_gpu.sbatch
"""
import os
import runpy
import sys
import time

T0 = time.time()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                     # noqa: E402

TAG = "_f1r4"
_record = ledger.record


def record(experiment, **kw):
    if experiment == "r3_fewshot":
        experiment = "f1_leaf_fit"
        kw["script"] = "experiments/f1_leaf_fit.py -> experiments/r3_fewshot.py"
        kw["metrics"] = dict(kw["metrics"], job_secs_to_record=time.time() - T0)
    return _record(experiment, **kw)


ledger.record = record

if __name__ == "__main__":
    task = sys.argv[sys.argv.index("--task") + 1]
    sys.argv = [os.path.join(ROOT, "experiments", "r3_fewshot.py"), "--task", task, "--rank", "4", "--tag", TAG]
    runpy.run_path(sys.argv[0], run_name="__main__")
