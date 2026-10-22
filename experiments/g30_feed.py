"""Plan D2: a throttled feeder for G30's fits and reads (docs/BASELINES.md).  Keeps the user's GPU jobs (bar the other agents' zdn*,
x1*, x3*) at or below --cap, submitting the next item when a slot frees; each item is idempotent (launch_pack skips result files
that exist, g30_baselines skips adapters that exist), so the feeder can be killed and rerun.  Exits when every item is submitted
and nothing of ours is queued.  Login node: only squeue / sbatch, no torch.

    nohup python3 experiments/g30_feed.py --cap 9 > slurm/g30_feed.log 2>&1 &
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results", "robot")
PY = os.path.expanduser("~/flash/miniconda3/envs/chunks/bin/python")
SEED = 13
E3_ARMS = ("fullft@_g30", "r3lora@_g30n5", "demodiff@g30")
NALL = "r3lora@_g30nall"
CL_ARMS = ("bcl@ewc_g30", "lora@bcl_packnet_g30", "lora@bcl_packnet75_g30")
OTHERS = ("zdn", "x1", "x3")
MINE = ("g30", "lolpack")


def e3_tasks():
    return sorted(int(t) for t in json.load(open(os.path.join(R, "e3_plan.json"))) if int(t) != 51)


def w1_order():
    return json.load(open(os.path.join(R, "w1_state.json")))["a"]["order"]


def n_mine():
    out = subprocess.run(["squeue", "-u", os.environ["USER"], "-h", "-o", "%j"], capture_output=True, text=True).stdout.split()
    return sum(1 for j in out if j.startswith(MINE) and not j.startswith(OTHERS))


def pack(tasks, arms, exp, fresh, trials, first=0):
    cmd = [PY, "experiments/launch_pack.py", "--tasks", "+".join(map(str, tasks)), "--arms", ",".join(arms), "--experiment", exp,
           "--first", str(first), "--trials", str(trials), "--seed", str(SEED), "--bind", "track", "--partition", "overcap", "--per", "2"]
    if fresh:
        cmd += ["--fresh", "--fresh-dir", "fresh_states_b"]
    return cmd


def items(extra):
    """-> [(name, ready(), cmd, jobs it submits)]: the fits first (the long pole), then the reads."""
    out = []
    nall = lambda t: os.path.exists(os.path.join(R, f"r3_lora_{t}_g30nall.pt"))           # noqa: E731
    for t in e3_tasks():
        if not nall(t):
            out.append((f"fit nall {t}", lambda: True, [sys.executable, "experiments/g30_baselines.py", "--submit", "multiall",
                                                         "--tasks", str(t), "--per-job", "1"], 1))
    reads = []
    for t in e3_tasks():                                                                   # E3 arms: acceptance (0-9) + set b
        reads.append((f"e3 {t}", lambda: True, pack([t], E3_ARMS, "g30_accept", False, 10), 2))
        reads.append((f"e3b {t}", lambda: True, pack([t], E3_ARMS, "g30_main", True, 20), 2))
    for t in w1_order():                                                                   # W1 stream, set b, no acceptance
        reads.append((f"cl {t}", lambda: True, pack([t], CL_ARMS, "g30_clmain", True, 20), 2))
    fits, out = out, []
    for i in range(max(len(fits), len(reads))):                                            # interleave: fits never hog every slot
        out += fits[i:i + 1] + reads[2 * i:2 * i + 2]
    out += reads[2 * max(len(fits), len(reads)):]
    out.append(("parity", lambda: all(nall(t) for t in (13, 36, 52)), [PY, "experiments/g30_baselines.py", "--parity"], 5))
    for t in e3_tasks():
        out.append((f"nall {t}", (lambda t=t: nall(t)), pack([t], (NALL,), "g30_accept", False, 10), 1))
        out.append((f"nallb {t}", (lambda t=t: nall(t)), pack([t], (NALL,), "g30_main", True, 20), 1))
    if "sdp" in extra:                                                                     # the SDP fits, once task 13's smoke fit exists
        out.append(("fit sdp", lambda: os.path.exists(os.path.join(R, "sdp_13_g30_router.pt")),
                    [sys.executable, "experiments/g30_sdp.py", "--submit", "--per-job", "4"], 7))
    for t in e3_tasks() if "sdp" in extra else []:
        f = lambda t=t: os.path.exists(os.path.join(R, f"sdp_{t}_g30_router.pt"))        # noqa: E731
        out.append((f"sdp {t}", f, pack([t], ("sdp@g30",), "g30_accept", False, 10), 1))
        out.append((f"sdpb {t}", f, pack([t], ("sdp@g30",), "g30_main", True, 20), 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=9)
    ap.add_argument("--extra", default="", help="sdp: also the SDP reads, as their routers appear")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    os.chdir(ROOT)
    todo = items(a.extra)
    print(f"[feed] {len(todo)} items", flush=True)
    while todo:
        n = n_mine()
        progressed = False
        for it in list(todo):
            name, ready, cmd, k = it
            if not ready():
                continue
            if not a.dry and n + k > a.cap:
                break
            r = subprocess.run(cmd + (["--dry"] if a.dry else []), capture_output=True, text=True, cwd=ROOT)
            sub = r.stdout.count("-> job") + r.stdout.count("multiall [")
            print(f"[feed] {time.strftime('%H:%M')} {name}: {sub} job(s) {r.stdout.strip()[-150:]!r} {r.stderr.strip()[-200:]!r}", flush=True)
            todo.remove(it)
            n += sub
            progressed = True
        if a.dry:
            break
        if not progressed:
            time.sleep(120)
    while n_mine() and not a.dry:
        time.sleep(300)
    print("[feed] done", flush=True)


if __name__ == "__main__":
    main()
