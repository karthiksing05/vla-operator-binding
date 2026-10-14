"""One screen of what is running and what is recorded.

    ~/flash/miniconda3/envs/chunks/bin/python experiments/status.py
"""
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
R = os.path.join(ROOT, "results", "robot")
READS = ["e1_factored", "e2_instance", "e3_hier", "e4_stream", "w1_stream", "w2_stream", "w0_select", "t0_transfer", "i4_factors",
         "e5_replicate"]                      # the current program's reads, in the order they ran


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def tail(path, n=1):
    if not os.path.exists(path):
        return ""
    lines = [l for l in open(path, errors="replace").read().splitlines() if l.strip() and "Warn" not in l]
    return "\n".join(lines[-n:])


def main():
    print("== drivers running")
    procs = sh(f"ps -u {os.environ.get('USER', '')} -o args= | grep -E '(_stream|_read|_chain[0-9]*)\\.(py|sh)' | grep -v grep")
    print(procs or "   none")
    print("\n== streams")
    for f in sorted(glob.glob(os.path.join(R, "*_state.json"))):
        name = os.path.basename(f)[:-11]
        try:
            st = json.load(open(f))
        except (OSError, ValueError):
            continue
        orders = [o for o in st if not o.startswith("_") and isinstance(st[o], dict) and "order" in st[o]]
        if not orders:
            continue
        prog = "  ".join(f"{o}: {st[o]['i']}/{len(st[o]['order'])} {st[o].get('phase', '')} demos {st[o].get('demos', '-')}" for o in orders)
        test = st.get("_test", {}).get("jobs")
        print(f"   {name:5s} {prog}  test {'launched' if test else '-'}  | {tail(os.path.join(R, name + '_driver.log'))[:90]}")
    for f in sorted(glob.glob(os.path.join(R, "*_chain.log"))):
        print(f"   {os.path.basename(f)[:-4]}: {tail(f)[:110]}")
    q = sh(f"squeue -u {os.environ.get('USER', '')} -h -o '%T'")
    print(f"\n== queue: {q.count('RUNNING')} running, {q.count('PENDING')} pending")
    print("\n== reads recorded (latest row)")
    from lifelong_ood import ledger
    for exp in READS:
        rows = [r for r in ledger.load(exp) if not r.get("retired")]
        pre = ledger.load("m3_prereg", tag=exp) or ledger.load("m3_prereg", tag=exp.split("_")[0])
        status = f"{rows[-1].get('tag')} @ {str(rows[-1].get('started', ''))[:16]}" if rows else "not recorded"
        print(f"   {exp:14s} {'pre-registered' if pre else 'exploratory   '}  {status}")


if __name__ == "__main__":
    main()
