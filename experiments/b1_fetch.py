"""B1 stage (b): fetch the CompoSuite EXPERT offline data from Dryad and convert it to compact per-task files.

Source: Dryad doi:10.5061/dryad.9cnp5hqps, version 298028 (2024-06-06).  The expert data is four per-robot tarballs
(no per-task files):  expert-{iiwa,jaco,kinova3,panda}-offline-comp-data.tar.gz, 15.85-15.96 GB each, 63.58 GB total,
sha-256 digests from the Dryad API (below).  Each holds 64 task folders <Robot>_<Object>_<Obstacle>_<Objective>/
data.hdf5 with observations (1e6 x 93), actions (1e6 x 8), rewards, terminals, timeouts, infos: 2000 episodes x 500
steps from one compositional PPO agent (paper: >= 90% success on every task).

ACCESS (verified 2026-10-21): the Dryad API answers 401 "must have current bearer token" for every file download,
and the web file_stream route is behind an AWS WAF (403 to curl).  Downloads therefore need Dryad API credentials:
put {"client_id": "...", "client_secret": "..."} in ~/.dryad_api.json (chmod 600).  Alternatively pass
--url-file with one direct (e.g. presigned) URL per robot, "<robot> <url>" per line.

Because each tarball holds every task of its robot and every cell of the split needs data (train cells for the
policy, test cells for the one demonstration and the multi-task ceiling), all four tarballs are fetched.  The
conversion keeps the first --keep episodes per task (default 200 of 2000) in float32 and drops the raw hdf5 after
conversion, so the kept data is ~10 GB.  Run through SLURM only (slurm/jobs/b1_cpu.sbatch), with python -I:

    --export=ALL,SCRIPT=experiments/b1_fetch.py,ARGS="--robot iiwa"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lifelong_ood import b1  # noqa: E402

FILES = {  # robot -> (dryad file id, bytes, sha256), from https://datadryad.org/api/v2/versions/298028/files
    "iiwa": (3215065, 15907905038, "f41c857157d503867499ba1b6237d8c9a619dcafc22ed2851ebdd8a05067bd77"),
    "jaco": (3215064, 15859597409, "1a013ae4fbc8e772a0c0096d58f387860010917987e155d57c2500dd6988304d"),
    "kinova3": (3215063, 15964813942, "55adda0a73261f393c0540433db2e425bac81493042dc9cf27388c63f013eacb"),
    "panda": (3215062, 15852191273, "970d4e20b3615968830586f942a27bb786429d9bd5cd31728a889981d5267439"),
}
RAW = b1.DATA / "raw"          # the downloaded tarballs: their own directory, never on sys.path
OUT = b1.DATA / "expert"       # converted per-task npz
EP_LEN = 500


def token():
    cred = json.loads((pathlib.Path.home() / ".dryad_api.json").read_text(encoding="utf-8"))
    r = subprocess.run(["curl", "-s", "-X", "POST", "https://datadryad.org/oauth/token", "-H",
                        "Content-Type: application/x-www-form-urlencoded;charset=UTF-8", "-d",
                        f"client_id={cred['client_id']}&client_secret={cred['client_secret']}"
                        "&grant_type=client_credentials"], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)["access_token"]


def download(robot, url=None):
    fid, size, sha = FILES[robot]
    RAW.mkdir(parents=True, exist_ok=True)
    dst = RAW / f"expert-{robot}-offline-comp-data.tar.gz"
    ok = RAW / (dst.name + ".sha256ok")
    if ok.exists():
        return dst
    hdr = []
    if url is None:
        url = f"https://datadryad.org/api/v2/files/{fid}/download"
        hdr = ["-H", f"Authorization: Bearer {token()}"]
    for attempt in range(20):           # resumable: curl -C - continues a partial file after preemption
        if dst.exists() and dst.stat().st_size == size:
            break
        subprocess.run(["curl", "-L", "--fail", "-sS", "-C", "-", "--retry", "5", *hdr, "-o", str(dst), url])
        time.sleep(5)
    assert dst.stat().st_size == size, f"{dst}: {dst.stat().st_size} bytes, want {size}"
    h = hashlib.sha256()
    with open(dst, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    assert h.hexdigest() == sha, f"sha256 mismatch for {dst}"
    ok.write_text(sha + "\n", encoding="utf-8")
    return dst


def convert_member(h5path, cell, keep):
    import h5py
    with h5py.File(h5path, "r") as f:
        n = keep * EP_LEN
        obs = f["observations"][:n].astype(np.float32)
        act = f["actions"][:n].astype(np.float32)
        rew = f["rewards"][:n].astype(np.float32).reshape(-1)
        term = f["terminals"][:n].reshape(-1).astype(bool)
        tout = f["timeouts"][:n].reshape(-1).astype(bool)
        shapes = {k: list(f[k].shape) for k in f.keys()}
    assert obs.shape[1] == 93 and act.shape[1] == 8, shapes
    ends = np.flatnonzero(term | tout)
    assert len(ends) >= keep and np.all(np.diff(np.r_[-1, ends[:keep]]) == EP_LEN), \
        "expert episodes are expected to be 500 steps each"
    E = keep
    obs, act, rew = obs.reshape(E, EP_LEN, -1), act.reshape(E, EP_LEN, -1), rew.reshape(E, EP_LEN)
    success = (rew >= 1.0).any(1)       # Dryad README: success = any state of the episode had reward 1
    d = OUT / b1.cell_name(cell)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "episodes.npz", obs=obs, act=act, rew=rew, success=success)
    meta = dict(cell=list(cell), episodes=E, ep_len=EP_LEN, success_rate=float(success.mean()),
                raw_shapes=shapes, terminals_any=bool(term.any()))
    (d / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return meta


def extract_and_convert(robot, tar_path, keep):
    """Stream the tarball once; each task's hdf5 is written to a scratch dir, converted, and deleted."""
    tmp = RAW / f"tmp_{robot}"
    tmp.mkdir(parents=True, exist_ok=True)
    done = []
    with tarfile.open(tar_path, "r|gz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith("data.hdf5"):
                continue
            task = pathlib.PurePosixPath(m.name).parent.name.replace(".hdf5", "")
            cell = tuple(task.split("_"))
            assert len(cell) == 4 and cell[0] in b1.ROBOTS, m.name
            if (OUT / task / "meta.json").exists():
                continue
            safe = tmp / "data.hdf5"            # never use the member's own path (no traversal)
            with tf.extractfile(m) as src, open(safe, "wb") as dst:
                while blk := src.read(1 << 24):
                    dst.write(blk)
            meta = convert_member(safe, cell, keep)
            safe.unlink()
            done.append(meta)
            print(task, meta["success_rate"], flush=True)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True, choices=sorted(FILES))
    ap.add_argument("--keep", type=int, default=200)
    ap.add_argument("--url-file", default=None)
    a = ap.parse_args()
    url = None
    if a.url_file:
        for line in open(a.url_file, encoding="utf-8"):
            r, u = line.split(maxsplit=1)
            if r == a.robot:
                url = u.strip()
    tar = download(a.robot, url)
    metas = extract_and_convert(a.robot, tar, a.keep)
    print(f"{a.robot}: converted {len(metas)} tasks")


if __name__ == "__main__":
    main()
