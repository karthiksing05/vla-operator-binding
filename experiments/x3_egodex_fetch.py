"""X3 (EgoDex) stage 2a/2b: fetch planned training episodes by byte range, and pre-extract frames + poses (resumable).

FETCH (--shard i): for each manifest row (x3_egodex_manifest.py), ONE ranged GET covers the episode's adjacent hdf5 and mp4
members; their local headers are parsed from the fetched bytes, each member is inflated (raw deflate) and checked against the
central directory's uncompressed size and the CRC-32 (local header, or the data descriptor when bit 3 is set), then written
atomically to   <data>/train_subset/<folder>/<ep>.hdf5 / .mp4.
EXTRACT: for each episode with both files, one compact file  <data>/derived/<split>/<folder>/<ep>.npz :
  jpg   uint8 [sum bytes]  the JPEG (q 90) of every STRIDE-th frame (0, 2, 4, ... at 30 Hz), 224x224, squashed from 1920x1080
                          (as x3_egodex_data.Episode.image), concatenated;   off int64 [F+1] their byte offsets;  fidx int32 [F]
  cam   f32 [T, 4, 4]      camera pose in the ARKit origin frame;  wrist f32 [2, T, 4, 4] (left, right);  tips f32 [2, T, 5, 3]
                          (thumb, index, middle, ring, little);  conf f32 [T, 12] (12 metric points; NaN when the file has none)
  instruction, T, n_video_frames
The 48-D state / action chunks are computed from these on the fly (x3_egodex_data.frame48 semantics; x3_egodex_train.py).
Every output is written to a temp name and renamed, so a preempted job resumes where it stopped (done = the file exists).

    python -I experiments/x3_egodex_fetch.py --shard 3 --nshards 8          # fetch + extract a training shard
    python -I experiments/x3_egodex_fetch.py --test --shard 0 --nshards 4   # extract the local test set
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import os
import struct
import sys
import threading
import time
import zlib

import numpy as np
import requests

try:
    import h5py
except ImportError:
    sys.path.append(os.environ.get("EGODEX_PYLIBS", "/coc/flash10/ksingara3/robot/ext/egodex_pylibs"))
    import h5py

BASE = "https://ml-site.cdn-apple.com/datasets/egodex/"
DATA = "/coc/flash10/ksingara3/robot/data/egodex"
TEST = os.path.join(DATA, "test")
OUT = "results/robot/egodex"
STRIDE, SIZE, QUALITY = 2, 224, 90
FINGERS = ("Thumb", "IndexFinger", "MiddleFinger", "RingFinger", "LittleFinger")
HANDS = ("left", "right")
_tls = threading.local()


def _session():
    if not hasattr(_tls, "s"):
        _tls.s = requests.Session()
    return _tls.s


def get_range(url, lo, hi):
    for a in range(8):
        try:
            r = _session().get(url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=300)
            if r.status_code == 206:
                return r.content
            err = r.status_code
        except Exception as e:
            err = repr(e)
        time.sleep(3 + 5 * a)
    raise RuntimeError(f"range failed {url} {lo}-{hi}: {err}")


def _member(buf, base, info):
    """Inflate one member from buf (bytes starting at absolute offset base); info = (name, off, cs, us)."""
    name, off, cs, us = info
    o = off - base
    sig, ver, flag, meth, _, _, crc, _, _, nlen, xlen = struct.unpack("<IHHHHHIIIHH", buf[o:o + 30])
    assert sig == 0x04034B50, (name, hex(sig))
    d0 = o + 30 + nlen + xlen
    raw = buf[d0:d0 + cs]
    assert len(raw) == cs, (name, len(raw), cs)
    data = zlib.decompressobj(-15).decompress(raw) if meth == 8 else raw
    assert len(data) == us, (name, len(data), us)
    if flag & 8:                                            # CRC in the data descriptor
        dd = buf[d0 + cs:d0 + cs + 16]
        crc = struct.unpack("<I", dd[4:8] if dd[:4] == b"PK\x07\x08" else dd[:4])[0]
    assert (zlib.crc32(data) & 0xFFFFFFFF) == crc, (name, "crc")
    return data


def _write(path, data):
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def fetch(row, root):
    d = os.path.join(root, row["folder"])
    os.makedirs(d, exist_ok=True)
    ph, pm = os.path.join(d, row["ep"] + ".hdf5"), os.path.join(d, row["ep"] + ".mp4")
    if os.path.exists(ph) and os.path.exists(pm):
        return 0
    buf = get_range(BASE + row["zip"], row["lo"], row["hi"])
    _write(ph, _member(buf, row["lo"], row["hdf5"]))
    _write(pm, _member(buf, row["lo"], row["mp4"]))
    return len(buf)


def extract(ph, pm, out):
    if os.path.exists(out):
        return 0
    import av
    import cv2
    with h5py.File(ph, "r") as f:
        cam = f["transforms/camera"][:].astype(np.float32)
        T = len(cam)
        wrist = np.stack([f[f"transforms/{h}Hand"][:] for h in HANDS]).astype(np.float32)
        tips = np.stack([np.stack([f[f"transforms/{h}{fi}Tip"][:, :3, 3] for fi in FINGERS], 1) for h in HANDS]).astype(np.float32)
        names = [n for h in HANDS for n in [f"{h}Hand"] + [f"{h}{fi}Tip" for fi in FINGERS]]
        conf = (np.stack([f[f"confidences/{n}"][:] for n in names], 1).astype(np.float32) if "confidences" in f
                else np.full((T, 12), np.nan, np.float32))
        at = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
    rev = str(at.get("llm_type", "")) == "reversible" and str(at.get("which_llm_description", "")) == "2"
    instr = str(at.get("llm_description2" if rev else "llm_description", ""))
    jpgs, fidx, n = [], [], 0
    with av.open(pm) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        for i, fr in enumerate(c.decode(st)):
            n += 1
            if i % STRIDE or i >= T:
                continue
            img = fr.to_ndarray(format="rgb24")
            img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            ok, enc = cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
            assert ok
            jpgs.append(enc.tobytes())
            fidx.append(i)
    off = np.cumsum([0] + [len(j) for j in jpgs]).astype(np.int64)
    bio = io.BytesIO()
    np.savez(bio, jpg=np.frombuffer(b"".join(jpgs), np.uint8), off=off, fidx=np.array(fidx, np.int32), cam=cam, wrist=wrist,
             tips=tips, conf=conf, instruction=np.array(instr), T=np.array(T), n_video_frames=np.array(n))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    _write(out, bio.getvalue())
    return len(fidx)


def summarize(out, record):
    """Every derived file: count, frames, video-vs-pose frame agreement; the fetch stats; one ledger row (x3_egodex_fetch)."""
    import glob
    res = {}
    for split in ("train", "test"):
        files = sorted(glob.glob(os.path.join(DATA, "derived", split, "*", "*.npz")))
        n_fr, mism, short, nbytes, folders = 0, [], 0, 0, set()
        for p in files:
            z = np.load(p)
            T, nv, F = int(z["T"]), int(z["n_video_frames"]), len(z["fidx"])
            n_fr += F
            nbytes += os.path.getsize(p)
            folders.add(os.path.basename(os.path.dirname(p)))
            if nv != T:
                mism.append((p, T, nv))
            if F < (T + 1) // 2:
                short += 1
        res[split] = dict(episodes=len(files), folders=len(folders), frames_224=n_fr, derived_gb=round(nbytes / 1e9, 2),
                          video_pose_len_mismatch=len(mism), mismatch_examples=mism[:5], fewer_frames_than_poses=short)
    man = [r for f in glob.glob(os.path.join(out, "fetch", "shard*.json")) for r in json.load(open(f))]
    raw = sum(os.path.getsize(os.path.join(DATA, "train_subset", r["folder"], r["ep"] + ext))
              for r in man for ext in (".hdf5", ".mp4") if os.path.exists(os.path.join(DATA, "train_subset", r["folder"], r["ep"] + ext)))
    errs = [e for f in glob.glob(os.path.join(out, "fetch", "stats_*.json")) for e in json.load(open(f))["errors"]]
    res.update(manifest_episodes=len(man), raw_train_gb=round(raw / 1e9, 2), errors=len(errs), error_examples=errs[:5])
    json.dump(res, open(os.path.join(out, "fetch", "summary.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    if record:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from lifelong_ood import ledger
        p = ledger.record("x3_egodex_fetch", bench="egodex", tag="train_subset_plan", seed=0,
                          metrics={k: v for k, v in res.items() if k != "error_examples"},
                          bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                          config=dict(stride=STRIDE, size=SIZE, jpeg_quality=QUALITY, manifest="results/robot/egodex/fetch/shard*.json"),
                          script="experiments/x3_egodex_fetch.py", command=" ".join(sys.argv),
                          artifacts=dict(raw=os.path.join(DATA, "train_subset"), derived=os.path.join(DATA, "derived")))
        print("ledger:", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--nshards", type=int, default=8)
    ap.add_argument("--test", action="store_true", help="extract the local test set instead of fetching")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--max-minutes", type=float, default=225)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.summarize:
        summarize(a.out, a.record)
        return
    t0 = time.time()
    if a.test:
        folders = sorted(os.listdir(TEST))[a.shard::a.nshards]
        jobs = [(os.path.join(TEST, fo, n), os.path.join(TEST, fo, n[:-5] + ".mp4"), os.path.join(DATA, "derived", "test", fo, n[:-5] + ".npz"))
                for fo in folders for n in sorted(os.listdir(os.path.join(TEST, fo))) if n.endswith(".hdf5")]
        rows = None
    else:
        rows = json.load(open(os.path.join(a.out, "fetch", f"shard{a.shard}.json")))
        root = os.path.join(DATA, "train_subset")
    stats = dict(fetched_bytes=0, fetched=0, extracted=0, frames=0, errors=[])
    lock = threading.Lock()

    def work(item):
        if time.time() - t0 > a.max_minutes * 60:
            return
        try:
            if rows is not None:
                nb = fetch(item, root)
                ph = os.path.join(root, item["folder"], item["ep"] + ".hdf5")
                out = os.path.join(DATA, "derived", "train", item["folder"], item["ep"] + ".npz")
                nf = extract(ph, ph[:-5] + ".mp4", out)
            else:
                nb, (ph, pm, out) = 0, item
                nf = extract(ph, pm, out)
            with lock:
                stats["fetched_bytes"] += nb
                stats["fetched"] += int(nb > 0)
                stats["extracted"] += int(nf > 0)
                stats["frames"] += nf
        except Exception as e:
            with lock:
                stats["errors"].append([str(item if rows is None else (item["folder"], item["ep"])), repr(e)[:300]])

    items = rows if rows is not None else jobs
    with cf.ThreadPoolExecutor(a.threads) as ex:
        for i, _ in enumerate(ex.map(work, items)):
            if i % 200 == 0:
                print(f"{i}/{len(items)} {time.time() - t0:.0f}s {stats['fetched_bytes'] / 1e9:.1f} GB fetched, "
                      f"{stats['extracted']} extracted, {stats['frames']} frames, {len(stats['errors'])} errors", flush=True)
    done = time.time() - t0 <= a.max_minutes * 60
    tag = "test" if a.test else "train"
    stats.update(items=len(items), seconds=round(time.time() - t0), complete_pass=done)
    json.dump(stats, open(os.path.join(a.out, "fetch", f"stats_{tag}{a.shard}_{int(t0)}.json"), "w"), indent=1)
    print("DONE" if done else "TIME LIMIT", {k: v for k, v in stats.items() if k != "errors"}, "errors:", stats["errors"][:5])


if __name__ == "__main__":
    main()
