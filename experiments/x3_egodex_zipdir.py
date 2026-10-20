"""X3 (EgoDex): list the training zips' task folders by reading only their central directories over HTTP range requests.

For each of part1..part5.zip and extra.zip at ml-site.cdn-apple.com, zipfile is opened on a ranged, block-cached file object, so
only the end-of-central-directory record and the central directory are fetched (a few MB per 300 GB zip).  Per task folder it
records: the zip, the number of episodes (hdf5 files), the number of mp4s, the total compressed bytes, and the byte span
[min local-header offset, max end] so that a later fetch can read each task's members by range.  Per-file offsets go to a
separate (untracked) JSON for the fetcher.  No member data is downloaded.

Run on a compute node (SLURM), never the login node:
    python -I experiments/x3_egodex_zipdir.py --out results/robot/egodex
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import os
import time
import zipfile

import requests

BASE = "https://ml-site.cdn-apple.com/datasets/egodex/"
ZIPS = ["part1.zip", "part2.zip", "part3.zip", "part4.zip", "part5.zip", "extra.zip"]
BLOCK = 1 << 20


class RangedFile(io.RawIOBase):
    """A seekable read-only file over HTTP byte ranges, cached in 1 MB blocks."""

    def __init__(self, url: str):
        self.url, self.pos, self.cache, self.fetched = url, 0, {}, 0
        r = requests.head(url, timeout=60, allow_redirects=True)
        r.raise_for_status()
        self.size = int(r.headers["Content-Length"])
        assert r.headers.get("Accept-Ranges", "bytes") == "bytes", r.headers

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _block(self, b):
        if b not in self.cache:
            lo, hi = b * BLOCK, min(self.size, (b + 1) * BLOCK) - 1
            for attempt in range(5):
                try:
                    r = requests.get(self.url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=120)
                    assert r.status_code == 206, r.status_code
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(5)
            self.cache[b] = r.content
            self.fetched += len(r.content)
        return self.cache[b]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while n > 0:
            b, o = divmod(self.pos, BLOCK)
            chunk = self._block(b)[o:o + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, buf):
        d = self.read(len(buf))
        buf[:len(d)] = d
        return len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/robot/egodex")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tasks, files, zips = {}, {}, {}
    for z in ZIPS:
        f = RangedFile(BASE + z)
        zf = zipfile.ZipFile(f)
        infos = zf.infolist()
        per = collections.defaultdict(lambda: dict(zip=z, episodes=0, mp4=0, other=0, bytes=0, lo=None, hi=0, prefix=None))
        flist = []
        for i in infos:
            if i.is_dir():
                continue
            parts = i.filename.strip("/").split("/")
            task = parts[-2] if len(parts) >= 2 else "<root>"
            t = per[task]
            t["prefix"] = "/".join(parts[:-1])
            if i.filename.endswith(".hdf5"):
                t["episodes"] += 1
            elif i.filename.endswith(".mp4"):
                t["mp4"] += 1
            else:
                t["other"] += 1
            t["bytes"] += i.compress_size
            # local header = 30 + name + extra; extra unknown without reading it, so take 30 + name + 64 as a safe upper end
            end = i.header_offset + 30 + len(i.filename.encode()) + 64 + i.compress_size
            t["lo"] = i.header_offset if t["lo"] is None else min(t["lo"], i.header_offset)
            t["hi"] = max(t["hi"], end)
            flist.append([i.filename, i.header_offset, i.compress_size, i.file_size, i.compress_type])
        for k, v in per.items():
            key = f"{z}:{k}" if k in tasks else k
            if k in tasks:
                print("DUPLICATE task across zips:", k, tasks[k]["zip"], z)
            tasks[key] = dict(v)
        files[z] = flist
        zips[z] = dict(size=f.size, members=len(infos), tasks=len(per), fetched_bytes=f.fetched)
        print(f"{z}: size {f.size/1e9:.1f} GB, {len(infos)} members, {len(per)} task folders, fetched {f.fetched/1e6:.1f} MB", flush=True)
    json.dump(dict(zips=zips, tasks=tasks), open(os.path.join(a.out, "train_zip_tasks.json"), "w"), indent=1)
    json.dump(files, open(os.path.join(a.out, "train_zip_files.json"), "w"))
    print("tasks:", len(tasks), "episodes:", sum(t["episodes"] for t in tasks.values()),
          "GB:", round(sum(t["bytes"] for t in tasks.values()) / 1e9, 1))


if __name__ == "__main__":
    main()
