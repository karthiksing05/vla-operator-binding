"""Plan X / X1: extract CALVIN's task_ABC_D.zip (555 GB) without a local copy of the zip, resumable, as a SLURM CPU job.

    python -I experiments/x1_calvin_fetch.py index <dest dir>               read the central directory once -> <dest>/_index.pkl
    python -I experiments/x1_calvin_fetch.py fetch <dest dir> <k> <n>       worker k of n: contiguous byte ranges of ~64 MB, each
                                                                            split into its members (stored or deflated), files
                                                                            already present at full size are skipped
"""
import os
import pickle
import struct
import sys
import time
import urllib.request
import zlib

URL = "http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip"
SIZE = 555309812705                                                        # Content-Length of the zip
CHUNK = 64 << 20


def index(dest):
    from remotezip import RemoteZip
    with RemoteZip(URL) as z:
        rows = sorted((i.header_offset, i.filename, i.compress_size, i.file_size, i.compress_type)
                      for i in z.infolist() if not i.is_dir())
    os.makedirs(dest, exist_ok=True)
    pickle.dump(rows, open(os.path.join(dest, "_index.pkl"), "wb"))
    print(f"[index] {len(rows)} members, {sum(r[2] for r in rows) / 1e9:.1f} GB compressed", flush=True)


def get(a, b):
    for attempt in range(6):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={a}-{b - 1}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) == b - a:
                return data
        except Exception as e:                                              # transient HTTP errors: back off and retry
            print(f"[fetch] range {a}-{b} attempt {attempt}: {e}", flush=True)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"range {a}-{b} failed")


def fetch(dest, k, n):
    k, n = int(k), int(n)
    rows = pickle.load(open(os.path.join(dest, "_index.pkl"), "rb"))
    groups, cur = [], []                                                    # contiguous groups of members, ~CHUNK each
    for r in rows:
        if cur and (r[0] - cur[0][0] > CHUNK):
            groups.append(cur)
            cur = []
        cur.append(r)
    if cur:
        groups.append(cur)
    mine = groups[k::n]
    t0, done, skipped = time.time(), 0, 0
    for gi, g in enumerate(mine):
        todo = [r for r in g if not (os.path.exists(os.path.join(dest, r[1])) and os.path.getsize(os.path.join(dest, r[1])) == r[3])]
        if not todo:
            skipped += len(g)
            continue
        a = g[0][0]
        b = min(SIZE, g[-1][0] + 30 + 1024 + g[-1][2] + 65536)              # last member's local header, name, extra, data
        buf = get(a, b)
        for off, name, csize, usize, ctype in todo:
            p = off - a
            sig, *_ = struct.unpack("<I", buf[p:p + 4])
            assert sig == 0x04034B50, (name, hex(sig))
            nlen, xlen = struct.unpack("<HH", buf[p + 26:p + 30])
            s = p + 30 + nlen + xlen
            data = buf[s:s + csize]
            if ctype == 8:
                data = zlib.decompress(data, -15)
            assert len(data) == usize, (name, len(data), usize)
            out = os.path.join(dest, name)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            tmp = out + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, out)
            done += 1
        if gi % 50 == 0:
            print(f"[fetch {k}] group {gi + 1}/{len(mine)}, {done} written, {skipped} present, {time.time() - t0:.0f}s", flush=True)
    print(f"[fetch {k}] DONE {done} written, {skipped} present, {len(mine)} groups, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "index":
        index(sys.argv[2])
    else:
        fetch(*sys.argv[2:5])
