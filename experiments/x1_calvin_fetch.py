"""Plan X / X1: extract CALVIN's task_ABC_D.zip member by member over HTTP range requests (no 555 GB zip copy on disk), resumable.

    python -I experiments/x1_calvin_fetch.py <dest dir> <worker k> <of n>      (one process per worker; skips existing files)
"""
import os
import sys
import time

from remotezip import RemoteZip

URL = "http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip"


def main(dest, k, n):
    k, n = int(k), int(n)
    done = err = 0
    t0 = time.time()
    with RemoteZip(URL) as z:
        infos = [i for j, i in enumerate(z.infolist()) if j % n == k and not i.is_dir()]
        for i in infos:
            p = os.path.join(dest, i.filename)
            if os.path.exists(p) and os.path.getsize(p) == i.file_size:
                continue
            for attempt in range(5):
                try:
                    z.extract(i, dest)
                    break
                except Exception as e:                                      # transient HTTP errors: back off and retry
                    if attempt == 4:
                        err += 1
                        print(f"[fetch {k}] FAILED {i.filename}: {e}", flush=True)
                    time.sleep(2 ** attempt)
            done += 1
            if done % 2000 == 0:
                print(f"[fetch {k}] {done} of {len(infos)} in {time.time() - t0:.0f}s", flush=True)
    print(f"[fetch {k}] DONE {done} extracted, {err} failed, {len(infos)} assigned, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
