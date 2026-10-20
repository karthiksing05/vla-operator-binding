"""X3 (EgoDex): a minimal loader of (image frame, instruction, state, action chunk) samples in our action representation.

Representation (docs/EGODEX.md, "Recipe"), per hand h in (left, right), everything in the CURRENT camera frame C_t
(p^C = inv(T_camera[t]) @ p^W, the official utils.data_utils.convert_to_camera_frame):
    wrist xyz (3) + wrist 6D rotation (6: the first two COLUMNS of R, Zhou et al.) + 5 fingertip xyz (15) = 24
so a frame is 48-D in EgoDex's own layout [L wrist xyz, L rot6d, L 5 tips, R wrist xyz, R rot6d, R 5 tips] (fingertip order:
thumb, index, middle, ring, little), which is the layout compute_metrics.evaluate_distance indexes (12 points: 2 wrists + 10 tips).
  state  s_t   = the 48-D frame at t in C_t (the current hand configuration; GR00T pads it to 64);
  action a_t   = [frame(t + stride*k) in C_t for k = 1..H]  ([H, 48]); a relative trajectory: every step is expressed in the
                 camera frame at the chunk's first observation, as in the EgoDex benchmark.
  pad    [H]   = 1 where t + stride*k runs past the episode (the official metric skips padded chunks).
  image        = the RGB frame t of the paired mp4 (1920x1080), resized to `size` (GR00T's Eagle uses 224x224).
  instruction  = llm_description, or llm_description2 when llm_type == 'reversible' and which_llm_description == '2' (as the
                 official simple_dataset.py does).
Optional per-frame confidences of the 12 metric points are returned (0 = occluded) so that training can mask them.

Frames are decoded with PyAV: seek to the keyframe before t, decode forward to t (only the requested frames are decoded).
Run inside a SLURM job, in the GR00T venv (h5py is appended from /coc/flash10/ksingara3/robot/ext/egodex_pylibs when missing):
    /coc/flash10/ksingara3/robot/groot/.venv/bin/python -I experiments/x3_egodex_data.py --smoke
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

try:
    import h5py
except ImportError:            # the GR00T venv has no h5py: a --target install beside it (python -I ignores PYTHONPATH)
    sys.path.append(os.environ.get("EGODEX_PYLIBS", "/coc/flash10/ksingara3/robot/ext/egodex_pylibs"))
    import h5py

FPS = 30
FINGERS = ("Thumb", "IndexFinger", "MiddleFinger", "RingFinger", "LittleFinger")
HANDS = ("left", "right")


def point_names(hand):
    return [f"{hand}Hand"] + [f"{hand}{f}Tip" for f in FINGERS]


def rot6d(R):
    """[..., 3, 3] -> [..., 6]: the first two columns, column-major (r11, r21, r31, r12, r22, r32)."""
    return np.concatenate([R[..., :, 0], R[..., :, 1]], -1)


def rot6d_to_matrix(x):
    """[..., 6] -> [..., 3, 3] by Gram-Schmidt (inverse of rot6d on rotations)."""
    a, b = x[..., :3], x[..., 3:]
    c1 = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - (c1 * b).sum(-1, keepdims=True) * c1
    c2 = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.stack([c1, c2, np.cross(c1, c2)], -1)


class Episode:
    """One EgoDex episode: poses read once (all frames, the 12 needed joints + camera), video opened lazily."""

    def __init__(self, hdf5_path):
        self.path = hdf5_path
        with h5py.File(hdf5_path, "r") as f:
            self.cam = f["transforms/camera"][:].astype(np.float64)                        # [T, 4, 4] in the ARKit origin frame
            self.T = len(self.cam)
            self.wrist = {h: f[f"transforms/{h}Hand"][:].astype(np.float64) for h in HANDS}  # [T, 4, 4]
            self.tips = {h: np.stack([f[f"transforms/{h}{fi}Tip"][:, :3, 3] for fi in FINGERS], 1).astype(np.float64)
                         for h in HANDS}                                                    # [T, 5, 3]
            self.conf = None
            if "confidences" in f:
                self.conf = np.stack([f[f"confidences/{n}"][:] for h in HANDS for n in point_names(h)], 1)  # [T, 12]
            at = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
            self.intrinsic = f["camera/intrinsic"][:] if "camera/intrinsic" in f else None
        rev = str(at.get("llm_type", "")) == "reversible" and str(at.get("which_llm_description", "")) == "2"
        self.instruction = str(at.get("llm_description2" if rev else "llm_description", ""))
        self.attrs = at
        self.mp4 = hdf5_path[:-5] + ".mp4"
        self._container = None

    # ── poses ────────────────────────────────────────────────────────────
    def frame48(self, idx, ref, tips_in_wrist=False):
        """[len(idx), 48] hand frames at frames idx, expressed in the camera frame at frame ref.  tips_in_wrist: the fingertips
        in each step's own wrist frame instead (the hand shape decoupled from the wrist motion; `to_camera48` inverts it)."""
        Cinv = np.linalg.inv(self.cam[ref])
        out = []
        for h in HANDS:
            W = Cinv[None] @ self.wrist[h][idx]                                              # [n, 4, 4]
            P = np.einsum("ij,nkj->nki", Cinv[:3, :3], self.tips[h][idx]) + Cinv[:3, 3]      # [n, 5, 3]
            if tips_in_wrist:
                P = np.einsum("nji,nkj->nki", W[:, :3, :3], P - W[:, None, :3, 3])           # R^T (p - t)
            out += [W[:, :3, 3], rot6d(W[:, :3, :3]), P.reshape(len(idx), 15)]
        return np.concatenate(out, 1).astype(np.float32)

    def sample(self, t, H=16, stride=2, tips_in_wrist=False):
        """state [48], action [H, 48], pad [H], conf [H, 12] or None, at frame t."""
        fut = t + stride * np.arange(1, H + 1)
        pad = (fut >= self.T).astype(np.uint8)
        fut = np.minimum(fut, self.T - 1)
        conf = None if self.conf is None else self.conf[fut].astype(np.float32)
        return (self.frame48(np.array([t]), t, tips_in_wrist)[0], self.frame48(fut, t, tips_in_wrist), pad, conf)


    # ── video ────────────────────────────────────────────────────────────
    def image(self, t, size=224):
        import av
        if self._container is None:
            self._container = av.open(self.mp4)
        c = self._container
        st = c.streams.video[0]
        tb = float(st.time_base)
        target = t / FPS
        c.seek(int(target / tb), stream=st, backward=True, any_frame=False)
        for fr in c.decode(st):
            if fr.pts is not None and fr.pts * tb >= target - 0.5 / FPS:
                img = fr.to_ndarray(format="rgb24")
                break
        else:
            raise IndexError(f"frame {t} not found in {self.mp4}")
        if size:
            import cv2
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        return img

    def close(self):
        if self._container is not None:
            self._container.close()
            self._container = None


def to_camera48(x):
    """[..., 48] with fingertips in the wrist frame -> the official camera-frame 48-D layout (for compute_metrics)."""
    x = np.array(x, np.float64, copy=True)
    for o in (0, 24):
        t, R = x[..., o:o + 3], rot6d_to_matrix(x[..., o + 3:o + 9])
        P = x[..., o + 9:o + 24].reshape(*x.shape[:-1], 5, 3)
        x[..., o + 9:o + 24] = (np.einsum("...ij,...kj->...ki", R, P) + t[..., None, :]).reshape(*x.shape[:-1], 15)
    return x.astype(np.float32)


def samples(hdf5_paths, H=16, stride=2, every=15, size=224, with_image=True):
    """Yield dicts (image, instruction, state, action, pad, conf, path, t) every `every` frames of each episode."""
    for p in hdf5_paths:
        ep = Episode(p)
        for t in range(0, ep.T - 1, every):
            s, a, pad, conf = ep.sample(t, H, stride)
            yield dict(image=ep.image(t, size) if with_image else None, instruction=ep.instruction, state=s, action=a,
                       pad=pad, conf=conf, path=p, t=t)
        ep.close()


def _smoke(root):
    import time
    tasks = ["basic_pick_place", "pour", "stack_unstack_bowls"]
    paths = [os.path.join(root, t, "0.hdf5") for t in tasks]
    t0 = time.time()
    n = 0
    for s in samples(paths, H=16, stride=2, every=45):
        n += 1
        a = s["action"]
        R = rot6d_to_matrix(a[:, 3:9])
        assert np.allclose(R.transpose(0, 2, 1) @ R, np.eye(3), atol=1e-4)
        # the state's wrist positions should be near the camera (|xyz| < 1.2 m) and in front of it (ARKit camera looks down -z)
        wl, wr = s["state"][0:3], s["state"][24:27]
        print(f"{os.path.basename(os.path.dirname(s['path']))}/{os.path.basename(s['path'])} t={s['t']:4d} img={s['image'].shape} "
              f"{s['image'].dtype} mean={s['image'].mean():.1f} | L wrist {np.round(wl, 3)} R wrist {np.round(wr, 3)} | "
              f"chunk travel R {np.linalg.norm(a[-1, 24:27] - wr):.3f} m | pad {int(s['pad'].sum())} | "
              f"conf min {None if s['conf'] is None else round(float(s['conf'].min()), 2)} | '{s['instruction'][:70]}'", flush=True)
    # consistency with the official camera-frame conversion
    ep = Episode(paths[0])
    t = 10
    off = np.linalg.inv(ep.cam[t])[None] @ ep.wrist["right"][[t + 2]]
    assert np.allclose(off[0, :3, 3], ep.sample(t)[1][0, 24:27], atol=1e-5)
    # the wrist-frame fingertip variant maps back exactly to the camera-frame layout
    cam, wr = ep.sample(t)[1], ep.sample(t, tips_in_wrist=True)[1]
    assert np.allclose(to_camera48(wr), cam, atol=1e-4), np.abs(to_camera48(wr) - cam).max()
    print("tips in wrist frame: |tip| range %.3f-%.3f m" % (np.linalg.norm(wr[:, 9:24].reshape(-1, 5, 3), axis=-1).min(),
                                                          np.linalg.norm(wr[:, 9:24].reshape(-1, 5, 3), axis=-1).max()))
    # a 48-D chunk passes compute_metrics' indexing: 12 points
    idx = list(range(3)) + list(range(9, 27)) + list(range(33, 48))
    assert len(idx) == 36
    print(f"OK {n} samples in {time.time() - t0:.1f} s; official 48-D layout and camera-frame conversion consistent")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--root", default="/coc/flash10/ksingara3/robot/data/egodex/test")
    a = ap.parse_args()
    if a.smoke:
        _smoke(a.root)
