"""PROBE R6 (EXPLORATORY closed-loop smoke; docs/PLAN_R6.md): the serving side.  Opt-in mode of experiments/serve_guided.py, default off
(the mode string must start with "r6:"):

    r6:<spec.json>?arm=<off|both|goal|orc>&w=<w>&log=<queries.jsonl>

The frozen pi05_l90c@5000 is served exactly as zs (MODE ours: GuidedPolicy with no hooks, plain Euler 10 steps, the server's seed + call
count as the noise seed) on every query that is not guided; those queries go through GuidedPolicy.infer itself.  Two guided phases,
both by R4-bind's x0-projection on the first 5 of the 10 Euler steps (rb4_bind.Guided.sample, early schedule, w):
  approach  (arms both / orc) while the gripper is open and before the first grasp (r5s_guide.Tracker): R5s's primary configuration,
            d* = J_D4 (c_obj - h) + b_D4 (D4's approach bank, the named object's type > category > role);
  carry     (arms both / goal / orc) from the first grasp (closed on 3 consecutive queries) while the chunk's gripper stays closed, until
            the first open chunk after the grasp (the release; guidance is off from then on): the predicted release point
            r* = h + J_D5 (c_goal - h) + b_D5 (D5's release bank, the target body's type > category > role), and the chunk's xy
            displacement is steered along r* - h, keeping its own length (at least 3 cm) and never past r*:
            d* = min(|r* - h|, max(|D|, 3 cm)) * unit(r* - h).
Targets: c_obj / c_goal = the vision anchor (arms both / goal): the argmax agentview SigLIP token of the named object's / the target
type's training prototype (results/rb4/protos.npz), back-projected through the LIVE agentview depth the client sends (R5S_GEOM=1,
meta/tok_xyz); the oracle (arm orc): the simulator positions (meta/obj_pos for the object; meta/tgt_xyz, the client's opt-in R6_TGT=1,
for the target body).  `off` never guides (the serving smoke).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

HIT_M = 0.05
DIR_MIN = 0.03
EARLY = "early"


def parse(mode):
    body, q = (mode[len("r6:"):].split("?", 1) + [""])[:2]
    kv = dict(x.split("=", 1) for x in q.split("&") if x)
    return body, kv


class Phase:
    """r5s_guide.Tracker's grasp rule plus the release: `closed` = the chunk's 5th commanded gripper > 0; grasped = closed on 3
    consecutive queries; released = the first open chunk after the grasp."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.run, self.closed, self.grasped, self.released = 0, False, False, False

    def observe(self, actions):
        a = np.asarray(actions)
        self.closed = bool(a[min(4, len(a) - 1), 6] > 0)
        if self.closed:
            self.run += 1
            if self.run >= 3:
                self.grasped = True
        else:
            self.run = 0
            if self.grasped:
                self.released = True

    @property
    def approach(self):
        return (not self.closed) and (not self.grasped)

    @property
    def carry(self):
        return self.grasped and self.closed and not self.released


def make_policy(bk, mode, num_steps=10, seed=0):
    import torch
    import rb4_bind as RB
    from lifelong_ood.policy.serve import GuidedPolicy

    class R6Guided(RB.Guided):
        def dstar(self, D, cfg, h, c, op):
            h2 = np.asarray(h, np.float64)[:2]
            c2 = np.asarray(c, np.float64)[:2]
            v = np.asarray(op["J"], np.float64) @ (c2 - h2) + np.asarray(op["b"], np.float64)
            if op.get("kind") == "goal":                        # steer along r* - h, own length (>= 3 cm), never past r*
                tv = torch.as_tensor(v, dtype=torch.float32, device=D.device)[None]
                dist = tv.norm(dim=1, keepdim=True)
                mag = torch.minimum(dist, torch.clamp(D.norm(dim=1, keepdim=True), min=DIR_MIN))
                return mag * tv / dist.clamp(min=1e-9)
            return torch.as_tensor(v, dtype=torch.float32, device=D.device)[None].expand_as(D)

    class R6Policy(GuidedPolicy):
        def __init__(self):
            super().__init__(bk, hooks=None, num_steps=num_steps, seed=seed)
            path, kv = parse(mode)
            self.spec = json.load(open(path, encoding="utf-8"))
            self.arm = kv.get("arm", "off")
            assert self.arm in ("off", "both", "goal", "orc"), self.arm
            self.cfg = dict(target="bank", sched=EARLY, w=float(kv.get("w", 1.0)))
            self.gd = R6Guided(bk)
            sc = np.asarray(self.spec["scale"], np.float64)
            self.gd.scale = sc
            self.gd.t_scale = torch.as_tensor(sc[:2], dtype=torch.float32, device=bk.device)
            pz = np.load(os.path.join(ROOT, "results", "rb4", "protos.npz"), allow_pickle=True)
            self.PR = torch.as_tensor(pz["sig"], device=bk.device)
            self.named = self.spec["named"]
            # D5's release operator per goal atom, its offset in the target frame the arm uses (oracle region site / vision anchor)
            self.goal = self.spec["goal_orc"] if self.arm == "orc" else self.spec["goal_vis"]
            self.ph = Phase()
            self.qi = 0
            self.logf = None
            if kv.get("log"):
                os.makedirs(os.path.dirname(kv["log"]), exist_ok=True)
                self.logf = open(kv["log"], "w", encoding="utf-8")
            logging.info(f"r6: arm {self.arm} w {self.cfg['w']} task {self.spec['task']} named "
                         f"{[(q['name'], q['level']) for q in self.named]} goal {[(q['body'], q['type'], q['level']) for q in self.goal]}")

        def _log(self, rec):
            if self.logf is not None:
                self.logf.write(json.dumps(rec) + "\n")
                self.logf.flush()

        def _siglip(self, observation):
            m = bk.model
            images, img_masks, lang_tokens, lang_masks, _ = m._preprocess_observation(observation, train=False)
            embs, _, _ = m.embed_prefix(images, img_masks, lang_tokens, lang_masks)
            return embs[0, :256].float()

        def _anchor(self, observation, meta, pids):
            with torch.no_grad():
                S = torch.nn.functional.normalize(self._siglip(observation), dim=-1)
                sim = self.PR[pids] @ S.T
                best, arg = sim.max(0)
                tok = int(torch.argmax(best))
                pid = pids[int(arg[tok])]
            xyz = np.asarray(meta["tok_xyz"], np.float64).reshape(256, 3)[tok]
            return xyz, pid, dict(tok=tok, xyz=[round(float(v), 4) for v in xyz], sim=round(float(best[tok]), 4), proto=pid)

        def _obj_target(self, observation, meta, h):
            if self.arm == "orc":
                if "obj_names" not in meta:
                    return None, None, dict(reason="no_named_pos")
                nm = [str(x) for x in meta["obj_names"]]
                P = np.asarray(meta["obj_pos"], np.float64).reshape(len(nm), 3)
                pos = {q["name"]: P[nm.index(q["name"])] for q in self.named if q["name"] in nm}
                if not pos:
                    return None, None, dict(reason="no_named_pos")
                n = min(pos, key=lambda q: float(np.linalg.norm(pos[q][:2] - h[:2])))
                return pos[n], next(q for q in self.named if q["name"] == n), dict(target=n)
            pids = sorted({q["proto"] for q in self.named if q["proto"] is not None})
            if not pids or "tok_xyz" not in meta:
                return None, None, dict(reason="no_proto" if not pids else "no_geom")
            xyz, pid, info = self._anchor(observation, meta, pids)
            if not np.all(np.isfinite(xyz)):
                return None, None, dict(info, reason="nonfinite")
            return xyz, next(q for q in self.named if q["proto"] == pid), info

        def _goal_target(self, observation, meta, h):
            if not self.goal:
                return None, None, dict(reason="no_goal")
            if self.arm == "orc":
                if "tgt_xyz" not in meta:
                    return None, None, dict(reason="no_tgt_pos")
                return np.asarray(meta["tgt_xyz"], np.float64).reshape(-1)[:3], self.goal[0], dict(target=self.goal[0]["body"])
            pids = sorted({q["proto"] for q in self.goal if q["proto"] is not None})
            if not pids or "tok_xyz" not in meta:
                return None, None, dict(reason="no_proto" if not pids else "no_geom")
            xyz, pid, info = self._anchor(observation, meta, pids)
            if "tgt_xyz" in meta:
                t = np.asarray(meta["tgt_xyz"], np.float64).reshape(-1)[:3]
                info.update(err=round(float(np.linalg.norm(xyz[:2] - t[:2])), 4), hit=bool(np.linalg.norm(xyz[:2] - t[:2]) < HIT_M))
            if not np.all(np.isfinite(xyz)):
                return None, None, dict(info, reason="nonfinite")
            return xyz, next(q for q in self.goal if q["proto"] == pid), info

        def infer(self, obs, *, noise=None):
            t0 = time.monotonic()
            meta = {k[5:]: v for k, v in obs.items() if k.startswith("meta/")}
            if meta.get("episode_start"):
                self.ph.reset()
                self.qi = 0
            phase = "approach" if self.ph.approach else ("carry" if self.ph.carry else "other")
            rec = dict(task=meta.get("task_id"), ep=meta.get("episode"), t=meta.get("t"), q=self.qi, arm=self.arm, phase=phase, guided=False)
            c = ent = None
            observation = None
            want = (phase == "approach" and self.arm in ("both", "orc")) or (phase == "carry" and self.arm in ("both", "goal", "orc"))
            if want and meta:
                h = np.asarray(obs["observation/state"], np.float64)[:3]
                observation = bk.observation({k: v for k, v in obs.items() if not k.startswith("meta/")})
                c, ent, info = (self._obj_target if phase == "approach" else self._goal_target)(observation, meta, h)
                rec.update(info)
            if c is None:
                out = GuidedPolicy.infer(self, obs, noise=noise)
            else:
                if meta.get("episode_start"):
                    self.ep_q = 0
                    self.branch = None
                sd = self._seed()
                pre = bk.prefix(observation)
                op = dict(J=ent["J"], b=ent["b"], kind=("goal" if phase == "carry" else "approach"))
                rows = self.gd.sample(1, pre, sd, self.cfg, h, c, op)
                with torch.no_grad():
                    D = self.gd.disp(torch.as_tensor(rows, device=bk.device)).cpu().numpy()[0]
                rstar = np.asarray(ent["J"]) @ (np.asarray(c)[:2] - h[:2]) + np.asarray(ent["b"])
                actions = bk.unnormalize(rows, state=observation.state[0].float().cpu().numpy())[0]
                self.calls += 1
                self.ep_q += 1
                self.secs += time.monotonic() - t0
                rec.update(guided=True, seed=sd, level=ent["level"], type=ent["type"], target_disp=[round(float(v), 4) for v in rstar],
                           D=[round(float(v), 4) for v in D], h=[round(float(v), 4) for v in h])
                out = {"actions": np.asarray(actions, np.float32), "state": np.asarray(obs["observation/state"], np.float32)}
            self.ph.observe(out["actions"])
            rec["ms"] = round(1000 * (time.monotonic() - t0), 1)
            if meta:
                self._log(rec)
            self.qi += 1
            return out

    return R6Policy()
