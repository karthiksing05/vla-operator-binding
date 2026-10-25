"""PROBE R5s (EXPLORATORY closed-loop smoke; docs/PLAN_R5S.md): the serving side.  Opt-in mode of experiments/serve_guided.py,
default off (the mode string must start with "r5s:"):

    r5s:<spec.json>?arm=<off|vis|orc>&sched=<early|all>&w=<w>&log=<queries.jsonl>

The frozen pi05_l90c@5000 is served exactly as zs (MODE ours: GuidedPolicy with no hooks, plain Euler 10 steps, the server's
seed + call count as the noise seed) on every query that is NOT guided; those queries go through GuidedPolicy.infer itself.
A query is guided only while the gripper is open and before the policy's first grasp (the _Grasp / PrimHooks phase rule: the
commanded gripper of the chunk's 5th action closed on 3 consecutive queries = the grasp; guidance is off from then on, for the
rest of the episode).  On a guided query the chunk is sampled by R4-bind's x0-projection sampler (rb4_bind.Guided.sample: at each
guided Euler step x0_hat = x_t - t v, its integrated xy displacement D is moved by w (d* - D), v' = (x_t - x0_hat') / t), with the
same noise seed the unguided query would have drawn.  The target displacement is D4's bank operator, d* = J (c - h) + b (J the
2x2 xy block, b the xy offset), for the named object's type (else its category, else the approach role), and c is
  vis  the vision anchor: the argmax agentview SigLIP token of the cosine to the named types' training prototypes
       (results/rb4/protos.npz, R4-bind's), back-projected through the LIVE agentview depth (robot visible) sent by the client
       (libero_client.py, opt-in R5S_GEOM=1: meta/tok_xyz);
  orc  the oracle: the named object nearest the hand (xy) at its simulator position (meta/obj_pos, --objects --all-objects).
`off` never guides (the serving smoke: it must reproduce zs).  Every query is logged to `log` (one JSON line per query).
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

HIT_M = 0.05                    # localisation: the anchor's xy within 5 cm of a named object's simulator position (R4's corridor)
GRASP_RUN = 3                   # the _Grasp rule: closed on 3 consecutive queries = grasped


def parse(mode):
    body, q = (mode[len("r5s:"):].split("?", 1) + [""])[:2]
    kv = dict(x.split("=", 1) for x in q.split("&") if x)
    return body, kv


class Tracker:
    """PrimHooks(goal_only)'s grasp rule (held_ops._Grasp): `closed` = the chunk's 5th commanded gripper > 0; the grasp = closed on
    3 consecutive queries.  Guidance runs while not closed and before the first grasp."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.run, self.closed, self.grasped = 0, False, False

    def observe(self, actions):
        a = np.asarray(actions)
        self.closed = bool(a[min(4, len(a) - 1), 6] > 0)
        if self.closed:
            self.run += 1
            if self.run >= GRASP_RUN:
                self.grasped = True
        else:
            self.run = 0

    @property
    def open_pre_grasp(self):
        return (not self.closed) and (not self.grasped)


def make_policy(bk, mode, num_steps=10, seed=0):
    import torch
    import rb4_bind as RB
    from lifelong_ood.policy.serve import GuidedPolicy

    class BankGuided(RB.Guided):
        """R4-bind's sampler with D4's bank operator as the target: d* = J (c - h) + b."""

        def dstar(self, D, cfg, h, c, op):
            ch = np.asarray(c, np.float64)[:2] - np.asarray(h, np.float64)[:2]
            v = np.asarray(op["J"], np.float64) @ ch + np.asarray(op["b"], np.float64)
            return torch.as_tensor(v, dtype=torch.float32, device=D.device)[None].expand_as(D)

    class R5SPolicy(GuidedPolicy):
        def __init__(self):
            super().__init__(bk, hooks=None, num_steps=num_steps, seed=seed)
            path, kv = parse(mode)
            self.spec = json.load(open(path, encoding="utf-8"))
            self.arm = kv.get("arm", "off")
            assert self.arm in ("off", "vis", "orc"), self.arm
            self.cfg = dict(target="bank", sched=kv.get("sched", "early"), w=float(kv.get("w", 1.0)))
            assert self.cfg["sched"] in RB.SCHEDS, self.cfg
            self.gd = BankGuided(bk)
            sc = np.asarray(self.spec["scale"], np.float64)      # D4's demo-calibrated metres per action unit (J's units)
            self.gd.scale = sc
            self.gd.t_scale = torch.as_tensor(sc[:2], dtype=torch.float32, device=bk.device)
            self.named = self.spec["named"]
            pz = np.load(os.path.join(ROOT, "results", "rb4", "protos.npz"), allow_pickle=True)
            self.PR = torch.as_tensor(pz["sig"], device=bk.device)
            self.pids = sorted({q["proto"] for q in self.named if q["proto"] is not None})
            self.tr = Tracker()
            self.qi = 0
            self.logf = None
            if kv.get("log"):
                os.makedirs(os.path.dirname(kv["log"]), exist_ok=True)
                self.logf = open(kv["log"], "w", encoding="utf-8")         # a requeued instance starts its log afresh
            logging.info(f"r5s: arm {self.arm} cfg {self.cfg} task {self.spec['task']} named "
                         f"{[(q['name'], q['type'], q['level']) for q in self.named]} protos {self.pids} scale {sc.tolist()}")

        def _log(self, rec):
            if self.logf is not None:
                self.logf.write(json.dumps(rec) + "\n")
                self.logf.flush()

        def _siglip(self, observation):
            m = bk.model
            images, img_masks, lang_tokens, lang_masks, _ = m._preprocess_observation(observation, train=False)
            embs, _, _ = m.embed_prefix(images, img_masks, lang_tokens, lang_masks)
            return embs[0, :256].float()

        def _named_pos(self, meta):
            if "obj_names" not in meta:
                return {}
            nm = [str(x) for x in meta["obj_names"]]
            P = np.asarray(meta["obj_pos"], np.float64).reshape(len(nm), 3)
            return {q["name"]: P[nm.index(q["name"])] for q in self.named if q["name"] in nm}

        def _target(self, observation, meta, h):
            """-> (c [3], named entry, info) or (None, None, info)."""
            pos = self._named_pos(meta)
            info = {}
            if self.arm == "orc":
                if not pos:
                    return None, None, dict(reason="no_named_pos")
                n = min(pos, key=lambda q: float(np.linalg.norm(pos[q][:2] - h[:2])))
                return pos[n], next(q for q in self.named if q["name"] == n), dict(target=n)
            if not self.pids or "tok_xyz" not in meta:
                return None, None, dict(reason="no_proto" if not self.pids else "no_geom")
            with torch.no_grad():
                S = torch.nn.functional.normalize(self._siglip(observation), dim=-1)
                sim = self.PR[self.pids] @ S.T                              # [P, 256]
                best, arg = sim.max(0)
                tok = int(torch.argmax(best))
                pid = self.pids[int(arg[tok])]
            xyz = np.asarray(meta["tok_xyz"], np.float64).reshape(256, 3)[tok]
            info.update(tok=tok, xyz=[round(float(v), 4) for v in xyz], sim=round(float(best[tok]), 4), proto=pid)
            if pos:
                dn = {n: float(np.linalg.norm(xyz[:2] - p[:2])) for n, p in pos.items()}
                info.update(err=round(min(dn.values()), 4), nearest=min(dn, key=dn.get), hit=bool(min(dn.values()) < HIT_M))
            if not np.all(np.isfinite(xyz)):
                return None, None, dict(info, reason="nonfinite")
            ent = next(q for q in self.named if q["proto"] == pid)
            return xyz, ent, info

        def infer(self, obs, *, noise=None):
            t0 = time.monotonic()
            meta = {k[5:]: v for k, v in obs.items() if k.startswith("meta/")}
            if meta.get("episode_start"):
                self.tr.reset()
                self.qi = 0
            rec = dict(task=meta.get("task_id"), ep=meta.get("episode"), t=meta.get("t"), q=self.qi, arm=self.arm,
                       phase_open=self.tr.open_pre_grasp, guided=False)
            c = ent = None
            observation = None
            if self.arm != "off" and meta and self.tr.open_pre_grasp:
                h = np.asarray(obs["observation/state"], np.float64)[:3]
                observation = bk.observation({k: v for k, v in obs.items() if not k.startswith("meta/")})
                c, ent, info = self._target(observation, meta, h)
                rec.update(info)
            if c is None:
                out = GuidedPolicy.infer(self, obs, noise=noise)          # exactly zs (MODE ours) on every unguided query
            else:
                if meta.get("episode_start"):
                    self.ep_q = 0
                    self.branch = None
                sd = self._seed()
                pre = bk.prefix(observation)
                op = dict(J=ent["J"], b=ent["b"])
                rows = self.gd.sample(1, pre, sd, self.cfg, h, c, op)
                with torch.no_grad():
                    D = self.gd.disp(torch.as_tensor(rows, device=bk.device)).cpu().numpy()[0]
                ds = np.asarray(ent["J"]) @ (np.asarray(c)[:2] - h[:2]) + np.asarray(ent["b"])
                actions = bk.unnormalize(rows, state=observation.state[0].float().cpu().numpy())[0]
                self.calls += 1
                self.ep_q += 1
                self.secs += time.monotonic() - t0
                rec.update(guided=True, seed=sd, level=ent["level"], type=ent["type"], dstar=[round(float(v), 4) for v in ds],
                           D=[round(float(v), 4) for v in D], h=[round(float(v), 4) for v in h])
                out = {"actions": np.asarray(actions, np.float32), "state": np.asarray(obs["observation/state"], np.float32)}
            self.tr.observe(out["actions"])
            rec["ms"] = round(1000 * (time.monotonic() - t0), 1)
            if meta:
                self._log(rec)
            self.qi += 1
            return out

    return R5SPolicy()
