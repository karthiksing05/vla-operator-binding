"""A synthetic compositional manipulation domain for testing the composition machinery (S0).

Codes z in R^16 are split into blocks:  T = dims 0-5 (unit type), O = dims 6-10 (object), G = 11-15 (goal).

    approach(o)  : T = t_approach, O = e_o,  G = 0
    grasp(o)     : T = t_grasp,    O = e_o,  G = 0
    carry(o, g)  : T = t_carry,    O = e_o,  G = f_g
    release(g)   : T = t_release,  O = 0,    G = f_g

A unit lasts a random 5-15 frames; frame t of a unit is  mean + drift * (progress - 1/2) * r_type + noise.
Tasks come from a grammar:
    Task  -> PP | PP PP          (p_double)
    PP    -> Pick(o) Place(o, g)
    Pick  -> approach(o) grasp(o)
    Place -> carry(o, g) release(g)

Observation features (k = 12): [gripper closure, holding, e_o (5), f_g (5)] of the current pick-place,
with small noise.  Gripper ramps closed during grasp and open during release; "holding" switches at the
end of grasp / release (the contact events used by the contact-event baseline).

Splits: objects 0..n_obj-2 and goals 0..n_goal-1 are seen, except the held-out (object, goal) combinations
(G-rec); object n_obj-1 is never seen (G-new).
"""
from dataclasses import dataclass, field

import numpy as np

TYPES = ("approach", "grasp", "carry", "release")
D, KOBS = 16, 12
TB, OB, GB = slice(0, 6), slice(6, 11), slice(11, 16)


@dataclass
class Demo:
    Z: np.ndarray
    O: np.ndarray
    units: list                    # [(type, arg, start, end)]
    frame_unit: np.ndarray         # unit index per frame
    constituents: list = field(default_factory=list)   # [(label, start, end)] Pick / Place / PP / Task
    combos: list = field(default_factory=list)          # [(o, g)]

    def boundaries(self):
        return [u[2] for u in self.units[1:]]

    def unit_seq(self):
        return [(u[0], u[1]) for u in self.units]

    def label(self, span):
        """Ground-truth label of a span (diagnostics only): the unit sequence it covers, majority-voted per
        unit (a unit counts if the span holds at least half of its frames inside the span, or the span
        lies inside it)."""
        s, e = span
        out = []
        for (ty, arg, us, ue) in self.units:
            ov = max(0, min(e, ue) - max(s, us))
            if ov > 0 and (ov >= 0.5 * (ue - us) or ov >= 0.5 * (e - s)):
                out.append((ty, arg))
        return tuple(out)


class Domain:
    def __init__(self, seed=0, n_obj=6, n_goal=4, noise=0.15, drift=0.3, obs_noise=0.05, t_scale=1.5,
                 heldout=((0, 3), (1, 0), (2, 1), (3, 2)), p_double=0.3, len_range=(5, 15)):
        self.rng = np.random.default_rng(seed)
        r = self.rng
        self.n_obj, self.n_goal = n_obj, n_goal
        self.noise, self.drift, self.obs_noise = noise, drift, obs_noise
        self.p_double, self.len_range = p_double, len_range
        self.t = {ty: t_scale * r.standard_normal(6) for ty in TYPES}
        self.dirs = {ty: (lambda v: v / np.linalg.norm(v))(r.standard_normal(6)) for ty in TYPES}
        self.e = r.standard_normal((n_obj, 5))
        self.f = r.standard_normal((n_goal, 5))
        self.new_obj = n_obj - 1
        self.heldout = [tuple(h) for h in heldout]
        self.train_combos = [(o, g) for o in range(n_obj - 1) for g in range(n_goal)
                             if (o, g) not in self.heldout]

    # ---- unit codes --------------------------------------------------------------------------------
    def unit_mean(self, ty, arg):
        z = np.zeros(D)
        z[TB] = self.t[ty]
        if ty in ("approach", "grasp"):
            z[OB] = self.e[arg]
        elif ty == "carry":
            o, g = arg
            z[OB], z[GB] = self.e[o], self.f[g]
        else:
            z[GB] = self.f[arg]
        return z

    def all_units(self, objects=None):
        objs = range(self.n_obj) if objects is None else objects
        out = []
        for o in objs:
            out += [("approach", o), ("grasp", o)]
            out += [("carry", (o, g)) for g in range(self.n_goal)]
        out += [("release", g) for g in range(self.n_goal)]
        return out

    def decode(self, z, objects=None):
        """Oracle decoder (diagnostics only): the nearest ground-truth unit mean."""
        us = self.all_units(objects)
        M = np.stack([self.unit_mean(*u) for u in us])
        return us[int(np.argmin(((M - z[None]) ** 2).sum(1)))]

    # ---- observations --------------------------------------------------------------------------------
    def obs_vec(self, grip, hold, o, g):
        v = np.concatenate([[grip, hold], self.e[o], self.f[g]])
        return v + self.obs_noise * self.rng.standard_normal(KOBS)

    # ---- demonstrations ------------------------------------------------------------------------------
    def demo(self, combos):
        """One demonstration of a task made of pick-places over `combos` [(o, g), ...]."""
        r = self.rng
        Z, O, units, fu, cons = [], [], [], [], []
        t = 0
        for (o, g) in combos:
            pp0 = t
            for ty, arg in (("approach", o), ("grasp", o), ("carry", (o, g)), ("release", g)):
                n = int(r.integers(self.len_range[0], self.len_range[1] + 1))
                m = self.unit_mean(ty, arg)
                for i in range(n):
                    prog = (i + 0.5) / n
                    z = m.copy()
                    z[TB] += self.drift * (prog - 0.5) * self.dirs[ty]
                    z += self.noise * r.standard_normal(D)
                    Z.append(z)
                    grip = {"approach": 0.0, "grasp": prog, "carry": 1.0, "release": 1.0 - prog}[ty]
                    hold = {"approach": 0.0, "grasp": 1.0 if i == n - 1 else 0.0, "carry": 1.0,
                            "release": 1.0 if i < n - 1 else 0.0}[ty]
                    O.append(self.obs_vec(grip, hold, o, g))
                    fu.append(len(units))
                units.append((ty, arg, t, t + n))
                t += n
            a, gr, ca, re = units[-4:]
            cons += [("Pick", a[2], gr[3]), ("Place", ca[2], re[3]), ("PP", pp0, t)]
        cons.append(("Task", 0, t))
        return Demo(np.array(Z), np.array(O), units, np.array(fu), cons, list(combos))

    def sample_task(self, combos_pool, n_pp=None):
        r = self.rng
        if n_pp is None:
            n_pp = 2 if r.random() < self.p_double else 1
        idx = r.choice(len(combos_pool), size=n_pp, replace=True)
        return [combos_pool[i] for i in idx]

    def contact_boundaries(self, demo, thr=0.5):
        """Contact-event baseline: frames where gripper closure crosses thr or 'holding' switches."""
        g, h = demo.O[:, 0], demo.O[:, 1]
        b = set()
        for t in range(1, len(g)):
            if (g[t - 1] < thr) != (g[t] < thr) or (h[t - 1] > 0.5) != (h[t] > 0.5):
                b.add(t)
        return sorted(b)


class Env:
    """Closed-loop stand-in for execution: executing a code decodes it (oracle physics) and updates the
    gripper / holding state; the observation shows the current pick-place's object and goal."""

    def __init__(self, dom, combos):
        self.dom, self.combos = dom, list(combos)
        self.pp = 0
        self.grip, self.hold = 0.0, 0.0
        self.executed = []

    def obs(self):
        o, g = self.combos[min(self.pp, len(self.combos) - 1)]
        return np.concatenate([[self.grip, self.hold], self.dom.e[o], self.dom.f[g]])

    def step(self, code):
        u = self.dom.decode(code)
        self.executed.append(u)
        ty = u[0]
        if ty == "grasp":
            self.grip, self.hold = 1.0, 1.0
        elif ty == "carry":
            self.grip, self.hold = 1.0, 1.0
        elif ty == "release":
            self.grip, self.hold = 0.0, 0.0
            self.pp += 1
        else:
            self.grip, self.hold = 0.0, 0.0
        return u
