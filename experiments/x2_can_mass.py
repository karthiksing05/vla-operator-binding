"""X2: the SAPIEN mass of each G4 can at its stock density (50), and the density that gives 0.15 kg (the registered G4 variant).
    python -I experiments/x2_can_mass.py        (.venv_pi0, GPU node)"""
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import x2_simpler_graded as G

out = {}
for name in ["X2G4PepsiOnTowel-v0", "X2G4RedBullOnPlate-v0", "X2G4SevenUpOnTowel-v0"]:
    env = G.make(name)
    env.reset(options={"obj_init_options": {"episode_id": 0}})
    u = env.unwrapped
    src = u.episode_source_obj
    d0 = u.model_db[src.name].get("density", 1000)
    m = float(src.mass)
    out[src.name] = dict(density=d0, mass=m, volume=m / d0, density_for_015kg=0.15 / (m / d0))
    print("[mass]", name, json.dumps(out[src.name]), flush=True)
    env.close()
print("[mass] all", json.dumps(out))
