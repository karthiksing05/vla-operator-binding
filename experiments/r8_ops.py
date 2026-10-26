"""PROBE R8 (docs/PLAN_R8.md): the region lexicon and the per-task operator spec.  Pure python, no policy.

LEX is transcribed from the TRAINING instructions (the l90c train split, results/r8/lex.json printed by r8_lex.py; split_sealed.json is
never opened): (body type, region suffix) -> the phrase the training instruction uses for that goal region, and the training tasks
whose goal is that region (the C3 prototype sources).

spec(language, goal, fixtures) -> dict or None, from the PARSE (goal atom) + the scene's fixture types + LEX only:
  phrase  the region phrase found verbatim in the instruction: an entry with the goal's region suffix, the same body type first,
          else any body type (category level, as R6's :29)
  sources the training tasks of that entry (C3)
  alts    the instruction with the phrase replaced by the phrase of every OTHER region of the same body type (C2)
"""
from __future__ import annotations

LEX = [
    # (body type, region suffix, phrase, training tasks with that goal region)
    ("white_cabinet", "top_side", "on top of the cabinet", ["libero_90:25"]),
    ("white_cabinet", "bottom_region", "in the bottom drawer of the cabinet", ["libero_90:26"]),
    ("wooden_cabinet", "top_region", "in the top drawer of the cabinet", ["libero_90:3", "libero_90:4", "libero_90:5"]),
    ("wooden_cabinet", "top_side", "on top of the cabinet", ["libero_goal:4"]),
    ("wooden_two_layer_shelf", "top_side", "on top of the cabinet", ["libero_90:43"]),
    ("wooden_two_layer_shelf", "top_region", "on the cabinet shelf", ["libero_90:88"]),
    ("wooden_two_layer_shelf", "bottom_region", "under the cabinet shelf", ["libero_90:89"]),
]


def goal_region(goal, fixtures):
    """The last In / On atom whose target is a region of a fixture -> (body, body type, region suffix) or None."""
    for atom in reversed(list(goal)):
        if atom[0] not in ("In", "On") or len(atom) < 3:
            continue
        tgt = str(atom[2])
        for body, typ in fixtures.items():
            if tgt.startswith(body + "_"):
                return body, typ, tgt[len(body) + 1:]
    return None


def spec(language, goal, fixtures):
    g = goal_region(goal, fixtures)
    if g is None:
        return None
    body, typ, reg = g
    cands = [e for e in LEX if e[1] == reg and e[2] in language]
    same = [e for e in cands if e[0] == typ]
    pick = (same or cands)
    if not pick:
        return None
    e = pick[0]
    level = "type" if same else "category"
    srcs = sorted({s for x in pick for s in x[3]} if not same else set(e[3]))
    alts = sorted({x[2] for x in LEX if x[0] == typ and x[1] != reg and x[2] != e[2]})
    return dict(body=body, type=typ, region=reg, phrase=e[2], level=level, sources=srcs,
                alt_prompts=[language.replace(e[2], a) for a in alts], alt_phrases=alts)
