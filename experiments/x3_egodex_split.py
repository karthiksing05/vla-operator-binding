"""X3 (EgoDex): verb / object units of every task folder, and the graded split (docs/EGODEX.md, "Split").

Inputs (results/robot/egodex/): test_inventory.json (x3_egodex_inventory.py), train_zip_tasks.json (x3_egodex_zipdir.py),
train_meta.json (x3_egodex_trainmeta.py: per-episode attrs of sampled training episodes).  Light: JSON only.

Units.  A folder's VERBS are read by rule from its name (the verb lexicon below; 'un'-verbs and reversible pairs kept as
separate verbs, each a task of its own in the EgoDex count of 194), checked against the episodes' llm_verbs.  Its OBJECTS are the
remaining name tokens, normalised (plural -> singular, a small synonym table), plus every llm_objects entry seen in >= 30% of its
sampled training episodes (the scene-level nouns 'table', 'tablecloth', 'background' dropped).  'Seen' is defined at the TASK
level from that text: a verb (object) is seen when some training folder carries it; a verb x object pair is seen when one
training folder carries both.  Grades (docs/SPLIT.md):
  G1 relation: every verb and object of the held-out folder seen in training folders, no training folder has the pair;
  G4 part:     at least one verb or object never seen in any training folder.
(G2 binding and G3 scene have no clean analogue here: EgoDex has no fixed scenes; backgrounds/tablecloths vary per session.)
Within basic_pick_place, the episode-level analogue uses the metadata fields object:<o> and to:<target>.

    python -I experiments/x3_egodex_split.py --out results/robot/egodex
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re

VERBS = {
    "add", "remove", "arrange", "topple", "assemble", "disassemble", "pick", "place", "boil", "serve", "braid", "unbraid", "build",
    "unstack", "charge", "uncharge", "clean", "clip", "unclip", "color", "crumple", "flatten", "deal", "gather", "declutter", "dry",
    "fidget", "flip", "fold", "unfold", "stack", "fry", "roll", "insert", "dump", "knead", "load", "dispense", "lock", "unlock",
    "make", "measure", "open", "close", "paint", "peel", "play", "reset", "point", "click", "pour", "push", "pop", "put", "rake",
    "smooth", "scoop", "screw", "unscrew", "sleeve", "unsleeve", "slot", "sort", "staple", "stock", "unstock", "sweep", "thread",
    "unthread", "throw", "catch", "collect", "tie", "untie", "type", "use", "wash", "wipe", "wrap", "unwrap", "write", "zip",
    "unzip", "blowdry", "knit", "stamp", "setup", "cleanup", "set", "take",
}
FILLER = {"and", "up", "down", "in", "out", "on", "from", "into", "away", "basic", "vertical", "free", "or", "the", "a", "with",
          "furniture", "bench", "magnetic", "kitchen", "surface", "surfaces", "objects"}
PHRASES = {"set_up": "set_up", "put_away": "put_away", "take_out": "take_out", "put_in": "put_in", "clean_up": "clean_up",
           "setup": "set_up", "cleanup": "clean_up", "put_down": "put_down", "pick_up": "pick"}
SYN = {"cups": "cup", "cloths": "cloth", "legos": "lego", "papers": "paper", "beads": "bead", "batteries": "battery",
       "bowls": "bowl", "plates": "plate", "dishes": "dish", "objects": "object", "cards": "card", "dominoes": "domino",
       "tiles": "tile", "structures": "structure", "blocks": "block", "utensils": "utensil", "glasses": "glasses",
       "airpods": "airpods", "pages": "page", "hands": "hand", "dice": "die", "tupperware": "tupperware", "shoelace": "shoelace",
       "rubberband": "rubber_band", "lid": "lid", "socket": "socket", "plug": "plug", "fingers": "finger", "jenga": "jenga_block",
       "plushie": "plush", "plushies": "plush", "duck": "duck", "chessboard": "chessboard", "dustpan": "dustpan",
       "toothbrush": "toothbrush", "toothpaste": "toothpaste", "fridge": "fridge", "necklace": "necklace", "ball": "ball",
       "remote": "remote", "keyboard": "keyboard", "piano": "piano", "screen": "screen", "bag": "bag", "case": "case"}
SCENE = {"table", "tablecloth", "background", "wooden table", "metal table", "wood table", "hand", "hands", "right hand",
         "left hand", "surface", "fingers", "finger"}


COMPOUNDS = {"tennis ball": "tennis_ball", "round table": "round_table", "square table": "square_table", "connect four": "connect_four",
             "zen garden": "zen_garden", "rubiks cube": "rubiks_cube", "board game": "board_game", "allen fixture": "allen_fixture",
             "spinner rings": "spinner_ring", "jigsaw puzzle": "jigsaw_puzzle", "soft lego": "soft_lego", "tetra board": "tetra_board",
             "bottle cap": "bottle_cap", "finger fixture": "finger_fixture", "origami paper": "origami_paper", "paper origami": "origami"}
HEAD = {"tennis_ball": "ball", "round_table": "table_piece", "square_table": "table_piece", "soft_lego": "lego",
        "bottle_cap": "bottle", "origami_paper": "paper", "allen_fixture": "fixture", "finger_fixture": "fixture", "bagging": "bag"}


VSYN = {"put_down": "place", "pick up": "pick", "put down": "place", "unstake": "unstack", "cleanup": "clean_up", "setup": "set_up",
        "put away": "put_away", "set up": "set_up", "clean up": "clean_up", "place back": "place"}


def norm_obj(w):
    """an object phrase -> its unit: the head noun (last word), singular; folder compounds (tennis_ball, ...) kept whole."""
    w = w.strip().lower().replace("-", " ")
    w = re.sub(r"\s+", " ", w)
    w = SYN.get(w, w)
    if " " in w:
        w = SYN.get(w.split(" ")[-1], w.split(" ")[-1])
    w = HEAD.get(w, w) if w == "bagging" else w
    w = {"tupperwear": "tupperware"}.get(w, w)
    for h in ("bowl", "plate", "tray", "basket", "container", "tupperware", "box"):
        if w.endswith(h) and w != h and len(w) - len(h) >= 3:      # colour-fused targets: bluebowl, graybowl
            w = h
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith(("ches", "shes", "xes", "sses")):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "glasses", "airpods", "chopsticks", "us")) and len(w) > 3:
        w = w[:-1]
    return w


def parse_folder(name):
    """folder name -> (verbs, objects, notes) by rule."""
    toks = name.split("_")
    verbs, objs, notes, i = [], [], [], 0
    while i < len(toks):
        two = "_".join(toks[i:i + 2])
        if two in PHRASES:
            verbs.append(PHRASES[two]); i += 2; continue
        t = toks[i]
        if t in PHRASES:
            verbs.append(PHRASES[t])
        elif t in VERBS:
            verbs.append(t)
        elif t not in FILLER:
            objs.append(norm_obj(t))
        i += 1
    joined = " ".join(objs)
    for k, v in COMPOUNDS.items():
        if k in joined:
            joined = joined.replace(k, v)
    objs = [HEAD.get(o, o) if o == "bagging" else o for o in joined.split()]
    if name.startswith("use_"):
        notes.append("'use X': X is a tool, the manipulation verb is implicit")
    if name in ("color", "write", "type_keyboard", "play_piano", "dry_hands"):
        notes.append("activity name; object implicit or the hand itself")
    if "furniture_bench" in name:
        notes.append("FurnitureBench part assembly; object = the furniture piece")
    if not objs:
        notes.append("no object token in the name")
    return sorted(set(verbs)), sorted(set(objs)), notes


def parse_list(s):
    """attrs like "['pick' 'place']" -> ['pick', 'place']."""
    if s is None:
        return []
    return [x for x in re.findall(r"'([^']*)'", str(s)) if x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/robot/egodex")
    a = ap.parse_args()
    te = json.load(open(os.path.join(a.out, "test_inventory.json")))["tasks"]
    tz = json.load(open(os.path.join(a.out, "train_zip_tasks.json")))["tasks"]
    meta = json.load(open(os.path.join(a.out, "train_meta.json")))
    by = collections.defaultdict(list)
    for m in meta:
        if "error" not in m:
            by[m["folder"]].append(m)
    folders = sorted(set(k.split(":")[-1] for k in tz))
    units = {}
    for f in folders:
        v, o, notes = parse_folder(f)
        eps = by.get(f, [])
        vc = collections.Counter(VSYN.get(x, x) for e in eps for x in set(parse_list(e.get("llm_verbs"))))
        oc = collections.Counter(norm_obj(x) for e in eps for x in set(parse_list(e.get("llm_objects"))))
        n = max(1, len(eps))
        llm_v = sorted(k for k, c in vc.items() if c >= 0.3 * n)
        llm_o = sorted(k for k, c in oc.items() if c >= 0.3 * n and k not in SCENE)
        ztr = [t for k, t in tz.items() if k.split(":")[-1] == f]
        v = sorted({VSYN.get(x, x) for x in v})
        top = [k for k, c in oc.most_common() if k not in SCENE and c >= 0.5 * n]
        units[f] = dict(verbs=v, objects_name=o, llm_verbs=llm_v, llm_objects=llm_o,
                        objects=sorted(set(o) | set(llm_o)),            # the vocabulary a TRAINING folder exposes
                        primary=sorted(set(o) or set(top[:2])),         # the objects a HELD-OUT folder is graded on
                        notes=notes,
                        verb_mismatch=sorted(set(llm_v) - set(v)),
                        train_episodes=sum(t["episodes"] for t in ztr), train_gb=round(sum(t["bytes"] for t in ztr) / 1e9, 2),
                        zips=sorted(t["zip"] for t in ztr), test_episodes=te.get(f, {}).get("episodes", 0),
                        test_hours=te.get(f, {}).get("hours", 0), sampled=len(eps),
                        reversible=round(sum(str(e.get("llm_type")) == "reversible" for e in eps) / n, 2))
    json.dump(units, open(os.path.join(a.out, "task_units.json"), "w"), indent=1)

    # basic_pick_place episode-level object x target table
    bpp = collections.Counter()
    for e in by.get("basic_pick_place", []):
        env, obj = str(e.get("environment", "")), str(e.get("object", ""))
        o = re.search(r"object:\s*([^,]+)", obj)
        to = re.search(r"to:\s*([^,]+)", env)
        bpp[(norm_obj(o.group(1)) if o else "?", (to.group(1).strip().lower() if to else "?"))] += 1
    json.dump([[k[0], k[1], c] for k, c in bpp.most_common()], open(os.path.join(a.out, "bpp_object_target.json"), "w"))
    print("folders", len(units), "| basic_pick_place (object, to) pairs", len(bpp), "objects", len({k[0] for k in bpp}),
          "targets", len({k[1] for k in bpp}))
    proposal(units, json.load(open(os.path.join(a.out, "train_zip_files.json"))), a.out)
    bpp_split(by.get("basic_pick_place", []), te["basic_pick_place"]["episodes_list"], a.out)
    for f, u in units.items():
        print(f"{f:50s} V={u['verbs']} O={u['objects']} llmV={u['llm_verbs']} test={u['test_episodes']} train={u['train_episodes']} "
              f"{u['train_gb']}GB {'; '.join(u['notes'])}")


_COLOURS = set("red orange yellow green blue purple pink brown black white gray grey silver gold beige tan navy clear transparent "
               "metal wood wooden multicolor color multi".split())


def _ot(e):
    env, obj = str(e.get("environment", "")), str(e.get("object", ""))
    o = re.search(r"object:\s*([^,]+)", obj)
    to = re.search(r"to:\s*([^,]+)", env)
    ov = o.group(1) if o else None
    c = re.search(r"color:\s*([^,]+)", obj)
    # swapped fields (session 2025-04-03_16-42-36: "object:brown, color:croissant"): the object is a colour word and the colour
    # field a noun -> the colour field holds the object (found by experiments/x3_egodex_verify_split.py, 2026-10-20)
    if ov and c and ov.strip().lower() in _COLOURS and c.group(1).strip().lower().replace("-", " ").split()[-1] not in _COLOURS:
        ov = c.group(1)
    return (norm_obj(ov) if ov else None, norm_obj(to.group(1)) if to else None)


def bpp_split(train_eps, test_eps, out, min_other=2):
    """Episode-level graded split inside basic_pick_place: units = (moved object, target container), from the metadata fields
    object:<o> and environment to:<target>, head nouns.  For each TEST episode:
      G0 the pair occurs in training episodes;
      G1 object and target each occur in >= min_other training episodes with other partners, the pair never;
      G4 the object (or the target) never occurs in training.
    Holding out a G1 pair means dropping its training episodes; that is free here, since the pair has none."""
    tr = collections.Counter(_ot(e) for e in train_eps)
    objs = collections.Counter(o for (o, t), c in tr.items() for _ in range(c))
    tgts = collections.Counter(t for (o, t), c in tr.items() for _ in range(c))
    res = collections.Counter()
    ex = collections.defaultdict(list)
    for e in test_eps:
        o, t = _ot(e)
        if o is None or t is None:
            g = "unparsed"
        elif tr[(o, t)]:
            g = "G0"
        elif objs[o] == 0 or tgts[t] == 0:
            g = "G4"
        elif objs[o] >= min_other and tgts[t] >= min_other:
            g = "G1"
        else:
            g = "rare"
        res[g] += 1
        if len(ex[g]) < 8:
            ex[g].append((e.get("ep"), o, t))
    # constructible grades: hold the pair (G1) or the object (G4) out of training by dropping its training episodes
    cons = collections.Counter()
    cand_pairs, cand_objs = collections.Counter(), collections.Counter()
    for e in test_eps:
        o, t = _ot(e)
        if o is None or t is None:
            continue
        o_other = objs[o] - tr[(o, t)]
        t_other = tgts[t] - tr[(o, t)]
        n_o_targets = len({tt for (oo, tt) in tr if oo == o and tt != t})
        if o_other >= min_other and t_other >= min_other and n_o_targets >= 1:
            cons["G1_pair_holdable"] += 1
            cand_pairs[(o, t)] += 1
        cand_objs[o] += 1
    pair_rows = sorted(((c, o, t, tr[(o, t)], objs[o] - tr[(o, t)], tgts[t] - tr[(o, t)]) for (o, t), c in cand_pairs.items()),
                       reverse=True)
    obj_rows = sorted(((c, o, objs[o]) for o, c in cand_objs.items()), reverse=True)
    d = dict(train_episodes=len(train_eps), train_pairs=len(tr), train_objects=len(objs), train_targets=len(tgts),
             test_grades=dict(res), constructible=dict(cons),
             g1_pairs=[dict(test_eps=c, object=o, target=t, train_eps_dropped=n, object_other=oo, target_other=to)
                       for c, o, t, n, oo, to in pair_rows],
             g4_objects=[dict(test_eps=c, object=o, train_eps_dropped=n) for c, o, n in obj_rows],
             examples=ex, top_targets=tgts.most_common(15), top_objects=objs.most_common(15))
    json.dump(d, open(os.path.join(out, "bpp_split.json"), "w"), indent=1, default=str)
    print("\n== basic_pick_place episode-level:", {k: v for k, v in d.items() if k not in ("examples", "g1_pairs", "g4_objects")})
    print("  G1 pair candidates (test eps, object, target, train eps dropped, object other, target other):", pair_rows[:25])
    print("  G4 object candidates (test eps, object, train eps dropped):", obj_rows[:30])
    for g, l in ex.items():
        print("  ", g, l)


# The proposal (docs/EGODEX.md): held-out folders by intended grade; every other folder of part1-5 is a training folder.
HELD = {
    "G1": ["charge_uncharge_airpods", "open_close_insert_remove_tupperware", "stack_unstack_cups", "stack_unstack_bowls"],
    "G4": ["pour", "lock_unlock_key", "zip_unzip_bag", "zip_unzip_case", "screw_unscrew_allen_fixture", "screw_unscrew_bottle_cap",
           "screw_unscrew_fingers_fixture", "tie_and_untie_shoelace", "tie_untie_rubberband", "throw_and_catch_ball",
           "throw_collect_objects"],
}
# < 100 training episodes and long multi-step (fry_*, wash_fruit, wash_kitchen_dishes; fry_egg's text also says "pour");
# put_toothpaste_on_toothbrush: every one of its instructions says "unscrew the toothpaste cap", which would make screw seen;
# pick_up_and_put_down_case_or_bag: "pick ... case" co-occurs in basic_pick_place / put_in_take_out_glasses text (not a clean G1).
EXCLUDE_FROM_TRAIN = {"fry_bread", "fry_egg", "wash_fruit", "wash_kitchen_dishes", "put_toothpaste_on_toothbrush",
                      "pick_up_and_put_down_case_or_bag"}
# Inside basic_pick_place (episode level; units = moved object x target, docs/EGODEX.md): pairs held out of training (G1) and
# objects held out of basic_pick_place training episodes (G2 when another training folder's text names them, G4 otherwise).
BPP_HELD_PAIRS = [("block", "bowl"), ("duck", "plate"), ("plush", "plate"), ("egg", "bowl"), ("die", "bowl"), ("donut", "tupperware")]
BPP_HELD_OBJECTS = ["cup", "banana", "croissant", "iphone", "stapler", "mouse"]
# the fetch plan: episodes per training folder (evenly spaced); donors of the G1 units get more
CAP_DEFAULT, CAP_DONOR, CAP_BPP = 150, 500, 3000
DONORS = {"charge_uncharge_device", "insert_remove_airpods", "stack_unstack_plates", "stack", "clean_cups", "add_remove_lid",
          "insert_remove_cups_from_rack", "stack_unstack_tupperware", "open_close_insert_remove_box", "open_close_insert_remove_case"}


def grade(f, units, train):
    """G0 / G1 / G4 of folder f against the training folders, at the task level (folder verbs x objects)."""
    u = units[f]
    V = set().union(*(units[t]["verbs"] for t in train))
    O = set().union(*(units[t]["objects"] for t in train))
    newv, newo = set(u["verbs"]) - V, set(u["primary"]) - O
    if newv or newo:
        return "G4", dict(unseen_verbs=sorted(newv), unseen_objects=sorted(newo), primary=u["primary"])
    pairs = {(v, o) for v in u["verbs"] for o in u["primary"]}
    shared = [t for t in train if pairs & {(v, o) for v in units[t]["verbs"] for o in units[t]["objects"]}]
    if shared:
        return "G0", dict(pair_seen_in=shared)
    donors_v = {v: [t for t in train if v in units[t]["verbs"]][:4] for v in u["verbs"]}
    donors_o = {o: [t for t in train if o in units[t]["objects"]][:4] for o in u["primary"]}
    return "G1", dict(verb_donors=donors_v, object_donors=donors_o)


def proposal(units, files, out, caps=(50, 100, 150, 200, 500, None)):
    held = [f for g in HELD.values() for f in g]
    main_folders = {k for k, u in units.items() if any(z != "extra.zip" for z in u["zips"])}
    train = sorted(main_folders - set(held) - EXCLUDE_FROM_TRAIN)
    rows = {f: dict(intended=g, **dict(zip(("grade", "why"), grade(f, units, train))), test_episodes=units[f]["test_episodes"])
            for g, l in HELD.items() for f in l}
    # bytes of the training subset, per-episode cap (evenly spaced episodes), hdf5 + mp4 of main-part zips only
    per = {}
    for z, lst in files.items():
        if z == "extra.zip":
            continue
        for n, off, cs, us, t in lst:
            fo, base = n.split("/")[-2], os.path.basename(n)
            if fo in train:
                per.setdefault(fo, {}).setdefault(base.split(".")[0], 0)
                per[fo][base.split(".")[0]] += cs
    sizes = {}
    for cap in caps:
        tot, neps = 0, 0
        for fo, eps in per.items():
            ks = sorted(eps, key=int)
            if cap is not None and len(ks) > cap:
                import numpy as np
                ks = [ks[i] for i in np.linspace(0, len(ks) - 1, cap).round().astype(int)]
            tot += sum(eps[k] for k in ks)
            neps += len(ks)
        sizes[str(cap)] = dict(episodes=neps, gb=round(tot / 1e9, 1))
    import numpy as np
    plan, tot, neps = {}, 0, 0
    for fo, eps in per.items():
        cap = CAP_BPP if fo == "basic_pick_place" else CAP_DONOR if fo in DONORS else CAP_DEFAULT
        ks = sorted(eps, key=int)
        if len(ks) > cap:
            ks = [ks[i] for i in np.linspace(0, len(ks) - 1, cap).round().astype(int)]
        plan[fo] = dict(episodes=len(ks), gb=round(sum(eps[k] for k in ks) / 1e9, 2))
        tot += sum(eps[k] for k in ks)
        neps += len(ks)
    sizes["plan"] = dict(episodes=neps, gb=round(tot / 1e9, 1), caps=dict(default=CAP_DEFAULT, donor=CAP_DONOR, bpp=CAP_BPP))
    hd = sum(cs for z, lst in files.items() if z != "extra.zip" for n, off, cs, us, t in lst
             if n.endswith(".hdf5") and n.split("/")[-2] in train)
    sizes["hdf5_only_all"] = dict(gb=round(hd / 1e9, 1))
    res = dict(train=train, held=rows, train_sizes=sizes, plan=plan, excluded=sorted(EXCLUDE_FROM_TRAIN),
               bpp_held_pairs=BPP_HELD_PAIRS, bpp_held_objects=BPP_HELD_OBJECTS)
    json.dump(res, open(os.path.join(out, "split_proposal.json"), "w"), indent=1)
    print("\n== proposal: train folders", len(train), "sizes by per-folder episode cap", sizes)
    for f, r in rows.items():
        print(f"  {r['intended']} -> {r['grade']}  {f:40s} test eps {r['test_episodes']:3d}  {r['why']}")


if __name__ == "__main__":
    main()
