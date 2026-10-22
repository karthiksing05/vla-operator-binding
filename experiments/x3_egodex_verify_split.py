"""X3 (EgoDex): verify, from the files on disk, that the training data excludes every held-out folder and pair (docs/EGODEX.md §2).

Independent of the manifest builder: reads the derived training files (<data>/derived/train/<folder>/<ep>.npz), the split
definitions (x3_egodex_split.HELD / EXCLUDE_FROM_TRAIN / BPP_HELD_PAIRS / BPP_HELD_OBJECTS, split_proposal.json, bpp_split.json)
and each basic_pick_place training episode's own metadata (train_meta.json, object / environment to:), and checks:
  1. no training file lies in a held-out (G1 / G4) or excluded folder, and every training folder is one of the 90;
  2. no basic_pick_place training file is a held-out pair (G1) or has a held-out object (G2 / G4), by its own metadata;
  3. every basic_pick_place training file has metadata (no unparsed episode slipped in);
  4. the stored instruction of every training file (all folders) never names a G1 pair or a G4 verb as listed below
     (word-boundary regex), and no basic_pick_place training instruction names a held-out object;
  5. the training index the trainer samples from (derived/train_index.json) covers exactly these files.
Writes results/robot/egodex/split_verify.json.  CPU only (SLURM).
"""
from __future__ import annotations

import collections
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x3_egodex_split import BPP_HELD_OBJECTS, BPP_HELD_PAIRS, EXCLUDE_FROM_TRAIN, HELD, _ot  # noqa: E402

DATA = "/coc/flash10/ksingara3/robot/data/egodex"
R = os.path.join(ROOT, "results/robot/egodex")
# G4 verbs never seen in training instructions (docs/EGODEX.md §2: pour, zip, tie, throw; lock as a verb; screw)
G4_VERB_RE = re.compile(r"\b(pour\w*|zip\w*|unzip\w*|tie|ties|tying|tied|untie\w*|throw\w*|threw|catch\w*|screw\w*|unscrew\w*|shoelace\w*|rubber ?band\w*)\b", re.I)
# G1 pairs as text co-occurrence: (verb regex, object regex)
G1_PAIRS = [(r"\b(charg|uncharg)\w*", r"\bairpods?\b"), (r"\b(open|clos|insert|remov)\w*", r"\btupperwa\w*"),
            (r"\b(stack|unstack)\w*", r"\bbowls?\b"), (r"\b(stack|unstack)\w*", r"\bcups?\b")]


def main():
    prop = json.load(open(os.path.join(R, "split_proposal.json")))
    train = set(prop["train"])
    held = {f for g in HELD.values() for f in g}
    files = sorted(glob.glob(os.path.join(DATA, "derived", "train", "*", "*.npz")))
    folders = collections.Counter(os.path.basename(os.path.dirname(p)) for p in files)
    res = dict(files=len(files), folders=len(folders))
    res["in_held_folder"] = sorted(set(folders) & held)
    res["in_excluded_folder"] = sorted(set(folders) & EXCLUDE_FROM_TRAIN)
    res["not_a_training_folder"] = sorted(set(folders) - train)
    res["training_folders_missing"] = sorted(train - set(folders))
    # basic_pick_place by its own metadata
    meta = {}
    for m in json.load(open(os.path.join(R, "train_meta.json"))):
        if m.get("folder") == "basic_pick_place" and m.get("zip") != "extra.zip" and "error" not in m:
            meta[os.path.basename(m["file"]).split(".")[0]] = m
    hp, ho = {tuple(p) for p in BPP_HELD_PAIRS}, set(BPP_HELD_OBJECTS)
    bpp = [p for p in files if os.path.basename(os.path.dirname(p)) == "basic_pick_place"]
    bad = collections.Counter()
    pairs = collections.Counter()
    for p in bpp:
        m = meta.get(os.path.basename(p)[:-4])
        if m is None:
            bad["no_metadata"] += 1
            continue
        o, t = _ot(m)
        pairs[(o, t)] += 1
        if o is None or t is None:
            bad["unparsed"] += 1
        if (o, t) in hp:
            bad["held_pair"] += 1
        if o in ho:
            bad["held_object"] += 1
    res["bpp"] = dict(files=len(bpp), violations=dict(bad), distinct_pairs=len(pairs),
                      held_pairs_present={f"{o}-{t}": pairs[(o, t)] for o, t in hp if pairs[(o, t)]})
    # instructions
    hits = collections.defaultdict(list)
    obj_re = re.compile(r"\b(" + "|".join(sorted({"cup", "cups", "banana", "bananas", "croissant", "croissants", "iphone", "iphones",
                                                       "stapler", "staplers", "mouse"})) + r")\b", re.I)
    pair_re = [(re.compile(v, re.I), re.compile(o, re.I)) for v, o in G1_PAIRS]
    for p in files:
        fo = os.path.basename(os.path.dirname(p))
        ins = str(np.load(p)["instruction"])
        mv = G4_VERB_RE.search(ins)
        if mv:
            hits["g4_verb"].append((fo, os.path.basename(p), mv.group(0), ins[:160]))
        for k, (vr, orr) in enumerate(pair_re):
            if vr.search(ins) and orr.search(ins):
                hits[f"g1_pair{k}"].append((fo, os.path.basename(p), ins[:160]))
        if fo == "basic_pick_place":
            mo = obj_re.search(ins)
            if mo:
                hits["bpp_held_object_word"].append((fo, os.path.basename(p), mo.group(0), ins[:160]))
                mv_obj = re.search(r"\b(pick\w*|grab\w*|take\w*|lift\w*|move\w*)\b(?:(?!\b(?:from|with|and|then|into|onto|on|in)\b).)*?" + obj_re.pattern,
                                   ins, re.I)
                if mv_obj:                                   # the held object is the MOVED object in the instruction
                    hits["bpp_held_object_moved"].append((fo, os.path.basename(p), mv_obj.group(mv_obj.lastindex), ins[:160]))
            # held pairs in the text: the moved phrase (pick ... up to from/with/and) names the object, the place phrase the target
            mp = re.search(r"\b(?:pick\w*|grab\w*|take\w*|lift\w*|move\w*)\b(.*?)(?:\bfrom\b|\bwith\b|\band\b|,|$)", ins, re.I)
            tp = re.search(r"\b(?:place\w*|put\w*|drop\w*|set\w*|position\w*)\b.*?\b(?:in|on|into|onto|inside|atop)\b(.*)$", ins, re.I)
            if mp and tp:
                for o, t in BPP_HELD_PAIRS:
                    ow = {"die": r"\b(die|dice)\b"}.get(o, rf"\b{o}s?\b")
                    if re.search(ow, mp.group(1), re.I) and re.search(rf"\b{t}s?\b", tp.group(1), re.I):
                        hits["bpp_held_pair_text"].append((fo, os.path.basename(p), f"{o}-{t}", ins[:160]))
    res["instruction_hits"] = {k: dict(n=len(v), by_folder=dict(collections.Counter(x[0] for x in v)), by_word=dict(collections.Counter(x[2].lower() for x in v if len(x) == 4)), examples=v)
                               for k, v in hits.items()}
    idx = json.load(open(os.path.join(DATA, "derived", "train_index.json")))
    res["index"] = dict(n_files=idx["n_files"], same_files=sorted(idx["paths"]) == files, samples=len(idx["samples"]))
    res["ok_folders_and_pairs"] = not (res["in_held_folder"] or res["in_excluded_folder"] or res["not_a_training_folder"]
                                       or bad.get("held_pair") or bad.get("held_object") or bad.get("no_metadata"))
    json.dump(res, open(os.path.join(R, "split_verify.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "instruction_hits"}, indent=1))
    print(json.dumps({k: dict(n=v["n"], by_folder=v["by_folder"], by_word=v["by_word"], ex=v["examples"][:6]) for k, v in res["instruction_hits"].items()}, indent=1))


if __name__ == "__main__":
    main()
