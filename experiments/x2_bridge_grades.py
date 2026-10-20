"""Plan X / X2: grade SimplerEnv object x target pairs against EVERY instruction of Bridge V2 (the WidowX checkpoints' training
data), not only SimplerEnv's 4 tasks.  Source: IPEC-COMMUNITY/bridge_orig_lerobot meta/tasks.jsonl (19,974 distinct instructions;
language only).  A part is seen if some instruction names it; a pair is seen if some instruction names both (an over-count: "take
carrot off plate" counts), so a pair with no co-mention is an unseen relation (G1) and a part never named is an unseen part (G4).

    python3 -I experiments/x2_bridge_grades.py <tasks.jsonl> [--record]
"""
import json
import re
import sys

SYN = {"carrot": r"carrot", "spoon": r"spoon", "eggplant": r"eggplant|aubergine", "green block": r"green (?:block|cube)",
       "yellow block": r"yellow (?:block|cube)", "towel": r"towel|cloth|rag", "plate": r"plate|dish", "basket": r"basket",
       "coke can": r"\bcoke\b|\bcola\b", "apple": r"\bapples?\b", "sponge": r"\bsponges?\b", "orange": r"\borange\b(?! (?:cloth|towel|block|cube|bowl|plate))",
       "pepsi can": r"\bpepsi\b", "redbull can": r"\bred ?bull\b", "7up can": r"\b7 ?up\b|\bsprite\b"}
PAIRS = [("carrot", "towel"), ("spoon", "plate"), ("eggplant", "plate"), ("green block", "plate"), ("carrot", "yellow block"),
         ("carrot", "basket"), ("coke can", "plate"), ("coke can", "towel"), ("apple", "plate"), ("apple", "towel"), ("sponge", "plate"),
         ("pepsi can", "towel"), ("redbull can", "plate"), ("7up can", "towel")]


def grade(T):
    out = {}
    for o, g in PAIRS:
        no = sum(bool(re.search(SYN[o], t)) for t in T)
        ng = sum(bool(re.search(SYN[g], t)) for t in T)
        both = sum(bool(re.search(SYN[o], t) and re.search(SYN[g], t)) for t in T)
        grade_ = "G4 part" if no == 0 or ng == 0 else "G1 relation" if both == 0 else "seen"
        out[f"{o} -> {g}"] = dict(object=no, target=ng, together=both, grade=grade_)
    return out


if __name__ == "__main__":
    T = [json.loads(l)["task"].lower() for l in open(sys.argv[1])]
    res = grade(T)
    for k, v in res.items():
        print(f"{k:26} object {v['object']:5} target {v['target']:5} together {v['together']:4}  {v['grade']}")
    if "--record" in sys.argv:
        sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
        from lifelong_ood import ledger
        print(ledger.record("x2_bridge_grades", bench="bridge_v2", tag="read", seed=0, metrics=dict(instructions=len(T), pairs=res),
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/x2_bridge_grades.py", command=" ".join(sys.argv)))
