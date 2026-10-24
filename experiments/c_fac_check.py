"""Pre-check of pi05_l90c_fac's factored-conditioning mixture (docs/L90C.md 3.1) before it trains: CPU only.

Checks, on the DEV split file only (results/l90c/split.json; the sealed file is never opened, only stat'ed):
  A  the prompt table (results/l90c/fac_prompts.json) is keyed by training-task instructions only, and every training
     instruction has an entry; full-instruction texts that a held-out task shares verbatim are listed (inherited from
     pi05_l90c's data; their demos are not in the data);
  B  the training data (lol/libero_l90c: data_manifest.json, meta/tasks.jsonl, meta/episodes.jsonl) holds training tasks only,
     so every part prompt is paired only with training-cell demos; every dataset text hits the table (FactoredPrompt's lookup);
  C  every part prompt is built from its own instruction's words and names no part atom (ketchup / caddy / turn on);
  D  compounds: the factored pairs (part prompt of task t, demo of t) expose only t's own atoms and compounds; no training
     task carries a held-out compound (recomputed with c_split's rule), and no (object part, goal part) pair of any table
     entry is a held-out object x target compound;
  E  sealed: no training task, manifest task or table text belongs to a sealed task (salted hashes in split.json);
     split_sealed.json's mtime / size are recorded (not read).

    python experiments/c_fac_check.py [--record]     (sbatch: slurm/jobs/l90c_cpu.sbatch ENV=chunks)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
R = os.path.join(ROOT, "results", "l90c")
DATA = "/coc/flash10/ksingara3/robot/data/hf/lerobot/lol/libero_l90c"
PART_WORDS = ("ketchup", "caddy", "turn on")


def words(s):
    return set(re.findall(r"[a-z]+", s.lower()))


def jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def main(record):
    from lifelong_ood import l90c
    from lifelong_ood.policy import factors as F
    import c_split as CS
    s = l90c.load_split()
    table = json.load(open(os.path.join(R, "fac_prompts.json"), encoding="utf-8"))
    manifest = json.load(open(os.path.join(R, "data_manifest.json"), encoding="utf-8"))
    train = list(s["train"])
    lang = s["languages"]
    held_tids = sorted(set(lang) - set(train))                  # dev bins + dev part + two-step tests (sealed: not listed)
    held_c = set(s["held_compounds"])
    res, fail = {}, []

    # A: table keys
    train_texts = {lang[t] for t in train}
    res["A"] = dict(table_entries=len(table), train_texts=len(train_texts),
                    keys_not_training=sorted(set(table) - train_texts), training_without_entry=sorted(train_texts - set(table)),
                    texts_shared_with_heldout={x: sorted(t for t in held_tids if lang[t] == x) for x in sorted(table)
                                               if any(lang[t] == x for t in held_tids)})
    if res["A"]["keys_not_training"] or res["A"]["training_without_entry"]:
        fail.append("A")

    # B: the data holds training tasks only
    man_tasks = set(manifest["per_task"])
    tasks_jsonl = jsonl(os.path.join(DATA, "meta", "tasks.jsonl"))
    ds_texts = {r["task"] for r in tasks_jsonl}
    ep_texts = {}
    for r in jsonl(os.path.join(DATA, "meta", "episodes.jsonl")):
        for x in r.get("tasks", []):
            ep_texts[x] = ep_texts.get(x, 0) + 1
    res["B"] = dict(manifest_tasks=len(man_tasks), manifest_not_train=sorted(man_tasks - set(train)),
                    train_not_in_manifest=sorted(set(train) - man_tasks),
                    manifest_text_mismatch=sorted(t for t, v in manifest["per_task"].items() if v["text"] != lang.get(t)),
                    dataset_texts=len(ds_texts), dataset_texts_not_training=sorted(ds_texts - train_texts),
                    dataset_texts_without_parts=sorted(x for x in ds_texts if not table.get(x)),
                    episodes=int(sum(ep_texts.values())), episode_texts_not_training=sorted(set(ep_texts) - train_texts),
                    manifest_episodes=int(manifest["episodes"]))
    if (res["B"]["manifest_not_train"] or res["B"]["dataset_texts_not_training"] or res["B"]["episode_texts_not_training"]
            or res["B"]["dataset_texts_without_parts"] or res["B"]["manifest_text_mismatch"]):
        fail.append("B")

    # C: part prompts come from their own instruction and name no part atom
    bad_words, part_named = {}, {}
    for x, parts in table.items():
        for p in parts:
            if not words(p) <= words(x):
                bad_words.setdefault(x, []).append(p)
            if any(w in p.lower() for w in PART_WORDS):
                part_named.setdefault(x, []).append(p)
    keys_part = sorted(x for x in table if any(w in x.lower() for w in PART_WORDS))
    distinct = sorted({p for v in table.values() for p in v})
    res["C"] = dict(part_prompts=sum(len(v) for v in table.values()), distinct_part_prompts=len(distinct),
                    words_not_in_own_instruction=bad_words, names_part_atom=part_named, instruction_names_part_atom=keys_part,
                    entries_by_len={str(k): sum(len(v) == k for v in table.values()) for k in (1, 2, 3)})
    if bad_words or part_named or keys_part:
        fail.append("C")

    # D: compounds of the factored pairs
    tasks = {t.tid: t for t in F.load(("libero_90", "libero_10", "libero_spatial", "libero_object", "libero_goal"))}
    tr_comp = {}
    for t in train:
        tr_comp[t] = {str(k) for k in CS.compounds(tasks[t])}
    in_train = sorted({k for v in tr_comp.values() for k in v} & held_c)
    # (object part, goal part) of every table entry -> the (o, t) compound it pairs, read from the first In / On step
    ot_pairs = {}
    for t in train:
        st = [x for x in CS.steps(tasks[t]) if x[0] in ("In", "On") and x[2]]
        if st:
            ot_pairs[t] = str(("o:" + st[0][1], "t:" + st[0][2]))
    ot_held = sorted(t for t, k in ot_pairs.items() if k in held_c)
    # the held-out tasks' own compounds never meet a held-out task's demos: there are none in the data (B)
    held_task_comp_in_pairs = sorted({k for t in held_tids if t in tasks for k in map(str, CS.compounds(tasks[t]))}
                                     & held_c & {k for v in tr_comp.values() for k in v})
    res["D"] = dict(held_compounds=len(held_c), held_compounds_in_training_tasks=in_train,
                    object_x_goal_pairs=len(ot_pairs), object_x_goal_pairs_held=ot_held,
                    heldout_task_compounds_exposed=held_task_comp_in_pairs,
                    heldout_cell_demos_in_data=sorted(man_tasks & set(held_tids)))
    if in_train or ot_held or held_task_comp_in_pairs or res["D"]["heldout_cell_demos_in_data"]:
        fail.append("D")

    # E: sealed
    sealed_train = sorted(t for t in train if l90c.is_sealed(t))
    sealed_man = sorted(t for t in man_tasks if l90c.is_sealed(t))
    # a table text that is the language of some sealed task: sealed tids are hashed, so test every LIBERO-90 tid not in the dev file
    sealed_text = sorted(x for x in table for tid, t in tasks.items()
                         if tid.startswith("libero_90:") and t.language == x and l90c.is_sealed(tid))
    st = os.stat(l90c.SEALED) if os.path.exists(l90c.SEALED) else None
    res["E"] = dict(sealed_in_train=sealed_train, sealed_in_manifest=sealed_man, n_sealed=sum(s["n_sealed"].values()),
                    table_texts_shared_with_a_sealed_task=len(sealed_text),
                    sealed_file=dict(exists=st is not None, size=st.st_size if st else None,
                                     mtime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)) if st else None),
                    sealed_file_opened=False)
    if sealed_train or sealed_man:
        fail.append("E")
    # A text a sealed task shares verbatim with a training task is the same kind of overlap as A's dev list (the sealed task's
    # demos are not in the data); it is counted, not listed, so no log names a sealed task.
    res["pass"] = not fail
    res["failed"] = fail
    print(json.dumps(res, indent=1))
    with open(os.path.join(R, "c_fac_check.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    if record:
        from lifelong_ood import ledger
        print(ledger.record("c_fac_check", bench="libero_90_c", tag="pi05_l90c_fac", seed=0, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config=dict(p_part=0.3, p_null=0.1, table="results/l90c/fac_prompts.json"),
                            script="experiments/c_fac_check.py", command=" ".join(sys.argv),
                            artifacts=[os.path.join(R, "c_fac_check.json")]))
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main("--record" in sys.argv))
