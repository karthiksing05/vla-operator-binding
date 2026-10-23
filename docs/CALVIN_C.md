# CALVIN-C: the compositional split of CALVIN ABC -> D (plan v4 P0 / P1)

Plan: docs/PLAN_V4.md 3.2 and docs/DESIGN.md 3.2. Live status: results/calvin_c/STATUS.md. Earlier CALVIN setup and the
parity gate of the public checkpoints: docs/CALVIN.md.

## Cells, atoms, compounds

CALVIN's 21 block subtasks are three small factorial families:
- lift: colour x place (9);
- push: colour x direction (6);
- rotate: colour x direction (6).

A cell is (verb, colour, slot). Atoms are verb, colour, place (lift only) and direction (push and rotate share it).

A cell's three compounds are verb-colour, verb-slot and colour-slot:
- colour-direction compounds are shared by push and rotate;
- a colour-place compound belongs to one lift cell only, so a held-out lift cell always has an unseen colour-place compound and
  is never grade 0.

Caveat: CALVIN's colour also fixes the block's size (red middle, blue small, pink big), so the colour atom is colour+size. A
colour generalisation is also a size generalisation.

## The split (experiments/c_calvin_split.py, lifelong_ood/calvin_c.py)

These rules were fixed before any result. Per split seed:
- 7 cells are held out;
- every colour, verb, place and direction stays in at least 2 training cells;
- exactly one colour x verb pair (the part set) is held out entirely, and no other verb-colour compound is new;
- a held-out cell's grade is the number of its compounds absent from training (0, 1 or 2).

The search maximises the smallest per-grade count, then evenness, then the number of held-out cells, then verb variety, then
minimises the atom divergence. The part set is pink x rotate for seed 0 (the plan's example), and red x lift and blue x push
for seeds 1 and 2, so each verb and each colour is a part once.

| seed | part | grade counts 0 / 1 / 2 | D_A | D_C |
|---|---|---|---|---|
| 0 | pink x rotate | 3 / 3 / 1 | 0.123 | 0.288 |
| 1 | red x lift | 2 / 2 / 3 | 0.018 | 0.412 |
| 2 | blue x push | 3 / 3 / 1 | 0.123 | 0.288 |

With a push or rotate part, only one grade-2 cell is possible: a second one would need a second fully held-out verb-colour
pair.

Sealing: one third of the held-out cells (2 of 7) is sealed per seed, stratified by grade and never a grade's last dev cell.
The sealed cells go to `results/calvin_c/split_s{k}_sealed.json`, which is git-ignored. They are read only through
`calvin_c.load_sealed(seed, prereg=<m3_prereg run_id>)` with `CALVINC_UNSEAL` set to the same id. Readers call
`calvin_c.check_cell`, and training filters call `calvin_c.held_out_mask`, which uses salted hashes and never opens the sealed
file. As in b1, the hashes stop accidents, not a determined reader: the train and dev lists imply the sealed cells.

## Training data for FLOWER-C

FLOWER's CALVIN recipe trains on the language-annotated 64-frame windows of task_ABC_D/training (17,870 windows). For each
seed:
- every window of a held-out cell is dropped;
- every other window that overlaps a held-out window by more than 16 frames is dropped (144 / 251 / 171 windows);
- about 78 % of the windows remain (`results/calvin_c/windows_s{k}.{npz,json}`);
- 2-3 % of the held-out frames still sit inside kept windows, at most 16 frames per overlap.

Unannotated play is never read by the recipe's language-only dataset. Other subtasks' windows still move every block (the
colour-free push_into_drawer, stack_block and place_in_*), so the held-out motions are not absent from the data, only their
instructions.

## FLOWER-C training (experiments/c_train_flower.py)

The recipe is the published checkpoint's own config and training log:
- flower_vla_pret 360000_model_weights.pt as the initialisation;
- 4 GPUs x batch 8, 35,000 steps (35 epochs of 1000 batches; the README's "40k" disagrees with the config and the log);
- AdamW 2e-5 with the TriStage schedule, EMA 0.999, bf16-mixed, seed 242.

Deviations, also listed in the script's docstring:
1. The data filter above.
2. No in-training scene-D rollouts and no best-of model selection on the test scene: the final EMA weights are evaluated once.
3. Full Lightning checkpoints, so resume restores the optimiser, the EMA and the loop. The run length is fixed by max_steps,
   because a mid-epoch resume restarts Lightning's epoch counter.
4. The <Flow> prompt embedding (a plain tensor, never saved by the authors) is saved and restored, and written into the
   exported token row. The evaluation's load fix therefore reads the training-time value.
5. Torch 2.10's TF32 API, CSV logging, and Lightning's local launcher with NCCL_P2P_LEVEL=SYS.

Throughput on A40 nodes: about 1.1 s/step (megabot 4.6 s/step, excluded). A run costs about 12.5 h on 4 A40s, about 50
A40-h. The full-ABC control uses the same wrapper with `--full`.

## Reads

- Parity (`experiments/c_calvin_parity.py`, ledger `c_calvin_parity`):
  - the standard 1,000 chains in scene D;
  - the chains free of held-out cells (562 / 443 / 574 per seed);
  - per-subtask success on every non-held-out subtask, against flower_calvin_abc with the load fix and against the control.
- Contamination audit (`experiments/c_calvin_audit.py`, ledger `c_calvin_audit`): flow loss and its dependence on the named
  colour or slot, on dev held-out against training-cell windows.
