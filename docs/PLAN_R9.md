# Probe R9: a failure-gated retry, feasibility from existing records (2026-10-26; EXPLORATORY; stopped before any GPU work)

**Proposed (coordinator, while the user sleeps).**
- After a release, a deployable gate (the SigLIP region anchor + the policy's own state) checks whether the object is in the parsed
  region. A privileged oracle-gate arm would use sim poses.
- If the gate says wrong, the policy keeps running with a fresh noise seed and the original instruction, within LIBERO's budget.
- Constraints: frozen weights, no demos, no simulator lookahead.
- First, a cheap feasibility check from existing records: stop without GPU work if the retry is infeasible.

**Check.** Code: `experiments/r9_feas.py`, run on CPU from records only: results/r9/feas.json, ledger none.
- **Records:**
  - R7's `start` runs: 42 wrong-goal episodes x 32 seeds. Each run continues after a release, with fresh noise at every query,
    until success or the 400-step budget.
  - The 42 recorded zs episodes (fresh_d4).
- **Release:** the first executed open command after the first sustained grasp (>= 3 consecutive closed queries).
- **Re-grasp:** another sustained grasp after that release.

## Result

| | zs episodes (42) | R7 seed runs from the start (1,344) |
|---|---|---|
| episode length (failures) | 400 in all 42 (LIBERO marks no failure; the episode runs to its budget) | budget 400 |
| reach a post-grasp release | 36 | 1,214 |
| steps left after the first release (median / 25th pct) | 262 / 252 | 260 / 237 |
| re-grasp after the release (spontaneous) | 0.44 (:16 1.00, :29 0.11, :40 0.00) | 0.44 (:16 0.95, :29 0.07, :40 0.43) |
| successes | 0 (by selection) | 75, of which 21 after a re-grasp (:16 12, :29 8, :40 1) |
| failures that never release | 6 (all :40) | 83 (all :40) |

## Reading and decision: R9 as specified is a no-op, so it is not run

1. **Time is not the constraint.** About 260 steps remain after a wrong release, and LIBERO keeps the episode running.
2. **The specified retry is what the baseline already does.**
   - After a wrong release, the zs policy and every R7 run keep acting under the original instruction.
   - Every query draws fresh flow noise. A gate that fires and then "continues with a fresh noise seed and the original instruction"
     leaves the action distribution unchanged.
   - So the smoke would measure zs against itself, and the gate's precision and recall would not matter for success.
3. **The policy already re-picks spontaneously,** but task-dependently:
   - :16: almost always (0.95), with 12 late successes;
   - :29: rarely (0.07). Once the bowl is on the cabinet top, the policy treats the task as done;
   - :40: in 0.43 of released runs. In 6 of 9 zs episodes it never releases the pan at all.
   The retry the proposal wants is therefore a property of the policy's own closed loop, and it is already in every zs number.
4. **What a gate could add.** It helps only if, on firing, it CHANGES something the policy conditions on or does. Two examples:
   - re-issuing a pick-up sub-instruction from the parse ("pick up the black bowl"), then the original one;
   - another prompt-level reset.
   That is a different operator from the one specified: it uses the parse's sub-goals. It would need its own design and the user's
   scope decision, because R8 showed that prompt-level re-conditioning can harm.

**Cost.** 1 CPU job; 0 A40-h.

**Disclosures.**
- The release / re-grasp definitions were fixed in r9_feas.py before it ran, but were not pre-registered: there is no ledger row,
  since no smoke was run.
- Success timing in R7's records is the run's step count at success.
