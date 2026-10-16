# Method: a factored, hierarchical prototype library for a frozen policy

The task is [THE_TASK.md](THE_TASK.md): a new task arrives with its instruction, its scene and at most one demonstration; new start
states out; storage charged; nothing the learner held may be damaged. The results are in [RESULTS.md](RESULTS.md); the earlier
methods (role codes, folded tokens, step primitives) are in [history/](history/).

## The units

A **prototype** is a rank-4 LoRA on the action expert of the frozen pi0.5 (`lifelong_ood/policy/adapt.py`), fitted by flow
matching on demonstration frames and switched on only during its phase (`primitives.PrimHooks`). There are three kinds, read off
the instruction (`lifelong_ood/policy/primitives.py`):

| kind | key | frames it is fitted on | acts |
|---|---|---|---|
| OBJECT (type) | `object_key`: verb and object, the instance word removed ("put / the black bowl") | each demonstration's frames before its first sustained grasp | from the start until the policy's own grasp |
| INSTANCE (child of a type) | `instance_key`: the object key and the named instance ("... / middle") | the same, for demonstrations that name that instance | in place of its type when the instruction names it |
| GOAL | `goal_key`: preposition and target ("in / the basket") | each demonstration's frames from its first grasp to its release | from the policy's grasp to its release |

Every prototype is consolidated over every demonstration of its concept, each demonstration taken from the task's own scene
(`data.Demos90.episodes_of_task`; the dataset merges tasks that share an instruction, see RESULTS.md, "DATA DEFECT"). A task is
the composition of its object (or instance) prototype and its goal prototype; a missing factor is left to the policy.

**A concept the policy already holds** (read G4b): when a concept's demonstrations are ones the policy was trained on, a fitted
prototype learns nothing (its flow loss does not fall). Such a concept is stored as the policy's own field under the instruction
of a training task that taught it (the source task in the test scene's base suite, else the first), at zero bytes; the step
runs under the object concept's instruction until the policy's grasp and under the goal concept's after it (`PrimHooks`
`phase_prompts`; arm `pfield@<object source>.<goal source>`). Practice chooses between the policy, the prototype composition and
this field composition. Both kinds of unit change the policy's velocity field for one phase: a prototype adds a low-rank change,
a held concept selects the conditional field the policy already has. The scope is flow-matching and diffusion policies.

## The learner

1. **Compose** the arrival from the current library.
2. **Practise**: ten episodes from stored start states for the composition and for the policy alone. Keep the better (ties: the
   policy). If it succeeds in at least 8 of 10, request nothing.
3. Otherwise **request the one demonstration**. It joins every concept on its path (instance child, type, goal), each is refit
   on all its demonstrations, and the refit composition is practised again; it is kept only if it beats the policy.
4. **Serve** every task with the final library where it kept the library, else with the policy.

`experiments/w1_stream.py` runs this rule; `experiments/e4_stream.py` (an earlier, looser rule) and `experiments/w2_stream.py`
(re-checks after every refit) are its variants.

## How it is tested

Pre-registered before any episode (`experiments/prereg/`, `ledger/m3_prereg`); scored on fresh start states from LIBERO's own
placement sampler at seeds no design decision has seen; paired episodes (same start state, same noise for every arm); exact sign
test on discordant pairs and a two-level bootstrap over tasks and episodes; the strong baseline is a one-demonstration LoRA per
task (rank 16, the task's own scene) given the same acceptance test.
