# The compositional split, graded by what is unseen

## Motivation

Robot manipulation is an ideal testbed for compositionality. A LIBERO task is a composition of parts: an object, its location
or instance, a goal relation and its location, set in a scene. The 40 training tasks expose the parts. The test tasks recombine
them.

Following Zekun Wang's criterion ("a set with parts and relations unseen"), every test task is graded by what it leaves unseen.
The grade measures how far prior knowledge must be bent to serve a new task: how expertise is donated in unorthodox ways. This is
the task-to-task transfer that cognitive science documents in humans, from Fodor's systematicity onward.

## The training set and the parts

The training set is pi0.5's 40 tasks (LIBERO-Spatial / Object / Goal / 10). It is fixed by the frozen checkpoints of every
policy we test (each was fine-tuned on those 40), so it is not chosen by us.

Parts are read by rule from each task's BDDL file (`lifelong_ood/policy/factors.py`), with one normalisation applied to every task
alike (`experiments/split_grades.py: norm`):
- **object**: the moved object's type;
- **object location**: its instance qualifier;
- **goal relation**: the placing predicate and target type;
- **goal location**: the region, when it names a location such as a drawer or a side;
- **scene**: the layout. Li's OOD suites use their base suite's table, with objects added.

## Grades

| grade | unseen | example |
|---|---|---|
| G0 seen | nothing: the (object, relation) pair in this scene is a training task | LIBERO-90 copies of LIBERO-10 steps |
| G1 relation | the pairing only: object and goal each seen in this scene, never together | Li's "put the cream cheese on the plate" |
| G2 binding | a part never seen in this scene, though seen elsewhere | Li's "put the butter on the plate" (butter only in LIBERO-Object) |
| G3 scene | the scene; every part seen in other scenes | LIBERO-90 kitchens and living rooms not in LIBERO-10 |
| G4 part | an object, goal relation or predicate never seen at all | LIBERO-90's ketchup in the tray, the caddy, the frying pan |

Counts over the 110 evaluated tasks: G0 15, G1 9, G2 20, G3 18, G4 48.

## Results by grade

Successes are summed per grade from the per-task reads (`results/robot/split_grades.json`).

| result | grade | tasks | the policy alone | composition, zero fit | fitted units |
|---|---|---|---|---|---|
| Li OOD, pi0.5 (G15) | G1 | 6 | 59 | 116 (chosen) | — |
| | G2 | 14 | 84 | 196 | — |
| Li OOD, pi0 (G15) | G1 / G2 | 6 / 14 | 11 / 24 | 101 / 235 | — |
| Li OOD, SmolVLA / X-VLA (G18) | G1 | 6 | 4 / 43 | 26 / 97 | — |
| | G2 | 14 | 1 / 37 | 70 / 159 | — |
| Li OOD, FLOWER-goal (Z1b) | G1 | 6 | 37 | 48 | — |
| LIBERO-90, pi0.5 (Z3, G8; set d) | G0 | 10 | 173 | 166 (pointers) | 161 library, 151 LoRA |
| | G2 | 5 | 0 | 0 | 56 library, 70 LoRA |
| | G3 | 8 | 68 | 67 | 154 library, 136 LoRA |
| | G4 | 2 | 4 | 0 | 35 library, 40 LoRA |
| LIBERO-90, pi0.5 (G8, all 66) | G3 | 15 | 90 | — | 225 leaves, 215 library |
| | G4 | 31 | 117 | — | 455 leaves, 403 library |

## Reading

When only relations or bindings are new (G1, G2) and the scene is familiar, composing held units, with nothing fitted, recovers
most of the behaviour. This holds on all five flow-matching architectures.

When the scene or a part is new (G3, G4), composing held units does nothing. A unit must be created, and created units compose
(the library and leaves beat the per-task LoRA).

LIBERO-90's G2 tasks need creation too. There the unseen binding is to a scene the policy knows only from LIBERO-10's two-step
tasks.

The grade predicts which of the two learning regimes a task needs:
- **reuse:** the supervised pointers, or the unsupervised segment-named programs;
- **create:** fitted prototypes and leaves.
