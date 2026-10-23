# Plan L4: a continual store of language prototypes (draft, 2026-10-22)

**Why this stage.** L2 (docs/PLAN_LANG.md, "L2 results") showed that a frozen VLA can be repaired for a new task by
*composing stored language states* (text latents of training sentences, word states), chosen from one demo, with no
gradient and no practice. On pi0.5 this beat zero-shot (91 vs 63 / 160) and matched Li's hand-picked pairs (84). That is
the end goal's core operation: an unsupervised combination of stored prototypes. L4 turns it into the full loop:
a hierarchical store that grows, is reused, and never damages the policy.

## 1. The store (hierarchical by granularity)

| level | prototype | source | what it carries (L0 trace) |
|---|---|---|---|
| sentence | text latent of a whole training or repaired sentence | training prompts; every accepted repair | the whole task: approach, carry, release |
| phrase / role | the object phrase or goal phrase state, keyed by role | spans of stored sentences | where the motion points while approaching (object) or carrying (goal) |
| word | a single word's layer-0..2 state | spans of stored sentences | the referent; L0 shows binding is decided at layers 0-1 |

Every prototype is executable: putting it in a prompt slot and re-evaluating the frozen policy at the current observation
gives a live action field. Prototypes are keys into the policy's own field, not stored actions.

## 2. Composition (the relation)

The operators measured in L2, used as the composition relation, cheapest first:
- **interpolate** two sentence prototypes (TLI, weight lambda);
- **transplant** a phrase or word prototype into a role of the new sentence (layers 1-k);
- **schedule** roles in time (object before the grasp, goal after; the P3 AND / THEN readout).

Selection is unsupervised: no labels, no rewards, no practice. One demo scores candidates by flow loss under a rule chosen
by L2b (the criterion study: demo-0 loss, split-demo validation, minimal-edit MDL penalty, zero-shot guard). The rule is
fixed once on dev tasks and then frozen.

## 3. The continual loop

For each arriving task (instruction + one demo):
1. **Compose.** Search candidates over the *current* store (training prototypes plus everything added so far).
2. **Accept or keep zero-shot**, by the frozen rule.
3. **Store.** An accepted composition becomes a new sentence prototype, and its role spans become new phrase prototypes,
   keyed by their words. Nothing in the policy changes, so old tasks cannot be damaged.
4. **Create only when composition fails** (L3): one inverted token for the failing word (about 2 KB). The rank-1 edit is
   dropped (L0b: no compact binding site).

## 4. What L4 must show

| claim | measure | pass |
|---|---|---|
| reuse | fraction of later tasks whose chosen candidate uses a prototype added by an earlier task; causal check by removing it | > 0, and removal lowers success |
| forward transfer | success of a task when it arrives late vs early in the stream (two mirrored orders) | late > early |
| no damage | training-task success with the full store | change >= -0.01 (expected 0, since the policy is frozen and the prompt for old tasks is unchanged) |
| cost | GPU-seconds and bytes per arrival vs per-task LoRA | <= 1/100 of LoRA bytes |

**Stream design.** Li's 20 OOD tasks plus LIBERO-90-C held-out cells (once pi05_l90c exists), in two mirrored random orders.
Order pairs so that a task sharing a part (an object or a goal) with a later task arrives first. That makes reuse possible,
and the mirrored order measures forward transfer.

## 5. Order of work

1. L2 confirm read (running): does the frozen l2sel rule hold on the 12 held-out tasks on pi0.5?
2. L2b (running): which offline criterion predicts closed-loop success, especially on pi0?
3. Pre-register L4 with the L2b rule, on pi0.5 first. L4 only adds the store-update step; it needs no new operators.
4. pi0 joins once a rule is found that does not collapse on pi0 in closed loop.

## 6. How it meets the end goal

- **hierarchical store:** sentence > phrase > word prototypes, all executable.
- **unsupervised composition:** one demo, flow-loss selection, no labels or rewards.
- **OOD efficiency:** no gradient, about the cost of scoring candidates on one demo.
- **continual:** the store grows by accepted compositions, the policy is frozen.
- **interpretable:** every repair names its source sentences, roles and layer span.

**Relation to plan v4.** Plan v4's action-space prototypes (modes under part prompts) await pi05_l90c. If they become
readable there, they enter this store as a fourth, action-level layer.
