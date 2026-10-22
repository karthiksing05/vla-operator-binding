# Experimental design: what we must show, how, and against what (2026-10-21)

This doc fixes the design before the framework is iterated. It answers docs/DESIGN_REVIEW.md, and it serves the user's goal
(docs/FRAMEWORK.md): **a hierarchical store of reusable prototypes, combined without supervision, that learns OOD tasks
efficiently and continually, as an alternative to traditional fine-tuning.**

## 1 The claims, stated so that each can fail

| # | claim | what would refute it |
|---|---|---|
| C1 Efficiency | On a new task, the store reaches a target success with a fraction of the budget (demonstrations, gradient steps, practice episodes, GPU-seconds) that fine-tuning needs | budget-to-threshold no smaller than per-task LoRA / full SFT |
| C2 Systematicity | Success on a held-out combination is predicted by whether its parts are in the store: high when all parts are held, low when a part is missing, until the store creates that part | no interaction between grade (parts held vs missing) and arm |
| C3 Reuse and hierarchy | The store grows with the number of distinct parts, not with tasks. Higher nodes are reused across more tasks. Removing the hierarchy (a flat store) costs success or efficiency | bytes linear in tasks; flat store equal on every metric |
| C4 Continual | Learning task k never lowers success on tasks < k (preservation), and the cost of task k falls with k (forward transfer) | backward transfer < 0, or no downward trend in cost |
| C5 Unsupervised | C1-C4 hold with units discovered from raw demonstrations. No component labels, instructions or segment annotations are used to build or select units. Labels are used only to score | the label-free store far below the label-supervised one |
| C6 Composition is the cause | The gains vanish under mechanism controls: shuffled unit assignment, a zero code, nearest-segment retrieval, and a store that leaves the target task out | controls as good as the method |

## 2 Benchmarks

The review's main threat was that grade is confounded with benchmark. The fix: every grade is formed **inside one benchmark
by a factorial split**, the split is fixed before any result, and one third of the test cells is **sealed** until the final
read.

**B1. CompoSuite (primary, factorial, clean).** robosuite / MuJoCo. 4 axes × 4 values: robot (IIWA, Jaco, Kinova3, Panda),
object (box, dumbbell, plate, hollow box), obstacle (none, wall, object door, goal wall), objective (pick-place, push,
shelf, trashcan). That gives 256 tasks, with expert offline data for all of them (Hussing, Mendez et al.,
[arXiv 2307.07091](https://arxiv.org/abs/2307.07091), with official train/test lists).
- We train our own flow-matching policy on the training cells, so there is no pretraining contamination.
- Grades:
  - **G-rec:** every value seen, the combination unseen (systematic recombination).
  - **G-new:** one value never seen (a new part: creation is required).
  - **G-new2:** two unseen values.
- The factors are used only to build the split and to score (C5). The policy is conditioned on a free per-task embedding
  that carries no factor structure.
- The axes give a natural parts × relations structure: objective is the relation, object and obstacle are the parts, and
  robot is the embodiment.

**B2. LIBERO-90 with frozen VLAs (realism: vision, language, large pretrained flow policies).**
- G0's design: 67 single-step pick-and-place tasks in 5 folds, where every unit is refit without the fold's tasks. It
  becomes a sealed-fold protocol: folds 1-4 for development, fold 5 sealed.
- Grades within it: object and goal both held (recombination), or one of them unseen.
- pi0.5 first; GR00T N1.5 / SmolVLA for architecture generality.

**B3 (later, realism).** EgoDex (human hands, offline metric) or CALVIN chains, once B1 / B2 settle the method.

## 3 Protocol

- **One demonstration per new task.** It is used only as the query; it is never executed and never scored.
- **Practice** is optional. When an arm uses practice episodes, they are counted in its budget, in every table.
- **Budgets:**
  - gradient steps 0 / 25 / 50 / 100 / 200 / 400 / 800;
  - demonstrations 1 / 2 / 5 / all;
  - practice 0 / 10 / 50;
  - GPU-seconds logged for everything.
- **Streams for C4:** the new tasks arrive in two mirrored random orders. Success on all earlier tasks is re-measured
  after every 10 arrivals.
- **Episodes and statistics:**
  - 50 episodes per task in confirmatory reads (20 in development).
  - IQM with stratified bootstrap CIs (rliable); paired sign tests per task; one Holm correction per read.
  - Kaplan-Meier "fraction of tasks converged vs budget", with a log-rank test, for C1.
  - The grade × arm interaction (mixed-effects logistic, task as a random effect) for C2.
- **Tuning parity:** every arm gets the same hyperparameter sweep budget, on development cells only.
- **Pre-registration:** every confirmatory read has a prereg row (ledger m3_prereg) before its first episode, as now.
  Sealed cells are read once.

## 4 Baselines

The set is chosen to answer one question: **is this method of compositionality the strong one?**

**Fine-tuning (what the store replaces):**
- per-task LoRA (rank 16 and rank 4);
- full SFT;
- multi-task SFT on all training cells plus the new task's demonstration;
- the policy alone (zero-shot).

**Continual fine-tuning (C4):**
- sequential LoRA (the 2026 finding that pretrained VLAs resist forgetting makes this a real baseline: Liu et al.
  [arXiv 2603.03818](https://arxiv.org/abs/2603.03818));
- experience replay;
- EWC;
- per-task adapters (TAIL-style, which preserve by construction).

**Other ways to compose (the decisive group):**

| baseline | what it composes | why it matters |
|---|---|---|
| Composable diffusion / PoCo ([arXiv 2402.02511](https://arxiv.org/abs/2402.02511)) | scores of whole per-task policies | composition by score sum, with no units or hierarchy |
| Wang et al. prototype PoE ([arXiv 2605.07078](https://arxiv.org/abs/2605.07078)) applied to actions | action-space modes, selected by coverage | our own earlier port (q3); tests whether prototypes must live in conditioning space |
| LoraHub / Arrow routing / TIES merging | existing per-task LoRAs, weighted from the one demonstration | LLM-style modular composition on the same units |
| Task arithmetic / LoRA averaging | per-task LoRAs | the simplest parameter composition |
| SPiRL-style flat latent skills (skill prior + posterior from the demonstration; [arXiv 2010.11944](https://arxiv.org/abs/2010.11944), [arXiv 2107.10253](https://arxiv.org/abs/2107.10253)) | a flat code space with a learned prior | our units without the hierarchy and without chunk composition |
| FIST, few-shot imitation with skill transition models ([arXiv 2107.08981](https://arxiv.org/abs/2107.08981)) | latent skills plus retrieval, from few demonstrations | the closest few-shot skill-composition method |
| BUDS / LOTUS (skill discovery and libraries) | discovered skill segments | unsupervised libraries |
| STRAP / nearest-segment retrieval | stored segments | retrieval without composition |

**Oracle (upper bound, not a competitor):** the same store built with the true factor labels (supervised units), and on B1
the multi-hot factor-conditioned policy.

**Ablations of our framework (each is one principle removed):**
- a flat store (no taxonomy);
- no chunking (units only, no composites);
- no content/context split (one taxonomy);
- no PoE within a step (a single prototype per segment);
- no creation;
- no consolidation (no merge / split);
- action-space prototypes in place of conditioning-space ones;
- a fixed segmentation (contact events) in place of parsed chunks.

## 5 Metrics per claim

| claim | primary metric | secondary |
|---|---|---|
| C1 | KM budget-to-threshold (τ = 80% of the all-demos multi-task SFT ceiling, per task) | AUC of success vs log budget; success at budget 0 / 1 demo |
| C2 | grade × arm interaction | success as a function of the coverage of the task's parts by the store |
| C3 | bytes and nodes vs tasks; reuse count per node, by depth | flat-store ablation |
| C4 | backward transfer, forward transfer (LIBERO's FWT / NBT / AUC), cost per arrival vs index | routing errors on old tasks after growth |
| C5 | label-free vs label-supervised store, on C1-C4 | unit purity against the factors (diagnostic only) |
| C6 | each control vs the method | |

## 6 Order

1. Build B1 (CompoSuite data, our flow policy, the factorial split with sealed cells) and its fine-tuning baselines. This
   gives the targets.
2. Iterate the framework (docs/FRAMEWORK.md) on B1's development cells.
3. Port to B2 (LIBERO-90 sealed fold) with the frozen VLAs.
4. Confirmatory reads: sealed cells, all baselines, pre-registered.
