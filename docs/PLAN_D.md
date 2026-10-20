# Plan D: what a prototype donates versus what the demonstration teaches, and the full baseline set (user direction, 2026-10-20)

The user asked for two things:
1. Interpretation of the information a prototype donates during composition, and of how it donates it. The question is which part of the
   action the prototype supports and which part the demonstration teaches: an action breakdown that measures the value of a
   prototype. This matters most for the unsupervised scheme.
2. Every frequently asked baseline (SFT, RL and others) and the published solutions, trained so that they fix the value of
   prototype composition.

## D1 The action breakdown (three sources, every action)

Each served action chunk on a held-out task splits into three nested fields. Every one already exists as a backbone arm:

    a_policy       the frozen policy under the task's instruction
    a_proto        the composition of held prototypes (pointers / library units; supervised) or of named units (unsupervised)
    a_demo         the leaf fitted on the one demonstration, initialised from a_proto (zf_leaf_fit --init-best)

    donation  = a_proto - a_policy     what prior knowledge supplies
    taught    = a_demo  - a_proto      what the demonstration had to add

**Measures** on the held-out demonstration 1 and on rollouts, all broken down the same way:
- by action component: translation xyz, rotation, gripper (the dim groups of blend.py);
- by phase: approach, grasp, carry, release, cut at the policy's own gripper events (program.segments);
- by flow time t (the coarse versus fine field, as in h1_donate);
- by grade G1 to G4 (docs/SPLIT.md).

**Readouts:**
- **Error share:** the share of the policy's error to the demonstration that the prototype removes, versus the share that only the
  demonstration removes, per component and phase (the variance explained by each source).
- **Causal swap:** serve each component group from one source and the rest from another. Example: a_proto's translation with
  a_demo's gripper. It uses BlendHooks with one-hot group weights, so no new sampler is needed. The success of each of the 2^3
  assignments gives a per-component Shapley value of "prototype versus demonstration" in closed-loop success. This is the
  action breakdown in success units.
- **Value of a prototype in demonstration units:** learning curves of the leaf over fit steps (50, 100, 200, 400, 800) and over
  the number of demonstrations (1, 2, 5). One curve starts from a_proto, the other from a_policy (the plain SFT/LoRA initialisation).
  A prototype is worth the number of demonstrations, or fit steps, that the start from a_policy needs to match the start from
  a_proto. The value is reported per component and grade.
- **What predicts value:** regress per-task value on the donation's size, its phase, the concept kind (object / goal / instance),
  the grade and the policy's flow loss. The goal is a rule that tells, before fitting, which components to take from a prototype
  and which to learn.

**Unsupervised scheme:** the same three sources with a_proto equal to the soft blend of named contact-cut units (blend.py, Z5). For
each contact segment, the read reports how much of each component the named units explain and how much the demonstration fit
must teach. A segment whose taught share is high in every component is a candidate new unit; this joins the queued
creation/hierarchy work.

Policies: pi0.5 first, then pi0, SmolVLA, X-VLA and GR00T on the G20 / Z3 task sets that already carry pointers, prototypes and
leaves. CALVIN (FLOWER) and EgoDex (GR00T) follow once their units exist.

## D2 The baseline set

Already run (paper sections "Against ..."; docs/RESULTS.md):
- one-demonstration per-task LoRA, with and without acceptance (the SFT-LoRA / TAIL family);
- sequential fine-tuning, experience replay, joint multi-task (G11, F1);
- task arithmetic and LoRA averaging (G2);
- STRAP (G6);
- DSRL on the policy, on the per-task adapter and on the library (G7, G7b);
- Li's interpolation and TLI.

To add (pre-registered, paired episodes, same task sets and start states):

| baseline | family | what it answers |
|---|---|---|
| full-parameter SFT on the one demonstration (action expert; and whole model where memory allows) | SFT | is the low-rank adapter the reason prototypes win? |
| multi-demonstration SFT (5 and 50 LIBERO demonstrations per task) | SFT ceiling | how many demonstrations equal one demonstration plus prototypes |
| EWC and PackNet on the W1 stream (LIBERO's own lifelong baselines) | continual | the canonical CL regularisers on the same stream as G11 |
| pi_RL / RLinf PPO on the flow policy (pi0, pi0.5; published LIBERO recipes, few-shot SFT then RL) | RL on weights | RL fine-tuning at the published budget and at our practice budget |
| ReinFlow (flow-policy RL with learned noise) | RL on weights | a second published flow-RL method |
| LOTUS or IsCiL (skill-library continual imitation; whichever has runnable code) | published library | a learned library against ours |
| Sparse Diffusion Policy (experts with a router) | published modular | routed experts against pointer composition |
| DemoDiffusion (one demonstration, no training) | published one-shot | re-run on the current task sets |

**Rules:**
- Budgets are reported two ways: at the method's published budget, and matched to ours (1 demonstration plus 10 practice
  episodes, or the same number of GPU-hours).
- Each baseline must reproduce its published number before it is read.
- RL runs are restricted to the G3 / G4 task subset, where creation is needed, unless the budget allows more.
- All training and rollouts run on SLURM (overcap, resumable, at most 4 h per job).

## Order

1. D1 offline error shares on pi0.5 (cheap, no rollouts).
2. D1 learning curves.
3. D1 causal swaps.
4. D2 inventory and parity runs, then pre-registration, then reads, in this order: full SFT, multi-demonstration SFT, EWC and
   PackNet, pi_RL, ReinFlow, the published library methods.
