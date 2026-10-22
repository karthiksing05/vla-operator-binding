# Baselines of plan D2: inventory, stage 1

(docs/PLAN_D.md section D2.) The aim is to train every frequently requested baseline, and each published solution, so that together
they fix the value of prototype composition. Stage 1 covers four steps: the inventory, cloning the code, the cheap baselines'
fits and parity smokes, and draft pre-registrations. No main read has run.

## Pairing

| set | tasks | start states | arms it pairs with |
|---|---|---|---|
| E3 | 30 LIBERO-90 tasks (29 have a demonstration; task 51 has none); grades G0 6, G2 4, G3 7, G4 13 | fresh set b (sampler seeds 200000-200019), 20 per task, noise seed 13 | E3 library with acceptance 480 / 580; per-task LoRA with acceptance 494 / 580; zero-shot 273 / 600; G7 DSRL (dsrlz 313, dsrls 503 / 580 with acceptance); G6 STRAP 296 |
| W1 stream | E3's 30 tasks in W1's order a | set b, seed 13 | W1 served 465 / 600; G11 sequential FT 52, ER 335 |
| G3/G4 subset (RL) | E3's 20 G3/G4 tasks: 9 10 12 13 14 15 30; 25 31 36 41 43 55 56 57 58 59 61 62 71 | set b, seed 13 | as E3 |

Acceptance and the matched budget are as in E3. A task gets 1 demonstration plus 10 practice episodes on stored states 0-9. An arm is
kept where it beats the frozen policy there.

Cost unit: an E3 read of one arm is 600 episodes, about 4.4 A40-hours (two instances per GPU, about 18 minutes per 20 episodes;
`g11_main` secs). One-demonstration LoRA fit: about 5 minutes with the prefix cached.

## Inventory

| # | baseline | paper | code / licence | fits our policies + LIBERO? | published LIBERO numbers | cost (A40-h): published budget / our matched budget | task set, pairing | status |
|---|---|---|---|---|---|---|---|---|
| 1 | full-parameter SFT, one demonstration (action expert) | (the SFT baseline; openpi's own fine-tuning recipe) | ours: `experiments/r3_fewshot.py --full` (adapt.train_full) | yes, pi0.5 (any backbone of c0_fit) | none at 1 demo; pi_RL's "few-shot SFT" is the nearest (pi0.5, 40 trajectories ≈ 1 per task over 4 suites: 84.6 / 95.4 / 84.6 / 43.9) | fit 29 × ~6 min ≈ 3 A40-h; read 4.4 + acceptance 1.5 | E3, set b | **fitted** (stage 1) |
| 1b | full SFT, whole model (VLM + expert) | | | **no** on one A40: 3.3B params × ~16 B (bf16 weights, fp32 masters, AdamW) ≈ 53 GB before activations | | needs 2-4 A40 with FSDP sharding, or 8-bit AdamW | | not run (decision) |
| 2 | multi-demonstration SFT, 5 and all (28-50) own-scene demos per task | (SFT ceiling) | ours: `r3_fewshot.py --ndemo` | yes | LIBERO-90 multi-task, all demos: pi0.5 92.7 (from scratch) / 96.0 (from generalist) (arXiv 2505.23705 Tab. 1); pi0-FAST 83.1 (arXiv 2505.13888); FLOWER 94.7 (arXiv 2509.04996); openpi's pi05_libero is not trained on LIBERO-90 | 5 demos: 29 × ~15 min ≈ 7 A40-h; all demos: 29 × ~2.2 h ≈ 64 A40-h (prefixes recomputed per step) | E3, set b | 5-demo **fitted**; all-demo fitted on parity tasks 13, 36, 52 only |
| 3 | EWC (online, LIBERO's) | Kirkpatrick et al. 2017; LIBERO (arXiv 2306.03310) | LIBERO repo, MIT; reimplemented on the LoRA: `experiments/g30_cl.py` (e_lambda 50000, gamma 0.9, Fisher = mean squared batch gradient, as `libero/lifelong/algos/ewc.py`) | yes | LIBERO Tab. 2 (ResNet-T, FWT / NBT / AUC): Long .13/.22/.02, Spatial .23/.33/.06, Object .56/.69/.16, Goal .32/.48/.06; TAIL LIBERO-10 FWT 0.30 | ~1 A40-h fit; read 4.4 | W1 order a, set b | **fitted** |
| 4 | PackNet | Mallya & Lazebnik 2018; LIBERO | as above, `g30_cl.py --mode packnet`: per-weight task ownership on LoRA A/B, prune, post-prune fine-tune, task t served with tasks ≤ t | yes (needs the task id at test, as in LIBERO) | LIBERO Tab. 2: Long .22/.08/.25, Spatial .55/.07/.63, Object .60/.17/.60, Goal .63/.06/.75 | ~2 A40-h fit (both prune rules); read 4.4 each | W1 order a, set b | **fitted**: equal share (primary) and LIBERO's prune_perc 0.75 |
| 5 | pi_RL / RLinf PPO (Flow-Noise, Flow-SDE), pi0 + pi0.5 | arXiv 2510.25889 (pi_RL); arXiv 2510.06710 (RLinf-VLA) | github.com/RLinf/RLinf @91a302e, Apache-2.0 | pi0 / pi0.5 via openpi (PyTorch), LIBERO incl. a `libero_90` env config; full-model PPO (lr 5e-6, FSDP) | pi0: few-shot SFT 65.3/64.4/49.8/51.2 → Flow-Noise PPO 99.0/99.2/98.2/93.8; pi0.5: 84.6/95.4/84.6/43.9 → 99.6/100/99.6/94.0 (Spatial/Object/Goal/Long) | published: 8× H100 per suite, 400 epochs × 8 rollout epochs × 64 envs; GPU-hours not stated. Our estimate (UNVERIFIED) ≈ 100-300 H100-h per suite ≈ 300-900 A40-h; scaled to 20 tasks ≈ 600-1800 A40-h. Matched: 10 practice episodes per task is too few for PPO; equal A40-hours to the library's fit + practice ≈ 0.3 A40-h per task | G3/G4 subset, set b | code cloned; env install running (job 4170910); **needs a budget decision** |
| 6 | ReinFlow | arXiv 2505.22094 (NeurIPS 2025) | github.com/ReinFlow/ReinFlow @e722e15, MIT | **no LIBERO / VLA in the paper or repo**: gym, Franka Kitchen, robomimic; its README points pi0 / pi0.5 LIBERO to RLinf (the same learned-noise idea is pi_RL's Flow-Noise) | none on LIBERO | n/a | (via RLinf) | cloned; **proposal: count pi_RL Flow-Noise as the ReinFlow-family arm**, or port ReinFlow's noise net to our action expert (decision) |
| 7 | LOTUS (skill-library continual imitation) | arXiv 2311.02058 (ICRA 2024) | github.com/UT-Austin-RPL/Lotus @ce38726, **no licence file** (research use only; no redistribution) | LIBERO yes (Object, Goal, LIBERO-50); own ResNet-T BC + cVAE meta-controller from scratch, DINOv2 features; py3.9, torch 1.11+cu113 | Tab. I (FWT/NBT/AUC): Object 74/11/65, Goal 61/30/56, LIBERO-50 39/43/45; ER 56/24/49 | published: 50 demos per base task, 10 per lifelong task; compute not stated (estimate ~20-60 A40-h per stream incl. DINOv2 + clustering). Matched: 1 demo per stream task | base = pi0.5's 40 training tasks, stream = W1 order a, set b | cloned; needs LIBERO hdf5 for the 40 base tasks (LIBERO-90 hdf5 already in robot/data/libero_raw); **decision** |
| 7b | IsCiL | arXiv 2410.22658 (NeurIPS 2024) | github.com/L2dulgi/IsCiL @03082e7, no licence file | JAX DDPM policy; **no LIBERO setup shipped** (README: "evaluated on LIBERO following DMPEL's setup", no numbers, Kitchen and Meta-World only) | none | | | cloned; not runnable on LIBERO as shipped, so LOTUS is the library arm |
| 8 | Sparse Diffusion Policy (MoE experts + router) | arXiv 2407.01531 (CoRL 2024) | github.com/AnthonyHuo/SDP @9d70d48, MIT | **no LIBERO**: MimicGen, DexArt, Adroit, robomimic; diffusion policy from scratch (130-150 h per 2D multitask run on one A6000) | none on LIBERO | published-style from scratch on our stream: ~150+ A40-h; **adapted**: SDP's top-k router over our per-task LoRA experts on pi0.5's action expert ≈ 2 A40-h fit + read 4.4 | E3, set b | cloned; **decision**: faithful (port to LIBERO, from scratch) or adapted (router over our experts) |
| 9 | DemoDiffusion (one demo, no training) | arXiv 2506.20668 | github.com/demodiffusion/demodiffusion @ad9cacc, no licence; real-robot pi0-DROID deploy code only | our SDEdit implementation (`DemoDiffusionHooks`, s* = 0.2, 20 steps, as the authors' openpi fork) on pi0.5 | none on LIBERO (real Franka: pi0 13.8% → 83.8%) | 0 training; read 4.4 + acceptance 1.5 | E3, set b | the M3-era read (36 / 420) predates the own-scene demo fix (2026-10-04) and used other tasks: **not current**. Demo programs rebuilt from current demos (`--tag _lib40g30`, arm `demodiff@g30`) |
| ref | TAIL (per-task LoRA) | arXiv 2310.05905 | no code found | = our per-task LoRA family | LIBERO-10 FWT: TAIL-LoRA 0.70, FFT 0.48, ER 0.45, EWC 0.30 | | | already run (per-task LoRA ± acceptance) |

Already run, for completeness: per-task one-demonstration LoRA (± acceptance), sequential FT, ER, joint multi-task (G11, F1); task
arithmetic and LoRA averaging (G2); STRAP (G6); DSRL ×3 (G7, G7b); Li's interpolation and TLI.

## Parity smokes (stage 1)

| baseline | reference | smoke | result |
|---|---|---|---|
| all-demonstration LoRA | pi0.5 LIBERO-90 multi-task 92.7 (arXiv 2505.23705) | per-task LoRA on all own-scene demos, tasks 13, 36, 52 (zero-shot 0 of 20 each on set b), stored states 30-49, seed 13 | pending |
| 5-demonstration LoRA, full SFT | none published | the same three tasks: the pipeline must serve, with successes above zero-shot | pending |
| EWC / PackNet | LIBERO's numbers are for ResNet-T from scratch with 50 demos and FWT/NBT metrics, so no like-for-like parity exists | code checks: the fit rows record PackNet's per-task kept / free weights; EWC at e_lambda 0 reduces to G11's sequential FT | see ledger g30_cl |
| RLinf | pi0.5 few-shot SFT 84.6 → PPO 99.6 on LIBERO-Spatial | not feasible within ~8 A40-h: full-model PPO of pi0.5 needs more than one 48 GB A40 (FSDP across ≥ 4 A40). Feasible instead: evaluate the released `RLinf/RLinf-Pi05-LIBERO-SFT` and `RLinf/RLinf-Pi0-PPO-LIBERO-spatial` checkpoints on LIBERO-Spatial in our harness (~2 A40-h) | after the env install |
| LOTUS | Object FWT 74 / AUC 65 | not run (needs the base stage) | decision |

## Decisions for the user

1. **RL budget.** pi_RL at the published scale on the 20 G3/G4 tasks is about 600-1800 A40-h (our estimate; the paper gives no
   GPU-hours). It also needs multi-GPU jobs (full-model PPO does not fit one A40). Three options:
   - (a) pi_RL with LoRA on the action expert, VLM frozen. This fits one A40, but the published recipe trains the whole model.
   - (b) full-model pi_RL on 4 A40s with FSDP, a few tasks only.
   - (c) published-checkpoint evaluation only, plus (a) at matched A40-hours.
2. **ReinFlow.** Either count pi_RL's Flow-Noise as the ReinFlow arm (ReinFlow's own README points pi0 LIBERO to RLinf), or port
   ReinFlow.
3. **Library method.** LOTUS is the only one with runnable LIBERO code. It has no licence (research use) and trains its own
   ResNet-T policy from scratch: about 20-60 A40-h per setting.
4. **SDP.** Either the faithful version (from scratch, no LIBERO code) or the adapted one (a router over our per-task experts).
5. **Whole-model full SFT.** It needs 2-4 A40s with FSDP, or an 8-bit optimiser.
6. **All-demonstration SFT on all 29 tasks.** About 64 A40-h, run as 29 single-task jobs.
