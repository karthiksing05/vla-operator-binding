# PLAN_SOLVE S2 smoke: adapter + a DEPLOYABLE binder (pre-registered 2026-10-26, before any S2 rollout)

EXPLORATORY. There is no confirmatory 15-task read and no sealed set; both need the user. Ledger: m3_prereg tag `ps2`; read row
`s2_smoke` tag read. split_sealed.json is never opened. No demo is executed.

## 1. Arms (the same 24 dev tasks x fresh_m1 states 0-4 = 120 units; seeds 0-3)

| arm | binding masks | source |
|---|---|---|
| zs | none (frozen pi05_l90c@5000) | S0's dev_zs runs, first 4 of 8 seeds |
| oracle | adapter + live simulator masks | S0's dev_oracle runs, first 4 of 8 seeds |
| **deploy** | adapter + **S1b** binder masks | new: results/bind/s2/roll/dev_binder |

- **Binder = S1b**, because S1c FAILED its pre-registered bars (PLAN_S1 section 9).
- **S1b masks, computed per replan on the current agentview frame** (experiments/s2_serve.py):
  1. GroundingDINO finds boxes for the role's body phrase (s1b_detect's settings).
  2. The S1b head picks one box from the policy's own SigLIP tokens and the relation phrase.
  3. Mask: the box's tokens if the relation is empty or "on the" / "in the"; otherwise the head's crop cells (sigmoid >= 0.5)
     plus the peak token. With no box, the mask is zero, so the adapter's edit is exactly zero.
- **Role phrases**: experiments/s2_plan.py splits each prompt into its object phrase and goal phrase, the same split S1's labels
  use (results/bind/s2/plan.json).
- **Pairing**: seeds 0-3 reuse S0's SEED_BASE, so each deploy run shares its state, seed and noise with the zs and oracle
  comparators. Reusing S0's runs, which have already been read, is disclosed. Only the deploy arm is new.

## 2. Damage

On S0's 50 training-cell units, deploy is compared with S0's train_zs runs (first 4 seeds).

## 3. Metrics (experiments/s2_read.py)

- Outcome classes follow S0's classify: success / binding error / motion failure.
- Rates use d4_read.rate, a two-level task-cluster bootstrap.
- Paired differences use a task bootstrap (B 2000).
- **Closure** c = (deploy - zs) / (oracle - zs), for binding error and for success. Its CI is a task bootstrap of the ratio of
  task-pooled means.
- **Binder diagnostic**: on the first call of each unit, the binder's masks are compared with the live oracle masks
  (fraction of the binder mask inside the oracle mask, IoU, and empty-mask rate).

## 4. Direction (pre-declared; no pass bar, exploratory)

- **D1**: the deploy binding error is below zs (paired difference < 0).
- **D2**: the binding closure lies in [0, 1]. We expect a partial closure, because S1b is accurate at depth 0 (0.82) and weak at
  depths 1-2 (held-out depth-2 0.60 on one task). Dev units mix the depths.
- **D3**: damage, success(deploy) - success(zs) on training cells, is >= -0.05 (S0's damage was -0.03 with oracle masks).

## 5. Cost (planned)

- Deploy dev: 120 units x 4 seeds; deploy train: 50 units x 4 seeds.
- Running GroundingDINO per replan adds server time. A 1-unit smoke (60 steps) measures it first.
- Budget: at most about 8 A40-h of the remaining ~10.9.
- If the smoke projects more than that, the train arm drops to states 0-1 before launch (disclosed).
