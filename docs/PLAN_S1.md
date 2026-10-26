# PLAN_SOLVE S1: binder accuracy (pre-registered 2026-10-26, before B1b is trained and before any dev prediction)

From docs/PLAN_SOLVE.md section 4. The user chose our own binder (B1b) as the headline. The adapter (S0) is not built.

Ledger: m3_prereg tag `ps1`; read row `s1_bind` tag read. split_sealed.json is never opened. No demo is loaded or executed.

## 1. Design

### Data (results/bind/s1/)

**Frames** (experiments/s1_render.py, LIBERO env).
- **Training:** every l90c training cell (76 tasks) on 8 FRESH sampler states (`env.seed(9300000 + k); env.reset()`), after the
  client's 10 settle steps. That is 608 frames.
- **Dev:** the 24 dev tasks on fresh_m1 states 0-4, the states of E2 / E2b. That is 120 frames.
- **Per frame:**
  - the policy's agentview / wrist images;
  - the segmentation coverage of every body over the 16 x 16 agentview tokens;
  - every BDDL region site's projected box coverage (`*_region`, `*_side`).

**Labels** (experiments/s1_labels.py; training cells only; counterfactual within the scene; no demos). Per training frame, EVERY
entity and region with a phrase gets an example, not only the cell's own target.
- **Object role:**
  - every movable instance with its type phrase (factors.object_phrases over the training tasks) -> all instances of that type;
  - the training instructions' instance phrases ("middle black bowl", "black bowl at the front", "book on the right") -> their
    instance. These are used scene-wide for libero_90 scenes and task-only elsewhere.
- **Goal role:**
  - every region site whose phrase is attested in a training instruction (GOAL_T in the code lists the source task of each);
  - table regions relative to a body ("to the right of the plate");
  - goal-object types ("on the plate", "on the black bowl");
  - each training instruction's own goal phrase.
- Totals: 5168 examples (object 2696, goal 2472) over 75 distinct phrases.
- Two training cells are skipped: libero_10:1 and :7 ("both X and Y"). Their objects still enter through the other cells'
  type phrases only when those scenes recur.

**Dev units.**
- Per dev task: the In / On step, with its object phrase -> the named instance and its goal phrase -> the target.
- That is 24 tasks x 5 frames x 2 roles = 240 units.
- Depth = E4's pre-registered code (experiments/bind_e4.code), unchanged. :76's goal codes 0 (the disclosed E4 coding miss);
  it is reported separately.
- **Held-out combination:** (role, phrase, scene) never occurs among the training examples.
  - Depth-2 held-out units are :1 ("on top of the top drawer of the cabinet", a known body) and the 8 caddy tasks (:73, :75,
    :77-:80, :82, :83).
  - **The desk caddy is in no training scene.** Most depth-2 held-out units therefore test a NEW BODY TYPE as well as a relational
    sub-region. This is reported as a split.
  - Depth-2 seen units: :29, :32 ("in the top drawer of the cabinet" in KITCHEN_SCENE5) and :40 ("on the cabinet shelf").
  - Depth-1 held-out units: :60's "black bowl on the left" (training said "left bowl"), :38's "right moka pot", :16's goal "on
    the black bowl in the middle".

### B1b, our binder (experiments/s1_feats.py, s1_train.py)

**Features: the FROZEN policy's own.**
- Image: the agentview's 256 image tokens as pi05_l90c@5000's prefix embeds them (SigLIP + projector, 2048-d). These are the
  same features as the SigLIP anchor.
- Phrase: **the policy's own word embeddings** (PaliGemma's embed_language_tokens of the phrase's SentencePiece ids). No other text
  encoder is used.

**Architecture.** One head per role (object, goal). Each is conditioned ONLY on its own role's phrase:
- a 2-layer Transformer encoder over the phrase's word embeddings (256-d);
- image tokens -> LayerNorm, Linear 256, plus a learned 16 x 16 position;
- 2 decoder layers: image self-attention plus cross-attention to the phrase;
- one logit per patch, which is the mask over patches.

**Training.**
- Loss: BCE on each token's coverage, plus the cross-entropy of the token softmax against the target mask.
- Settings: AdamW lr 3e-4, wd 0.01, batch 64, 40 epochs, dropout 0.1, seed 0.
- **Fixed in advance:** no dev-based choice; the dev units are predicted once after the last epoch.
- Training frames with k >= 7 are a within-training check, reported only.

### Comparisons

- **SigLIP prototype anchor (D3 / D5):** the argmax token of the target type's training prototype (results/rb4/protos.npz).
  - Object role: the named type. Goal role: the target body's type; for a table region, its reference body.
  - Types without a prototype (e.g. the caddy) are reported separately.
- **B1a, off-the-shelf:** GroundingDINO-base (transformers; HF weights).
  - Query: the role phrase with its goal preposition removed. The best box becomes a token mask; its centre token is the peak.
  - **Deviations (disclosed):**
    - No depth is used, because S1 scores image-space masks.
    - RoboRefer was not tried: it is a separate 2-8B VLM with its own code, outside this phase's budget.

## 2. Metrics

- **hit (primary):** the method's peak token lies in the target's mask, AND it is covered by the target at least as much as by
  any competitor.
  - Competitors: the other regions of the same body (drawers / top side, shelf levels, caddy compartments), the other table
    regions relative to the same body, or the other same-type instances.
  - **Deviation from the brief's "target-in-mask" (disclosed):** the projected region boxes of one body overlap in the image. For
    example, a peak on the cabinet front lies in the top, middle and bottom drawer boxes at once. Without the competitor rule, a
    region hit is not a resolution. The loose hit (target-in-mask only) is reported as descriptive.
- **iou:** the predicted mask (B1b sigmoid >= 0.5; B1a box coverage >= 0.5) vs the target (coverage >= 0.5). This applies to B1b
  and B1a only; the anchor has no mask.
- **Statistics:** per depth (0 / 1 / 2) x {all, seen, held-out}, per role, and per task. Two-level task-cluster bootstrap
  (d4_read.rate).

## 3. Bar (pre-declared)

**PASS iff both hold:**
- (i) B1b's hit on depth-2 HELD-OUT units >= 0.70 (point estimate);
- (ii) at every depth (0, 1, 2), B1b's hit >= the anchor's, on the units where the anchor has a prototype (point estimates).

**Otherwise FAIL.**

**Pre-stated expectation (honest).**
- Depth-2 held-out is 8 / 9 caddy tasks: a body that no training label ever names.
- A binder trained on training cells has to transfer "caddy" and "compartment" from word embeddings alone. FAIL on (i) is likely.
- The known-body depth-2 cases (:1 held-out; :29, :32, :40 seen) are the informative split, and are reported separately.

**Power.**
- 24 tasks, 5 frames each.
- Depth 2 held-out: 9 tasks. Depth 1: 7 object + 1 goal task. The frames within a task are near-duplicates of one layout.

## 4. Cost (planned)

- Render: CPU, about 0.05 A40-h (the GPU-allocated shards).
- Features: about 0.1 A40-h. Training: about 0.2 A40-h. B1a: about 0.05 A40-h.
- The read is on CPU. Total: under 1 A40-h.
