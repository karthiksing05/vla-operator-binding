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

## 5. Result (read 2026-10-26; ledger s1_bind tag read; results/bind/s1.json)

**FAIL on both pre-declared conditions.**
- (i) B1b's depth-2 held-out hit is **0.378 [0.111, 0.667]** (45 units, 9 tasks), below 0.70.
- (ii) At depth 0, B1b is below the anchor on the units with a prototype: 0.733 [0.56, 0.88] vs 0.867 [0.76, 0.95].
  - At depth 1: 0.60 vs 0.45. At depth 2: 0.90 vs 0.00 (4 tasks).

**Hit by method** (two-level task-cluster bootstrap):

| depth / split | n (tasks) | B1b (ours) | B1a GroundingDINO | anchor (proto units) |
|---|---|---|---|---|
| 0 seen | 90 (14) | 0.967 [0.89, 1.00] | 0.722 [0.54, 0.89] | 0.911 |
| 0 held-out | 50 (9) | 0.240 [0.07, 0.42] | 0.900 [0.72, 1.00] | 0.778 (45) |
| 1 seen | 25 (5) | 0.800 [0.56, 1.00] | 0.200 [0.00, 0.60] | 0.600 |
| 1 held-out | 15 (3) | 0.267 [0.00, 0.60] | 0.000 | 0.200 |
| 2 seen | 15 (3) | 1.000 [1.00, 1.00] | 0.000 | 0.000 |
| 2 held-out | 45 (9) | 0.378 [0.11, 0.67] | 0.111 [0.00, 0.33] | 0.000 (5; the caddy has no prototype) |

**Reading.**
- **B1b resolves what the training labels taught, in the scenes they taught it, at every depth.**
  - Seen depth 2 is 1.00: "in the top drawer of the cabinet" on :29 / :32, "on the cabinet shelf" on :40. The anchor and
    GroundingDINO both score 0 there: they find the body, not the region.
  - Seen depth 1 is 0.80: "middle black bowl" 1.00, "black bowl at the front" 0.80.
- **It does not transfer out of the training scenes.**
  - The STUDY_SCENE1-3 tasks are a scene family with no training cell. Even the TYPE phrase "book" falls to 0.00-0.60 (depth 0
    held-out 0.24), where GroundingDINO gets 1.00.
  - The caddy compartments (a body never labelled) are 0 on 5 of 8 tasks and 0.8-1.0 on 3. With four compartments, this is not
    above chance.
  - New phrasings in familiar scenes partly transfer:
    - :1 "on top of the top drawer of the cabinet" 0.60;
    - :38 "right moka pot" 0.60;
    - :16's goal "on the black bowl in the middle" 0.20;
    - :60 "black bowl on the left" 0.00 (training said "left bowl"; the anchor gets 0.60).
- **The two binders fail in complementary places.**
  - GroundingDINO handles new scenes at type level (0.90 held-out depth 0) and fails every relational sub-region and every
    instance qualifier (0.00-0.20).
  - B1b handles sub-regions and qualifiers in the training scene distribution and fails new scenes.
  - The headline claim (B1b resolves held-out combinations) is not supported.
- **IoU:** B1b's sigmoid masks are under-confident (IoU 0.22 at depth 0, 0.35 at depth 2; mostly empty at 0.5). GroundingDINO's
  boxes give 0.53 at depth 0 and 0.09 at depth 2. The peak hit is the informative measure here.

**Disclosures.**
- The competitor rule replaces the brief's plain "target-in-mask" for overlapping region boxes. It was declared before the read; the
  loose hit is in results/bind/s1.json.
- RoboRefer was not run, and GroundingDINO was run without depth.
- :76's goal is coded depth 0 by E4's pre-registered code. It is held-out and missed by all three methods.
- The within-training check (frames k >= 7, loose hit) was 0.93 object / 0.98 goal. B1b fits its training distribution.
- The ledger rows carry dirty=true, because other agents' edits are in the shared tree.

## 6. S1b: the factorised binder (pre-registered 2026-10-26, before S1b is trained or any dev unit is scored)

From PLAN_SOLVE section 7: body by open-vocabulary detection x relation in the body's own frame. Reuses S1's frames, labels,
features and scoring.

### Design

**Split** (experiments/s1b_detect.split).
- The role phrase's BODY is the longest known type / fixture noun phrase in it. The lexicon is:
  - the training tasks' type phrases;
  - the attested goal templates' fixture nouns, and "caddy" from factors' fallback table.
- The RELATION is the remainder:
  - "in the top drawer of the cabinet" -> body "cabinet", relation "in the top drawer of the";
  - "middle black bowl" -> "black bowl" + "middle";
  - "black bowl on the left" -> "black bowl" + "on the left".

**Body stage** (s1b_detect.py).
- GroundingDINO-base proposes boxes for the body phrase on the policy's 224 image.
- Settings: score >= 0.05, NMS 0.5, at most 6 boxes.
- Run on all 728 S1 frames (3403 queries; every query returns a box).
- **Set before registration (disclosed):**
  - A first run at 0.15 returned no box for "cabinet" on most training frames, which dropped half the goal examples. A
    training-only diagnostic (experiments/s1b_diag.py) showed this, and the threshold was lowered.
  - Separately, a type phrase with several instances now matches the box covering any ONE instance.
  - The remaining training drops are mostly the black-bowl instances of KITCHEN_SCENE2 ("black bowl at the front" 0 / 24
    matched), where the detector's boxes do not isolate one bowl.

**Relation / instance stage** (s1b_train.py). One head per role, trained once on the TRAINING cells' S1 labels. Per candidate box
it sees only:
- the frozen policy's image tokens resampled on an 8 x 8 grid over the box enlarged 2x (the body's frame);
- the box geometry and its rank position among the body's candidates (needed to pick an instance by qualifier);
- the policy's own word embeddings of the RELATION phrase.

It outputs a softmax over the candidates (instance choice) and a mask over the crop cells. Settings are S1's: AdamW 3e-4, 40
epochs, seed 0, no dev-based choice.
- **Deviation from "only the crop" (disclosed):** the geometry / rank features are needed for instance qualifiers, as the brief
  asks ("detect all same-type instances and have the qualifier head pick among them").
- The 2x enlargement lets table regions beside a body ("to the right of the plate") fall inside the frame.

### Scoring

S1's hit (the competitor rule) and IoU, on the same 240 dev units (results/bind/s1b.json; s1b_read.py).
- **body_seen:** the target's body type (the named type; a region's body; a table region's reference body) is the body type of
  some training example's target.

### Bar (pre-declared)

**PASS iff both hold:**
- (i) S1b's hit on depth-2 HELD-OUT units with a SEEN body type >= 0.70 (point estimate);
- (ii) S1b's depth-0 hit >= GroundingDINO alone (S1's B1a) on the same units (point estimates).

The caddy (unseen body type) is reported separately.

**Power (stated honestly).** (i) has ONE task: :1, "on top of the top drawer of the cabinet", 5 units. It is a weak test; all
cells are reported with CIs.
