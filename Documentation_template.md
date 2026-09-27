# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** CodeOps
**Team Members:** Dushyant Kumar (Team Leader), Kawaljeet Singh, Mayank, Harsh Pachauri
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

Candidate generation runs from the **Source 2/3 side**: each Source 2/3 record retrieves its
best Source 1 entities, rather than each Source 1 entity searching the 10M-record side. This
follows from a property we verified across all 7,638,365 training links — **no Source 2/3
record is ever claimed by two Source 1 entities** — which makes matching a constrained
assignment and bounds the candidate set at `(|S2| + |S3|) x k` however many matches an
individual entity attracts. A LightGBM pair classifier scores the candidates; a second
"collective" model re-scores each assignment against the other records assigned to the same
entity, which is what distinguishes a genuine match from a plausible-looking distractor. The
collective stage is applied only where it was measured to help.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from EDA (`reports/eda.md`, reproducible with `scripts/run_eda.py`) that shaped every
later decision:

| Finding | Measurement | Consequence |
| --- | --- | --- |
| **Exclusivity** | 0 of 7,638,365 links claimed by two entities | Matching is an assignment; retrieve from the S2/S3 side |
| **Country never crosses** | 0 mismatches in 693,069 links | Country is a safe hard partition |
| Singletons | 123,247 of 2,206,821 (5.58%) | Each correct empty prediction is worth a full 1.0 |
| Distractors | 26.6% of S2, 25.4% of S3 match nothing | The model must learn to withhold |
| Empty addresses | 3.3% of S2/S3 | Needs a name-only retrieval path |
| Matches per entity | mean 3.46, max 11 | Groups are large enough for collective reasoning |

Noise is heavy and structured: cross-script transliteration (Devanagari, Bengali, Gujarati,
Kannada, Tamil), typos, DBA/shell-company prefixes, bare-domain names, word transposition,
address component reordering and dropping.

**Two hypotheses we tested and rejected**, because they shaped what we did not build:

* *Exact normalised keys.* Full normalised addresses are exactly equal on only **10.79%** of
  true links; no deterministic key exceeded 61% coverage, and the high-coverage ones had poor
  selectivity (digit tokens alone: 829 entities per lookup). There is no deterministic shortcut.
* *Plain character TF-IDF is sufficient.* On an identical index, plain char 3-5 gram TF-IDF
  reached R@5 = 0.9473 against 0.9654 for the view-based retrieval below. Simpler was not better.

### 2.2 Solution Strategy

**Approach Type:** Blocking + pairwise classifier + collective second stage
**Core Innovation:** Exploiting match exclusivity end to end — the retrieval direction, the
reverse-best-match features, the argmax decision rule and the sibling-agreement second stage
all derive from it.

---

## 3. Candidate Generation (Blocking)

Two complementary **views** of each record, indexed separately, results unioned:

* `composite` — romanised address and name tokens plus digit-anchored composite keys pairing
  each house/plot number with each locality token. Numbers survive reordering, abbreviation and
  transliteration better than words: on true pairs the numeric-token Jaccard has median 1.0
  against 0.0 for hard negatives.
* `ngram` — character 3/4-grams of the romanised name. This is the only thing that bridges
  cross-script names, where transliteration yields `tteknolojii` for `technology` — no shared
  token, but most trigrams in common.

Both sides are romanised with `unidecode` before tokenising, turning native-script names into
approximate Latin that fuzzy and n-gram comparison can reach.

- **Blocking keys used:** IDF-weighted tokens of the two views above, scored as cosine over
  L2-normalised sparse rows, with country as a hard partition (never a feature).

- **Parameters:** `k = 5` per view, `max_df_ratio = 0.02`, blocking-score floor `0.2469`.

The document-frequency cap was the highest-leverage parameter. We tuned it as a controlled
experiment on a matched 200,000-entity harness, holding everything else fixed:

| `max_df_ratio` | longest postings list | R@5 (harness) |
| ---: | ---: | ---: |
| 0.002 | 400 | 0.9466 |
| 0.01 | 1,997 | 0.9577 |
| **0.02** | **3,998** | **0.9627** |
| 0.05 | 9,911 | 0.9639 |
| uncapped | 82,667 | 0.9649 |

The cap bounds the sparse matrix product: the longest postings list grows with it, and uncapped
it reaches the size of the index, making the intermediate product infeasible at full scale.
**0.02 captures 97% of the available recall while bounding the longest postings list roughly
20x lower.** Measured at **full scale** on sampled records, moving from 0.002 to 0.02 raised
blocking recall from 0.9108 to **0.9334** (India) and 0.9556 to **0.9688** (US).

- **Candidate pairs generated:** at `k = 5` with `max_df 0.002` and no floor, the test set
  produced **93,466,649** candidate pairs across 1,732,544 Source 1 entities — about **54 per
  entity**. Because retrieval runs from the Source 2/3 side, nominations accumulate unevenly on
  Source 1 entities, so candidates per entity is considerably larger than `k` and has to be
  managed explicitly. The score floor is the control. Measured on the 200k-entity tuning
  **harness** (figures below are harness measurements; the per-entity counts are scaled to
  production by the ratio of queries to entities):

| Configuration | Link recall (harness) | Candidates per entity (scaled) |
| --- | ---: | ---: |
| k=5, cap 0.002, no floor | 0.9474 | ~54 |
| k=5, cap 0.02, no floor | 0.9650 | ~47 |
| **k=5, cap 0.02, floor 0.2469 (selected)** | **0.9620** | **~35** |
| k=3, cap 0.02, floor 0.2469 | 0.9522 | ~21 |

  The selected configuration improves on both axes relative to the earlier one: higher recall
  with roughly **35% fewer candidates per Source 1 entity**. A per-entity cap was evaluated and
  rejected: capping at 20 saved a further 11% of candidates for 0.3pt of recall, and it cannot
  be applied consistently in a sharded run — a global per-entity cap requires knowledge no
  single worker holds, so it would have to be applied after scoring, leaving
  `candidate_pairs.tsv` inconsistent with the set the model actually scored.

- **How we ensured true matches were not lost:** a union of two independent views (they overlap
  only ~4%, so each recovers links the other buries); a name n-gram view covering the 3.3% of
  records with no address at all; and blocking recall measured directly against ground truth at
  every configuration rather than assumed.

---

## 4. Matching Model

**Features used** (33, `src/features.py`):

- **Name features:** token Jaccard, token-set / token-sort / partial ratio, Jaro-Winkler,
  character-trigram Jaccard, IDF-weighted overlap, rarest shared token weight, acronym match,
  length ratio, empty-name flag.
- **Address features:** token Jaccard, token-set ratio, trigram Jaccard, IDF-weighted overlap,
  length ratio, either/both-empty flags, and the numeric family: numeric-token Jaccard, shared
  numeric count, an explicit **conflict** flag (both sides carry house numbers and they
  disagree), and a numbers-missing flag. Under a precision-heavy metric an explicit
  disagreement signal is worth more than another shade of similarity.
- **Other:** blocking context — per-view rank and score, best rank and score, margin to the
  record's next-best entity, shortlist size, and a source flag. `score_margin_to_next` is
  consistently the highest-gain feature; this is exclusivity entering the model.

**Model type:** LightGBM binary classifier (500 trees, 127 leaves, learning rate 0.06), trained
on **4,711,239** candidate pairs sampled from India and US training records, holdout average
precision **0.9914**. Sampling is by **Source 2/3 record**, not by entity, so the training
distribution matches inference exactly and distractors appear at their true ~26% rate.

**Threshold selection method:** exact macro F_0.5 on a grouped holdout (18% of Source 1
entities, stratified by country, split by entity so no match group straddles the boundary),
sweeping threshold and margin. The optimum was flat between 0.60 and 0.70; **0.70** was chosen
because at a flat optimum the precision-leaning end is safer — it gave 16% fewer false merges
on singletons and 16% fewer distractors assigned for 0.0005 of macro F_0.5.

### 4.1 Second stage: collective scoring

Stage one scores each pair in isolation, which is where most of the remaining loss sits. A
Source 1 entity has on average 3.46 true records and those records resemble *each other*; a
distractor — **90% of our wrong accepts (10,487 of 11,675)** — has no siblings. It may look
plausible beside the Source 1 record but looks like nothing beside the records assigned
alongside it.

**Model type:** LightGBM binary classifier over **24 features** (`src/collective.py`) in three
groups:

- **Pair (5):** stage-1 probability, margin, rank within the entity's group, difference from and
  ratio to the group's best probability.
- **Group shape (10):** group size, counts above probability 0.5 and 0.7, sum/max/mean
  probability, whether the group contains Source 2 and Source 3 records, whether this record is
  the only one from its source, and the record's own source.
- **Sibling agreement (9):** character n-gram and fuzzy similarity of name and address against
  the other records assigned to the same entity (max and mean), the combined maximum, the
  sibling count, and the same similarities against the Source 1 record itself.

**Training and evaluation:** trained on decisions whose Source 1 entity lies on the **train**
side of the grouped holdout; evaluated only on **holdout** entities, which were further split
50/50 into halves A and B by a hash of the entity id. A change was accepted only if it improved
**both** halves.

### 4.2 Final decision rule

Exclusivity is enforced rather than hoped for: **each Source 2/3 record is assigned to at most
one Source 1 entity** — its argmax — and only if that probability clears the threshold.
Thresholding pairs independently would let one record be handed to several entities, and every
surplus copy is a false merge charged against a different entity's precision.

The collective stage is then applied **to India only**, at threshold 0.60; **US and France keep
their stage-1 assignment** at threshold 0.70.

This asymmetry is an empirical result, not a design preference. The collective model was
trained on India decisions. Applied to all three countries it produced, on US, **4,324
singleton entities turned into matched entities against only 1,435 in the reverse direction —
a 3:1 skew**, where a singleton wrongly given a match scores a hard 0. The leaderboard
confirmed the diagnosis: stage-2 everywhere scored **0.914**, identical to stage-1 alone, while
stage-2 on India only scored **0.916**. The likely mechanisms are that US blocking recall is
higher (0.9688 against 0.9334), so US groups are fuller and more confident than the model
expects, and that Indian sibling similarity is legitimately lower because of cross-script
transliteration, so a US distractor scoring moderately looks acceptable by Indian standards.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** on the India holdout, stage 1 alone scores **0.9035**. Adding the
  collective stage raises it to **0.9134** on half A and **0.9137** on half B (+0.0100 /
  +0.0102), improving both halves of the split.

**Leaderboard history:**

| # | Configuration | Public LB |
| ---: | --- | ---: |
| 1 | stage-1 only, threshold 0.70, `max_df 0.002`, no floor | 0.914 |
| 2 | stage-2 applied to all countries | 0.914 |
| 3 | **stage-2 India only, stage-1 elsewhere** | **0.916** |
| 4 | stage-1, `max_df 0.02` + floor 0.2469, retrained matcher | TBD |
| 5 | as #4 plus stage-2 India only, re-tuned threshold | TBD |

Note that the holdout predicted +0.010 from the collective stage while the leaderboard moved
+0.002 — the offline estimate overstated the gain by roughly 5x on the full test set, and by
about 2x once restricted to India. Later changes were therefore accepted only with a clear
margin on both holdout halves **and** checked per country rather than in aggregate.

**Final configuration:** `k = 5` per view, `max_df_ratio = 0.02`, blocking-score floor
`0.2469`, stage-1 threshold **0.70**, stage-2 threshold **0.60** applied to India only.

**Loss decomposition** on the India holdout (549,932 true links), each bucket priced by an
oracle that fixes only that failure mode:

| Bucket | Links | Share | F_0.5 if fixed |
| --- | ---: | ---: | ---: |
| correctly assigned | 455,100 | 82.76% | — |
| (a) never retrieved | 37,307 | 6.78% | +0.0380 |
| (c) correct top-1, below threshold | 39,914 | 7.26% | +0.0334 |
| (d) wrong entity / distractor accepted | 11,675 | — | +0.0190 |
| (b) wrong top-1 chosen | 17,611 | 3.20% | +0.0179 |
| (e) singleton false merges | — | — | +0.0054 |
| all combined | | | **1.0000** |

The decomposition is complete — the combined oracle reaches exactly 1.0 — and it shows the loss
is spread rather than concentrated. Notably, **perfect blocking with this matcher would reach
only 0.9415**, so retrieval alone was never the whole answer. Accepting every top-1 regardless
of confidence scores **0.7671**, far below the thresholded 0.9035, which is the clearest
demonstration of how heavily this metric punishes false merges.

- **Common false positives (wrong merges):** unowned distractor records that superficially
  resemble the Source 1 record. They dominate the wrong-accept bucket (10,487 of 11,675), and
  the collective stage targets them directly.
- **Common false negatives (missed matches):** cross-script names (57% of India's blocking
  misses against 3% of the US's), records with an empty address, severe typos
  (`Hopitaihfty` for `Hospitality`), and bare-domain name forms (`realtrading.com`).

---

## 6. Conclusion

Most of the score came from reading the data rather than from model capacity: exclusivity set
the architecture, the numeric-conflict signal set the precision, and a document-frequency cap
originally chosen to bound memory turned out to be the largest single drag on recall — worth
+2.3 points on India once measured and corrected. The most useful discipline was distrusting
our own offline numbers: the holdout overstated leaderboard gains several-fold, and the one
change that looked best offline (collective scoring everywhere) was neutral on the leaderboard
until we restricted it to the country it was trained on.

---

## Appendix

### A. Code Artefacts

```
code/business_entity_resolution/
  src/
    config.py       paths and seed (BER_DATA_ROOT / BER_ARTIFACTS override everything)
    data_io.py      TSV readers and submission writers
    normalize.py    name/address canonicalisation and romanisation
    keys.py         token views used as blocking keys
    blocking.py     vocabulary, IDF weighting, sparse top-k
    candidates.py   per-country index and per-record shortlists
    features.py     the 33 stage-1 pair features
    inference.py    streaming scorer, one row kept per record
    decide.py       exclusivity argmax, threshold and margin
    collective.py   the 24 stage-2 group and sibling features
    metrics.py      exact macro F_0.5 with the singleton rule
    splits.py       grouped holdout and leave-one-country-out splits
  scripts/          command-line entry points
  README.md         exact reproduction commands
  requirements.txt  pinned dependencies
```

`README.md` carries the exact command sequence that regenerates
`output/matching_results.tsv` and `output/candidate_pairs.tsv` for the submitted configuration.
The three blocking parameters must match across every stage, since the matcher's
blocking-context features are calibrated to the candidate set they describe.

### B. Additional Results

**Hardware and runtime.** Developed and run on a machine with **8 cores and 64 GB RAM**. Full
test-set scoring (9,969,589 Source 2/3 records against 1,732,544 Source 1 entities) takes
roughly **1.5-2 hours** there. Throughput depends strongly on index size, because the
document-frequency cap scales with it: about 520 records/second per worker for the smallest
country against about 135 for the largest. A 16 GB laptop completes the same pipeline but
considerably more slowly, and the worker pool sizes itself down to fit; per-worker checkpoints
let an interrupted run resume rather than restart.

**Handling the country absent from training.** The test set contains France, which appears
nowhere in the training data. Our approach was to make the pipeline country-agnostic by
construction rather than to validate against a proxy: the country label is carried as an opaque
string and used only as a partition key — never one-hot encoded, never enumerated, never a
model feature — so an unseen label flows through the same code path as a familiar one.

We did **not** run leave-one-country-out evaluation; the splits are implemented in
`src/splits.py` but time was spent on the loss decomposition and blocking work instead. The
evidence we do have is twofold. First, prediction-rate sanity checks: France's predicted
singleton rate (5.40%) is the closest of the three countries to the training rate (5.58%), and
its mean matches per matched entity (3.56) is nearest training's 3.46 — the unseen country
behaves like the seen ones. Second, per-country leaderboard evidence: comparing submissions 2
and 3 isolated the effect of the collective stage on US and France and showed it was harmful
there, which is why the final rule applies it to India only.

**Fair play.** No external data was used at any point: no geocoding, no business registries, no
web lookups, no pretrained embeddings, no outside augmentation. Every signal is derived from the
provided files. The only third-party resources are open-source Python libraries listed in
`requirements.txt`, used as general-purpose string-processing and modelling tools.

---

**Note:** Section numbering and headings follow the provided template; subsections 4.1, 4.2 and
the appendix items were added to document the second stage, the final decision rule and the
runtime environment.
