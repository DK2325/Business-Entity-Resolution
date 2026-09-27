# Business Entity Resolution

Matching business records across three independent, noisy sources. Source 1 is a
deduplicated reference set; for every Source 1 entity the task is to find all
Source 2 and Source 3 records describing the same real-world business — zero,
one, or many.

Scored by **macro-averaged F_0.5**: computed per Source 1 entity, then averaged.
Precision is weighted about 2x over recall, and an entity with no true matches
scores 1.0 for an empty prediction and 0.0 for any non-empty one.

## Requirements

| | |
| --- | --- |
| Python | 3.10 or newer (developed on 3.12) |
| Cores | 8 or more recommended; the scoring stage scales across processes |
| Memory | 32 GB or more recommended for the full test set |
| Disk | ~15 GB for the dataset, intermediates and outputs |

The scoring stage sizes its worker pool from the available cores and memory. On
platforms providing `fork` the per-country index is built once and shared
copy-on-write, so N workers cost roughly one index; elsewhere each worker builds
its own, and the pool is sized down accordingly. With 8 cores the full test set
scores in about 70 minutes; with fewer cores or less memory it still completes,
more slowly, and per-worker checkpoints let an interrupted run resume.

```bash
python -m pip install -r requirements.txt
```

## Layout

```
src/
  config.py       paths and seed
  data_io.py      TSV readers and submission writers
  normalize.py    name/address canonicalisation and romanisation
  keys.py         token views used as blocking keys
  blocking.py     vocabulary, IDF weighting, sparse top-k
  candidates.py   per-country index and per-record shortlists
  features.py     the 33 pair features
  inference.py    streaming scorer, one row kept per record
  decide.py       exclusivity argmax, threshold and margin
  collective.py   second-stage group and sibling features
  metrics.py      exact macro F_0.5 with the singleton rule
  splits.py       grouped holdout and leave-one-country-out
scripts/          command-line entry points
reports/          generated analysis
output/           matching_results.tsv, candidate_pairs.tsv
```

The dataset is not stored in this repository. Point the code at a local copy of
`student_resource/` with `BER_DATA_ROOT`; intermediates go to `BER_ARTIFACTS`.
Both have repo-relative defaults in `src/config.py`.

```bash
export BER_DATA_ROOT=/path/to/student_resource     # Windows: set BER_DATA_ROOT=...
export BER_ARTIFACTS=/path/to/scratch
```

## Reproducing the submission end to end

The configuration below is the one that produced the submitted
`output/matching_results.tsv` and `output/candidate_pairs.tsv`. The three
blocking parameters (`--k 5`, `--max-df-ratio 0.02`, `--score-floor 0.2469`)
must match across every stage: they define the candidate set, and the matcher's
blocking-context features are calibrated to it.

```bash
# 0. sanity: the scorer must reproduce the worked example from the problem statement
python scripts/test_metrics.py

# 1. validation splits (grouped holdout + leave-one-country-out)
python scripts/make_splits.py

# 2. labelled training pairs, sampled by Source 2/3 record so the training
#    distribution matches inference (distractors included at their true rate)
python scripts/build_training_pairs.py --country India --n-records 250000 \
    --k 5 --max-df-ratio 0.02 --chunk-rows 8000 --out "$BER_ARTIFACTS/pairs_India.parquet"
python scripts/build_training_pairs.py --country US --n-records 250000 \
    --k 5 --max-df-ratio 0.02 --chunk-rows 8000 --out "$BER_ARTIFACTS/pairs_US.parquet"

# 3. stage-1 pair classifier
python scripts/train_matcher.py \
    --pairs "$BER_ARTIFACTS/pairs_India.parquet" "$BER_ARTIFACTS/pairs_US.parquet" \
    --out "$BER_ARTIFACTS/matcher.txt"

# 4. holdout decisions, used to train the second stage and to tune the threshold
python scripts/predict_parallel.py --split train --countries India \
    --workers auto --k 5 --max-df-ratio 0.02 --score-floor 0.2469 \
    --model "$BER_ARTIFACTS/matcher.txt" --chunk-rows 8000 \
    --checkpoint-every 200000 --out-dir "$BER_ARTIFACTS/shards_train"

python scripts/eval_shards.py --country India --shards "$BER_ARTIFACTS/shards_train"

# 5. stage-2 collective model (group and sibling agreement)
python scripts/train_collective.py --country India \
    --shards "$BER_ARTIFACTS/shards_train" --floor 0.30 \
    --out "$BER_ARTIFACTS/collective.txt"

# 6. score the test set; writes one shard per worker per country
python scripts/predict_parallel.py --split test --countries France US India \
    --workers auto --k 5 --max-df-ratio 0.02 --score-floor 0.2469 \
    --model "$BER_ARTIFACTS/matcher.txt" --chunk-rows 8000 \
    --checkpoint-every 200000 --out-dir "$BER_ARTIFACTS/shards_test"

# 7. write both output files from the shards (stage-1 decision rule)
python scripts/write_submission.py --shards "$BER_ARTIFACTS/shards_test" \
    --threshold 0.70 --margin 0.0

# 8. apply the collective stage where it was validated to help. Only
#    matching_results.tsv is rewritten; candidate_pairs.tsv from step 7 still
#    describes exactly the set the model scored, so matches stay a subset of it.
#
#    The collective stage is applied to India only. It is trained on India
#    decisions, and applied to US it turned singleton entities into matched ones
#    at a 3:1 ratio against the reverse, which this metric punishes hard; the
#    leaderboard agreed (0.914 applied everywhere against 0.916 on India only).
python scripts/apply_collective.py --shards "$BER_ARTIFACTS/shards_test" \
    --model "$BER_ARTIFACTS/collective.txt" \
    --threshold 0.70 --floor 0.30 \
    --stage2-countries India --stage1-threshold 0.70

# 9. validate before submitting
python "$BER_DATA_ROOT/utils/validate_submission.py" \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir "$BER_DATA_ROOT/dataset/test"
```

### Variant: collective stage on all three countries

If a collective model is trained on **both** India and US decisions
(`scripts/train_collective_multi.py`), and per-country evaluation on both halves
of the holdout shows it beats stage 1 in **every** country, step 8 becomes:

```bash
# 5b. multi-country collective model (replaces step 5)
python scripts/train_collective_multi.py --countries India US \
    --shards "$BER_ARTIFACTS/shards_train" --floor 0.30 \
    --out "$BER_ARTIFACTS/collective_multi.txt"

# 8b. apply it everywhere (replaces step 8)
python scripts/apply_collective.py --shards "$BER_ARTIFACTS/shards_test" \
    --model "$BER_ARTIFACTS/collective_multi.txt" \
    --threshold 0.70 --floor 0.30
```

Which of the two was used for the submitted file is recorded in
`submissions_log.csv` alongside its leaderboard score.

Steps 4, 6 and 8 are the long ones. Every scoring run checkpoints each worker,
so re-running the same command after an interruption resumes from the last
checkpoint rather than starting over.

## Analysis

```bash
python scripts/run_eda.py                  # regenerates reports/eda.md
python scripts/loss_decomposition.py       # where macro-F_0.5 points are lost
python scripts/probe_candidate_budget.py   # recall against candidate-set size
python scripts/probe_structure.py          # exact-key coverage and selectivity
```

## Approach

Three measured facts shape the design; `Documentation_template.md` has the full
account.

1. **Exclusivity.** No Source 2/3 record is claimed by two Source 1 entities
   (0 of 7,638,365 training links). Matching is a constrained assignment, so
   retrieval runs from the Source 2/3 side and each record is assigned to at
   most one entity.
2. **Country agreement.** True links never cross a country boundary (0
   mismatches in 693,069 links), so country partitions the problem exactly.
3. **Address carries the signal, names carry the noise.** Matched records share
   near-identical addresses subject to reordering and abbreviation, while names
   vary by transliteration, typo, truncation, trade name and legal suffix.

The test set contains a country absent from training. Nothing branches on the
country label — it is carried as an opaque string and used only as a partition
key — and leave-one-country-out validation is the proxy used to judge whether a
change generalises.

No external data is used: no geocoding, no business registries, no web lookups,
no outside augmentation. Every signal is derived from the provided files.
