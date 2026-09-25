# Business Entity Resolution

Matching business records across three independent, noisy sources. Source 1 is a
deduplicated reference set; for every Source 1 entity the task is to find all
Source 2 and Source 3 records describing the same real-world business — zero,
one, or many.

Scored by **macro-averaged F_0.5**: computed per Source 1 entity, then averaged.
Precision is weighted about 2x over recall, and an entity with no true matches
scores 1.0 for an empty prediction and 0.0 for any non-empty one.

## Layout

```
src/            importable modules
  config.py     paths, seed
  data_io.py    TSV readers/writers (tab-separated, string-typed)
  normalize.py  name and address canonicalisation, transliteration
  metrics.py    macro F_0.5 with the singleton rule
  splits.py     grouped holdout and leave-one-country-out
scripts/        command-line entry points
reports/        generated analysis
output/         matching_results.tsv, candidate_pairs.tsv
artifacts/      caches and intermediates (not tracked)
```

The dataset is **not** stored in this repository. Point the code at a local copy
of `student_resource/` with the `BER_DATA_ROOT` environment variable; caches go
to `BER_ARTIFACTS`. Both have defaults in `src/config.py`.

## Setup

```bash
python -m pip install -r requirements.txt
export BER_DATA_ROOT=/path/to/student_resource      # Windows: set BER_DATA_ROOT=...
```

## Running

```bash
python scripts/test_metrics.py     # verify the scorer against the worked example
python scripts/make_splits.py      # write artifacts/splits.json
python scripts/run_eda.py          # regenerate reports/eda.md
```

`run_eda.py` caches each section under `artifacts/`; pass `--force` to recompute,
`--section <name>` to run one, `--render-only` to rebuild the report from caches.

## Approach

Three facts from the data shape the design:

1. **Exclusivity.** No Source 2/3 record is ever claimed by two Source 1
   entities. Matching is therefore a constrained assignment, and the decision
   rule makes candidates compete rather than thresholding them independently.
2. **Country agreement.** True links never cross a country boundary, so country
   is a safe hard blocking key.
3. **Address carries the signal, names carry the noise.** Matched records share
   near-identical addresses subject to reordering and abbreviation, while names
   vary by transliteration, typo, truncation, trade name and legal suffix.

The test set contains a country absent from training. Nothing in the pipeline
branches on the country label — it is carried as an opaque string and used only
as a partition key — and `leave-one-country-out` validation is the proxy used to
judge whether a change generalises.

No external data is used: no geocoding, no business registries, no web lookups,
no outside augmentation. Every signal is derived from the provided files.
