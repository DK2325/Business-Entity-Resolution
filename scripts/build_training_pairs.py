"""Generate labelled (record, candidate) pairs for training the matcher.

The unit sampled is the **Source 2/3 record**, not the Source 1 entity, because
that is the unit the model faces at inference: one record, its shortlist of
candidate entities, at most one of them correct. Sampling records rather than
entities reproduces the inference distribution exactly, including two things
entity-side sampling would distort:

* **Distractors.** About 26% of Source 2/3 records match nothing at all. Sampled
  by record they appear at their true rate and contribute all-negative
  shortlists, which is what teaches the model to withhold a match. Sampled by
  entity they would never appear.
* **Shortlist competition.** A record's candidates compete with each other, and
  the margin between them is a feature. That structure only exists per record.

The Source 1 index covers every entity in the country, so candidates face the
full field they will face at test time. Train/validation membership follows the
owning entity's side of the grouped holdout; no feature references an entity
identity, so an entity appearing as a competitor on the other side leaks nothing.

Usage::

    python scripts/build_training_pairs.py --country India --n-records 150000 --k 5
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.candidates import build_country_index, iter_country_rows, retrieve, shortlist_context
from src.features import FEATURE_NAMES, PairFeaturizer
from src.splits import load_splits


def load_owner_map(ground_truth_path: Path) -> dict[str, str]:
    """Map every matched Source 2/3 record id to its owning Source 1 entity."""
    owner: dict[str, str] = {}
    with open(ground_truth_path, encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or not parts[1].strip():
                continue
            entity_id = parts[0]
            for record_id in parts[1].split(","):
                if record_id:
                    owner[record_id] = entity_id
    return owner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", default="India")
    parser.add_argument("--n-records", type=int, default=150_000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--chunk-rows", type=int, default=20000)
    parser.add_argument("--max-df-ratio", type=float, default=0.002)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config.ensure_dirs()
    started = time.time()

    splits = load_splits(config.ARTIFACTS / "splits.json")
    valid_entities = set(splits["holdout"]["valid"])
    print(f"holdout valid entities: {len(valid_entities):,}")

    owner = load_owner_map(config.TRAIN_FILES["ground_truth"])
    print(f"owned Source 2/3 records: {len(owner):,}")

    index = build_country_index(
        config.TRAIN_FILES["source1"],
        args.country,
        max_df_ratio=args.max_df_ratio,
    )
    print(f"index: {index.size:,} Source 1 entities  [{time.time() - started:.0f}s]")
    row_of_entity = {entity_id: row for row, entity_id in enumerate(index.entity_ids)}
    idf_map = index.token_idf()
    print(f"idf map: {len(idf_map):,} tokens  [{time.time() - started:.0f}s]")

    # Reservoir-free sampling: decide inclusion per record with a fixed rate, so
    # the pass is single-shot and memory stays flat.
    total_records = 0
    for source in ("source2", "source3"):
        total_records += sum(1 for _ in iter_country_rows(config.TRAIN_FILES[source], args.country))
    rate = min(1.0, args.n_records / max(total_records, 1))
    print(f"sampling {rate:.4f} of {total_records:,} Source 2/3 records")

    rng = random.Random(config.SEED)
    featurizer = PairFeaturizer(idf_map)

    out_path = Path(args.out) if args.out else config.ARTIFACTS / f"pairs_{args.country}.parquet"
    schema = pa.schema(
        [pa.field(name, pa.float32()) for name in FEATURE_NAMES]
        + [
            pa.field("label", pa.int8()),
            pa.field("is_valid", pa.int8()),
            pa.field("record_id", pa.string()),
            pa.field("entity_id", pa.string()),
        ]
    )
    writer = pq.ParquetWriter(out_path, schema, compression="zstd")

    buffer: list[tuple[str, str, str, bool]] = []
    n_pairs = 0
    n_positive = 0
    n_records_used = 0
    n_owner_retrieved = 0
    n_owner_in_country = 0

    def flush() -> None:
        nonlocal n_pairs, n_positive, n_owner_retrieved, n_owner_in_country
        if not buffer:
            return
        shortlists = retrieve(index, buffer, k=args.k, chunk_rows=args.chunk_rows)
        rows: list[list[float]] = []
        labels: list[int] = []
        is_valid: list[int] = []
        record_ids: list[str] = []
        entity_ids: list[str] = []

        for shortlist in shortlists:
            true_owner = owner.get(shortlist.record_id)
            owner_row = row_of_entity.get(true_owner) if true_owner else None
            if owner_row is not None:
                n_owner_in_country += 1
                if owner_row in shortlist.ranks:
                    n_owner_retrieved += 1

            # Validation membership follows the owner; distractors are split at
            # the same rate so both sides see them.
            if true_owner is not None:
                valid_flag = 1 if true_owner in valid_entities else 0
            else:
                valid_flag = 1 if rng.random() < 0.18 else 0

            right = featurizer.parse(shortlist.record_id, shortlist.name, shortlist.address)
            for index_row in shortlist.rows():
                left = featurizer.parse(
                    index.entity_ids[index_row],
                    index.names[index_row],
                    index.addresses[index_row],
                )
                context = shortlist_context(shortlist, index_row)
                rows.append(featurizer.features(left, right, context))
                label = 1 if index_row == owner_row else 0
                labels.append(label)
                n_positive += label
                is_valid.append(valid_flag)
                record_ids.append(shortlist.record_id)
                entity_ids.append(index.entity_ids[index_row])

        if rows:
            matrix = np.asarray(rows, dtype=np.float32)
            arrays = [pa.array(matrix[:, i]) for i in range(matrix.shape[1])]
            arrays += [
                pa.array(labels, type=pa.int8()),
                pa.array(is_valid, type=pa.int8()),
                pa.array(record_ids, type=pa.string()),
                pa.array(entity_ids, type=pa.string()),
            ]
            writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
            n_pairs += len(rows)
        buffer.clear()
        # The per-record cache only helps within a chunk; clearing it bounds
        # memory across the whole pass.
        if len(featurizer._cache) > 400_000:
            featurizer.clear()

    for source in ("source2", "source3"):
        is_s3 = source == "source3"
        for record_id, name, address in iter_country_rows(config.TRAIN_FILES[source], args.country):
            if rng.random() >= rate:
                continue
            buffer.append((record_id, name, address, is_s3))
            n_records_used += 1
            if len(buffer) >= args.chunk_rows:
                flush()
                print(
                    f"    {n_records_used:,} records -> {n_pairs:,} pairs "
                    f"({n_positive:,} positive)  [{time.time() - started:.0f}s]",
                    flush=True,
                )
        flush()

    writer.close()
    blocking_recall = n_owner_retrieved / max(n_owner_in_country, 1)
    print(
        f"\nwrote {out_path}\n"
        f"  records sampled : {n_records_used:,}\n"
        f"  pairs           : {n_pairs:,} ({n_pairs / max(n_records_used, 1):.2f} per record)\n"
        f"  positives       : {n_positive:,} ({n_positive / max(n_pairs, 1):.2%})\n"
        f"  owner retrieved : {blocking_recall:.4f} (blocking recall on sampled records)\n"
        f"  seconds         : {time.time() - started:.0f}"
    )


if __name__ == "__main__":
    main()
