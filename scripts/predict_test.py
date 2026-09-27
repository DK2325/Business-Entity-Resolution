"""Score the test set and write both submission files.

Countries are discovered by reading the test data, never enumerated in code. The
test set contains France, which appears nowhere in training; it flows through
exactly the same path as US and India because nothing here branches on the label.

Memory stays flat by processing one country at a time and, within a country,
reducing each chunk to one row per Source 2/3 record before the next chunk is
read. Candidate pairs are held as int32 index pairs rather than strings -- at
~93M pairs the string form would cost several GB, the packed form about 0.7.

Usage::

    python scripts/predict_test.py --k 5 --threshold 0.4 --margin 0.0
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb

from src import config
from src.candidates import build_country_index
from src.features import PairFeaturizer
from src.candidates import retrieve, shortlist_context
from src.inference import iter_chunks


def discover_countries(path: Path) -> list[str]:
    """Distinct country labels in a source file, most frequent first."""
    counts: Counter = Counter()
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                counts[parts[3]] += 1
    return [country for country, _n in counts.most_common()]


def iter_country_records(path: Path, country: str, is_s3: bool):
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4 and parts[3] == country:
                yield parts[0], parts[1], parts[2], is_s3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--chunk-rows", type=int, default=20000)
    parser.add_argument("--max-df-ratio", type=float, default=0.002)
    parser.add_argument("--keep-floor", type=float, default=0.02,
                        help="drop records whose best probability is below this before saving")
    parser.add_argument("--skip-candidates", action="store_true",
                        help="write matching_results only (candidate_pairs is ~1.2 GB)")
    args = parser.parse_args()

    config.ensure_dirs()
    started = time.time()

    model_path = Path(args.model) if args.model else config.ARTIFACTS / "matcher.txt"
    booster = lgb.Booster(model_file=str(model_path))
    print(f"model: {model_path}")

    countries = discover_countries(config.TEST_FILES["source1"])
    print(f"countries found in test Source 1: {countries}")

    matching_path = config.OUTPUT_DIR / "matching_results.tsv"
    candidate_path = config.OUTPUT_DIR / "candidate_pairs.tsv"
    matching_out = open(matching_path, "w", encoding="utf-8", newline="")
    matching_out.write("source1_entity_id\tmatched_entity_ids\n")
    candidate_out = None
    if not args.skip_candidates:
        candidate_out = open(candidate_path, "w", encoding="utf-8", newline="")
        candidate_out.write("source1_entity_id\tcandidate_entity_ids\n")

    grand_entities = grand_assigned = grand_pairs = 0

    for country in countries:
        country_started = time.time()
        print(f"\n=== {country} ===", flush=True)
        index = build_country_index(
            config.TEST_FILES["source1"], country, max_df_ratio=args.max_df_ratio
        )
        print(f"  index: {index.size:,} Source 1 entities", flush=True)

        featurizer = PairFeaturizer(index.token_idf())

        # Packed candidate store: parallel int32 arrays into the entity index and
        # into the record-id list built as we stream.
        record_ids: list[str] = []
        pair_entity: list[np.ndarray] = []
        pair_record: list[np.ndarray] = []
        # One row per record: its winning entity, probability and margin. This
        # is what survives each chunk, and what is saved so the decision rule can
        # be re-tuned without re-running inference.
        decision_record: list[int] = []
        decision_entity: list[int] = []
        decision_prob: list[float] = []
        decision_margin: list[float] = []

        n_records = n_pairs = 0

        def records():
            yield from iter_country_records(config.TEST_FILES["source2"], country, False)
            yield from iter_country_records(config.TEST_FILES["source3"], country, True)

        for batch in iter_chunks(records(), args.chunk_rows):
            base = len(record_ids)
            shortlists = retrieve(index, batch, k=args.k, chunk_rows=args.chunk_rows)

            rows: list[list[float]] = []
            owner_entity: list[int] = []
            owner_record: list[int] = []

            for offset, shortlist in enumerate(shortlists):
                record_index = base + offset
                record_ids.append(shortlist.record_id)
                right = featurizer.parse(
                    shortlist.record_id, shortlist.name, shortlist.address
                )
                for index_row in shortlist.rows():
                    left = featurizer.parse(
                        index.entity_ids[index_row],
                        index.names[index_row],
                        index.addresses[index_row],
                    )
                    rows.append(
                        featurizer.features(
                            left, right, shortlist_context(shortlist, index_row)
                        )
                    )
                    owner_entity.append(index_row)
                    owner_record.append(record_index)

            if rows:
                matrix = np.asarray(rows, dtype=np.float32)
                probabilities = booster.predict(
                    matrix, num_iteration=booster.best_iteration or None
                )
                del matrix, rows

                entity_arr = np.asarray(owner_entity, dtype=np.int32)
                record_arr = np.asarray(owner_record, dtype=np.int32)
                if candidate_out is not None:
                    pair_entity.append(entity_arr)
                    pair_record.append(record_arr)
                n_pairs += entity_arr.size

                # Best and second-best per record, then assign.
                best_prob: dict[int, float] = {}
                best_entity: dict[int, int] = {}
                second_prob: dict[int, float] = {}
                for entity_row, record_index, probability in zip(
                    owner_entity, owner_record, probabilities
                ):
                    probability = float(probability)
                    current = best_prob.get(record_index)
                    if current is None or probability > current:
                        if current is not None and current > second_prob.get(record_index, 0.0):
                            second_prob[record_index] = current
                        best_prob[record_index] = probability
                        best_entity[record_index] = entity_row
                    elif probability > second_prob.get(record_index, 0.0):
                        second_prob[record_index] = probability

                # Keep every record's winner, not just those over the current
                # threshold. Inference costs hours; the threshold is one number.
                # Saving the decisions lets it be re-tuned in seconds against a
                # holdout result, instead of re-running the whole pass.
                for record_index, probability in best_prob.items():
                    if probability < args.keep_floor:
                        continue
                    decision_record.append(record_index)
                    decision_entity.append(best_entity[record_index])
                    decision_prob.append(probability)
                    decision_margin.append(
                        probability - second_prob.get(record_index, 0.0)
                    )

            n_records += len(batch)
            featurizer.clear()
            if n_records % (args.chunk_rows * 10) == 0:
                elapsed = time.time() - country_started
                print(
                    f"    {n_records:,} records, {n_pairs:,} pairs, "
                    f"{n_records / max(elapsed, 1e-9):,.0f} rec/s  [{elapsed:.0f}s]",
                    flush=True,
                )

        # ---- persist the decisions, then apply the rule ---------------------
        decision_path = config.ARTIFACTS / f"decisions_test_{country}.npz"
        np.savez_compressed(
            decision_path,
            record_index=np.asarray(decision_record, dtype=np.int32),
            entity_row=np.asarray(decision_entity, dtype=np.int32),
            probability=np.asarray(decision_prob, dtype=np.float32),
            margin=np.asarray(decision_margin, dtype=np.float32),
            record_ids=np.asarray(record_ids),
            entity_ids=np.asarray(index.entity_ids),
        )
        print(f"  saved decisions to {decision_path.name}", flush=True)

        assigned: dict[int, list[int]] = {}
        for record_index, entity_row, probability, margin_value in zip(
            decision_record, decision_entity, decision_prob, decision_margin
        ):
            if probability >= args.threshold and margin_value >= args.margin:
                assigned.setdefault(entity_row, []).append(record_index)

        for entity_row, entity_id in enumerate(index.entity_ids):
            ids = assigned.get(entity_row)
            if ids:
                matching_out.write(
                    f"{entity_id}\t{','.join(record_ids[i] for i in ids)}\n"
                )
            else:
                matching_out.write(f"{entity_id}\t\n")

        n_assigned_country = sum(len(v) for v in assigned.values())
        if candidate_out is not None and pair_entity:
            all_entity = np.concatenate(pair_entity)
            all_record = np.concatenate(pair_record)
            del pair_entity, pair_record
            order = np.argsort(all_entity, kind="stable")
            all_entity = all_entity[order]
            all_record = all_record[order]
            boundaries = np.searchsorted(
                all_entity, np.arange(index.size + 1), side="left"
            )
            for entity_row, entity_id in enumerate(index.entity_ids):
                lo, hi = boundaries[entity_row], boundaries[entity_row + 1]
                if lo == hi:
                    candidate_out.write(f"{entity_id}\t\n")
                else:
                    seen: dict[str, None] = {}
                    for i in all_record[lo:hi]:
                        seen.setdefault(record_ids[i], None)
                    candidate_out.write(f"{entity_id}\t{','.join(seen)}\n")
            del all_entity, all_record
        elif candidate_out is not None:
            for entity_id in index.entity_ids:
                candidate_out.write(f"{entity_id}\t\n")

        grand_entities += index.size
        grand_assigned += n_assigned_country
        grand_pairs += n_pairs
        print(
            f"  {country}: {index.size:,} entities, {n_records:,} records, "
            f"{n_pairs:,} pairs, {n_assigned_country:,} assigned "
            f"({len(assigned):,} entities matched)  [{time.time() - country_started:.0f}s]",
            flush=True,
        )
        del index, assigned, record_ids

    matching_out.close()
    if candidate_out is not None:
        candidate_out.close()

    print(
        f"\nwrote {matching_path} ({matching_path.stat().st_size / 1e6:.1f} MB)"
    )
    if candidate_out is not None:
        print(f"wrote {candidate_path} ({candidate_path.stat().st_size / 1e9:.2f} GB)")
    print(
        f"\ntotals: {grand_entities:,} entities, {grand_pairs:,} candidate pairs, "
        f"{grand_assigned:,} records assigned "
        f"({grand_assigned / max(grand_entities, 1):.2f} per entity)"
    )
    print(f"total seconds: {time.time() - started:.0f}")


if __name__ == "__main__":
    main()
