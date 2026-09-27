"""Streaming inference over a whole country.

Memory is the binding constraint: at k=5 the full test set produces roughly 93M
candidate pairs, and a float32 feature matrix that size would be about 12 GB.
It is never built.

The retrieval direction makes that avoidable. Because candidates are retrieved
*per Source 2/3 record*, every pair belonging to a record is produced in the same
chunk, and the decision for that record -- which entity owns it, if any -- can be
made immediately and the features discarded. What survives a chunk is one small
row per record: its best entity, that probability, and the margin to its
runner-up. Peak memory is therefore set by the chunk size, not the dataset.

Candidate pairs, which the submission also asks for, are streamed straight to
disk as flat ``entity_id<TAB>record_id`` lines and grouped later by an external
sort, for the same reason: 93M of them will not fit in a dictionary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, TextIO

import numpy as np

from .candidates import CountryIndex, Shortlist, retrieve, shortlist_context
from .features import PairFeaturizer


@dataclass
class RecordDecision:
    """The single row kept per Source 2/3 record after its chunk is discarded."""

    entity_id: str
    probability: float
    margin: float


def iter_chunks(
    rows: Iterator[tuple[str, str, str, bool]], size: int
) -> Iterator[list[tuple[str, str, str, bool]]]:
    """Group a record stream into fixed-size chunks."""
    batch: list[tuple[str, str, str, bool]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def score_chunk(
    index: CountryIndex,
    featurizer: PairFeaturizer,
    shortlists: list[Shortlist],
    predict: Callable[[np.ndarray], np.ndarray],
) -> tuple[dict[str, RecordDecision], list[tuple[str, str]]]:
    """Featurise and score one chunk, reducing it to one decision per record.

    Returns the per-record decisions and the flat candidate pairs, the latter
    only so the caller can stream them to disk.
    """
    feature_rows: list[list[float]] = []
    owners: list[tuple[str, str]] = []  # (record_id, entity_id) parallel to rows

    for shortlist in shortlists:
        right = featurizer.parse(shortlist.record_id, shortlist.name, shortlist.address)
        for index_row in shortlist.rows():
            left = featurizer.parse(
                index.entity_ids[index_row],
                index.names[index_row],
                index.addresses[index_row],
            )
            feature_rows.append(
                featurizer.features(left, right, shortlist_context(shortlist, index_row))
            )
            owners.append((shortlist.record_id, index.entity_ids[index_row]))

    if not feature_rows:
        return {}, []

    matrix = np.asarray(feature_rows, dtype=np.float32)
    probabilities = predict(matrix)
    del feature_rows, matrix

    # Reduce to the best and second-best entity per record.
    best: dict[str, tuple[float, str]] = {}
    second: dict[str, float] = {}
    for (record_id, entity_id), probability in zip(owners, probabilities):
        probability = float(probability)
        current = best.get(record_id)
        if current is None or probability > current[0]:
            if current is not None and current[0] > second.get(record_id, 0.0):
                second[record_id] = current[0]
            best[record_id] = (probability, entity_id)
        elif probability > second.get(record_id, 0.0):
            second[record_id] = probability

    decisions = {
        record_id: RecordDecision(
            entity_id=entity_id,
            probability=probability,
            margin=probability - second.get(record_id, 0.0),
        )
        for record_id, (probability, entity_id) in best.items()
    }
    return decisions, owners


def run_country(
    index: CountryIndex,
    records: Iterator[tuple[str, str, str, bool]],
    predict: Callable[[np.ndarray], np.ndarray],
    k: int,
    chunk_rows: int = 20000,
    keep_floor: float = 0.02,
    candidate_sink: TextIO | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, RecordDecision]:
    """Score every record of one country, keeping one row per record.

    ``keep_floor`` drops records whose best probability is hopeless. It must stay
    below every threshold that will later be considered, or tuning would be
    reading a truncated distribution; at 0.02 it discards only records no rule
    would ever assign, while removing most of the dictionary's bulk.

    ``candidate_sink``, when given, receives every candidate pair as
    ``entity_id<TAB>record_id`` for later grouping into ``candidate_pairs.tsv``.
    """
    featurizer = PairFeaturizer(index.token_idf())
    kept: dict[str, RecordDecision] = {}
    n_records = 0
    n_pairs = 0

    for batch in iter_chunks(records, chunk_rows):
        shortlists = retrieve(index, batch, k=k, chunk_rows=chunk_rows)
        decisions, pairs = score_chunk(index, featurizer, shortlists, predict)

        if candidate_sink is not None:
            candidate_sink.writelines(
                f"{entity_id}\t{record_id}\n" for record_id, entity_id in pairs
            )

        for record_id, decision in decisions.items():
            if decision.probability >= keep_floor:
                kept[record_id] = decision

        n_records += len(batch)
        n_pairs += len(pairs)
        # The parse cache pays off inside a chunk; across the whole country it
        # would grow without bound.
        featurizer.clear()
        if progress is not None:
            progress(n_records, n_pairs)

    return kept


def write_matching_results(
    path: str | Path,
    entity_ids: list[str],
    decisions: dict[str, RecordDecision],
    threshold: float,
    margin: float,
) -> tuple[int, int]:
    """Write ``matching_results.tsv``.

    Every entity gets exactly one row, including those matched to nothing: an
    empty list is the singleton prediction and is worth full credit when right.
    """
    matches: dict[str, list[str]] = {entity_id: [] for entity_id in entity_ids}
    n_assigned = 0
    for record_id, decision in decisions.items():
        if decision.probability < threshold or decision.margin < margin:
            continue
        bucket = matches.get(decision.entity_id)
        if bucket is not None:
            bucket.append(record_id)
            n_assigned += 1

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write("source1_entity_id\tmatched_entity_ids\n")
        for entity_id in entity_ids:
            handle.write(f"{entity_id}\t{','.join(matches[entity_id])}\n")

    n_singletons = sum(1 for ids in matches.values() if not ids)
    return n_assigned, n_singletons
