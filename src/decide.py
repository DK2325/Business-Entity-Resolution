"""Turning pair probabilities into final match lists.

The rule is built around the one hard structural fact in the data: **no Source
2/3 record belongs to more than one Source 1 entity**, verified across all
7,638,365 training links. So the decision is not "is this pair a match?" asked
independently of every other pair -- it is "which single entity, if any, owns
this record?".

That reframing is worth a lot under macro F_0.5. Thresholding pairs
independently lets one record be handed to several entities, and every surplus
copy is a false merge charged against a different entity's precision. Forcing
each record to pick at most one owner makes those errors impossible by
construction, and costs nothing real, because a record genuinely has one owner.

Two knobs are then tuned on the holdout:

``threshold``
    How confident the winning pair must be. This is the precision/recall dial,
    and under F_0.5 its optimum sits well above 0.5.

``margin``
    How far the winner must lead the runner-up. When a record cannot tell two
    entities apart, assigning it to the marginally higher-scoring one is close to
    a coin flip, and a coin flip that lands wrong is a false merge on one entity
    and a miss on another. Requiring a clear lead withholds those.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np


def assign_records(
    record_ids: Sequence[str],
    entity_ids: Sequence[str],
    probabilities: Sequence[float],
    threshold: float,
    margin: float = 0.0,
) -> dict[str, str]:
    """Assign each record to at most one entity.

    Input is the flat pair table: three parallel sequences over candidate pairs.
    Returns ``{record_id: entity_id}`` for the records that were assigned.

    A record is assigned when its best pair clears ``threshold`` and leads its
    own runner-up by at least ``margin``.
    """
    best: dict[str, tuple[float, str]] = {}
    second: dict[str, float] = {}

    for record_id, entity_id, probability in zip(record_ids, entity_ids, probabilities):
        current = best.get(record_id)
        if current is None or probability > current[0]:
            if current is not None:
                second[record_id] = max(second.get(record_id, 0.0), current[0])
            best[record_id] = (float(probability), entity_id)
        else:
            if probability > second.get(record_id, 0.0):
                second[record_id] = float(probability)

    assignment: dict[str, str] = {}
    for record_id, (probability, entity_id) in best.items():
        if probability < threshold:
            continue
        if probability - second.get(record_id, 0.0) < margin:
            continue
        assignment[record_id] = entity_id
    return assignment


def assignment_to_matches(
    assignment: Mapping[str, str],
    all_entity_ids: Iterable[str],
) -> dict[str, list[str]]:
    """Invert ``{record: entity}`` into ``{entity: [records]}``.

    Every entity in ``all_entity_ids`` gets a key, including those that were
    assigned nothing. Those empty lists are the singleton predictions, and under
    this metric a correct one is worth a full 1.0 -- dropping them would forfeit
    that credit and also fail the submission's one-row-per-entity rule.
    """
    matches: dict[str, list[str]] = {entity_id: [] for entity_id in all_entity_ids}
    for record_id, entity_id in assignment.items():
        bucket = matches.get(entity_id)
        if bucket is None:
            matches[entity_id] = [record_id]
        else:
            bucket.append(record_id)
    return matches


def tune_threshold(
    record_ids: Sequence[str],
    entity_ids: Sequence[str],
    probabilities: Sequence[float],
    truth: Mapping[str, Sequence[str]],
    thresholds: Iterable[float] = tuple(np.arange(0.05, 1.0, 0.025)),
    margins: Iterable[float] = (0.0,),
    scorer=None,
) -> list[dict]:
    """Score every (threshold, margin) combination against the exact metric.

    ``truth`` defines the evaluation set, so entities that received no candidate
    at all are still scored -- correctly as singleton predictions.
    """
    if scorer is None:
        from .metrics import macro_fbeta_report

        scorer = macro_fbeta_report

    results: list[dict] = []
    for margin in margins:
        for threshold in thresholds:
            assignment = assign_records(
                record_ids, entity_ids, probabilities, threshold=threshold, margin=margin
            )
            predictions = assignment_to_matches(assignment, truth.keys())
            report = scorer(predictions, truth)
            report["threshold"] = round(float(threshold), 4)
            report["margin"] = round(float(margin), 4)
            report["n_assigned"] = len(assignment)
            results.append(report)
    results.sort(key=lambda row: row["macro_f0.5"], reverse=True)
    return results
