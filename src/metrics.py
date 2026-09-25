"""The competition metric: macro-averaged F-beta with beta = 0.5.

F_0.5 is computed per Source 1 entity and then averaged over every entity in the
evaluation set, singletons included. Two properties of this metric drive the
whole design of the pipeline:

* Precision is weighted roughly 2x over recall, so a false merge costs more than
  a missed link.
* An entity whose true match list is empty scores 1.0 for an empty prediction and
  0.0 for any non-empty one. Singletons are 5.6% of the training entities, so
  predicting them correctly is worth about 5.6 points of the final score and
  over-predicting on them is pure loss.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

BETA = 0.5
_BETA_SQ = BETA * BETA


def entity_fbeta(predicted: Iterable[str], truth: Iterable[str], beta: float = BETA) -> float:
    """F-beta for one Source 1 entity.

    Both arguments are treated as sets: the submission format forbids duplicates
    within an ID list, and ordering carries no meaning.

    The empty/empty case returns 1.0 — correctly identifying a singleton earns
    full credit. Any other case involving an empty side returns 0.0.
    """
    pred_set = set(predicted)
    true_set = set(truth)

    if not pred_set and not true_set:
        return 1.0
    if not pred_set or not true_set:
        return 0.0

    hits = len(pred_set & true_set)
    if hits == 0:
        return 0.0

    precision = hits / len(pred_set)
    recall = hits / len(true_set)

    beta_sq = beta * beta
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def macro_fbeta(
    predictions: Mapping[str, Sequence[str]],
    truth: Mapping[str, Sequence[str]],
    beta: float = BETA,
) -> float:
    """Macro-average of :func:`entity_fbeta` over every entity in ``truth``.

    ``truth`` defines the evaluation set: an entity missing from ``predictions``
    is scored as an empty prediction, which mirrors the scorer's treatment and
    keeps an incomplete submission from looking better than it is.
    """
    if not truth:
        return 0.0
    total = 0.0
    for entity_id, true_ids in truth.items():
        total += entity_fbeta(predictions.get(entity_id, ()), true_ids, beta)
    return total / len(truth)


def macro_fbeta_report(
    predictions: Mapping[str, Sequence[str]],
    truth: Mapping[str, Sequence[str]],
    beta: float = BETA,
) -> dict[str, float]:
    """Score plus the diagnostics needed to steer the precision/recall trade-off.

    Reports the macro score split over singleton and non-singleton entities,
    because the two are improved by opposite moves: singletons want a more
    conservative decision rule, non-singletons a more permissive one.
    """
    singleton_scores: list[float] = []
    matched_scores: list[float] = []
    micro_hits = micro_pred = micro_true = 0

    for entity_id, true_ids in truth.items():
        pred_ids = predictions.get(entity_id, ())
        score = entity_fbeta(pred_ids, true_ids, beta)
        (singleton_scores if not true_ids else matched_scores).append(score)

        pred_set, true_set = set(pred_ids), set(true_ids)
        micro_hits += len(pred_set & true_set)
        micro_pred += len(pred_set)
        micro_true += len(true_set)

    all_scores = singleton_scores + matched_scores
    return {
        "macro_f0.5": sum(all_scores) / len(all_scores) if all_scores else 0.0,
        "macro_f0.5_singletons": (
            sum(singleton_scores) / len(singleton_scores) if singleton_scores else float("nan")
        ),
        "macro_f0.5_matched": (
            sum(matched_scores) / len(matched_scores) if matched_scores else float("nan")
        ),
        "n_singletons": float(len(singleton_scores)),
        "n_matched": float(len(matched_scores)),
        "micro_precision": micro_hits / micro_pred if micro_pred else 0.0,
        "micro_recall": micro_hits / micro_true if micro_true else 0.0,
    }
