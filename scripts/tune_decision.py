"""Improve the decision layer using saved decisions, without re-running inference.

Inference produced, for every Source 2/3 record, the entity it scored highest,
that probability, and the margin to its runner-up. Everything here re-reads that
and asks a different question of it: given a record's best entity and how
confident the model is, should the assignment be kept?

What this cannot do, and why: the saved decisions hold only each record's
*argmax* entity, so a record can be accepted or rejected but never moved to a
different entity. Every rule below therefore operates inside "keep or drop the
argmax", which is where the 11.6% singleton false-merge rate lives anyway.

**Overfitting control.** Holdout entities are split 50/50 into halves A and B by
a hash of the entity id. Every rule is tuned on A and reported on both, then the
roles are swapped. A rule is only worth keeping if it improves *both* halves in
*both* directions -- with ~159k entities and rules carrying a handful of
parameters it is easy to fit noise, and the whole point of a decision layer is
that it generalises to a test set we cannot see.

Usage::

    python scripts/tune_decision.py --country India
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.candidates import iter_country_rows
from src.splits import load_splits

BETA_SQ = 0.25  # beta = 0.5


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_decisions(shard_dir: Path, country: str):
    """Flatten the shards into parallel arrays of (record_id, entity_row, p, margin)."""
    paths = sorted(shard_dir.glob(f"shard_{country}_*.npz"))
    if not paths:
        raise SystemExit(f"no shards for {country} in {shard_dir}")
    record_ids: list[str] = []
    entity_rows: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    margins: list[np.ndarray] = []
    for path in paths:
        data = np.load(path, allow_pickle=False)
        shard_records = data["record_ids"]
        record_ids.extend(shard_records[data["record_index"]].tolist())
        entity_rows.append(data["entity_row"])
        probs.append(data["probability"])
        margins.append(data["margin"])
    return (
        np.asarray(record_ids),
        np.concatenate(entity_rows),
        np.concatenate(probs),
        np.concatenate(margins),
    )


def half_of(entity_id: str) -> int:
    """Stable 50/50 split of entities, independent of ordering or run."""
    digest = hashlib.blake2b(entity_id.encode("utf-8"), digest_size=8).digest()
    return digest[0] & 1


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def macro_fbeta_from_counts(
    predicted: np.ndarray, true_positive: np.ndarray, truth_count: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    """Macro F_0.5 over the entities selected by ``mask``, from per-entity counts."""
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, true_positive / np.maximum(predicted, 1), 0.0)
        recall = np.where(truth_count > 0, true_positive / np.maximum(truth_count, 1), 0.0)
        denominator = BETA_SQ * precision + recall
        fbeta = np.where(
            denominator > 0, 1.25 * precision * recall / np.maximum(denominator, 1e-12), 0.0
        )
    fbeta = np.where((truth_count == 0) & (predicted == 0), 1.0, fbeta)
    fbeta = np.where((truth_count == 0) & (predicted > 0), 0.0, fbeta)

    singles = mask & (truth_count == 0)
    matched = mask & (truth_count > 0)
    total_pred = int(predicted[mask].sum())
    total_tp = int(true_positive[mask].sum())
    total_true = int(truth_count[mask].sum())
    return {
        "macro": float(fbeta[mask].mean()) if mask.any() else 0.0,
        "singleton": float(fbeta[singles].mean()) if singles.any() else float("nan"),
        "matched": float(fbeta[matched].mean()) if matched.any() else float("nan"),
        "precision": total_tp / total_pred if total_pred else 0.0,
        "recall": total_tp / total_true if total_true else 0.0,
        "assigned": total_pred,
        "fp_singletons": int((singles & (predicted > 0)).sum()),
    }


def counts_from_keep(
    keep: np.ndarray,
    entity_row: np.ndarray,
    is_correct: np.ndarray,
    n_entities: int,
) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.bincount(entity_row[keep], minlength=n_entities).astype(np.int32)
    true_positive = np.bincount(
        entity_row[keep & is_correct], minlength=n_entities
    ).astype(np.int32)
    return predicted, true_positive


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

def fit_isotonic(probabilities: np.ndarray, correct: np.ndarray):
    """Isotonic calibration of raw scores into P(assignment is correct)."""
    from sklearn.isotonic import IsotonicRegression

    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model.fit(probabilities, correct.astype(np.float64))
    return model


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------

def rule_threshold(prob: np.ndarray, margin: np.ndarray, threshold: float, margin_cut: float):
    return (prob >= threshold) & (margin >= margin_cut)


def rule_rank_threshold(
    prob: np.ndarray,
    entity_row: np.ndarray,
    first_threshold: float,
    extra_threshold: float,
):
    """One threshold for an entity's best record, another for its additional ones.

    The first record an entity gets is the one that decides singleton-vs-matched,
    and that decision is worth a full 1.0 either way. Additional records only
    trade precision against recall inside an already-matched entity, so they can
    be held to a different standard.
    """
    order = np.lexsort((-prob, entity_row))
    sorted_entity = entity_row[order]
    is_first = np.empty(order.shape[0], dtype=bool)
    is_first[0] = True
    is_first[1:] = sorted_entity[1:] != sorted_entity[:-1]
    rank_is_first = np.zeros_like(is_first)
    rank_is_first[order] = is_first
    return np.where(rank_is_first, prob >= first_threshold, prob >= extra_threshold)


def apply_singleton_gate(
    keep: np.ndarray,
    calibrated: np.ndarray,
    entity_row: np.ndarray,
    n_entities: int,
    gate: float,
) -> np.ndarray:
    """Drop every record of an entity whose best candidate is too weak.

    The expected-F_0.5 rule decides how *many* records to keep, but its
    empty-set score -- the product of (1 - p) over the candidates -- collapses
    once an entity has several mediocre candidates, so it almost never predicts
    a singleton. That is exactly backwards for this metric: a singleton
    predicted empty is worth a full 1.0, and any match on it is worth 0. This
    gate restores the decision the subset search cannot make, by requiring the
    entity's single best candidate to clear a bar before anything is kept.
    """
    if gate <= 0.0:
        return keep
    best = np.zeros(n_entities, dtype=np.float32)
    np.maximum.at(best, entity_row, calibrated.astype(np.float32))
    return keep & (best[entity_row] >= gate)


def rule_expected_fbeta(
    calibrated: np.ndarray,
    entity_row: np.ndarray,
    n_entities: int,
    floor: float = 0.02,
    recall_inflation: float = 1.0,
):
    """Per entity, keep the prefix of its candidates that maximises expected F_0.5.

    For an entity whose candidates have calibrated probabilities p1 >= p2 >= ...,
    keeping the top m gives expected true positives ``sum(p_i, i<=m)`` and an
    expected truth size estimated as ``sum(p_i)`` over all its candidates. The
    empty set is scored as the probability that the entity really has no match,
    ``prod(1 - p_i)``, which is exactly the singleton case the metric rewards.

    ``recall_inflation`` scales the estimated truth size to account for true
    matches that blocking never retrieved; those are invisible here but do count
    against recall in the real metric.
    """
    active = calibrated >= floor
    idx = np.nonzero(active)[0]
    if idx.size == 0:
        return np.zeros(calibrated.shape[0], dtype=bool)

    order = idx[np.lexsort((-calibrated[idx], entity_row[idx]))]
    sorted_entity = entity_row[order]
    sorted_prob = calibrated[order]

    starts = np.flatnonzero(
        np.concatenate(([True], sorted_entity[1:] != sorted_entity[:-1]))
    )
    ends = np.concatenate((starts[1:], [order.shape[0]]))

    keep = np.zeros(calibrated.shape[0], dtype=bool)
    for start, end in zip(starts.tolist(), ends.tolist()):
        probs = sorted_prob[start:end]
        cumulative = np.cumsum(probs)
        expected_truth = cumulative[-1] * recall_inflation
        counts = np.arange(1, probs.shape[0] + 1, dtype=np.float64)

        # Plug-in expectation: F_0.5 evaluated at the expected counts.
        precision = cumulative / counts
        recall = cumulative / max(expected_truth, 1e-9)
        denominator = BETA_SQ * precision + recall
        scores = np.where(
            denominator > 0, 1.25 * precision * recall / np.maximum(denominator, 1e-12), 0.0
        )

        empty_score = float(np.prod(1.0 - probs))
        best_m = int(np.argmax(scores)) + 1
        if empty_score >= scores[best_m - 1]:
            continue
        keep[order[start : start + best_m]] = True
    return keep


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", default="India")
    parser.add_argument("--shards", default=None)
    args = parser.parse_args()

    shard_dir = Path(args.shards) if args.shards else config.ARTIFACTS / "shards_train"
    record_ids, entity_row, prob, margin = load_decisions(shard_dir, args.country)
    print(f"decisions: {record_ids.shape[0]:,}")

    entity_ids = [e for e, _n, _a in iter_country_rows(config.TRAIN_FILES["source1"], args.country)]
    n_entities = len(entity_ids)
    print(f"{args.country} entities: {n_entities:,}")

    truth_all: dict[str, list[str]] = {}
    with open(config.TRAIN_FILES["ground_truth"], encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            ids = parts[1] if len(parts) > 1 else ""
            truth_all[parts[0]] = [x for x in ids.split(",") if x]

    owner_of: dict[str, str] = {}
    for entity, ids in truth_all.items():
        for record in ids:
            owner_of[record] = entity

    splits = load_splits(config.ARTIFACTS / "splits.json")
    valid = set(splits["holdout"]["valid"])

    truth_count = np.zeros(n_entities, dtype=np.int32)
    in_holdout = np.zeros(n_entities, dtype=bool)
    half = np.zeros(n_entities, dtype=np.int8)
    for row, entity in enumerate(entity_ids):
        if entity in valid:
            in_holdout[row] = True
            truth_count[row] = len(truth_all.get(entity, []))
            half[row] = half_of(entity)

    entity_id_array = np.asarray(entity_ids)
    is_correct = np.fromiter(
        (owner_of.get(record_ids[i]) == entity_id_array[entity_row[i]] for i in range(record_ids.shape[0])),
        dtype=bool,
        count=record_ids.shape[0],
    )
    decision_in_holdout = in_holdout[entity_row]
    decision_half = half[entity_row]

    mask_a = in_holdout & (half == 0)
    mask_b = in_holdout & (half == 1)
    print(
        f"holdout entities: {int(in_holdout.sum()):,} "
        f"(A {int(mask_a.sum()):,}, B {int(mask_b.sum()):,})"
    )
    print(
        f"singletons: A {int((mask_a & (truth_count == 0)).sum()):,}, "
        f"B {int((mask_b & (truth_count == 0)).sum()):,}\n"
    )

    def evaluate(keep_all: np.ndarray) -> tuple[dict, dict]:
        keep = keep_all & decision_in_holdout
        predicted, true_positive = counts_from_keep(keep, entity_row, is_correct, n_entities)
        return (
            macro_fbeta_from_counts(predicted, true_positive, truth_count, mask_a),
            macro_fbeta_from_counts(predicted, true_positive, truth_count, mask_b),
        )

    results: list[tuple[str, dict, dict]] = []

    # ---- baseline ---------------------------------------------------------
    for threshold in (0.65, 0.70, 0.75, 0.80):
        keep = rule_threshold(prob, margin, threshold, 0.0)
        a, b = evaluate(keep)
        results.append((f"threshold {threshold:.2f}", a, b))

    # ---- first vs additional ---------------------------------------------
    for first, extra in ((0.60, 0.80), (0.65, 0.80), (0.70, 0.60), (0.70, 0.80), (0.60, 0.70)):
        keep = rule_rank_threshold(prob, entity_row, first, extra)
        a, b = evaluate(keep)
        results.append((f"first {first:.2f} / extra {extra:.2f}", a, b))

    # ---- calibrated expected-F0.5 ----------------------------------------
    # Calibrate on A, apply everywhere; then calibrate on B and re-apply, so the
    # reported number for each half comes from a model fitted on the other.
    fit_mask_a = decision_in_holdout & (decision_half == 0)
    fit_mask_b = decision_in_holdout & (decision_half == 1)
    iso_a = fit_isotonic(prob[fit_mask_a], is_correct[fit_mask_a])
    iso_b = fit_isotonic(prob[fit_mask_b], is_correct[fit_mask_b])
    cal_from_a = iso_a.predict(prob)
    cal_from_b = iso_b.predict(prob)

    # Cache the subset search per inflation; the gate is then a cheap filter on
    # top, so the (inflation x gate) grid costs one search per inflation.
    searched: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for inflation in (1.0, 1.20, 1.40, 1.60, 2.00):
        searched[inflation] = (
            rule_expected_fbeta(cal_from_b, entity_row, n_entities, recall_inflation=inflation),
            rule_expected_fbeta(cal_from_a, entity_row, n_entities, recall_inflation=inflation),
        )

    for inflation, (keep_using_b, keep_using_a) in searched.items():
        for gate in (0.0, 0.50, 0.60, 0.70, 0.80, 0.90):
            # Half A is always scored with the model fitted on B, and vice versa.
            keep_a_side = apply_singleton_gate(
                keep_using_b, cal_from_b, entity_row, n_entities, gate
            ) & decision_in_holdout
            keep_b_side = apply_singleton_gate(
                keep_using_a, cal_from_a, entity_row, n_entities, gate
            ) & decision_in_holdout
            pred_a, tp_a = counts_from_keep(keep_a_side, entity_row, is_correct, n_entities)
            pred_b, tp_b = counts_from_keep(keep_b_side, entity_row, is_correct, n_entities)
            a = macro_fbeta_from_counts(pred_a, tp_a, truth_count, mask_a)
            b = macro_fbeta_from_counts(pred_b, tp_b, truth_count, mask_b)
            label = f"expF0.5 infl {inflation:.2f} gate {gate:.2f}"
            results.append((label, a, b))

    # ---- report -----------------------------------------------------------
    header = (
        f"{'rule':<34}{'F0.5 A':>9}{'F0.5 B':>9}{'single A':>10}{'single B':>10}"
        f"{'matched A':>11}{'prec A':>8}{'rec A':>8}{'FP-sgl A':>10}"
    )
    print(header)
    print("-" * len(header))
    baseline = None
    for name, a, b in results:
        if name == "threshold 0.70":
            baseline = (a, b)
        print(
            f"{name:<34}{a['macro']:>9.4f}{b['macro']:>9.4f}"
            f"{a['singleton']:>10.4f}{b['singleton']:>10.4f}"
            f"{a['matched']:>11.4f}{a['precision']:>8.4f}{a['recall']:>8.4f}"
            f"{a['fp_singletons']:>10,}"
        )

    if baseline is not None:
        print(f"\nbaseline (threshold 0.70): A {baseline[0]['macro']:.4f}  B {baseline[1]['macro']:.4f}")
        print("\nrules improving BOTH halves over baseline:")
        any_better = False
        for name, a, b in results:
            if name == "threshold 0.70":
                continue
            if a["macro"] > baseline[0]["macro"] and b["macro"] > baseline[1]["macro"]:
                any_better = True
                print(
                    f"  {name:<34} A {a['macro']:+.4f}  B {b['macro']:+.4f}"
                    f"   (A {a['macro']:.4f}, B {b['macro']:.4f})".replace(
                        f"A {a['macro']:+.4f}", f"A {a['macro'] - baseline[0]['macro']:+.4f}"
                    ).replace(
                        f"B {b['macro']:+.4f}", f"B {b['macro'] - baseline[1]['macro']:+.4f}"
                    )
                )
        if not any_better:
            print("  none")


if __name__ == "__main__":
    main()
