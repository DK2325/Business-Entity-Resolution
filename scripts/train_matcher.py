"""Train the pair classifier on the generated candidate pairs.

Trains only on pairs whose Source 1 entity is on the train side of the grouped
holdout, and reports on the held-out side. The reported figures are *pair-level*
(average precision, and precision/recall of the per-record assignment): they are
honest about precision, because a record's whole shortlist is present, but they
understate per-entity recall, because only a sample of each entity's records was
generated. The trustworthy macro F_0.5 comes from a full country pass in
``scripts/run_holdout_eval.py``; this script is for model selection, not for
choosing the final threshold.

Usage::

    python scripts/train_matcher.py --pairs artifacts/pairs_India.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import pyarrow.parquet as pq

from src import config
from src.features import FEATURE_NAMES


def load_pairs(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read pair tables into feature matrix, labels, split flag and ids."""
    feature_blocks, label_blocks, valid_blocks = [], [], []
    record_blocks, entity_blocks = [], []
    for path in paths:
        table = pq.read_table(path)
        n = table.num_rows
        print(f"  {path.name}: {n:,} pairs")
        block = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)
        for i, name in enumerate(FEATURE_NAMES):
            block[:, i] = table.column(name).to_numpy(zero_copy_only=False)
        feature_blocks.append(block)
        label_blocks.append(table.column("label").to_numpy(zero_copy_only=False))
        valid_blocks.append(table.column("is_valid").to_numpy(zero_copy_only=False))
        record_blocks.append(np.asarray(table.column("record_id").to_pylist()))
        entity_blocks.append(np.asarray(table.column("entity_id").to_pylist()))
        del table
    return (
        np.concatenate(feature_blocks),
        np.concatenate(label_blocks).astype(np.int8),
        np.concatenate(valid_blocks).astype(bool),
        np.concatenate(record_blocks),
        np.concatenate(entity_blocks),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="+", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--num-leaves", type=int, default=127)
    args = parser.parse_args()

    config.ensure_dirs()
    started = time.time()

    paths = [Path(p) for p in args.pairs]
    X, y, is_valid, record_ids, entity_ids = load_pairs(paths)
    print(
        f"\ntotal: {X.shape[0]:,} pairs x {X.shape[1]} features, "
        f"{int(y.sum()):,} positive ({y.mean():.2%})"
    )
    print(f"train pairs: {int((~is_valid).sum()):,}   holdout pairs: {int(is_valid.sum()):,}")

    model = lgb.LGBMClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=100,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=config.SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X[~is_valid],
        y[~is_valid],
        eval_set=[(X[is_valid], y[is_valid])],
        eval_metric="average_precision",
        feature_name=FEATURE_NAMES,
        callbacks=[lgb.log_evaluation(100)],
    )
    print(f"trained  [{time.time() - started:.0f}s]")

    probabilities = model.predict_proba(X[is_valid])[:, 1]
    y_valid = y[is_valid]
    records_valid = record_ids[is_valid]

    print("\ntop features by split gain:")
    for name, gain in sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda kv: -kv[1])[:15]:
        print(f"  {name:<28} {gain}")

    # Per-record assignment quality. Precision here is exact: every candidate a
    # record has is in the table, so the argmax is the real argmax.
    best: dict[str, tuple[float, int]] = {}
    truth_has_owner: dict[str, bool] = {}
    for record_id, probability, label in zip(records_valid, probabilities, y_valid):
        probability = float(probability)
        current = best.get(record_id)
        if current is None or probability > current[0]:
            best[record_id] = (probability, int(label))
        truth_has_owner[record_id] = truth_has_owner.get(record_id, False) or bool(label)

    n_with_owner = sum(truth_has_owner.values())
    print(
        f"\nholdout records: {len(best):,} "
        f"({n_with_owner:,} have their true owner in the shortlist, "
        f"{len(best) - n_with_owner:,} do not)"
    )
    print(f"\n{'thr':>6}{'assigned':>10}{'correct':>10}{'precision':>11}{'recall':>9}")
    for threshold in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        assigned = [(p, lab) for p, lab in best.values() if p >= threshold]
        correct = sum(lab for _p, lab in assigned)
        precision = correct / len(assigned) if assigned else 0.0
        recall = correct / max(n_with_owner, 1)
        print(f"{threshold:>6.1f}{len(assigned):>10,}{correct:>10,}{precision:>11.4f}{recall:>9.4f}")

    out = Path(args.out) if args.out else config.ARTIFACTS / "matcher.txt"
    model.booster_.save_model(str(out))
    print(f"\nsaved model to {out}")
    print(f"total seconds: {time.time() - started:.0f}")


if __name__ == "__main__":
    main()
