"""Score sharded holdout decisions and choose the decision rule.

Consumes the ``shard_*.npz`` files written by ``scripts/predict_parallel.py`` on
the training split and reports macro F_0.5 over holdout entities only, sweeping
threshold and margin.

The conditions are the test conditions: every Source 1 entity of the country was
indexed, every Source 2/3 record was queried -- distractors that belong to
nobody, and records owned by train-split entities, both of which can be stolen by
a holdout entity and become a false merge. Only entities on the holdout side are
scored, and the model never saw their pairs.

Beyond the headline score it reports the two diagnostics that decide whether the
rule is safe: how many singletons were given a match (each costs a full 1.0), and
how many unowned distractor records were assigned to something.

Usage::

    python scripts/eval_shards.py --country India --shards artifacts/shards_train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.candidates import iter_country_rows
from src.metrics import macro_fbeta_report
from src.splits import load_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", default="India")
    parser.add_argument("--shards", default=None)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    shard_dir = Path(args.shards) if args.shards else config.ARTIFACTS / "shards_train"
    shard_paths = sorted(shard_dir.glob(f"shard_{args.country}_*.npz"))
    if not shard_paths:
        raise SystemExit(f"no shards for {args.country} in {shard_dir}")
    print(f"shards: {len(shard_paths)}")

    # Entity rows are shard-independent (every worker built the same index from
    # the same file), but record indices are local to a shard, so record ids are
    # resolved per shard.
    entity_ids = [e for e, _n, _a in iter_country_rows(config.TRAIN_FILES["source1"], args.country)]
    print(f"{args.country} Source 1 entities: {len(entity_ids):,}")

    rec_ids: list[str] = []
    ent_rows: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    margins: list[np.ndarray] = []
    for path in shard_paths:
        data = np.load(path, allow_pickle=False)
        shard_record_ids = data["record_ids"]
        idx = data["record_index"]
        rec_ids.extend(shard_record_ids[idx].tolist())
        ent_rows.append(data["entity_row"])
        probs.append(data["probability"])
        margins.append(data["margin"])
    entity_row = np.concatenate(ent_rows)
    probability = np.concatenate(probs)
    margin = np.concatenate(margins)
    del ent_rows, probs, margins
    print(f"decisions: {len(rec_ids):,}")

    truth_all: dict[str, list[str]] = {}
    with open(config.TRAIN_FILES["ground_truth"], encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            ids = parts[1] if len(parts) > 1 else ""
            truth_all[parts[0]] = [x for x in ids.split(",") if x]

    owner_of: dict[str, str] = {}
    for entity_id, ids in truth_all.items():
        for record_id in ids:
            owner_of[record_id] = entity_id

    splits = load_splits(config.ARTIFACTS / "splits.json")
    valid_entities = set(splits["holdout"]["valid"])
    holdout = [e for e in entity_ids if e in valid_entities]
    truth = {e: truth_all.get(e, []) for e in holdout}
    holdout_rows = {i for i, e in enumerate(entity_ids) if e in valid_entities}
    n_singletons = sum(1 for v in truth.values() if not v)
    print(
        f"holdout entities: {len(truth):,} "
        f"({n_singletons:,} singletons, {n_singletons / max(len(truth), 1):.2%})\n"
    )

    # The sweep is vectorised. Done as a Python loop it would be ~90 passes over
    # a few million decisions, which costs longer than the inference that made
    # them. Everything below reduces to bincounts over entity rows.
    n_entities = len(entity_ids)
    entity_is_holdout = np.zeros(n_entities, dtype=bool)
    for row in holdout_rows:
        entity_is_holdout[row] = True

    # Per decision: is the assignment correct, and does the record belong to
    # anyone at all (an unowned record is a distractor).
    is_correct = np.fromiter(
        (owner_of.get(rec_ids[i]) == entity_ids[int(entity_row[i])] for i in range(len(rec_ids))),
        dtype=bool,
        count=len(rec_ids),
    )
    has_owner = np.fromiter(
        (rec_ids[i] in owner_of for i in range(len(rec_ids))), dtype=bool, count=len(rec_ids)
    )
    in_holdout = entity_is_holdout[entity_row]

    truth_count = np.zeros(n_entities, dtype=np.int32)
    for row, entity_id in enumerate(entity_ids):
        if entity_is_holdout[row]:
            truth_count[row] = len(truth_all.get(entity_id, []))
    holdout_mask = entity_is_holdout
    n_holdout = int(holdout_mask.sum())
    singleton_mask = holdout_mask & (truth_count == 0)
    matched_mask = holdout_mask & (truth_count > 0)

    def evaluate(threshold: float, margin_cut: float) -> dict:
        keep = (probability >= threshold) & (margin >= margin_cut) & in_holdout
        rows_kept = entity_row[keep]
        predicted = np.bincount(rows_kept, minlength=n_entities).astype(np.int32)
        true_positive = np.bincount(
            entity_row[keep & is_correct], minlength=n_entities
        ).astype(np.int32)

        with np.errstate(divide="ignore", invalid="ignore"):
            precision = np.where(predicted > 0, true_positive / np.maximum(predicted, 1), 0.0)
            recall = np.where(truth_count > 0, true_positive / np.maximum(truth_count, 1), 0.0)
            denominator = 0.25 * precision + recall
            fbeta = np.where(denominator > 0, 1.25 * precision * recall / np.maximum(denominator, 1e-12), 0.0)
        # An entity with no true matches scores 1.0 exactly when nothing was
        # predicted for it, and 0.0 otherwise.
        fbeta = np.where((truth_count == 0) & (predicted == 0), 1.0, fbeta)
        fbeta = np.where((truth_count == 0) & (predicted > 0), 0.0, fbeta)

        total_predicted = int(predicted[holdout_mask].sum())
        total_tp = int(true_positive[holdout_mask].sum())
        total_true = int(truth_count[holdout_mask].sum())
        return {
            "threshold": round(float(threshold), 3),
            "margin": float(margin_cut),
            "macro_f0.5": float(fbeta[holdout_mask].mean()) if n_holdout else 0.0,
            "singletons": float(fbeta[singleton_mask].mean()) if singleton_mask.any() else float("nan"),
            "matched": float(fbeta[matched_mask].mean()) if matched_mask.any() else float("nan"),
            "micro_p": total_tp / total_predicted if total_predicted else 0.0,
            "micro_r": total_tp / total_true if total_true else 0.0,
            "n_assigned": total_predicted,
            "fp_singletons": int((singleton_mask & (predicted > 0)).sum()),
            "n_distractor": int((keep & ~has_owner).sum()),
        }

    print(
        f"{'thr':>6}{'margin':>8}{'F0.5':>9}{'single':>9}{'matched':>9}"
        f"{'micro-P':>9}{'micro-R':>9}{'assigned':>10}{'FP-single':>11}{'distract':>10}"
    )
    rows = [
        evaluate(threshold, margin_cut)
        for margin_cut in (0.0, 0.05, 0.10, 0.20, 0.30)
        for threshold in np.arange(0.10, 0.96, 0.05)
    ]
    rows.sort(key=lambda r: r["macro_f0.5"], reverse=True)
    for row in rows[: args.top]:
        print(
            f"{row['threshold']:>6.2f}{row['margin']:>8.2f}{row['macro_f0.5']:>9.4f}"
            f"{row['singletons']:>9.4f}{row['matched']:>9.4f}"
            f"{row['micro_p']:>9.4f}{row['micro_r']:>9.4f}"
            f"{row['n_assigned']:>10,}{row['fp_singletons']:>11,}{row['n_distractor']:>10,}"
        )

    best = rows[0]
    n_singleton_entities = int(singleton_mask.sum())
    print(
        f"\nBEST  threshold={best['threshold']}  margin={best['margin']}\n"
        f"  macro F_0.5             : {best['macro_f0.5']:.4f}\n"
        f"  on singletons           : {best['singletons']:.4f} ({n_singleton_entities:,} entities)\n"
        f"  on matched entities     : {best['matched']:.4f} ({int(matched_mask.sum()):,} entities)\n"
        f"  micro precision         : {best['micro_p']:.4f}\n"
        f"  micro recall            : {best['micro_r']:.4f}\n"
        f"  false merges on singles : {best['fp_singletons']:,} "
        f"({best['fp_singletons'] / max(n_singleton_entities, 1):.2%} of singletons)\n"
        f"  distractors assigned    : {best['n_distractor']:,}\n"
        f"  records assigned        : {best['n_assigned']:,}"
    )


if __name__ == "__main__":
    main()
