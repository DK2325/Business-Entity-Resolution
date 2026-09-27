"""Score the matcher under test conditions and tune the decision rule.

This is the only evaluation whose macro F_0.5 can be trusted, because it is the
only one that reproduces what happens at test time:

* the index holds **every** Source 1 entity of the country, so candidates face
  the full field of competitors;
* **every** Source 2/3 record of the country is queried -- distractors that
  belong to nobody, and records belonging to train-split entities, both of which
  can steal a holdout entity's record and cause a false merge;
* the score is taken only over holdout entities, whose records the model never
  trained on.

Anything less optimistic-biases the result. Restricting the record pool makes
unprocessed records look like misses; excluding other entities' records removes
the hardest negatives.

Usage::

    python scripts/run_holdout_eval.py --country India --model artifacts/matcher.txt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb

from src import config
from src.candidates import build_country_index, iter_country_rows
from src.inference import run_country
from src.metrics import macro_fbeta_report
from src.splits import load_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", default="India")
    parser.add_argument("--model", default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--chunk-rows", type=int, default=20000)
    parser.add_argument("--max-df-ratio", type=float, default=0.002)
    args = parser.parse_args()

    config.ensure_dirs()
    started = time.time()

    model_path = Path(args.model) if args.model else config.ARTIFACTS / "matcher.txt"
    booster = lgb.Booster(model_file=str(model_path))
    print(f"model: {model_path}")

    truth_all: dict[str, list[str]] = {}
    with open(config.TRAIN_FILES["ground_truth"], encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            ids = parts[1] if len(parts) > 1 else ""
            truth_all[parts[0]] = [x for x in ids.split(",") if x]

    splits = load_splits(config.ARTIFACTS / "splits.json")
    valid_entities = set(splits["holdout"]["valid"])

    index = build_country_index(
        config.TRAIN_FILES["source1"], args.country, max_df_ratio=args.max_df_ratio
    )
    print(f"index: {index.size:,} Source 1 entities  [{time.time() - started:.0f}s]", flush=True)

    country_entities = set(index.entity_ids)
    holdout_entities = country_entities & valid_entities
    truth = {e: truth_all.get(e, []) for e in holdout_entities}
    n_singletons = sum(1 for v in truth.values() if not v)
    print(
        f"holdout entities in {args.country}: {len(truth):,} "
        f"({n_singletons:,} singletons, {n_singletons / max(len(truth), 1):.2%})"
    )

    def records():
        for source in ("source2", "source3"):
            is_s3 = source == "source3"
            for record_id, name, address in iter_country_rows(
                config.TRAIN_FILES[source], args.country
            ):
                yield record_id, name, address, is_s3

    def predict(matrix: np.ndarray) -> np.ndarray:
        return booster.predict(matrix, num_iteration=booster.best_iteration or None)

    def progress(n_records: int, n_pairs: int) -> None:
        if n_records % (args.chunk_rows * 10) == 0:
            elapsed = time.time() - started
            print(
                f"    {n_records:,} records, {n_pairs:,} pairs, "
                f"{n_records / max(elapsed, 1e-9):,.0f} rec/s  [{elapsed:.0f}s]",
                flush=True,
            )

    decisions = run_country(
        index,
        records(),
        predict,
        k=args.k,
        chunk_rows=args.chunk_rows,
        progress=progress,
    )
    print(
        f"\nscored records kept above floor: {len(decisions):,}  "
        f"[{time.time() - started:.0f}s]",
        flush=True,
    )

    # Which records truly belong to a holdout entity, and which belong to nobody.
    owner_of: dict[str, str] = {}
    for entity_id, ids in truth_all.items():
        for record_id in ids:
            owner_of[record_id] = entity_id

    print(f"\n{'thr':>6}{'margin':>8}{'F0.5':>9}{'single':>9}{'matched':>9}"
          f"{'micro-P':>9}{'micro-R':>9}{'assigned':>10}{'FP-single':>11}{'distract':>10}")

    best_row = None
    for margin in (0.0, 0.05, 0.10, 0.20):
        for threshold in np.arange(0.10, 0.96, 0.05):
            predictions: dict[str, list[str]] = {e: [] for e in truth}
            n_assigned = 0
            n_distractor_assigned = 0
            for record_id, decision in decisions.items():
                if decision.probability < threshold or decision.margin < margin:
                    continue
                bucket = predictions.get(decision.entity_id)
                if bucket is None:
                    continue  # assigned to a train-split entity; not scored here
                bucket.append(record_id)
                n_assigned += 1
                if record_id not in owner_of:
                    n_distractor_assigned += 1

            report = macro_fbeta_report(predictions, truth)
            # False merges on singletons: entities with no true match that we
            # nonetheless assigned something. Each costs a full 1.0.
            fp_singletons = sum(
                1 for e, ids in truth.items() if not ids and predictions[e]
            )
            row = {
                "threshold": round(float(threshold), 3),
                "margin": margin,
                "report": report,
                "n_assigned": n_assigned,
                "fp_singletons": fp_singletons,
                "n_distractor_assigned": n_distractor_assigned,
            }
            if best_row is None or report["macro_f0.5"] > best_row["report"]["macro_f0.5"]:
                best_row = row
            if threshold in (0.10, 0.25, 0.40, 0.55, 0.70, 0.85):
                print(
                    f"{threshold:>6.2f}{margin:>8.2f}{report['macro_f0.5']:>9.4f}"
                    f"{report['macro_f0.5_singletons']:>9.4f}{report['macro_f0.5_matched']:>9.4f}"
                    f"{report['micro_precision']:>9.4f}{report['micro_recall']:>9.4f}"
                    f"{n_assigned:>10,}{fp_singletons:>11,}{n_distractor_assigned:>10,}"
                )

    report = best_row["report"]
    print(
        f"\nBEST  threshold={best_row['threshold']}  margin={best_row['margin']}\n"
        f"  macro F_0.5            : {report['macro_f0.5']:.4f}\n"
        f"  on singletons          : {report['macro_f0.5_singletons']:.4f} "
        f"({int(report['n_singletons']):,} entities)\n"
        f"  on matched entities    : {report['macro_f0.5_matched']:.4f} "
        f"({int(report['n_matched']):,} entities)\n"
        f"  micro precision        : {report['micro_precision']:.4f}\n"
        f"  micro recall           : {report['micro_recall']:.4f}\n"
        f"  false merges on singles: {best_row['fp_singletons']:,}\n"
        f"  distractors assigned   : {best_row['n_distractor_assigned']:,}\n"
        f"  records assigned       : {best_row['n_assigned']:,}"
    )
    print(f"total seconds: {time.time() - started:.0f}")


if __name__ == "__main__":
    main()
