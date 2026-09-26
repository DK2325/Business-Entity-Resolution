"""End-to-end matcher on a self-contained dev subset.

Purpose is a trustworthy macro F_0.5 in minutes rather than hours, so the
decision rule can be tuned before committing to a full run.

Why a subset of *entities* rather than a sample of pairs: the metric is computed
per Source 1 entity over the records assigned to it, so scoring an entity
honestly means processing every record that could have been assigned to it.
Sampling records breaks that -- unprocessed records silently become misses and
recall reads low. Restricting the Source 1 index to N entities instead keeps the
evaluation exact: every record in the dev pool is scored against the whole index
that produced it.

The dev pool is the records owned by those entities plus a proportional share of
unowned distractors, so the roughly 26% distractor rate of the real data is
preserved -- without them the model never learns to withhold a match, which is
the single most valuable thing it can learn under this metric.

Usage::

    python scripts/dev_pipeline.py --country India --n-entities 100000 --k 5
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb

from src import config
from src.candidates import (
    CountryIndex,
    iter_country_rows,
    retrieve,
    shortlist_context,
)
from src.blocking import TokenVocabulary, build_matrix
from src.decide import tune_threshold
from src.features import FEATURE_NAMES, PairFeaturizer
from src.keys import VIEWS
from src.metrics import macro_fbeta_report
from src.splits import load_splits


def build_index_from_rows(
    country: str,
    rows: list[tuple[str, str, str]],
    view_names: tuple[str, ...],
    max_df_ratio: float,
) -> CountryIndex:
    """Index an explicit list of Source 1 records (the dev subset)."""
    index = CountryIndex(
        country=country,
        entity_ids=[r[0] for r in rows],
        names=[r[1] for r in rows],
        addresses=[r[2] for r in rows],
        view_names=view_names,
    )
    for view_name in view_names:
        view = VIEWS[view_name]
        vocab = TokenVocabulary()
        for _eid, name, address in rows:
            vocab.add_document(view(name, address))
        idf = vocab.finalise(len(rows), max_df_ratio, min_df=1)
        matrix = build_matrix((view(n, a) for _e, n, a in rows), vocab, idf)
        index.vocabs.append(vocab)
        index.idfs.append(idf)
        index.matrices.append(matrix)
        index.transposes.append(matrix.T.tocsr())
        print(f"  [{view_name}] {len(vocab):,} tokens, {matrix.nnz:,} nnz", flush=True)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", default="India")
    parser.add_argument("--n-entities", type=int, default=100_000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--chunk-rows", type=int, default=20000)
    parser.add_argument("--max-df-ratio", type=float, default=0.002)
    parser.add_argument("--valid-fraction", type=float, default=0.30)
    args = parser.parse_args()

    config.ensure_dirs()
    started = time.time()
    rng = random.Random(config.SEED)

    # ---- ground truth and the dev entity subset ----------------------------
    truth_all: dict[str, list[str]] = {}
    with open(config.TRAIN_FILES["ground_truth"], encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            ids = parts[1] if len(parts) > 1 else ""
            truth_all[parts[0]] = [x for x in ids.split(",") if x]

    country_rows = list(iter_country_rows(config.TRAIN_FILES["source1"], args.country))
    print(f"{args.country} Source 1 records: {len(country_rows):,}")
    rng.shuffle(country_rows)
    dev_rows = country_rows[: args.n_entities]
    dev_entities = {r[0] for r in dev_rows}
    truth = {e: truth_all.get(e, []) for e in dev_entities}
    n_singleton = sum(1 for v in truth.values() if not v)
    print(
        f"dev entities: {len(dev_entities):,} "
        f"({n_singleton:,} singletons, {n_singleton / len(dev_entities):.2%})"
    )

    owned = {r for ids in truth.values() for r in ids}
    print(f"records owned by dev entities: {len(owned):,}")

    # ---- dev record pool: owned records + proportional distractors ---------
    # The real data is ~26% unowned; keeping that ratio is what teaches the
    # model to say "no match", which the metric rewards heavily.
    all_owned_anywhere: set[str] = set()
    for ids in truth_all.values():
        all_owned_anywhere.update(ids)

    distractor_target = int(len(owned) * 0.26 / 0.74)
    pool: list[tuple[str, str, str, bool]] = []
    n_distractor = 0
    for source in ("source2", "source3"):
        is_s3 = source == "source3"
        for record_id, name, address in iter_country_rows(
            config.TRAIN_FILES[source], args.country
        ):
            if record_id in owned:
                pool.append((record_id, name, address, is_s3))
            elif record_id not in all_owned_anywhere and n_distractor < distractor_target:
                pool.append((record_id, name, address, is_s3))
                n_distractor += 1
    rng.shuffle(pool)
    print(
        f"dev record pool: {len(pool):,} "
        f"({len(owned):,} owned, {n_distractor:,} distractors)  [{time.time() - started:.0f}s]"
    )

    # ---- index and retrieval ------------------------------------------------
    index = build_index_from_rows(
        args.country, dev_rows, ("composite", "ngram"), args.max_df_ratio
    )
    row_of_entity = {e: i for i, e in enumerate(index.entity_ids)}
    owner = {r: e for e, ids in truth.items() for r in ids}
    featurizer = PairFeaturizer(index.token_idf())
    print(f"index + idf ready  [{time.time() - started:.0f}s]", flush=True)

    # Entities are split so validation measures generalisation; the record pool
    # follows its owner, distractors are split at the same rate.
    valid_entities = {e for e in dev_entities if rng.random() < args.valid_fraction}

    # Features are converted to a float32 block per chunk rather than piling up
    # as Python lists: at a few million pairs the list-of-lists representation
    # costs several times the array it becomes.
    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    valid_blocks: list[np.ndarray] = []
    rec_ids: list[str] = []
    ent_ids: list[str] = []
    owner_retrieved = owner_present = 0
    n_pairs_total = 0

    for start in range(0, len(pool), args.chunk_rows):
        batch = pool[start : start + args.chunk_rows]
        shortlists = retrieve(index, batch, k=args.k, chunk_rows=args.chunk_rows)
        feats: list[list[float]] = []
        labels: list[int] = []
        valid_flags: list[int] = []
        for shortlist in shortlists:
            true_owner = owner.get(shortlist.record_id)
            owner_row = row_of_entity.get(true_owner) if true_owner else None
            if owner_row is not None:
                owner_present += 1
                if owner_row in shortlist.ranks:
                    owner_retrieved += 1
            if true_owner is not None:
                is_valid = 1 if true_owner in valid_entities else 0
            else:
                is_valid = 1 if rng.random() < args.valid_fraction else 0

            right = featurizer.parse(shortlist.record_id, shortlist.name, shortlist.address)
            for index_row in shortlist.rows():
                left = featurizer.parse(
                    index.entity_ids[index_row],
                    index.names[index_row],
                    index.addresses[index_row],
                )
                feats.append(
                    featurizer.features(left, right, shortlist_context(shortlist, index_row))
                )
                labels.append(1 if index_row == owner_row else 0)
                valid_flags.append(is_valid)
                rec_ids.append(shortlist.record_id)
                ent_ids.append(index.entity_ids[index_row])

        if feats:
            feature_blocks.append(np.asarray(feats, dtype=np.float32))
            label_blocks.append(np.asarray(labels, dtype=np.int8))
            valid_blocks.append(np.asarray(valid_flags, dtype=bool))
            n_pairs_total += len(feats)
        # Bound the per-record parse cache; entities recur across chunks but the
        # cache would otherwise grow to the whole dev pool.
        if len(featurizer._cache) > 600_000:
            featurizer.clear()
        if (start // args.chunk_rows) % 5 == 0:
            print(
                f"    {start + len(batch):,}/{len(pool):,} records, {n_pairs_total:,} pairs"
                f"  [{time.time() - started:.0f}s]",
                flush=True,
            )

    X = np.concatenate(feature_blocks) if feature_blocks else np.zeros((0, 33), np.float32)
    y = np.concatenate(label_blocks) if label_blocks else np.zeros(0, np.int8)
    v = np.concatenate(valid_blocks) if valid_blocks else np.zeros(0, bool)
    rec_ids_arr = np.asarray(rec_ids)
    ent_ids_arr = np.asarray(ent_ids)
    del feature_blocks, label_blocks, valid_blocks, rec_ids, ent_ids

    print(
        f"\npairs: {X.shape[0]:,} x {X.shape[1]} features, "
        f"{int(y.sum()):,} positive ({y.mean():.2%})\n"
        f"blocking recall on dev pool: {owner_retrieved / max(owner_present, 1):.4f}\n"
        f"[{time.time() - started:.0f}s]",
        flush=True,
    )

    # ---- model --------------------------------------------------------------
    train_mask = ~v
    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.08,
        num_leaves=96,
        min_child_samples=50,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=config.SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X[train_mask],
        y[train_mask],
        eval_set=[(X[v], y[v])],
        eval_metric="average_precision",
        feature_name=FEATURE_NAMES,
        callbacks=[lgb.log_evaluation(100)],
    )
    probabilities = model.predict_proba(X[v])[:, 1]
    print(f"model trained  [{time.time() - started:.0f}s]", flush=True)

    importances = sorted(
        zip(FEATURE_NAMES, model.feature_importances_), key=lambda kv: -kv[1]
    )
    print("\ntop features by split gain:")
    for name, gain in importances[:12]:
        print(f"  {name:<28} {gain}")

    # ---- decision rule ------------------------------------------------------
    valid_truth = {e: truth[e] for e in dev_entities if e in valid_entities}
    print(f"\nvalidation entities: {len(valid_truth):,}")

    results = tune_threshold(
        rec_ids_arr[v].tolist(),
        ent_ids_arr[v].tolist(),
        probabilities.tolist(),
        valid_truth,
        thresholds=np.arange(0.10, 0.96, 0.05),
        margins=(0.0, 0.05, 0.10, 0.20),
    )

    print("\ntop decision rules by macro F_0.5:")
    header = f"{'thr':>6}{'margin':>8}{'F0.5':>9}{'single':>9}{'matched':>9}{'micro-P':>9}{'micro-R':>9}{'assigned':>10}"
    print(header)
    for row in results[:10]:
        print(
            f"{row['threshold']:>6.2f}{row['margin']:>8.2f}{row['macro_f0.5']:>9.4f}"
            f"{row['macro_f0.5_singletons']:>9.4f}{row['macro_f0.5_matched']:>9.4f}"
            f"{row['micro_precision']:>9.4f}{row['micro_recall']:>9.4f}{row['n_assigned']:>10,}"
        )

    best = results[0]
    print(
        f"\nBEST: threshold={best['threshold']} margin={best['margin']} "
        f"macro F_0.5={best['macro_f0.5']:.4f}"
    )
    print(f"total seconds: {time.time() - started:.0f}")


if __name__ == "__main__":
    main()
