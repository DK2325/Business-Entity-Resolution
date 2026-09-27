"""Train and evaluate the second-stage collective model on saved decisions.

Runs entirely on shards already on disk: no retrieval, no stage-one inference.

Training entities come from the *train* side of the grouped holdout; evaluation
uses holdout entities only, split 50/50 into halves A and B by a hash of the
entity id. A change is only worth keeping if it improves both halves, since with
~159k entities it is easy to fit noise.

Usage::

    python scripts/train_collective.py --country India --floor 0.30
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb

from src import config
from src.candidates import iter_country_rows
from src.collective import FEATURE_NAMES, Group, group_features, parse_record
from src.splits import load_splits

BETA_SQ = 0.25


def half_of(entity_id: str) -> int:
    return hashlib.blake2b(entity_id.encode("utf-8"), digest_size=8).digest()[0] & 1


def macro(predicted, true_positive, truth, mask) -> dict:
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(predicted > 0, true_positive / np.maximum(predicted, 1), 0.0)
        r = np.where(truth > 0, true_positive / np.maximum(truth, 1), 0.0)
        den = BETA_SQ * p + r
        f = np.where(den > 0, 1.25 * p * r / np.maximum(den, 1e-12), 0.0)
    f = np.where((truth == 0) & (predicted == 0), 1.0, f)
    f = np.where((truth == 0) & (predicted > 0), 0.0, f)
    singles = mask & (truth == 0)
    matched = mask & (truth > 0)
    tp, pred, tru = (
        int(true_positive[mask].sum()),
        int(predicted[mask].sum()),
        int(truth[mask].sum()),
    )
    return {
        "macro": float(f[mask].mean()) if mask.any() else 0.0,
        "singleton": float(f[singles].mean()) if singles.any() else float("nan"),
        "matched": float(f[matched].mean()) if matched.any() else float("nan"),
        "precision": tp / pred if pred else 0.0,
        "recall": tp / tru if tru else 0.0,
        "assigned": pred,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", default="India")
    parser.add_argument("--shards", default=None)
    parser.add_argument("--floor", type=float, default=0.30,
                        help="minimum stage-1 probability to enter a group")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    started = time.time()
    shard_dir = Path(args.shards) if args.shards else config.ARTIFACTS / "shards_train"

    record_ids: list[str] = []
    ent_rows, probs, margins = [], [], []
    for path in sorted(shard_dir.glob(f"shard_{args.country}_*.npz")):
        d = np.load(path, allow_pickle=False)
        record_ids.extend(d["record_ids"][d["record_index"]].tolist())
        ent_rows.append(d["entity_row"])
        probs.append(d["probability"])
        margins.append(d["margin"])
    entity_row = np.concatenate(ent_rows)
    prob = np.concatenate(probs)
    margin = np.concatenate(margins)
    record_ids_arr = np.asarray(record_ids)
    print(f"decisions: {record_ids_arr.shape[0]:,}   [{time.time() - started:.0f}s]", flush=True)

    keep = prob >= args.floor
    entity_row, prob, margin = entity_row[keep], prob[keep], margin[keep]
    record_ids_arr = record_ids_arr[keep]
    print(f"above floor {args.floor}: {record_ids_arr.shape[0]:,}", flush=True)

    entity_ids = [e for e, _n, _a in iter_country_rows(config.TRAIN_FILES["source1"], args.country)]
    n_entities = len(entity_ids)

    truth_all: dict[str, list[str]] = {}
    with open(config.TRAIN_FILES["ground_truth"], encoding="utf-8") as h:
        h.readline()
        for line in h:
            p = line.rstrip("\n").split("\t")
            ids = p[1] if len(p) > 1 else ""
            truth_all[p[0]] = [x for x in ids.split(",") if x]
    owner_of = {rec: e for e, ids in truth_all.items() for rec in ids}

    splits = load_splits(config.ARTIFACTS / "splits.json")
    valid = set(splits["holdout"]["valid"])
    in_holdout = np.zeros(n_entities, dtype=bool)
    truth_count = np.zeros(n_entities, dtype=np.int32)
    half = np.zeros(n_entities, dtype=np.int8)
    for row, e in enumerate(entity_ids):
        if e in valid:
            in_holdout[row] = True
            truth_count[row] = len(truth_all.get(e, []))
            half[row] = half_of(e)
    print(f"holdout entities: {int(in_holdout.sum()):,}", flush=True)

    # ---- load the text we need ------------------------------------------------
    needed_records = set(record_ids_arr.tolist())
    needed_entity_rows = set(entity_row.tolist())
    rec_text: dict[str, tuple[str, str]] = {}
    for src in ("source2", "source3"):
        for rid, name, address in iter_country_rows(config.TRAIN_FILES[src], args.country):
            if rid in needed_records:
                rec_text[rid] = (name, address)
    ent_text: dict[int, tuple[str, str]] = {}
    for row, (eid, name, address) in enumerate(
        iter_country_rows(config.TRAIN_FILES["source1"], args.country)
    ):
        if row in needed_entity_rows:
            ent_text[row] = (name, address)
    print(f"text loaded: {len(rec_text):,} records, {len(ent_text):,} entities"
          f"   [{time.time() - started:.0f}s]", flush=True)

    # ---- build groups ---------------------------------------------------------
    by_entity: dict[int, list[int]] = collections.defaultdict(list)
    for i in range(record_ids_arr.shape[0]):
        by_entity[int(entity_row[i])].append(i)
    print(f"groups: {len(by_entity):,}", flush=True)

    feature_blocks, label_blocks = [], []
    meta_entity, meta_record = [], []
    processed = 0
    for row, members in by_entity.items():
        text = ent_text.get(row)
        if text is None:
            continue
        group = Group(entity_parsed=parse_record(*text))
        for i in members:
            rid = record_ids_arr[i]
            rt = rec_text.get(rid)
            if rt is None:
                continue
            group.record_ids.append(rid)
            group.parsed.append(parse_record(*rt))
            group.probs.append(float(prob[i]))
            group.margins.append(float(margin[i]))
            group.is_s3.append(rid.startswith("S3-"))
        if not group.record_ids:
            continue
        X = group_features(group)
        y = np.fromiter(
            (owner_of.get(r) == entity_ids[row] for r in group.record_ids),
            dtype=np.int8, count=len(group.record_ids),
        )
        feature_blocks.append(X)
        label_blocks.append(y)
        meta_entity.extend([row] * len(group.record_ids))
        meta_record.extend(group.record_ids)
        processed += 1
        if processed % 50000 == 0:
            print(f"    {processed:,} groups  [{time.time() - started:.0f}s]", flush=True)

    X = np.concatenate(feature_blocks)
    y = np.concatenate(label_blocks)
    meta_entity_arr = np.asarray(meta_entity, dtype=np.int32)
    del feature_blocks, label_blocks
    print(f"\npairs: {X.shape[0]:,} x {X.shape[1]}, positives {int(y.sum()):,} "
          f"({y.mean():.2%})   [{time.time() - started:.0f}s]", flush=True)

    is_holdout_pair = in_holdout[meta_entity_arr]
    pair_half = half[meta_entity_arr]

    model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.06, num_leaves=63,
        min_child_samples=100, subsample=0.85, subsample_freq=1,
        colsample_bytree=0.85, reg_lambda=2.0,
        random_state=config.SEED, n_jobs=-1, verbose=-1,
    )
    train_mask = ~is_holdout_pair
    print(f"train pairs {int(train_mask.sum()):,}   holdout pairs {int(is_holdout_pair.sum()):,}")
    model.fit(
        X[train_mask], y[train_mask],
        eval_set=[(X[is_holdout_pair], y[is_holdout_pair])],
        eval_metric="average_precision", feature_name=FEATURE_NAMES,
        callbacks=[lgb.log_evaluation(200)],
    )
    stage2 = model.predict_proba(X)[:, 1]
    print(f"model trained  [{time.time() - started:.0f}s]")

    print("\ntop features:")
    for name, gain in sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda kv: -kv[1])[:12]:
        print(f"  {name:<28}{gain}")

    # ---- evaluate -------------------------------------------------------------
    correct = np.fromiter(
        (owner_of.get(meta_record[i]) == entity_ids[int(meta_entity_arr[i])]
         for i in range(len(meta_record))),
        dtype=bool, count=len(meta_record),
    )
    mask_a = in_holdout & (half == 0)
    mask_b = in_holdout & (half == 1)

    def score(keep_pairs: np.ndarray) -> tuple[dict, dict]:
        sel = keep_pairs & is_holdout_pair
        rows = meta_entity_arr[sel]
        pred = np.bincount(rows, minlength=n_entities).astype(np.int32)
        tp = np.bincount(rows[correct[sel]], minlength=n_entities).astype(np.int32)
        return macro(pred, tp, truth_count, mask_a), macro(pred, tp, truth_count, mask_b)

    # The stage-1 probability must come from the group-aligned feature matrix,
    # not from the original decision array: groups are built by iterating a dict,
    # so their row order differs from the order the decisions were loaded in.
    # Using the unaligned array compares one pair's probability against another
    # pair's label and produces a meaningless baseline.
    stage1_prob = X[:, FEATURE_NAMES.index("prob")]
    base_a, base_b = score(stage1_prob >= 0.70)
    print(f"\nstage-1 baseline (thr 0.70):  A {base_a['macro']:.4f}   B {base_b['macro']:.4f}")

    print(f"\n{'rule':<28}{'F0.5 A':>9}{'F0.5 B':>9}{'sgl A':>8}{'mat A':>8}{'prec A':>8}{'rec A':>8}")
    print(f"{'stage-1 thr 0.70':<28}{base_a['macro']:>9.4f}{base_b['macro']:>9.4f}"
          f"{base_a['singleton']:>8.4f}{base_a['matched']:>8.4f}"
          f"{base_a['precision']:>8.4f}{base_a['recall']:>8.4f}")

    best = None
    for t in np.arange(0.20, 0.91, 0.05):
        a, b = score(stage2 >= t)
        if best is None or min(a["macro"], b["macro"]) > min(best[1]["macro"], best[2]["macro"]):
            best = (t, a, b)
        if abs(t - round(t, 2)) < 1e-9 and int(round(t * 100)) % 10 == 0:
            print(f"{'stage-2 thr ' + format(t, '.2f'):<28}{a['macro']:>9.4f}{b['macro']:>9.4f}"
                  f"{a['singleton']:>8.4f}{a['matched']:>8.4f}{a['precision']:>8.4f}{a['recall']:>8.4f}")

    t, a, b = best
    print(f"\nBEST stage-2 threshold {t:.2f}:  A {a['macro']:.4f} ({a['macro'] - base_a['macro']:+.4f})"
          f"   B {b['macro']:.4f} ({b['macro'] - base_b['macro']:+.4f})")
    print(f"  singleton A {a['singleton']:.4f} (was {base_a['singleton']:.4f})")
    print(f"  matched   A {a['matched']:.4f} (was {base_a['matched']:.4f})")
    print(f"  precision A {a['precision']:.4f} (was {base_a['precision']:.4f})")
    print(f"  recall    A {a['recall']:.4f} (was {base_a['recall']:.4f})")
    improves_both = a["macro"] > base_a["macro"] and b["macro"] > base_b["macro"]
    print(f"\nimproves BOTH halves: {'YES' if improves_both else 'NO'}")

    out = Path(args.out) if args.out else config.ARTIFACTS / "collective.txt"
    model.booster_.save_model(str(out))
    print(f"saved model to {out}\ntotal seconds: {time.time() - started:.0f}")


if __name__ == "__main__":
    main()
