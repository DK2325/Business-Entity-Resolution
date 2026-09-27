"""Train one collective model across several countries, evaluated per country.

The India-only collective model gained on India but lost on US, where it flipped
singletons to matched 3:1. The hypothesis this script tests is that the loss came
from *never having seen US data* rather than from something irreducible: US
groups are fuller (blocking recall 0.9688 against 0.9334) and US sibling
similarity is higher (no cross-script transliteration), so an India-trained model
reads a typical US group as unusually complete and accepts too much.

Training on both countries at once should fix that if the hypothesis is right.
The decisive evidence is per-country: a single combined number could improve
while US still degrades, so each country is scored separately, and each on both
halves of its holdout split.

One subtlety the single-country script does not have to handle: ``entity_row``
indices are relative to each country's own index, so row 5 means a different
entity in India than in US. Entities are keyed by ``(country, row)`` throughout.

Usage::

    python scripts/train_collective_multi.py --countries India US \
        --shards artifacts/shards_train2 --floor 0.30
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
    tp, pred, tru = (int(true_positive[mask].sum()), int(predicted[mask].sum()),
                     int(truth[mask].sum()))
    return {
        "macro": float(f[mask].mean()) if mask.any() else 0.0,
        "singleton": float(f[singles].mean()) if singles.any() else float("nan"),
        "precision": tp / pred if pred else 0.0,
        "recall": tp / tru if tru else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--countries", nargs="+", default=["India", "US"])
    parser.add_argument("--shards", required=True)
    parser.add_argument("--floor", type=float, default=0.30)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    started = time.time()
    shard_dir = Path(args.shards)

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

    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    meta_country: list[int] = []
    meta_key: list[int] = []   # global entity index
    key_of: dict[tuple[str, int], int] = {}
    entity_of_key: list[str] = []
    country_of_key: list[int] = []

    for ci, country in enumerate(args.countries):
        paths = sorted(shard_dir.glob(f"shard_{country}_*.npz"))
        if not paths:
            raise SystemExit(f"no shards for {country} in {shard_dir}")
        record_ids: list[str] = []
        ent_rows, probs, margins = [], [], []
        for path in paths:
            d = np.load(path, allow_pickle=False)
            record_ids.extend(d["record_ids"][d["record_index"]].tolist())
            ent_rows.append(d["entity_row"])
            probs.append(d["probability"])
            margins.append(d["margin"])
        entity_row = np.concatenate(ent_rows)
        prob = np.concatenate(probs)
        margin = np.concatenate(margins)
        rec_arr = np.asarray(record_ids)
        del ent_rows, probs, margins, record_ids

        keep = prob >= args.floor
        entity_row, prob, margin, rec_arr = (
            entity_row[keep], prob[keep], margin[keep], rec_arr[keep]
        )
        print(f"{country}: {rec_arr.shape[0]:,} decisions above floor  "
              f"[{time.time() - started:.0f}s]", flush=True)

        rows = list(iter_country_rows(config.TRAIN_FILES["source1"], country))
        entity_ids = [r[0] for r in rows]

        needed = set(rec_arr.tolist())
        rec_text: dict[str, tuple[str, str]] = {}
        for src in ("source2", "source3"):
            for rid, name, address in iter_country_rows(config.TRAIN_FILES[src], country):
                if rid in needed:
                    rec_text[rid] = (name, address)
        del needed

        by_entity: dict[int, list[int]] = collections.defaultdict(list)
        for i in range(rec_arr.shape[0]):
            by_entity[int(entity_row[i])].append(i)
        print(f"  groups {len(by_entity):,}, text {len(rec_text):,}  "
              f"[{time.time() - started:.0f}s]", flush=True)

        done = 0
        for row, members in by_entity.items():
            text = rows[row]
            group = Group(entity_parsed=parse_record(text[1], text[2]))
            for i in members:
                rid = rec_arr[i]
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
            key = key_of.get((country, row))
            if key is None:
                key = len(entity_of_key)
                key_of[(country, row)] = key
                entity_of_key.append(entity_ids[row])
                country_of_key.append(ci)
            feature_blocks.append(group_features(group))
            label_blocks.append(
                np.fromiter((owner_of.get(r) == entity_ids[row] for r in group.record_ids),
                            dtype=np.int8, count=len(group.record_ids))
            )
            meta_key.extend([key] * len(group.record_ids))
            meta_country.extend([ci] * len(group.record_ids))
            done += 1
            if done % 100_000 == 0:
                print(f"    {done:,} groups  [{time.time() - started:.0f}s]", flush=True)
        del rec_text, by_entity, rec_arr, entity_row, prob, margin

    X = np.concatenate(feature_blocks)
    y = np.concatenate(label_blocks)
    key_arr = np.asarray(meta_key, dtype=np.int32)
    country_arr = np.asarray(meta_country, dtype=np.int8)
    del feature_blocks, label_blocks, meta_key, meta_country
    print(f"\npairs {X.shape[0]:,} x {X.shape[1]}, positives {int(y.sum()):,} ({y.mean():.2%})"
          f"  [{time.time() - started:.0f}s]", flush=True)

    n_keys = len(entity_of_key)
    truth_count = np.zeros(n_keys, dtype=np.int32)
    in_holdout = np.zeros(n_keys, dtype=bool)
    half = np.zeros(n_keys, dtype=np.int8)
    for k, eid in enumerate(entity_of_key):
        if eid in valid:
            in_holdout[k] = True
            truth_count[k] = len(truth_all.get(eid, []))
            half[k] = half_of(eid)
    is_holdout_pair = in_holdout[key_arr]

    model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.06, num_leaves=63, min_child_samples=100,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.85, reg_lambda=2.0,
        random_state=config.SEED, n_jobs=-1, verbose=-1,
    )
    model.fit(X[~is_holdout_pair], y[~is_holdout_pair],
              eval_set=[(X[is_holdout_pair], y[is_holdout_pair])],
              eval_metric="average_precision", feature_name=FEATURE_NAMES,
              callbacks=[lgb.log_evaluation(200)])
    stage2 = model.predict_proba(X)[:, 1]
    stage1 = X[:, FEATURE_NAMES.index("prob")]
    print(f"trained  [{time.time() - started:.0f}s]")

    correct = np.zeros(X.shape[0], dtype=bool)
    # a pair is correct when the record's true owner is this key's entity
    # (recomputed from labels, which were built exactly that way)
    correct[:] = y.astype(bool)

    print("\ntop features:")
    for name, gain in sorted(zip(FEATURE_NAMES, model.feature_importances_),
                             key=lambda kv: -kv[1])[:10]:
        print(f"  {name:<28}{gain}")

    def score(keep_pairs, ci, h) -> dict:
        mask = in_holdout & (half == h) & (np.asarray(country_of_key) == ci)
        sel = keep_pairs & is_holdout_pair & (country_arr == ci)
        rows_kept = key_arr[sel]
        pred = np.bincount(rows_kept, minlength=n_keys).astype(np.int32)
        tp = np.bincount(rows_kept[correct[sel]], minlength=n_keys).astype(np.int32)
        return macro(pred, tp, truth_count, mask)

    print(f"\n{'country':<8}{'rule':<22}{'F0.5 A':>9}{'F0.5 B':>9}{'sgl A':>8}{'prec A':>8}{'rec A':>8}")
    results = {}
    for ci, country in enumerate(args.countries):
        b_a, b_b = score(stage1 >= 0.70, ci, 0), score(stage1 >= 0.70, ci, 1)
        results[(country, "stage1")] = (b_a, b_b)
        print(f"{country:<8}{'stage-1 @0.70':<22}{b_a['macro']:>9.4f}{b_b['macro']:>9.4f}"
              f"{b_a['singleton']:>8.4f}{b_a['precision']:>8.4f}{b_a['recall']:>8.4f}")
        best = None
        for t in np.arange(0.30, 0.91, 0.05):
            a, bb = score(stage2 >= t, ci, 0), score(stage2 >= t, ci, 1)
            if best is None or min(a["macro"], bb["macro"]) > min(best[1]["macro"], best[2]["macro"]):
                best = (t, a, bb)
            if int(round(t * 100)) % 10 == 0:
                print(f"{'':<8}{'stage-2 @' + format(t, '.2f'):<22}{a['macro']:>9.4f}{bb['macro']:>9.4f}"
                      f"{a['singleton']:>8.4f}{a['precision']:>8.4f}{a['recall']:>8.4f}")
        t, a, bb = best
        results[(country, "stage2")] = (t, a, bb)
        gain_a = a["macro"] - b_a["macro"]
        gain_b = bb["macro"] - b_b["macro"]
        verdict = "HELPS both halves" if gain_a > 0 and gain_b > 0 else "DOES NOT help both halves"
        print(f"{'':<8}BEST @{t:.2f}: A {a['macro']:.4f} ({gain_a:+.4f})  "
              f"B {bb['macro']:.4f} ({gain_b:+.4f})  -> {verdict}\n")

    out = Path(args.out) if args.out else config.ARTIFACTS / "collective_multi.txt"
    model.booster_.save_model(str(out))
    print(f"saved model to {out}\ntotal seconds: {time.time() - started:.0f}")


if __name__ == "__main__":
    main()
