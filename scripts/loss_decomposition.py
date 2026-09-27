"""Where the macro-F_0.5 points are actually lost, on the India holdout.

Each bucket is priced by an oracle: fix that one failure mode perfectly, leave
everything else alone, and measure how far the score moves. The deltas say where
effort is worth spending, and they are additive only loosely -- fixing blocking
also changes what the threshold sees -- but they rank the causes honestly.

Buckets:

``a`` true link never retrieved (no decision row for the record at all)
``b`` record retrieved but its top-1 entity is the wrong one
``c`` top-1 is correct but scored below the threshold
``d`` an accepted record belongs to a different entity, or to nobody
``e`` a singleton was given a match

Usage::

    python scripts/loss_decomposition.py --threshold 0.70
"""

from __future__ import annotations

import argparse
import collections
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.candidates import iter_country_rows
from src.splits import load_splits

BETA_SQ = 0.25


def macro(predicted: np.ndarray, true_positive: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(predicted > 0, true_positive / np.maximum(predicted, 1), 0.0)
        r = np.where(truth > 0, true_positive / np.maximum(truth, 1), 0.0)
        den = BETA_SQ * p + r
        f = np.where(den > 0, 1.25 * p * r / np.maximum(den, 1e-12), 0.0)
    f = np.where((truth == 0) & (predicted == 0), 1.0, f)
    f = np.where((truth == 0) & (predicted > 0), 0.0, f)
    return float(f[mask].mean()) if mask.any() else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", default="India")
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--shards", default=None)
    parser.add_argument("--examples", type=int, default=15)
    args = parser.parse_args()

    shard_dir = Path(args.shards) if args.shards else config.ARTIFACTS / "shards_train"
    paths = sorted(shard_dir.glob(f"shard_{args.country}_*.npz"))
    record_ids: list[str] = []
    ent_rows, probs = [], []
    for path in paths:
        d = np.load(path, allow_pickle=False)
        record_ids.extend(d["record_ids"][d["record_index"]].tolist())
        ent_rows.append(d["entity_row"])
        probs.append(d["probability"])
    entity_row = np.concatenate(ent_rows)
    prob = np.concatenate(probs)
    record_ids_arr = np.asarray(record_ids)
    print(f"decisions: {record_ids_arr.shape[0]:,}")

    entity_ids = [e for e, _n, _a in iter_country_rows(config.TRAIN_FILES["source1"], args.country)]
    n_entities = len(entity_ids)
    entity_index = {e: i for i, e in enumerate(entity_ids)}

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
    in_holdout = np.zeros(n_entities, dtype=bool)
    truth_count = np.zeros(n_entities, dtype=np.int32)
    for row, entity in enumerate(entity_ids):
        if entity in valid:
            in_holdout[row] = True
            truth_count[row] = len(truth_all.get(entity, []))
    print(f"holdout entities: {int(in_holdout.sum()):,}")

    # decision lookup by record
    decision_of: dict[str, tuple[int, float]] = {}
    for i in range(record_ids_arr.shape[0]):
        decision_of[record_ids_arr[i]] = (int(entity_row[i]), float(prob[i]))

    # classify every true link of every holdout entity
    holdout_entities = [e for e in entity_ids if e in valid]
    bucket = collections.Counter()
    examples: dict[str, list[tuple[str, str, float, str]]] = collections.defaultdict(list)
    rng = random.Random(config.SEED)

    for entity in holdout_entities:
        row = entity_index[entity]
        for record in truth_all.get(entity, []):
            found = decision_of.get(record)
            if found is None:
                bucket["a_not_retrieved"] += 1
                if len(examples["a"]) < args.examples * 4:
                    examples["a"].append((entity, record, 0.0, ""))
            elif found[0] != row:
                bucket["b_wrong_top1"] += 1
                if len(examples["b"]) < args.examples * 4:
                    examples["b"].append((entity, record, found[1], entity_ids[found[0]]))
            elif found[1] < args.threshold:
                bucket["c_below_threshold"] += 1
                if len(examples["c"]) < args.examples * 4:
                    examples["c"].append((entity, record, found[1], ""))
            else:
                bucket["ok"] += 1

    total_links = sum(bucket.values())
    print(f"\n=== true links of holdout entities: {total_links:,} ===")
    for key in ("ok", "a_not_retrieved", "b_wrong_top1", "c_below_threshold"):
        print(f"  {key:<20}{bucket[key]:>10,}{bucket[key] / total_links:>9.2%}")

    # accepted records that are wrong, split into distractor vs wrong-entity
    accepted = prob >= args.threshold
    acc_rows = entity_row[accepted]
    acc_records = record_ids_arr[accepted]
    holdout_acc = in_holdout[acc_rows]
    wrong_entity = 0
    distractor = 0
    ex_d: list[tuple[str, str, float]] = []
    for row, record, keep in zip(acc_rows.tolist(), acc_records.tolist(), holdout_acc.tolist()):
        if not keep:
            continue
        owner = owner_of.get(record)
        if owner is None:
            distractor += 1
            if len(ex_d) < args.examples * 4:
                ex_d.append((entity_ids[row], record, 0.0))
        elif owner != entity_ids[row]:
            wrong_entity += 1
            if len(ex_d) < args.examples * 8:
                ex_d.append((entity_ids[row], record, -1.0))
    print(f"\n  d_wrong_or_distractor accepted: {wrong_entity + distractor:,} "
          f"(wrong entity {wrong_entity:,}, distractor {distractor:,})")

    # ---- oracle pricing -----------------------------------------------------
    def counts(keep_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        is_correct = np.fromiter(
            (owner_of.get(record_ids_arr[i]) == entity_ids[int(entity_row[i])]
             for i in np.nonzero(keep_mask)[0]),
            dtype=bool, count=int(keep_mask.sum()),
        )
        rows = entity_row[keep_mask]
        predicted = np.bincount(rows, minlength=n_entities).astype(np.int32)
        true_positive = np.bincount(rows[is_correct], minlength=n_entities).astype(np.int32)
        return predicted, true_positive

    base_pred, base_tp = counts(accepted)
    base = macro(base_pred, base_tp, truth_count, in_holdout)
    print(f"\n=== baseline macro F_0.5 (threshold {args.threshold}) : {base:.4f} ===\n")

    singleton_mask = in_holdout & (truth_count == 0)

    rows = []

    # (e) no false merges on singletons
    pred_e = base_pred.copy(); tp_e = base_tp.copy()
    pred_e[singleton_mask] = 0; tp_e[singleton_mask] = 0
    rows.append(("e: singleton false merges fixed", macro(pred_e, tp_e, truth_count, in_holdout)))

    # (d) every accepted-but-wrong record removed
    pred_d = base_tp.copy()  # keeping only the correct ones
    rows.append(("d: wrong/distractor accepts removed", macro(pred_d, base_tp, truth_count, in_holdout)))

    # (c) accept correct top-1 regardless of threshold
    pred_c = base_pred.copy(); tp_c = base_tp.copy()
    for entity in holdout_entities:
        row = entity_index[entity]
        gained = 0
        for record in truth_all.get(entity, []):
            found = decision_of.get(record)
            if found is not None and found[0] == row and found[1] < args.threshold:
                gained += 1
        if gained:
            pred_c[row] += gained; tp_c[row] += gained
    rows.append(("c: below-threshold correct top-1 accepted", macro(pred_c, tp_c, truth_count, in_holdout)))

    # (b) records whose top-1 was the wrong entity, given to the right one
    pred_b = base_pred.copy(); tp_b = base_tp.copy()
    for entity, record, _p, _other in examples["b"][:0]:
        pass
    gained_b = collections.Counter()
    for entity in holdout_entities:
        row = entity_index[entity]
        for record in truth_all.get(entity, []):
            found = decision_of.get(record)
            if found is not None and found[0] != row:
                gained_b[row] += 1
    for row, n in gained_b.items():
        pred_b[row] += n; tp_b[row] += n
    rows.append(("b: wrong top-1 corrected", macro(pred_b, tp_b, truth_count, in_holdout)))

    # (a) links never retrieved, added
    pred_a = base_pred.copy(); tp_a = base_tp.copy()
    gained_a = collections.Counter()
    for entity in holdout_entities:
        row = entity_index[entity]
        for record in truth_all.get(entity, []):
            if record not in decision_of:
                gained_a[row] += 1
    for row, n in gained_a.items():
        pred_a[row] += n; tp_a[row] += n
    rows.append(("a: blocking misses recovered", macro(pred_a, tp_a, truth_count, in_holdout)))

    # combined oracles
    pred_all = base_tp.copy(); tp_all = base_tp.copy()
    for row, n in gained_a.items():
        pred_all[row] += n; tp_all[row] += n
    for row, n in gained_b.items():
        pred_all[row] += n; tp_all[row] += n
    for entity in holdout_entities:
        row = entity_index[entity]
        gained = sum(
            1 for record in truth_all.get(entity, [])
            if (f := decision_of.get(record)) is not None and f[0] == row and f[1] < args.threshold
        )
        pred_all[row] += gained; tp_all[row] += gained
    pred_all[singleton_mask] = 0; tp_all[singleton_mask] = 0
    rows.append(("ALL oracles combined", macro(pred_all, tp_all, truth_count, in_holdout)))

    # threshold 0: accept every top-1
    pred_t0, tp_t0 = counts(prob > 0)
    rows.append(("threshold 0 (accept every top-1)", macro(pred_t0, tp_t0, truth_count, in_holdout)))

    print(f"{'oracle':<44}{'macro F0.5':>12}{'gain':>10}")
    for name, value in rows:
        print(f"{name:<44}{value:>12.4f}{value - base:>+10.4f}")

    # ---- examples -----------------------------------------------------------
    need = {"a": "true link never retrieved", "b": "record's top-1 was a different entity"}
    s1_text: dict[str, tuple[str, str]] = {}
    want_entities = {e for k in need for e, _r, _p, _o in examples[k]}
    want_records = {r for k in need for _e, r, _p, _o in examples[k]}
    with open(config.TRAIN_FILES["source1"], encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if parts[0] in want_entities:
                s1_text[parts[0]] = (parts[1], parts[2])
    rec_text: dict[str, tuple[str, str]] = {}
    for source in ("source2", "source3"):
        with open(config.TRAIN_FILES[source], encoding="utf-8") as handle:
            handle.readline()
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if parts[0] in want_records:
                    rec_text[parts[0]] = (parts[1], parts[2] if len(parts) > 2 else "")

    for key, title in need.items():
        picked = examples[key][: args.examples]
        print(f"\n=== examples: {title} ({bucket['a_not_retrieved' if key=='a' else 'b_wrong_top1']:,} links) ===")
        for entity, record, p, other in picked:
            left = s1_text.get(entity, ("?", "?"))
            right = rec_text.get(record, ("?", "?"))
            print(f"  S1 {left[0][:44]:<44} | {left[1][:58]}")
            print(f"  {record[:2]} {right[0][:44]:<44} | {right[1][:58]}"
                  + (f"   [top1={other} p={p:.2f}]" if other else ""))
            print()


if __name__ == "__main__":
    main()
