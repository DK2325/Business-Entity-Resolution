"""Apply the collective second-stage model to saved test decisions.

Re-scores existing shards and rewrites ``matching_results.tsv``. The candidate
set is untouched: this stage only decides which of the already-generated
candidates to keep, so ``candidate_pairs.tsv`` from the original run stays valid
and the matches remain a subset of it.

Processing is per country so peak memory tracks the largest country rather than
the whole test set.

Usage::

    python scripts/apply_collective.py --shards <dir> --model <collective.txt> \
        --threshold 0.60 --floor 0.30
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb

from src import config
from src.candidates import iter_country_rows
from src.collective import FEATURE_NAMES, Group, group_features, parse_record


def discover_countries(path: Path) -> list[str]:
    counts: Counter = Counter()
    with open(path, encoding="utf-8") as h:
        h.readline()
        for line in h:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4:
                counts[p[3]] += 1
    return [c for c, _n in counts.most_common()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--floor", type=float, default=0.30)
    parser.add_argument("--out", default=None)
    parser.add_argument("--stage2-countries", nargs="+", default=None,
                        help="apply stage 2 only to these countries; others keep their "
                             "stage-1 assignment at --stage1-threshold")
    parser.add_argument("--stage1-threshold", type=float, default=0.70,
                        help="threshold used for countries stage 2 is not applied to")
    args = parser.parse_args()

    started = time.time()
    shard_dir = Path(args.shards)
    model_path = Path(args.model) if args.model else config.ARTIFACTS / "collective.txt"
    booster = lgb.Booster(model_file=str(model_path))
    booster.params["num_threads"] = max(1, (__import__("os").cpu_count() or 2) - 1)
    print(f"model: {model_path}\nthreshold {args.threshold}  floor {args.floor}\n")

    out_path = Path(args.out) if args.out else config.OUTPUT_DIR / "matching_results.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    handle_out = open(out_path, "w", encoding="utf-8", newline="")
    handle_out.write("source1_entity_id\tmatched_entity_ids\n")

    countries = discover_countries(config.TEST_FILES["source1"])
    grand_entities = grand_assigned = grand_matched = 0

    for country in countries:
        country_started = time.time()
        entity_rows = list(iter_country_rows(config.TEST_FILES["source1"], country))
        entity_ids = [e for e, _n, _a in entity_rows]
        n_entities = len(entity_ids)

        paths = sorted(shard_dir.glob(f"shard_{country}_*.npz"))
        if not paths:
            raise SystemExit(f"no shards for {country}")
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
        record_ids_arr = np.asarray(record_ids)
        del ent_rows, probs, margins, record_ids

        keep = prob >= args.floor
        entity_row, prob, margin = entity_row[keep], prob[keep], margin[keep]
        record_ids_arr = record_ids_arr[keep]
        print(
            f"=== {country}: {n_entities:,} entities, "
            f"{record_ids_arr.shape[0]:,} decisions above floor  "
            f"[{time.time() - country_started:.0f}s]",
            flush=True,
        )

        needed = set(record_ids_arr.tolist())
        rec_text: dict[str, tuple[str, str]] = {}
        for src in ("source2", "source3"):
            for rid, name, address in iter_country_rows(config.TEST_FILES[src], country):
                if rid in needed:
                    rec_text[rid] = (name, address)
        del needed

        by_entity: dict[int, list[int]] = collections.defaultdict(list)
        for i in range(record_ids_arr.shape[0]):
            by_entity[int(entity_row[i])].append(i)
        print(f"  groups {len(by_entity):,}, text {len(rec_text):,}  "
              f"[{time.time() - country_started:.0f}s]", flush=True)

        assigned: dict[int, list[str]] = {}
        done = 0
        # Stage 2 is applied only where it has been validated to help. Elsewhere
        # the stage-1 decision stands: a model trained on one country's group
        # sizes and sibling-similarity levels does not necessarily transfer, and
        # under this metric a wrong transfer costs more than a missed gain.
        use_stage2 = args.stage2_countries is None or country in args.stage2_countries
        if not use_stage2:
            print(f"  {country}: stage 1 only (threshold {args.stage1_threshold})", flush=True)
            for i in range(record_ids_arr.shape[0]):
                if prob[i] >= args.stage1_threshold:
                    assigned.setdefault(int(entity_row[i]), []).append(record_ids_arr[i])
            for row, eid in enumerate(entity_ids):
                ids = assigned.get(row)
                handle_out.write(f"{eid}\t{','.join(ids) if ids else ''}\n")
            n_assigned = sum(len(v) for v in assigned.values())
            grand_entities += n_entities
            grand_assigned += n_assigned
            grand_matched += len(assigned)
            print(
                f"  {country}: {n_assigned:,} records assigned, {len(assigned):,} matched "
                f"({len(assigned) / max(n_entities, 1):.1%})  "
                f"[{time.time() - country_started:.0f}s]",
                flush=True,
            )
            del assigned, rec_text, by_entity, record_ids_arr, entity_row, prob, margin
            continue
        batch_rows: list[np.ndarray] = []
        batch_meta: list[tuple[int, str]] = []

        def flush_batch() -> None:
            nonlocal batch_rows, batch_meta
            if not batch_rows:
                return
            X = np.concatenate(batch_rows)
            scores = booster.predict(X, num_iteration=booster.best_iteration or None)
            for (row, rid), score in zip(batch_meta, scores):
                if score >= args.threshold:
                    assigned.setdefault(row, []).append(rid)
            batch_rows = []
            batch_meta = []

        for row, members in by_entity.items():
            text = entity_rows[row]
            group = Group(entity_parsed=parse_record(text[1], text[2]))
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
            batch_rows.append(group_features(group))
            batch_meta.extend((row, r) for r in group.record_ids)
            done += 1
            if len(batch_meta) >= 200_000:
                flush_batch()
            if done % 100_000 == 0:
                print(f"    {done:,} groups  [{time.time() - country_started:.0f}s]", flush=True)
        flush_batch()

        for row, eid in enumerate(entity_ids):
            ids = assigned.get(row)
            handle_out.write(f"{eid}\t{','.join(ids) if ids else ''}\n")

        n_assigned = sum(len(v) for v in assigned.values())
        grand_entities += n_entities
        grand_assigned += n_assigned
        grand_matched += len(assigned)
        print(
            f"  {country}: {n_assigned:,} records assigned, {len(assigned):,} matched "
            f"({len(assigned) / max(n_entities, 1):.1%}), "
            f"{n_entities - len(assigned):,} singleton  "
            f"[{time.time() - country_started:.0f}s]",
            flush=True,
        )
        del assigned, rec_text, by_entity, record_ids_arr, entity_row, prob, margin

    handle_out.close()
    print(
        f"\ntotals: {grand_entities:,} entities, {grand_assigned:,} assigned, "
        f"{grand_matched:,} matched, {grand_entities - grand_matched:,} singleton "
        f"({(grand_entities - grand_matched) / max(grand_entities, 1):.2%})"
    )
    print(f"mean ids per matched entity: {grand_assigned / max(grand_matched, 1):.2f}")
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"total seconds: {time.time() - started:.0f}")


if __name__ == "__main__":
    main()
