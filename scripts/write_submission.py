"""Turn sharded test decisions into the two submission files.

Reads the ``shard_*.npz`` files written by ``scripts/predict_parallel.py`` and
applies the decision rule. Because inference saved one row per record rather than
a finished answer, the threshold and margin can be changed here and the files
rewritten in a couple of minutes, without repeating the hours of scoring.

Both files are written with one row per test Source 1 entity, in the order the
entities appear in ``test_source1.tsv``. Entities with nothing assigned get an
empty field: that is the singleton prediction, and it is worth full credit when
correct.

Usage::

    python scripts/write_submission.py --threshold 0.75 --margin 0.0
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.candidates import iter_country_rows


def discover_countries(path: Path) -> list[str]:
    counts: Counter = Counter()
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                counts[parts[3]] += 1
    return [c for c, _n in counts.most_common()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", default=None)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--no-candidates", action="store_true")
    parser.add_argument("--validate", action="store_true", help="run the organisers' validator")
    parser.add_argument("--countries", nargs="+", default=None,
                        help="restrict to these countries (testing only; a real run needs all)")
    args = parser.parse_args()

    config.ensure_dirs()
    started = time.time()
    shard_dir = Path(args.shards) if args.shards else config.ARTIFACTS / "shards_test"

    countries = args.countries or discover_countries(config.TEST_FILES["source1"])
    print(f"countries: {countries}")
    print(f"threshold={args.threshold}  margin={args.margin}\n")

    matching_path = config.OUTPUT_DIR / "matching_results.tsv"
    candidate_path = config.OUTPUT_DIR / "candidate_pairs.tsv"
    matching_out = open(matching_path, "w", encoding="utf-8", newline="")
    matching_out.write("source1_entity_id\tmatched_entity_ids\n")
    candidate_out = None
    if not args.no_candidates:
        candidate_out = open(candidate_path, "w", encoding="utf-8", newline="")
        candidate_out.write("source1_entity_id\tcandidate_entity_ids\n")

    grand_entities = grand_assigned = grand_matched = grand_pairs = 0

    for country in countries:
        entity_ids = [
            e for e, _n, _a in iter_country_rows(config.TEST_FILES["source1"], country)
        ]
        n_entities = len(entity_ids)
        shard_paths = sorted(shard_dir.glob(f"shard_{country}_*.npz"))
        if not shard_paths:
            raise SystemExit(f"no shards for {country} in {shard_dir}")

        assigned: dict[int, list[str]] = {}
        n_pairs = 0
        # Candidates are kept as int32 index pairs into a single per-country
        # record-id table. Held as Python lists of id strings instead, India's
        # ~44M candidates would cost several GB.
        all_record_ids: list[np.ndarray] = []
        pair_entity_blocks: list[np.ndarray] = []
        pair_record_blocks: list[np.ndarray] = []
        record_offset = 0

        for path in shard_paths:
            data = np.load(path, allow_pickle=False)
            record_ids = data["record_ids"]

            keep = (data["probability"] >= args.threshold) & (data["margin"] >= args.margin)
            rows = data["entity_row"][keep]
            recs = record_ids[data["record_index"][keep]]
            for entity_row, record_id in zip(rows.tolist(), recs.tolist()):
                assigned.setdefault(entity_row, []).append(record_id)

            if candidate_out is not None and "pair_entity" in data:
                pair_entity_blocks.append(data["pair_entity"])
                # Shard-local record indices become global by offsetting.
                pair_record_blocks.append(
                    data["pair_record"].astype(np.int64) + record_offset
                )
                n_pairs += data["pair_entity"].size
            all_record_ids.append(record_ids)
            record_offset += record_ids.shape[0]
            del data

        record_table = (
            np.concatenate(all_record_ids) if all_record_ids else np.empty(0, dtype="<U1")
        )
        del all_record_ids

        boundaries = None
        pair_entity = pair_record = None
        if candidate_out is not None and pair_entity_blocks:
            pair_entity = np.concatenate(pair_entity_blocks)
            pair_record = np.concatenate(pair_record_blocks)
            del pair_entity_blocks, pair_record_blocks
            order = np.argsort(pair_entity, kind="stable")
            pair_entity = pair_entity[order]
            pair_record = pair_record[order]
            del order
            boundaries = np.searchsorted(pair_entity, np.arange(n_entities + 1), side="left")

        for entity_row, entity_id in enumerate(entity_ids):
            ids = assigned.get(entity_row)
            matching_out.write(f"{entity_id}\t{','.join(ids) if ids else ''}\n")
            if candidate_out is not None:
                if boundaries is None:
                    candidate_out.write(f"{entity_id}\t\n")
                    continue
                lo, hi = int(boundaries[entity_row]), int(boundaries[entity_row + 1])
                if lo == hi:
                    candidate_out.write(f"{entity_id}\t\n")
                    continue
                seen: dict[str, None] = {}
                for record_index in pair_record[lo:hi].tolist():
                    seen.setdefault(str(record_table[record_index]), None)
                # Matches must be a subset of candidates. True by construction,
                # but a violation means a pipeline bug, so check rather than trust.
                if ids:
                    missing = [i for i in ids if i not in seen]
                    if missing:
                        raise AssertionError(
                            f"{entity_id}: matched ids absent from candidates: {missing[:3]}"
                        )
                candidate_out.write(f"{entity_id}\t{','.join(seen)}\n")

        n_assigned = sum(len(v) for v in assigned.values())
        grand_entities += n_entities
        grand_assigned += n_assigned
        grand_matched += len(assigned)
        grand_pairs += n_pairs
        print(
            f"  {country:<8} {n_entities:>9,} entities  {n_assigned:>10,} records assigned  "
            f"{len(assigned):>9,} matched ({len(assigned) / max(n_entities, 1):.1%})  "
            f"{n_entities - len(assigned):>9,} singleton",
            flush=True,
        )
        del assigned, record_table, pair_entity, pair_record

    matching_out.close()
    if candidate_out is not None:
        candidate_out.close()

    print(
        f"\ntotals: {grand_entities:,} entities, {grand_assigned:,} records assigned, "
        f"{grand_matched:,} entities matched "
        f"({grand_entities - grand_matched:,} predicted singleton, "
        f"{(grand_entities - grand_matched) / max(grand_entities, 1):.2%})"
    )
    print(f"  mean matches per matched entity: {grand_assigned / max(grand_matched, 1):.2f}")
    print(f"\nwrote {matching_path} ({matching_path.stat().st_size / 1e6:.1f} MB)")
    if candidate_out is not None:
        print(
            f"wrote {candidate_path} ({candidate_path.stat().st_size / 1e9:.2f} GB, "
            f"{grand_pairs:,} pairs)"
        )
    print(f"seconds: {time.time() - started:.0f}")

    if args.validate:
        command = [
            sys.executable,
            str(config.VALIDATOR),
            "--matching", str(matching_path),
            "--test-dir", str(config.TEST_DIR),
        ]
        if candidate_out is not None:
            command[4:4] = ["--candidate", str(candidate_path)]
        print("\n$ " + " ".join(command), flush=True)
        completed = subprocess.run(command, capture_output=True, text=True)
        print(completed.stdout)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
        print(f"validator exit code: {completed.returncode}")
        sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
