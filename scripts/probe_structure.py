"""How much of the matching is recoverable by exact keys on normalised text?

The leaderboard says ~0.99 is achievable. Our blocking tops out near 0.91 on
India, which caps the score no matter how good the matcher is, so the question
is not "how do we rank better" but "what makes a true link findable at all".

This measures, for true links, how often a simple deterministic key is *exactly*
equal on both sides, and how selective that key is (how many distinct entities
share it). A key that is both high-coverage and highly selective would mean the
data is far more regular than our IDF-weighted retrieval assumes -- and that the
document-frequency cap we added for speed is throwing away the signal.

Usage::

    python scripts/probe_structure.py --country India --sample 40000
"""

from __future__ import annotations

import argparse
import collections
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.normalize import (
    numeric_keys,
    romanized_address_tokens,
    romanized_tokens,
)


def keys_for(name: str, address: str) -> dict[str, str | None]:
    """A family of deterministic keys, cheapest and strictest first."""
    addr = romanized_address_tokens(address)
    core = romanized_tokens(name)
    numbers = sorted(numeric_keys(addr))
    addr_sorted = sorted(addr)
    core_sorted = sorted(core)

    return {
        # whole normalised address, order-insensitive
        "addr_set": " ".join(addr_sorted) if addr_sorted else None,
        # address words with the digits removed, order-insensitive
        "addr_words": " ".join(t for t in addr_sorted if not any(c.isdigit() for c in t)) or None,
        # just the digit-bearing tokens
        "numbers": " ".join(numbers) if numbers else None,
        # name core only
        "name_core": " ".join(core_sorted) if core_sorted else None,
        # name core + digits: the pairing we guessed was discriminative
        "name_numbers": (" ".join(core_sorted) + "|" + " ".join(numbers))
        if (core_sorted and numbers)
        else None,
        # digits + the last address component (locality anchor)
        "numbers_locality": (
            " ".join(numbers) + "|" + (addr_sorted[-1] if addr_sorted else "")
        )
        if numbers
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", default="India")
    parser.add_argument("--sample", type=int, default=40000)
    args = parser.parse_args()

    rng = random.Random(config.SEED)

    truth: dict[str, list[str]] = {}
    with open(config.TRAIN_FILES["ground_truth"], encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            ids = parts[1] if len(parts) > 1 else ""
            if ids.strip():
                truth[parts[0]] = [x for x in ids.split(",") if x]

    s1_rows: dict[str, tuple[str, str]] = {}
    with open(config.TRAIN_FILES["source1"], encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4 and parts[3] == args.country and parts[0] in truth:
                s1_rows[parts[0]] = (parts[1], parts[2])
    print(f"{args.country} Source 1 entities with matches: {len(s1_rows):,}")

    sampled = set(rng.sample(list(s1_rows), min(args.sample, len(s1_rows))))
    wanted = {r for e in sampled for r in truth[e]}
    print(f"sampled entities: {len(sampled):,}, their true links: {len(wanted):,}")

    partners: dict[str, tuple[str, str]] = {}
    for source in ("source2", "source3"):
        with open(config.TRAIN_FILES[source], encoding="utf-8") as handle:
            handle.readline()
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if parts[0] in wanted:
                    partners[parts[0]] = (parts[1], parts[2] if len(parts) > 2 else "")
    print(f"partner records loaded: {len(partners):,}\n")

    # --- coverage: how often is each key exactly equal across a true link? ---
    key_names = list(keys_for("x", "y").keys())
    equal = collections.Counter()
    present = collections.Counter()
    total_links = 0

    s1_keys: dict[str, dict[str, str | None]] = {}
    for entity in sampled:
        name, address = s1_rows[entity]
        s1_keys[entity] = keys_for(name, address)

    for entity in sampled:
        left = s1_keys[entity]
        for record in truth[entity]:
            right_row = partners.get(record)
            if right_row is None:
                continue
            total_links += 1
            right = keys_for(*right_row)
            for key in key_names:
                if left[key] is not None and right[key] is not None:
                    present[key] += 1
                    if left[key] == right[key]:
                        equal[key] += 1

    print("=== exact-key coverage on true links ===")
    print(f"{'key':<20}{'both present':>14}{'exactly equal':>15}{'coverage':>11}")
    for key in key_names:
        cov = equal[key] / total_links if total_links else 0.0
        print(f"{key:<20}{present[key]:>14,}{equal[key]:>15,}{cov:>10.2%}")
    print(f"\ntrue links examined: {total_links:,}")

    # --- selectivity: how many distinct entities share a key value? ---
    print("\n=== key selectivity over the whole country's Source 1 ===")
    all_s1: list[tuple[str, str, str]] = []
    with open(config.TRAIN_FILES["source1"], encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4 and parts[3] == args.country:
                all_s1.append((parts[0], parts[1], parts[2]))
    print(f"indexing {len(all_s1):,} Source 1 records")

    for key in key_names:
        buckets: collections.Counter = collections.Counter()
        for _eid, name, address in all_s1:
            value = keys_for(name, address)[key]
            if value is not None:
                buckets[value] += 1
        if not buckets:
            continue
        sizes = list(buckets.values())
        singleton_keys = sum(1 for v in sizes if v == 1)
        weighted = sum(v * v for v in sizes) / sum(sizes)
        print(
            f"{key:<20} distinct {len(buckets):>9,}  "
            f"unique-bucket {singleton_keys / len(buckets):>6.1%}  "
            f"mean entities per lookup {weighted:>8.1f}"
        )


if __name__ == "__main__":
    main()
