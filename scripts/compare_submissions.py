"""Compare submission files per country, against reference distributions.

A threshold tuned on one model does not necessarily suit another, and the only
signal available before uploading is whether the prediction *shape* looks sane:
what fraction of entities get a match, how many records each matched entity
gets, and how many entities are predicted singleton. Training gives reference
values for the last two (5.58% singleton, 3.46 matches per entity); a new model
whose rates sit far from both those and from earlier submissions is a warning,
not a result.

Usage::

    python scripts/compare_submissions.py a.tsv b.tsv --labels "#1" "#4"
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config

TRAIN_SINGLETON_RATE = 5.58
TRAIN_MEAN_MATCHES = 3.46


def load_country_map() -> dict[str, str]:
    country: dict[str, str] = {}
    with open(config.TEST_FILES["source1"], encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                country[parts[0]] = parts[3]
    return country


def summarise(path: Path, country: dict[str, str]) -> dict:
    stats: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            ids = [x for x in (parts[1] if len(parts) > 1 else "").split(",") if x]
            row = stats[country.get(parts[0], "?")]
            row["n"] += 1
            if ids:
                row["matched"] += 1
                row["ids"] += len(ids)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--labels", nargs="+", default=None)
    args = parser.parse_args()

    labels = args.labels or [Path(f).stem for f in args.files]
    if len(labels) != len(args.files):
        raise SystemExit("--labels must match the number of files")

    country = load_country_map()
    summaries = [summarise(Path(f), country) for f in args.files]
    countries = sorted({c for s in summaries for c in s})

    print(f"{'country':<9}{'label':<10}{'entities':>10}{'matched':>11}{'matched%':>10}"
          f"{'singleton%':>12}{'mean ids':>10}")
    for c in countries:
        for label, s in zip(labels, summaries):
            row = s.get(c)
            if not row:
                continue
            n, matched, ids = row["n"], row["matched"], row["ids"]
            print(f"{c:<9}{label:<10}{n:>10,}{matched:>11,}"
                  f"{100 * matched / n:>9.2f}%{100 * (n - matched) / n:>11.2f}%"
                  f"{ids / max(matched, 1):>10.2f}")
        print()

    print(f"{'TOTAL':<9}{'label':<10}{'entities':>10}{'matched':>11}{'matched%':>10}"
          f"{'singleton%':>12}{'mean ids':>10}")
    flags: list[str] = []
    for label, s in zip(labels, summaries):
        n = sum(r["n"] for r in s.values())
        matched = sum(r["matched"] for r in s.values())
        ids = sum(r["ids"] for r in s.values())
        singleton_pct = 100 * (n - matched) / n
        mean_ids = ids / max(matched, 1)
        print(f"{'':<9}{label:<10}{n:>10,}{matched:>11,}"
              f"{100 * matched / n:>9.2f}%{singleton_pct:>11.2f}%{mean_ids:>10.2f}")
        # Sanity bands: training says 5.58% singleton and 3.46 matches per entity.
        # A precision-leaning rule should sit a little above the singleton rate,
        # not far from it in either direction.
        if not 4.0 <= singleton_pct <= 12.0:
            flags.append(f"{label}: singleton rate {singleton_pct:.2f}% outside 4-12% "
                         f"(training {TRAIN_SINGLETON_RATE}%)")
        if not 2.6 <= mean_ids <= 4.2:
            flags.append(f"{label}: mean ids {mean_ids:.2f} outside 2.6-4.2 "
                         f"(training {TRAIN_MEAN_MATCHES})")

    print(f"\nreference: training singleton rate {TRAIN_SINGLETON_RATE}%, "
          f"mean matches per entity {TRAIN_MEAN_MATCHES}")
    if flags:
        print("\nWARNINGS:")
        for f in flags:
            print(f"  ! {f}")
    else:
        print("\nno distribution warnings")


if __name__ == "__main__":
    main()
