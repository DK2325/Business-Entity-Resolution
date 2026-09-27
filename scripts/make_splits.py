"""Build and persist the validation splits.

Writes ``artifacts/splits.json`` holding the grouped holdout and the
leave-one-country-out folds, plus a short summary to stdout.

Run: ``python scripts/make_splits.py``
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.splits import grouped_holdout, leave_one_country_out, save_splits


def main() -> None:
    config.ensure_dirs()

    entity_country: dict[str, str] = {}
    with open(config.TRAIN_FILES["source1"], encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                entity_country[parts[0]] = parts[3]
    print(f"Source 1 training entities: {len(entity_country):,}")

    holdout = grouped_holdout(entity_country, seed=config.SEED)
    loco = leave_one_country_out(entity_country)

    out = config.ARTIFACTS / "splits.json"
    save_splits(out, holdout, loco)

    def mix(ids: list[str]) -> str:
        counts = collections.Counter(entity_country[i] for i in ids)
        return ", ".join(f"{c} {n:,}" for c, n in sorted(counts.items()))

    print(f"\nGrouped holdout  train {len(holdout['train']):,}  ({mix(holdout['train'])})")
    print(f"                 valid {len(holdout['valid']):,}  ({mix(holdout['valid'])})")
    for country, fold in loco.items():
        print(f"LOCO hold out {country:<6} train {len(fold['train']):,}  valid {len(fold['valid']):,}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
