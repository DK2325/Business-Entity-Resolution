"""Validation splits.

Two splits, serving different questions:

**Grouped holdout** — a random 18% of Source 1 entities, stratified by country.
Grouping by Source 1 entity is the only safe unit: because every Source 2/3
record belongs to at most one Source 1 entity, holding out an entity also holds
out its whole match group, so no true link can straddle the boundary and leak.
This is the split that estimates the leaderboard score.

**Leave-one-country-out** — train on US, evaluate on India, and the reverse.
The test set contains France, which appears nowhere in training, so the number
that matters is not how well the model does on countries it has seen but how far
it falls on one it has not. LOCO is the only proxy we have for that, and a rule
that survives it is worth more than one that scores marginally higher in-domain.

Five-fold cross-validation was considered and dropped: at 2.2M entities the
extra folds cost several hours for a tighter confidence interval on a number
that LOCO shows is not the binding risk.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

HOLDOUT_FRACTION = 0.18


def grouped_holdout(
    entity_country: dict[str, str],
    fraction: float = HOLDOUT_FRACTION,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Split Source 1 entity IDs into train/validation, stratified by country.

    Stratifying keeps the country mix identical on both sides, so a score
    movement reflects the model rather than a shift in the country balance.
    """
    by_country: dict[str, list[str]] = defaultdict(list)
    for entity_id, country in entity_country.items():
        by_country[country].append(entity_id)

    rng = random.Random(seed)
    train: list[str] = []
    valid: list[str] = []
    for country, ids in sorted(by_country.items()):
        ids = sorted(ids)
        rng.shuffle(ids)
        cut = int(round(len(ids) * fraction))
        valid.extend(ids[:cut])
        train.extend(ids[cut:])
    return {"train": sorted(train), "valid": sorted(valid)}


def leave_one_country_out(entity_country: dict[str, str]) -> dict[str, dict[str, list[str]]]:
    """One fold per country: train on everything else, evaluate on that country.

    With US and India in training this yields two folds, each a full domain
    shift, which is the closest available stand-in for France.
    """
    by_country: dict[str, list[str]] = defaultdict(list)
    for entity_id, country in entity_country.items():
        by_country[country].append(entity_id)

    folds: dict[str, dict[str, list[str]]] = {}
    for held in sorted(by_country):
        train = [e for c, ids in by_country.items() if c != held for e in ids]
        folds[held] = {"train": sorted(train), "valid": sorted(by_country[held])}
    return folds


def save_splits(path: str | Path, holdout: dict, loco: dict) -> None:
    """Persist both splits as a single JSON document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "holdout": holdout,
        "loco": loco,
        "holdout_fraction": HOLDOUT_FRACTION,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_splits(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
