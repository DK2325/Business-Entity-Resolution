"""Second-stage "collective" scoring: judge an assignment by its siblings.

Stage one scores each (record, entity) pair on its own. That is where most of
the remaining loss lives, and the reason is visible in the data: a Source 1
entity has on average 3.46 true records spread over Source 2 and Source 3, and
those records resemble *each other* because they describe the same business.
A distractor -- a record belonging to nobody, which is 90% of our wrong accepts
-- has no such siblings. It may look plausible next to the Source 1 record, but
it looks like nothing next to the other records assigned alongside it.

So the second stage asks a question the first one cannot: given everything else
assigned to this entity, does this record belong with them?

Three families of feature:

* **pair** -- what stage one already knew: probability, margin, and where the
  record sits among its own alternatives.
* **group** -- the shape of the entity's assigned set: how many records, how
  confident, which sources they came from. A record that is the entity's only
  nomination is in a very different position from one of four agreeing records.
* **sibling** -- textual agreement with the other records assigned to the same
  entity, and with the Source 1 record itself. This is the signal that separates
  a true match from a plausible-looking distractor.

Nothing here re-runs retrieval; it re-scores decisions already on disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rapidfuzz import fuzz

from .keys import char_ngrams
from .normalize import romanized_address_tokens, romanized_tokens

FEATURE_NAMES: list[str] = [
    # pair-level (what stage one saw)
    "prob",
    "margin",
    "prob_rank_in_group",
    "prob_minus_group_max",
    "prob_over_group_max",
    # group shape
    "group_size",
    "group_size_above_50",
    "group_size_above_70",
    "group_prob_sum",
    "group_prob_max",
    "group_prob_mean",
    "group_has_s2",
    "group_has_s3",
    "is_only_record_from_its_source",
    "source_is_s3",
    # sibling agreement
    "sib_name_max",
    "sib_name_mean",
    "sib_addr_max",
    "sib_addr_mean",
    "sib_combined_max",
    "n_siblings",
    # agreement with the Source 1 record itself
    "self_name_sim",
    "self_addr_sim",
    "self_combined_sim",
]

N_FEATURES = len(FEATURE_NAMES)


@dataclass
class Parsed:
    """Cached text representation of one record."""

    name_grams: set
    addr_grams: set
    name_joined: str
    addr_joined: str


def parse_record(name: str, address: str) -> Parsed:
    name_tokens = romanized_tokens(name)
    addr_tokens = romanized_address_tokens(address)
    name_joined = " ".join(name_tokens)
    addr_joined = " ".join(addr_tokens)
    return Parsed(
        name_grams=char_ngrams(name_joined),
        addr_grams=char_ngrams(addr_joined),
        name_joined=name_joined,
        addr_joined=addr_joined,
    )


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _similarity(a: Parsed, b: Parsed) -> tuple[float, float]:
    """(name similarity, address similarity) between two parsed records."""
    name = max(
        _jaccard(a.name_grams, b.name_grams),
        fuzz.token_set_ratio(a.name_joined, b.name_joined) / 100.0,
    )
    addr = _jaccard(a.addr_grams, b.addr_grams)
    return name, addr


@dataclass
class Group:
    """One Source 1 entity and every record currently assigned to it."""

    entity_parsed: Parsed
    record_ids: list[str] = field(default_factory=list)
    parsed: list[Parsed] = field(default_factory=list)
    probs: list[float] = field(default_factory=list)
    margins: list[float] = field(default_factory=list)
    is_s3: list[bool] = field(default_factory=list)


def group_features(group: Group) -> np.ndarray:
    """Feature matrix for every record in one group, one row per record."""
    n = len(group.record_ids)
    if n == 0:
        return np.zeros((0, N_FEATURES), dtype=np.float32)

    probs = np.asarray(group.probs, dtype=np.float32)
    margins = np.asarray(group.margins, dtype=np.float32)
    is_s3 = np.asarray(group.is_s3, dtype=bool)

    group_max = float(probs.max())
    group_sum = float(probs.sum())
    group_mean = float(probs.mean())
    n_above_50 = int((probs >= 0.5).sum())
    n_above_70 = int((probs >= 0.7).sum())
    has_s2 = bool((~is_s3).any())
    has_s3 = bool(is_s3.any())
    n_s2 = int((~is_s3).sum())
    n_s3 = int(is_s3.sum())

    order = np.argsort(-probs)
    rank = np.empty(n, dtype=np.float32)
    rank[order] = np.arange(n, dtype=np.float32)

    # Pairwise sibling similarity, computed once per unordered pair.
    name_sim = np.zeros((n, n), dtype=np.float32)
    addr_sim = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            ns, as_ = _similarity(group.parsed[i], group.parsed[j])
            name_sim[i, j] = name_sim[j, i] = ns
            addr_sim[i, j] = addr_sim[j, i] = as_

    rows = np.zeros((n, N_FEATURES), dtype=np.float32)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        if others:
            sib_name = name_sim[i, others]
            sib_addr = addr_sim[i, others]
            sib_name_max, sib_name_mean = float(sib_name.max()), float(sib_name.mean())
            sib_addr_max, sib_addr_mean = float(sib_addr.max()), float(sib_addr.mean())
            sib_combined_max = float((sib_name + sib_addr).max() / 2.0)
        else:
            sib_name_max = sib_name_mean = sib_addr_max = sib_addr_mean = 0.0
            sib_combined_max = 0.0

        self_name, self_addr = _similarity(group.parsed[i], group.entity_parsed)
        own_source_count = n_s3 if is_s3[i] else n_s2

        rows[i] = (
            probs[i],
            margins[i],
            rank[i],
            probs[i] - group_max,
            probs[i] / group_max if group_max > 0 else 0.0,
            n,
            n_above_50,
            n_above_70,
            group_sum,
            group_max,
            group_mean,
            1.0 if has_s2 else 0.0,
            1.0 if has_s3 else 0.0,
            1.0 if own_source_count == 1 else 0.0,
            1.0 if is_s3[i] else 0.0,
            sib_name_max,
            sib_name_mean,
            sib_addr_max,
            sib_addr_mean,
            sib_combined_max,
            float(len(others)),
            self_name,
            self_addr,
            (self_name + self_addr) / 2.0,
        )
    return rows
