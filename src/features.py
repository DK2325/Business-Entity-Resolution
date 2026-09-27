"""Features for a (Source 1, candidate) pair.

Grouped by what they measure, because the groups behave very differently under
the metric:

* **Agreement** features say the two records describe the same business. The EDA
  found the strongest of these is agreement on the digit-bearing address tokens:
  true pairs have a median numeric Jaccard of 1.0 against 0.0 for hard negatives.
* **Conflict** features say they do not, which matters more than it sounds. Under
  macro F_0.5 a false merge is far more expensive than a miss, so an explicit
  "both records carry house numbers and they disagree" signal is worth more than
  another shade of similarity.
* **Context** features come from blocking rather than the text: where the Source 1
  entity ranked in this candidate's own shortlist, and how far behind the
  candidate's best alternative it sits. Because no Source 2/3 record belongs to
  more than one Source 1 entity, a candidate that ranks this entity first with a
  wide margin is far likelier to be a true match than one that ranks it third in
  a crowded field. These are the features that let the model exploit exclusivity
  instead of judging each pair in isolation.

Nothing here reads the country label or any external resource.
"""

from __future__ import annotations

import math

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from .keys import char_ngrams
from .normalize import (
    numeric_keys,
    romanized_address_tokens,
    romanized_tokens,
)

# Order matters: the model consumes a plain float vector, and this list is the
# single source of truth for what each position means.
FEATURE_NAMES: list[str] = [
    # name agreement
    "name_jaccard",
    "name_token_set_ratio",
    "name_token_sort_ratio",
    "name_partial_ratio",
    "name_jaro_winkler",
    "name_trigram_jaccard",
    "name_idf_overlap",
    "name_rare_shared",
    "name_acronym_match",
    "name_len_ratio",
    "name_either_empty",
    # address agreement
    "addr_jaccard",
    "addr_token_set_ratio",
    "addr_trigram_jaccard",
    "addr_idf_overlap",
    "addr_rare_shared",
    "addr_len_ratio",
    # numbers: the sharpest signal in both directions
    "num_jaccard",
    "num_shared",
    "num_conflict",
    "num_either_missing",
    # emptiness
    "addr_either_empty",
    "addr_both_empty",
    # context from blocking
    "rank_composite",
    "rank_ngram",
    "rank_best",
    "score_composite",
    "score_ngram",
    "score_best",
    "is_reverse_best",
    "score_margin_to_next",
    "n_candidates_for_record",
    "candidate_is_s3",
]

N_FEATURES = len(FEATURE_NAMES)
_MISSING_RANK = 99.0


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _idf_overlap(left: set[str], right: set[str], idf: dict[str, float]) -> tuple[float, float]:
    """Summed IDF of shared tokens, normalised, plus the rarest shared weight.

    Sharing one distinctive token (a plot number, an unusual trade name) is much
    stronger evidence than sharing several generic ones, so the maximum shared
    IDF is reported next to the normalised sum.
    """
    shared = left & right
    if not shared:
        return 0.0, 0.0
    weights = [idf.get(token, 0.0) for token in shared]
    total = sum(weights)
    denominator = sum(idf.get(token, 0.0) for token in (left | right)) or 1.0
    return total / denominator, max(weights, default=0.0)


def _acronym(tokens: list[str]) -> str:
    return "".join(token[0] for token in tokens if token)


def _ratio(a: int, b: int) -> float:
    """Size ratio in [0, 1]; 0 when either side is empty."""
    if a <= 0 or b <= 0:
        return 0.0
    return min(a, b) / max(a, b)


class PairFeaturizer:
    """Computes the feature vector for pairs, caching per-record parsing.

    A Source 1 entity is compared against many candidates and each candidate
    against several entities, so the tokenisation of each record is memoised;
    re-deriving it per pair dominates the runtime otherwise.
    """

    def __init__(self, idf: dict[str, float] | None = None) -> None:
        self.idf = idf or {}
        self._cache: dict[str, dict] = {}

    def parse(self, entity_id: str, name: str, address: str) -> dict:
        cached = self._cache.get(entity_id)
        if cached is not None:
            return cached
        name_tokens = romanized_tokens(name)
        addr_tokens = romanized_address_tokens(address)
        parsed = {
            "name_tokens": name_tokens,
            "name_set": set(name_tokens),
            "name_joined": " ".join(name_tokens),
            "name_grams": char_ngrams(" ".join(name_tokens)),
            "acronym": _acronym(name_tokens),
            "addr_tokens": addr_tokens,
            "addr_set": set(addr_tokens),
            "addr_joined": " ".join(addr_tokens),
            "addr_grams": char_ngrams(" ".join(addr_tokens)),
            "numbers": numeric_keys(addr_tokens),
            "addr_empty": not address.strip(),
            "name_empty": not name.strip(),
        }
        self._cache[entity_id] = parsed
        return parsed

    def clear(self) -> None:
        self._cache.clear()

    def features(self, left: dict, right: dict, context: dict) -> list[float]:
        """Feature vector for one pair.

        ``context`` carries the blocking-derived fields: per-view rank and score,
        whether the Source 1 entity is this candidate's top choice, the margin to
        the candidate's next best entity, how many candidates the record kept,
        and which source it came from.
        """
        idf = self.idf

        name_idf_sum, name_idf_max = _idf_overlap(left["name_set"], right["name_set"], idf)
        addr_idf_sum, addr_idf_max = _idf_overlap(left["addr_set"], right["addr_set"], idf)

        left_numbers, right_numbers = left["numbers"], right["numbers"]
        both_have_numbers = bool(left_numbers) and bool(right_numbers)
        # A conflict is only meaningful when both sides actually carry a number:
        # a missing number is silence, not disagreement.
        num_conflict = 1.0 if both_have_numbers and not (left_numbers & right_numbers) else 0.0

        return [
            # name agreement
            _jaccard(left["name_set"], right["name_set"]),
            fuzz.token_set_ratio(left["name_joined"], right["name_joined"]) / 100.0,
            fuzz.token_sort_ratio(left["name_joined"], right["name_joined"]) / 100.0,
            fuzz.partial_ratio(left["name_joined"], right["name_joined"]) / 100.0,
            JaroWinkler.similarity(left["name_joined"], right["name_joined"]),
            _jaccard(left["name_grams"], right["name_grams"]),
            name_idf_sum,
            name_idf_max,
            1.0 if left["acronym"] and left["acronym"] == right["acronym"] else 0.0,
            _ratio(len(left["name_joined"]), len(right["name_joined"])),
            1.0 if left["name_empty"] or right["name_empty"] else 0.0,
            # address agreement
            _jaccard(left["addr_set"], right["addr_set"]),
            fuzz.token_set_ratio(left["addr_joined"], right["addr_joined"]) / 100.0,
            _jaccard(left["addr_grams"], right["addr_grams"]),
            addr_idf_sum,
            addr_idf_max,
            _ratio(len(left["addr_joined"]), len(right["addr_joined"])),
            # numbers
            _jaccard(left_numbers, right_numbers),
            float(len(left_numbers & right_numbers)),
            num_conflict,
            0.0 if both_have_numbers else 1.0,
            # emptiness
            1.0 if left["addr_empty"] or right["addr_empty"] else 0.0,
            1.0 if left["addr_empty"] and right["addr_empty"] else 0.0,
            # context from blocking
            float(context.get("rank_composite", _MISSING_RANK)),
            float(context.get("rank_ngram", _MISSING_RANK)),
            float(context.get("rank_best", _MISSING_RANK)),
            float(context.get("score_composite", 0.0)),
            float(context.get("score_ngram", 0.0)),
            float(context.get("score_best", 0.0)),
            1.0 if context.get("rank_best", _MISSING_RANK) == 0 else 0.0,
            float(context.get("score_margin_to_next", 0.0)),
            float(context.get("n_candidates_for_record", 0)),
            1.0 if context.get("candidate_is_s3") else 0.0,
        ]


def feature_index(name: str) -> int:
    """Position of a named feature, for inspecting model importances."""
    return FEATURE_NAMES.index(name)


def sanity_check() -> None:
    """Assert the vector length matches the declared names."""
    featurizer = PairFeaturizer()
    left = featurizer.parse("a", "Dhule Roadlines Limited", "Plot No.F-3, Midc Avadhan, Dhule")
    right = featurizer.parse("b", "Dhule Roadlines Ltd", "Plot No.f-003, Midc Avadhan, Dhule")
    vector = featurizer.features(left, right, {})
    if len(vector) != N_FEATURES:
        raise AssertionError(f"{len(vector)} values for {N_FEATURES} declared names")
    if not math.isfinite(sum(vector)):
        raise AssertionError("non-finite feature value")
