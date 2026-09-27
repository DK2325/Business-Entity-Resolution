"""Production candidate generation, shared by training and inference.

Same retrieval as the probe, packaged for reuse: build one Source 1 index per
country per view, then stream Source 2/3 records through it and emit, for each
record, its shortlist of Source 1 entities with the blocking context the model
needs (per-view rank and score, margin to the next best, shortlist size).

The unit of work is the **Source 2/3 record**, not the Source 1 entity. Since no
Source 2/3 record belongs to more than one Source 1 entity, each record needs at
most one partner, so a record and its shortlist is a self-contained decision.
This also makes training and inference share a distribution exactly: at both
times the model sees "one record, its k candidates, at most one correct".
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .blocking import TokenVocabulary, build_matrix, topk_matches
from .keys import VIEWS

DEFAULT_VIEWS = ("composite", "ngram")


def iter_country_rows(path: str | Path, country: str) -> Iterator[tuple[str, str, str]]:
    """Stream ``(entity_id, name, address)`` for one country."""
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4 and parts[3] == country:
                yield parts[0], parts[1], parts[2]


@dataclass
class CountryIndex:
    """One country's Source 1 side, indexed under each view."""

    country: str
    entity_ids: list[str]
    names: list[str]
    addresses: list[str]
    view_names: tuple[str, ...]
    vocabs: list[TokenVocabulary] = field(default_factory=list)
    idfs: list[np.ndarray] = field(default_factory=list)
    matrices: list = field(default_factory=list)
    transposes: list = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.entity_ids)

    def token_idf(self) -> dict[str, float]:
        """Flat token -> IDF map, used by the pair features.

        Merged across views, keeping the largest weight when a token appears in
        more than one, so the feature code can look up any token without needing
        to know which view produced it.
        """
        merged: dict[str, float] = {}
        for vocab, idf in zip(self.vocabs, self.idfs):
            for token, position in vocab.token_to_id.items():
                weight = float(idf[position])
                if weight > 0.0 and weight > merged.get(token, 0.0):
                    merged[token] = weight
        return merged


def build_country_index(
    source1_path: str | Path,
    country: str,
    view_names: tuple[str, ...] = DEFAULT_VIEWS,
    max_df_ratio: float = 0.002,
    verbose: bool = True,
) -> CountryIndex:
    """Index every Source 1 record of one country under each view."""
    entity_ids: list[str] = []
    names: list[str] = []
    addresses: list[str] = []
    for entity_id, name, address in iter_country_rows(source1_path, country):
        entity_ids.append(entity_id)
        names.append(name)
        addresses.append(address)

    index = CountryIndex(
        country=country,
        entity_ids=entity_ids,
        names=names,
        addresses=addresses,
        view_names=tuple(view_names),
    )
    if not entity_ids:
        return index

    for view_name in view_names:
        view = VIEWS[view_name]
        vocab = TokenVocabulary()
        for name, address in zip(names, addresses):
            vocab.add_document(view(name, address))
        idf = vocab.finalise(len(entity_ids), max_df_ratio, min_df=1)
        matrix = build_matrix(
            (view(name, address) for name, address in zip(names, addresses)), vocab, idf
        )
        index.vocabs.append(vocab)
        index.idfs.append(idf)
        index.matrices.append(matrix)
        index.transposes.append(matrix.T.tocsr())
        if verbose:
            print(
                f"  [{country}/{view_name}] {len(vocab):,} tokens, {matrix.nnz:,} nnz",
                flush=True,
            )
    return index


@dataclass
class Shortlist:
    """One Source 2/3 record and the Source 1 entities retrieved for it."""

    record_id: str
    name: str
    address: str
    is_s3: bool
    # index row -> per-view rank and score
    ranks: dict[int, dict[str, int]]
    scores: dict[int, dict[str, float]]
    best_rank: dict[int, int]
    best_score: dict[int, float]

    def rows(self) -> list[int]:
        return list(self.ranks)


def retrieve(
    index: CountryIndex,
    records: list[tuple[str, str, str, bool]],
    k: int,
    chunk_rows: int = 20000,
    score_floor: float = 0.0,
) -> list[Shortlist]:
    """Retrieve the top ``k`` Source 1 entities per view for each record.

    ``records`` is ``(record_id, name, address, is_s3)``. Results merge the views:
    a candidate is kept if any view ranked it in its own top ``k``, and carries
    that view's rank and score plus the best across views.

    ``score_floor`` drops nominations whose blocking score falls below it. The
    organisers rank a smaller candidate set per Source 1 entity higher, and weak
    nominations are almost never matched, so the floor buys a substantially
    smaller candidate set for very little recall. It is applied during retrieval
    rather than afterwards so that the candidates we report are exactly the ones
    the model scores.
    """
    if not records or not index.matrices:
        return []

    merged_ranks: list[dict[int, dict[str, int]]] = [{} for _ in records]
    merged_scores: list[dict[int, dict[str, float]]] = [{} for _ in records]

    for view_name, vocab, idf, matrix, transpose in zip(
        index.view_names, index.vocabs, index.idfs, index.matrices, index.transposes
    ):
        view = VIEWS[view_name]
        query = build_matrix(
            (view(name, address) for _rid, name, address, _s3 in records),
            vocab,
            idf,
            n_columns=matrix.shape[1],
        )
        ids, scores = topk_matches(
            query,
            matrix,
            k=k,
            chunk_rows=chunk_rows,
            index_transpose=transpose,
            min_score=score_floor,
        )
        for row in range(len(records)):
            rank_bucket = merged_ranks[row]
            score_bucket = merged_scores[row]
            for rank, (index_row, score) in enumerate(zip(ids[row], scores[row])):
                if index_row < 0:
                    continue
                key = int(index_row)
                rank_bucket.setdefault(key, {})[view_name] = rank
                score_bucket.setdefault(key, {})[view_name] = float(score)

    out: list[Shortlist] = []
    for row, (record_id, name, address, is_s3) in enumerate(records):
        ranks = merged_ranks[row]
        scores = merged_scores[row]
        best_rank = {key: min(view_ranks.values()) for key, view_ranks in ranks.items()}
        best_score = {key: max(view_scores.values()) for key, view_scores in scores.items()}
        out.append(
            Shortlist(
                record_id=record_id,
                name=name,
                address=address,
                is_s3=is_s3,
                ranks=ranks,
                scores=scores,
                best_rank=best_rank,
                best_score=best_score,
            )
        )
    return out


def shortlist_context(shortlist: Shortlist, index_row: int) -> dict:
    """Blocking-derived context for one (record, entity) pair.

    ``score_margin_to_next`` is the gap between this entity's best score and the
    best score of the record's strongest *other* candidate. A large positive
    margin means the record has one clear owner, which under exclusivity is
    strong evidence; a margin near zero means the record cannot tell two
    entities apart and the model should be reluctant.
    """
    ranks = shortlist.ranks.get(index_row, {})
    scores = shortlist.scores.get(index_row, {})
    own_score = shortlist.best_score.get(index_row, 0.0)

    rival_best = 0.0
    for other_row, other_score in shortlist.best_score.items():
        if other_row != index_row and other_score > rival_best:
            rival_best = other_score

    return {
        "rank_composite": ranks.get("composite", 99),
        "rank_ngram": ranks.get("ngram", 99),
        "rank_best": shortlist.best_rank.get(index_row, 99),
        "score_composite": scores.get("composite", 0.0),
        "score_ngram": scores.get("ngram", 0.0),
        "score_best": own_score,
        "score_margin_to_next": own_score - rival_best,
        "n_candidates_for_record": len(shortlist.ranks),
        "candidate_is_s3": shortlist.is_s3,
    }
