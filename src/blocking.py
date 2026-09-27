"""Candidate generation.

Blocking sets the ceiling on everything downstream: a true link that never
enters the candidate set cannot be recovered by any model. The design here is
driven by three measurements in ``reports/eda.md``.

**Retrieval runs from the Source 2/3 side, not the Source 1 side.** No Source 2/3
record is ever claimed by two Source 1 entities, so each one needs exactly one
Source 1 partner and a shortlist of ``k`` is generous. That bounds the candidate
set at ``|S2| + |S3|`` times ``k`` — about 10M * k pairs on the test set —
independently of how many matches any one Source 1 entity attracts. Retrieving
from the Source 1 side instead needs a large ``K`` for the entities with many
matches, and spends that ``K`` on every entity including the 5.6% singletons.
The reverse view also falls out for free: the rank a Source 1 entity holds in a
candidate's own shortlist is the reverse-best-match signal the classifier wants.

**Country partitions the problem exactly.** True links never cross countries, so
every index is built per country. The label is carried as an opaque string and
never enumerated, which is what lets an unseen country flow through unchanged.

**Postal codes are not available.** They appear in 10.9% of US addresses and
under 0.5% of Indian and French ones, so postal-code blocking keys were dropped;
locality tokens from the tail of the address carry the geographic signal instead.

Scoring is IDF-weighted overlap, computed as a cosine product between L2-normalised
sparse rows. Tokens above a document-frequency cap are dropped from the index --
they cost a great deal of intersection work and carry almost no evidence.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy import sparse

DEFAULT_MAX_DF_RATIO = 0.002   # a token in >0.2% of a country's records is noise
DEFAULT_MIN_DF = 1
DEFAULT_TOPK = 5


class TokenVocabulary:
    """Maps string tokens to int32 column indices, with document frequencies.

    Built from the side being searched (the Source 1 index), so the IDF weights
    describe the reference source and both sides share one coordinate system.
    """

    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {}
        self.doc_freq: list[int] = []

    def add_document(self, tokens: set[str]) -> None:
        for token in tokens:
            index = self.token_to_id.get(token)
            if index is None:
                self.token_to_id[token] = len(self.doc_freq)
                self.doc_freq.append(1)
            else:
                self.doc_freq[index] += 1

    def finalise(self, n_docs: int, max_df_ratio: float, min_df: int) -> np.ndarray:
        """Return per-token IDF weights, with capped tokens zeroed out.

        A zero weight removes the token from scoring without disturbing the
        column numbering, which keeps the two matrices aligned.
        """
        freq = np.asarray(self.doc_freq, dtype=np.float64)
        max_df = max(1.0, max_df_ratio * n_docs)
        idf = np.log(1.0 + n_docs / np.maximum(freq, 1.0))
        idf[freq > max_df] = 0.0
        idf[freq < min_df] = 0.0
        return idf.astype(np.float32)

    def __len__(self) -> int:
        return len(self.doc_freq)


def build_matrix(
    documents: Iterable[set[str]],
    vocab: TokenVocabulary,
    idf: np.ndarray,
    n_columns: int | None = None,
) -> sparse.csr_matrix:
    """Build an L2-normalised IDF-weighted sparse matrix over ``vocab``.

    ``documents`` is consumed lazily, so the caller can stream a 5M-row source
    file straight into the matrix rather than materialising its token sets.

    Rows whose tokens are all out-of-vocabulary or all capped come out empty;
    they retrieve nothing and are reported as blocking misses rather than
    silently scoring zero against everything.
    """
    token_to_id = vocab.token_to_id
    indptr: list[int] = [0]
    indices: list[int] = []
    data: list[float] = []

    for tokens in documents:
        row_cols: list[int] = []
        row_vals: list[float] = []
        for token in tokens:
            col = token_to_id.get(token)
            if col is None:
                continue
            weight = idf[col]
            if weight > 0.0:
                row_cols.append(col)
                row_vals.append(float(weight))
        if row_vals:
            norm = float(np.sqrt(np.sum(np.square(row_vals, dtype=np.float64))))
            if norm > 0.0:
                inv = 1.0 / norm
                row_vals = [v * inv for v in row_vals]
        indices.extend(row_cols)
        data.extend(row_vals)
        indptr.append(len(indices))

    n_rows = len(indptr) - 1
    width = n_columns if n_columns is not None else max(len(vocab), 1)
    return sparse.csr_matrix(
        (
            np.asarray(data, dtype=np.float32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(n_rows, width),
    )


def topk_matches(
    query: sparse.csr_matrix,
    index: sparse.csr_matrix,
    k: int = DEFAULT_TOPK,
    chunk_rows: int = 20000,
    min_score: float = 0.0,
    index_transpose: sparse.csr_matrix | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """For each query row, the ``k`` highest-scoring index rows.

    Returns ``(ids, scores)``, both ``(n_queries, k)``, padded with ``-1`` and
    ``0.0`` where fewer than ``k`` rows scored above ``min_score``.

    The product is taken in row chunks: the full query-by-index product is far
    too large to materialise, but one chunk of a few tens of thousands of rows
    stays small once high-frequency tokens have been capped out of the index.
    """
    n_queries = query.shape[0]
    out_ids = np.full((n_queries, k), -1, dtype=np.int32)
    out_scores = np.zeros((n_queries, k), dtype=np.float32)

    # Transposing the index is the expensive part and does not depend on the
    # query, so callers running many chunks should pass a prepared transpose.
    index_t = index_transpose if index_transpose is not None else index.T.tocsr()

    for start in range(0, n_queries, chunk_rows):
        stop = min(start + chunk_rows, n_queries)
        block = (query[start:stop] @ index_t).tocsr()

        for local_row in range(block.shape[0]):
            lo, hi = block.indptr[local_row], block.indptr[local_row + 1]
            if lo == hi:
                continue
            cols = block.indices[lo:hi]
            vals = block.data[lo:hi]

            if min_score > 0.0:
                keep = vals > min_score
                if not keep.any():
                    continue
                cols, vals = cols[keep], vals[keep]

            if vals.size > k:
                part = np.argpartition(vals, vals.size - k)[-k:]
                cols, vals = cols[part], vals[part]

            order = np.argsort(vals)[::-1]
            cols, vals = cols[order], vals[order]

            row = start + local_row
            out_ids[row, : cols.size] = cols
            out_scores[row, : vals.size] = vals

    return out_ids, out_scores


def invert_to_source1(
    candidate_ids: np.ndarray,
    candidate_scores: np.ndarray,
    query_entity_ids: list[str],
    index_entity_ids: list[str],
) -> dict[str, list[tuple[str, float, int]]]:
    """Turn Source 2/3-side shortlists into per-Source-1 candidate lists.

    Each entry is ``(candidate_id, score, rank)`` where ``rank`` is the position
    the Source 1 entity held in that candidate's own shortlist. Rank 0 means the
    candidate considers this entity its single best partner, which given
    exclusivity is the strongest cheap evidence available.
    """
    grouped: dict[str, list[tuple[str, float, int]]] = {}
    for query_row, (ids, scores) in enumerate(zip(candidate_ids, candidate_scores)):
        candidate_id = query_entity_ids[query_row]
        for rank, (index_row, score) in enumerate(zip(ids, scores)):
            if index_row < 0:
                continue
            entity_id = index_entity_ids[index_row]
            grouped.setdefault(entity_id, []).append((candidate_id, float(score), rank))
    return grouped
