"""Measure the recall ceiling and cost of candidate generation.

This decides the architecture, so the setup is deliberately realistic rather than
convenient:

* The index holds **every** Source 1 record of a country, not just the sampled
  ones, so candidates face the full field of competitors they will face at test
  time.
* The queries are **every** Source 2/3 record of that country, so the pair counts
  and runtimes extrapolate honestly.
* Recall is scored on a random sample of Source 1 entities, stratified by
  country, using their full ground-truth match lists.

Retrieval runs from the Source 2/3 side: each Source 2/3 record takes the top
``k`` Source 1 entities. Exclusivity means one partner per record is all that is
ever needed, which bounds the candidate set at ``(|S2| + |S3|) * k`` regardless
of how many matches an individual Source 1 entity attracts.

Usage::

    python scripts/probe_blocking.py --country India --view composite --k 10
    python scripts/probe_blocking.py --all-views --country India
    python scripts/probe_blocking.py --split test --view composite --k 5 --no-eval
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.blocking import DEFAULT_MAX_DF_RATIO, TokenVocabulary, build_matrix, topk_matches
from src.keys import VIEWS
from src.normalize import has_non_latin


# Candidate budgets reported alongside recall, so one run yields the full
# cost-versus-recall curve rather than one point on it.
PAIR_CUTS = (1, 2, 3, 5, 8, 10)


def peak_rss_gb() -> float:
    """Peak resident memory of this process, in GB."""
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / 1e9, 2)
    except Exception:
        return float("nan")


def iter_country_rows(path: Path, country: str):
    """Stream ``(entity_id, name, address)`` for one country."""
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4 and parts[3] == country:
                yield parts[0], parts[1], parts[2]


def read_truth(path: Path) -> dict[str, list[str]]:
    truth: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            ids = parts[1] if len(parts) > 1 else ""
            truth[parts[0]] = [x for x in ids.split(",") if x]
    return truth


def run_probe_s1_side(
    split: str,
    country: str,
    view_name: str,
    k: int,
    n_eval: int,
    max_df_ratio: float,
    chunk_rows: int,
) -> dict:
    """The mirror image: index Source 2/3, and take the top ``k`` for each Source 1.

    Kept for comparison against the Source 2/3-side probe. It is the more natural
    reading of the task but the worse engineering choice: the candidate budget
    ``k`` has to be large enough for the entity with eleven true matches, it is
    spent identically on the 5.6% of entities that have none, and the index it
    searches is the 10M-record side rather than the 1.7M-record one.
    """
    view = VIEWS[view_name]
    files = config.TRAIN_FILES if split == "train" else config.TEST_FILES
    started = time.time()

    def iter_index_rows():
        for source in ("source2", "source3"):
            yield from iter_country_rows(files[source], country)

    vocab = TokenVocabulary()
    index_ids: list[str] = []
    for entity_id, name, address in iter_index_rows():
        index_ids.append(entity_id)
        vocab.add_document(view(name, address))
    n_index = len(index_ids)
    idf = vocab.finalise(n_index, max_df_ratio, min_df=1)
    print(f"  index: {n_index:,} S2/S3 records, {len(vocab):,} tokens", flush=True)

    index_matrix = build_matrix(
        (view(name, address) for _eid, name, address in iter_index_rows()), vocab, idf
    )
    print(f"  index matrix: {index_matrix.nnz:,} nnz", flush=True)

    truth = read_truth(files["ground_truth"])
    rng = random.Random(config.SEED)
    s1_rows = [(e, n, a) for e, n, a in iter_country_rows(files["source1"], country)]
    with_links = [r for r in s1_rows if truth.get(r[0])]
    sampled = rng.sample(with_links, min(n_eval, len(with_links)))
    eval_targets = {e: set(truth[e]) for e, _n, _a in sampled}
    print(f"  scoring recall on {len(eval_targets):,} sampled entities", flush=True)

    # Only the sampled entities are queried: the pair-count figure is scaled up
    # to the full Source 1 side afterwards.
    query_matrix = build_matrix(
        [view(n, a) for _e, n, a in sampled], vocab, idf, n_columns=index_matrix.shape[1]
    )
    ids, _scores = topk_matches(query_matrix, index_matrix, k=k, chunk_rows=chunk_rows)

    retrieved: dict[str, dict[str, int]] = {}
    for row, (entity_id, _n, _a) in enumerate(sampled):
        bucket: dict[str, int] = {}
        for rank, index_row in enumerate(ids[row]):
            if index_row >= 0:
                bucket.setdefault(index_ids[index_row], rank)
        retrieved[entity_id] = bucket

    found_pairs = int((ids >= 0).sum())
    n_s1_total = len(s1_rows)
    result = {
        "split": split,
        "country": country,
        "view": view_name,
        "side": "s1",
        "k": k,
        "max_df_ratio": max_df_ratio,
        "n_index_s2s3": n_index,
        "n_query_s1": len(sampled),
        "n_s1_total": n_s1_total,
        "n_tokens": len(vocab),
        "index_nnz": int(index_matrix.nnz),
        "pairs_on_sample": found_pairs,
        "pairs_per_s1": round(found_pairs / len(sampled), 2),
        "total_pairs_scaled": int(round(found_pairs / len(sampled) * n_s1_total)),
        "total_seconds": round(time.time() - started, 1),
        "peak_rss_gb": peak_rss_gb(),
    }
    result.update(_score_recall(retrieved, eval_targets, k, files, country))
    return result


def run_probe(
    split: str,
    country: str,
    view_name: str,
    k: int,
    n_eval: int,
    max_df_ratio: float,
    chunk_rows: int,
    evaluate: bool,
    fast_eval: bool = False,
    stats_sample: int = 200_000,
) -> dict:
    """Index one country's Source 1, query its Source 2/3, and score recall.

    ``fast_eval`` exploits a property of Source 2/3-side retrieval: each query
    record takes its own top ``k`` against the Source 1 index, and query records
    never compete with one another. Recall on a sample of Source 1 entities is
    therefore determined entirely by the records that are their true matches --
    querying the other ~24 out of 25 records cannot change it, only the cost
    figures. So the fast path queries the true-match records for recall plus a
    random sample of the rest for pairs-per-query, and scales the sample up.
    The full path remains available to verify that the shortcut agrees.
    """
    # ``view_name`` may name several views separated by "+". Each gets its own
    # vocabulary, IDF weighting and index, and a candidate counts as retrieved if
    # any one of them puts it in its top k. Views disagree about which evidence
    # matters -- address words, name words, character n-grams -- so a link that
    # one ranking buries another often surfaces, and the union recovers it.
    view_names = [v for v in view_name.split("+") if v]
    views = [VIEWS[v] for v in view_names]
    files = config.TRAIN_FILES if split == "train" else config.TEST_FILES
    started = time.time()

    s1_ids: list[str] = []
    for entity_id, _name, _address in iter_country_rows(files["source1"], country):
        s1_ids.append(entity_id)
    n_index = len(s1_ids)
    if n_index == 0:
        raise SystemExit(f"no Source 1 records for country {country!r} in {split}")

    vocabs: list[TokenVocabulary] = []
    idfs: list[np.ndarray] = []
    matrices: list = []
    transposes: list = []
    n_active_total = 0
    n_tokens_total = 0

    for name_of_view, view in zip(view_names, views):
        vocab = TokenVocabulary()
        for _eid, name, address in iter_country_rows(files["source1"], country):
            vocab.add_document(view(name, address))
        idf = vocab.finalise(n_index, max_df_ratio, min_df=1)
        n_active = int((idf > 0).sum())
        n_active_total += n_active
        n_tokens_total += len(vocab)

        matrix = build_matrix(
            (
                view(name, address)
                for _eid, name, address in iter_country_rows(files["source1"], country)
            ),
            vocab,
            idf,
        )
        empty_index_rows = int((np.diff(matrix.indptr) == 0).sum())
        print(
            f"  [{name_of_view}] index: {len(vocab):,} tokens ({n_active:,} active), "
            f"{matrix.nnz:,} nnz, {empty_index_rows:,} empty rows  "
            f"[{time.time() - started:.0f}s]",
            flush=True,
        )
        vocabs.append(vocab)
        idfs.append(idf)
        matrices.append(matrix)
        transposes.append(matrix.T.tocsr())

    index_matrix = matrices[0]
    index_time = time.time() - started

    # ---- evaluation target ---------------------------------------------------
    truth: dict[str, list[str]] = {}
    eval_targets: dict[str, set[str]] = {}
    if evaluate:
        truth = read_truth(files["ground_truth"])
        s1_set = set(s1_ids)
        rng = random.Random(config.SEED)
        in_country = [e for e in s1_ids if truth.get(e)]
        sampled = set(rng.sample(in_country, min(n_eval, len(in_country))))
        eval_targets = {e: set(truth[e]) for e in sampled}
        del s1_set
        print(f"  scoring recall on {len(eval_targets):,} sampled entities", flush=True)

    # In fast mode, query only the records that can affect recall, plus a random
    # sample used purely to estimate pairs-per-query.
    must_query: set[str] = set()
    sample_rng = random.Random(config.SEED + 2)
    sample_rate = 0.0
    if fast_eval and evaluate:
        for true_ids in eval_targets.values():
            must_query.update(true_ids)
        total_queries_available = 0
        for source in ("source2", "source3"):
            total_queries_available += sum(1 for _ in iter_country_rows(files[source], country))
        sample_rate = min(1.0, stats_sample / max(total_queries_available, 1))
        print(
            f"  fast eval: {len(must_query):,} recall-relevant queries "
            f"+ ~{int(sample_rate * total_queries_available):,} sampled for cost stats "
            f"(of {total_queries_available:,})",
            flush=True,
        )

    # ---- query pass ----------------------------------------------------------
    # retrieved[s1_entity] -> {candidate_id: rank}; only entities under
    # evaluation are retained, which keeps the probe's memory flat.
    retrieved: dict[str, dict[str, int]] = {e: {} for e in eval_targets}
    total_pairs = 0
    n_queries = 0
    query_started = time.time()
    empty_queries = 0
    # Pairs and query count restricted to the randomly sampled records, which is
    # the unbiased basis for the cost extrapolation in fast mode.
    sampled_pairs = 0
    sampled_queries = 0
    sampled_rows: set[int] = set()
    n_source_records = 0
    pairs_at_cut: dict[int, int] = {c: 0 for c in PAIR_CUTS}
    sampled_pairs_at_cut: dict[int, int] = {c: 0 for c in PAIR_CUTS}

    for source in ("source2", "source3"):
        buffer_ids: list[str] = []
        buffer_docs: list[set[str]] = []
        # Raw text is kept only when several views must each re-tokenise it.
        buffer_text: list[tuple[str, str]] = []

        def flush() -> None:
            nonlocal total_pairs, empty_queries, sampled_pairs, sampled_queries
            if not buffer_ids:
                return
            # Retrieve under each view, then merge: a candidate keeps the best
            # (lowest) rank it achieved in any view.
            merged: list[dict[int, int]] = [{} for _ in buffer_ids]
            for view_i, view_fn in enumerate(views):
                docs = (
                    buffer_docs
                    if len(views) == 1
                    else [view_fn(n, a) for n, a in buffer_text]
                )
                query_matrix = build_matrix(
                    docs, vocabs[view_i], idfs[view_i], n_columns=matrices[view_i].shape[1]
                )
                if view_i == 0:
                    empty_queries += int((np.diff(query_matrix.indptr) == 0).sum())
                ids, _scores = topk_matches(
                    query_matrix,
                    matrices[view_i],
                    k=k,
                    chunk_rows=chunk_rows,
                    index_transpose=transposes[view_i],
                )
                for local_row in range(len(buffer_ids)):
                    bucket = merged[local_row]
                    for rank, index_row in enumerate(ids[local_row]):
                        if index_row < 0:
                            continue
                        row = int(index_row)
                        if rank < bucket.get(row, 1 << 30):
                            bucket[row] = rank

            per_row_found = np.array([len(m) for m in merged], dtype=np.int64)
            total_pairs += int(per_row_found.sum())

            # Pairs the union would emit at each smaller k, so one run yields the
            # whole cost curve instead of needing a run per k.
            for cut in PAIR_CUTS:
                if cut > k:
                    continue
                pairs_at_cut[cut] += sum(
                    1 for bucket in merged for rank in bucket.values() if rank < cut
                )
                if sampled_rows:
                    sampled_pairs_at_cut[cut] += sum(
                        1
                        for local_row in sampled_rows
                        for rank in merged[local_row].values()
                        if rank < cut
                    )
            if sampled_rows:
                for local_row in sampled_rows:
                    sampled_pairs += int(per_row_found[local_row])
                sampled_queries += len(sampled_rows)
                sampled_rows.clear()
            if retrieved:
                for local_row, candidate_id in enumerate(buffer_ids):
                    for index_row, rank in merged[local_row].items():
                        entity_id = s1_ids[index_row]
                        bucket = retrieved.get(entity_id)
                        if bucket is not None and rank < bucket.get(candidate_id, 1 << 30):
                            bucket[candidate_id] = rank
            buffer_ids.clear()
            buffer_docs.clear()
            buffer_text.clear()

        for entity_id, name, address in iter_country_rows(files[source], country):
            n_source_records += 1
            if fast_eval and evaluate:
                in_sample = sample_rng.random() < sample_rate
                if entity_id not in must_query and not in_sample:
                    continue
                if in_sample:
                    # Only the sampled rows feed the cost estimate; the
                    # recall-relevant rows are a biased subset by construction.
                    sampled_rows.add(len(buffer_ids))
            buffer_ids.append(entity_id)
            if len(views) == 1:
                buffer_docs.append(views[0](name, address))
            else:
                buffer_text.append((name, address))
            n_queries += 1
            if len(buffer_ids) >= chunk_rows:
                flush()
                if n_queries % (chunk_rows * 10) == 0:
                    elapsed = time.time() - query_started
                    print(
                        f"    {source}: {n_queries:,} queried, {total_pairs:,} pairs, "
                        f"{n_queries / max(elapsed, 1e-9):,.0f} rec/s  [{elapsed:.0f}s]",
                        flush=True,
                    )
        flush()
        print(f"  queried {source}: running total {n_queries:,} records", flush=True)

    query_time = time.time() - query_started

    # In fast mode the measured pairs cover only the queried subset, so the
    # country-wide figures are estimated from the random sample.
    if fast_eval and evaluate and sampled_queries:
        pairs_per_query = sampled_pairs / sampled_queries
        estimated_pairs = int(round(pairs_per_query * n_source_records))
        pairs_curve = {
            str(cut): int(round(sampled_pairs_at_cut[cut] / sampled_queries * n_source_records))
            for cut in PAIR_CUTS
            if cut <= k
        }
        per_query_curve = {
            str(cut): round(sampled_pairs_at_cut[cut] / sampled_queries, 3)
            for cut in PAIR_CUTS
            if cut <= k
        }
    else:
        pairs_per_query = total_pairs / max(n_queries, 1)
        estimated_pairs = total_pairs
        pairs_curve = {str(cut): pairs_at_cut[cut] for cut in PAIR_CUTS if cut <= k}
        per_query_curve = {
            str(cut): round(pairs_at_cut[cut] / max(n_queries, 1), 3)
            for cut in PAIR_CUTS
            if cut <= k
        }

    result = {
        "split": split,
        "country": country,
        "view": view_name,
        "side": "s2s3",
        "k": k,
        "fast_eval": bool(fast_eval),
        "max_df_ratio": max_df_ratio,
        "n_index_s1": n_index,
        "n_query_s2s3": n_source_records,
        "n_queries_run": n_queries,
        "n_tokens": len(vocab),
        "n_tokens_active": n_active,
        "index_nnz": int(index_matrix.nnz),
        "queries_with_no_usable_token": empty_queries,
        "pairs_per_query": round(pairs_per_query, 3),
        "pairs_per_query_by_k": per_query_curve,
        "pairs_by_k": pairs_curve,
        "total_pairs": estimated_pairs,
        "pairs_per_s1": round(estimated_pairs / n_index, 2),
        "index_seconds": round(index_time, 1),
        "query_seconds": round(query_time, 1),
        "total_seconds": round(time.time() - started, 1),
        "peak_rss_gb": peak_rss_gb(),
    }

    if evaluate:
        result.update(_score_recall(retrieved, eval_targets, k, files, country))
    return result


def _score_recall(
    retrieved: dict[str, dict[str, int]],
    eval_targets: dict[str, set[str]],
    k: int,
    files: dict,
    country: str,
) -> dict:
    """Link recall at each k, plus a breakdown of what the misses look like."""
    per_k = {}
    for cut in sorted({1, 2, 3, 5, 10, k}):
        if cut > k:
            continue
        hits = links = 0
        fully = 0
        for entity_id, true_ids in eval_targets.items():
            got = {c for c, rank in retrieved.get(entity_id, {}).items() if rank < cut}
            found = len(true_ids & got)
            hits += found
            links += len(true_ids)
            if found == len(true_ids):
                fully += 1
        per_k[str(cut)] = {
            "link_recall": round(hits / links, 4) if links else 0.0,
            "entities_fully_covered": round(fully / len(eval_targets), 4) if eval_targets else 0.0,
        }

    # Why did the misses miss? Load only the records involved.
    missed: set[str] = set()
    for entity_id, true_ids in eval_targets.items():
        got = set(retrieved.get(entity_id, {}))
        missed.update(true_ids - got)

    reasons = {"empty_address": 0, "cross_script": 0, "other": 0}
    if missed:
        for source in ("source2", "source3"):
            for entity_id, name, address in iter_country_rows(files[source], country):
                if entity_id not in missed:
                    continue
                if not address.strip():
                    reasons["empty_address"] += 1
                elif has_non_latin(name) or has_non_latin(address):
                    reasons["cross_script"] += 1
                else:
                    reasons["other"] += 1

    return {
        "eval_entities": len(eval_targets),
        "eval_links": sum(len(v) for v in eval_targets.values()),
        "recall_by_k": per_k,
        "missed_links": len(missed),
        "miss_reasons": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=("train", "test"))
    parser.add_argument("--country", default="India")
    parser.add_argument(
        "--view",
        default="composite",
        help="one view, or several joined by \"+\" to union their top-k (e.g. composite+name)",
    )
    parser.add_argument("--all-views", action="store_true", help="run every view in turn")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-eval", type=int, default=50000)
    parser.add_argument("--max-df-ratio", type=float, default=DEFAULT_MAX_DF_RATIO)
    parser.add_argument("--chunk-rows", type=int, default=20000)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument(
        "--fast-eval",
        action="store_true",
        help=(
            "query only the recall-relevant records plus a random cost sample; "
            "recall is exact, pair counts are estimated"
        ),
    )
    parser.add_argument("--stats-sample", type=int, default=200_000)
    parser.add_argument(
        "--side",
        default="s2s3",
        choices=("s2s3", "s1"),
        help="which side is queried: s2s3 indexes Source 1, s1 indexes Source 2/3",
    )
    parser.add_argument("--out", default=None, help="append the result to this JSON-lines file")
    args = parser.parse_args()

    views = sorted(VIEWS) if args.all_views else [args.view]
    results = []
    for view_name in views:
        print(
            f"\n=== {args.split} / {args.country} / view={view_name} / "
            f"k={args.k} / side={args.side} ===",
            flush=True,
        )
        if args.side == "s1":
            result = run_probe_s1_side(
                split=args.split,
                country=args.country,
                view_name=view_name,
                k=args.k,
                n_eval=args.n_eval,
                max_df_ratio=args.max_df_ratio,
                chunk_rows=args.chunk_rows,
            )
        else:
            result = run_probe(
                split=args.split,
                country=args.country,
                view_name=view_name,
                k=args.k,
                n_eval=args.n_eval,
                max_df_ratio=args.max_df_ratio,
                chunk_rows=args.chunk_rows,
                evaluate=not args.no_eval,
                fast_eval=args.fast_eval,
                stats_sample=args.stats_sample,
            )
        results.append(result)
        print(json.dumps(result, indent=2))

        out_path = Path(args.out) if args.out else config.ARTIFACTS / "blocking_probe.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")

    if len(results) > 1:
        print("\n=== summary ===")
        print(f"{'view':<12}{'recall@k':>10}{'full-cov':>10}{'pairs':>14}{'pairs/S1':>10}{'sec':>8}{'RSS GB':>8}")
        for r in results:
            top = r.get("recall_by_k", {}).get(str(r["k"]), {})
            print(
                f"{r['view']:<12}{top.get('link_recall', float('nan')):>10.4f}"
                f"{top.get('entities_fully_covered', float('nan')):>10.4f}"
                f"{r['total_pairs']:>14,}{r['pairs_per_s1']:>10.2f}"
                f"{r['total_seconds']:>8.0f}{r['peak_rss_gb']:>8.2f}"
            )


if __name__ == "__main__":
    main()
