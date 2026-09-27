"""Recall against candidate-set size, to pick a blocking configuration.

The organisers rank a smaller candidate set per Source 1 entity higher, so
blocking is judged on two axes at once: how many true links it retains, and how
many candidates each Source 1 entity ends up carrying.

Those two are not the same as "pairs per query record". Retrieval runs from the
Source 2/3 side, so a Source 1 entity accumulates one nomination from every
record that shortlists it -- and the nominations pile up unevenly. Total pairs
divided by entities is what the organisers see, and it is far larger than ``k``.

Three levers are swept together:

``k``      candidates each record nominates, per view
``floor``  minimum blocking score for a nomination to survive
``cap``    maximum nominations kept per Source 1 entity, best-scoring first

The floor and the cap attack different halves of the same problem: the floor
removes weak nominations everywhere, the cap protects against a single popular
entity collecting hundreds of them.

Usage::

    python scripts/probe_candidate_budget.py --country India --index-size 200000
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.blocking import TokenVocabulary, build_matrix, topk_matches
from src.keys import VIEWS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", default="India")
    parser.add_argument("--index-size", type=int, default=200_000)
    parser.add_argument("--eval-entities", type=int, default=5000)
    parser.add_argument("--max-k", type=int, default=5)
    args = parser.parse_args()

    rng = random.Random(config.SEED)
    started = time.time()

    truth: dict[str, list[str]] = {}
    with open(config.TRAIN_FILES["ground_truth"], encoding="utf-8") as h:
        h.readline()
        for line in h:
            p = line.rstrip("\n").split("\t")
            ids = p[1] if len(p) > 1 else ""
            if ids.strip():
                truth[p[0]] = [x for x in ids.split(",") if x]

    s1: list[tuple[str, str, str]] = []
    with open(config.TRAIN_FILES["source1"], encoding="utf-8") as h:
        h.readline()
        for line in h:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4 and p[3] == args.country:
                s1.append((p[0], p[1], p[2]))

    with_matches = [r for r in s1 if r[0] in truth]
    ev = rng.sample(with_matches, min(args.eval_entities, len(with_matches)))
    ev_ids = {r[0] for r in ev}
    others = [r for r in s1 if r[0] not in ev_ids]
    rng.shuffle(others)
    index_rows = ev + others[: max(0, args.index_size - len(ev))]
    rng.shuffle(index_rows)
    row_of = {r[0]: i for i, r in enumerate(index_rows)}
    n_index = len(index_rows)

    # Queries: the true partners (for recall) plus a random slice of everything
    # else (so the candidates-per-entity figure reflects a realistic load, not
    # just the records that happen to have an owner here).
    wanted = {rec for e in ev_ids for rec in truth[e]}
    partners: list[tuple[str, str, str]] = []
    ballast: list[tuple[str, str, str]] = []
    keep_rate = 0.05
    for src in ("source2", "source3"):
        with open(config.TRAIN_FILES[src], encoding="utf-8") as h:
            h.readline()
            for line in h:
                p = line.rstrip("\n").split("\t")
                if len(p) < 4 or p[3] != args.country:
                    continue
                if p[0] in wanted:
                    partners.append((p[0], p[1], p[2]))
                elif rng.random() < keep_rate:
                    ballast.append((p[0], p[1], p[2]))
    queries = partners + ballast
    owner = {rec: e for e in ev_ids for rec in truth[e]}
    truth_row = np.array(
        [row_of.get(owner[r], -1) if r in owner else -1 for r, _n, _a in queries],
        dtype=np.int32,
    )
    n_true = int((truth_row >= 0).sum())
    print(
        f"index {n_index:,}   queries {len(queries):,} "
        f"({len(partners):,} true partners, {len(ballast):,} ballast)   "
        f"[{time.time() - started:.0f}s]",
        flush=True,
    )

    # ---- retrieve once at max k, uncapped; every setting is a filter of this --
    view_names = ("composite", "ngram")
    all_ids: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    for vn in view_names:
        view = VIEWS[vn]
        vocab = TokenVocabulary()
        for _e, n, a in index_rows:
            vocab.add_document(view(n, a))
        idf = vocab.finalise(n_index, 1.0, 1)  # uncapped
        matrix = build_matrix((view(n, a) for _e, n, a in index_rows), vocab, idf)
        transpose = matrix.T.tocsr()
        query = build_matrix(
            [view(n, a) for _r, n, a in queries], vocab, idf, n_columns=matrix.shape[1]
        )
        ids, scores = topk_matches(
            query, matrix, k=args.max_k, chunk_rows=4000, index_transpose=transpose
        )
        all_ids.append(ids)
        all_scores.append(scores)
        print(f"  [{vn}] retrieved  [{time.time() - started:.0f}s]", flush=True)

    # Merge views: best score per (query, entity).
    merged_entity = np.concatenate(all_ids, axis=1)
    merged_score = np.concatenate(all_scores, axis=1)

    score_pool = merged_score[merged_entity >= 0]
    print("\nblocking-score distribution over all nominations:")
    for q in (10, 25, 50, 75, 90, 95, 99):
        print(f"  p{q:<3} {np.percentile(score_pool, q):.4f}")
    print()

    def evaluate(k: int, floor: float, cap: int) -> dict:
        ent = merged_entity[:, : k].copy()
        sco = merged_score[:, : k].copy()
        if merged_entity.shape[1] > args.max_k:  # second view block
            ent2 = merged_entity[:, args.max_k : args.max_k + k]
            sco2 = merged_score[:, args.max_k : args.max_k + k]
            ent = np.concatenate([ent, ent2], axis=1)
            sco = np.concatenate([sco, sco2], axis=1)
        valid = (ent >= 0) & (sco >= floor)

        # per-entity nomination counts, and recall
        rows = ent[valid]
        scores = sco[valid]
        if cap > 0 and rows.size:
            order = np.lexsort((-scores, rows))
            rows_sorted = rows[order]
            starts = np.flatnonzero(
                np.concatenate(([True], rows_sorted[1:] != rows_sorted[:-1]))
            )
            ends = np.concatenate((starts[1:], [rows_sorted.shape[0]]))
            keep_positions = np.concatenate(
                [np.arange(s, min(s + cap, e)) for s, e in zip(starts, ends)]
            ) if starts.size else np.empty(0, dtype=np.int64)
            kept_rows = rows_sorted[keep_positions]
            kept_set = np.zeros(rows.shape[0], dtype=bool)
            kept_set[order[keep_positions]] = True
        else:
            kept_rows = rows
            kept_set = np.ones(rows.shape[0], dtype=bool)

        counts = np.bincount(kept_rows, minlength=n_index)
        per_entity_mean = counts.mean()
        per_entity_p90 = np.percentile(counts, 90)

        # recall: is the true owner among this query's surviving nominations?
        flat_query = np.repeat(np.arange(ent.shape[0]), ent.shape[1])[valid.ravel()]
        surviving = kept_set
        hit = np.zeros(ent.shape[0], dtype=bool)
        tq = flat_query[surviving]
        te = rows[surviving]
        want = truth_row[tq]
        good = (want >= 0) & (te == want)
        np.logical_or.at(hit, tq[good], True)
        recall = hit[truth_row >= 0].sum() / max(n_true, 1)

        return {
            "k": k,
            "floor": floor,
            "cap": cap,
            "recall": float(recall),
            "pairs": int(kept_rows.shape[0]),
            "per_entity_mean": float(per_entity_mean),
            "per_entity_p90": float(per_entity_p90),
        }

    floors = [0.0] + [round(float(np.percentile(score_pool, q)), 4) for q in (25, 50, 65, 75)]
    print(f"{'k':>3}{'floor':>9}{'cap':>6}{'recall':>9}{'pairs':>12}{'cand/S1 mean':>14}{'p90':>8}")
    rows_out = []
    for k in (3, 4, 5):
        for floor in floors:
            for cap in (0, 20, 15, 10):
                r = evaluate(k, floor, cap)
                rows_out.append(r)
                print(
                    f"{r['k']:>3}{r['floor']:>9.4f}{r['cap'] if r['cap'] else '-':>6}"
                    f"{r['recall']:>9.4f}{r['pairs']:>12,}"
                    f"{r['per_entity_mean']:>14.1f}{r['per_entity_p90']:>8.0f}",
                    flush=True,
                )

    print(f"\ntotal seconds: {time.time() - started:.0f}")


if __name__ == "__main__":
    main()
