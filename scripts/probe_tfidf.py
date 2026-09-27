"""Plain character n-gram TF-IDF retrieval, with no document-frequency cap.

Our production blocking caps tokens above a document-frequency threshold, which
was a speed decision. An earlier diagnostic showed 2.38% of true links share no
token at all *after* that cap against 0.01% before it, so the cap is a prime
suspect for the recall ceiling that limits the score.

This tests the simplest possible alternative: hash character n-grams of one
concatenated "name + address" string, weight by IDF, and take the nearest Source
1 entities by cosine. No cap, no composite keys, no hand-built views. A
``HashingVectorizer`` is used so a vocabulary of tens of millions of n-grams
never has to be materialised.

Index and queries are both restricted to a subsample so the comparison runs in
minutes; the same subsample is used for every variant *and* for the production
view, so the numbers are directly comparable even though all of them are
optimistic relative to a full-size index.

Usage::

    python scripts/probe_tfidf.py --country India --index-size 300000 --eval-entities 8000
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.normalize import romanized_address_tokens, romanized_tokens

N_FEATURES = 2 ** 20


def combined_text(name: str, address: str) -> str:
    return " ".join(romanized_tokens(name)) + " " + " ".join(romanized_address_tokens(address))


def name_text(name: str, address: str) -> str:
    return " ".join(romanized_tokens(name))


def address_text(name: str, address: str) -> str:
    return " ".join(romanized_address_tokens(address))


VARIANTS = {
    "name+address": combined_text,
    "address only": address_text,
    "name only": name_text,
}


def topk(query: sparse.csr_matrix, index_t: sparse.csr_matrix, k: int, chunk: int = 4000):
    out = np.full((query.shape[0], k), -1, dtype=np.int32)
    scores = np.zeros((query.shape[0], k), dtype=np.float32)
    for start in range(0, query.shape[0], chunk):
        stop = min(start + chunk, query.shape[0])
        block = (query[start:stop] @ index_t).tocsr()
        for row in range(block.shape[0]):
            lo, hi = block.indptr[row], block.indptr[row + 1]
            if lo == hi:
                continue
            cols, vals = block.indices[lo:hi], block.data[lo:hi]
            if vals.size > k:
                part = np.argpartition(vals, vals.size - k)[-k:]
                cols, vals = cols[part], vals[part]
            order = np.argsort(vals)[::-1]
            out[start + row, : cols.size] = cols[order]
            scores[start + row, : vals.size] = vals[order]
    return out, scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", default="India")
    parser.add_argument("--index-size", type=int, default=300_000)
    parser.add_argument("--eval-entities", type=int, default=8000)
    parser.add_argument("--ngram-min", type=int, default=3)
    parser.add_argument("--ngram-max", type=int, default=5)
    args = parser.parse_args()

    from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

    rng = random.Random(config.SEED)
    started = time.time()

    truth: dict[str, list[str]] = {}
    with open(config.TRAIN_FILES["ground_truth"], encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            ids = parts[1] if len(parts) > 1 else ""
            if ids.strip():
                truth[parts[0]] = [x for x in ids.split(",") if x]

    s1: list[tuple[str, str, str]] = []
    with open(config.TRAIN_FILES["source1"], encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4 and parts[3] == args.country:
                s1.append((parts[0], parts[1], parts[2]))
    print(f"{args.country} Source 1: {len(s1):,}")

    with_matches = [r for r in s1 if r[0] in truth]
    eval_entities = rng.sample(with_matches, min(args.eval_entities, len(with_matches)))
    eval_ids = {r[0] for r in eval_entities}

    others = [r for r in s1 if r[0] not in eval_ids]
    rng.shuffle(others)
    index_rows = eval_entities + others[: max(0, args.index_size - len(eval_entities))]
    rng.shuffle(index_rows)
    row_of = {r[0]: i for i, r in enumerate(index_rows)}
    print(f"index: {len(index_rows):,} entities   eval entities: {len(eval_entities):,}")

    wanted = {rec for e in eval_ids for rec in truth[e]}
    partners: list[tuple[str, str, str, str]] = []
    for source in ("source2", "source3"):
        with open(config.TRAIN_FILES[source], encoding="utf-8") as handle:
            handle.readline()
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if parts[0] in wanted:
                    partners.append((parts[0], parts[1], parts[2] if len(parts) > 2 else "", ""))
    owner_of = {rec: e for e in eval_ids for rec in truth[e]}
    print(f"query records (true partners): {len(partners):,}   [{time.time() - started:.0f}s]\n")

    print(f"{'variant':<16}{'nnz/doc':>9}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'R@10':>8}{'build s':>9}{'query s':>9}")
    for label, fn in VARIANTS.items():
        t0 = time.time()
        vectorizer = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(args.ngram_min, args.ngram_max),
            n_features=N_FEATURES,
            norm=None,
            alternate_sign=False,
        )
        index_matrix = vectorizer.transform(fn(n, a) for _e, n, a in index_rows)
        tfidf = TfidfTransformer(sublinear_tf=True)
        index_matrix = tfidf.fit_transform(index_matrix)
        index_t = index_matrix.T.tocsr()
        build_s = time.time() - t0

        t1 = time.time()
        query_matrix = tfidf.transform(
            vectorizer.transform(fn(n, a) for _r, n, a, _x in partners)
        )
        ids, scores = topk(query_matrix, index_t, k=10)
        query_s = time.time() - t1

        truth_rows = np.array(
            [row_of.get(owner_of[r], -1) for r, _n, _a, _x in partners], dtype=np.int32
        )
        hits = ids == truth_rows[:, None]
        recalls = {k: float(hits[:, :k].any(axis=1).mean()) for k in (1, 3, 5, 10)}
        print(
            f"{label:<16}{index_matrix.nnz / index_matrix.shape[0]:>9.0f}"
            f"{recalls[1]:>8.4f}{recalls[3]:>8.4f}{recalls[5]:>8.4f}{recalls[10]:>8.4f}"
            f"{build_s:>9.0f}{query_s:>9.0f}"
        )

        if label == "name+address":
            print("\n  top-1 precision at cosine cut-offs (name+address):")
            top1_correct = hits[:, 0]
            for cut in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
                sel = scores[:, 0] >= cut
                if sel.sum() == 0:
                    continue
                print(
                    f"    cut {cut:.2f}: kept {sel.sum():>7,} of {sel.shape[0]:,} "
                    f"({sel.mean():>6.1%}), top-1 precision {top1_correct[sel].mean():.4f}"
                )
            print()

    print(f"\ntotal seconds: {time.time() - started:.0f}")


if __name__ == "__main__":
    main()
