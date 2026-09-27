"""Parallel scoring of a whole split, used for both the holdout eval and the test run.

Measured single-process throughput against a full-country index is about 300
records/second, which puts the ~10M-record test set at roughly nine hours. The
work is embarrassingly parallel per Source 2/3 record, so it is sharded across
processes.

Two decisions are forced by Windows having no ``fork``:

* **Each worker builds its own index** rather than receiving one. Serialising a
  country index (sparse matrices plus a 1.8M-token vocabulary) costs more than
  rebuilding it, and the rebuilds run concurrently, so the wall-clock cost is one
  index build rather than N.
* **Each worker reads its own stride of the source files** (``row % workers ==
  worker_id``). No record is ever pickled; workers touch the same files
  read-only and write their partial results to separate files, which the parent
  merges.

Peak memory is ``workers`` times one index, so ``--workers`` is the memory dial.

Usage::

    python scripts/predict_parallel.py --split test --k 5 --workers 6
    python scripts/predict_parallel.py --split train --country India --workers 6
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys

# Pin the math libraries to one thread per process. With several worker
# processes, the default (one thread per core, per process) oversubscribes
# the machine badly. Thread count does not affect any result here: LightGBM
# prediction parallelises over rows, and scipy's sparse matmul is
# single-threaded regardless.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config

_STATE: dict = {}


def _iter_stride(
    path: Path, country: str, is_s3: bool, worker_id: int, workers: int, limit: int = 0
):
    """Yield this worker's share of one source file, without materialising it."""
    row = 0
    emitted = 0
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4 and parts[3] == country:
                if row % workers == worker_id:
                    yield parts[0], parts[1], parts[2], is_s3
                    emitted += 1
                    if limit and emitted >= limit:
                        return
                row += 1



def _write_shard(path, record_ids, dec_record, dec_entity, dec_prob, dec_margin,
                 pair_entity_blocks, pair_record_blocks, n_records=0,
                 top3=None):
    """Write one shard's results. Used for both checkpoints and the final file.

    Written to a temporary file and renamed, so a crash mid-write cannot leave a
    half-written checkpoint that a later resume would trust.
    """
    payload = {
        "record_index": np.asarray(dec_record, dtype=np.int32),
        "entity_row": np.asarray(dec_entity, dtype=np.int32),
        "probability": np.asarray(dec_prob, dtype=np.float32),
        "margin": np.asarray(dec_margin, dtype=np.float32),
        "record_ids": np.asarray(record_ids),
        "n_records": np.asarray([n_records], dtype=np.int64),
    }
    if pair_entity_blocks:
        payload["pair_entity"] = np.concatenate(pair_entity_blocks)
        payload["pair_record"] = np.concatenate(pair_record_blocks)
    if top3 is not None:
        t_record, t_entity, t_prob, t_rank = top3
        payload["top3_record"] = np.asarray(t_record, dtype=np.int32)
        payload["top3_entity"] = np.asarray(t_entity, dtype=np.int32)
        payload["top3_prob"] = np.asarray(t_prob, dtype=np.float32)
        payload["top3_rank"] = np.asarray(t_rank, dtype=np.int8)
    tmp = Path(str(path) + ".tmp.npz")
    np.savez(tmp, **payload)
    tmp.replace(path)


def _init_worker(split: str, country: str, k: int, max_df_ratio: float, model_path: str) -> None:
    """Build this worker's index and load the model once."""
    import lightgbm as lgb

    from src.candidates import build_country_index
    from src.features import PairFeaturizer

    files = config.TRAIN_FILES if split == "train" else config.TEST_FILES
    index = build_country_index(
        files["source1"], country, max_df_ratio=max_df_ratio, verbose=False
    )
    _STATE["index"] = index
    _STATE["booster"] = lgb.Booster(model_file=model_path)
    _STATE["booster"].params["num_threads"] = 1
    _STATE["featurizer"] = PairFeaturizer(index.token_idf())
    _STATE["k"] = k
    _STATE["files"] = files
    _STATE["country"] = country


def _run_shard(task: tuple[int, int, int, float, bool, str]) -> dict:
    """Score one stride of the records and write partial results to disk."""
    from src.candidates import retrieve, shortlist_context
    from src.inference import iter_chunks

    (worker_id, workers, chunk_rows, keep_floor, want_candidates, out_dir, limit,
     checkpoint_every, expected_records) = task
    index = _STATE["index"]
    booster = _STATE["booster"]
    featurizer = _STATE["featurizer"]
    k = _STATE["k"]
    files = _STATE["files"]
    country = _STATE["country"]

    started = time.time()
    final_path = Path(out_dir) / f"shard_{country}_{worker_id}.npz"
    partial_path = Path(out_dir) / f"shard_{country}_{worker_id}.partial.npz"

    # A finished shard means this worker already completed on an earlier run.
    if final_path.exists():
        print(f"    [w{worker_id}] shard already complete, skipping", flush=True)
        done = np.load(final_path, allow_pickle=False)
        return {
            "worker_id": worker_id,
            "n_records": int(done["record_ids"].shape[0]),
            "n_pairs": int(done["pair_entity"].shape[0]) if "pair_entity" in done else 0,
            "n_decisions": int(done["record_index"].shape[0]),
            "seconds": 0.0,
            "path": str(final_path),
            "resumed": "complete",
        }

    record_ids: list[str] = []
    dec_record: list[int] = []
    dec_entity: list[int] = []
    dec_prob: list[float] = []
    dec_margin: list[float] = []
    top3_record: list[int] = []
    top3_entity: list[int] = []
    top3_prob: list[float] = []
    top3_rank: list[int] = []
    pair_entity_blocks: list[np.ndarray] = []
    pair_record_blocks: list[np.ndarray] = []
    n_records = n_pairs = 0
    skip_records = 0

    # A checkpoint means the worker was interrupted; restore its state and skip
    # the records it already scored, so a restart resumes rather than repeats.
    if partial_path.exists():
        try:
            saved = np.load(partial_path, allow_pickle=False)
            skip_records = int(saved["n_records"][0])
            record_ids = saved["record_ids"].tolist()
            dec_record = saved["record_index"].tolist()
            dec_entity = saved["entity_row"].tolist()
            dec_prob = saved["probability"].tolist()
            dec_margin = saved["margin"].tolist()
            if "top3_record" in saved:
                top3_record = saved["top3_record"].tolist()
                top3_entity = saved["top3_entity"].tolist()
                top3_prob = saved["top3_prob"].tolist()
                top3_rank = saved["top3_rank"].tolist()
            if "pair_entity" in saved:
                pair_entity_blocks = [saved["pair_entity"]]
                pair_record_blocks = [saved["pair_record"]]
                n_pairs = int(saved["pair_entity"].shape[0])
            n_records = skip_records
            print(
                f"    [w{worker_id}] resuming from checkpoint at {skip_records:,} records",
                flush=True,
            )
        except Exception as exc:  # a truncated checkpoint is not fatal
            print(f"    [w{worker_id}] checkpoint unusable ({exc}); starting over", flush=True)
            skip_records = 0
            record_ids, dec_record, dec_entity, dec_prob, dec_margin = [], [], [], [], []
            top3_record, top3_entity, top3_prob, top3_rank = [], [], [], []
            pair_entity_blocks, pair_record_blocks = [], []
            n_records = n_pairs = 0

    def records():
        seen = 0
        for row in _iter_stride(files["source2"], country, False, worker_id, workers, limit):
            seen += 1
            if seen > skip_records:
                yield row
        for row in _iter_stride(files["source3"], country, True, worker_id, workers, limit):
            seen += 1
            if seen > skip_records:
                yield row

    for batch in iter_chunks(records(), chunk_rows):
        base = len(record_ids)
        shortlists = retrieve(index, batch, k=k, chunk_rows=chunk_rows)

        rows: list[list[float]] = []
        owner_entity: list[int] = []
        owner_record: list[int] = []
        for offset, shortlist in enumerate(shortlists):
            record_index = base + offset
            record_ids.append(shortlist.record_id)
            right = featurizer.parse(shortlist.record_id, shortlist.name, shortlist.address)
            for index_row in shortlist.rows():
                left = featurizer.parse(
                    index.entity_ids[index_row],
                    index.names[index_row],
                    index.addresses[index_row],
                )
                rows.append(
                    featurizer.features(left, right, shortlist_context(shortlist, index_row))
                )
                owner_entity.append(index_row)
                owner_record.append(record_index)

        if rows:
            matrix = np.asarray(rows, dtype=np.float32)
            probabilities = booster.predict(
                matrix, num_iteration=booster.best_iteration or None
            )
            del matrix, rows

            if want_candidates:
                pair_entity_blocks.append(np.asarray(owner_entity, dtype=np.int32))
                pair_record_blocks.append(np.asarray(owner_record, dtype=np.int32))
            n_pairs += len(owner_entity)

            best_prob: dict[int, float] = {}
            best_entity: dict[int, int] = {}
            second_prob: dict[int, float] = {}
            for entity_row, record_index, probability in zip(
                owner_entity, owner_record, probabilities
            ):
                probability = float(probability)
                current = best_prob.get(record_index)
                if current is None or probability > current:
                    if current is not None and current > second_prob.get(record_index, 0.0):
                        second_prob[record_index] = current
                    best_prob[record_index] = probability
                    best_entity[record_index] = entity_row
                elif probability > second_prob.get(record_index, 0.0):
                    second_prob[record_index] = probability

            for record_index, probability in best_prob.items():
                if probability < keep_floor:
                    continue
                dec_record.append(record_index)
                dec_entity.append(best_entity[record_index])
                dec_prob.append(probability)
                dec_margin.append(probability - second_prob.get(record_index, 0.0))

            # Diagnostic only: each record's top three candidates. Computed in a
            # separate pass over the same arrays so it cannot influence the
            # assignment above, which is what the submission is built from.
            rec_arr = np.asarray(owner_record, dtype=np.int64)
            ent_arr = np.asarray(owner_entity, dtype=np.int32)
            prob_arr = np.asarray(probabilities, dtype=np.float32)
            order = np.lexsort((-prob_arr, rec_arr))
            rec_sorted = rec_arr[order]
            group_start = np.flatnonzero(
                np.concatenate(([True], rec_sorted[1:] != rec_sorted[:-1]))
            )
            for rank in range(3):
                positions = group_start + rank
                if rank > 0:
                    # keep only positions still inside their own record's group
                    limit_pos = np.concatenate((group_start[1:], [rec_sorted.shape[0]]))
                    positions = positions[positions < limit_pos]
                positions = positions[positions < rec_sorted.shape[0]]
                if positions.size == 0:
                    continue
                chosen = order[positions]
                top3_record.extend(rec_arr[chosen].tolist())
                top3_entity.extend(ent_arr[chosen].tolist())
                top3_prob.extend(prob_arr[chosen].tolist())
                top3_rank.extend([rank] * chosen.shape[0])

        n_records += len(batch)
        featurizer.clear()

        # Progress and checkpoint. On a multi-hour remote run the log is the
        # only way to see whether anything is happening, and a checkpoint means
        # an interrupted run resumes from the last flush rather than from zero.
        if n_records % (chunk_rows * 5) == 0:
            elapsed = time.time() - started
            rate = n_records / max(elapsed, 1e-9)
            if expected_records and rate > 0:
                remaining = max(expected_records - n_records, 0) / rate
                eta = f", {100.0 * n_records / expected_records:.1f}% done, ETA {remaining / 60:.0f} min"
            else:
                eta = ""
            print(
                f"    [w{worker_id}] {n_records:,} records, {n_pairs:,} pairs, "
                f"{rate:,.0f} rec/s{eta}  [{elapsed:.0f}s]",
                flush=True,
            )
        if checkpoint_every and n_records % checkpoint_every == 0:
            _write_shard(
                partial_path,
                record_ids, dec_record, dec_entity, dec_prob, dec_margin,
                pair_entity_blocks if want_candidates else None,
                pair_record_blocks if want_candidates else None,
                n_records=n_records,
                top3=(top3_record, top3_entity, top3_prob, top3_rank),
            )

    out_path = final_path
    _write_shard(
        out_path, record_ids, dec_record, dec_entity, dec_prob, dec_margin,
        pair_entity_blocks if want_candidates else None,
        pair_record_blocks if want_candidates else None,
        n_records=n_records,
        top3=(top3_record, top3_entity, top3_prob, top3_rank),
    )
    if partial_path.exists():
        partial_path.unlink()

    return {
        "worker_id": worker_id,
        "n_records": n_records,
        "n_pairs": n_pairs,
        "n_decisions": len(dec_record),
        "seconds": round(time.time() - started, 1),
        "path": str(out_path),
    }



def choose_workers(requested: str, index_gb_estimate: float = 2.0) -> int:
    """Pick a worker count from the machine's cores and free memory.

    Under ``fork`` the index is shared, so memory per worker is small and the
    core count governs. Under ``spawn`` each worker holds its own index, so
    memory governs -- which is exactly the constraint that made the local
    Windows run swap.
    """
    if requested != "auto":
        return max(1, int(requested))

    cores = os.cpu_count() or 4
    try:
        import psutil

        free_gb = psutil.virtual_memory().available / 1e9
    except Exception:
        free_gb = 8.0

    if "fork" in mp.get_all_start_methods():
        # Shared index: leave a couple of cores for the parent and the OS.
        workers = max(1, min(cores - 1, 30))
        # Each worker still holds its own chunk buffers and feature block.
        workers = max(1, min(workers, int(free_gb // 1.5)))
    else:
        workers = max(1, min(cores - 1, int(free_gb // index_gb_estimate)))
    print(
        f"auto workers: {workers} (cores={cores}, free={free_gb:.1f} GB, "
        f"start_method={'fork' if 'fork' in mp.get_all_start_methods() else 'spawn'})"
    )
    return workers


def discover_countries(path: Path) -> list[str]:
    counts: Counter = Counter()
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                counts[parts[3]] += 1
    return [c for c, _n in counts.most_common()]


def score_country(
    split: str,
    country: str,
    k: int,
    workers: int,
    chunk_rows: int,
    keep_floor: float,
    max_df_ratio: float,
    model_path: str,
    want_candidates: bool,
    out_dir: Path,
    limit: int = 0,
    checkpoint_every: int = 0,
    expected_per_worker: int = 0,
) -> list[dict]:
    """Run one country across ``workers`` processes; return the shard summaries.

    Where ``fork`` is available (Linux) the index is built **once** in the parent
    and inherited copy-on-write, so N workers cost one index build and roughly
    one index of memory. Under ``spawn`` (Windows) nothing can be inherited, so
    each worker builds its own -- correct but N times the memory, which is what
    made the local run swap.
    """
    tasks = [
        (worker_id, workers, chunk_rows, keep_floor, want_candidates, str(out_dir), limit,
         checkpoint_every, expected_per_worker)
        for worker_id in range(workers)
    ]

    use_fork = "fork" in mp.get_all_start_methods()
    if use_fork:
        started = time.time()
        _init_worker(split, country, k, max_df_ratio, model_path)
        print(
            f"  index built once in parent, shared by fork "
            f"({_STATE['index'].size:,} entities)  [{time.time() - started:.0f}s]",
            flush=True,
        )
        context = mp.get_context("fork")
        with context.Pool(processes=workers) as pool:
            return list(pool.imap_unordered(_run_shard, tasks))

    context = mp.get_context("spawn")
    with context.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(split, country, k, max_df_ratio, model_path),
    ) as pool:
        return list(pool.imap_unordered(_run_shard, tasks))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--country", default=None, help="a single country")
    parser.add_argument("--countries", nargs="+", default=None,
                        help="explicit ordered list, e.g. France US India (smallest first)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--workers", default="auto",
                        help="number of worker processes, or 'auto' to size from cores/RAM")
    parser.add_argument("--chunk-rows", type=int, default=20000)
    parser.add_argument("--keep-floor", type=float, default=0.02)
    parser.add_argument("--max-df-ratio", type=float, default=0.002)
    parser.add_argument("--no-candidates", action="store_true")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=200_000,
                        help="records between per-worker checkpoints; 0 disables")
    parser.add_argument("--limit", type=int, default=0,
                        help="records per worker per source file; 0 = all (smoke testing)")
    args = parser.parse_args()

    config.ensure_dirs()
    started = time.time()

    model_path = str(Path(args.model) if args.model else config.ARTIFACTS / "matcher.txt")
    files = config.TRAIN_FILES if args.split == "train" else config.TEST_FILES
    out_dir = Path(args.out_dir) if args.out_dir else config.ARTIFACTS / f"shards_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    workers = choose_workers(str(args.workers))
    if args.countries:
        countries = args.countries
    elif args.country:
        countries = [args.country]
    else:
        countries = discover_countries(files["source1"])
    print(f"split={args.split}  countries={countries}  workers={workers}  k={args.k}")
    print(f"model={model_path}\nshards -> {out_dir}\n", flush=True)

    for country in countries:
        country_started = time.time()
        print(f"=== {country} ===", flush=True)
        # One cheap pass to size the work, so each worker can report a real ETA
        # rather than just a rate.
        n_country_records = 0
        for source in ("source2", "source3"):
            with open(files[source], encoding="utf-8") as handle:
                handle.readline()
                for line in handle:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 4 and parts[3] == country:
                        n_country_records += 1
        expected_per_worker = n_country_records // max(workers, 1)
        print(
            f"  {n_country_records:,} Source 2/3 records "
            f"(~{expected_per_worker:,} per worker)",
            flush=True,
        )
        summaries = score_country(
            args.split,
            country,
            args.k,
            workers,
            args.chunk_rows,
            args.keep_floor,
            args.max_df_ratio,
            model_path,
            not args.no_candidates,
            out_dir,
            args.limit,
            args.checkpoint_every,
            expected_per_worker,
        )
        total_records = sum(s["n_records"] for s in summaries)
        total_pairs = sum(s["n_pairs"] for s in summaries)
        total_decisions = sum(s["n_decisions"] for s in summaries)
        elapsed = time.time() - country_started
        print(
            f"  {total_records:,} records, {total_pairs:,} pairs, "
            f"{total_decisions:,} decisions kept, "
            f"{total_records / max(elapsed, 1e-9):,.0f} rec/s aggregate  [{elapsed:.0f}s]",
            flush=True,
        )
        for s in sorted(summaries, key=lambda d: d["worker_id"]):
            print(
                f"    worker {s['worker_id']}: {s['n_records']:,} records, "
                f"{s['seconds']:.0f}s"
            )

    print(f"\ntotal seconds: {time.time() - started:.0f}")


if __name__ == "__main__":
    main()
