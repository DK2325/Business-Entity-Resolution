"""Loading and writing the challenge TSVs.

Two rules govern every read here, and getting either wrong corrupts the data
silently rather than raising:

* ``sep="\t"`` — addresses and ID lists both contain commas, so a comma-parsed
  read yields one column holding the whole line.
* ``dtype=str, keep_default_na=False`` — entity IDs and names must stay strings,
  and a business genuinely named "NA" (or an empty address) must not become NaN.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def read_source(path: str | Path, usecols: list[str] | None = None) -> pd.DataFrame:
    """Read one ``*_source{1,2,3}.tsv`` into a string-typed frame."""
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        usecols=usecols,
    )


def iter_source(path: str | Path, chunksize: int = 500_000) -> Iterator[pd.DataFrame]:
    """Stream a source file in chunks.

    The test sources hold ~5M rows each; with the memory budget on this machine
    the blocking and feature stages read them chunk-wise rather than whole.
    """
    yield from pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        chunksize=chunksize,
    )


def read_ground_truth(path: str | Path) -> dict[str, list[str]]:
    """Read ``train_ground_truth.tsv`` into ``{s1_id: [matched ids]}``.

    Entities with no matches map to an empty list; they are the singletons, and
    dropping them would silently remove 5.6% of the evaluation set.
    """
    truth: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as handle:
        header = handle.readline()
        if not header.startswith("source1_entity_id"):
            raise ValueError(f"unexpected ground-truth header: {header!r}")
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            ids = parts[1] if len(parts) > 1 else ""
            truth[parts[0]] = [x for x in ids.split(",") if x]
    return truth


def write_id_lists(path: str | Path, rows: dict[str, list[str]], value_column: str) -> None:
    """Write a submission-shaped TSV: one row per S1 entity, comma-joined IDs.

    Used for both ``matching_results.tsv`` (``matched_entity_ids``) and
    ``candidate_pairs.tsv`` (``candidate_entity_ids``). Duplicates within a list
    are dropped while preserving order, since they are a rejection condition.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(f"source1_entity_id\t{value_column}\n")
        for s1_id, ids in rows.items():
            seen: dict[str, None] = {}
            for i in ids:
                seen.setdefault(i, None)
            handle.write(f"{s1_id}\t{','.join(seen)}\n")
