"""Checks for the scoring code in ``src.metrics``.

The first case is the worked example from the problem statement; if that does not
reproduce exactly, every offline number we report is meaningless.

Run: ``python scripts/test_metrics.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import entity_fbeta, macro_fbeta, macro_fbeta_report

FAILURES: list[str] = []


def check(label: str, got: float, want: float, tol: float = 1e-3) -> None:
    ok = abs(got - want) <= tol
    print(f"[{'ok ' if ok else 'FAIL'}] {label}: got {got:.6f} want {want:.6f}")
    if not ok:
        FAILURES.append(label)


# The worked example from the problem statement.
check(
    "problem-statement example",
    entity_fbeta(
        ["S2-00047", "S2-00193", "S3-00812"],
        ["S2-00047", "S3-00812"],
    ),
    0.714,
)

# Singleton rule: empty prediction on an empty truth is full credit, any
# prediction on a singleton is a total loss for that entity.
check("singleton predicted empty", entity_fbeta([], []), 1.0)
check("singleton with a false merge", entity_fbeta(["S2-1"], []), 0.0)
check("missed everything", entity_fbeta([], ["S2-1"]), 0.0)
check("exact match", entity_fbeta(["S2-1", "S3-2"], ["S2-1", "S3-2"]), 1.0)
check("no overlap", entity_fbeta(["S2-9"], ["S2-1"]), 0.0)

# Precision is weighted above recall: with the same number of errors, the
# precision-preserving prediction must score higher.
half_recall = entity_fbeta(["S2-1"], ["S2-1", "S2-2"])          # P=1.0, R=0.5
half_precision = entity_fbeta(["S2-1", "S2-9"], ["S2-1"])        # P=0.5, R=1.0
check("precision-preserving (P=1.0, R=0.5)", half_recall, 0.8333)
check("recall-preserving  (P=0.5, R=1.0)", half_precision, 0.5556)
if half_recall <= half_precision:
    FAILURES.append("beta orientation: precision must outweigh recall")
    print("[FAIL] beta orientation: precision must outweigh recall")
else:
    print("[ok ] beta orientation: precision outweighs recall")

# Duplicates in a prediction list must not inflate the score; the scorer
# rejects them outright, and set semantics keep our offline numbers honest.
check("duplicates collapse", entity_fbeta(["S2-1", "S2-1"], ["S2-1"]), 1.0)

# Macro average: an entity absent from the predictions is scored as empty, so a
# missing singleton still earns 1.0 and a missing matched entity earns 0.0.
truth = {"S1-a": ["S2-1"], "S1-b": [], "S1-c": ["S2-2", "S3-3"]}
check(
    "macro with a missing entity",
    macro_fbeta({"S1-a": ["S2-1"]}, truth),
    (1.0 + 1.0 + 0.0) / 3,
)

report = macro_fbeta_report({"S1-a": ["S2-1"], "S1-c": ["S2-2"]}, truth)
check("report singleton slice", report["macro_f0.5_singletons"], 1.0)
check("report micro precision", report["micro_precision"], 1.0)
check("report micro recall", report["micro_recall"], 2 / 3)

print()
if FAILURES:
    print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
    sys.exit(1)
print("all metric checks passed")
