"""Turn the blocking probe's JSON-lines log into ``reports/blocking.md``.

Reads every run recorded in ``artifacts/blocking_probe.jsonl`` and renders the
comparison table, the miss breakdown, and the extrapolation from the measured
training-country runs to the full test set.

Run: ``python scripts/render_blocking_report.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config

# Test-set sizes, from reports/eda.md. Used only to extrapolate the pair counts
# measured on the training countries.
TEST_SIZES = {
    "France": {"s1": 259_452, "s2s3": 703_378 + 731_615},
    "India": {"s1": 809_986, "s2s3": 2_312_565 + 2_405_000},
    "US": {"s1": 663_106, "s2s3": 1_871_330 + 1_945_701},
}
TEST_TOTAL_S1 = sum(v["s1"] for v in TEST_SIZES.values())
TEST_TOTAL_S2S3 = sum(v["s2s3"] for v in TEST_SIZES.values())


def load_runs(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"no probe log at {path}; run scripts/probe_blocking.py first")
    runs = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    return runs


def render(runs: list[dict]) -> str:
    lines = [
        "# Candidate generation",
        "",
        "Built by `scripts/render_blocking_report.py` from the runs logged in",
        "`artifacts/blocking_probe.jsonl`. Each run indexes every Source 1 record of one",
        "country and queries every Source 2/3 record of that country, so pair counts and",
        "runtimes carry over to the test set directly.",
        "",
        "## Runs",
        "",
        "| split | country | view | side | k | link recall | entities fully covered | pairs | pairs/S1 | sec | peak RSS |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for run in runs:
        side = run.get("side", "s2s3")
        top = run.get("recall_by_k", {}).get(str(run["k"]), {})
        pairs = run.get("total_pairs", run.get("total_pairs_scaled", 0))
        recall = top.get("link_recall")
        covered = top.get("entities_fully_covered")
        lines.append(
            f"| {run['split']} | {run['country']} | {run['view']} | {side} | {run['k']} | "
            f"{recall if recall is None else f'{recall:.4f}'} | "
            f"{covered if covered is None else f'{covered:.4f}'} | "
            f"{pairs:,} | {run.get('pairs_per_s1', 0):.2f} | "
            f"{run.get('total_seconds', 0):.0f} | {run.get('peak_rss_gb', float('nan')):.2f} GB |"
        )

    lines += ["", "## Recall as k varies", ""]
    for run in runs:
        by_k = run.get("recall_by_k")
        if not by_k:
            continue
        side = run.get("side", "s2s3")
        lines += [
            f"**{run['split']} / {run['country']} / {run['view']} / side={side}**",
            "",
            "| k | link recall | entities fully covered |",
            "| ---: | ---: | ---: |",
        ]
        for cut in sorted(by_k, key=int):
            row = by_k[cut]
            lines.append(
                f"| {cut} | {row['link_recall']:.4f} | {row['entities_fully_covered']:.4f} |"
            )
        lines.append("")

    lines += ["## Where the misses come from", ""]
    any_misses = False
    for run in runs:
        reasons = run.get("miss_reasons")
        if not reasons:
            continue
        any_misses = True
        total = sum(reasons.values()) or 1
        side = run.get("side", "s2s3")
        lines += [
            f"**{run['split']} / {run['country']} / {run['view']} / side={side}** — "
            f"{run.get('missed_links', 0):,} missed links of {run.get('eval_links', 0):,}",
            "",
            "| reason | links | share of misses |",
            "| --- | ---: | ---: |",
        ]
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {reason} | {count:,} | {count / total:.1%} |")
        lines.append("")
    if not any_misses:
        lines += ["No miss breakdown recorded.", ""]

    lines += [
        "## Extrapolation to the test set",
        "",
        f"Test set: **{TEST_TOTAL_S1:,}** Source 1 entities and **{TEST_TOTAL_S2S3:,}** "
        "Source 2/3 records across three countries.",
        "",
        "Source 2/3-side retrieval produces one shortlist of `k` per Source 2/3 record, so the "
        "total pair count is bounded by `|S2| + |S3|` times `k` no matter how the entities are "
        "distributed:",
        "",
        "| k | bounded test pairs |",
        "| ---: | ---: |",
    ]
    for k in (1, 2, 3, 5, 10):
        lines.append(f"| {k} | {TEST_TOTAL_S2S3 * k:,} |")
    lines.append("")

    measured = [r for r in runs if r.get("side", "s2s3") == "s2s3" and r.get("n_query_s2s3")]
    if measured:
        lines += [
            "Measured pairs per query record, and the implied test-set totals:",
            "",
            "| country | view | k | pairs/query | implied test pairs |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        for run in measured:
            per_query = run["total_pairs"] / max(run["n_query_s2s3"], 1)
            lines.append(
                f"| {run['country']} | {run['view']} | {run['k']} | {per_query:.2f} | "
                f"{int(per_query * TEST_TOTAL_S2S3):,} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    runs = load_runs(config.ARTIFACTS / "blocking_probe.jsonl")
    out = config.REPO_ROOT / "reports" / "blocking.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(runs), encoding="utf-8")
    print(f"wrote {out} ({len(runs)} run(s))")


if __name__ == "__main__":
    main()
