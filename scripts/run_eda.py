"""Exploratory data analysis over the challenge dataset.

Every number quoted in ``reports/eda.md`` is produced here, so the report can be
regenerated from the raw TSVs at any time.

The script runs in four independent sections, each cached as JSON under
``artifacts/`` so an expensive full-file scan is paid for once:

``counts``         row counts, file sizes and country mixes (streams every file)
``truth``          ground-truth structure: singletons, exclusivity, distractors
``dupes``          how alike the S2 and S3 records inside one S1 group are
``separability``   true pairs vs hard negatives, by name and address similarity

Usage::

    python scripts/run_eda.py                  # all sections, reusing caches
    python scripts/run_eda.py --section truth  # one section
    python scripts/run_eda.py --force          # ignore caches and recompute
    python scripts/run_eda.py --render-only    # rebuild the markdown from caches

Memory: the full scans stream line by line and hold only counters, and the two
sampled sections cap their record pools, keeping peak usage well inside the
budget for this machine.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rapidfuzz import fuzz

from src import config
from src.normalize import (
    address_tokens,
    name_core,
    numeric_keys,
    romanized_address_tokens,
    romanized_tokens,
)

SEED = config.SEED
SECTIONS = ("counts", "truth", "dupes", "separability")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def cache_path(section: str) -> Path:
    return config.ARTIFACTS / f"eda_{section}.json"


def load_cache(section: str) -> dict | None:
    path = cache_path(section)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def save_cache(section: str, payload: dict) -> None:
    config.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    cache_path(section).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def iter_rows(path: Path):
    """Yield ``[entity_id, name, address, country]`` for a source TSV.

    Splits on tabs directly rather than going through pandas: these files are up
    to 5.3M rows and we only ever need a streaming pass over four string fields.
    """
    with open(path, encoding="utf-8") as handle:
        handle.readline()  # header
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                yield parts[:4]
            elif parts and parts[0]:
                # Defensive: a short row still has an ID we must not lose.
                yield (parts + ["", "", ""])[:4]


def read_truth(path: Path) -> dict[str, list[str]]:
    truth: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            ids = parts[1] if len(parts) > 1 else ""
            truth[parts[0]] = [x for x in ids.split(",") if x]
    return truth


def percentiles(values: list[float], points=(5, 25, 50, 75, 95)) -> dict[str, float]:
    if not values:
        return {f"p{p}": float("nan") for p in points}
    ordered = sorted(values)
    out = {}
    for p in points:
        idx = min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))
        out[f"p{p}"] = round(ordered[idx], 4)
    out["mean"] = round(statistics.fmean(ordered), 4)
    return out


# ---------------------------------------------------------------------------
# section: counts
# ---------------------------------------------------------------------------

def section_counts() -> dict:
    """Row counts, file sizes and country mix for all seven TSVs."""
    result: dict[str, dict] = {}
    files = {f"train_{k}": v for k, v in config.TRAIN_FILES.items()}
    files.update({f"test_{k}": v for k, v in config.TEST_FILES.items()})

    for label, path in files.items():
        size_mb = round(path.stat().st_size / 1e6, 1)
        if label.endswith("ground_truth"):
            rows = sum(1 for _ in open(path, encoding="utf-8")) - 1
            result[label] = {"rows": rows, "size_mb": size_mb}
            continue
        countries: collections.Counter = collections.Counter()
        empty_address = 0
        empty_name = 0
        rows = 0
        for _eid, name, address, country in iter_rows(path):
            rows += 1
            countries[country] += 1
            if not address.strip():
                empty_address += 1
            if not name.strip():
                empty_name += 1
        result[label] = {
            "rows": rows,
            "size_mb": size_mb,
            "countries": dict(countries.most_common()),
            "empty_address": empty_address,
            "empty_address_pct": round(100 * empty_address / rows, 2) if rows else 0.0,
            "empty_name": empty_name,
        }
        print(f"  {label}: {rows:,} rows, {size_mb} MB, empty addr {empty_address:,}")
    return result


# ---------------------------------------------------------------------------
# section: truth
# ---------------------------------------------------------------------------

def section_truth() -> dict:
    """Structure of the ground truth.

    The exclusivity question is the important one: if no Source 2/3 record is
    ever claimed by two Source 1 entities, matching is a constrained assignment
    and the decision rule can force candidates to compete.
    """
    truth = read_truth(config.TRAIN_FILES["ground_truth"])

    size_hist: collections.Counter = collections.Counter()
    s2_hist: collections.Counter = collections.Counter()
    s3_hist: collections.Counter = collections.Counter()
    claim_count: collections.Counter = collections.Counter()

    for ids in truth.values():
        size_hist[len(ids)] += 1
        s2_hist[sum(1 for i in ids if i.startswith("S2-"))] += 1
        s3_hist[sum(1 for i in ids if i.startswith("S3-"))] += 1
        for i in ids:
            claim_count[i] += 1

    total_links = sum(claim_count.values())
    multi_claimed = sum(1 for v in claim_count.values() if v > 1)

    # Country agreement across true links, and distractor share per source.
    s1_country = {}
    for eid, _n, _a, country in iter_rows(config.TRAIN_FILES["source1"]):
        s1_country[eid] = country

    country_mismatch = 0
    links_checked = 0
    distractors = {}
    # One pass per source: count distractors and, for the records that do appear
    # in the truth, keep their country for the agreement check below.
    link_country: dict[str, str] = {}
    for src, path in (("S2", config.TRAIN_FILES["source2"]), ("S3", config.TRAIN_FILES["source3"])):
        total = matched = 0
        for eid, _n, _a, country in iter_rows(path):
            total += 1
            if eid in claim_count:
                matched += 1
                link_country[eid] = country
        distractors[src] = {
            "total": total,
            "matched": matched,
            "distractors": total - matched,
            "distractor_pct": round(100 * (total - matched) / total, 2) if total else 0.0,
        }
        print(f"  {src}: {total:,} rows, {total - matched:,} distractors")

    for s1_id, ids in truth.items():
        home = s1_country.get(s1_id)
        for i in ids:
            other = link_country.get(i)
            if other is None:
                continue
            links_checked += 1
            if other != home:
                country_mismatch += 1

    n_entities = len(truth)
    singletons = size_hist[0]
    return {
        "n_s1_entities": n_entities,
        "singletons": singletons,
        "singleton_pct": round(100 * singletons / n_entities, 2),
        "total_links": total_links,
        "mean_matches_per_entity": round(total_links / n_entities, 3),
        "match_count_hist": {str(k): v for k, v in sorted(size_hist.items())},
        "s2_per_entity_hist": {str(k): v for k, v in sorted(s2_hist.items())},
        "s3_per_entity_hist": {str(k): v for k, v in sorted(s3_hist.items())},
        "distinct_linked_ids": len(claim_count),
        "ids_claimed_by_multiple_s1": multi_claimed,
        "exclusivity_holds": multi_claimed == 0,
        "links_checked_for_country": links_checked,
        "country_mismatches": country_mismatch,
        "distractors": distractors,
    }


# ---------------------------------------------------------------------------
# sampling support for the two similarity sections
# ---------------------------------------------------------------------------

def _sample_groups(n_groups: int, rng: random.Random, non_singleton_only: bool = True):
    """Pick Source 1 entities at random and load every record in their groups."""
    truth = read_truth(config.TRAIN_FILES["ground_truth"])
    keys = [k for k, v in truth.items() if v] if non_singleton_only else list(truth)
    chosen = rng.sample(keys, min(n_groups, len(keys)))
    groups = {k: truth[k] for k in chosen}

    wanted = set(groups)
    for ids in groups.values():
        wanted.update(ids)

    records: dict[str, tuple[str, str, str]] = {}
    for path in (
        config.TRAIN_FILES["source1"],
        config.TRAIN_FILES["source2"],
        config.TRAIN_FILES["source3"],
    ):
        for eid, name, address, country in iter_rows(path):
            if eid in wanted:
                records[eid] = (name, address, country)
    return groups, records


def _pair_features(rec_a: tuple[str, str, str], rec_b: tuple[str, str, str]) -> dict[str, float]:
    """The similarity signals we are deciding whether to build features from."""
    name_a, addr_a, _ = rec_a
    name_b, addr_b, _ = rec_b

    core_a, core_b = set(name_core(name_a)), set(name_core(name_b))
    rom_a, rom_b = set(romanized_tokens(name_a)), set(romanized_tokens(name_b))
    tok_a, tok_b = set(address_tokens(addr_a)), set(address_tokens(addr_b))
    rtok_a, rtok_b = set(romanized_address_tokens(addr_a)), set(romanized_address_tokens(addr_b))
    num_a, num_b = numeric_keys(rtok_a), numeric_keys(rtok_b)

    def jaccard(x: set, y: set) -> float:
        if not x and not y:
            return float("nan")
        union = x | y
        return len(x & y) / len(union) if union else float("nan")

    return {
        "name_jaccard": jaccard(core_a, core_b),
        "name_jaccard_romanized": jaccard(rom_a, rom_b),
        "name_token_set_ratio": fuzz.token_set_ratio(" ".join(sorted(rom_a)), " ".join(sorted(rom_b))) / 100,
        "addr_jaccard": jaccard(tok_a, tok_b),
        "addr_jaccard_romanized": jaccard(rtok_a, rtok_b),
        "addr_numeric_jaccard": jaccard(num_a, num_b),
        "addr_empty_either": float(not addr_a.strip() or not addr_b.strip()),
    }


# ---------------------------------------------------------------------------
# section: dupes
# ---------------------------------------------------------------------------

def section_dupes(n_groups: int = 20000) -> dict:
    """How similar are the Source 2 and Source 3 records inside one group?

    If an S2 record and an S3 record describing the same business are usually
    near-identical, then a match found on one side can be propagated to the
    other, which is a cheap recall gain for records the blocker would otherwise
    miss (for instance those with an empty address).
    """
    rng = random.Random(SEED)
    groups, records = _sample_groups(n_groups, rng)

    cross_best: list[float] = []
    exact_norm_dupes = 0
    groups_with_cross = 0
    within_s2: list[float] = []
    within_s3: list[float] = []

    for _s1_id, ids in groups.items():
        s2 = [records[i] for i in ids if i.startswith("S2-") and i in records]
        s3 = [records[i] for i in ids if i.startswith("S3-") and i in records]

        if s2 and s3:
            groups_with_cross += 1
            best = 0.0
            for a in s2:
                for b in s3:
                    feats = _pair_features(a, b)
                    name_sim = feats["name_token_set_ratio"]
                    addr_sim = feats["addr_jaccard_romanized"]
                    combined = name_sim if addr_sim != addr_sim else (name_sim + addr_sim) / 2
                    best = max(best, combined)
                    key_a = (tuple(sorted(romanized_tokens(a[0]))), tuple(sorted(romanized_address_tokens(a[1]))))
                    key_b = (tuple(sorted(romanized_tokens(b[0]))), tuple(sorted(romanized_address_tokens(b[1]))))
                    if key_a == key_b:
                        exact_norm_dupes += 1
            cross_best.append(best)

        # Same-source siblings: two S2 records under one S1 are two renderings of
        # the same business, which is the clearest picture of intra-source noise.
        for bucket, pool in ((within_s2, s2), (within_s3, s3)):
            for i in range(len(pool)):
                for j in range(i + 1, len(pool)):
                    feats = _pair_features(pool[i], pool[j])
                    bucket.append(feats["name_token_set_ratio"])

    return {
        "groups_sampled": len(groups),
        "groups_with_both_sources": groups_with_cross,
        "cross_source_best_similarity": percentiles(cross_best),
        "share_groups_with_near_identical_cross_pair": round(
            sum(1 for v in cross_best if v >= 0.9) / len(cross_best), 4
        )
        if cross_best
        else 0.0,
        "exact_normalized_duplicate_pairs": exact_norm_dupes,
        "within_s2_name_similarity": percentiles(within_s2),
        "within_s3_name_similarity": percentiles(within_s3),
    }


# ---------------------------------------------------------------------------
# section: separability
# ---------------------------------------------------------------------------

def section_separability(n_groups: int = 6000, pool_size: int = 400000, top_negatives: int = 5) -> dict:
    """Similarity of true pairs against the hardest look-alikes we can find.

    A hard negative here is a same-country record that shares at least one
    address or name token with the Source 1 record but is not a true match --
    the kind of pair the classifier actually has to separate. Comparing the two
    distributions tells us which features carry signal before we build any.
    """
    rng = random.Random(SEED + 1)
    groups, records = _sample_groups(n_groups, rng)

    s1_records = {k: records[k] for k in groups if k in records}

    # Build a modest pool of same-country records to mine negatives from.
    wanted_countries = {v[2] for v in s1_records.values()}
    truth_ids = {i for ids in groups.values() for i in ids}
    pool: list[tuple[str, tuple[str, str, str]]] = []
    keep_prob = 0.06
    for path in (config.TRAIN_FILES["source2"], config.TRAIN_FILES["source3"]):
        for eid, name, address, country in iter_rows(path):
            if country not in wanted_countries:
                continue
            if eid in truth_ids or rng.random() < keep_prob:
                pool.append((eid, (name, address, country)))
                if len(pool) >= pool_size + len(truth_ids):
                    break

    # Inverted index over address and name tokens, per country, with a frequency
    # cap so ubiquitous tokens (city names) do not dominate the negative pool.
    index: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for idx, (_eid, (name, address, country)) in enumerate(pool):
        for tok in set(romanized_address_tokens(address)) | set(romanized_tokens(name)):
            index[(country, tok)].append(idx)
    index = {k: v for k, v in index.items() if len(v) <= 3000}

    true_feats: list[dict[str, float]] = []
    neg_feats: list[dict[str, float]] = []

    for s1_id, ids in groups.items():
        rec = s1_records.get(s1_id)
        if rec is None:
            continue
        for i in ids:
            if i in records:
                true_feats.append(_pair_features(rec, records[i]))

        country = rec[2]
        keys = set(romanized_address_tokens(rec[1])) | set(romanized_tokens(rec[0]))
        hits: collections.Counter = collections.Counter()
        for tok in keys:
            for idx in index.get((country, tok), ()):  # noqa: B038
                hits[idx] += 1
        truth_set = set(ids)
        ranked = [idx for idx, _ in hits.most_common(top_negatives * 4) if pool[idx][0] not in truth_set]
        for idx in ranked[:top_negatives]:
            neg_feats.append(_pair_features(rec, pool[idx][1]))

    def summarise(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        if not rows:
            return out
        for key in rows[0]:
            values = [r[key] for r in rows if r[key] == r[key]]  # drop NaN
            out[key] = percentiles(values)
        return out

    # Separation: how far apart the medians are, in units that survive a quick read.
    true_summary = summarise(true_feats)
    neg_summary = summarise(neg_feats)
    separation = {}
    for key in true_summary:
        if key in neg_summary:
            separation[key] = round(true_summary[key]["p50"] - neg_summary[key]["p50"], 4)

    return {
        "groups_sampled": len(groups),
        "pool_size": len(pool),
        "true_pairs": len(true_feats),
        "hard_negative_pairs": len(neg_feats),
        "true_pair_similarity": true_summary,
        "hard_negative_similarity": neg_summary,
        "median_separation": separation,
    }


# ---------------------------------------------------------------------------
# markdown rendering
# ---------------------------------------------------------------------------

def render(results: dict[str, dict]) -> str:
    lines: list[str] = ["# Exploratory data analysis", ""]
    lines.append("Regenerate with `python scripts/run_eda.py`. Every figure below comes from that script.")
    lines.append("")

    counts = results.get("counts")
    if counts:
        lines += ["## 1. Files", "", "| file | rows | size (MB) | empty address |", "| --- | ---: | ---: | ---: |"]
        for label, info in counts.items():
            empty = f"{info.get('empty_address', 0):,} ({info.get('empty_address_pct', 0)}%)" if "empty_address" in info else "-"
            lines.append(f"| {label} | {info['rows']:,} | {info['size_mb']} | {empty} |")
        lines += ["", "### Country mix", "", "| file | " + " | ".join(
            sorted({c for i in counts.values() for c in i.get("countries", {})})
        ) + " |"]
        cols = sorted({c for i in counts.values() for c in i.get("countries", {})})
        lines.append("| --- |" + " ---: |" * len(cols))
        for label, info in counts.items():
            if "countries" not in info:
                continue
            row = " | ".join(f"{info['countries'].get(c, 0):,}" for c in cols)
            lines.append(f"| {label} | {row} |")
        lines.append("")

    truth = results.get("truth")
    if truth:
        lines += [
            "## 2. Ground-truth structure",
            "",
            f"- Source 1 entities: **{truth['n_s1_entities']:,}**",
            f"- Singletons: **{truth['singletons']:,} ({truth['singleton_pct']}%)** — each worth a full 1.0 if predicted empty",
            f"- Total links: **{truth['total_links']:,}**, mean **{truth['mean_matches_per_entity']}** per entity",
            f"- Distinct S2/S3 IDs appearing in the truth: **{truth['distinct_linked_ids']:,}**",
            "",
            "### Exclusivity",
            "",
            f"IDs claimed by more than one Source 1 entity: **{truth['ids_claimed_by_multiple_s1']:,}** "
            f"of {truth['distinct_linked_ids']:,}.",
            "",
            f"> **Exclusivity {'holds' if truth['exclusivity_holds'] else 'does NOT hold'}.** "
            "Every Source 2/3 record belongs to at most one Source 1 entity, so matching is a "
            "constrained assignment: candidates can be made to compete and only the best claim kept. "
            "This is the backbone of the decision rule.",
            "",
            "### Country agreement",
            "",
            f"Country mismatches across true links: **{truth['country_mismatches']:,}** of "
            f"{truth['links_checked_for_country']:,} — country is a safe hard blocking key.",
            "",
            "### Distractors",
            "",
            "| source | rows | matched | distractors |",
            "| --- | ---: | ---: | ---: |",
        ]
        for src, info in truth["distractors"].items():
            lines.append(
                f"| {src} | {info['total']:,} | {info['matched']:,} | "
                f"{info['distractors']:,} ({info['distractor_pct']}%) |"
            )
        lines += ["", "### Matches per entity", "", "| matches | entities |", "| ---: | ---: |"]
        for k, v in truth["match_count_hist"].items():
            lines.append(f"| {k} | {v:,} |")
        lines.append("")

    dupes = results.get("dupes")
    if dupes:
        lines += [
            "## 3. Source 2 vs Source 3 inside one group",
            "",
            f"Sampled **{dupes['groups_sampled']:,}** non-singleton groups; "
            f"{dupes['groups_with_both_sources']:,} contain records from both sources.",
            "",
            "Best cross-source similarity within a group (name token-set ratio, averaged with "
            "address Jaccard where both addresses exist):",
            "",
            "| p5 | p25 | p50 | p75 | p95 | mean |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        cs = dupes["cross_source_best_similarity"]
        lines.append(
            f"| {cs['p5']} | {cs['p25']} | {cs['p50']} | {cs['p75']} | {cs['p95']} | {cs['mean']} |"
        )
        lines += [
            "",
            f"- Groups containing a near-identical (>= 0.9) S2/S3 pair: "
            f"**{dupes['share_groups_with_near_identical_cross_pair']:.1%}**",
            f"- Pairs identical after normalisation: **{dupes['exact_normalized_duplicate_pairs']:,}**",
            "",
            "Same-source siblings (two records from one source under one entity) — this is the "
            "intra-source noise level, name token-set ratio:",
            "",
            "| source | p5 | p25 | p50 | p75 | p95 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for label, key in (("S2", "within_s2_name_similarity"), ("S3", "within_s3_name_similarity")):
            d = dupes[key]
            lines.append(f"| {label} | {d['p5']} | {d['p25']} | {d['p50']} | {d['p75']} | {d['p95']} |")
        lines.append("")

    sep = results.get("separability")
    if sep:
        lines += [
            "## 4. True pairs vs hard negatives",
            "",
            f"**{sep['true_pairs']:,}** true pairs against **{sep['hard_negative_pairs']:,}** hard negatives "
            f"(same country, sharing at least one token, not a true match), drawn from a pool of "
            f"{sep['pool_size']:,} records.",
            "",
            "Median similarity, and the gap between them:",
            "",
            "| feature | true p50 | hard-neg p50 | separation |",
            "| --- | ---: | ---: | ---: |",
        ]
        for key, gap in sorted(sep["median_separation"].items(), key=lambda kv: -abs(kv[1])):
            t = sep["true_pair_similarity"][key]["p50"]
            n = sep["hard_negative_similarity"][key]["p50"]
            lines.append(f"| {key} | {t} | {n} | **{gap:+.4f}** |")
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=SECTIONS, action="append", help="run only these sections")
    parser.add_argument("--force", action="store_true", help="recompute even when a cache exists")
    parser.add_argument("--render-only", action="store_true", help="rebuild the report from caches")
    parser.add_argument("--out", default="reports/eda.md")
    args = parser.parse_args()

    wanted = tuple(args.section) if args.section else SECTIONS
    runners = {
        "counts": section_counts,
        "truth": section_truth,
        "dupes": section_dupes,
        "separability": section_separability,
    }

    results: dict[str, dict] = {}
    for name in SECTIONS:
        cached = load_cache(name)
        if args.render_only:
            if cached:
                results[name] = cached
            continue
        if name not in wanted:
            if cached:
                results[name] = cached
            continue
        if cached and not args.force:
            print(f"[cached] {name}")
            results[name] = cached
            continue
        print(f"[run] {name} ...")
        payload = runners[name]()
        save_cache(name, payload)
        results[name] = payload

    out_path = config.REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(results), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
