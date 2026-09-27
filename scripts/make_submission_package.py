"""Assemble the final submission archive.

Builds the exact structure the organisers specify:

    <team>_submission.zip
      output/matching_results.tsv     the file uploaded to the leaderboard
      output/candidate_pairs.tsv      the candidate set the model scored
      code/business_entity_resolution/
        src/  scripts/  README.md  requirements.txt
      Documentation_template.md

The two output files are passed in explicitly rather than read from ``output/``,
so the archive is built from the *exact* files that were validated and uploaded,
not from whatever happens to be lying around. The archive is written outside the
repository.

Usage::

    python scripts/make_submission_package.py \
        --matching /path/to/sub5_safe.tsv \
        --candidates /path/to/sub4_candidates.tsv \
        --out /path/to/CodeOps_submission.zip
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config

CODE_ROOT = "code/business_entity_resolution"
EXCLUDE_DIRS = {"__pycache__", ".git", ".venv", "venv", "artifacts", "work", "data"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".log", ".tmp", ".md"}
# Only `src/` and `scripts/` are walked, and the root-level files that ship are
# named explicitly below, so local notes sitting in the repository root are
# never picked up. Markdown inside the walked folders is excluded too: the
# documentation that ships is added by name, not by discovery.


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def want(path: Path) -> bool:
    if path.suffix in EXCLUDE_SUFFIXES:
        return False
    return not any(part in EXCLUDE_DIRS for part in path.parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--team", default="CodeOps")
    args = parser.parse_args()

    root = config.REPO_ROOT
    matching = Path(args.matching)
    candidates = Path(args.candidates)
    for path in (matching, candidates):
        if not path.exists():
            raise SystemExit(f"missing: {path}")

    out = Path(args.out) if args.out else root.parent / f"{args.team}_submission.zip"
    if out.resolve().is_relative_to(root.resolve()):
        raise SystemExit(f"refusing to write the archive inside the repo: {out}")

    print(f"matching   : {matching}  ({matching.stat().st_size / 1e6:.1f} MB)")
    print(f"candidates : {candidates}  ({candidates.stat().st_size / 1e9:.2f} GB)")
    print(f"archive    : {out}\n")

    # Sanity: both files must have one row per test Source 1 entity.
    expected = sum(1 for _ in open(config.TEST_FILES["source1"], encoding="utf-8")) - 1
    for label, path in (("matching_results", matching), ("candidate_pairs", candidates)):
        rows = sum(1 for _ in open(path, encoding="utf-8")) - 1
        status = "ok" if rows == expected else "MISMATCH"
        print(f"  {label}: {rows:,} rows (expected {expected:,}) -> {status}")
        if rows != expected:
            raise SystemExit("row count mismatch; refusing to package")

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(matching, "output/matching_results.tsv")
        print("\n  added output/matching_results.tsv")
        archive.write(candidates, "output/candidate_pairs.tsv")
        print("  added output/candidate_pairs.tsv")

        for folder in ("src", "scripts"):
            for path in sorted((root / folder).rglob("*")):
                if path.is_file() and want(path.relative_to(root)):
                    archive.write(path, f"{CODE_ROOT}/{path.relative_to(root).as_posix()}")
            print(f"  added {CODE_ROOT}/{folder}/")

        archive.write(root / "Readme.md", f"{CODE_ROOT}/README.md")
        archive.write(root / "requirements.txt", f"{CODE_ROOT}/requirements.txt")
        print(f"  added {CODE_ROOT}/README.md, requirements.txt")

        archive.write(root / "Documentation_template.md", "Documentation_template.md")
        print("  added Documentation_template.md")

        log = root / "submissions_log.csv"
        if log.exists():
            archive.write(log, f"{CODE_ROOT}/submissions_log.csv")
            print(f"  added {CODE_ROOT}/submissions_log.csv")

    size = out.stat().st_size
    print(f"\nwrote {out} ({size / 1e6:.1f} MB)")
    print(f"sha256 {sha256(out)}")

    print("\ncontents:")
    with zipfile.ZipFile(out) as archive:
        top: dict[str, int] = {}
        for info in archive.infolist():
            key = "/".join(info.filename.split("/")[:2]) if "/" in info.filename else info.filename
            top[key] = top.get(key, 0) + 1
        for key in sorted(top):
            print(f"  {key:<45}{top[key]:>4} file(s)")


if __name__ == "__main__":
    main()
