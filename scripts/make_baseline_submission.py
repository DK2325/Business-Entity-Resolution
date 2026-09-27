"""Write the trivial "predict nothing" submission.

Two purposes:

* It is the score floor. Every Source 1 entity gets an empty match list, so the
  macro F_0.5 equals the singleton rate -- about 0.056 if the test set has the
  same 5.58% singleton share as training. Any real model must beat that, and it
  is a useful sanity check on the leaderboard's scoring.
* It exercises the output path end to end against the organisers' validator
  before any modelling exists, so a format problem surfaces now rather than
  while burning one of the five daily submissions.

Run: ``python scripts/make_baseline_submission.py``
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config


def main() -> None:
    config.ensure_dirs()

    entity_ids: list[str] = []
    with open(config.TEST_FILES["source1"], encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            entity_id = line.split("\t", 1)[0].strip()
            if entity_id:
                entity_ids.append(entity_id)
    print(f"test Source 1 entities: {len(entity_ids):,}")

    targets = [
        (config.OUTPUT_DIR / "matching_results.tsv", "matched_entity_ids"),
        (config.OUTPUT_DIR / "candidate_pairs.tsv", "candidate_entity_ids"),
    ]
    for path, column in targets:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(f"source1_entity_id\t{column}\n")
            for entity_id in entity_ids:
                handle.write(f"{entity_id}\t\n")
        print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")

    if not config.VALIDATOR.exists():
        print(f"validator not found at {config.VALIDATOR}; skipping check")
        return

    command = [
        sys.executable,
        str(config.VALIDATOR),
        "--matching", str(config.OUTPUT_DIR / "matching_results.tsv"),
        "--candidate", str(config.OUTPUT_DIR / "candidate_pairs.tsv"),
        "--test-dir", str(config.TEST_DIR),
    ]
    print("\n$ " + " ".join(command))
    completed = subprocess.run(command, capture_output=True, text=True)
    print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)
    print(f"validator exit code: {completed.returncode}")
    sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
