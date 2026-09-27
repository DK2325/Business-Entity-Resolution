"""Package the code and trained model for a remote run.

Writes a single archive **outside the repository** so nothing large or
machine-specific enters git history. The dataset is deliberately excluded: the
test TSVs are ~1.2 GB and are transferred separately, since they change far less
often than the code.

    python deploy/make_bundle.py
    python deploy/make_bundle.py --out /path/to/bundle.tar.gz --with-test-data
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config

# Everything the pipeline needs to run inference on another machine.
CODE_PATHS = [
    "src",
    "scripts",
    "deploy",
    "requirements.txt",
    "Readme.md",
]

EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".log", ".tmp"}
EXCLUDE_DIRS = {"__pycache__", ".git", ".venv", "venv", "artifacts", "work", "output", "data"}


def _filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(info.name).parts
    if any(part in EXCLUDE_DIRS for part in parts):
        return None
    if Path(info.name).suffix in EXCLUDE_SUFFIXES:
        return None
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    parser.add_argument("--with-test-data", action="store_true",
                        help="also include the three test TSVs (~1.2 GB)")
    args = parser.parse_args()

    root = config.REPO_ROOT
    default_out = root.parent / f"ber_bundle_{time.strftime('%Y%m%d_%H%M')}.tar.gz"
    out_path = Path(args.out) if args.out else default_out
    if out_path.resolve().is_relative_to(root.resolve()):
        raise SystemExit(f"refusing to write the bundle inside the repo: {out_path}")

    model_path = config.ARTIFACTS / "matcher.txt"
    if not model_path.exists():
        raise SystemExit(f"trained model not found at {model_path}")

    print(f"repo  : {root}")
    print(f"model : {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")
    print(f"out   : {out_path}\n")

    with tarfile.open(out_path, "w:gz") as archive:
        for relative in CODE_PATHS:
            source = root / relative
            if not source.exists():
                print(f"  skip (absent): {relative}")
                continue
            archive.add(source, arcname=relative, filter=_filter)
            print(f"  added: {relative}")

        # The model lands where config.ARTIFACTS points by default on the host.
        archive.add(model_path, arcname="work/matcher.txt")
        print("  added: work/matcher.txt")

        splits = config.ARTIFACTS / "splits.json"
        if splits.exists():
            archive.add(splits, arcname="work/splits.json")
            print("  added: work/splits.json")

        if args.with_test_data:
            for name, path in config.TEST_FILES.items():
                arcname = f"data/student_resource/dataset/test/{path.name}"
                archive.add(path, arcname=arcname)
                print(f"  added: {arcname} ({path.stat().st_size / 1e6:.0f} MB)")
            validator = config.VALIDATOR
            if validator.exists():
                archive.add(validator, arcname="data/student_resource/utils/validate_submission.py")
                print("  added: validator")

    size = out_path.stat().st_size
    print(f"\nwrote {out_path} ({size / 1e6:.1f} MB)")
    print("\ncontents (top level):")
    with tarfile.open(out_path, "r:gz") as archive:
        seen: set[str] = set()
        for name in archive.getnames():
            top = name.split("/")[0]
            if top not in seen:
                seen.add(top)
                print(f"  {top}/")


if __name__ == "__main__":
    main()
