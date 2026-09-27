"""Central configuration: dataset paths, artifact locations and the random seed.

The dataset is deliberately kept outside the repository. Override the location
with the ``BER_DATA_ROOT`` environment variable or the ``--data-root`` flag that
the scripts in ``scripts/`` expose, so nothing here is machine-specific beyond
the default.
"""

from __future__ import annotations

import os
from pathlib import Path

SEED = 42

# Repository root (this file lives in <repo>/src/).
REPO_ROOT = Path(__file__).resolve().parent.parent

# ``student_resource`` as shipped by the organisers, kept outside the repo so the
# 2.4 GB of TSVs are never synced or committed.
#
# The default is relative to the repository so the code runs unchanged on any
# machine; set ``BER_DATA_ROOT`` to point at a copy held elsewhere.
DATA_ROOT = Path(
    os.environ.get("BER_DATA_ROOT", str(Path(__file__).resolve().parent.parent / "data" / "student_resource"))
)

DATASET_DIR = DATA_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
VALIDATOR = DATA_ROOT / "utils" / "validate_submission.py"

# Heavy intermediates (blocking indexes, feature matrices, model files). Ignored
# by git; kept off the OneDrive-synced repo path by default.
ARTIFACTS = Path(
    os.environ.get("BER_ARTIFACTS", str(Path(__file__).resolve().parent.parent / "work"))
)

OUTPUT_DIR = REPO_ROOT / "output"

TRAIN_FILES = {
    "source1": TRAIN_DIR / "train_source1.tsv",
    "source2": TRAIN_DIR / "train_source2.tsv",
    "source3": TRAIN_DIR / "train_source3.tsv",
    "ground_truth": TRAIN_DIR / "train_ground_truth.tsv",
}

TEST_FILES = {
    "source1": TEST_DIR / "test_source1.tsv",
    "source2": TEST_DIR / "test_source2.tsv",
    "source3": TEST_DIR / "test_source3.tsv",
}


def ensure_dirs() -> None:
    """Create the writable directories the pipeline assumes exist."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
