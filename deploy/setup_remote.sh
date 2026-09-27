#!/usr/bin/env bash
# Prepare a clean Ubuntu host to run the inference pipeline.
#
# Creates a virtual environment, installs the pinned dependencies, and checks
# that the dataset and model are where the pipeline expects them. Idempotent:
# safe to re-run.
#
#   bash deploy/setup_remote.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== host ==="
echo "cores : $(nproc)"
echo "memory: $(free -g | awk '/^Mem:/{print $2" GB total, "$7" GB available"}')"
echo "disk  : $(df -h . | awk 'NR==2{print $4" free"}')"
echo "python: $(python3 --version)"
echo

# ``import venv`` succeeding is not enough: Debian/Ubuntu ship the module but
# split out ``ensurepip``, and without it venv creation fails at the last step.
if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
    echo "python3-venv/ensurepip missing; installing"
    sudo apt-get update -qq
    sudo apt-get install -y -qq "python3-venv" "python$(python3 -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')-venv" python3-pip || true
fi

if [ ! -d .venv ]; then
    echo "=== creating virtualenv ==="
    python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --quiet --upgrade pip wheel

echo "=== installing dependencies ==="
# torch is only needed for optional experiments, not for inference; skip it on
# the remote host to avoid a multi-GB download.
grep -v -E '^(torch|xgboost)' requirements.txt > /tmp/requirements-remote.txt
python -m pip install --quiet -r /tmp/requirements-remote.txt
python -m pip install --quiet psutil

echo
echo "=== verifying layout ==="
DATA_ROOT="${BER_DATA_ROOT:-$ROOT/data/student_resource}"
ARTIFACTS="${BER_ARTIFACTS:-$ROOT/work}"
mkdir -p "$ARTIFACTS" output

missing=0
for f in test_source1.tsv test_source2.tsv test_source3.tsv; do
    if [ -f "$DATA_ROOT/dataset/test/$f" ]; then
        printf '  ok      %-20s %s\n' "$f" "$(du -h "$DATA_ROOT/dataset/test/$f" | cut -f1)"
    else
        printf '  MISSING %s\n' "$DATA_ROOT/dataset/test/$f"
        missing=1
    fi
done
if [ -f "$ARTIFACTS/matcher.txt" ]; then
    printf '  ok      %-20s %s\n' "matcher.txt" "$(du -h "$ARTIFACTS/matcher.txt" | cut -f1)"
else
    printf '  MISSING %s\n' "$ARTIFACTS/matcher.txt"
    missing=1
fi
if [ -f "$DATA_ROOT/utils/validate_submission.py" ]; then
    printf '  ok      %-20s\n' "validate_submission.py"
else
    printf '  note    validator not present (can be run locally instead)\n'
fi

echo
python - <<'PY'
import sys
sys.path.insert(0, ".")
from src import config, features, metrics  # noqa: F401
print(f"imports ok; {features.N_FEATURES} features")
print(f"data root : {config.DATA_ROOT}")
print(f"artifacts : {config.ARTIFACTS}")
PY

if [ "$missing" -ne 0 ]; then
    echo
    echo "setup incomplete: upload the missing files listed above"
    exit 1
fi
echo
echo "setup complete. next: bash deploy/run_remote.sh <threshold> [margin]"
