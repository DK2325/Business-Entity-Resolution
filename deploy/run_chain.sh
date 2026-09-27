#!/usr/bin/env bash
# Unattended end-to-end run: inference -> submission -> validation -> archive -> halt.
#
# Designed to complete with nothing else attached: it runs inside tmux, logs
# every stage, checkpoints each worker so an interrupted run resumes, and powers
# the machine down when finished.
#
#   bash deploy/run_chain.sh <threshold> [margin]
#
# Countries run smallest-index first (France, US, India). Two reasons: the
# smallest country finishes soonest and so proves the whole path end to end
# while the long ones are still ahead, and if anything is wrong it surfaces in
# minutes rather than hours.
#
set -uo pipefail

THRESHOLD="${1:?usage: run_chain.sh <threshold> [margin]}"
MARGIN="${2:-0.0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export BER_DATA_ROOT="${BER_DATA_ROOT:-$ROOT/data/student_resource}"
export BER_ARTIFACTS="${BER_ARTIFACTS:-$ROOT/work}"
export PYTHONUNBUFFERED=1
# One thread per process: several workers each spawning a thread per core
# oversubscribes the machine badly. Cannot change any prediction.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

LOG_DIR="$BER_ARTIFACTS"
RESULTS="$ROOT/output/results.txt"
mkdir -p "$LOG_DIR" "$ROOT/output"

source .venv/bin/activate

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

{
    echo "==================== run started $(stamp) ===================="
    echo "threshold=$THRESHOLD margin=$MARGIN"
    echo "cores=$(nproc)  mem=$(free -g | awk '/^Mem:/{print $2}')GB"
    echo
} | tee "$RESULTS"

# ---------------------------------------------------------------- (a) inference
echo "=== [a] test inference $(stamp) ===" | tee -a "$RESULTS"
python -u scripts/predict_parallel.py \
    --split test \
    --countries France US India \
    --workers auto \
    --k 5 \
    --checkpoint-every 100000 \
    --out-dir "$BER_ARTIFACTS/shards_test" \
    > "$LOG_DIR/test_infer.log" 2>&1
INFER_RC=$?
echo "inference exit code: $INFER_RC" | tee -a "$RESULTS"
tail -5 "$LOG_DIR/test_infer.log" | tee -a "$RESULTS"

if [ "$INFER_RC" -ne 0 ]; then
    echo "INFERENCE FAILED - not shutting down so the state can be inspected" | tee -a "$RESULTS"
    exit 1
fi

# ------------------------------------------------------------- (b) submission
echo | tee -a "$RESULTS"
echo "=== [b] writing submission $(stamp) ===" | tee -a "$RESULTS"
python -u scripts/write_submission.py \
    --shards "$BER_ARTIFACTS/shards_test" \
    --threshold "$THRESHOLD" \
    --margin "$MARGIN" \
    > "$LOG_DIR/submission.log" 2>&1
WRITE_RC=$?
echo "write exit code: $WRITE_RC" | tee -a "$RESULTS"
cat "$LOG_DIR/submission.log" | tee -a "$RESULTS"

if [ "$WRITE_RC" -ne 0 ]; then
    echo "WRITE FAILED - not shutting down" | tee -a "$RESULTS"
    exit 1
fi

# --------------------------------------------------- (c) validate + statistics
echo | tee -a "$RESULTS"
echo "=== [c] validator $(stamp) ===" | tee -a "$RESULTS"
python "$BER_DATA_ROOT/utils/validate_submission.py" \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir "$BER_DATA_ROOT/dataset/test" \
    > "$LOG_DIR/validator.log" 2>&1
VALIDATOR_RC=$?
cat "$LOG_DIR/validator.log" | tee -a "$RESULTS"
echo "validator exit code: $VALIDATOR_RC" | tee -a "$RESULTS"

echo | tee -a "$RESULTS"
echo "=== per-country match / singleton rates $(stamp) ===" | tee -a "$RESULTS"
python - <<'PY' 2>&1 | tee -a "$RESULTS"
import os
from pathlib import Path

root = Path(os.environ["BER_DATA_ROOT"])
test1 = root / "dataset" / "test" / "test_source1.tsv"

country = {}
with open(test1, encoding="utf-8") as handle:
    handle.readline()
    for line in handle:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 4:
            country[parts[0]] = parts[3]

stats = {}
total_matched = total_entities = total_ids = 0
with open("output/matching_results.tsv", encoding="utf-8") as handle:
    handle.readline()
    for line in handle:
        parts = line.rstrip("\n").split("\t")
        entity = parts[0]
        ids = [x for x in (parts[1] if len(parts) > 1 else "").split(",") if x]
        c = country.get(entity, "?")
        row = stats.setdefault(c, {"n": 0, "matched": 0, "ids": 0})
        row["n"] += 1
        total_entities += 1
        if ids:
            row["matched"] += 1
            row["ids"] += len(ids)
            total_matched += 1
            total_ids += len(ids)

print(f"{'country':<10}{'entities':>12}{'matched':>12}{'match rate':>12}"
      f"{'singleton':>12}{'mean ids':>10}")
for c in sorted(stats):
    row = stats[c]
    match_rate = row["matched"] / row["n"] if row["n"] else 0.0
    singleton = 1.0 - match_rate
    mean_ids = row["ids"] / row["matched"] if row["matched"] else 0.0
    print(f"{c:<10}{row['n']:>12,}{row['matched']:>12,}{match_rate:>11.2%}"
          f"{singleton:>12.2%}{mean_ids:>10.2f}")

overall_singleton = 1.0 - (total_matched / total_entities if total_entities else 0.0)
print()
print(f"overall predicted-singleton rate : {overall_singleton:.2%}")
print(f"training singleton rate          : 5.58%")
print(f"mean matched ids per matched ent : "
      f"{total_ids / total_matched if total_matched else 0.0:.2f}")
print(f"training mean matches per entity : 3.46")
print()
print("A France rate far from the US/India rates is the warning sign:")
print("France is absent from training, so it is the only country whose")
print("behaviour was never validated against labels.")
PY

# ------------------------------------------------------------------ (d) gzip
echo | tee -a "$RESULTS"
echo "=== [d] compressing outputs $(stamp) ===" | tee -a "$RESULTS"
GZIP_CMD=$(command -v pigz || command -v gzip)
for f in output/matching_results.tsv output/candidate_pairs.tsv; do
    if [ -f "$f" ]; then
        "$GZIP_CMD" -6 -k -f "$f"
        echo "  $(ls -la "$f.gz" | awk '{print $9, $5" bytes"}')" | tee -a "$RESULTS"
    fi
done
sha256sum output/*.tsv.gz | tee -a "$RESULTS"

echo | tee -a "$RESULTS"
echo "==================== run finished $(stamp) ====================" | tee -a "$RESULTS"
ls -la output/ | tee -a "$RESULTS"

# --------------------------------------------------------------- (e) shutdown
echo | tee -a "$RESULTS"
echo "halting in 60s; cancel with: sudo shutdown -c" | tee -a "$RESULTS"
sync
sleep 60
sudo shutdown -h now
