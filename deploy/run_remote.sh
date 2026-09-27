#!/usr/bin/env bash
# Run test inference and write both submission files.
#
# Starts inside tmux so the run survives an SSH disconnect, sizes the worker
# pool from the host's cores and memory, and checkpoints each worker so an
# interrupted run resumes rather than restarting.
#
#   bash deploy/run_remote.sh 0.75          # threshold, margin defaults to 0
#   bash deploy/run_remote.sh 0.75 0.05
#
# Watch progress:   tmux attach -t ber
# Detach:           Ctrl-b then d
# Tail the log:     tail -f work/test_infer.log
#
set -euo pipefail

THRESHOLD="${1:?usage: run_remote.sh <threshold> [margin]}"
MARGIN="${2:-0.0}"
SESSION="${SESSION:-ber}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ARTIFACTS="${BER_ARTIFACTS:-$ROOT/work}"
mkdir -p "$ARTIFACTS" output

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux missing; installing"
    sudo apt-get update -qq && sudo apt-get install -y -qq tmux
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "session '$SESSION' already exists. Attach with: tmux attach -t $SESSION"
    echo "To start over: tmux kill-session -t $SESSION"
    exit 1
fi

cat > "$ARTIFACTS/_run_inside_tmux.sh" <<INNER
set -euo pipefail
cd "$ROOT"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export BER_ARTIFACTS="$ARTIFACTS"
[ -n "\${BER_DATA_ROOT:-}" ] && export BER_DATA_ROOT="\${BER_DATA_ROOT}"

echo "=== test inference started \$(date -Is) ==="
python -u scripts/predict_parallel.py \\
    --split test \\
    --workers auto \\
    --k 5 \\
    --checkpoint-every 200000 \\
    --out-dir "$ARTIFACTS/shards_test" \\
    2>&1 | tee "$ARTIFACTS/test_infer.log"

echo "=== writing submission \$(date -Is) ==="
python -u scripts/write_submission.py \\
    --shards "$ARTIFACTS/shards_test" \\
    --threshold $THRESHOLD \\
    --margin $MARGIN \\
    2>&1 | tee "$ARTIFACTS/submission.log"

echo "=== done \$(date -Is) ==="
ls -la output/
INNER

chmod +x "$ARTIFACTS/_run_inside_tmux.sh"

echo "threshold=$THRESHOLD margin=$MARGIN"
echo "starting tmux session '$SESSION'"
tmux new-session -d -s "$SESSION" "bash $ARTIFACTS/_run_inside_tmux.sh; echo; echo '[finished - press any key]'; read -n 1"

sleep 2
tmux list-sessions
echo
echo "running in background. useful commands:"
echo "  tmux attach -t $SESSION      # watch live (Ctrl-b d to detach)"
echo "  tail -f $ARTIFACTS/test_infer.log"
echo "  ls -la $ARTIFACTS/shards_test/"
