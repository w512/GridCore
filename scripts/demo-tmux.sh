#!/usr/bin/env bash
# Opens a tmux layout for demos and recordings:
#
#   ┌──────────────────────┬──────────────────┐
#   │ gridcore status      │ chat             │
#   │   --watch            ├──────────────────┤
#   │                      │ indexer          │
#   ├──────────────────────┴──────────────────┤
#   │ nvtop / nvidia-smi (or gridcore log)    │
#   └─────────────────────────────────────────┘
#
#   scripts/demo-tmux.sh                 # against a running gridcore at 127.0.0.1:8080
#   GRIDCORE_ADDR=host:port scripts/demo-tmux.sh
#   scripts/demo-tmux.sh --fake          # also start gridcore on examples/fake-demo.yaml
#
# The chat pane waits for you to type; the indexer pane has the command
# typed but not started, so you control when the background load begins.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v tmux >/dev/null || { echo "tmux is required" >&2; exit 1; }

ADDR="${GRIDCORE_ADDR:-127.0.0.1:8080}"
SESSION="${SESSION:-gridcore-demo}"
FAKE=0
[[ "${1:-}" == "--fake" ]] && FAKE=1

# Build the demo tools into bin/ so panes start instantly.
mkdir -p bin
go build -o bin/gridcore ./cmd/gridcore
go build -o bin/gc-chat ./scripts/chat
go build -o bin/gc-indexer ./scripts/demo-indexer

if [[ $FAKE == 1 ]]; then
  if ! curl -sf "http://$ADDR/health" >/dev/null 2>&1; then
    bin/gridcore serve --config examples/fake-demo.yaml --state-dir /tmp/gridcore-demo-state > /tmp/gridcore-demo.log 2>&1 &
    for _ in $(seq 1 30); do curl -sf "http://$ADDR/health" >/dev/null 2>&1 && break; sleep 0.2; done
  fi
fi
curl -sf "http://$ADDR/health" >/dev/null || { echo "gridcore is not answering at $ADDR" >&2; exit 1; }

# Bottom pane: nvtop if present, else nvidia-smi loop, else the daemon log.
if command -v nvtop >/dev/null; then
  BOTTOM="nvtop"
elif command -v nvidia-smi >/dev/null; then
  BOTTOM="watch -n1 -t 'nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader; echo; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader'"
elif [[ $FAKE == 1 ]]; then
  BOTTOM="tail -f /tmp/gridcore-demo.log"
else
  BOTTOM="echo 'no GPU tooling here'; sleep infinity"
fi

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x 200 -y 50 -e GRIDCORE_ADDR="$ADDR"
# Pane ids (%N) are stable regardless of layout position.
DASH=$(tmux display-message -t "$SESSION" -p '#{pane_id}')
BOT=$(tmux split-window -v -t "$DASH" -p 25 -P -F '#{pane_id}')      # full-width bottom
CHAT=$(tmux split-window -h -t "$DASH" -p 42 -P -F '#{pane_id}')     # right column
IDX=$(tmux split-window -v -t "$CHAT" -p 35 -P -F '#{pane_id}')      # right-bottom

tmux send-keys -t "$DASH" "bin/gridcore status --watch --addr $ADDR" C-m
tmux send-keys -t "$CHAT" "clear; bin/gc-chat -addr $ADDR" C-m
tmux send-keys -t "$IDX"  "clear; bin/gc-indexer -addr $ADDR -docs 3000 -loop"   # typed, not run
tmux send-keys -t "$BOT"  "$BOTTOM" C-m
tmux select-pane -t "$CHAT"
tmux set -t "$SESSION" status off
tmux set -t "$SESSION" pane-border-style fg=colour240
tmux set -t "$SESSION" pane-active-border-style fg=colour245
if [[ -n "${NO_ATTACH:-}" ]]; then
  echo "session $SESSION ready: tmux attach -t $SESSION"
elif [[ -n "${TMUX:-}" ]]; then
  tmux switch-client -t "$SESSION"
else
  tmux attach -t "$SESSION"
fi
