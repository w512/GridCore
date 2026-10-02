#!/usr/bin/env bash
# Demo 1: background indexing keeps running, an interactive chat arrives and
# gets the GPU immediately; indexing resumes once the user is done.
#
#   gridcore serve --config examples/simulation.yaml &   # or a real config
#   scripts/demo-background-vs-interactive.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/demo-lib.sh
check_server

CHAT=${CHAT_MODEL:-$(model_with chat)}
EMBED=${EMBED_MODEL:-$(model_with embedding)}
[[ -n "$CHAT" && -n "$EMBED" ]] || { echo "need one chat and one embedding model configured" >&2; exit 1; }

say "Warm up: load the chat model with one interactive request"
chat "$CHAT" interactive "Say hello." 8
sleep 3   # let the interactive idle window pass

say "Start background indexing: 1000 documents in 32-document chunks"
T0=$(now_iso)
( embed "$EMBED" 1000 ) &
BG=$!
sleep 0.3

say "150 ms later a user starts chatting"
chat "$CHAT" interactive "In one sentence, what does a GPU scheduler do?" 40

say "Indexing finishes after the chat"
wait $BG

say "What the scheduler did"
events "$T0"
echo
resident
