#!/usr/bin/env bash
# Demo 2: VRAM pressure. A cold model occupies the card; an interactive
# request for a model that does not fit evicts it and loads its own.
#
# Needs at least two chat models whose footprints do not fit together in
# the budget (examples/simulation.yaml is set up that way).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/demo-lib.sh
check_server

# Two chat models that do not fit together (override with FIRST_MODEL/SECOND_MODEL).
read -r FIRST SECOND < <(curl -s "http://$GC/v1/models" | python3 -c '
import json,sys
ms=[m["id"] for m in json.load(sys.stdin)["data"] if "chat" in (m.get("capabilities") or []) and not m.get("root")]
print(" ".join(ms[:2]))')
FIRST=${FIRST_MODEL:-$FIRST}; SECOND=${SECOND_MODEL:-$SECOND}
[[ -n "$FIRST" && -n "$SECOND" ]] || { echo "need two chat models configured" >&2; exit 1; }

say "Make $FIRST resident with a background request"
chat "$FIRST" background "Say hello." 8
resident

say "Interactive request for $SECOND: does not fit next to $FIRST -> evict (it is idle), load, run"
T0=$(now_iso)
chat "$SECOND" interactive "Name three colors." 16

say "Now $SECOND is hot. A background request for $FIRST must NOT evict it: it waits and times out"
code=$(curl -s -o /dev/null -w '%{http_code} %{time_total}' \
  -H 'X-GridCore-Class: background' -H 'X-GridCore-Max-Wait-Ms: 3000' -H 'Content-Type: application/json' \
  -d "{\"model\":\"$FIRST\",\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" "http://$GC/v1/chat/completions")
printf '   %-11s %-14s HTTP %s %5.2fs  (max_wait 3s)\n' background "$FIRST" "${code% *}" "${code#* }"

say "What the scheduler did"
events "$T0"
echo
resident
