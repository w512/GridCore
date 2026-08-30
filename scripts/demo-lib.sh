# Shared helpers for demo scripts. Source, do not execute.
GC="${GRIDCORE_ADDR:-127.0.0.1:8080}"

need() { command -v "$1" >/dev/null 2>&1 || { echo "error: $1 is required" >&2; exit 1; }; }
need curl; need python3

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
note() { printf '   %s\n' "$*"; }

check_server() {
  curl -sf "http://$GC/health" >/dev/null || { echo "error: gridcore is not answering at $GC (set GRIDCORE_ADDR)" >&2; exit 1; }
}

# model_with CAP  -> first configured model id that has the capability
model_with() {
  curl -s "http://$GC/v1/models" | python3 -c '
import json,sys
cap=sys.argv[1]
for m in json.load(sys.stdin)["data"]:
    if cap in (m.get("capabilities") or []) and not m.get("root"):
        print(m["id"]); break' "$1"
}

# chat MODEL CLASS TEXT [MAX_TOKENS] -> prints "HTTP <code> <total>s queue <ms>ms: <content>"
chat() {
  local model=$1 class=$2 text=$3 max=${4:-32}
  local out hdr
  hdr=$(mktemp)
  out=$(curl -s -D "$hdr" -w '\n%{http_code} %{time_total}' \
    -H "X-GridCore-Class: $class" -H 'Content-Type: application/json' \
    -d "$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"max_tokens":int(sys.argv[3]),"messages":[{"role":"user","content":sys.argv[2]}]}))' "$model" "$text" "$max")" \
    "http://$GC/v1/chat/completions")
  local body=${out%$'\n'*} tail=${out##*$'\n'}
  local code=${tail% *} secs=${tail#* }
  local queue; queue=$(grep -i '^x-gridcore-queue-ms:' "$hdr" | tr -d '\r' | awk '{print $2}')
  rm -f "$hdr"
  local content
  content=$(printf '%s' "$body" | python3 -c '
import json,sys
try:
    d=json.load(sys.stdin)
    if "error" in d: print("ERROR:", d["error"].get("code"), d["error"].get("message","")[:120]); sys.exit()
    print((d["choices"][0]["message"].get("content") or "").strip().replace("\n"," ")[:90])
except Exception as e: print("(unparseable)")')
  printf '   %-11s %-14s HTTP %s %5.2fs  queue %5sms  %s\n' "$class" "$model" "$code" "$secs" "${queue:-0}" "$content"
}

# embed MODEL N  -> "HTTP <code> <total>s items=<n>"
embed() {
  local model=$1 n=$2
  local out
  out=$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"input":["document %d about gpu scheduling" % i for i in range(int(sys.argv[2]))]}))' "$model" "$n" \
    | curl -s -w '\n%{http_code} %{time_total}' -H 'X-GridCore-Class: background' -H 'Content-Type: application/json' -d @- "http://$GC/v1/embeddings")
  local body=${out%$'\n'*} tail=${out##*$'\n'}
  local items; items=$(printf '%s' "$body" | python3 -c 'import json,sys
try: print(len(json.load(sys.stdin)["data"]))
except Exception: print("?")')
  printf '   %-11s %-14s HTTP %s %5.2fs  items=%s\n' background "$model" "${tail% *}" "${tail#* }" "$items"
}

resident() {
  curl -s "http://$GC/admin/state" | python3 -c '
import json,sys
s=json.load(sys.stdin); g=s["gpu"]
print("   GPU %s: %.1f / %.1f GB committed, mode=%s" % (g["name"], g["committed_mb"]/1024, g["budget_mb"]/1024, s["mode"]))
for r in s["resident"]:
    print("   %-16s %-9s %-7s %5.1f GB  slots %d/%d" % (r["id"], r["state"], r["tier"], r["vram_mb"]/1024, r["busy_slots"], r["slots"]))'
}

# events SINCE_ISO  -> scheduler events after the given timestamp
events() {
  curl -s "http://$GC/admin/state" | python3 -c '
import json,sys
since=sys.argv[1]
for e in json.load(sys.stdin)["recent_events"]:
    if e["at"] > since:
        print("   %s %-9s %-16s %s" % (e["at"][11:23], e["kind"], e["subject"][:16], e["detail"][:70]))' "$1"
}

now_iso() { python3 -c 'import datetime; print(datetime.datetime.now().astimezone().isoformat(timespec="milliseconds"))'; }
