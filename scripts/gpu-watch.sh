#!/usr/bin/env bash
# Compact ground-truth view of the NVIDIA GPU for demos: the card's memory
# and utilisation, then every compute process with the GridCore model it
# serves (read from the llama-server command line). Fits in a few lines.
#
#   scripts/gpu-watch.sh            # refresh every 0.5s
#   scripts/gpu-watch.sh --once
set -uo pipefail
IDX="${GPU_INDEX:-0}"
B=$'\033[1m'; D=$'\033[2m'; G=$'\033[32m'; Y=$'\033[33m'; C=$'\033[36m'; R=$'\033[0m'

once() {
  local line name used total util temp pwr
  line=$(nvidia-smi --id="$IDX" --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null) || { echo "nvidia-smi failed"; return; }
  IFS=',' read -r name used total util temp pwr <<<"$line"
  name=${name# }; used=${used# }; total=${total# }; util=${util# }; temp=${temp# }; pwr=${pwr# }
  local pct=$(( used * 100 / total ))
  local width=30 filled=$(( pct * 30 / 100 )) bar=""
  bar+=$(printf '%*s' "$filled" '' | tr ' ' '█')
  bar+=$D$(printf '%*s' $(( width - filled )) '' | tr ' ' '·')$R
  printf '%snvidia-smi%s  %s\n' "$B" "$R" "$name"
  printf '  [%s] %s%5d%s / %d MiB   util %s%3s%%%s   %s°C   %s W\n' "$bar" "$B" "$used" "$R" "$total" "$C" "$util" "$R" "$temp" "${pwr%%.*}"
  echo
  local any=0
  while IFS=',' read -r pid mem; do
    pid=${pid# }; mem=${mem# }
    [[ -z "$pid" ]] && continue
    any=1
    local alias="" port="" argv
    argv=$(tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null)
    alias=$(awk '/^--alias$/{getline; print; exit}' <<<"$argv")
    port=$(awk '/^--port$/{getline; print; exit}' <<<"$argv")
    if [[ -n "$alias" ]]; then
      printf '  pid %-7s %s%6s MiB%s  %s%-16s%s %s:%s%s\n' "$pid" "$G" "$mem" "$R" "$B" "$alias" "$R" "$D" "$port" "$R"
    else
      printf '  pid %-7s %s%6s MiB%s  %s%s%s\n' "$pid" "$Y" "$mem" "$R" "$D" "$(basename "$(readlink /proc/$pid/exe 2>/dev/null || echo unknown)")" "$R"
    fi
  done < <(nvidia-smi --id="$IDX" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null)
  [[ $any == 0 ]] && printf '  %sno compute processes%s\n' "$D" "$R"
}

if [[ "${1:-}" == "--once" ]]; then once; exit 0; fi
printf '\033[?25l\033[2J'; trap 'printf "\033[?25h"; exit' INT TERM
while true; do
  out=$(once | sed $'s/$/\033[K/')     # clear to end of each line: no leftovers
  printf '\033[H%s\033[J' "$out"
  sleep 0.5
done
