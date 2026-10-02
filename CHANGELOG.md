# Changelog

## 0.3.0 — 2026-10-02

Apple Silicon: GridCore runs real models on a Mac, through llama.cpp's
Metal backend. Measured on an M4 Pro 24 GB with macOS 15.8 and llama.cpp
b11146 (Homebrew). And a dashboard at `/admin/ui`. On the RTX 4060 Ti the
v0.2 load tests (mixed, thrash, ambient) give the same results as v0.2.0.

### Added
- `gpu.device: apple` (and `auto` on macOS): the device total is Metal's
  working-set limit (`llama-server --list-devices`, else
  `iogpu.wired_limit_mb`, else two thirds of RAM); memory in use comes
  from `ioreg`, a model's size from `footprint` (all the RAM its process
  takes, CPU-side tensors included), memory pressure and swap from
  `sysctl`. No cgo, no new dependencies. `gridcore check` names the device
  and where its limit came from.
- Estimates on unified memory count the host side too (embeddings, a tied
  model's input copy of `token_embd`, weights left on the CPU) with Metal's
  overhead instead of CUDA's: -2..+4 % against measured on Gemma 4 12B, E4B
  and E2B. `models --explain` says that `-ngl` / `--cpu-moe` save no memory
  there.
- Memory pressure: at "warn" background and batch work may not load
  models (family requests take a resident variant), at "critical" cold
  models are unloaded one every 5 s, ignoring `min_residency`; a minute of
  no lower-class loads follows each such unload. Interactive work is never
  held. `gridcore_memory_pressure_level`, pressure and swap in
  `/admin/state` and `gridcore status`.
- Device out of memory while serving: llama.cpp on an overcommitted Metal
  device keeps answering `/health` and fails every request, neighbours
  included. Running instances' OOM errors are counted
  (`gridcore_device_oom_total`); the model loaded within the last minute is
  unloaded with its measurement discarded, and background loads wait 30 s.
- `examples/apple-24gb.yaml`, `deploy/gridcore.plist` (launchd agent).
- Dashboard at `/admin/ui` (`/` redirects there): the memory budget split
  by model, memory / queue / utilisation charts over ten minutes, resident
  models, running and queued jobs with why they wait, and every scheduler
  event since the page was opened (the daemon keeps only its last 100).
  One page compiled into the binary that polls `/admin/state`: no external
  resources (a CSP forbids them), read-only.

### Changed
- llama.cpp launch defaults on unified memory, each only if the binary
  knows the flag and the model's `args` do not set it: `--load-mode none`
  (or `--no-mmap` on older builds), `--cache-ram 0`, `--ctx-checkpoints 0`.
  Each of them otherwise grows a process after it was measured: the RAM
  prompt cache alone took a 12B from 8.8 to 18.2 GB in a dozen prompts.
- Default `headroom_mb` is 1024 on Apple Silicon (512 elsewhere).
- On unified memory models load one at a time.
- While an interactive request is freeing memory for its model or waiting
  for the load slot, background and batch work start no loads (all
  devices). On a 24 GB Mac a starved background job kept slipping its load
  in first: 64 loads in 3 minutes and no chat answered.

### Fixed
- A load whose warmup ran out of memory is a failed (OOM) load even when
  `/health` already answers.
- Orphan reaping after a crash checks reused PIDs through `ps` where there
  is no `/proc` (macOS); before, it could kill an unrelated process.
- `gridcore bench` measures on monitors that must be told the PID.

## 0.2.0 — 2026-10-02

Smarter residency on one card: models are evicted by what they would cost
to lose, models that do not fit together take turns instead of thrashing,
no class starves, and a request can name a model family and get the variant
that fits the moment. Measured on an RTX 4060 Ti 16 GB with llama.cpp b11060.

### Added
- Model families (`families:` with `preferred` / `balanced` / `compact`
  tiers and aliases). Interactive work gets the best variant it can have
  (evicting as for a plain model, falling back only when the better ones
  cannot be made resident); background and batch work get the best variant
  that runs without evicting anything or sharing the model the user is
  chatting with. `"gridcore": {"quality": ...}` / `X-GridCore-Quality`
  excludes lower tiers; requests with images (audio) only go to variants
  with the `vision` (new: `audio`) capability. Responses name the variant
  (`X-GridCore-Model`, `X-GridCore-Family`); `variant` events explain the
  choice; `gridcore_variant_selected_total{family,variant,class}`;
  `/v1/models` lists families with their variants.
- Cost-based eviction (`policy.eviction: cost`, the default): among the
  models the residency rules allow, the scheduler evicts the set that is
  cheapest to lose — reload time times recent demand, a request count that
  halves every 5 minutes and is weighted by class priority. Equal costs
  fall back to LRU; `eviction: lru` restores the 0.1 order. Each resident
  model's cost is in `/admin/state` (`evict_cost`) and `gridcore status`,
  and `evict` events state it.
- `policy.min_residency` (default 30s) against model thrash: batch work
  does not evict a model background used that recently, and a model loaded
  for background (batch) work is not replaced by other background (batch)
  work before it has been resident that long. Interactive work is not held
  back. Two models 200 MB over budget were reloaded 60 times in 5 minutes by
  0.1, 5 times by 0.2.
- `policy.background_share` (default 0.5) caps how much of the time under
  continuous interactive load background/batch steps run alongside it:
  after a step of length d the next one waits d·(1/share − 1), so steps of
  any length add up to the share. With six chatting clients it took chat
  p50 from 4.0 s to 2.7 s (p95 4.8 → 3.5 s) and halved background
  throughput while the chat lasted; `1` is the 0.1 behaviour.
- `policy.batch_max_starvation` (default 2m): a batch job that has made no
  progress for this long gets one step even while background has work
  queued, and may take a model background is using (never a hot or pinned
  one). `0s` restores strict ordering.
- Thrash detection: a model loaded 6 times within 5 minutes produces a
  `thrash` event, a warning with a hint and `gridcore_model_thrash_total`.
- A job waiting for VRAM says which models are kept and why
  (`waiting for VRAM (kept: e4b in use by background)`); a job held back by
  the share says when its next step may run.
- `gridcore profiles` lists stored measurements and says which ones the
  scheduler would use and why the others no longer apply (other runtime
  build, other GPU, changed model settings, model gone from the config);
  `gridcore profiles prune [--dry-run]` removes the stale ones.
- `scripts/loadtest -scenario ambient`: chat, a coding agent, screenshot
  analysis, a file indexer and OCR on one GPU, with per-role latency, the
  model that served each role, and loads / evictions / thrash from
  `/metrics`.
- VRAM estimate honours the llama.cpp offload flags in a model's args
  (`-ngl` / `--n-gpu-layers`, `--cpu-moe`, `--n-cpu-moe`, `-ot <regex>=CPU`)
  and skips multi-token-prediction blocks, so models larger than the card
  (Qwen3.8-27B with `-ngl 64`) and MoE models with experts in host RAM
  (Ornith-1.5-35B-A3B with `--cpu-moe`) are admitted without a prior
  `gridcore bench`. `gridcore models --explain` shows the host part.
- `gridcore status` dashboard: colors, per-model VRAM bars, relative event
  times, flicker-free `--watch`; honours `--no-color` and `NO_COLOR`.
- Demo tools: `make tools` builds `bin/gc-chat` (streaming chat with queue
  time, TTFT and tokens/s) and `bin/gc-indexer` (background embedding or
  LLM classification load with a progress bar that visibly pauses);
  `scripts/demo-tmux.sh` opens the demo layout (`--simulation` works without a
  GPU); `scripts/gpu-watch.sh` lists GPU processes by GridCore model.
- `config.example.yaml`: a `gemma4` family, Qwen3.8-27B (IQ4_XS on the GPU,
  Q4_K_M with one layer on the CPU) and Ornith-1.5-35B-A3B with experts in
  RAM; `examples/simulation.yaml`: a `qwen3` family to try without a GPU.

### Changed
- Measured profiles are keyed by the llama.cpp build (`llama-server
  --version`: build number, commit, GPU backend) instead of the binary's
  path, mtime and size, so reinstalling or redeploying the same build keeps
  them. Profiles recorded by 0.1.0 for the binary that is installed now are
  carried over on start; `gridcore profiles prune` removes the old entries.
- README: a coding agent that works on its own should send `background`.
  In the acceptance scenario that cost the agent ~10% and gave the indexer
  next to it 4.6 times the throughput.
- CI and release builds use the Go toolchain pinned in `go.mod`
  (`toolchain go1.26.3`) instead of Go 1.25.0; the minimum Go for building
  from source is still 1.25. GitHub Actions moved to Node 24 releases.
- `make deploy` checks for `rsync` and `HOST` before building.

### Fixed
- `gridcore bench` (or a prune) while the daemon was running lost its
  results: the daemon rewrote `profiles.json` from memory on its next
  update. The store now merges changes made by other processes before it
  writes, and skips writes that change nothing (it used to rewrite the
  file on every GPU poll).
- An instance that was exiting between the two `nvidia-smi` queries made
  its memory look like external usage for one poll, and the planner
  evicted a second, healthy model to make room.
- The orphan reaper identifies processes by command line and port instead
  of `/proc/<pid>/exe`, which pointed at the interpreter for wrapped
  runtimes and leaked the child.

## 0.1.0 — 2026-09-13

### Added
- OpenAI-compatible gateway (`/v1/chat/completions`, `/v1/completions`,
  `/v1/embeddings`, `/v1/models`) in front of one or more `llama-server`
  processes, one per model.
- Three workload classes (`interactive`, `background`, `batch`) selected per
  request via header, body or model suffix; strict ordering with bounded
  background starvation (`policy.background_max_starvation`).
- VRAM-aware admission with GGUF-based estimates (weights, per-layer KV cache
  including sliding-window / shared-KV / hybrid layouts, mmproj, buffers),
  replaced by measured per-process VRAM after the first load.
- Model residency: pinned / hot / cold tiers, LRU eviction, drain before stop,
  reservations against over-commit.
- Chunked embeddings with yield points between chunks.
- Failure handling: crash detection, per-model circuit breaker, profile
  reset on out-of-memory, orphan reaping on restart.
- `gridcore bench` to measure models; `gridcore models --explain` for
  estimate breakdowns; `gridcore status --watch` dashboard; Prometheus
  metrics; `/admin/state`.
- Simulated runtime and GPU for running and testing without hardware.
- `scripts/install-llamacpp.sh`, demo scripts, systemd user unit,
  mixed-class load generator.
