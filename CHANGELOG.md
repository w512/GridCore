# Changelog

## Unreleased

### Added
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
  `scripts/demo-tmux.sh` opens the demo layout (`--fake` works without a
  GPU); `scripts/gpu-watch.sh` lists GPU processes by GridCore model.
- `config.example.yaml`: Qwen3.8-27B (IQ4_XS on the GPU, Q4_K_M with one
  layer on the CPU) and Ornith-1.5-35B-A3B with experts in RAM.
- `gridcore profiles` lists stored measurements and says which ones the
  scheduler would use and why the others no longer apply (other runtime
  build, other GPU, changed model settings, model gone from the config);
  `gridcore profiles prune [--dry-run]` removes the stale ones.

- Cost-based eviction (`policy.eviction: cost`, the default): among the
  models the residency rules allow, the scheduler evicts the set that is
  cheapest to lose — reload time times recent demand, a request count that
  halves every 5 minutes and is weighted by class priority. Equal costs
  fall back to LRU; `eviction: lru` restores the 0.1 order. The cost of
  each resident model is in `/admin/state` (`evict_cost`) and `gridcore
  status`, and `evict` events state it.
- `policy.min_residency` (default 30s) against model thrash: batch work
  does not evict a model background used that recently, and a model loaded
  for background (batch) work is not replaced by other background (batch)
  work before it has been resident that long. Interactive work is not held
  back. In the 0.1 load test two models 200 MB over budget were reloaded
  30 times in 5 minutes.
- Thrash detection: a model loaded 6 times within 5 minutes produces a
  `thrash` event, a warning with a hint and `gridcore_model_thrash_total`.
- A job waiting for VRAM says which models are kept and why
  (`waiting for VRAM (kept: e4b in use by background)`).

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

### Changed
- Measured profiles are keyed by the llama.cpp build (`llama-server
  --version`: build number, commit, GPU backend) instead of the binary's
  path, mtime and size, so reinstalling or redeploying the same build keeps
  them. Profiles recorded by 0.1.0 for the binary that is installed now are
  carried over on start; `gridcore profiles prune` removes the old entries.
- CI and release builds use the Go toolchain pinned in `go.mod`
  (`toolchain go1.26.3`) instead of Go 1.25.0; the minimum Go for building
  from source is still 1.25. GitHub Actions moved to Node 24 releases.
- `make deploy` checks for `rsync` and `HOST` before building.

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
- Fake runtime and fake GPU for running and testing without hardware.
- `scripts/install-llamacpp.sh`, demo scripts, systemd user unit,
  mixed-class load generator.
