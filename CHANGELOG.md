# Changelog

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
