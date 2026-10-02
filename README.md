# GridCore

**A workload scheduler for local AI computers.**

One GPU. Several things want it at once: an interactive chat, a coding agent,
an indexer embedding your files in the background, a screenshot analyser
that wakes up every few seconds. Today each of them talks to its own
`llama-server` or Ollama and they fight over VRAM and compute; whoever loaded
last wins, and the chat you are typing into stalls because an indexer is
busy.

GridCore sits between your applications and the inference runtime and
**owns the GPU**. Applications keep using the OpenAI API. GridCore decides
what runs now, what waits, which model stays resident and which one gets
evicted to make room.

```
Apps / Agents / Desktop services
        │  OpenAI-compatible API  (+ optional X-GridCore-* headers)
        ▼
┌──────────────────────────────────────────┐
│  gridcore                                │
│   priority queues   ·  VRAM planner      │
│   model residency   ·  admission control │
└───────┬──────────────┬───────────────────┘
        ▼              ▼
  llama-server    llama-server   ...   (one process per model)
        └────── GPU ──────┘
```

It is not a model router ("which LLM is smartest for this prompt"). It is
closer to an operating-system scheduler: it manages *workload classes*,
*models* and *VRAM* on a machine you own.

## What it does

- **Three workload classes.** `interactive` always goes first. `background`
  and `batch` run when the GPU is free, are chunked, and yield between chunks
  so a chat request never waits behind an embedding job. Under continuous
  interactive traffic background still makes progress (one small step at a
  time), so indexing never starves.
- **VRAM-aware admission.** Before a model is loaded GridCore knows whether
  it fits: it reads the GGUF header and estimates weights, KV cache and
  buffers (within about 10% on tested models), honouring the llama.cpp
  offload flags in the model's args (`-ngl`, `--cpu-moe`, `--n-cpu-moe`,
  `-ot ...=CPU`), then replaces the estimate with the measured footprint
  after the first load.
- **Model residency.** Models stay loaded while they are useful. A model the
  user is actively chatting with is *hot* and cannot be evicted by background
  work; `pinned` models (embeddings) are never evicted. When something else
  needs the room, the cold models that are cheapest to lose go first: reload
  time weighed by how often each class asked for them lately, so a model
  used every few seconds outlives one used once a moment ago. Two models
  that do not fit together take turns (`min_residency`) instead of pushing
  each other out on every request, and a model that keeps coming back is
  reported as thrash with a hint.
- **Self-correcting.** If a load runs out of memory the stored measurement is
  discarded and the next attempt is refused up front with a clear error
  instead of crashing three more times. Repeated failures trip a circuit
  breaker per model. A crashed daemon's orphaned runtimes are reaped on the
  next start.
- **Observable.** Prometheus metrics, a JSON state endpoint, and
  `gridcore status --watch` showing what is running, queued and resident.

## Quickstart

Requirements: Linux x86_64, an NVIDIA GPU with a working driver
(`nvidia-smi` prints your card), and some GGUF models.

```bash
# 1. GridCore binary
curl -fsSL https://github.com/w512/GridCore/releases/latest/download/gridcore-linux-amd64 \
  -o ~/.local/bin/gridcore && chmod +x ~/.local/bin/gridcore

# 2. llama.cpp (prebuilt CUDA release into /opt/llama.cpp/current)
curl -fsSL https://raw.githubusercontent.com/w512/GridCore/main/scripts/install-llamacpp.sh | bash

# 3. Config: point it at your models
mkdir -p ~/.config/gridcore
curl -fsSL https://raw.githubusercontent.com/w512/GridCore/main/config.example.yaml \
  -o ~/.config/gridcore/config.yaml
$EDITOR ~/.config/gridcore/config.yaml        # paths under models:

# 4. Check, measure, run
gridcore check
gridcore bench --all                          # loads each model once, records VRAM / speed
gridcore serve
```

Then, from any OpenAI client:

```bash
curl localhost:8080/v1/chat/completions -d '{
  "model": "gpt-4o",
  "messages": [{"role": "user", "content": "hello"}]
}'
```

`gpt-4o` here is an alias from the config, so existing tools work without
changes. Run `gridcore status --watch` in another terminal to see the
scheduler work.

No GPU at hand? `gridcore serve --config examples/fake-demo.yaml` runs the
whole scheduler against a simulated 16 GB card, and `scripts/demo-*.sh`
replay the two scenarios below.

## Telling GridCore what a request is

Everything defaults to `interactive`. Background work says so, in whichever
way is convenient for the client:

| Channel | Example |
|---|---|
| Header | `X-GridCore-Class: background` |
| Body (any OpenAI SDK via `extra_body`) | `"gridcore": {"class": "background", "max_wait_ms": 30000}` |
| Model suffix | `"model": "nomic-embed@batch"` |

Header wins over body, body over suffix. `X-GridCore-Max-Wait-Ms` (or
`max_wait_ms`) turns an unbounded wait into a `503 queue_timeout` with
`Retry-After`.

`interactive` is for requests a person is waiting on. A coding agent that
works on its own should send `background`: running back to back as
interactive it keeps the GPU in interactive mode, and everything else gets
only `background_share` of it. On the 4060 Ti that cost the agent itself
~10% when switched to background, while the indexer next to it did 4.6
times more work and screenshot analysis got twice as fast.

Responses carry `X-GridCore-Job-Id`, `X-GridCore-Model` (the resolved id),
`X-GridCore-Queue-Ms` (how long the request waited for the GPU) and, for
family requests, `X-GridCore-Family`.

| Class | Runs | Evicts |
|---|---|---|
| `interactive` | immediately if at all possible | cold models, and idle hot ones |
| `background` | when no interactive work is active, or one step at a time after `background_max_starvation` | cold models only |
| `batch` | like background, but only when background has nothing queued | cold models only |

Large `/v1/embeddings` inputs are split into chunks (32 by default) and each
chunk is a separate scheduling step; the client still gets one response.

### Model families

A family is one name for several variants of the same model, so the
scheduler can pick the size that fits the moment:

```yaml
families:
  gemma4: { preferred: gemma4-12b, balanced: gemma4-e4b, compact: gemma4-e2b }
```

`"model": "gemma4"` from a chat gets the best variant it can have — the
12B, evicting cold or idle models just as a request for `gemma4-12b`
would — and a smaller one only when the 12B cannot be made resident at all.
From background work it gets the best variant that runs without evicting
anything and without taking a slot of the model the user is chatting with:
an ambient task next to a busy 12B lands on the E4B instead of waiting.
`"gridcore": {"quality": "balanced"}` (or `X-GridCore-Quality`) excludes
the tiers below; a request with an image only goes to vision variants;
`X-GridCore-Model` in the response names the variant.

## Configuration

```yaml
server: { listen: 127.0.0.1:8080 }

gpu:
  device: nvidia:0          # or fake
  headroom_mb: 512          # budget = min(total, vram_limit_mb) - headroom
  vram_limit_mb: null       # cap GridCore below the card, e.g. to leave room for other apps

runtimes:
  llamacpp:
    binary: /opt/llama.cpp/current/llama-server
    default_args: ["--flash-attn", "on"]

models:
  gemma4-12b:
    runtime: llamacpp
    path: ~/models/gemma-4-12b-it-qat-q4_0.gguf
    mmproj: ~/models/mmproj-gemma-4-12b-it-qat-q4_0.gguf
    capabilities: [chat, vision]
    ctx: 16384              # total context pool, shared by slots (llama-server --ctx-size)
    parallel: 2             # slots = concurrent requests for this model
    aliases: [gpt-4o, default-chat]

  nomic-embed:
    runtime: llamacpp
    path: ~/models/nomic-embed-text-v1.5.f16.gguf
    capabilities: [embedding]
    pinned: true            # preload, never evict

policy:
  classes:
    interactive: { hot_ttl: 5m }          # how long a model stays protected after interactive use
  interactive_idle_before_background: 2s  # quiet period before background may start
  background_max_starvation: 3s           # bound on how long background waits under constant chat
  embedding_chunk_size: 32
```

`gridcore models --explain` shows the VRAM estimate per model and where it
comes from. See [`config.example.yaml`](config.example.yaml) for every knob.

## What it looks like

Two scenarios from a 16 GB RTX 4060 Ti with real models
(`scripts/demo-background-vs-interactive.sh`, `scripts/demo-eviction.sh`):

```
Start background indexing: 1000 documents in 32-document chunks
150 ms later a user starts chatting
   interactive gemma4-12b     HTTP 200  1.23s  queue     0ms  A GPU scheduler manages and coordinates ...
Indexing finishes after the chat
   background  nomic-embed    HTTP 200  4.04s  items=1000

   11:20:27.194 enqueue   background embedding nomic-embed steps=32
   11:20:27.194 dispatch  background step 1/32 -> nomic-embed
   11:20:27.500 enqueue   interactive chat gemma4-12b steps=1
   11:20:27.500 dispatch  interactive step 1/1 -> gemma4-12b
   11:20:27.500 mode      gpu              interactive
   11:20:28.731 complete  interactive gemma4-12b
   11:20:30.833 mode      gpu              idle
   11:20:31.032 dispatch  background step 32/32 -> nomic-embed
   11:20:31.078 complete  background nomic-embed
```

```
Interactive request for qwen3-14b: does not fit next to gemma4-12b -> evict (it is idle), load, run
   interactive qwen3-14b      HTTP 200  2.35s  queue  1789ms  Sure! Here are three colors: 1. Red 2. ...
Now qwen3-14b is hot. A background request for gemma4-12b must NOT evict it: it waits and times out
   background  gemma4-12b     HTTP 503  3.01s  (max_wait 3s)

   11:20:33.057 evict     gemma4-12b       make room for qwen3-14b (running=0)
   11:20:33.237 unloaded  gemma4-12b       evicted
   11:20:33.237 load      qwen3-14b        reserve 9732 MB port=41002
   11:20:34.847 loaded    qwen3-14b        1.6s pid=228902 addr=127.0.0.1:41002
   11:20:34.847 dispatch  interactive step 1/1 -> qwen3-14b
   11:20:38.434 timeout   max_wait exceeded while queued
```

To watch it live, `scripts/demo-tmux.sh` opens a tmux layout with the
dashboard, a streaming chat client and a background indexer whose progress
bar visibly pauses while a chat runs (`--fake` starts the simulated GPU
first, so it works on any laptop):

```
scripts/demo-tmux.sh --fake
```

`gridcore status --watch`:

```
11:08:12  mode=idle
GPU NVIDIA GeForce RTX 4060 Ti [###################.]  15.5 / 16.0 GB  committed 15.5  util 0%

RESIDENT
  gemma4-12b       ready    hot       8.3 GB  slots 0/2
  gemma4-e2b       ready    cold      2.8 GB  slots 0/4
  gemma4-e4b       ready    cold      4.0 GB  slots 0/2
  nomic-embed      ready    pinned    0.4 GB  slots 0/4
```

Under a 5-minute mixed load (6 chatting clients, 8 background, 4 batch, 2
embedding) on that card, every class made progress and no request failed;
interactive p50 was 3.4 s for 32-token answers from a 12B model while
background steps ran alongside. Measurements of seven models on that card
informed the defaults in [`config.example.yaml`](config.example.yaml).

## Commands

| | |
|---|---|
| `gridcore serve` | run the daemon (`--config`, `--state-dir`, `--log-level`) |
| `gridcore check` | validate config, model files, GPU access |
| `gridcore models [--explain]` | list models with VRAM estimates |
| `gridcore bench [--all] <model>` | load, measure VRAM / load time / tokens/s, store the profile |
| `gridcore profiles [prune]` | list stored measurements and whether they still apply; remove the stale ones |
| `gridcore status [--watch]` | live view of GPU, queues, resident models, events |
| `make tools` | build `bin/gc-chat` (streaming chat) and `bin/gc-indexer` (background load with progress bar) |

HTTP: `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`,
`/v1/models`, `/health`, `/metrics`, `/admin/state`,
`POST /admin/models/{id}/{load,unload,enable}`.

Run it as a service with [`deploy/gridcore.service`](deploy/gridcore.service)
(`systemctl --user enable --now gridcore`).

## Status and scope

v0.1: one NVIDIA GPU, llama.cpp as the runtime, three classes, residency
and admission as described above. Tested on Ubuntu 24.04 with an RTX 4060 Ti.

Deliberately not in v0.1: cloud fallback, automatic quality tiers (pick a
smaller variant of the same model under pressure), preempting a chat
mid-generation, vLLM/MLX runtimes, AMD GPUs, multiple GPUs, authentication.
The architecture has room for all of them.

## Building

```bash
make build            # bin/gridcore for this machine
make linux            # static linux/amd64 binary
make test             # go test -race ./...
make run              # serve examples/fake-demo.yaml (no GPU needed)
```

Go 1.25+, no cgo, two dependencies (`yaml.v3`, `prometheus/client_golang`).

## License

Copyright (c) 2026 Nick Blokhin. All rights reserved. This is proprietary
software: no license is granted to copy, modify, distribute or use it
without written permission. Any use is at your own risk; the software is
provided "as is" without warranty or liability of any kind. See
[`LICENSE`](LICENSE).
