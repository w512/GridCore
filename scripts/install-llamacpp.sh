#!/usr/bin/env bash
# Install a prebuilt llama.cpp release (server binaries + CUDA runtime) into
# /opt/llama.cpp/<tag> and point /opt/llama.cpp/current at it.
#
#   scripts/install-llamacpp.sh              # latest build, CUDA 12.8 (works with driver >= 525)
#   scripts/install-llamacpp.sh b11060       # specific build
#   FLAVOR=cuda-13.3 scripts/install-llamacpp.sh
#   FLAVOR=vulkan scripts/install-llamacpp.sh # no CUDA runtime needed
#   PREFIX=$HOME/llama.cpp scripts/install-llamacpp.sh   # without sudo
#
# Then in gridcore config:
#   runtimes:
#     llamacpp:
#       binary: /opt/llama.cpp/current/llama-server
set -euo pipefail

TAG="${1:-}"
FLAVOR="${FLAVOR:-cuda-12.8}"
PREFIX="${PREFIX:-/opt/llama.cpp}"
REPO="ggml-org/llama.cpp"

need() { command -v "$1" >/dev/null 2>&1 || { echo "error: $1 is required" >&2; exit 1; }; }
need curl; need tar; need python3

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "error: prebuilt CUDA releases are Linux x86_64 only; build from source on this platform" >&2
  exit 1
fi

if [[ -z "$TAG" ]]; then
  echo "resolving latest build..."
  TAG=$(curl -fsSL "https://api.github.com/repos/$REPO/releases?per_page=10" | python3 -c '
import json,sys
for r in json.load(sys.stdin):
    t=r["tag_name"]
    if t.startswith("b") and t[1:].isdigit():
        print(t); break')
  [[ -n "$TAG" ]] || { echo "error: could not resolve latest build tag" >&2; exit 1; }
fi
echo "tag:    $TAG"
echo "flavor: $FLAVOR"
echo "prefix: $PREFIX"

DEST="$PREFIX/llama-$TAG"
if [[ -x "$DEST/llama-server" ]]; then
  echo "already installed at $DEST"
else
  SUDO=""
  if [[ ! -d "$PREFIX" ]]; then
    if mkdir -p "$PREFIX" 2>/dev/null; then :; else need sudo; SUDO=sudo; sudo mkdir -p "$PREFIX"; sudo chown "$(id -u):$(id -g)" "$PREFIX"; fi
  fi
  [[ -w "$PREFIX" ]] || { echo "error: $PREFIX is not writable; set PREFIX or fix permissions" >&2; exit 1; }

  TMP=$(mktemp -d)
  trap 'rm -rf "$TMP"' EXIT
  BASE="https://github.com/$REPO/releases/download/$TAG"
  MAIN="llama-$TAG-bin-ubuntu-$FLAVOR-x64.tar.gz"
  echo "downloading $MAIN"
  curl -fSL --retry 3 --progress-bar -o "$TMP/$MAIN" "$BASE/$MAIN"
  mkdir -p "$DEST"
  tar -xzf "$TMP/$MAIN" -C "$DEST" --strip-components=1

  if [[ "$FLAVOR" == cuda-* ]]; then
    CUDART="cudart-llama-$TAG-bin-ubuntu-$FLAVOR-x64.tar.gz"
    echo "downloading $CUDART (CUDA runtime libraries)"
    curl -fSL --retry 3 --progress-bar -o "$TMP/$CUDART" "$BASE/$CUDART"
    tar -xzf "$TMP/$CUDART" -C "$DEST" --strip-components=1
  fi
fi

ln -sfn "llama-$TAG" "$PREFIX/current"
echo
"$PREFIX/current/llama-server" --version 2>&1 | grep -E "^version" || true
if [[ "$FLAVOR" == cuda-* ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    "$PREFIX/current/llama-server" --list-devices 2>/dev/null | grep -E "CUDA|Available" || echo "warning: llama-server did not list a CUDA device; check the NVIDIA driver"
  else
    echo "warning: nvidia-smi not found; install the NVIDIA driver before running models"
  fi
fi
echo
echo "installed: $PREFIX/current -> llama-$TAG"
echo "config:    runtimes.llamacpp.binary: $PREFIX/current/llama-server"
