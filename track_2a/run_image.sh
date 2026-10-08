#!/bin/bash
# Build and run the image. This script was not executed on the machine that graded the files.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
docker build -t spandiff:local -f "${ROOT}/track_2a/Dockerfile" "$ROOT"
# The graded files do not need the weight cache. Mount it only when it exists.
if [[ -n "${HOME:-}" && -d "${HOME}/.cache/huggingface" ]]; then
  docker run --rm \
    -e HF_TOKEN \
    -e HUGGING_FACE_HUB_TOKEN \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface:ro" \
    spandiff:local
else
  docker run --rm \
    -e HF_TOKEN \
    -e HUGGING_FACE_HUB_TOKEN \
    spandiff:local
fi
