#!/bin/bash
# Build and run the image. This script was not executed on the machine that graded the files.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if ! command -v docker >/dev/null 2>&1; then
  echo "refusing: docker is not on PATH. The image was not run. Grade with make run-local." >&2
  exit 1
fi
docker build -t spandiff:local -f "${ROOT}/track_2a/Dockerfile" "$ROOT"
# Do not mount the Hugging Face cache. The grade does not load weights, and a
# read-only cache can make the evaluator's imports fail when they try to write.
docker run --rm spandiff:local
