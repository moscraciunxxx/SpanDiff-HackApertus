#!/bin/bash
# Image entrypoint. grade_dev.sh fetches the development kit when it is absent.
# Does not load Apertus weights.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "${ROOT}/track_2a/grade_dev.sh"
