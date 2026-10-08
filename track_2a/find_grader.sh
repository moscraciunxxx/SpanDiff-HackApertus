#!/bin/bash
# Print the absolute path of an interpreter that can import the evaluator.
# Exit 1 if none on PATH can. No hardcoded install prefix.
set -euo pipefail

probe="import nlpstats, numpy, jsonlines, torch, tokenizers, transformers, sys; print(sys.executable)"
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 \
    && exe=$("$candidate" -c "$probe" 2>/dev/null); then
    printf '%s\n' "$exe"
    exit 0
  fi
done
exit 1
