#!/usr/bin/env bash
# Strip output.weight from a Clef GGUF (gguf-py at commit 4413a19, inside the pinned full image).
# Usage: strip_clef.sh SRC_GGUF_PATH   -> ~/srv/models/gguf/clef-flash/derived/<name>-nooutput.gguf (+ MANIFEST.md)
# NOTE: Experiment 0 found this is NOT lossless for clef (the head reads output.weight rows); kept for the record.
set -euo pipefail
FULL_IMG=${FULL_IMG:-ghcr.io/ggml-org/llama.cpp:full-cuda-b11580}
MODELS=$HOME/srv/models; DER=$MODELS/gguf/clef-flash/derived; LAB=$(cd "$(dirname "$0")" && pwd)
src=$(realpath "$1"); base=$(basename "$src" .gguf); out=$base-nooutput.gguf
rel=${src#$MODELS/}
docker run --rm --user "$(id -u):$(id -g)" -v "$MODELS:/models" -v "$LAB:/lab:ro" --entrypoint python3 "$FULL_IMG" \
  /lab/gguf_tool.py strip "/models/$rel" "/models/gguf/clef-flash/derived/$out" output.weight
src_sha=$(sha256sum "$src" | cut -d' ' -f1); out_sha=$(sha256sum "$DER/$out" | cut -d' ' -f1)
{
  echo "## $out"
  echo "- made: $(date -Is), image \`$FULL_IMG\` (gguf-py at commit 4413a1924)"
  echo "- command: \`python3 lab/exp0/gguf_tool.py strip $rel derived/$out output.weight\`"
  echo "- source: \`$rel\` sha256 \`$src_sha\`"
  echo "- output sha256: \`$out_sha\`, size $(stat -c %s "$DER/$out") bytes"
  echo "- NOT lossless for clef: without output.weight llama.cpp falls back to token_embd for the head's lexical/prior terms (see lab/exp0/REPORT.md)"
  echo
} >> "$DER/MANIFEST.md"
