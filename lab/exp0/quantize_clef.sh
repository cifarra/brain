#!/usr/bin/env bash
# Re-quantize Clef-Flash from BF16 with the pinned llama-quantize (full-cuda-b11580 = commit 4413a19),
# keeping the decision head (decision.*, dec.*) at its original BF16 precision.
# Usage: quantize_clef.sh TYPE [TYPE...]     e.g. quantize_clef.sh Q5_K_M Q6_K IQ4_XS Q4_K_M
#   Mixed recipes: SUFFIX=-a8 EXTRA_TT="attn_=q8_0 ssm_=q8_0" quantize_clef.sh Q6_K   (extra --tensor-type regexes,
#   applied after the head overrides; first match wins)
# Output: ~/srv/models/gguf/clef-flash/derived/Clef-Flash-<TYPE><SUFFIX>-hbf16.gguf (+ MANIFEST.md entry)
# CPU only (no GPU lock needed).
set -euo pipefail
FULL_IMG=${FULL_IMG:-ghcr.io/ggml-org/llama.cpp:full-cuda-b11580}
SRC_DIR=$HOME/srv/models/gguf/clef-flash
DER=$SRC_DIR/derived
SRC=Clef-Flash-BF16.gguf
THREADS=${THREADS:-8}
mkdir -p "$DER"
src_sha=$(cat "$DER/.sha256-$SRC" 2>/dev/null || true)
if [ -z "$src_sha" ]; then src_sha=$(sha256sum "$SRC_DIR/$SRC" | cut -d' ' -f1); echo "$src_sha" > "$DER/.sha256-$SRC"; fi
SUFFIX=${SUFFIX:-}
extra=(); for e in ${EXTRA_TT:-}; do extra+=(--tensor-type "$e"); done
for T in "$@"; do
  out=Clef-Flash-$T$SUFFIX-hbf16.gguf
  [ -f "$DER/$out" ] && { echo "exists: $out"; continue; }
  cmd=(/app/llama-quantize --tensor-type '^decision\.=bf16' --tensor-type '^dec\.blk\.=bf16' "${extra[@]}"
       /src/$SRC /der/$out.tmp "$T" "$THREADS")
  echo "+ ${cmd[*]}"
  t0=$(date +%s)
  docker run --rm --user "$(id -u):$(id -g)" -v "$SRC_DIR:/src:ro" -v "$DER:/der" --entrypoint "${cmd[0]}" "$FULL_IMG" "${cmd[@]:1}" \
    > "$DER/$out.log" 2>&1
  mv "$DER/$out.tmp" "$DER/$out"
  out_sha=$(sha256sum "$DER/$out" | cut -d' ' -f1)
  {
    echo "## $out"
    echo "- made: $(date -Is) in $(( $(date +%s) - t0 )) s, image \`$FULL_IMG\` (build b11580, commit 4413a1924)"
    echo "- command: \`docker run ... ${cmd[*]}\`"
    echo "- source: \`$SRC\` sha256 \`$src_sha\` (ggml-org/Clef-Flash-GGUF rev 6b4fe4c)"
    echo "- output sha256: \`$out_sha\`, size $(stat -c %s "$DER/$out") bytes"
    echo "- decision head (decision.*, dec.blk.*) kept at BF16 via --tensor-type overrides${EXTRA_TT:+; extra overrides: $EXTRA_TT}; no imatrix (clef has no logits)"
    echo
  } >> "$DER/MANIFEST.md"
  echo "done $out"
done
