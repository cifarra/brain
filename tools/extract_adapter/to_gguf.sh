#!/usr/bin/env bash
# PEFT LoRA dir -> llama.cpp LoRA GGUF, with the pinned llama.cpp source (convert_lora_to_gguf.py) run
# inside the matching full-cuda image (which has torch/transformers), then verify_gguf.py.
#
#   tools/extract_adapter/to_gguf.sh <peft dir> <out.gguf> [outtype=f32] [base HF dir]
#
# f32 is the default on purpose: for small ranks it decodes faster than f16 on CUDA (see README).
set -euo pipefail
PEFT=$(realpath "$1"); OUTF=$(realpath -m "$2"); OUTTYPE=${3:-f32}
BASE=$(realpath "${4:-$HOME/srv/models/hf/qwen3.5-9b}")
SRC=${LLAMA_CPP_SRC:-$HOME/srv/brain-lab/llama.cpp}   # git checkout 4413a19241 (b11580)
IMAGE=${IMAGE:-ghcr.io/ggml-org/llama.cpp:full-cuda-b11580}
HERE=$(cd "$(dirname "$0")" && pwd)
[[ $(git -C "$SRC" rev-parse --short=10 HEAD) == 4413a19241 ]] || echo "WARNING: $SRC is not at 4413a19241" >&2
mkdir -p "$(dirname "$OUTF")"
mounts=(-v "$SRC:/src:ro" -v "$HERE:/tool:ro" -v "$PEFT:$PEFT:ro" -v "$BASE:$BASE:ro" -v "$(dirname "$OUTF"):$(dirname "$OUTF")")
docker run --rm --user "$(id -u):$(id -g)" -e HF_HOME=/tmp/hf -e HOME=/tmp --entrypoint python3 "${mounts[@]}" -w /src "$IMAGE" \
  /src/convert_lora_to_gguf.py --base "$BASE" "$PEFT" --outfile "$OUTF" --outtype "$OUTTYPE" 2>&1 | grep -v "hf-to-gguf:blk\." | grep -v "^Writing"
docker run --rm --user "$(id -u):$(id -g)" --entrypoint python3 "${mounts[@]}" "$IMAGE" \
  /tool/verify_gguf.py "$OUTF" "$PEFT" "$BASE/config.json"
sha256sum "$OUTF"; stat -c '%s bytes' "$OUTF"
