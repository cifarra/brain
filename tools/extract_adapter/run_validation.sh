#!/usr/bin/env bash
# GPU validation of a generation LoRA on ranger: one llama-server at a time, each phase under the
# shared GPU lock. Produces collect files + validation.json in $OUT.
#
#   tools/extract_adapter/run_validation.sh
#
# Override with env: BASE_GGUF, MERGED_GGUF, STOCK_LOCAL_GGUF (optional), ADAPTER, OUT, IMAGE, PORT.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
M=${BRAIN_MODELS:-$HOME/srv/models}
LAB=${BRAIN_LAB:-$HOME/srv/brain-lab}
BASE_GGUF=${BASE_GGUF:-$M/gguf/qwen3.5-9b/Qwen3.5-9B-Q8_0.gguf}
MERGED_GGUF=${MERGED_GGUF:-$LAB/abliteration/wangzhang-merged-Q8_0.gguf}
STOCK_LOCAL_GGUF=${STOCK_LOCAL_GGUF:-$LAB/abliteration/stock-local-Q8_0.gguf}
ADAPTER=${ADAPTER:-$M/adapters/abliteration-wangzhang-r1.gguf}
OUT=${OUT:-$LAB/abliteration/validation}
IMAGE=${IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda-b11580@sha256:c4348afb91499b10fad12f92626c7a8ca8b69d6475d10c793b810cb61237ce00}
PORT=${PORT:-18080}
LOCK=$LAB/gpu.lock
URL=http://127.0.0.1:$PORT
mkdir -p "$OUT" "$LAB"
V="python3 $HERE/validate.py"

serve() {  # serve <name> <model> [extra llama-server args...]
  local name=$1 model=$2; shift 2
  docker rm -f "abl-val" >/dev/null 2>&1 || true
  docker run -d --rm --name abl-val --gpus all -p 127.0.0.1:$PORT:8080 \
    -v $M:$M:ro -v $LAB:$LAB:ro "$IMAGE" \
    -m "$model" -ngl 99 -c 8192 -np 1 --host 0.0.0.0 --port 8080 --seed 42 --no-webui "$@" >/dev/null
  echo "[$name] started $(basename "$model") $*"
}
stop() { docker logs abl-val > "$OUT/server-$1.log" 2>&1 || true; docker rm -f abl-val >/dev/null 2>&1 || true; }

inner() {  # runs while holding the GPU lock: inner <name> <model> <server args...> -- <collect args...>
  local name=$1 model=$2; shift 2
  local sargs=()
  while [[ $# -gt 0 && $1 != -- ]]; do sargs+=("$1"); shift; done
  [[ ${1:-} == -- ]] && shift
  local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  echo "[$name] free VRAM ${free} MiB"
  [[ $free -gt 11500 ]] || { echo "[$name] not enough free VRAM"; exit 3; }
  serve "$name" "$model" "${sargs[@]}"
  trap "stop $name" EXIT
  $V collect --url $URL --label "$name" --out "$OUT/$name.json" "$@"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader > "$OUT/vram-$name.txt" || true
}
if [[ ${1:-} == _inner ]]; then shift; inner "$@"; exit; fi
if [[ ${1:-} == _speed ]]; then
  shift; name=$1 configs=$2; shift 3
  serve "$name" "$BASE_GGUF" --cache-ram 0 "$@"; trap "stop $name" EXIT
  $V speed --url $URL --configs "$configs" --reps 7 --tokens 512 --out "$OUT/$name.json"
  exit
fi

phase() {  # phase <name> ... : skip if done, else run inner under the lock
  [[ -s $OUT/$1.json ]] && { echo "[$1] exists, skipping"; return; }
  flock "$LOCK" "$0" _inner "$@"
}

phase merged "$MERGED_GGUF" -- --ref self
phase stock "$BASE_GGUF" -- --ref "$OUT/merged.json" --speed
# --cache-ram 0: llama-server's RAM prompt cache is keyed by tokens only, not by the LoRA set, so
# with it on, a request can resume from a state computed under another adapter scale (see README).
phase lora "$BASE_GGUF" --lora "$ADAPTER" --lora-init-without-apply --cache-ram 0 -- \
  --lora-scales 1,0,none --tf-scales 1 --ref "$OUT/merged.json" --speed
phase stock_after "$BASE_GGUF" -- --speed
[[ -f $STOCK_LOCAL_GGUF ]] && phase stock_local "$STOCK_LOCAL_GGUF" -- --ref "$OUT/merged.json"

# dedicated decode-speed runs: stock, then LoRA off/on interleaved, then stock again (bracketing)
speed_phase() {  # speed_phase <name> <collect-free validate.py speed args> -- <server args>
  local name=$1; shift
  [[ -s $OUT/$name.json ]] && { echo "[$name] exists, skipping"; return; }
  flock "$LOCK" "$0" _speed "$name" "$@"
}
if [[ ${1:-} != _speed ]]; then
  speed_phase speed_stock_a "stock=" --
  speed_phase speed_lora "off=0:0;on=0:1" -- --lora "$ADAPTER" --lora-init-without-apply
  speed_phase speed_stock_b "stock=" --
fi

$V report --merged "$OUT/merged.json" --stock "$OUT/stock.json" --lora "$OUT/lora.json" \
  --stock-after "$OUT/stock_after.json" \
  --speed-stock "$OUT/speed_stock_a.json" "$OUT/speed_stock_b.json" --speed-lora "$OUT/speed_lora.json" \
  $( [[ -s $OUT/stock_local.json ]] && echo --stock-local "$OUT/stock_local.json" ) \
  --out "$OUT/validation.json"
