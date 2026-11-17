#!/usr/bin/env bash
# Extend the abliteration gate to other quants + the 100-probe refusal set (see README).
# Reference = the merged wangzhang Q8_0 (validation/merged.json); the stock Q8_0 / adapter Q8_0
# teacher-forced runs (stock.json, lora.json) from run_validation.sh are reused.
# Each server run holds the GPU lock on its own, so other jobs can interleave.
#
#   tools/extract_adapter/run_quant_gate.sh [quants=Q4_K_M,Q5_K_M,Q6_K,IQ4_XS]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
M=${BRAIN_MODELS:-$HOME/srv/models}; LAB=${BRAIN_LAB:-$HOME/srv/brain-lab}
GG=$M/gguf/qwen3.5-9b
MERGED_GGUF=${MERGED_GGUF:-$LAB/abliteration/wangzhang-merged-Q8_0.gguf}
ADAPTER=${ADAPTER:-$M/adapters/abliteration-wangzhang-r1.gguf}
OUT=${OUT:-$LAB/abliteration/validation}
IMAGE=${IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda-b11580@sha256:c4348afb91499b10fad12f92626c7a8ca8b69d6475d10c793b810cb61237ce00}
PORT=${PORT:-18080}; URL=http://127.0.0.1:$PORT; LOCK=$LAB/gpu.lock
V="python3 $HERE/validate.py"
LORA_ARGS="--lora $ADAPTER --lora-init-without-apply"

if [[ ${1:-} == _run ]]; then   # _run <name> <model> "<server args>" "<validate.py args>"
  name=$2 model=$3 sargs=$4 vargs=$5
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  echo "[$name] free VRAM ${free} MiB"; [[ $free -gt 11500 ]] || { echo "[$name] not enough VRAM"; exit 3; }
  docker rm -f abl-val >/dev/null 2>&1 || true
  docker run -d --rm --name abl-val --gpus all -p 127.0.0.1:$PORT:8080 -v $M:$M:ro -v $LAB:$LAB:ro "$IMAGE" \
    -m "$model" -ngl 99 -c 8192 -np 1 --host 0.0.0.0 --port 8080 --seed 42 --no-webui --cache-ram 0 $sargs >/dev/null
  trap 'docker logs abl-val > "$OUT/server-'$name'.log" 2>&1 || true; docker rm -f abl-val >/dev/null 2>&1 || true' EXIT
  $V $vargs --url $URL --out "$OUT/$name.json"
  exit
fi
run() { [[ -s $OUT/$1.json ]] && { echo "[$1] exists, skipping"; return; }; flock "$LOCK" "$0" _run "$@"; }

# 100-probe refusal rerun on Q8_0
run q8_probes_merged "$MERGED_GGUF" "" "collect --label merged --sets probes"
run q8_probes_stock  "$GG/Qwen3.5-9B-Q8_0.gguf" "" "collect --label stock --sets probes"
run q8_probes_lora   "$GG/Qwen3.5-9B-Q8_0.gguf" "$LORA_ARGS" "collect --label lora --sets probes --lora-scales 1,0"

QUANTS=${1:-Q4_K_M,Q5_K_M,Q6_K,IQ4_XS}
for Q in ${QUANTS//,/ }; do
  F=$GG/Qwen3.5-9B-$Q.gguf
  run q_${Q}_speed_stock_a "$F" "" "speed --configs stock= --reps 7 --tokens 512"
  run q_${Q}_stock "$F" "" "collect --label stock-$Q"
  run q_${Q}_lora  "$F" "$LORA_ARGS" "collect --label lora-$Q --lora-scales 1,0 --tf-scales 0,1 --ref $OUT/merged.json"
  run q_${Q}_speed_lora "$F" "$LORA_ARGS" "speed --configs off=0:0;on=0:1 --reps 7 --tokens 512"
  run q_${Q}_speed_stock_b "$F" "" "speed --configs stock= --reps 7 --tokens 512"
done

$V quants --dir "$OUT" --quants "$QUANTS" --base-validation "$OUT/validation.json" --out "$OUT/validation_v2.json"
