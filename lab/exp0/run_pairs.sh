#!/usr/bin/env bash
# Experiment 0, pairs: S1 + S2 serving concurrently; combined per-PID peak. One lock hold per pair.
# Edit PAIRS to the candidates chosen from the S1/S2 sweeps. Format per line:
#   tag | S1 model | S1 ctx | S1 output on CPU (0/1) | S1 ngl | S2 quant | S2 ctx/slot | S2 KV | S2 np | mmproj (0 none, 1 GPU, 2 CPU)
set -uo pipefail
cd "$(dirname "$0")"
B="$HOME/.local/bin/uv run -q bench.py"
OUT=${OUT:-$HOME/srv/brain-lab/exp0/results/pairs.jsonl}
G=gguf/clef-flash; D=$G/derived; Q=gguf/qwen3.5-9b
ADP=adapters/abliteration-wangzhang-r1.gguf
LORA=""; [ -f "$HOME/srv/models/$ADP" ] && LORA="--lora $ADP"
PAIRS_FILE=${1:-pairs.txt}
grep -v '^\s*#' "$PAIRS_FILE" | grep -v '^\s*$' | while IFS='|' read -r tag s1 c1 ocpu ngl q2 c2 kv np mm; do
  tag=$(echo $tag); s1=$(echo $s1); q2=$(echo $q2)
  args=(pair --tag "$tag" --s1-model "$s1" --s1-ctx $c1 --s1-ngl $(echo $ngl) --s2-model "$Q/Qwen3.5-9B-$q2.gguf" --s2-ctx $c2 --s2-kv $kv --s2-np $np --out "$OUT")
  [ "$(echo $ocpu)" = 1 ] && args+=(--s1-output-cpu)
  [ "$(echo $mm)" != 0 ] && args+=(--s2-mmproj $Q/mmproj-F16.gguf)
  [ "$(echo $mm)" = 2 ] && args+=(--s2-mmproj-cpu)
  $B "${args[@]}" $LORA < /dev/null
done
