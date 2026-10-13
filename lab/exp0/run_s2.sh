#!/usr/bin/env bash
# Experiment 0, S2 (Qwen3.5-9B) sweep. Each bench.py call takes ~/srv/brain-lab/gpu.lock for its own duration only.
# Stages: grid (quant x ctx, f16 KV, np 1, mmproj F16, adapter if present), nommproj, quality (n_probs vs Q8_0),
#         refine (KV q8_0 and np 2 for $TOP quants)
set -uo pipefail
cd "$(dirname "$0")"
B="$HOME/.local/bin/uv run -q bench.py"
OUT=${OUT:-$HOME/srv/brain-lab/exp0/results/s2.jsonl}
Q=gguf/qwen3.5-9b
MM="--mmproj $Q/mmproj-F16.gguf"
ADP=adapters/abliteration-wangzhang-r1.gguf
LORA=""; [ -f "$HOME/srv/models/$ADP" ] && LORA="--lora $ADP"
QUANTS=${QUANTS:-"Q8_0 Q6_K Q5_K_M Q4_K_M IQ4_XS"}
CTXS=${CTXS:-"8192 16384 32768"}
TOP=${TOP:-"IQ4_XS Q4_K_M"}
STAGE=${1:-all}

if [[ $STAGE == all || $STAGE == grid ]]; then
  for q in $QUANTS; do for c in $CTXS; do
    $B s2 --model $Q/Qwen3.5-9B-$q.gguf --ctx $c $MM $LORA --out "$OUT"
  done; done
fi
if [[ $STAGE == all || $STAGE == nommproj ]]; then
  $B s2 --model $Q/Qwen3.5-9B-Q4_K_M.gguf --ctx 16384 $LORA --out "$OUT"
fi
if [[ $STAGE == all || $STAGE == quality ]]; then
  # Q8_0 first: it writes the reference continuations (results/s2q_ref_continuations.json)
  for q in Q8_0 Q6_K Q5_K_M Q4_K_M IQ4_XS; do
    $B s2q --model $Q/Qwen3.5-9B-$q.gguf --tag s2q-$q --out "$OUT"
  done
fi
if [[ $STAGE == all || $STAGE == refine ]]; then
  for q in $TOP; do
    for c in 16384 32768; do $B s2 --model $Q/Qwen3.5-9B-$q.gguf --ctx $c --kv q8_0 $MM $LORA --out "$OUT"; done
    $B s2 --model $Q/Qwen3.5-9B-$q.gguf --ctx 16384 --np 2 $MM $LORA --out "$OUT"
    # vision encoder on the CPU (frees the mmproj VRAM; image latency measured)
    $B s2 --model $Q/Qwen3.5-9B-$q.gguf --ctx 16384 --kv q8_0 $MM --mmproj-cpu $LORA --out "$OUT"
  done
  $B s2 --model $Q/Qwen3.5-9B-IQ4_XS.gguf --ctx 8192 --kv q8_0 $MM --mmproj-cpu $LORA --out "$OUT"
  $B s2 --model $Q/Qwen3.5-9B-Q8_0.gguf --ctx 32768 --np 2 $MM $LORA --out "$OUT"
fi
