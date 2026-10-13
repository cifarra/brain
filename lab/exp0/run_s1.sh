#!/usr/bin/env bash
# Experiment 0, S1 (Clef) sweep. Each bench.py call takes ~/srv/brain-lab/gpu.lock for its own duration only.
# Results: ~/srv/brain-lab/exp0/results/s1.jsonl and results/calib/<tag>.jsonl
set -uo pipefail
cd "$(dirname "$0")"
B="$HOME/.local/bin/uv run -q bench.py"
OUT=${OUT:-$HOME/srv/brain-lab/exp0/results/s1.jsonl}
G=gguf/clef-flash; D=$G/derived
QUANTS=${QUANTS:-"$G/Clef-Flash-Q8_0.gguf $D/Clef-Flash-Q6_K-hbf16.gguf $D/Clef-Flash-Q5_K_M-hbf16.gguf $G/Clef-Flash-Q4_K_M.gguf $D/Clef-Flash-Q4_K_M-hbf16.gguf $D/Clef-Flash-IQ4_XS-hbf16.gguf"}
STAGE=${1:-all}

if [[ $STAGE == all || $STAGE == ref ]]; then
  # BF16 reference: output.weight on CPU and partial offload so it fits next to the other residents; speed is irrelevant
  $B s1 --model $G/Clef-Flash-BF16.gguf --ctx 4096 --output-cpu --ngl ${BF16_NGL:-24} --no-workload --calib --calib-name BF16 --out "$OUT"
fi
if [[ $STAGE == all || $STAGE == quants ]]; then
  for m in $QUANTS; do
    n=$(basename $m .gguf); n=${n#Clef-Flash-}
    # ctx 4096: cold+warm load, workload, calibration
    $B s1 --model $m --ctx 4096 --cold --calib --calib-name $n --out "$OUT"
    for c in 2048 8192; do $B s1 --model $m --ctx $c --out "$OUT"; done
  done
fi
if [[ $STAGE == all || $STAGE == strip ]]; then
  # stripped vs unstripped (official Q4_K_M), and the lossless alternative: output.weight kept on the CPU
  $B s1 --model $D/Clef-Flash-Q4_K_M-nooutput.gguf --ctx 4096 --calib --calib-name Q4_K_M-nooutput --out "$OUT"
  $B s1 --model $G/Clef-Flash-Q4_K_M.gguf --ctx 4096 --output-cpu --calib --calib-name Q4_K_M-ocpu --out "$OUT"
fi
if [[ $STAGE == extra ]]; then
  # reference noise floor: BF16 again with a different CPU/GPU split (ngl 28 vs 24)
  $B s1 --model $G/Clef-Flash-BF16.gguf --ctx 4096 --output-cpu --ngl 28 --no-workload --calib --calib-name BF16-ngl28 --out "$OUT"
  # head precision at Q8_0, and mixed recipes (attention + linear-attention tensors at Q8_0, FFN at Q6_K / Q5_K)
  for n in Q8_0-hbf16 Q6_K-a8-hbf16 Q5_K_M-a8-hbf16; do
    $B s1 --model $D/Clef-Flash-$n.gguf --ctx 4096 --cold --calib --calib-name $n --out "$OUT"
  done
fi
if [[ $STAGE == probe ]]; then
  # can anything quantized pass the gate? attention + linear-attention tensors at BF16 on a Q8_0 body
  $B s1 --model $D/Clef-Flash-Q8_0-abf16-hbf16.gguf --ctx 4096 --output-cpu --calib --calib-name Q8_0-abf16-hbf16 --out "$OUT"
  # BF16 with everything on the GPU except token_embd/output (the "max" S1 if it fits); ub 2048 to keep it small
  $B s1 --model $G/Clef-Flash-BF16.gguf --ctx 2048 --output-cpu --ngl all --out "$OUT"
fi
if [[ $STAGE == offload ]]; then
  # partial offload: large-batch ops on CPU-resident layers still run on the GPU, streaming the weights over PCIe
  # (gen 1 here), so S1 latency grows ~ CPU-resident bytes / PCIe bandwidth while VRAM shrinks
  $B s1 --model $D/Clef-Flash-IQ4_XS-a8-hbf16.gguf --ctx 2048 --output-cpu --calib --calib-name IQ4_XS-a8-hbf16 --passes 2 --out "$OUT"
  for spec in "$G/Clef-Flash-Q8_0.gguf 16" "$G/Clef-Flash-Q8_0.gguf 24" "$D/Clef-Flash-Q6_K-a8-hbf16.gguf 20" "$D/Clef-Flash-Q6_K-a8-hbf16.gguf 26" \
              "$G/Clef-Flash-Q4_K_M.gguf 24" "$D/Clef-Flash-Q5_K_M-a8-hbf16.gguf 24"; do
    set -- $spec
    $B s1 --model $1 --ctx 2048 --output-cpu --ngl $2 --passes 2 --out "$OUT"
  done
  for m in $G/Clef-Flash-Q8_0.gguf $D/Clef-Flash-Q6_K-a8-hbf16.gguf $D/Clef-Flash-Q5_K_M-a8-hbf16.gguf; do
    $B s1 --model $m --ctx 2048 --output-cpu --passes 2 --out "$OUT"
  done
fi
if [[ $STAGE == mixbf16 ]]; then
  # attention + linear-attention tensors at BF16, FFN at lower precision (Q8_0-abf16 passed the gate)
  for n in Q6_K-abf16-hbf16 Q5_K_M-abf16-hbf16 Q4_K_M-abf16-hbf16; do
    $B s1 --model $D/Clef-Flash-$n.gguf --ctx 4096 --output-cpu --cold --calib --calib-name $n --passes 2 --out "$OUT"
  done
fi
if [[ $STAGE == final ]]; then
  # solo S1 latency/VRAM for the profile candidates (partial offload sized for the pairs)
  for spec in "$D/Clef-Flash-Q8_0-abf16-hbf16.gguf 20" "$D/Clef-Flash-Q8_0-abf16-hbf16.gguf 14" "$D/Clef-Flash-Q8_0-abf16-hbf16.gguf all" \
              "$G/Clef-Flash-Q8_0.gguf 26" "$D/Clef-Flash-Q6_K-a8-hbf16.gguf 22"; do
    set -- $spec
    $B s1 --model $1 --ctx 2048 --output-cpu --ngl $2 --passes 2 --out "$OUT"
  done
  $B s1 --model $D/Clef-Flash-Q8_0-abf16-hbf16.gguf --ctx 8192 --output-cpu --cold --passes 2 --out "$OUT"
fi
