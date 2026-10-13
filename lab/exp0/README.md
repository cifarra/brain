# Experiment 0: the 4090 sweet spot

Measures the brain's own VRAM per configuration (spec rev 5): S1 Clef-Flash quants and S2 Qwen3.5-9B quants on
ranger's RTX 4090, plus the S1 calibration gate against BF16 and an S2 quality proxy. Findings are in `REPORT.md`, and
all tables are in `results/tables.md` (regenerate with `analyze.py`).

Runtime: the pinned server image `ghcr.io/ggml-org/llama.cpp:server-cuda-b11580@sha256:c4348afb…` (build b11580,
commit 4413a1924). `llama-quantize` and gguf-py come from `ghcr.io/ggml-org/llama.cpp:full-cuda-b11580`, which is the same
commit, so nothing is built from source.

## Files

| file | what |
|---|---|
| `gen_data.py` | deterministic synthetic data, `data/clef_records.jsonl` (150 calibration records), `data/s1_workload.jsonl` (24 systemone requests in buckets ~0.5k / ~1.7k / ~6k prompt tokens), `data/s2_prompts.jsonl` (30 chat prompts) |
| `quantize_clef.sh` | BF16 → `derived/Clef-Flash-<TYPE>-hbf16.gguf`, keeping `decision.*` and `dec.blk.*` at BF16 (`--tensor-type` overrides), and appends to `derived/MANIFEST.md` |
| `gguf_tool.py`, `strip_clef.sh` | inspect a GGUF (tensor types; verifies the head stayed BF16), and strip `output.weight` (it is not lossless; see the report) |
| `bench.py` | the harness (PEP 723, `uv run`): `s1`, `s2`, `s2q`, `pair`; one JSONL line per configuration |
| `run_s1.sh` | S1 stages: `ref` (BF16 reference), `quants` (ub 2048/4096/8192 + calibration), `strip`, `extra` (noise floor, `-a8` mixes), `probe` (`Q8_0-abf16`, BF16 on GPU), `offload` (partial `-ngl`), `mixbf16` (`-abf16` mixes), `final` (profile candidates) |
| `run_s2.sh` | S2 stages: `grid`, `nommproj`, `quality`, `refine` (KV q8_0, np 2, mmproj on CPU) |
| `run_pairs.sh` + `pairs.txt` | S1+S2 pairs (one line per pair: S1 model/ub/output-on-CPU/ngl, S2 quant/ctx/KV/np/mmproj) |
| `run_queue.sh` | runs stages in sequence, e.g. `./run_queue.sh s1:final pairs:pairs.txt` (log in `~/srv/brain-lab/exp0/logs/queue.log`) |
| `analyze.py` | calibration gate and tables → `results/tables.md`, `results/summary.json` |
| `results/` | raw JSONL copied back from ranger (`s1.jsonl`, `s2.jsonl`, `pairs.jsonl`, `calib/*.jsonl`, `s2q/*.jsonl`) |

## How the measurement works

- Each configuration starts the pinned container (`--gpus all`, `~/srv/models` mounted read-only, `-fit off -ngl all`
  unless stated), waits for `/health`, and samples VRAM every 100 ms through NVML. NVML is the same counter
  `nvidia-smi --query-compute-apps=pid,used_memory` reads. Only the container's host PIDs (`docker top`) are counted, so
  other residents of the GPU are excluded. Free device memory at the start is recorded too.
- Phases: `load` (container start → healthy), `steady` (2 s idle after load), `work`/`calib` (requests running).
  "peak" is the maximum over all phases. "Load" is wall time from `docker run` to healthy. "cold" means the GGUF was evicted
  from the page cache (`posix_fadvise DONTNEED`) first, and "warm" means it was already in the page cache.
- S1: `-c = -b = -ub`, `-np 1`, `-fa on`. The workload runs one warm-up pass, then 3 timed passes per bucket. The first
  request after load is reported separately. Requests larger than `-ub` are rejected by the server and reported as
  "doesn't fit". Calibration runs every record at `ub 4096` and stores the per-option probabilities.
- S2: `-np N -c ctx·N -ub 512 -fa on -ctk/-ctv <kv> --jinja --reasoning-format deepseek --ctx-checkpoints 4`, plus
  `--mmproj mmproj-F16.gguf` and `--lora abliteration-wangzhang-r1.gguf --lora-init-without-apply` (requests use scale 1). The workload is a
  256-token chat with thinking off, one 768×768 image, then a raw-token prefill of `ctx − 192` tokens with 128 decoded
  tokens (with `np 2`, two at once). pp/tg are the server's `timings`.
- S2 quality (`s2q`): Q8_0 greedily continues 30 prompts by 24 tokens. Every quant then scores the same token prefixes
  (teacher forcing) with `n_probs 20`. The script reports top-1 agreement and an approximate KL over the top 20 plus a rest bucket.
- `pair`: S1 and S2 containers at once. An S1 loop (the largest bucket that fits, then ~0.5k) runs concurrently with the
  S2 workload, and the combined peak is the maximum of the per-sample sum.
- GPU lock: each configuration holds `~/srv/brain-lab/gpu.lock` (flock(2), compatible with `flock(1)`) only while it runs.

## Rerun

```bash
# laptop
python3 lab/exp0/gen_data.py                  # data is committed; only needed if the generator changes
tools/sync-ranger.sh lab/exp0                 # -> ranger:~/srv/brain-src/lab/exp0
# ranger (through: ssh hx 'ssh ranger ...')
cd ~/srv/brain-src/lab/exp0
./quantize_clef.sh Q6_K Q5_K_M Q4_K_M IQ4_XS Q8_0       # CPU only, ~1-3 min each; skips existing outputs
SUFFIX=-a8    EXTRA_TT="attn_=q8_0 ssm_=q8_0" ./quantize_clef.sh Q6_K Q5_K_M IQ4_XS
SUFFIX=-abf16 EXTRA_TT="attn_=bf16 ssm_=bf16" ./quantize_clef.sh Q8_0 Q6_K Q5_K_M Q4_K_M   # Q8_0-abf16 = the gate-passing S1
./strip_clef.sh ~/srv/models/gguf/clef-flash/Clef-Flash-Q4_K_M.gguf
./run_queue.sh s1:all s1:extra s1:probe s1:offload s1:mixbf16 s1:final \
               s2:grid s2:nommproj s2:quality s2:refine pairs:pairs.txt
# single configurations, e.g.
~/.local/bin/uv run bench.py s1 --model gguf/clef-flash/derived/Clef-Flash-IQ4_XS-hbf16.gguf --ctx 2048 --output-cpu
~/.local/bin/uv run bench.py s2 --model gguf/qwen3.5-9b/Qwen3.5-9B-IQ4_XS.gguf --ctx 16384 --kv q8_0 \
    --mmproj gguf/qwen3.5-9b/mmproj-F16.gguf --mmproj-cpu --lora adapters/abliteration-wangzhang-r1.gguf
~/.local/bin/uv run bench.py pair --tag 12g-A --s1-model gguf/clef-flash/derived/Clef-Flash-Q8_0-abf16-hbf16.gguf \
    --s1-ctx 2048 --s1-ngl 20 --s1-output-cpu --s2-model gguf/qwen3.5-9b/Qwen3.5-9B-IQ4_XS.gguf --s2-ctx 16384 \
    --s2-kv q8_0 --s2-mmproj gguf/qwen3.5-9b/mmproj-F16.gguf --s2-mmproj-cpu --lora adapters/abliteration-wangzhang-r1.gguf
# results land in ~/srv/brain-lab/exp0/results; copy back and analyze on the laptop:
ssh hx 'ssh ranger "tar -C ~/srv/brain-lab/exp0 -c results"' | tar -x -C lab/exp0
python3 lab/exp0/analyze.py
```
