# Experiment 0: the 4090 sweet spot (report)

Date 2026-10-13. ranger RTX 4090 24 GiB, PCIe gen 1. Runtime: pinned server-cuda-b11580@sha256:c4348afb… (commit 4413a1924).
Scope per spec rev 5: the brain's own per-PID peak VRAM (NVML, host PIDs of our containers only).
Other residents are ignored. Free VRAM at every recorded start was 14.80 GiB (others held ~8.7 GiB).
Budgets are in GiB, as in the spec's budget table. Full tables: results/tables.md. Raw data: results/*.jsonl. Rerun: README.md.

## Recommendation
| profile | S1 (Clef) | S2 (Qwen3.5-9B + abliteration) | combined peak (measured) | S1 speed (solo, warm) | S2 speed (solo) | S1 gate |
|---|---|---|---|---|---|---|
| ranger-12g | Clef-Flash-Q8_0-abf16-hbf16, -c/-b/-ub 2048, -ngl 20, -ot output.weight=CPU | IQ4_XS, ctx 16384, KV q8_0, np 1, ub 512, mmproj F16 on CPU | 11.39 GiB (S1 6.19 + S2 5.20) | p50/p95 1.41/1.44 s at ~1.7k tok; 1.25 s at ~0.6k | pp 8.7k t/s, tg 128-134; image 3.7 s | PASS |
| ranger-10g | same S1 file, ub 2048, -ngl 14, output on CPU | IQ4_XS, ctx 8192, KV q8_0, np 1, mmproj on CPU | 9.58 GiB (4.55 + 5.03) | 1.94/1.97 s at ~1.7k | pp 8.9k, tg 132-134; image 3.6 s | PASS |
| ranger-max | same S1 file, ub 8192, all layers on GPU, output on CPU | Q8_0, ctx 32768, KV f16, np 1, mmproj F16 on GPU | 21.87 GiB (sum of solo peaks 11.22 + 10.65) | 0.08 / 0.23 / 0.83 s at 0.6k / 1.7k / 5.9k | pp 8.0k, tg 80-87; image 0.6 s | PASS |

12g and 10g miss the spec's S1 p50 ≤ 1 s bar because S1 is partially offloaded.
Fast alternatives (measured):
- ranger-12g-fast: official Q8_0, ub 2048, ngl 26, output on CPU, same S2. 11.59 GiB, 0.73 s. Gate fails on max |Δp| 0.119 (one record); top-1 99.2% and p99 0.026 pass.
- ranger-10g-fast: Q6_K-a8-hbf16, ngl 22, same S2 as 10g. 9.83 GiB, 0.92 s. Gate fails: p99 0.042, max 0.108.

I recommend the gate-passing set: calibrated probabilities are S1's purpose, and 1.4-1.9 s per decision is fine for pipeline work.

Notes:
- 12g has ~0.6 GiB of headroom: ngl 22 or ub 4096 (+0.43 GiB) would use it. Neither was measured.
- ub 2048 caps S1 prompts at 2048 tokens.
- ranger-max was not run as a pair (14.8 GiB free). Peaks are additive: the add-1 pair measured 13.11 GiB against a solo sum of 13.13. With S1 at ub 4096 it would be 20.93 GiB.
- If decimal GB was meant, drop about 1 S1 layer per 0.25 GiB (+~0.08 s per layer).

## Findings
1. No standard Clef quant passes the gate (top-1 ≥ 98% excluding near-ties, p99 |Δp| ≤ 0.03, max |Δp| ≤ 0.10).
   Attention and linear-attention precision dominate.
   Q8_0 FFN with attn_*/ssm_* at BF16 (Q8_0-abf16-hbf16) passes: top-1 100%, p99 0.013, max 0.038, 149/150 records.
2. The official Q4_K_M keeps attention and SSM tensors at Q8_0 and only the FFN at Q4_K, so it beats a stock Q5_K_M.
3. Head precision doesn't matter: Q8_0 with a Q8_0 head ≈ Q8_0 with a BF16 head (p99 0.0261 vs 0.0260, max 0.119 vs 0.118).
4. Stripping output.weight is not lossless.
   The head reads output.weight rows (option lexical embeddings and the prior); without it llama.cpp falls back to token_embd.
   0/150 records stay within 1e-6, max |Δp| is 0.25, and it saves only 0.25 GiB.
5. -ot output.weight=CPU is bit-identical (max |Δp| 0). It saves 0.78 GiB (Q6_K output) or 1.0 GiB (Q8_0), for +10-15 ms.
6. S1 partial offload: the CPU-resident weights stream over PCIe gen 1 (~3 GiB/s) on every request.
   Each GiB on CPU adds ~0.33 s at any prompt size; each layer saves ~0.21-0.27 GiB.
7. The reference is stable: BF16 at ngl 24 vs ngl 28 differs by at most 0.0001.
8. S2:
   - KV costs 32 KiB/token in f16; q8_0 halves it.
   - mmproj on GPU costs +1.14 GiB; on CPU an image takes 3.7 s instead of 0.5 s.
   - IQ4_XS ≈ Q4_K_M in quality (KL 0.020 vs 0.021) and is smaller and faster.
   - Steady VRAM ≈ peak.
9. Contention: S2 decode drops to 41 t/s while S1 prefills back to back at full GPU, and to 83-103 t/s in the profile pairs.
   S1 rises to 2.5-3.5 s while S2 prefills (n = 3-10).

## S1 VRAM and latency (warm p50 s at ~0.6k / ~1.7k / ~5.9k tokens)
| S1 | ub | output.weight | file GiB | peak GiB | 0.6k | 1.7k | 5.9k | load warm (cold) s |
|---|---|---|---|---|---|---|---|---|
| BF16 all layers | 2048 | CPU | 16.92 | 14.06 | 0.095 | 0.257 | — | 15.7 |
| Q8_0-abf16-hbf16 | 2048/4096/8192 | CPU | 10.92 | 9.83/10.28/11.22 | 0.080 | 0.221 | 0.829 | 4.7-8.9 (22.0) |
| Q8_0 official | 2048/4096/8192 | GPU | 8.99 | 8.91/9.36/10.30 | 0.073 | 0.197 | 0.737 | 4.7-24.6 (11.4) |
| Q8_0 official | 2048 | CPU | 8.99 | 7.90 | 0.077 | 0.207 | — | 10.3 |
| Q6_K-abf16-hbf16 | 4096 | CPU | 9.37 | 9.19 | 0.089 | 0.242 | — | 4.8 (18.5) |
| Q6_K-a8-hbf16 | 2048 / 4096 | CPU / GPU | 7.55 | 6.92 / 8.14 | 0.084 | 0.228 | — | 3.9-15.8 |
| Q6_K-hbf16 | 2048/4096/8192 | GPU | 7.08 | 7.23/7.67/8.62 | 0.080 | 0.218 | 0.835 | 3.6-8.1 (8.0) |
| Q5_K_M-abf16-hbf16 | 4096 | CPU | 8.75 | 8.69 | 0.084 | 0.233 | — | 4.2 (19.2) |
| Q5_K_M-a8-hbf16 | 2048 | CPU | 6.93 | 6.42 | 0.079 | 0.217 | — | 15.0 |
| Q5_K_M-hbf16 | 2048/4096/8192 | GPU | 6.24 | 6.51/6.96/7.90 | 0.076 | 0.207 | 0.779 | 3.3-7.6 (7.2) |
| Q4_K_M official | 2048/4096/8192 | GPU | 6.04 | 6.43/6.88/7.82 | 0.073 | 0.198 | 0.749 | 3.3-3.7 (7.4) |
| Q4_K_M official | 4096 | CPU | 6.04 | 6.10 | 0.079 | 0.213 | — | 7.4 |
| Q4_K_M nooutput (stripped) | 4096 | absent | 5.26 | 6.63 | 0.074 | 0.199 | — | 3.5 |
| IQ4_XS-a8-hbf16 | 2048 | CPU | 6.01 | 5.65 | 0.077 | 0.210 | — | 3.1 |
| Q4_K_M-hbf16 | 2048/4096/8192 | GPU | 5.45 | 5.84/6.29/7.23 | 0.074 | 0.202 | 0.761 | 3.0-4.8 (6.6) |
| IQ4_XS-hbf16 | 2048/4096/8192 | GPU | 5.10 | 5.52/5.96/6.90 | 0.071 | 0.196 | 0.751 | 3.3-14.3 (7.2) |

- The compute buffer grows +0.43 GiB from ub 2048 to 4096, and +0.95-1.35 GiB from 4096 to 8192.
- The first request after a load takes 0.6-0.9 s.
- Warm loads are noisy (page-cache churn on the 30 GB host). Cold means the file was evicted first.

Partial offload (ub 2048, output on CPU):
| S1 | ngl | peak GiB | 0.6k p50 | 1.7k p50 |
|---|---|---|---|---|
| Q8_0-abf16-hbf16 | all / 20 / 14 | 9.83 / 6.19 / 4.55 | 0.08 / 1.25 / 1.78 | 0.22 / 1.41 / 1.94 |
| Q8_0 | all / 26 / 24 / 16 | 7.90 / 6.39 / 5.96 / 4.25 | 0.08 / 0.59 / 0.72 / 1.28 | 0.21 / 0.73 / 0.87 / 1.43 |
| Q6_K-a8-hbf16 | all / 26 / 22 / 20 | 6.92 / 5.51 / 4.79 / 4.44 | 0.08 / 0.52 / 0.76 / 0.87 | 0.23 / 0.68 / 0.92 / 1.03 |
| Q5_K_M-a8-hbf16 | 24 | 4.79 | 0.59 | 0.75 |
| Q4_K_M official | 24 | 4.29 | 0.52 | 0.67 |

## S1 calibration vs BF16
Setup: 150 records (data/clef_records.jsonl): 60 shot assignments (2-15 shots, 1-3 questions), 20 routing, 22 sentiment, 18 urgency scores (3-10 levels), 15 noul and 15 mixed.
Prompts are 130-3735 tokens; all ran at ub 4096 (IQ4_XS-a8 at ub 2048, on the 111 records that fit).
A record passes if top-1 agrees on all its non-tie questions and its max |Δp| ≤ 0.03.

| quant | recipe | top-1 | p99 | max | mean KL | records passing | gate |
|---|---|---|---|---|---|---|---|
| Q8_0-abf16-hbf16 | FFN Q8_0, attn/ssm BF16, head BF16 | 100.0% | 0.0129 | 0.0380 | 0.00008 | 149/150 | PASS |
| Q6_K-abf16-hbf16 | FFN Q6_K, attn/ssm BF16 | 99.6% | 0.0310 | 0.0681 | 0.00039 | 141/150 | fail (p99, by 0.001) |
| Q8_0 official | all Q8_0 | 99.2% | 0.0261 | 0.1188 | 0.00039 | 142/150 | fail (max) |
| Q8_0-hbf16 | all Q8_0, head BF16 | 99.2% | 0.0260 | 0.1181 | 0.00038 | 142/150 | fail (max) |
| Q6_K-a8-hbf16 | FFN Q6_K, attn/ssm Q8_0 | 99.2% | 0.0420 | 0.1078 | 0.00072 | 136/150 | fail |
| Q5_K_M-abf16-hbf16 | FFN Q5_K, attn/ssm BF16 | 99.6% | 0.0500 | 0.1051 | 0.00094 | 135/150 | fail |
| Q5_K_M-a8-hbf16 | FFN Q5_K, attn/ssm Q8_0 | 98.4% | 0.0612 | 0.1500 | 0.00157 | 125/150 | fail |
| Q6_K-hbf16 | stock | 99.2% | 0.0629 | 0.1212 | 0.00135 | 129/150 | fail |
| Q4_K_M official | FFN Q4_K, attn/ssm Q8_0 | 98.8% | 0.0827 | 0.2562 | 0.00410 | 105/150 | fail |
| IQ4_XS-a8-hbf16 | FFN IQ4_XS, attn/ssm Q8_0 | 98.8% | 0.0914 | 0.1720 | 0.00250 | 89/111 | fail |
| Q4_K_M-abf16-hbf16 | FFN Q4_K, attn/ssm BF16 | 98.0% | 0.0944 | 0.2914 | 0.00379 | 110/150 | fail |
| Q5_K_M-hbf16 | stock | 96.5% | 0.1191 | 0.3097 | 0.00653 | 95/150 | fail |
| Q4_K_M-hbf16 | stock | 97.2% | 0.1499 | 0.4266 | 0.01320 | 80/150 | fail |
| IQ4_XS-hbf16 | stock, no imatrix | 95.3% | 0.2040 | 0.6690 | 0.02059 | 89/150 | fail |
| Q4_K_M nooutput | stripped | 97.6% | 0.1709 | 0.2937 | 0.01712 | 39/150 | fail |
| Q4_K_M, output on CPU | | identical to Q4_K_M | | | | 105/150 | fail |
| BF16 ngl 28 | noise floor | 100% | 0.0000 | 0.0001 | 0 | 150/150 | — |

Where quants fail:
- Shot assignment and urgency fail most (Q8_0 4/60 and 2/18; Q4_K_M 23/60 and 9/18), then mixed multi-question records. Noul and routing are robust.
- 4-15 options fail far more often than 2-3 (Q4_K_M 38/96 vs 7/54).
- Longer prompts fail more often (Q4_K_M 6/11 at ≥ 3k tokens vs 13/64 under 500).
- The worst records are near coin flips: r011 is 0.445 vs 0.374 in BF16, the worst or second-worst record for 7 quants. Q8_0 flips it.
- Stripping uniquely breaks noul and routing (12/15 and 18/20 fail).

## S2 (np 1 unless noted; ub 512, fa on, jinja, deepseek reasoning, 4 checkpoints, abliteration at scale 1)
Workload: chat, then one image (answered correctly in every configuration), then a prefill of ctx − 192 tokens and 128 decoded tokens.

| S2 | ctx | KV | mmproj | peak GiB | tg short / long t/s | pp long t/s |
|---|---|---|---|---|---|---|
| Q8_0 | 8k/16k/32k | f16 | GPU | 9.88/10.13/10.65 | 87 / 76-86 | 8.0-8.6k |
| Q8_0, np 2 | 32k | f16 | GPU | 11.70 | 88 / 23+68 | 7.0k+6.7k |
| Q6_K | 8k/16k/32k | f16 | GPU | 8.18/8.44/8.95 | 105 / 94-103 | 7.3-8.0k |
| Q5_K_M | 8k/16k/32k | f16 | GPU | 7.48/7.74/8.26 | 115 / 102-112 | 7.6-8.2k |
| Q4_K_M | 8k/16k/32k | f16 | GPU | 6.77/7.03/7.54 | 127 / 111-124 | 7.7-8.4k |
| Q4_K_M | 16k/32k | q8_0 | GPU | 6.81/7.16 | 124 / 113-120 | 7.6-8.2k |
| Q4_K_M | 16k | f16 | none | 5.89 | 127 / 119 | 8.3k |
| Q4_K_M | 16k | q8_0 | CPU | 5.68 | 125 / 119 | 8.2k |
| Q4_K_M, np 2 | 16k | f16 | GPU | 7.57 | 127 / 45+100 | 6.6k+6.7k |
| IQ4_XS | 8k/16k/32k | f16 | GPU | 6.29/6.55/7.06 | 137 / 119-133 | 8.2-9.0k |
| IQ4_XS | 16k/32k | q8_0 | GPU | 6.34/6.68 | 134 / 122-128 | 8.1-8.7k |
| IQ4_XS | 8k/16k | q8_0 | CPU | 5.04/5.20 | 134 / 128-132 | 8.7-8.9k |
| IQ4_XS, np 2 | 16k | f16 | GPU | 7.10 | 137 / 48+106 | 7.1k+7.1k |

Quality proxy vs Q8_0 (30 prompts × 24 teacher-forced positions, n_probs 20):
| quant | top-1 | KL | p99 |Δp| |
|---|---|---|---|
| Q6_K | 97.8% | 0.0020 | 0.087 |
| Q5_K_M | 97.0% | 0.0045 | 0.102 |
| Q4_K_M | 94.1% | 0.0210 | 0.264 |
| IQ4_XS | 94.1% | 0.0203 | 0.293 |

## Pairs (S1 loop concurrent with the S2 workload; combined = max of the per-sample sum)
| pair | S1 | S2 | combined GiB | S1 / S2 peak | S2 pp / tg under load | S1 p50/p95 under load |
|---|---|---|---|---|---|---|
| 12g-A (ranger-12g) | Q8_0-abf16 ub2048 ngl20 ocpu | IQ4_XS 16k q8_0 mmproj-CPU | 11.39 | 6.19 / 5.20 | 6302 / 89.6 | 2.55 / 3.15 (n=5) |
| 12g-B (fast) | Q8_0 ub2048 ngl26 ocpu | same | 11.59 | 6.39 / 5.20 | 6082 / 82.9 | 1.09 / 1.98 (n=10) |
| 10g-A (ranger-10g) | Q8_0-abf16 ub2048 ngl14 ocpu | IQ4_XS 8k q8_0 mmproj-CPU | 9.58 | 4.55 / 5.03 | 7683 / 102.7 | 3.46 (n=3) |
| 10g-B (fast) | Q6_K-a8 ub2048 ngl22 ocpu | same | 9.83 | 4.79 / 5.04 | 6636 / 87.8 | 1.60 / 2.44 (n=6) |
| add-1 (additivity) | Q4_K_M ub4096 all ocpu | Q4_K_M 16k f16 mmproj-GPU | 13.11 (solo sum 13.13) | 6.08 / 7.03 | 4585 / 40.8 | 0.24 / 1.03 (n=34) |

## Derived GGUFs
- Location: ~/srv/models/gguf/clef-flash/derived/; MANIFEST.md records each file's command, source sha256 and output sha256.
- Head kept at BF16 via --tensor-type '^decision\.=bf16' '^dec\.blk\.=bf16', verified by inspection. No imatrix.
- -a8 = attn_*/ssm_* at Q8_0; -abf16 = those at BF16.
- Recommended: Clef-Flash-Q8_0-abf16-hbf16.gguf, sha256 56dcd3f662d38901715fe47144b38df21f8261d7ab33060e5d138ec22e9b342c.
- The -nooutput file is kept as evidence only. Don't use it.

## Caveats and not done
- The ranger-max pair was not run (it doesn't fit next to the current residents); its number is the solo sum.
- S1 latency under contention has few samples.
- The upstream joint_schema_model.py cross-check was not run.
- Calibration ran at ub 4096 only.
- Clef image input (S1 mmproj) was not measured.
- The adapter's own VRAM was not isolated.
- Possible refinements, untested: BF16 for only the linear-attention or only the full-attention tensors, or Q6_K-abf16 variants (it misses p99 by 0.001). Either could shrink the gate-passing S1 and cut the 12g/10g latency.
- full-cuda-b11580 supplied llama-quantize and gguf-py, so nothing was built from source.
