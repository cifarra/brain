# extract_adapter

Turn "a fine-tuned or edited checkpoint" into "the base checkpoint + a LoRA adapter" that
llama.cpp can switch on and off per request. The first user is the abliteration adapter for S2
(spec `docs/specs/system12-api.md`, A8); the same tool will extract the Clef LoRA (rank 256) later.

| File | What it does |
|---|---|
| `extract.py` | base HF dir + tuned HF dir → PEFT LoRA dir (`adapter_model.safetensors`, `adapter_config.json`, `extraction_report.json`). numpy + safetensors only. |
| `to_gguf.sh` | PEFT dir → llama.cpp LoRA GGUF with the pinned `convert_lora_to_gguf.py` (llama.cpp `4413a19241`, run in the `full-cuda-b11580` image), then `verify_gguf.py`. |
| `verify_gguf.py` | Checks GGUF metadata, tensor names and that every `B·A` in the GGUF equals the PEFT one (including the V-head reorder on `ssm_out`). |
| `validate.py` | Talks to llama-server: greedy outputs, teacher-forced top-100 distributions, refusal probes, decode speed; `report` turns the dumps into the gate metrics. Stdlib only. |
| `run_validation.sh` | Runs the whole GPU validation on ranger: one server at a time, each under `flock ~/srv/brain-lab/gpu.lock`. |
| `run_quant_gate.sh` | Extends the gate to other quants (merged Q8_0 stays the reference) and reruns the 100-probe refusal set; `validate.py quants` writes the combined report. |
| `out/` | The abliteration manifest facts, extraction report and validation summary. |

## Usage (on ranger)

```bash
tools/sync-ranger.sh tools/extract_adapter          # from the laptop; lands in ~/srv/brain-src

# 1. extract (CPU, ~4 min; memory-maps both checkpoints and reads them in full)
uv run tools/extract_adapter/extract.py \
  --base  ~/srv/models/hf/qwen3.5-9b \
  --tuned ~/srv/models/hf/qwen3.5-9b-abliterated-wangzhang \
  --out   ~/srv/models/adapters/peft/abliteration-wangzhang-r1 \
  --max-rank 1 --ignore-missing '^mtp\.' --analyze-projection \
  --base-model-id Qwen/Qwen3.5-9B --source-id wangzhang/Qwen3.5-9B-abliterated

# 2. convert + verify (byte-reproducible)
tools/extract_adapter/to_gguf.sh ~/srv/models/adapters/peft/abliteration-wangzhang-r1 \
  ~/srv/models/adapters/abliteration-wangzhang-r1.gguf          # f32 by default, see "Speed"

# 3. validate on the GPU (needs the merged model as a GGUF for comparison, see below)
tools/extract_adapter/run_validation.sh                         # -> ~/srv/brain-lab/abliteration/validation/
```

Useful `extract.py` options: `--check-only` (just classify every tensor: identical / changed / artefact
/ error), `--max-rank`, `--energy`, `--noise-mult`, `--full-svd`, `--rename-tuned REGEX=REPL` (if the
tuned checkpoint uses different tensor names), `--ignore-missing/--ignore-extra REGEX`. For the Clef
LoRA, something like `--max-rank 256 --energy 1.0` (rank is then set by the cap and the noise rule).

The tool **fails (exit 2)** on any difference a LoRA cannot express: a changed tensor that is not a
2-D `.weight` (norms, biases, `A_log`, conv kernels), a changed embedding or `lm_head` (excluded by
default), a shape change, or a tensor present on one side only. Pure dtype re-casts (the tuned
value equals the base value rounded to the tuned dtype) are reported as artefacts and skipped.

## How it works: LoRA, abliteration, and why one is the other

**A LoRA is a low-rank correction to a weight matrix.** A linear layer computes `y = W x`, with `W`
of shape `d_out × d_in` (4096 × 4096 or 4096 × 12288 here). A LoRA of rank `r` adds two thin
matrices, `B` (`d_out × r`) and `A` (`r × d_in`), and the layer becomes

    y = W x + s · B (A x),        s = scale · lora_alpha / r

So the effective weight is `W + s·B·A`, but `B·A` is never built: the runtime does two skinny
matmuls, `A x` (squeeze the input to `r` numbers) and `B (…)` (spread them back out). With `r = 1`
that is one dot product and one vector times a number. The `scale` is chosen per request, so the
same loaded base can run with the change fully on (1.0), off (0.0) or anywhere in between.

**Abliteration is a rank-1 edit.** Refusal behaviour in chat models is largely carried by one
direction `r` (a unit vector, 4096 numbers) in the residual stream, the running hidden state that
every layer reads from and writes into. Abliteration removes that direction from what the layers
write: every matrix whose output is added to the residual stream (attention `o_proj`, the
linear-attention `out_proj`, MLP `down_proj`) is replaced by

    W' = W − λ · r rᵀ W        (λ ≈ 1: project out the component along r)

`r rᵀ W x` is "the part of this layer's output that points along r", so the layer can no longer
push the hidden state towards refusing. Rewrite the change:

    ΔW = W' − W = −λ · r (rᵀ W) = B · A   with   B = r  (4096 × 1),   A = −λ · rᵀW  (1 × d_in)

That is **exactly a rank-1 LoRA**. `A` asks "how much would this layer write along r for this
input?", `B` says "write that much, negated, along r". Scale 1.0 reproduces the abliterated model;
scale 0 is the stock model; 0.5 is half-way.

**Recovering it from merged weights: SVD.** We only have the merged checkpoint, so we compute
`D = W_tuned − W_base` for each changed matrix and take its singular value decomposition
`D = Σ σᵢ uᵢ vᵢᵀ`. The best rank-k approximation (Eckart–Young) keeps the k largest terms; we store
`B = U_k·√Σ_k`, `A = √Σ_k·V_kᵀ` (the √ splits the magnitude evenly between the two factors). The
per-matrix rank is the smallest of: `--max-rank`; the rank reaching `--energy` of `‖D‖²`; and the
**noise rule**: stop once the leftover `‖D − B·A‖` is within `--noise-mult` (1.1) of the rounding
noise you get just from storing `W_tuned` in bf16 (`√(Σ ulp²/12)`). Anything below that is not a
learned change; it is the merge being rounded to bf16, and no rank can or should reproduce it.

**What the data says for wangzhang** (`out/abliteration.extraction_report.json`):
- All 775 stock tensors compared in full: 658 byte-identical, 54 changed 2-D weights, 0 errors.
  Artefacts: the 48 F32 tensors `linear_attn.A_log` / `linear_attn.norm.weight` are stored as bf16 by
  wangzhang but equal `round_bf16(stock)` (a storage cast; the adapter keeps stock F32), and the 15
  `mtp.*` tensors (the multi-token-prediction head) were dropped.
- The 54 matrices: `linear_attn.out_proj` in 22 of the 24 linear-attention layers (not 0-1),
  `mlp.down_proj` in layers 8-31, and `self_attn.o_proj` in all 8 full-attention layers.
- Rank 1 captures 99.07-99.71% of each delta's energy. The remainder is 0.93-0.96× the expected
  bf16 rounding noise, i.e. pure rounding; with `--max-rank 8 --energy 1.0` the noise rule alone
  picks rank 1 for all 54. Re-merging `round_bf16(W + B·A)` reproduces 92.8-94.6% of wangzhang's
  stored bf16 values bit-for-bit; the rest differ by one ulp.
- The projection model fits: `D ≈ −λ·u uᵀW` with λ = 0.89-1.38 and the same fit error as the free
  SVD, and in layers 8-31 the attention and MLP matrices share one direction (|cos| = 1.0000).

## llama.cpp mapping (b11580 / 4413a19241)

- PEFT names are `base_model.model.<HF module>.lora_{A,B}.weight`, with HF modules like
  `model.language_model.layers.10.linear_attn.out_proj`. `adapter_config.json` lists the exact
  changed modules as `target_modules`. The converter strips `base_model.model.` and reuses the
  `Qwen3_5ForConditionalGeneration` → `Qwen3_5TextModel` tensor map, so the GGUF has arch `qwen35`
  and tensors `blk.N.{ssm_out,ffn_down,attn_output}.weight.lora_{a,b}` (108 for 54 matrices).
- The linear-attention `out_proj` (`ssm_out`) needs care: Qwen3.5-9B has 16 key heads and 32 value
  heads, and the converter reorders the value heads from grouped to tiled order. For the base weight
  that permutes `out_proj`'s input columns; for a LoRA `_reorder_v_heads` applies the same
  permutation to `lora_a`'s columns. This works at this commit, and `verify_gguf.py` confirms it
  (the error is exactly 0 with the permutation and 100% without it).
- llama.cpp scales every LoRA tensor by `scale · alpha / rank(tensor)` with one global `alpha`. The
  tool therefore zero-pads all matrices to one shared rank `R` and sets `lora_alpha = r = R`, so
  scale 1.0 is exactly the extracted delta in both PEFT and llama.cpp.
- Adapters at scale 0 are dropped from the graph (`llama_context::set_adapters_lora`), so scale 0
  is bit-identical to stock and costs nothing.

## Gotchas found while validating (affect the gateway)

1. **The RAM prompt cache ignores the LoRA set.** With llama-server's default `--cache-ram`, a slot
   saves its state to the RAM cache and reloads the best token match *before*
   `launch_slot_with_task` compares the LoRA set, and cache entries are keyed by tokens only
   (`server-context.cpp` `get_available_slot` → `prompt_save`/`prompt_load`, then line ~1855). So
   after a request at scale 1, the same prompt at scale 0 can resume the scale-1 state. Measured:
   scale 0 matched stock on only 17/45 prompts with the default cache, and 45/45 with
   `--cache-ram 0`. **Workaround: run S2 with `--cache-ram 0`** (in-slot prefix reuse still works
   and is cleared correctly on a LoRA change). The proper fix is to include the LoRA set in the
   cache key upstream.
2. **A request without a `lora` field runs at the global scale, which is 1.0** even with
   `--lora-init-without-apply` (that flag only skips applying at startup). Measured: no-field
   output identical to scale 1 on 45/45. The gateway must always send an explicit scale for every
   loaded adapter (the spec already requires this).
3. **Speed: use an f32 GGUF for small ranks.** Rank-1 f16: 0.951× stock decode; f32: 0.973×;
   zero-padding to rank 2 or 8 gives 0.962×. The overhead is per-op (54 matrices × mul_mat, mul_mat,
   scale, add ≈ 0.3 ms per token), and f16 adds conversions. f32 costs 1.2 MB more.
4. wangzhang's `tokenizer_config.json` (transformers 5, `TokenizersBackend`) breaks
   `convert_hf_to_gguf.py`; to build the merged comparison GGUF, use its weights with stock's
   config/tokenizer files (a symlink staging dir), plus `--no-mtp`.

## Validation (RTX 4090; details in `out/abliteration.validation.json`)

Q8_0 (the full gate):

| Check | Result | Gate |
|---|---|---|
| scale 0 vs no `--lora` | 45/45 greedy prompts + 100/100 probes token-identical | identical: pass |
| top-1 agreement vs merged (teacher-forced, 1213 positions) | 99.18% (stock vs merged: 96.95%) | |
| mean KL(merged ‖ adapter@1), harmless | 0.00052 (stock vs merged: 0.0176; Q8_0 conversion noise floor: 0.00041) | ≤ 0.01: pass |
| refusals on 100 mild probes: stock / adapter / merged | 32 / 0 / 0 | within ±3 per 100: pass |
| decode speed vs stock (90.6 t/s) | scale 1: 0.973, scale 0: 0.999 | ≥ 0.97 / ≥ 0.99: pass |

Other unsloth quants, all against the merged **Q8_0** as a fixed reference (quantizing our own merged
model per quant would confound the comparison with unsloth's imatrix). The rule is
KL(merged Q8_0 ‖ quant + adapter) ≤ KL(merged Q8_0 ‖ Q8_0 + adapter) [0.00052] + floor + 0.01,
where floor = KL(stock Q8_0 ‖ stock quant):

| Quant | KL adapter | floor | bound | top-1 | refusals /100 (stock) | scale 0 | speed s1 / s0 | gate |
|---|---|---|---|---|---|---|---|---|
| Q6_K | 0.00239 | 0.00247 | 0.0130 | 98.7% | 0 (32) | 120/120 | 0.967 / 0.999 | speed s1 fails |
| Q5_K_M | 0.00500 | 0.00468 | 0.0152 | 97.4% | 0 (33) | 120/120 | 0.965 / 1.000 | speed s1 fails |
| Q4_K_M | 0.01605 | 0.01589 | 0.0264 | 95.3% | 0 (31) | 120/120 | 0.961 / 1.000 | speed s1 fails |
| IQ4_XS | 0.02517 | 0.02437 | 0.0349 | 95.6% | 0 (31) | 120/120 | 0.960 / 1.001 | speed s1 fails |

The adapter's KL against the merged model is the quant's own floor plus about 0.0005: the adapter
behaves the same on every quant. The adapter adds a constant ~0.3 ms per token, so the ratio drops
as the base gets faster (Q4_K_M: 133 → 128 t/s). Every quant passes everything except the 0.97
speed gate at scale 1.

Not done yet: the stacking smoke test (needs a second adapter).
