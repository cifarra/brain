# Feature: "brain", a modular local System 1 / System 2 service

Status: DRAFT rev 4 for user review (design only, no implementation).
Date: 2026-10-13 (rev 1: 2026-10-07; rev 2: 2026-10-09)
Owner: Ale

Deployments:
- **Primary:** homelab host `ranger` (RTX 4090 24 GB, shared GPU).
- **Edge:** MacBook, used when the lab is unreachable. Later phase.
- **Sharing:** other people and enterprise customers. Later phase.

First capability test: CucoStudio shot assignment (Experiment 1).

### Revision history
- **Rev 1 (2026-10-07):** one Qwen3.5-9B base with the Clef LoRA toggled on and off, running on MLX on a Mac.
- **Rev 2 (2026-10-09):** 16 GB Mac as the minimum target, an in-process Python library first, structured output, a lab harness, and Experiment 1.
- **Rev 3 (2026-10-13):** supersedes rev 1-2 wherever they conflict.
  - ranger is the primary deployment and the main lab API.
  - Runtime is **llama.cpp**, the only runtime with working Clef on CUDA and Metal with quantized weights. The MLX engine, step loop and adapter switch are superseded.
  - A gateway with child-process backends and a declarative registry.
  - S2 = stock Qwen3.5-9B + a rank-1 abliteration adapter.
  - A measured GPU budget, a GPU lease, Experiment 0 (finding the 4090 sweet spot) first, and Experiment 1 reframed as a capability test.
- **Rev 4 (this document):**
  - **Ocular is out of brain scope.** The brain is "just LLM and decision". Ocular calling the brain is an optional change on Ocular's side. The Phase A budget is designed for **both** cases: (a) Ollama stays resident, brain ≈ 9-10 GB at peak; (b) Ollama retired, 12 GB target. The recommended strategy for case (a) is given below.
  - **Adapters are first-class and expected to grow:**
    - an adapter registry, with base identity, compatibility, `requires`, gates and provenance;
    - per-request stacking via `x_adapters: {name: scale}`;
    - stacks are evaluated as units: unapproved stacks are rejected, except for tokens with the experimental scope;
    - a task → stack router: static in Phase A, Clef-driven later;
    - a mapping onto llama.cpp, including stack-affine slots and the fact that adding an adapter restarts S2;
    - a light fine-tune → adapter workflow.
  - **Phase A is concrete enough to implement:** files, config formats, compose, endpoints and tests.
  - Review fixes are applied: lease atomicity, NVML PIDs inside containers, `batch ≥ ubatch`, Clef prompt rendering for token counting, health semantics, and a firewall instead of binding to the tailnet IP.

---

## Overview

"Brain" is one service that exposes two thinking modes behind a stable, OpenAI-style API:

- **System 1 (`system1`)** is Cloudflare **Clef-Flash**. Given a state and typed questions, it returns calibrated probabilities for every allowed answer in a single prefill. It generates nothing.
- **System 2 (`system2`)** is **Qwen3.5-9B** with thinking, JSON-schema output and vision. Each request can choose a stack of **generation adapters**. Abliteration is the first adapter, and it is on by default.

The design is modular without being fragmented: **one service, one container, one port**.
- A Python **gateway** owns the public API, auth, routing (including the adapter router), the GPU lease and the backend lifecycle.
- **Backends** are llama.cpp server processes that run as the gateway's children in the same container.
- What gets loaded is decided by declarative **profiles**, **model manifests** and **adapter manifests**.
- Clients address **roles** (`system1`, `system2`) and **adapters by name**, never model files. Swapping a model or adding an adapter means editing the registry and reloading.

Rollout is in two phases:
- **Phase A** uses stock llama.cpp in two processes: Clef GGUF, and Qwen + generation adapters.
- **Phase B** keeps one resident Qwen3.5-9B. Decision adapters (Clef LoRA + head) and generation adapters are applied per call in one patched llama.cpp server. Phase B is gated on calibration.

---

## Verified facts this design depends on

### Models

**Clef-Flash**
- Frozen Qwen3.5-9B + a rank-256 LoRA + a 0.24 GB joint decision head, calibrated with Brier loss and RLCD.
- `Cloudflare/clef-flash` ships **merged** weights plus the head.
- Source: Cloudflare blog; previous-session SVD.

**Clef schema**
- `choice` returns the choice, its confidence and per-option probabilities.
- `score` returns an expected 0-based score, its confidence, a legend and per-level probabilities.
- `noul` returns P(true).
- All questions are answered jointly.
- Source: model card; `joint_schema_model.py`.

**Qwen3.5-9B**
- 32 layers: 8 full-attention (4 KV heads × 256) and 24 Gated-DeltaNet.
- KV ≈ 32 KB/token in fp16. The recurrent state is fixed and small.
- Thinking is on by default.
- Vision is supported.
- Source: Qwen model card.

**Abliteration: `wangzhang/Qwen3.5-9B-abliterated` (Abliterix)**
- 2/200 refusals, KL 0.0105, Apache-2.0.
- The delta from stock is **exactly rank-1 per matrix** over 54 matrices: `linear_attn.out_proj`, `mlp.down_proj` (layers 8-31) and all `self_attn.o_proj`.
- W' = W0 − λ·r·rᵀ·W0, with λ ≈ 1.05-1.36. All other tensors are identical.
- Source: range-read safetensors + SVD (scratchpad `svd_delta.py`, `abl/`).

**Rule: S1 never runs with any generation adapter.** Abliteration removes a residual direction, and Clef's head was calibrated on stock hidden states.

### llama.cpp (master af50f2f, 2026-10-13)

**Clef support**
- Clef is GGUF arch `clef`, with the head inside the GGUF.
- `ggml-org/Clef-Flash-GGUF` ships BF16 (18.2 GB), Q8_0 (9.66 GB) and Q4_K_M (6.49 GB), plus mmproj files (BF16 0.92 GB, Q8_0 0.62 GB).
- `POST /v1/systemone` takes `state`, optional `images` (data URLs; needs `--mmproj`) and `questions`.
  - Limits: `choice` up to 255 options; `score` 2-10 levels.
  - No streaming.
  - No prefix reuse.
  - **The whole prompt must fit in `--ubatch-size`** (#30073).
  - No batching across decisions.
- **ubatch is capped at `--batch-size`** (default 2048).
- Builds must be newer than #30100 (fixes Metal bug #30064).
- A clef-arch server cannot generate text.
- A LoRA's arch must equal the model's arch.
- systemone ignores per-request `lora`.

**Chat and adapters**
- Adapters are preloaded with `--lora <file>` (repeatable) plus `--lora-init-without-apply`. Each request then sets `lora: [{id, scale}]`.
- **Requests with different LoRA configs don't batch together.**
- **A LoRA change clears that slot's prompt cache.**
- `GET/POST /lora-adapters` lists adapters and sets global scales. It cannot load new files.
- `id_slot` gives slot affinity (to verify on the pinned build).

**Other**
- Hybrid prefix reuse via checkpoints.
- JSON schema works together with thinking.
- Per-request `reasoning_budget_tokens`.
- `chat_template_kwargs.enable_thinking`.
- Avoid MTP with json_schema (#30419).
- Watch the `<think>` whitespace issue (#30357).
- Qwen3.5 vision via mmproj is inferred, and verified in Experiment 0.

### ranger (measured 2026-10-13)

| Load | VRAM |
|---|---|
| Frigate detector + 2 NVDEC (always on) | ≈ 1.2 GB |
| TTS (always on) | ≈ 2.0 GB |
| Jellyfin NVENC | ≈ 0.4 GB per transcode |
| Typical total (median) | ≈ 3.6 GB |
| Ocular → Ollama `qwen3.5:4b` | 0.6 → 4.9 GB per request; Ollama holds 5.2 GB at 32k ctx with a 5-min keep-alive; ~2 requests per 7 days; bursts reach 10-16 GB total |
| `tts-trainer.service` (~3-4 h per voice, a couple per week) | total ≈ 22 GB |

Other facts about ranger:
- PCIe gen 1, ~3.3 GB/s, so loading 7-9 GB takes ~2-3 s.
- Compose stacks live in `~/srv/docker/apps/<service>/`, with `TZ` set.
- Watchtower updates containers on Sundays at 04:00.
- TTS uses a `data/state/train.lock` file while training.
- Containers run as uid 1000.

### CucoStudio
- Styles have 2-15 shots.
- Each shot is an After Effects comp with a text precomp and a frame precomp.
- Each section is fitted into one shot.
- Notes of use are not implemented yet.
- There are **no fixed sequencing rules**, only optional "general notes".
- English only.
- The model never runs during After Effects renders.

---

## Scope

### In scope (v1)
1. **Experiment 0:** find the 4090 sweet spot and produce measured profiles for both Ollama cases.
2. **Phase A brain on ranger:**
   - the gateway: API, auth, roles, model/adapter/profile registry, adapter stacking and the static router, a supervisor with on-demand residency, a VRAM guard, the lease and health;
   - two llama.cpp backends;
   - the abliteration adapter;
   - the compose deployment;
   - Caddy, Kuma and Beszel;
   - the GPU lease with `tts-trainer`;
   - a Python SDK with failover.
3. **Experiment 1 pilot.**
4. **Phase B (gated).**

### Later (outlined only)
- Edge Mac.
- Sharing and enterprise.
- S1-driven or learned routing.
- Real fine-tunes beyond the described workflow.

### Not the brain's concern
- Captioning and detection. **Ocular remains its own service.**
- Ollama's future on ranger. That is the user's decision, and the brain's budget works either way.

### Deferred
- Runtime profile swap through the API (v1: change `.env` and restart).
- Hot-loading adapters without a backend restart.
- `reasoning_budget_message`.
- Tools.
- Other runtimes.
- Clef 27B.
- Runtime cloud calls.

---

## Architecture

```
  clients (CucoStudio, n8n on hx, laptop scripts, optionally Ocular, later: other people)
     │  HTTPS brain.hx (Caddy on hx, tailnet)        │  http://brain:8790 (docker network brain_net)
     ▼                                               ▼
┌──────────────────────────── container "brain" on ranger (one service) ────────────────────────────┐
│ Gateway (Python, FastAPI) :8790                                                                    │
│   auth/policy → role + adapter-stack resolution (role default ⊕ router ⊕ request) → validation     │
│   → per-role queues (+ stack-affine slots) → proxy to backend (SSE passthrough, cancel)            │
│   registry: models/ adapters/ profiles/ router.yaml   supervisor (resident | on_demand residency)  │
│   lease watcher (/run/gpu-lease)   VRAM guard   /health* /metrics /admin/reload                    │
│        │ 127.0.0.1 only                                │                                         │
│  Phase A:  llama-server "s1" (clef arch)          llama-server "s2" (qwen35 arch)                  │
│            Clef-Flash GGUF → /v1/systemone        Qwen3.5-9B GGUF + mmproj + --lora × N            │
│  Phase B:  one patched llama-server (qwen35): decision adapters (LoRA+head) + generation adapters   │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
   weights + adapters: ~/srv/models/brain (ro, not in restic)    config/registry/state: ~/srv/docker/apps/brain
```

**Why one container with a supervising gateway:**
- The lease, on-demand residency and crash recovery all need to start and stop backends.
- Doing that across containers would need the Docker socket, which is root-equivalent, or duplicated watcher logic in every container.
- The cost is that a gateway restart restarts the models (~2-3 s each).

**Why a gateway at all:**
- A stable API across model and adapter changes.
- Auth, policy and adapter rules.
- Validation against llama.cpp limits before any GPU time is spent.
- The lease, residency and the VRAM guard.
- Hiding the Phase A → Phase B change from clients.

---

## Adapters (first-class)

### Concepts
- **Base:** a specific checkpoint (Qwen3.5-9B at a pinned HF revision with a recorded weights hash), materialized as one or more GGUF quants. Each quant has its own model manifest.
- **Generation adapter:** a LoRA applied to S2 generation, such as abliteration or a future task fine-tune. Several can be **stacked**: llama.cpp sums their deltas, each with its own scale.
- **Decision adapter:** a LoRA plus a decision head that serves `/v1/systemone` (Clef). It is never stacked. In Phase A it is bound to its own backend, because the Clef GGUF is base + LoRA merged + head. In Phase B it is applied per call on the shared base.
- **Stack:** the set `{adapter: scale}` applied to one request. LoRA deltas add, so order is irrelevant.

### Adapter manifest (`registry/adapters/<name>.yaml`)
```yaml
name: abliteration                 # public name, [a-z0-9-]+
version: 1
kind: generation                   # generation | decision
description: "Removes the refusal direction (rank-1, Abliterix/wangzhang)."
router_description: "Uncensored general assistant behaviour."   # used by S1-driven routing later
base:
  model: Qwen/Qwen3.5-9B           # base model id
  hf_revision: <commit sha>        # must equal the backend base manifest's source revision
  weights_sha256: <sha256 of the bf16 safetensors index>   # identity of the exact checkpoint trained/extracted against
  quant_compat: [BF16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, IQ4_XS] # quants on which this adapter's eval gate passed
artifact:
  format: gguf-lora                # gguf-lora | merged-gguf (Phase A decision binding) | gguf-lora+head (Phase B decision)
  arch: qwen35
  file: adapters/abliteration-wangzhang-r1.gguf
  sha256: <...>
  size_mb: 1.3
target_modules: [linear_attn.out_proj, mlp.down_proj, self_attn.o_proj]
rank: 1
scale: {default: 1.0, min: 0.0, max: 1.5}
roles: [system2, system2-fast]
compat:
  incompatible_with: [clef, "kind:decision"]   # adapter names or kinds that may never be in the same stack/role
  requires: []                                 # adapters that must be present (e.g. a task LoRA trained on top of abliteration)
  exclusive_group: null                        # at most one adapter per group (e.g. "persona")
provenance:
  source: wangzhang/Qwen3.5-9B-abliterated@<rev>
  method: "rank-1 SVD extraction of (abliterated − stock), tools/extract_adapter vX"
  license: Apache-2.0
  created_at: 2026-10-xx
evals:
  gate: passed                     # passed | failed | pending
  report: eval/reports/adapters/abliteration-v1.md
  metrics: {refusals_per_100: null, kl_harmless_vs_merged: null, decode_speed_ratio: null}
```

**Clef in Phase A** (`registry/adapters/clef.yaml`):
- `kind: decision`, `roles: [system1]`, `compat.incompatible_with: ["kind:generation"]`.
- `artifact.format: merged-gguf`, pointing to the Clef-Flash GGUF model manifest.
- It is listed for discovery and policy, and is not stackable.

### Per-request selection and stacking

**Canonical field:** `x_adapters: {"abliteration": 1.0, "cuco-shots-v1": 0.8}`.
- OpenAI SDK clients send it through `extra_body`.
- `system1` accepts only decision adapters. In Phase A that is `clef`, which is implicit.

**Sugar for clients that can only set `model`:** a qualified model name, `<role>(+<adapter>[@<scale>])*`.
- `system2+abliteration@0.5`
- `system2+none` gives stock Qwen.
- `system2+abliteration+cuco-shots-v1`

Sending both forms is a 400 (`ambiguous_adapters`).

**Resolution:**
1. Start from the role default stack in the profile.
2. Merge the router stack, if `x_task` is set.
3. Merge the request stack.

**Merge rules:**
- A later layer overrides the scale of the same adapter and adds new adapters.
- Scale 0 removes an adapter.
- `+none`, or `"x_adapters_mode": "replace"`, starts from an empty stack.

**Validation before any GPU time** returns `400 invalid_adapter` with a reason code:

| Reason | Meaning |
|---|---|
| `unknown` | adapter not registered |
| `not_loaded` | not in the active profile's preload list; Phase A cannot hot-load |
| `role` | role not in the adapter's `roles` |
| `base_mismatch` | adapter `base.hf_revision` / `weights_sha256` ≠ the backend base, or the base quant ∉ `quant_compat` |
| `scale_range` | scale outside the allowed range |
| `incompatible` | an `incompatible_with` rule is violated |
| `missing_requirement` | a `requires` rule is not met |
| `exclusive_group` | two adapters from the same group |
| `stack_too_large` | more than the profile's `max_stack` (default 3) |
| `gate` | adapter `evals.gate != passed` |

**Token policy** returns `403 adapter_forbidden` when an adapter is not on the token's allow-list or its scale exceeds the token's `max_scale`.

**Stacks are evaluated as units.**
- A stack with one non-zero adapter is approved if that adapter's gate passed.
- A stack with two or more non-zero adapters is approved only if it appears in the profile's `approved_stacks`, which links each stack to an eval report.
- **Recommendation: reject unapproved stacks** with `403 stack_not_approved` for normal tokens. Tokens with the `experimental` scope may run them, and they are flagged with `x_brain.stack_approved: false`. This keeps untested combinations away from shared and production clients while still allowing exploration.
- Scale is part of the unit only coarsely: an approved stack is approved for scales within each adapter's manifest range. Stacks with very different scales should be added as separate entries if they behave differently.

`x_brain.adapters` returns the resolved stack.

### Router (task → stack)

**Phase A: static map** (`registry/router.yaml`):
```yaml
tasks:
  default:   {role: system2,      x_adapters: {abliteration: 1.0}}
  caption:   {role: system2-fast, x_adapters: {}}                     # stock behaviour
  translate: {role: system2-fast, x_adapters: {abliteration: 1.0}, defaults: {temperature: 0.3}}
  # future: shot-notes: {role: system2, x_adapters: {cuco-shots-v1: 1.0}}
```
- A request carries `x_task: "translate"`.
- The router supplies the role, the stack and sampling defaults. Explicit request fields still win.
- An unknown task is a 400.
- `GET /v1/router` lists the tasks.

**Later: S1-driven routing** (`x_task: "auto"`, designed and deferred). Clef picks the adapter:
1. The gateway builds a `systemone` request. The state is the user message, truncated to 1k tokens. It contains one `choice` question whose criteria are the candidate tasks' `router_description`s.
2. The argmax task is used if its probability ≥ τ. τ is calibrated per router on a small labelled set.
3. Otherwise the request goes to `default`.
4. `x_brain.route` records the choice and its probability.

This costs one extra S1 call, a few hundred ms. A learned router (a classifier, or an S1 decision adapter trained for routing) can replace it later behind the same contract.

### Mapping onto llama.cpp (Phase A)

**Preload**
- Every generation adapter listed in the profile is passed to S2 as `--lora <file>`, in profile order, together with `--lora-init-without-apply`.
- At startup the gateway reads `GET /lora-adapters` and verifies the name → `id` map against the file paths.

**Per request**
- The gateway sends `lora: [{id, scale}]` for **every** loaded adapter, with explicit 0 for inactive ones. This never relies on global scales.

**Batching and cache**
- Different stacks don't batch together.
- Changing a slot's stack clears that slot's prompt cache.

**Stack-affine slots**
- With `-np 2` or more, the gateway pins slot 0 to the role's default stack via `id_slot`.
- Other stacks are sent to the remaining slots, and identical stacks reuse the same slot where possible.
- Default profiles use `parallel: 1`, where a stack change just costs a re-prefill.
- Experiment 0 measures whether a second slot (~0.5 GB KV at 16k) is worth it.

**Memory**
- Loaded adapters stay resident: rank-1 is ~1 MB; a rank-16-64 task LoRA on 9B is ~20-150 MB in f16.
- `measured_peak_gb` includes them.
- If an adapter is added after measurement, the validator adds its manifest `size_mb` and warns until the profile is re-measured.

**Compute**
- Active adapters add small matmuls.
- Experiment 0 checks that zero-scale adapters cost nothing.

**Adding an adapter requires an S2 restart in Phase A**, because the preload list is fixed at launch. The steps:
1. Add the manifest.
2. Add the adapter to the profile's `loras`.
3. `POST /admin/reload`, which restarts only S2 (~3 s).

### Mapping onto Phase B
- Generation adapters work as in Phase A: per-request `lora`, with stack-affine slots.
- Decision adapters (Clef, and future decision fine-tunes) are applied per `/v1/systemone` call on a **reserved S1 slot**. The slot's adapter set is `{decision adapter: 1, all generation adapters: 0}`, with embeddings on and chunked prefill. The head runs on the GPU, and state is restored afterwards.
- llama.cpp applies the adapter set per batch within one context, so the S1 slot never shares a batch with S2 slots.
- Several decision adapters can share the base. `system1` selects one with `x_adapters: {"clef": 1}`, which is the default.

### Fine-tune → adapter → registry (light workflow, for later and for learning)
1. **Define the task first.** Create `lab/tasks/<task>/` with a dev/test set, a metric and a target ("better than base by X"), plus regression checks. Nothing is registered without an eval set.
2. **Train against the pinned base.**
   - Use the bf16 HF weights at `base.hf_revision`.
   - Train on ranger's 4090 **under the GPU lease** (`holder: brain-train`). QLoRA fits in 24 GB.
   - Tooling: PEFT + TRL, or Unsloth (both Apache-2.0).
   - Start with rank 8-32 on the attention and MLP projections, and record the target modules.
   - If the adapter is trained with abliteration applied, declare `requires: [abliteration]`.
3. **Convert.** `convert_lora_to_gguf.py --base <pinned HF dir>` produces a `qwen35` LoRA GGUF (f16).
4. **Register.** `brain adapters add <gguf> --name … --kind generation --roles system2`:
   - reads the rank, target modules and base revision from the GGUF and PEFT metadata;
   - computes the sha256 values;
   - writes the manifest with `gate: pending`;
   - copies the file to `~/srv/models/brain/adapters/`.
5. **Run the eval gate.** `brain adapters eval <name> --profile <p>` runs the task eval with and without the adapter on the profile's quant(s), plus a regression suite: KL on harmless text vs the base, JSON validity, a refusal probe if relevant, and decode speed. It writes the report and sets `gate` and `quant_compat`.
6. **Enable.**
   - Add the adapter to the profile's `loras`.
   - Optionally add a router task.
   - Add an `approved_stacks` entry, with its own eval run, if it will be stacked.
   - `POST /admin/reload`.

---

## ranger deployment

### VRAM budget: two cases (GiB)
| Item | (b) Ollama retired | (a) Ollama stays |
|---|---|---|
| 4090 usable | 23.5 | 23.5 |
| Frigate + TTS (always on) | 3.2 | 3.2 |
| Jellyfin, 2 transcodes | 0.8 | 0.8 |
| Ocular's own burst | 0.6 … 4.9 (verify per PID) | 0.6 … 4.9 |
| Ollama (resident 5 min after each Ocular call) | 0 | 5.2 |
| Safety headroom | 1.5 | 1.5 (or 0) |
| **Left for the brain at peak** | **13.1 … 17.4** | **7.9 … 12.2** (≈ 9-10 at the worst Ocular burst with zero headroom) |
| Steady state (no Ocular burst) | ≈ 18 | ≈ 18 |

**Phase A two-process estimates** (inferred; Experiment 0 measures):

| Configuration | Est. peak |
|---|---|
| Q4_K_M × 2 | ≈ 15 GB |
| IQ4_XS × 2, S1 ubatch 2048-4096, S2 ctx 16k | ≈ 12.7-13.4 GB |
| S2 alone (Q4_K_M or IQ4_XS + mmproj, ctx 16k) | ≈ 7.4-8.4 GB |
| S1 alone (IQ4_XS or Q4_K_M, stripped) | ≈ 5.3-7.4 GB |

### Phase A strategy per case

**Case (a): Ollama stays. Recommended default profile `ranger-od-a` (S2 resident, S1 on demand).**

*Steady state:*
- S2 alone, about 7.5-8.4 GB, which fits the ~9-10 GB worst case.
- Reducers if a measurement exceeds the budget: IQ4_XS, S2 ctx 8k, KV q8_0.

*S1 lifecycle:*
1. **Load gate.** S1 loads only if NVML free memory ≥ S1 measured peak + 0.5 GB. The cold load takes about 2-3 s at Q4/IQ4 on PCIe gen-1 (measured in Experiment 0).
2. **Idle unload.** S1 unloads after `idle_ttl_s` (600 s by default).
3. **Insufficient memory** (e.g. during an Ocular burst while Ollama is resident). S1 requests wait, polling every 2 s up to their `timeout_ms`, then return 503 `waiting_for_vram`. **The brain never causes another service's OOM by loading.**

*Accepted contention:*
- The residual risk is Ollama loading *while* S1 is resident.
- It is rare (about 2 Ocular calls a week), and Ollama degrades gracefully by offloading layers to the CPU.
- The cost is slower captions during the overlap, and the S1-active peak is about 13 GB.

*Rejected alternatives for case (a):*
- **Smaller quants or contexts alone** cannot bring two 9B processes under about 9-10 GB at acceptable quality.
- **Permanently giving up S1 residency in both directions** (S1 and S2 swap) would make S1→S2 escalation slow.

**Case (b): Ollama retired. Upgrade path `ranger-12g-a` (both backends resident).**
- Peak ≤ 12 GB as measured, with no cold starts.
- A ≤ 14 GB interim is allowed only if the measured Ocular own-burst is ≤ 1 GB.
- Switching to it is a profile change.

**Long term: Phase B**, about 8.5-9.5 GB in one process, fully resident in both cases.

`tts-trainer` (about 22 GB) can't coexist with the brain in either case, so it uses the **GPU lease**.

### VRAM guard
llama.cpp has no hard cap, and NVML inside a container reports host PIDs (we don't use `pid: host`). The guard works like this:

1. **Validator:**
   - `budget.steady_gb` ≥ the sum of measured peaks of the resident backends;
   - `budget.peak_gb` ≥ the sum across all backends;
   - peaks are measured per PID on the host by Experiment 0.
2. **Before every backend load** (startup, on demand, after a lease), require NVML free memory ≥ that backend's measured peak + 0.5 GB. Otherwise:
   - at startup, the state becomes `waiting_for_vram`;
   - on demand, the request waits.
3. **Footprint:**
   - computed from llama-server load-log buffer sizes (model, KV, compute) plus a CUDA-context estimate;
   - cross-checked against the device-used delta across the load;
   - reported as `brain_vram_gb`, with an `over_budget` flag.
4. **Static after warm-up.** Each load warms up with a max-size request; S2 includes an image. Allocations by others that fail are theirs. There is no auto-yield in v1.
5. **Host audit.** `nvidia-smi --query-compute-apps` lists the `llama-server` PIDs, and the verification checks use it.

### GPU lease (files-as-state)

**Location**
- `/run/gpu-lease/`, on tmpfs, recreated at boot by `tmpfiles.d/gpu-lease.conf` (owner `ale`, mode 0775).
- Bind-mounted read-write into the container.

**Files**
| File | Content |
|---|---|
| `request` | The lease itself: `{holder, need_gb, requested_at, max_hold_s}`. Only one can exist |
| `granted` | `{holder, granted_at}` |
| `brain.heartbeat` | Touched every 5 s |
| `lock` | Brief flock for creating and deleting the files above |

**Host tool:** `gpu-lease`, stdlib only, in `/usr/local/bin`.

`acquire --holder H --need-gb N --timeout S`:
1. Waits until there is no `request`, then creates one with `O_EXCL` under the flock.
2. Waits for `granted`. If the heartbeat is more than 30 s old (brain absent), it writes `granted` itself.
3. Waits up to 60 s for NVML free memory ≥ N. If that doesn't happen, it proceeds with a warning.
4. Exits 0.

On timeout or SIGTERM before step 4, it deletes its own `request` and exits non-zero.

Other subcommands:
- `release --holder H`: holder-checked and idempotent.
- `status`.

**tts-trainer drop-in**
- `ExecStartPre=/usr/local/bin/gpu-lease acquire --holder tts-trainer --need-gb <measured> --timeout 900`
- `ExecStopPost=/usr/local/bin/gpu-lease release --holder tts-trainer`

**Other holders:** `brain-exp0` and `brain-train`.

**State machine**
```
SERVING ──request──► DRAINING ──drained──► stop backends (SIGTERM, waitpid, NVML drop) ──► LEASED (write granted)
   ▲                    └─ request withdrawn ──► SERVING                                  │ request gone
   └──────────────────────────────────── LOADING (VRAM check, warm-up) ◄──────────────────┘
startup with request present ──► LEASED directly (no backend starts)
```
- **Draining:** returns 503 `brain_unavailable` with `reason: gpu_lease`, `holder` and `Retry-After`. Drain timeout is 60 s, after which streams are cancelled.
- **Overdue:** a lease past `max_hold_s` sets the `lease_overdue` flag and alerts. The brain **never reclaims** the GPU.

### Health semantics
| State / flag | `/health` HTTP | `/health/kuma` | `/health/deep` |
|---|---|---|---|
| `serving` | 200 | `BRAIN_OK` | `DEEP_OK` if the last synthetic S2 call (and S1 call, if resident) was < 15 min ago |
| `serving`, S1 `idle_unloaded` | 200 | `BRAIN_OK` | `DEEP_OK` (S2 only) |
| `draining` | 200 | `BRAIN_OK` | `DEEP_OK` (skipped) |
| `leased` within `max_hold_s` | 200 | `BRAIN_OK` | `DEEP_OK` (skipped) |
| `leased` + `lease_overdue` | 503 | `BRAIN_DOWN:lease_overdue` | `DEEP_DOWN` |
| `loading`, ≤ 180 s | 503 | `BRAIN_OK` | `DEEP_OK` (skipped) |
| `loading`, > 180 s | 503 | `BRAIN_DOWN:loading_stuck` | `DEEP_DOWN` |
| `waiting_for_vram`, ≤ 10 min | 503 | `BRAIN_OK` | `DEEP_OK` (skipped) |
| `waiting_for_vram`, > 10 min | 503 | `BRAIN_DOWN:waiting_for_vram` | `DEEP_DOWN` |
| `degraded` | 503 | `BRAIN_DOWN:degraded` | `DEEP_DOWN` |
| `over_budget` | unchanged | `BRAIN_DOWN:over_budget` | unchanged |

The deep check never loads an on-demand backend.

### Optional: Ocular as a brain client (Ocular-side change, user's decision)
- Ocular could call `system2` like any client: OpenAI chat with an `image_url` data URL, `x_task: "caption"` and its own token.
- That would allow retiring Ollama on ranger, which unlocks `ranger-12g-a`, and would move captioning from the 4B model to the 9B.
- The A/B comparison, Ocular's code change and the check for other Ollama clients (n8n) are Ocular-side work and out of brain scope.
- The brain only provides the role, a token and the `caption` router task.

---

## Phase A: concrete design (implementable)

### Repository layout (`/Users/ale/gits/clef`, needs `git init`)
```
gateway/brain/
  app.py            FastAPI app, routes, error envelope, middleware (auth, request id, logging)
  config.py         loads config/brain.yaml + env; pydantic models for all config files
  registry.py       loads + validates models/, adapters/, profiles/, router.yaml (pure)
  stacks.py         stack resolution (role default ⊕ router ⊕ request), x_adapters + qualified-name parsing, validation, approval
  policy.py         tokens (sha256), scopes, adapter policy
  render.py         Clef prompt renderer (port of llama.cpp server-decision.cpp template) for token-fit checks
  supervisor.py     llama-server argv from profile; start/stop/restart/backoff; residency (resident | on_demand)
  vram.py           NVML device free/used, load-log buffer parsing, footprint
  lease.py          lease watcher + state machine; heartbeat
  backends.py       HTTP clients to llama-server (systemone, chat SSE passthrough, tokenize, lora-adapters, health)
  queues.py         per-role queues, timeouts, stack-affine slots, cancellation
  health.py         /health, /health/kuma, /health/deep, /metrics
  cli.py            `brain` CLI: serve | models pull/build/verify | adapters add/list/verify/eval | tokens add/revoke | profile check
sdk/brain_client/   Client with failover; tasks/shot_assignment.py
tools/
  gpu-lease         host script (stdlib only)
  extract_adapter/  SVD extraction → PEFT → GGUF LoRA (+ manifest stub)
  gguf/             strip output.weight; quantize clef keeping head; (Phase B) extract head
registry/
  models/*.yaml  adapters/*.yaml  profiles/*.yaml  router.yaml
deploy/ranger/
  Dockerfile  compose.yaml  .env.example  config/brain.yaml.example  config/tokens.yaml.example
  host/brain-firewall.service  host/tmpfiles.d/gpu-lease.conf  host/tts-trainer.service.d/gpu-lease.conf
  docs: caddy.md  kuma.md  ops.md
lab/ exp0/  exp1/  tasks/
tests/ unit/  integration/ (fake llama-server, fake NVML, fake lease dir)  gpu/ (ranger only)
CLAUDE.md  README.md
```

### Config formats

**Model manifest** (`registry/models/<id>.yaml`)
```yaml
id: qwen3.5-9b-q4km
kind: base                 # base | mmproj | decision-merged
arch: qwen35               # qwen35 | clef
family: qwen3.5-9b
source: {model: Qwen/Qwen3.5-9B, hf_revision: <sha>, weights_sha256: <sha>, gguf_from: <repo or "built">}
quant: Q4_K_M
file: base/qwen3.5-9b-Q4_K_M.gguf          # relative to /models
sha256: <...>
size_gb: 5.9
derived_from: null                           # or {model: <id>, recipe: "strip output.weight" | "llama-quantize IQ4_XS keep-head"}
license: Apache-2.0
```

**Profile** (`registry/profiles/<id>.yaml`)
```yaml
id: ranger-od-a
phase: A
budget: {steady_gb: 9.0, peak_gb: 13.5}       # GiB
max_stack: 3
backends:
  s1:
    model: clef-flash-iq4xs-stripped
    args: {ctx: 4096, batch: 4096, ubatch: 4096, parallel: 1, flash_attn: on, ngl: all}
    residency: {mode: on_demand, idle_ttl_s: 600}
    measured_peak_gb: null                     # Experiment 0
  s2:
    model: qwen3.5-9b-q4km
    mmproj: qwen3.5-9b-mmproj-q8
    loras: [abliteration]                      # adapter names; load order = llama.cpp id order
    args: {ctx: 16384, batch: 2048, ubatch: 512, parallel: 1, flash_attn: on, ngl: all,
           ctx_checkpoints: 4, cache_type_k: f16, cache_type_v: f16, reasoning_format: deepseek, jinja: true}
    slot_policy: {stack_affinity: false}       # true only when parallel >= 2 (slot 0 = role default stack)
    residency: {mode: resident}
    measured_peak_gb: null
roles:
  system1: {backend: s1, endpoint: systemone, default_adapters: {clef: 1.0},
            capabilities: {systemone: true, images: false, max_prompt_tokens: 4096}}
  system2: {backend: s2, endpoint: chat,
            defaults: {thinking: true, reasoning_budget: -1},
            default_adapters: {abliteration: 1.0},
            capabilities: {chat: true, thinking: true, json_schema: true, vision: true, max_context: 16384}}
  system2-fast: {inherits: system2, defaults: {thinking: false}}
approved_stacks: []                            # e.g. [{adapters: [abliteration, cuco-shots-v1], report: eval/reports/stacks/...}]
```

**Validator rules:**
- Files exist and their sha256 values match.
- `batch ≥ ubatch`.
- S1 `max_prompt_tokens ≤ ubatch`.
- Adapters match their backend base: `hf_revision`, `weights_sha256` and `quant_compat`.
- Decision adapters appear only on `systemone` roles.
- `requires` and `incompatible_with` hold for the default stacks and `approved_stacks`.
- Budgets hold against measured peaks. `--allow-unmeasured` overrides this, for Experiment 0 only.
- Phase B profiles share one base sha256.

**Gateway settings** (`config/brain.yaml`)
```yaml
listen: {host: 0.0.0.0, port: 8790}
profile: ${BRAIN_PROFILE}
models_dir: /models
lease: {dir: /run/gpu-lease, heartbeat_s: 5, drain_timeout_s: 60}
vram: {load_margin_gb: 0.5, waiting_alert_after_s: 600}
queues: {system1: {max: 32, timeout_ms: 30000}, system2: {max: 8}}
supervisor: {restart_backoff_s: [1, 5, 30], crash_loop: {max: 5, window_s: 600}, loading_alert_after_s: 180}
deep_check: {interval_s: 600}
limits: {max_body_mb: 20}
logging: {level: info, bodies: false}
```

**Tokens** (`config/tokens.yaml`, managed with `brain tokens add`)
```yaml
- name: ale-laptop
  sha256: <hash of token>
  scopes: [system1, system2, admin, experimental]
  adapters: {allow: ["*"], max_scale: {abliteration: 1.5}}
- name: n8n-hx
  sha256: <...>
  scopes: [system1, system2]
  adapters: {allow: [abliteration], max_scale: {abliteration: 1.0}}
```

### Supervisor: profile args → llama-server flags
| Profile key | llama-server flag |
|---|---|
| `model` | `-m /models/<file>` |
| `mmproj` | `--mmproj …` |
| each lora | `--lora /models/<file>`; any lora also adds `--lora-init-without-apply` |
| `ctx` | `-c` |
| `batch` | `-b` |
| `ubatch` | `-ub` |
| `parallel` | `-np` |
| `flash_attn` | `-fa` |
| `ngl: all` | `-ngl 999` |
| `ctx_checkpoints` | `--ctx-checkpoints` |
| `cache_type_k/v` | `-ctk` / `-ctv` |
| `reasoning_format` | `--reasoning-format` |
| `jinja` | `--jinja` |
| (always) | `--host 127.0.0.1 --port <internal> --no-webui --metrics` |

Task A1 verifies the exact flags against the pinned build. A unit test renders the argv for each profile. Unknown keys are a validation error, and raw flags are not passed through.

### Gateway endpoints (Phase A)

**`POST /v1/systemone`** (scope `system1`)
- Clef decision, official schema.
- Validates the question limits.
- Renders and tokenizes the prompt and checks it against `max_prompt_tokens`.
- Unsupported images → 501.
- Any `x_adapters` other than `{clef}` → 400.
- Loads S1 on demand.
- Returns the answers plus `x_brain`.

**`POST /v1/chat/completions`** (scope `system2`)
- OpenAI chat with SSE and `reasoning_content`.
- Extensions:
  - `x_adapters`;
  - qualified model names;
  - `x_adapters_mode`;
  - `x_task`;
  - `x_thinking_budget` (maps to `reasoning_budget_tokens`).
- Also supports `response_format` (json_schema / json_object) and `image_url` data URLs.
- Rejects raw `lora`, `tools`, `n>1` and unknown `x_*` fields.

**`GET /v1/models`** (any scope)
- Roles with their capabilities and defaults (including default stacks).
- Concrete models (revision + sha256).
- Profile, phase and residency state.

**`GET /v1/adapters`** (any scope)
- Registered adapters: public manifest fields, `loaded` per backend, `gate`, roles, scale range, `incompatible_with` / `requires`, and an eval summary.
- Also lists the profile's `approved_stacks`.

**`GET /v1/router`** (any scope): router tasks and their resolved stacks.

**`GET /health`, `/health/kuma`, `/health/deep`** (no auth; kuma and deep return no details): see the health table.

**`GET /metrics`** (no auth, tailnet only): Prometheus text with per-role latency and queue depth, `brain_vram_gb`, lease state, residency, and counts per adapter and per stack.

**`POST /admin/reload`** (scope `admin`): re-reads the registry and restarts the affected backends. Returns 409 `leased` during a lease.

**`x_brain` response metadata** (in the JSON body; in the final chunk for SSE):
- `profile` and `phase`;
- `backend_model`;
- `adapters` (the resolved stack) and `stack_approved`;
- `route`;
- `latency_ms`, `queue_ms` and `prompt_tokens`;
- `cold_start_ms`, when S1 was loaded on demand.

**Errors** (OpenAI envelope):

| HTTP | Code |
|---|---|
| 400 | `invalid_request`, `context_length_exceeded` (with count and limit), `unsupported_parameter`, `invalid_adapter` (with reason), `ambiguous_adapters` |
| 401 | `unauthorized` |
| 403 | `forbidden`, `adapter_forbidden`, `stack_not_approved` |
| 404 | `model_not_found` |
| 409 | `leased` |
| 429 | `queue_full` (+ `Retry-After`) |
| 501 | `capability_unavailable` |
| 503 | `brain_unavailable`, with `reason ∈ {loading, gpu_lease, waiting_for_vram, draining, degraded}` (+ `Retry-After`) |
| 504 | `queue_timeout` |

### Image and compose stack

**Dockerfile**
- `FROM ghcr.io/ggml-org/llama.cpp:server-cuda-<build>@sha256:<digest>`. The build must be newer than #30100 and include #29969.
- Adds `tini`, Python 3.12 and a venv with the gateway wheel.
- `USER 1000:1000`.
- `ENTRYPOINT ["tini","--","brain","serve"]`.

**`compose.yaml`** (`~/srv/docker/apps/brain/`)
- **Service:** `brain`, image `brain:<version>` (built locally and pinned), `container_name: brain`, `restart: unless-stopped`.
- **GPU:** `deploy.resources.reservations.devices: [{driver: nvidia, count: all, capabilities: [gpu]}]` and `NVIDIA_DRIVER_CAPABILITIES=compute,utility`.
- **Environment:** `TZ`, `BRAIN_PROFILE`.
- **Ports:** `["8790:8790"]`.
- **Volumes:**
  - `./config:/etc/brain:ro`
  - `./registry:/opt/brain/registry:ro`
  - `./state:/var/lib/brain`
  - `${HOME}/srv/models/brain:/models:ro`
  - `/run/gpu-lease:/run/gpu-lease`
- **Networks:** `default` and `brain_net` (external).
- **Labels:** `com.centurylinklabs.watchtower.enable: "false"`.
- **Healthcheck:** `/health/kuma` contains `BRAIN_OK`; interval 30 s, start period 180 s.
- **Logging:** json-file, 20m × 5.
- **No Docker socket and no `pid: host`.**

**Host units**
- `brain-firewall.service` (After/Requires docker): `DOCKER-USER` rules so that port 8790 accepts only `tailscale0`, loopback and the docker bridges, and drops everything else. This is needed because published ports bypass ufw.
- `tmpfiles.d/gpu-lease.conf`.
- The tts-trainer drop-in.

**hx and monitoring**
- **Caddy on hx:** `brain.hx` → `ranger:8790` with `flush_interval -1` and a 15-minute response timeout.
- **Kuma:** keyword monitors for `BRAIN_OK` and `DEEP_OK`.

**Backups and upgrades**
- **restic** includes `~/srv/docker/apps/brain` and excludes `~/srv/models`. `brain models pull/build` re-creates the models.
- **Upgrades:**
  1. Bump the llama.cpp build.
  2. Re-run the Experiment 0 `clef_reference` subset and the adapter gates.
  3. Rebuild the image.
  4. `docker compose up -d`.

### Python SDK
**Construction:** `brain_client.Client(endpoints=["https://brain.hx", "http://127.0.0.1:8790"], token=…)`.

**Methods:**
- `decide(request)`
- `chat(messages, adapters=None, task=None, thinking=None, response_format=None, stream=False, …)`. `adapters` is a `{name: scale}` dict that maps to `x_adapters`.
- `models()`, `adapters()`, `health()`

**Failover:**
- Endpoints are tried in order, with a 2 s connect timeout.
- It fails over on connection errors and on any 503 `brain_unavailable`.
- It never fails over on 4xx, 429 or 504.
- The working endpoint stays sticky for 5 minutes.
- Results carry `served_by`, `profile` and `adapters`.

### Phase A tasks
| # | Task |
|---|---|
| A1 | Verify on the pinned llama.cpp build: flag names, `/lora-adapters`, `id_slot`, `reasoning_budget_tokens` and the load-log buffer lines. Record them in `deploy/ranger/docs/ops.md`. |
| A2 | Registry + validator + `brain profile check`. Model, adapter and profile manifests. Stack resolution: `x_adapters`, qualified names, merge, validation reasons, stack approval. Token policy. Static router. |
| A3 | Clef prompt renderer + token-fit check, with a parity test. |
| A4 | Supervisor: argv builder, start/stop/restart/backoff, crash loop → `degraded`, on-demand residency with idle TTL and the VRAM-gated load. |
| A5 | VRAM guard + lease state machine + heartbeat. `gpu-lease` host tool. |
| A6 | Endpoints, error envelope, SSE passthrough with cancellation, queues with stack-affine slots, `x_brain`, health/metrics/admin, `/v1/adapters`, `/v1/router`. |
| A7 | CLI: `models pull/build/verify`, `adapters add/list/verify/eval`, `tokens add/revoke`. |
| A8 | Abliteration adapter: extraction → GGUF → manifest → gate (with Experiment 0). |
| A9 | Deploy: Dockerfile, compose, host units, Caddy, Kuma, restic exclude, `brain_net`, tts-trainer drop-in. |
| A10 | SDK with failover. |
| A11 | Tests (below). |

A2-A7 and A10 don't need the GPU, so they run **in parallel with Experiment 0**. A9 waits for E0's measured profiles.

### Phase A tests

**Unit**
- **Registry validator:** each rule, including `requires`, `incompatible_with`, `weights_sha256` / `quant_compat` mismatch, and `batch < ubatch`.
- **Stack resolution:**
  - precedence;
  - scale-0 removal;
  - `+none`;
  - replace mode;
  - every `invalid_adapter` reason;
  - `stack_not_approved` vs the experimental scope.
- **Parsing and policy:** qualified-name parser, token policy, router.
- **Other:** argv builder per profile, the health mapping table, the lease state machine (fake fs + fake NVML, startup-with-request, withdrawal), and the load-log parser.

**Integration (no GPU)**
- A **fake llama-server** (a small HTTP app for `/v1/systemone`, chat SSE, `/tokenize`, `/lora-adapters` and `/health`) with controllable latency, crashes and memory, plus a fake NVML.
- Covers:
  - routing;
  - adapter name → `lora` id mapping, with explicit zeros;
  - stack-affine `id_slot` when `parallel: 2`;
  - queues and 429;
  - disconnect cancellation;
  - on-demand load and idle unload;
  - waiting-for-VRAM;
  - lease drain and resume;
  - crash-loop degradation;
  - SDK failover to a second gateway.

**GPU (`-m gpu`, on ranger)**
- renderer parity;
- adapter scale-0 identity and speed;
- two-adapter stack smoke test;
- load timings and footprint vs host `nvidia-smi`;
- JSON-schema 50/50.

---

## Experiment 0: the 4090 sweet spot (first; parallel with the Phase A build)

**Goal:** `ranger-od-a` (case a: steady state + S1 peak and cold start), `ranger-12g-a` (case b), and a ≤ 10 GB reference point. It also validates S1 calibration per quant, the abliteration adapter, stacking and the budget. Everything runs **on the host** with per-PID NVML, one process per configuration.

### Staged matrix
1. **S1 Clef.**
   - Quants: BF16 (reference, in a lease window), Q8_0, Q6_K, Q5_K_M, Q4_K_M and IQ4_XS, all stripped. A one-time check confirms stripping is lossless.
   - `batch = ubatch = ctx` ∈ {2048, 4096, 8192}.
   - Measure the cold-load time.
   - Use `llama-quantize` with the head kept at full precision.
2. **S2 Qwen + adapters.**
   - Stage 1: quant {Q8_0, Q6_K, Q5_K_M, Q4_K_M, IQ4_XS} × ctx {8k, 16k, 32k}, with parallel 1, ubatch 512, f16 KV, mmproj Q8_0 and 4 checkpoints.
   - Stage 2, around the top 3: KV q8_0, ubatch 1024, and **parallel 2 with stack-affine slots**.
3. **Pairs.**
   - Combined peak and interference.
   - Thinking budget.
   - Ocular's own per-PID burst (refines the case (a)/(b) numbers).
   - The trainer's per-PID peak (sets `need_gb`).

### S1 calibration gate (`clef_reference`)
- **Data:** ~150 synthetic records.
- **Reference:** the BF16 Clef GGUF (stripped, ~16.4 GB) under a lease.
- **Upstream cross-check:** the official `joint_schema_model.py` in bf16 on CUDA, with merged weights and no vision tower, on 50 records of ≤ 2k tokens. Fall back to 20 records on CPU if needed. Pass if max |Δp| ≤ 0.01.
- **Per-quant gate:**
  - top-1 agreement ≥ 98%, excluding near-ties (margin < 0.02);
  - p99 |Δp| ≤ 0.03;
  - max |Δp| ≤ 0.10.

### Adapter checks (the abliteration gate, written into its manifest)
- **Compared with the merged `wangzhang` GGUF at the same quant:**
  - refusals within ±3 per 100 probe prompts;
  - mean KL ≤ 0.01 on harmless text.
- **Scale 0:** token-identical to stock on 20 greedy prompts.
- **Speed:** ≥ 97% of stock when active, and ≥ 99% when loaded at scale 0.
- **Stacking smoke test:** abliteration + a dummy rank-8 LoRA, both at non-zero scale, produce output that differs from either one alone. This tests the wiring, not quality.

### Other S2 checks
- Per-quant KL vs BF16.
- Vision sanity on 10 images.
- JSON schema valid on 50/50 prompts, with thinking on and off.

### Selection
1. **Feasible configurations:** measured peaks fit each case's budget, and the S1 quant passes the gate.
2. **Minimum bars:**
   - S2 ≥ 60 tok/s;
   - warm S1 p50 ≤ 1 s at 2k tokens;
   - S1 cold start ≤ 5 s for `ranger-od-a`.
3. **Ranking:** by S1 quant, then S2 KL, then speed.

**Output:** the two profiles with measured peaks, a report in `eval/reports/exp0/`, and the E0 checkpoint.

---

## Experiment 1: shot assignment pilot (a capability test, not the product)

### Questions
1. Is S1 good enough for this **class** of task: picking from a catalog of short descriptions given natural text?
2. What does deployment cost?

The answers feed the **now / later / enterprise** decision.

### Task
- **Input:**
  - a style with 2-15 shots `{shot_id, name, note}`;
  - optional **general notes**: free-text rules;
  - an ordered script of sections.
- **Output:** one shot per section.
- **No fixed rules.** Both systems get the general notes. The rev-2 solver is dropped.
- **Note style is a variable:** terse vs descriptive.

### S1 framing
The prompt must be ≤ the S1 `max_prompt_tokens`.
- **F-joint-windowed:** one `choice` per section, over groups of G sections. The state holds the style, the general notes and the full script.
- **F-single:** one request per section. That is about 27k tokens per video at S=12, K=15, because there is no prefix reuse.

The pilot compares both.

### Arms
| Arm | Description |
|---|---|
| `floor-bm25` | Lexical baseline |
| `s1` | S1 argmax |
| `s2-think` / `s2-nothink` | Constrained JSON for the whole assignment |
| `s1-esc` | S1, with sections below τ escalated to S2 in one call per video, the enum restricted to S1's top-5. τ is set on dev |

**Profiles:**
- the Experiment 0 profile;
- S1 at BF16 in a lease window;
- S2 with the default stack (`abliteration: 1.0`) and with `system2+none`. This checks that abliteration doesn't hurt the task, and it is the first real use of per-request adapter selection.

### Dataset
- **Pilot:**
  - 2 styles (6 and 14 shots), each in a terse and a descriptive version;
  - one style has general notes with 2 rules;
  - 20 videos, ~180 sections;
  - split 8 dev / 12 test.
- **Full run** (only if warranted):
  - 6 styles with K ∈ {2, 4, 6, 9, 12, 15};
  - ~1,200 sections;
  - split 40 dev / 80 test.
- **Generation:**
  - written by Claude, offline;
  - shot sequence first, then text;
  - two blind label passes produce acceptable-set gold;
  - mandatory review of flagged items.
- **Blind user audit:** 60 pilot sections (≥ 300 for the full run). Target ≥ 90% agreement on clear items and ≥ 80% on ambiguous ones.
- **Integrity:**
  - no Qwen generation or labelling;
  - a hard slice of sections with overlap margin ≤ 0;
  - the dataset is frozen and hashed.

### Metrics
- **Primary:** top-1 acceptable-set accuracy.
- **Secondary:**
  - top-3 and video-exact accuracy;
  - rule adherence;
  - S1 ECE, Brier and AUROC;
  - risk-coverage;
  - latency and tokens per video;
  - peak VRAM;
  - S1 agreement between the profile and BF16;
  - breakdowns.
- **Statistics:** cluster-bootstrap CIs. The pilot is directional only.

### Decision output (proposed)
| Verdict | Criteria |
|---|---|
| **S1 good enough for this class** | Within 3 pts of the best S2 arm, ECE ≤ 0.05, and ≥ 3× faster |
| **S1 + escalation** | AUROC ≥ 0.80, and escalating ≤ 25% closes the gap to within 2 pts |
| **S2** | Otherwise. S1 is still a shortlister if top-5 recall ≥ 98% |

**Cost card per tier** (ranger profiles, estimated 16 GB Mac, customer ≥ 12 GB GPU): latency per video, memory and quality vs the reference.

---

## Phase B: one resident base (gated)

**Goal:** about 8.5-9.5 GB in a single process, fully resident in both Ollama cases. It is also the natural edge profile.

**B1: patch llama-server** (recommended; aim to upstream it as "decision head on a base model + LoRA")
- **Loading:**
  - the stock qwen35 GGUF;
  - `--lora` for every generation and decision adapter;
  - `--lora-init-without-apply`;
  - `--decision-head <name>=<head.gguf>`, with the head extracted from the Clef GGUF.
- **`/v1/systemone` on a reserved slot:**
  1. Set adapters to decision = 1, everything else = 0.
  2. Turn on embeddings with pooling none.
  3. Run a **chunked prefill**, which removes #30073.
  4. Run the head graph (ported from `build_head` in `src/models/clef.cpp`).
  5. Restore.
- **S2 slots** are unchanged.
- **To verify:**
  - the hidden-state tap point (post-final-norm);
  - zero-scale adapters cost nothing;
  - the recurrent cell for the reserved sequence;
  - the Clef LoRA survives the linear-attention permutation.

**Rejected alternatives**
- **B2, a new libllama server:** it would have to reimplement chat, grammar, the reasoning parser and multimodal.
- **B3, SGLang:** untested, its LoRA skips `in_proj_a/b`, it needs a GPTQ base, and it has no Mac support.

**Clef extraction**
- Rank ≤ 256, energy ≥ 0.998, element error ≤ max(2 bf16 ULP, 1e-6).
- PEFT → qwen35 LoRA GGUF.
- The head goes to `clef-head.gguf`.
- Manifest: `kind: decision`, `artifact.format: gguf-lora+head`.

**Ship gate**
- `clef_reference` passes against the BF16 Clef GGUF.
- Pilot S1 argmax agreement with Phase A is ≥ 98%.
- S2 output is token-identical to Phase A at the same quant and stack.
- Peak ≤ 10 GB.
- An 8k S1 prompt works with ubatch 512.
- The S2 cache survives an S1 call.

**Maintenance:** a pinned patch, with the gates re-run on every bump.

---

## Later phases (outline)

**Edge Mac**
- **Same software:** the same gateway, run natively (uv tool or a bundle) under launchd, with pinned llama.cpp Metal builds newer than #30100, the same registry and a `mac-16g` profile.
- **Memory:** the default wired limit is about 10.7 GB on a 16 GB Mac (inferred).
  - **Phase B** fits within it.
  - **Phase A** needs `iogpu.wired_limit_mb` raised. That is acceptable because the model never runs during renders.
- **Operation:**
  - No lease. Unloading uses `unload` or the idle TTL.
  - SDK failover tries the lab first, then the local brain.
- **MLX:** only an optional future backend.

**Sharing and enterprise**
- **Distribution:** the image and a bundle, `brain models pull/build`, and cost cards.
- **Access:** Tailscale node sharing, plus per-person tokens with scopes, adapter policies (OQ-3) and rate limits.
- **Licences:** Apache-2.0 weights and MIT llama.cpp; ship a NOTICE.
- **Routing:** S1-driven and learned routing.

---

## Components Affected
- [ ] Repository: `gateway/`, `sdk/`, `tools/`, `registry/`, `deploy/ranger/`, `lab/`, `tests/`, `CLAUDE.md`, `README.md` (layout above). Later: `deploy/mac/` and `phaseb/`.
- [ ] External, done during deployment:
  - Caddy on hx;
  - Kuma monitors;
  - the tts-trainer drop-in;
  - `brain-firewall.service` and `tmpfiles.d`;
  - the restic exclude;
  - the ranger admin notes.
- [ ] Not in scope: changes to Ocular or Ollama.

**Licences**

| Dependency | Licence | Used in |
|---|---|---|
| llama.cpp | MIT | |
| fastapi, pydantic | MIT | |
| uvicorn | BSD-3 | |
| httpx | BSD-3 | |
| nvidia-ml-py | BSD | |
| PyYAML | MIT | |
| jsonschema | MIT | |
| rank_bm25 | Apache-2.0 | lab |
| numpy, matplotlib | BSD / PSF | lab |
| torch, transformers, peft, trl | BSD / Apache-2.0 | lab and fine-tuning only |
| Model weights | Apache-2.0 | |

No copyleft.

---

## Data Flow
```
chat:      client ─token─► gateway ─ auth ─ resolve(role, stack = role default ⊕ router(x_task) ⊕ x_adapters) ─ validate + approve
           ─ queue (stack-affine slot) ─► s2 llama-server {lora:[{id,scale} for all loaded], chat_template_kwargs,
           reasoning_budget_tokens, response_format} ─► SSE passthrough (+ x_brain in last chunk) ─► client
systemone: client ─► gateway ─ validate (limits, render+tokenize ≤ max_prompt_tokens) ─ [on_demand: VRAM check → load s1]
           ─► s1 llama-server /v1/systemone ─► answers + x_brain ─► client
lease:     trainer ExecStartPre ─► gpu-lease acquire (O_EXCL) ─► gateway DRAINING ─► stop backends ─► granted ─► trainer
           ─► ExecStopPost release ─► gateway LOADING ─► SERVING
adapter:   train (lease brain-train) ─► convert GGUF ─► brain adapters add ─► brain adapters eval (gate) ─► profile loras ─► /admin/reload (restarts s2)
```

---

## Implementation Plan
1. **Experiment 0**, in parallel with the GPU-free Phase A tasks (A2-A7, A10). E0 needs `gpu-lease` (A5), the abliteration extraction (A8) and the lab tooling.
   - *Checkpoint E0:* the profiles.
2. **Phase A deploy** (A9 + A11 GPU tests) with `ranger-od-a`. Switch to `ranger-12g-a` if Ollama is ever retired.
3. **Experiment 1 pilot.**
   - *Checkpoint D1:* is S1 good enough for this class of task? Now, later or enterprise? Full run?
4. **Phase B** (gated): Clef extraction + head, the llama-server patch, the gate, and a `ranger-10g-b` profile.
5. **Phase C:** edge Mac.
6. **Phase D:** sharing and enterprise.

The first real fine-tuned adapter can enter at any time after Phase A, using the workflow above. S1-driven routing is optional after Phase A.

---

## Verification Plan

#### Behavioral Checks

**Experiment 0**
- [ ] **Matrix records:** each cell records the per-PID VRAM peak (host), load time, latency or throughput, the GGUF sha256 and the build tag.
- [ ] **Stripping is lossless:** removing `output.weight` changes probabilities by ≤ 1e-6 on 20 records.
- [ ] **Upstream cross-check:** max |Δp| ≤ 0.01.
- [ ] **Per-quant gate report:** `clef_reference` lists top-1 agreement and p99/max |Δp| for each quant, with pass flags that agree with the thresholds.
- [ ] **Abliteration gate:** the manifest has `evals.gate: passed`, with filled metrics:
  - refusals within ±3 per 100;
  - KL ≤ 0.01;
  - scale 0 identical to stock;
  - speed ≥ 97% active and ≥ 99% at scale 0.
- [ ] **Stacking smoke:** on 5 greedy prompts, two non-zero adapters give output different from each adapter alone.
- [ ] **`ranger-od-a`:** all measured peaks filled in, validator passes, S2 steady state ≤ `steady_gb`, S1 cold load ≤ 5 s.
- [ ] **`ranger-12g-a`:** either passes the validator with a sum ≤ 12.0, or the report says why not and gives the measured Ocular own-burst.

**Registry, adapters, stacking**
- [ ] **Profile check catches:**
  - an adapter whose `hf_revision` or `weights_sha256` differs from its base (both values named in the error);
  - a base quant not in `quant_compat`;
  - `batch < ubatch` (both values named);
  - a default stack that violates `requires` or `incompatible_with`.
- [ ] **Each invalid stack returns 400 `invalid_adapter`** with the right reason, and the backend request counter stays unchanged:
  - unknown adapter (`x_adapters: {"nope": 1}`);
  - not loaded;
  - wrong role (`clef` on `system2`);
  - scale out of range (`abliteration: 2.0`);
  - `incompatible_with` violated;
  - `requires` missing;
  - exclusive group violated;
  - more than `max_stack` adapters;
  - an ungated adapter.
- [ ] **Unapproved two-adapter stack:**
  - normal token → 403 `stack_not_approved`;
  - token with experimental scope → served, with `x_brain.stack_approved: false`.
- [ ] **Approved stacks:** after adding the stack to `approved_stacks` and running `/admin/reload`, a normal token is served and `stack_approved` is true.
- [ ] **Token scale limit:** a scale above the token's `max_scale` → 403 `adapter_forbidden`.
- [ ] **Stack resolution:**
  - `model: "system2+abliteration@0.5"` → `x_brain.adapters == {abliteration: 0.5}`;
  - `system2+none` → `{}`;
  - a qualified name plus `x_adapters` → 400 `ambiguous_adapters`;
  - `x_adapters: {abliteration: 0}` removes the default.
- [ ] **Explicit scales for every loaded adapter.** With only abliteration loaded, the fake backend receives `lora: [{id: 0, scale: 1.0}]` by default and `[{id: 0, scale: 0.0}]` for `+none`.
- [ ] **Stack-affine slots:** with `parallel: 2` and stack affinity on, requests with the default stack carry `id_slot: 0` and other stacks carry `id_slot: 1`.
- [ ] **Router:**
  - `x_task: "caption"` → role `system2-fast`, stack `{}`;
  - unknown task → 400;
  - `GET /v1/router` lists the tasks.
- [ ] **`GET /v1/adapters`:**
  - `abliteration`: generation, loaded on s2, gate passed, `incompatible_with` includes `clef`;
  - `clef`: decision, bound to s1, not stackable.
- [ ] **Adding an adapter:** a manifest + profile entry + `/admin/reload` restarts s2 only. The s1 PID is unchanged, the new adapter shows `loaded: true`, and a "profile not re-measured" warning appears in `/health`.

**Gateway and API**
- [ ] **Prompt-size parity:** on 20 requests, the renderer's token count equals the backend's prompt tokens.
- [ ] **systemone validation:**
  - 256 options → 400;
  - `score` with 11 levels → 400;
  - over `max_prompt_tokens` → 400 `context_length_exceeded` with count and limit;
  - images on a role without them → 501;
  - `x_adapters: {abliteration: 1}` → 400 `invalid_adapter` (role).
- [ ] **systemone happy path:** the official example returns 200 with 3 answers, choice probabilities summing to 1 ± 1e-4, and `x_brain.profile == BRAIN_PROFILE`.
- [ ] **Thinking:**
  - `system2` streaming emits `reasoning_content`, then `content`, then `[DONE]`;
  - `system2-fast` emits no `reasoning_content`;
  - `x_thinking_budget: 64` keeps reasoning ≤ 64 tokens plus the close, and `content` is non-empty.
- [ ] **JSON schema:** valid output on 50/50 prompts, with thinking on and off.
- [ ] **Unsupported parameters:** raw `lora`, `tools` and `n: 2` each return 400 `unsupported_parameter`.
- [ ] **Profile swap:** change `BRAIN_PROFILE` and `compose up -d`. The brain serves again within 60 s, with role names unchanged and the new concrete models shown.

**Residency, VRAM guard, lease**
- [ ] **On-demand S1:** the first S1 request after idle has `cold_start_ms > 0`, and a second within the TTL has 0. After `idle_ttl_s + 30 s`, the host `nvidia-smi` shows exactly one `llama-server` PID.
- [ ] **VRAM-gated S1 load:**
  - with a dummy CUDA process holding enough memory that free < S1 peak + 0.5 GB, an S1 request with `timeout_ms: 10000` returns 503 `waiting_for_vram` after ≈ 10 s, and no S1 PID appears;
  - with a longer timeout, the request succeeds after the dummy exits.
- [ ] **Startup without enough VRAM:** free memory < steady budget + 0.5 → `waiting_for_vram`. `/health/kuma` returns `BRAIN_OK` for ≤ 10 min, then `BRAIN_DOWN:waiting_for_vram`. Once memory is freed, the brain is `serving` within 60 s.
- [ ] **Footprint accuracy:** after warm-up, the host sum of `llama-server` memory stays ≤ the active budget over 100 max-size mixed requests, and `brain_vram_gb` is within ±0.5 GB of it.
- [ ] **Lease handshake:** `gpu-lease acquire --holder test --need-gb 10` while serving gives:
  - 503 `gpu_lease` for new requests;
  - in-flight requests finish or are cancelled by 60 s;
  - no `llama-server` PIDs;
  - `granted` is written;
  - exit code 0.

  After `release --holder test`, the brain is `serving` within 60 s, and `/health/kuma` sampled every 5 s shows only `BRAIN_OK`.
- [ ] **Lease concurrency:** two concurrent acquires never yield more than one `request`, and B exits 0 only after A releases.
- [ ] **Release ownership:** a wrong-holder release leaves `request` intact.
- [ ] **Acquire timeout:** a timed-out acquirer deletes its own `request` and exits non-zero.
- [ ] **Brain absent:** with the brain stopped, `acquire` returns within 5 s.
- [ ] **Startup during a lease:** starting with `request` present goes straight to `leased`.
- [ ] **Overdue lease:** `BRAIN_DOWN:lease_overdue`, and no reload happens.
- [ ] **Reboot:** after a reboot, `/run/gpu-lease` exists and is empty.

**Deployment**
- [ ] **Remote access:** from the laptop over the tailnet, `curl https://brain.hx/health` returns 200, and the first SSE delta through Caddy arrives in < 2 s.
- [ ] **Network exposure:**
  - from a LAN host, `192.168.1.215:8790` fails;
  - from a tailnet host it succeeds;
  - `docker port brain` lists only 8790.
- [ ] **Auth:** no token → 401.
- [ ] **Watchtower:** `docker inspect brain` shows the opt-out label and the pinned image.
- [ ] **Backups:** the latest restic snapshot contains `~/srv/docker/apps/brain/registry` and nothing under `~/srv/models`.
- [ ] **Monitoring:** both Kuma monitors are green.

**SDK** (second gateway with fake backends as `local`)
- [ ] **Failover on a lab lease:** `decide()` returns `served_by == local`. After the lease is released and the 5-minute stickiness expires, it returns `brain.hx`.
- [ ] **No failover on client errors:** a 400 from the lab is raised and never sent to `local`.
- [ ] **Adapter mapping:** `chat(adapters={"abliteration": 0.5})` is sent as `x_adapters: {"abliteration": 0.5}`, and `x_brain.adapters` comes back.

**Experiment 1 pilot**
- [ ] **Dataset validator** exits 0. It checks that:
  - every acceptable set contains the intended shot;
  - every flag is reviewed or excluded with a reason;
  - hashes match;
  - audit agreement is recorded.
- [ ] **Prompt fit:** the run log has no prompt-fit 400s.
- [ ] **Report contents:** per arm, the report has:
  - primary accuracy with CI;
  - top-3;
  - S1 ECE and AUROC;
  - latency and tokens per video;
  - peak VRAM;
  - BF16 agreement;
  - breakdowns;
  - rule adherence;
  - default stack vs `+none`;
  - the verdict;
  - the cost card.

**Phase B ship gate**
- [ ] `clef_reference` passes.
- [ ] Pilot S1 argmax agreement with Phase A ≥ 98%.
- [ ] S2 token-identical on 20 prompts.
- [ ] Peak ≤ 10.0 GB.
- [ ] An 8k S1 prompt runs at ubatch 512.
- [ ] An S2 slot keeps `cached_tokens > 0` after an S1 call.

#### Edge Cases
- [ ] **Reload during a lease:** `POST /admin/reload` → 409 `leased`.
- [ ] **S2 crash:** `kill -9` on s2 makes `system2` return 503 `degraded` while `system1` keeps serving. Restart backoff is 1, 5, 30 s, and the restart count appears in `/health`.
- [ ] **Crash loop:** 5 failures in 10 min → `degraded` / `BRAIN_DOWN:degraded`. Restarts stop until `/admin/reload`.
- [ ] **S1 queue overflow:** 40 concurrent S1 requests → 32 are queued and the rest get 429. Queued requests whose client disconnected are dropped without reaching the backend.
- [ ] **Lease vs on-demand load:** an on-demand S1 request during a lease gets 503 `gpu_lease`, and nothing loads.
- [ ] **Idle unload vs S2:** an idle unload of S1 doesn't interrupt a running S2 stream.
- [ ] **Bad image:** a malformed image data URL → 400.
- [ ] **Non-Qwen S2:** with S2 swapped to a non-Qwen3.5 GGUF and no adapters:
  - `system2` serves;
  - `abliteration` shows `loaded: false`;
  - requesting it returns 400 `invalid_adapter: not_loaded`;
  - `system1` is unaffected.
- [ ] **Base mismatch:** a profile that loads `abliteration` on a non-matching base fails `brain profile check` with `base_mismatch`.
- [ ] **Stale grant:** a `granted` file without a `request` is treated as released, with a warning.
- [ ] **Withdrawn request during drain:** the brain returns to `serving` without stopping its backends.
- [ ] **Free VRAM inside the margin:** free VRAM between the budget and budget + 0.5 at startup → `waiting_for_vram`.

---

## Cross-cutting concerns
- **Security:**
  - Reachable from the tailnet only (`DOCKER-USER`).
  - Tokens with scopes and adapter policy.
  - Unapproved stacks are rejected for normal tokens.
  - Backends listen on loopback.
  - No Docker socket and no `pid: host`.
  - Request bodies are not logged.
- **Failure modes:**
  - The VRAM footprint is static, and loads are VRAM-gated, so the brain never causes another service's OOM.
  - A crash only affects its own role.
  - Leases drain cleanly.
  - The SDK fails over.
- **Idempotency:** S1 is deterministic per profile. Lease operations are holder-checked and idempotent. Model pulls and adapter registration are verified by sha256.
- **Versioning:**
  - Profiles and manifests (model, adapter, router, approved stacks) live in git.
  - The image is pinned.
  - Any change to the build, a quant or an adapter re-runs `clef_reference` and the adapter and stack gates.
  - `/v1/models` and `/v1/adapters` show the exact artifacts.
- **Performance:**
  - S1 is a full prefill, queued.
  - A stack change costs a re-prefill (mitigated by stack-affine slots).
  - The S1 cold start applies only to `ranger-od-a`.
  - Interference between S1 and S2 is measured in Experiment 0.
- **Testing:** unit tests, fake-backend integration tests, GPU tests on ranger, and lab gates.

---

## CLAUDE.md Updates
Create `/Users/ale/gits/clef/CLAUDE.md`. Start a stub in Experiment 0 and finish it in Phase A.

**Purpose:** "brain" is the lab's LLM and decision API. `system1` serves Clef decisions; `system2` serves Qwen with stackable generation adapters. Built on llama.cpp, hosted on ranger. **No captioning or detection.**

**Invariants:**
- Clients use roles and adapter names, never file paths.
- Generation adapters never run on `system1`, and decision adapters never run on `system2`.
- Every adapter has a manifest with a pinned base (revision + weights sha256) and `quant_compat`, and must pass its gate before default use.
- Multi-adapter stacks must be in `approved_stacks`, otherwise they are rejected unless the token has the experimental scope.
- The gateway always sends explicit scales for every loaded adapter.
- Adding an adapter restarts s2.
- Profiles carry measured peaks, and `batch ≥ ubatch`.
- No Docker socket and no `pid: host`.
- Backends listen on loopback only.
- Loads require free VRAM.
- The brain never reclaims an overdue lease, and only the holder releases a lease.
- The image is pinned and excluded from Watchtower.
- No cloud calls at runtime.

**Commands:**
- `brain profile check`
- `brain models pull|build|verify`
- `brain adapters add|list|verify|eval`
- `brain tokens add|revoke`
- `gpu-lease status|acquire|release`
- `docker compose up -d`
- `pytest -m "not gpu"` / `pytest -m gpu`
- `python -m lab.exp0` / `python -m lab.exp1`

**Key locations:**
- `registry/` (models, adapters, profiles, `router.yaml`)
- `~/srv/docker/apps/brain/`
- `~/srv/models/brain/`
- `/run/gpu-lease/`
- `eval/`

**Upstream behaviours to respect:**
- #30073: in Phase A, the S1 prompt must fit in the ubatch.
- #30419: no MTP together with JSON schema.
- #30357: `<think>` whitespace.
- Builds must be newer than #30100.
- `batch` caps `ubatch`.
- A stack change clears the slot cache.

**Outside this repo:**
- Update ranger's admin notes: add the brain stack, the lease and the adapter workflow, and fix the outdated Ocular/Ollama sections.
- The global `~/.claude/CLAUDE.md` is unaffected.

---

## Open Questions (remaining)
1. **OQ-1 Ollama's future on ranger.** Your call, and outside brain scope. The brain ships `ranger-od-a` (works in both cases) and switches to `ranger-12g-a` if Ollama is retired.
2. **OQ-2 ≤ 14 GB interim.** Is a ≤ 14 GB interim acceptable for `ranger-12g-a`? It would apply only in case (b), and only if Experiment 0 measures Ocular's own burst at ≤ 1 GB.
3. **OQ-3 Abliteration default for shared users.** On for everyone, or only for your own tokens, with shared users getting stock?
4. **OQ-4 Phase B patch.** Are you willing to carry a pinned llama-server patch until it is upstreamed?
5. **OQ-5 Trainer delay.** Is it acceptable for `tts-trainer` to start up to ~15 minutes late while the brain drains?
6. **OQ-6 Pilot inputs.** Will you write or approve the two pilot styles, including the general notes, and do the 60-section blind audit?
7. **OQ-7 First fine-tune.** Which task should be the first real fine-tuned adapter? This can wait, but it shapes the first `lab/tasks/` eval set.
8. **OQ-8 Hostname.** Is `brain.hx` → `ranger:8790` OK?

---

## Risks & Mitigations
| Risk | Impact | Mitigation |
|---|---|---|
| Phase A doesn't fit at peak while Ollama stays | Contention | `ranger-od-a`: S2 resident, S1 on demand with VRAM-gated loads; Ollama falls back to CPU offload in the rare overlap; Phase B |
| Low quants degrade Clef calibration | Bad S1 decisions | Per-quant `clef_reference` gate plus an upstream cross-check |
| Adapter stacks interact unpredictably | Quality regressions | Stacks evaluated as units (`approved_stacks`); unapproved stacks rejected for normal tokens; `stack_approved` in responses |
| Adapter used on the wrong base or quant | Garbage outputs | Manifest `hf_revision` + `weights_sha256` + `quant_compat`, validated at load and on every request |
| Growing adapter count inflates VRAM or restarts | Budget creep, brief S2 downtime | Manifest `size_mb` counted by the validator; re-measure warning; S2-only restart (~3 s); Phase B hot-load is a later option |
| Stack churn kills prompt-cache reuse | Slower S2 | Stack-affine slots when `parallel ≥ 2`; router-chosen defaults; measured in Experiment 0 |
| A generation adapter leaks into S1 | Miscalibrated decisions | Separate S1 process (Phase A); reserved slot with forced zeros (Phase B); `incompatible_with` and tests |
| Lease races or a crashed holder | Two holders, or the brain stays down | `O_EXCL` request, holder-checked release, timeout cleanup, tmpfs, `ExecStopPost`, overdue alert |
| On-demand S1 cold starts | Latency spikes | ~2-3 s measured; `cold_start_ms` reported; switch to `ranger-12g-a` in case (b) |
| Phase B patch is hard | No single base | Phase A is a complete product on its own; the patch is pinned and aimed upstream |
| llama.cpp regressions | Silent bugs | Pinned digest; gates on every bump; deep health check |
| Synthetic pilot bias | Wrong now/later call | Sequence-first gold, blind labels, user audit, hard slice; the pilot is directional only |
