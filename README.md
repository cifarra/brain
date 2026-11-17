# brain

System 1 and System 2 for LLMs, behind one local API. It runs on a 16 GB Mac mini.

- **System 1 judges.** You send a situation and typed questions. It returns calibrated probabilities from a
  single forward pass and generates no text. `POST /v1/systemone`, served by Cloudflare Clef-Flash.
- **System 2 reasons.** OpenAI-compatible chat with thinking, JSON output, vision and stackable LoRA adapters.
  `POST /v1/chat/completions`, served by Qwen3.5-9B.

Clients ask for a role (`system1`, `system2`, `system2-fast`) and adapters, never for a model file. The
`registry/` decides what runs, so a model can be swapped without changing any client.

```sh
curl localhost:8790/v1/systemone -d '{
  "state": "Customer message: I was charged twice for my order last week and nobody has replied.",
  "questions": {
    "route":   {"type": "choice", "instructions": "Which team should handle this?",
                "criteria": {"billing": null, "shipping": null, "technical": null}},
    "angry":   {"type": "noul",   "instructions": "Is the customer angry?"},
    "urgency": {"type": "score",  "instructions": "How urgent is this?",
                "criteria": ["can wait", "this week", "today", "right now"]}
  }
}'
```
```json
{"answers": {
   "route":   {"choice": "billing", "probabilities": {"billing": 0.977, "shipping": 0.015, "technical": 0.009}},
   "angry":   {"noul": 0.42},
   "urgency": {"score": 2.21, "probabilities": {"0": 0.029, "1": 0.155, "2": 0.391, "3": 0.425}}},
 "x_brain": {"role": "system1", "backend_model": "clef-flash-q8_0-abf16", "latency_ms": 1578, "prompt_tokens": 324}}
```
That is a real response from the Mac mini, trimmed.

## On a 16 GB Mac mini

Profile `cerebro` (Mac mini M4, 16 GB, Metal):

| | model | peak memory | speed |
|---|---|---|---|
| System 1 | Clef-Flash Q8_0, attention kept in BF16 | 12.4 GiB | median 1.7 s under 1k prompt tokens, 6.3 s at ~1.4k, 13.8 s at ~2.8k |
| System 2 | Qwen3.5-9B Q4_K_M + vision | 7.7 GiB | 18 tok/s decode |

The two models don't fit together, so the gateway keeps one resident and swaps on demand, which takes about
5 s and is reported as `x_brain.swap_ms`. Through the gateway, System 1 passes the calibration gate against
the BF16 reference on 150 of 150 records (p99 |Δp| 0.010). macOS caps GPU memory at about 10.7 GB, so raise the limit
to 14 GiB first: `deploy/cerebro/install.sh` sets it at every boot and runs the gateway under launchd.

The same API runs on CUDA. On an RTX 4090 both models stay resident in 11.4 GiB (`ranger-12g`,
measurements in `lab/exp0/REPORT.md`).

## Run

You need Python 3.12+, [uv](https://docs.astral.sh/uv/), llama.cpp `llama-server` (b11580: calibration was
measured on that build) and the weights.

```sh
uv sync
uv run pytest -q                         # no GPU or weights needed
uv run brain check cerebro               # validate the profile, list missing files, print the llama-server argv
BRAIN_PROFILE=cerebro BRAIN_MODELS_DIR=~/models uv run brain serve     # :8790, web UI at /ui/
```

The weights are not in the repo. Each manifest in `registry/models/` names its upstream repo, revision,
path and sha256. The calibrated Clef is a derived quant, and its `llama-quantize` recipe is in its manifest.
The abliteration adapter is built by `tools/extract_adapter/`.

| Env | Default | |
|---|---|---|
| `BRAIN_PROFILE` | `ranger-12g` | profile in `registry/profiles/` |
| `BRAIN_REGISTRY` | `./registry` | registry dir |
| `BRAIN_MODELS_DIR` | `/models` | root for the manifest file paths |
| `BRAIN_LLAMA_SERVER` | `llama-server` | binary |
| `BRAIN_TOKEN` | unset | optional bearer token. `/health` is always open |
| `BRAIN_HOST` / `BRAIN_PORT` | `127.0.0.1` / `8790` | gateway address. Set a token before binding `0.0.0.0` |
| `BRAIN_BACKEND_PORT_BASE` | `8791` | llama-server backends listen on 127.0.0.1 at base, base+1, ... |
| `BRAIN_CONFIG` | unset | YAML with the same keys plus supervisor timeouts |

## API

| Endpoint | |
|---|---|
| `POST /v1/systemone` | System 1: `{state, questions}` → probabilities per question |
| `POST /v1/chat/completions` | System 2: OpenAI chat, SSE streaming, `response_format`, `image_url` (base64 data URLs) |
| `GET /v1/models`, `/v1/adapters`, `/v1/router` | roles and their models, adapters, router tasks |
| `GET /health` | 200 `BRAIN_OK` when ready, otherwise 503 `BRAIN_DOWN:<status>` |
| `GET /ui/` | web UI: chat, decide, evals, status |

System 1 question types: `choice` returns a distribution over the named options. `noul` returns the probability
that the statement is true. `score` returns a distribution over up to 10 ordered levels, plus the expected level.

System 2 extensions to the OpenAI request:

- `model`: `system2` (thinking on), `system2-fast` (thinking off), or a qualified name such as
  `system2+abliteration@0.5` or `system2+none`.
- `x_adapters: {name: scale}` merges over the role's default stack. A scale of 0 removes an adapter, and
  `x_adapters_mode: "replace"` replaces the whole stack.
- `x_task` picks a task from `registry/router.yaml`, and `x_thinking_budget` caps the reasoning tokens.

System 2 applies the abliteration adapter (rank-1, refusal direction removed) by default. Use `system2+none` to
get the stock model. An invalid adapter stack fails before any backend call, with `400 invalid_adapter` and a
`reason` code. Every response carries `x_brain`: profile, role, backend model, adapters, latency and prompt
tokens. With SSE, `x_brain` arrives in a final chunk just before `[DONE]`.

## Python client

```python
from brain_client import Client

c = Client(["http://mac-mini.local:8790", "http://127.0.0.1:8790"], token="...")
r = c.decide({"state": "...", "questions": {"q": {"type": "noul", "instructions": "..."}}})
r = c.chat([{"role": "user", "content": "hi"}], thinking=False)
for chunk in c.chat(messages, task="translate", stream=True): ...
```

The client fails over to the next endpoint on connection errors and on 503, and stays on the working one for
5 minutes.

## Layout

```
src/brain/          gateway: registry validation, adapter stacks, llama-server supervisor, FastAPI app, web UI
src/brain_client/   Python client
registry/           models, adapters, profiles, router tasks, eval suites
deploy/cerebro/     16 GB Mac mini: launchd + GPU memory limit
deploy/ranger/      CUDA host: Docker
lab/                measurements behind the profiles
tools/              adapter extraction, eval builder
```

## License

Apache 2.0. The models are separate downloads under their own licenses. Clef-Flash, Qwen3.5-9B and the
abliterated checkpoint are each Apache 2.0.
