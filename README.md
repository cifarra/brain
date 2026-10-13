# brain

The lab's local LLM and decision API: one gateway, two resident llama.cpp backends.

- **`system1`**: Cloudflare Clef-Flash (`/v1/systemone`), which returns calibrated probabilities for typed questions in one prefill.
- **`system2`** / **`system2-fast`**: Qwen3.5-9B with thinking, JSON output, vision and stackable **generation adapters** (`/v1/chat/completions`). Abliteration is the first adapter, and it is on by default.

Clients address roles and adapter names, never model files. What runs is declared in `registry/`.
The design is in `docs/specs/system12-api.md`. Rev 5 of that spec sets the scope implemented here.

## Layout

```
src/brain/          the gateway
  config.py         settings: env vars over an optional YAML file (BRAIN_CONFIG)
  registry.py       model/adapter/profile/router manifests (pydantic, unknown keys are errors) + validator
  stacks.py         adapter stacks: resolution, qualified model names, reason codes, the llama.cpp `lora` array
  supervisor.py     llama-server argv from a profile; start/stop/restart with backoff; crash loop -> degraded
  backends.py       async HTTP clients for the backends (SSE streaming, /lora-adapters, /health, /tokenize)
  app.py            FastAPI routes, auth, error envelope, x_brain metadata
  cli.py            brain serve | check | adapters list
src/brain_client/   thin synchronous SDK with failover
registry/           models/ adapters/ profiles/ router.yaml (paths are relative to the models dir)
tests/              unit tests + integration tests against a fake llama-server (no GPU)
```

## Run

```sh
uv sync
uv run pytest -q                                              # everything, no GPU needed
uv run brain check registry/profiles/ranger-12g.yaml --no-files   # validate + print llama-server argv
uv run brain check ranger-12g --sha256                        # on ranger: also verify file hashes (cached)
uv run brain adapters list
BRAIN_MODELS_DIR=~/srv/models uv run brain serve              # gateway on :8790
```

| Env | Default | |
|---|---|---|
| `BRAIN_PROFILE` | `ranger-12g` | profile id |
| `BRAIN_REGISTRY` | `./registry` | registry dir |
| `BRAIN_MODELS_DIR` | `/models` | root for manifest file paths (ranger: `~/srv/models` mounted at `/models`) |
| `BRAIN_LLAMA_SERVER` | `llama-server` | binary (CUDA image on ranger, `brew install llama.cpp` on a Mac) |
| `BRAIN_TOKEN` | unset | one optional bearer token; unset means no auth. `/health` is always open |
| `BRAIN_HOST` / `BRAIN_PORT` | `0.0.0.0` / `8790` | gateway listen address |
| `BRAIN_BACKEND_PORT_BASE` | `8791` | backends listen on 127.0.0.1 at base, base+1, ... in profile order |
| `BRAIN_CONFIG` | unset | YAML with the same keys plus `supervisor:` (backoff, crash loop, timeouts) and `backend_ports:` |

`brain serve` refuses to start if the profile does not validate. Both backends are always resident. `/health` reports `loading` until both are ready, and `degraded` after a crash loop (5 crashes in 10 minutes). After a crash loop, run `docker compose restart brain`.

## API

| Endpoint | |
|---|---|
| `POST /v1/systemone` | Clef decision: `{state, images?, questions}`. `x_adapters` may only name `clef`. An oversized prompt returns 400 `context_length_exceeded` (`count`, `limit`) |
| `POST /v1/chat/completions` | OpenAI chat, SSE streaming, `response_format`, `image_url` (base64 data URLs only), `chat_template_kwargs` |
| `GET /v1/models`, `/v1/adapters`, `/v1/router` | roles and concrete models, the adapter registry with `loaded` per backend, router tasks |
| `GET /health` | `{status: ok/loading/degraded, backends, warnings}`. Returns 200 with the literal `BRAIN_OK` when ok, otherwise 503 with `BRAIN_DOWN:<status>` |

Chat extensions:

- `x_adapters: {name: scale}`, which merges over the role default; scale 0 removes an adapter.
- `x_adapters_mode: "replace"`.
- Qualified model names `system2+abliteration@0.5` and `system2+none`. Sending a qualified name and `x_adapters` together is a 400 `ambiguous_adapters`.
- `x_task` (router), and `x_thinking_budget`, which maps to `reasoning_budget_tokens`.

Raw `lora`, `id_slot`, `tools`, `n>1` and unknown `x_*` fields are rejected.

The stack is resolved as role default ⊕ router task ⊕ request. It is validated before any backend call, and an invalid stack returns `400 invalid_adapter` with one of these `reason` codes: `unknown`, `not_loaded`, `role`, `base_mismatch`, `scale_range`, `incompatible`, `missing_requirement`, `exclusive_group`, `stack_too_large`, `gate`.

The backend always receives `lora: [{id, scale}]` for **every** adapter it has loaded, with an explicit 0 for inactive ones. The ids come from `GET /lora-adapters` and are checked against the preload paths at startup.

A stack is **evaluated** when every adapter's gate has passed and the stack is a single adapter or appears in the profile's `approved_stacks`. Unevaluated stacks are served and flagged with `x_brain.stack_evaluated: false`.

Every JSON response carries `x_brain`: `profile`, `phase`, `role`, `backend`, `backend_model`, `adapters`, `stack_evaluated`, `route`, `latency_ms` and `prompt_tokens`. For SSE, `x_brain` comes in an extra final chunk (`choices: []`) just before `data: [DONE]`.

## SDK

```python
from brain_client import Client

c = Client(["https://brain.hx", "http://127.0.0.1:8790"], token="...")
r = c.decide({"state": "...", "questions": {"q": {"type": "noul", "instructions": "..."}}})
r = c.chat([{"role": "user", "content": "hi"}], adapters={"abliteration": 0.5}, thinking=False)
print(r.served_by, r.adapters, r["choices"][0]["message"]["content"])
for chunk in c.chat(messages, task="translate", stream=True): ...
```

The SDK fails over to the next endpoint on connection errors and on 503, never on any other 4xx or 5xx. After a failover it sticks to the working endpoint for 5 minutes.
