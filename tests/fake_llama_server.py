"""A fake llama-server for tests. Accepts llama-server-style flags (unknown flags are an
error, like the real one), serves /health, /v1/systemone (clef models), /v1/chat/completions
(with SSE), /lora-adapters and /tokenize, and records every request at GET /_fake/requests.

Model file names containing "clef" select clef mode (only /v1/systemone, ignores `lora`).
Env: FAKE_STARTUP_DELAY (seconds of 503 "Loading model"), FAKE_EXIT_CODE (exit at once).
Prompt triggers: "OVERSIZE" (clef ubatch error, a 500 like the real server), "OVERCTX"
(exceed_context_size_error), "SLOW" (a long, slow stream, for cancellation tests).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

p = argparse.ArgumentParser()
p.add_argument("-m", required=True)
p.add_argument("--mmproj")
p.add_argument("--lora", action="append", default=[])
p.add_argument("--lora-init-without-apply", action="store_true")
for flag in ("-c", "-b", "-ub", "-np", "-fa", "-ngl", "--ctx-checkpoints", "-ctk", "-ctv",
             "--reasoning-format", "--reasoning-budget", "--cache-ram"):
    p.add_argument(flag)
p.add_argument("--jinja", action="store_true")
p.add_argument("--no-jinja", action="store_true")
p.add_argument("-ot", action="append", default=[])
p.add_argument("--no-mmproj-offload", action="store_true")
p.add_argument("--mmproj-offload", action="store_true")
p.add_argument("--host", default="127.0.0.1")
p.add_argument("--port", type=int, required=True)
p.add_argument("--no-webui", action="store_true")
p.add_argument("--metrics", action="store_true")
args = p.parse_args()

if code := os.environ.get("FAKE_EXIT_CODE"):
    print("fake llama-server: failing on purpose", flush=True)
    sys.exit(int(code))

CLEF = "clef" in os.path.basename(args.m).lower()
READY_AT = time.monotonic() + float(os.environ.get("FAKE_STARTUP_DELAY", "0"))
REQUESTS: list[dict] = []
app = FastAPI()


def err(status: int, message: str, type_: str, **extra) -> JSONResponse:
    return JSONResponse({"error": {"code": status, "message": message, "type": type_, **extra}}, status)


@app.get("/health")
async def health():
    if time.monotonic() < READY_AT:
        return err(503, "Loading model", "unavailable_error")
    return {"status": "ok"}


@app.get("/lora-adapters")
async def lora_adapters():
    scale = 0.0 if args.lora_init_without_apply else 1.0
    return [{"id": i, "path": path, "scale": scale} for i, path in enumerate(args.lora)]


@app.post("/tokenize")
async def tokenize(request: Request):
    body = await request.json()
    return {"tokens": list(range(len(str(body.get("content", "")).split())))}


@app.get("/_fake/requests")
async def requests_():
    return REQUESTS


@app.delete("/_fake/requests")
async def clear():
    REQUESTS.clear()
    return {"ok": True}


@app.get("/_fake/argv")
async def argv():
    return sys.argv[1:]


@app.post("/v1/systemone")
async def systemone(request: Request):
    body = await request.json()
    REQUESTS.append({"path": "/v1/systemone", "body": body})
    if not CLEF:
        return err(501, "This server does not support decisions", "not_supported_error")
    state = json.dumps(body.get("state"))
    if "OVERSIZE" in state:
        return err(500, "input (5000 tokens) is too large to process. increase the physical batch size "
                        "(current batch size: 4096)", "server_error")
    answers = {}
    for qid, q in body["questions"].items():
        if q["type"] == "choice":
            opts = list(q["criteria"])
            probs = {o: 1 / len(opts) for o in opts}
            answers[qid] = {"type": "choice", "choice": opts[0], "probabilities": probs, "confidence": 0.0}
        elif q["type"] == "score":
            n = len(q["criteria"])
            answers[qid] = {"type": "score", "score": (n - 1) / 2, "probabilities": {str(i): 1 / n for i in range(n)},
                            "legend": {str(i): c for i, c in enumerate(q["criteria"])}, "confidence": 0.0}
        else:
            answers[qid] = {"type": "noul", "noul": 0.5}
    return {"model": "clef", "answers": answers, "usage": {"input_tokens": len(state) // 4, "output_tokens": 0}}


def chunk(delta: dict, finish: str | None = None) -> str:
    c = {"id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": 1, "model": "fake",
         "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return f"data: {json.dumps(c)}\n\n"


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    rec = {"path": "/v1/chat/completions", "body": body, "completed": False, "cancelled": False}
    REQUESTS.append(rec)
    if CLEF:
        return err(501, "text generation is not available for clef", "not_supported_error")
    text = json.dumps(body.get("messages"))
    if "OVERCTX" in text:
        return err(400, "request (20000 tokens) exceeds the available context size (16384 tokens), try increasing it",
                   "exceed_context_size_error", n_prompt_tokens=20000, n_ctx=16384)
    thinking = (body.get("chat_template_kwargs") or {}).get("enable_thinking", True)
    slow = "SLOW" in text
    if not body.get("stream"):
        rec["completed"] = True
        msg = {"role": "assistant", "content": "hello"}
        if thinking:
            msg["reasoning_content"] = "thinking..."
        return {"id": "chatcmpl-fake", "object": "chat.completion", "created": 1, "model": "fake",
                "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8}}

    async def gen():
        try:
            if thinking:
                yield chunk({"reasoning_content": "thinking..."})
            for i in range(200 if slow else 3):
                if slow:
                    await asyncio.sleep(0.05)
                yield chunk({"content": f"tok{i} "})
            yield chunk({}, "stop")
            yield "data: [DONE]\n\n"
            rec["completed"] = True
        except asyncio.CancelledError:
            rec["cancelled"] = True
            raise
        finally:
            if not rec["completed"]:
                rec["cancelled"] = True

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
