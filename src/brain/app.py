"""The gateway: OpenAI-style API in front of two resident llama-server backends."""

from __future__ import annotations

import base64
import binascii
import contextlib
import hmac
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import httpx
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import stacks
from .backends import BackendHTTP
from .config import Settings, load_settings
from .registry import CHAT, SYSTEMONE, RegistryError, ResolvedRole, load_registry, validate_profile
from .supervisor import BackendProcess, BackendUnavailable, Supervisor, lora_paths

log = logging.getLogger("brain.app")

ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    429: "rate_limit_error",
    501: "not_supported_error",
    502: "backend_error",
    503: "unavailable_error",
    504: "timeout_error",
}
CHAT_X = {"x_adapters", "x_adapters_mode", "x_task", "x_thinking_budget"}
SYSTEMONE_X = {"x_adapters"}
CHAT_REJECT = {"lora", "id_slot", "tools", "tool_choice", "functions", "function_call"}
DATA_URL = re.compile(r"^data:image/[\w.+-]+;base64,(.*)$", re.S)
# clef: "input (N tokens) is too large to process. increase the physical batch size (current batch size: M)"
TOO_LARGE = re.compile(r"\((\d+) tokens\) is too large to process.*?batch size: (\d+)", re.S)
UI = Path(__file__).parent / "ui" / "index.html"
SUITE = re.compile(r"^[\w.-]+$")


class APIError(Exception):
    def __init__(self, status: int, code: str, message: str, *, param: str | None = None,
                 headers: dict | None = None, **extra: Any):
        super().__init__(message)
        self.status, self.code, self.message, self.param = status, code, message, param
        self.headers, self.extra = headers, extra

    def response(self) -> JSONResponse:
        err = {"message": self.message, "type": ERROR_TYPES.get(self.status, "server_error"),
               "code": self.code, "param": self.param, **self.extra}
        return JSONResponse({"error": err}, self.status, headers=self.headers)


def unavailable(reason: str) -> APIError:
    return APIError(503, "brain_unavailable", f"backend unavailable: {reason}", reason=reason,
                    headers={"Retry-After": "5"})


def backend_error(r: httpx.Response) -> APIError:
    try:
        err = r.json().get("error") or {}
    except (ValueError, AttributeError):
        err = {}
    msg = (err.get("message") if isinstance(err, dict) else None) or r.text[:500] or f"HTTP {r.status_code}"
    if err.get("type") == "exceed_context_size_error":
        return APIError(400, "context_length_exceeded", msg, count=err.get("n_prompt_tokens"), limit=err.get("n_ctx"))
    if m := TOO_LARGE.search(msg):
        return APIError(400, "context_length_exceeded", msg, count=int(m[1]), limit=int(m[2]))
    if r.status_code in (400, 404, 413, 422):
        return APIError(400, "invalid_request", msg)
    if r.status_code == 501:
        return APIError(501, "capability_unavailable", msg)
    if r.status_code == 503:
        return unavailable("loading")
    return APIError(502, "backend_error", msg)


def find_images(obj: Any) -> list[Any]:
    """URLs of every OpenAI-style image_url part inside messages/state."""
    out: list[Any] = []
    if isinstance(obj, dict):
        if obj.get("type") == "image_url":
            iu = obj.get("image_url")
            out.append(iu.get("url") if isinstance(iu, dict) else iu)
        else:
            for v in obj.values():
                out += find_images(v)
    elif isinstance(obj, list):
        for v in obj:
            out += find_images(v)
    return out


def check_images(urls: list[Any]) -> None:
    """Only base64 image data URLs: llama-server would otherwise fetch remote URLs or read local files."""
    for u in urls:
        m = DATA_URL.match(u) if isinstance(u, str) else None
        if not m:
            raise APIError(400, "invalid_request", "images must be data:image/...;base64,... URLs", param="image_url")
        try:
            base64.b64decode(m[1], validate=True)
        except (binascii.Error, ValueError):
            raise APIError(400, "invalid_request", "malformed base64 in image data URL", param="image_url") from None


def check_questions(questions: Any) -> None:
    if not isinstance(questions, dict) or not questions:
        raise APIError(400, "invalid_request", '"questions" must be a non-empty object', param="questions")
    for qid, q in questions.items():
        if not isinstance(q, dict):
            raise APIError(400, "invalid_request", f"question {qid!r} must be an object", param="questions")
        crit, t = q.get("criteria"), q.get("type")
        if t == "choice":
            if not isinstance(crit, dict) or not 2 <= len(crit) <= 255:
                raise APIError(400, "invalid_request", f"question {qid!r}: choice needs 2-255 options",
                               param="questions")
        elif t == "score":
            if not isinstance(crit, list) or not 2 <= len(crit) <= 10:
                raise APIError(400, "invalid_request", f"question {qid!r}: score needs 2-10 levels",
                               param="questions")
        elif t != "noul":
            raise APIError(400, "invalid_request", f"question {qid!r}: type must be choice, score or noul",
                           param="questions")


async def json_body(request: Request) -> dict:
    try:
        body = await request.json()
    except ValueError:
        raise APIError(400, "invalid_request", "request body must be JSON") from None
    if not isinstance(body, dict):
        raise APIError(400, "invalid_request", "request body must be a JSON object")
    return body


def reject_x(body: dict, allowed: set[str]) -> None:
    for k in body:
        if k.startswith("x_") and k not in allowed:
            raise APIError(400, "unsupported_parameter", f"unknown extension field {k!r}", param=k)


def ms(t0: float) -> int:
    return round((time.perf_counter() - t0) * 1000)


class Brain:
    """Gateway state: registry, active profile, supervisor, backend clients, LoRA id maps."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.reg = load_registry(settings.registry)
        self.profile = self.reg.profile(settings.profile)
        report = validate_profile(self.reg, self.profile, settings.models_dir)
        if report.errors:
            raise RegistryError(report.errors)
        self.warnings = report.warnings
        self.loaded = {b: set(self.reg.configured_adapters(self.profile, b)) for b in self.profile.backends}
        self.lora_ids: dict[str, dict[int, str | None]] = {b: {} for b in self.profile.backends}
        self.supervisor = Supervisor(self.reg, self.profile, settings, on_ready=self.on_backend_ready)
        self.http = {n: BackendHTTP(p.url) for n, p in self.supervisor.backends.items()}

    async def on_backend_ready(self, proc: BackendProcess) -> None:
        """Verify llama-server's LoRA path -> id map against the profile's preload list."""
        be = self.profile.backends[proc.name]
        if not be.loras:
            return
        listed = await self.http[proc.name].lora_adapters()
        by_path = {os.path.normpath(a["path"]): a["id"] for a in listed}
        id_map: dict[int, str | None] = {a["id"]: None for a in listed}
        verified = set()
        for name, path in lora_paths(self.reg, be, self.settings.models_dir).items():
            if (i := by_path.get(os.path.normpath(path))) is None:
                msg = f"backend {proc.name}: adapter {name} ({path}) missing from /lora-adapters; marked not loaded"
                log.error(msg)
                if msg not in self.warnings:
                    self.warnings.append(msg)
            else:
                id_map[i] = name
                verified.add(name)
        self.lora_ids[proc.name] = id_map
        decision = {n for n in self.loaded[proc.name] if self.reg.adapters[n].kind == "decision"}
        self.loaded[proc.name] = verified | decision
        log.info("backend %s lora ids: %s", proc.name, id_map)

    def ctx(self) -> stacks.StackContext:
        return stacks.StackContext.from_profile(self.reg, self.profile, self.loaded)

    def role(self, name: str, endpoint: str) -> ResolvedRole:
        if name not in self.profile.roles:
            raise APIError(404, "model_not_found", f"unknown model/role {name!r}", param="model")
        role = self.profile.role(name)
        if role.endpoint != endpoint:
            other = "/v1/systemone" if role.endpoint == SYSTEMONE else "/v1/chat/completions"
            raise APIError(400, "invalid_request", f"role {name!r} is served by {other}", param="model")
        return role

    async def acquire(self, backend: str) -> float:
        """Hold `backend` for one request (swap profiles may switch models first); pair with
        supervisor.release(). Returns the seconds spent switching."""
        try:
            return await self.supervisor.acquire(backend)
        except BackendUnavailable as e:
            raise unavailable(e.reason) from None

    def x_brain(self, role: ResolvedRole, stack: stacks.Stack, route: dict | None) -> dict:
        return {
            "profile": self.profile.id,
            "phase": self.profile.phase,
            "role": role.name,
            "backend": role.backend,
            "backend_model": self.profile.backends[role.backend].model,
            "adapters": stack.adapters,
            "stack_evaluated": stack.evaluated,
            "route": route,
        }

    def resolve(self, role: ResolvedRole, adapters, replace: bool, task=None) -> stacks.Stack:
        if adapters is not None and not (
            isinstance(adapters, dict) and all(isinstance(k, str) for k in adapters)
        ):
            raise APIError(400, "invalid_request", "x_adapters must be an object {name: scale}", param="x_adapters")
        try:
            return stacks.resolve(self.ctx(), role, adapters, replace, task)
        except stacks.AdapterError as e:
            raise APIError(400, "invalid_adapter", str(e), param="x_adapters", reason=e.reason,
                           adapter=e.adapter) from None

    # ------------------------------------------------------------ /v1/systemone

    async def systemone(self, body: dict) -> JSONResponse:
        t0 = time.perf_counter()
        reject_x(body, SYSTEMONE_X)
        role_name, q_adapters, q_replace = self.parse_model(body.pop("model", None) or "system1")
        if (q_adapters is not None or q_replace) and "x_adapters" in body:
            raise APIError(400, "ambiguous_adapters", "use either a qualified model name or x_adapters")
        role = self.role(role_name, SYSTEMONE)
        stack = self.resolve(role, body.pop("x_adapters", None) if q_adapters is None else q_adapters, q_replace)
        if stack.adapters != role.default_adapters:  # Phase A: S1 runs exactly its bound decision adapter
            raise APIError(400, "invalid_adapter", f"{role.name} only runs {role.default_adapters}",
                           param="x_adapters", reason="scale_range", adapter=next(iter(role.default_adapters), None))
        check_questions(body.get("questions"))
        images = [u for k in ("images", "files") for u in (body.get(k) or [])] + find_images(body.get("state"))
        if images:
            if not role.capabilities.images:
                raise APIError(501, "capability_unavailable", f"role {role.name} has no image input")
            check_images(images)
        swap_s = await self.acquire(role.backend)
        try:
            r = await self.http[role.backend].post("/v1/systemone", body)
        except httpx.TransportError:
            raise unavailable("loading") from None
        finally:
            await self.supervisor.release(role.backend)
        if r.status_code != 200:
            raise backend_error(r)
        data = r.json()
        xb = self.x_brain(role, stack, None)
        xb.update(latency_ms=ms(t0), prompt_tokens=(data.get("usage") or {}).get("input_tokens"))
        if swap_s:
            xb["swap_ms"] = int(swap_s * 1000)
        data["x_brain"] = xb
        return JSONResponse(data)

    def parse_model(self, model: Any) -> tuple[str, dict | None, bool]:
        if not isinstance(model, str):
            raise APIError(400, "invalid_request", "model must be a string", param="model")
        try:
            return stacks.parse_model(model)
        except ValueError as e:
            raise APIError(400, "invalid_request", str(e), param="model") from None

    # ------------------------------------------------------------ /v1/chat/completions

    async def chat(self, body: dict):
        t0 = time.perf_counter()
        reject_x(body, CHAT_X)
        for k in CHAT_REJECT:
            if k in body:
                raise APIError(400, "unsupported_parameter", f"{k!r} is not supported", param=k)
        if body.get("n") not in (None, 1):
            raise APIError(400, "unsupported_parameter", "n > 1 is not supported", param="n")
        if not isinstance(body.get("messages"), list) or not body["messages"]:
            raise APIError(400, "invalid_request", "messages must be a non-empty array", param="messages")

        task_name, route = body.get("x_task"), None
        model = body.get("model")
        if model in (None, "", "auto"):  # no explicit role: the router decides
            task_name = task_name or "default"
        task = None
        if task_name is not None:
            task = self.reg.router.tasks.get(task_name) if isinstance(task_name, str) else None
            if task is None:
                raise APIError(400, "invalid_request", f"unknown task {task_name!r}", param="x_task")
            route = {"task": task_name, "source": "x_task" if body.get("x_task") else "default"}
        # explicit request fields win: an explicit model picks the role even with x_task
        if model in (None, "", "auto"):
            role_name, q_adapters, q_replace = task.role, None, False
        else:
            role_name, q_adapters, q_replace = self.parse_model(model)
        qualified = q_adapters is not None or q_replace
        if qualified and "x_adapters" in body:
            raise APIError(400, "ambiguous_adapters", "use either a qualified model name or x_adapters")
        role = self.role(role_name, CHAT)
        mode = body.get("x_adapters_mode", "merge")
        if mode not in ("merge", "replace"):
            raise APIError(400, "invalid_request", "x_adapters_mode must be merge or replace", param="x_adapters_mode")
        stack = self.resolve(role, q_adapters if qualified else body.get("x_adapters"),
                             q_replace or mode == "replace", task)

        caps = role.capabilities
        if images := find_images(body["messages"]):
            if not caps.vision:
                raise APIError(501, "capability_unavailable", f"role {role.name} has no vision")
            check_images(images)
        rf = body.get("response_format")
        if isinstance(rf, dict) and rf.get("type") in ("json_schema", "json_object") and not caps.json_schema:
            raise APIError(501, "capability_unavailable", f"role {role.name} has no structured output")
        tb = body.get("x_thinking_budget")
        if tb is not None and (isinstance(tb, bool) or not isinstance(tb, int) or tb < -1):
            raise APIError(400, "invalid_request", "x_thinking_budget must be an integer >= -1",
                           param="x_thinking_budget")
        out = {k: v for k, v in body.items() if not k.startswith("x_")}
        d = role.defaults.model_dump(exclude_none=True)
        if task:
            d.update(task.defaults.model_dump(exclude_none=True))
        thinking, budget = d.pop("thinking", None), d.pop("reasoning_budget", None)
        for k, v in d.items():
            out.setdefault(k, v)
        kwargs = out.get("chat_template_kwargs") or {}
        if thinking is not None and isinstance(kwargs, dict) and "enable_thinking" not in kwargs:
            out["chat_template_kwargs"] = {**kwargs, "enable_thinking": thinking}
        if tb is not None:
            out["reasoning_budget_tokens"] = tb
        elif budget is not None and budget >= 0:
            out.setdefault("reasoning_budget_tokens", budget)
        out["model"] = self.profile.backends[role.backend].model
        out["lora"] = stacks.lora_payload(self.lora_ids[role.backend], stack.adapters)
        xb = self.x_brain(role, stack, route)

        backend = role.backend
        swap_s = await self.acquire(backend)
        if swap_s:
            xb["swap_ms"] = int(swap_s * 1000)
        if out.get("stream"):
            return await self._stream(backend, out, xb, t0)  # releases when the stream ends
        try:
            r = await self.http[backend].post("/v1/chat/completions", out)
        except httpx.TransportError:
            raise unavailable("loading") from None
        finally:
            await self.supervisor.release(backend)
        if r.status_code != 200:
            raise backend_error(r)
        data = r.json()
        xb.update(latency_ms=ms(t0), prompt_tokens=(data.get("usage") or {}).get("prompt_tokens"))
        data["x_brain"] = xb
        return JSONResponse(data)

    async def _stream(self, backend: str, out: dict, xb: dict, t0: float) -> StreamingResponse:
        stack = contextlib.AsyncExitStack()
        stack.push_async_callback(self.supervisor.release, backend)  # runs last, exactly once
        try:
            resp = await stack.enter_async_context(self.http[backend].stream("/v1/chat/completions", out))
            if resp.status_code != 200:
                await resp.aread()
                raise backend_error(resp)
        except httpx.TransportError:
            await stack.aclose()
            raise unavailable("loading") from None
        except BaseException:
            await stack.aclose()
            raise

        async def events():
            last: dict = {}
            try:
                async for line in resp.aiter_lines():
                    payload = line[5:].strip() if line.startswith("data:") else None
                    if payload == "[DONE]":
                        break
                    if payload:
                        with contextlib.suppress(ValueError):
                            chunk = json.loads(payload)
                            last = {k: chunk[k] for k in ("id", "created", "model") if k in chunk} or last
                            if usage := chunk.get("usage"):
                                xb["prompt_tokens"] = usage.get("prompt_tokens")
                    yield line + "\n"
            except httpx.TransportError as e:
                err = {"error": {"message": f"backend stream failed: {e}", "type": "backend_error",
                                 "code": "backend_error"}}
                yield f"data: {json.dumps(err)}\n\n"
                return
            finally:  # also runs when the client disconnects: closing upstream cancels the generation
                with anyio.CancelScope(shield=True):
                    await stack.aclose()
            xb["latency_ms"] = ms(t0)
            final = {**last, "object": "chat.completion.chunk", "choices": [], "x_brain": xb}
            yield f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ------------------------------------------------------------ discovery

    def health(self) -> JSONResponse:
        status = self.supervisor.status
        body = {
            "status": status,
            "kuma": "BRAIN_OK" if status == "ok" else f"BRAIN_DOWN:{status}",
            "profile": self.profile.id,
            "phase": self.profile.phase,
            "backends": {n: p.info() for n, p in self.supervisor.backends.items()},
            "warnings": self.warnings,
        }
        if sw := self.supervisor.swapper:
            body["residency"] = {"mode": "swap", "active": sw.active, "switching": sw.switching, "switches": sw.switches}
        return JSONResponse(body, 200 if status == "ok" else 503)

    def models(self) -> dict:
        data = []
        for name in self.profile.roles:
            role = self.profile.role(name)
            be = self.profile.backends[role.backend]
            m = self.reg.models[be.model]
            data.append({
                "id": name, "object": "model", "created": 0, "owned_by": "brain",
                "endpoint": role.endpoint, "backend": role.backend,
                "state": self.supervisor.backends[role.backend].state,
                "capabilities": role.capabilities.model_dump(),
                "defaults": role.defaults.model_dump(exclude_none=True),
                "default_adapters": role.default_adapters,
                "model": m.model_dump(include={"id", "arch", "family", "quant", "file", "sha256", "source"}),
                "mmproj": be.mmproj and self.reg.models[be.mmproj].model_dump(include={"id", "file", "sha256"}),
            })
        return {"object": "list", "profile": self.profile.id, "phase": self.profile.phase, "data": data}

    def adapters(self) -> dict:
        data = []
        for a in self.reg.adapters.values():
            ids = {b: i for b, m in self.lora_ids.items() for i, n in m.items() if n == a.name}
            data.append({
                **a.model_dump(include={"name", "version", "kind", "description", "roles", "scale", "compat",
                                        "base", "provenance", "evals", "rank", "target_modules"}),
                "artifact": a.artifact.model_dump(),
                "gate": a.evals.gate,
                "stackable": a.kind == "generation",
                "loaded": {b: a.name in names for b, names in self.loaded.items()},
                "lora_ids": ids,
            })
        return {"object": "list", "data": data,
                "approved_stacks": [s.model_dump() for s in self.profile.approved_stacks]}

    def router(self) -> dict:
        tasks = {}
        for name, t in self.reg.router.tasks.items():
            entry = t.model_dump(exclude_none=True)
            try:
                st = stacks.resolve(self.ctx(), self.profile.role(t.role), task=t)
                entry["resolved"] = {"adapters": st.adapters, "stack_evaluated": st.evaluated}
            except (stacks.AdapterError, KeyError) as e:
                entry["error"] = str(e)
            tasks[name] = entry
        return {"object": "router", "tasks": tasks}


def create_app(settings: Settings | None = None, start_backends: bool = True) -> FastAPI:
    settings = settings or load_settings()
    brain = Brain(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if start_backends:
            brain.supervisor.start()
        try:
            yield
        finally:
            await brain.supervisor.stop()
            for h in brain.http.values():
                await h.aclose()

    app = FastAPI(title="brain", version="0.1.0", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.brain = brain

    async def auth(request: Request) -> None:
        if not settings.token:
            return
        h = request.headers.get("authorization", "")
        scheme, _, tok = h.partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(tok.strip().encode(), settings.token.encode()):
            raise APIError(401, "unauthorized", "missing or invalid bearer token",
                           headers={"WWW-Authenticate": "Bearer"})

    api = APIRouter(dependencies=[Depends(auth)])

    @api.post("/v1/systemone")
    async def systemone(request: Request):
        return await brain.systemone(await json_body(request))

    @api.post("/v1/chat/completions")
    async def chat(request: Request):
        return await brain.chat(await json_body(request))

    @api.get("/v1/models")
    async def models():
        return brain.models()

    @api.get("/v1/adapters")
    async def adapters():
        return brain.adapters()

    @api.get("/v1/router")
    async def router():
        return brain.router()

    @app.get("/health")
    async def health():
        return brain.health()

    app.include_router(api)

    # The web UI is a plain client of the API above: one static page plus the eval suites
    # (registry/evals/*.json, built by tools/build_evals.py). The page sends the bearer token itself.
    evals_dir = Path(settings.registry) / "evals"

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse("/ui/")

    @app.get("/ui/", include_in_schema=False)
    async def ui():
        return FileResponse(UI, headers={"Cache-Control": "no-cache"})

    @app.get("/ui/evals")
    async def eval_suites():
        data = []
        for f in sorted(evals_dir.glob("*.json")):
            with contextlib.suppress(ValueError, OSError):
                suite = json.loads(f.read_text())
                data.append({k: suite.get(k) for k in ("name", "title", "description", "endpoint", "model", "gate")}
                            | {"n": len(suite.get("cases") or [])})
        return {"object": "list", "data": data}

    @app.get("/ui/evals/{name}")
    async def eval_suite(name: str):
        f = evals_dir / f"{name}.json"
        if not SUITE.match(name) or not f.is_file():
            raise APIError(404, "not_found", f"no eval suite {name!r}")
        return FileResponse(f, media_type="application/json")

    @app.exception_handler(APIError)
    async def _api_error(_: Request, e: APIError):
        return e.response()

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, e: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(e.status_code, "error")
        return APIError(e.status_code, code, str(e.detail)).response()

    return app
