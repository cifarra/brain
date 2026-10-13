"""Thin synchronous client for the brain gateway, with failover across endpoints.

    from brain_client import Client
    c = Client(["https://brain.hx", "http://127.0.0.1:8790"], token="...")
    r = c.chat([{"role": "user", "content": "hi"}], adapters={"abliteration": 0.5})
    r.served_by, r.x_brain["adapters"], r["choices"][0]["message"]["content"]

Failover happens on connection errors and 503 only (never on other 4xx/5xx, 429 or 504).
After a failover the working endpoint stays sticky for 5 minutes.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

import httpx

__all__ = ["Client", "Result", "ChatStream", "BrainError", "BrainUnavailable"]

FAILOVER_EXC = (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError)


class BrainError(Exception):
    """A non-failover error response (4xx, 429, 504, ...)."""

    def __init__(self, status: int, body: Any, served_by: str):
        err = body.get("error", {}) if isinstance(body, dict) else {}
        self.status, self.body, self.served_by = status, body, served_by
        self.code = err.get("code") if isinstance(err, dict) else None
        self.reason = err.get("reason") if isinstance(err, dict) else None
        msg = err.get("message") if isinstance(err, dict) else body
        super().__init__(f"{status} {self.code}: {msg} (from {served_by})")


class BrainUnavailable(Exception):
    """Every endpoint failed with a connection error or a 503."""

    def __init__(self, failures: list[tuple[str, str]]):
        self.failures = failures
        super().__init__("all brain endpoints unavailable: " + "; ".join(f"{e}: {why}" for e, why in failures))


class Result(dict):
    """A response body (dict) plus where it was served."""

    served_by: str

    @property
    def x_brain(self) -> dict:
        return self.get("x_brain") or {}

    @property
    def profile(self) -> str | None:
        return self.x_brain.get("profile")

    @property
    def adapters(self) -> dict | None:
        return self.x_brain.get("adapters")


class ChatStream:
    """Iterate to get parsed SSE chunks; `x_brain` is filled from the final chunk."""

    def __init__(self, response: httpx.Response, served_by: str):
        self.response, self.served_by, self.x_brain = response, served_by, {}

    def __iter__(self) -> Iterator[dict]:
        try:
            for line in self.response.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                if "x_brain" in chunk:
                    self.x_brain = chunk["x_brain"]
                yield chunk
        finally:
            self.close()

    def close(self) -> None:
        self.response.close()

    @property
    def profile(self) -> str | None:
        return self.x_brain.get("profile")

    @property
    def adapters(self) -> dict | None:
        return self.x_brain.get("adapters")


class Client:
    def __init__(
        self,
        endpoints: list[str] | str,
        token: str | None = None,
        *,
        timeout: float = 900.0,
        connect_timeout: float = 2.0,
        sticky_s: float = 300.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.endpoints = [endpoints] if isinstance(endpoints, str) else [e.rstrip("/") for e in endpoints]
        if not self.endpoints:
            raise ValueError("at least one endpoint is required")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.http = httpx.Client(headers=headers, transport=transport,
                                 timeout=httpx.Timeout(timeout, connect=connect_timeout))
        self.sticky_s = sticky_s
        self._sticky: tuple[str, float] | None = None  # (endpoint, until)

    def close(self) -> None:
        self.http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _order(self) -> list[str]:
        if self._sticky and self._sticky[1] > time.monotonic():
            first = self._sticky[0]
            return [first] + [e for e in self.endpoints if e != first]
        self._sticky = None
        return list(self.endpoints)

    def _send(self, method: str, path: str, body: dict | None = None, stream: bool = False):
        failures: list[tuple[str, str]] = []
        for ep in self._order():
            req = self.http.build_request(method, ep + path, json=body)
            try:
                resp = self.http.send(req, stream=stream)
            except FAILOVER_EXC as e:
                failures.append((ep, f"{type(e).__name__}: {e}"))
                continue
            if resp.status_code == 503:
                resp.read()
                failures.append((ep, f"503 {_err_reason(resp)}"))
                resp.close()
                continue
            if resp.status_code >= 400:
                if stream:
                    resp.read()
                    resp.close()
                raise BrainError(resp.status_code, _json(resp), ep)
            if ep != self.endpoints[0] and (self._sticky is None or self._sticky[0] != ep):
                self._sticky = (ep, time.monotonic() + self.sticky_s)
            return resp, ep
        raise BrainUnavailable(failures)

    def _call(self, method: str, path: str, body: dict | None = None) -> Result:
        resp, ep = self._send(method, path, body)
        r = Result(resp.json())
        r.served_by = ep
        return r

    # ---------------------------------------------------------------- API

    def decide(self, request: dict) -> Result:
        """POST /v1/systemone: {state, images?, questions: {id: {type, instructions, criteria}}}."""
        return self._call("POST", "/v1/systemone", request)

    def chat(
        self,
        messages: list[dict],
        adapters: dict[str, float] | None = None,
        task: str | None = None,
        thinking: bool | int | None = None,
        response_format: dict | None = None,
        stream: bool = False,
        model: str | None = None,
        **kw: Any,
    ) -> Result | ChatStream:
        """`adapters` -> x_adapters; `task` -> x_task (the router picks the role unless `model`
        is given); `thinking`: True/False toggles thinking, an int sets a thinking budget."""
        body: dict[str, Any] = {"messages": messages, **kw}
        if model or not task:
            body["model"] = model or "system2"
        if adapters is not None:
            body["x_adapters"] = adapters
        if task:
            body["x_task"] = task
        if thinking is not None:
            kwargs = dict(body.get("chat_template_kwargs") or {})
            kwargs["enable_thinking"] = thinking is not False  # an int budget implies thinking on
            body["chat_template_kwargs"] = kwargs
            if not isinstance(thinking, bool):
                body["x_thinking_budget"] = int(thinking)
        if response_format is not None:
            body["response_format"] = response_format
        if not stream:
            return self._call("POST", "/v1/chat/completions", body)
        body["stream"] = True
        resp, ep = self._send("POST", "/v1/chat/completions", body, stream=True)
        return ChatStream(resp, ep)

    def models(self) -> Result:
        return self._call("GET", "/v1/models")

    def adapters(self) -> Result:
        return self._call("GET", "/v1/adapters")

    def router(self) -> Result:
        return self._call("GET", "/v1/router")

    def health(self) -> dict[str, dict]:
        """/health of every endpoint (no failover): {endpoint: body or {"status": "unreachable"}}."""
        out = {}
        for ep in self.endpoints:
            try:
                out[ep] = _json(self.http.get(ep + "/health"))
            except httpx.HTTPError as e:
                out[ep] = {"status": "unreachable", "error": str(e)}
        return out


def _json(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        return resp.text


def _err_reason(resp: httpx.Response) -> str:
    body = _json(resp)
    err = body.get("error") if isinstance(body, dict) else None
    return (err or {}).get("reason") or (err or {}).get("code") or "unavailable"
