"""Async HTTP clients for the llama-server backends (loopback only)."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

import httpx

TIMEOUT = httpx.Timeout(connect=5, read=900, write=60, pool=60)


class BackendHTTP:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url, timeout=TIMEOUT)

    async def aclose(self) -> None:
        await self.client.aclose()

    async def health(self) -> bool:
        with contextlib.suppress(httpx.HTTPError):
            return (await self.client.get("/health", timeout=2)).status_code == 200
        return False

    async def lora_adapters(self) -> list[dict]:
        r = await self.client.get("/lora-adapters", timeout=10)
        r.raise_for_status()
        return r.json()

    async def tokenize(self, content: str) -> list[int]:
        r = await self.client.post("/tokenize", json={"content": content}, timeout=30)
        r.raise_for_status()
        return r.json()["tokens"]

    async def post(self, path: str, body: dict) -> httpx.Response:
        """/v1/systemone, or non-streaming /v1/chat/completions."""
        return await self.client.post(path, json=body)

    @contextlib.asynccontextmanager
    async def stream(self, path: str, body: dict) -> AsyncIterator[httpx.Response]:
        """Streaming /v1/chat/completions. Leaving the context closes the upstream
        connection, which makes llama-server cancel the generation."""
        req = self.client.build_request("POST", path, json=body)
        resp = await self.client.send(req, stream=True)
        try:
            yield resp
        finally:
            await resp.aclose()
