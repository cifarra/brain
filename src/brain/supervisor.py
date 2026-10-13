"""llama-server child processes: argv from a profile backend, start/stop, restart with
backoff, crash-loop -> degraded, readiness by polling /health. Both backends are resident."""

from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
import os
import signal
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx

from .config import Settings, SupervisorSettings
from .registry import Backend, Profile, Registry

log = logging.getLogger("brain.supervisor")

# profile arg -> llama-server flag (verified against the server README at 4413a19)
FLAGS = {
    "ctx": "-c",
    "batch": "-b",
    "ubatch": "-ub",
    "parallel": "-np",
    "flash_attn": "-fa",
    "ngl": "-ngl",
    "ctx_checkpoints": "--ctx-checkpoints",
    "cache_type_k": "-ctk",
    "cache_type_v": "-ctv",
    "reasoning_format": "--reasoning-format",
    "reasoning_budget": "--reasoning-budget",
}


def build_argv(
    reg: Registry, backend: Backend, *, binary: str, models_dir: Path, host: str, port: int
) -> list[str]:
    model = reg.models[backend.model]
    argv = [binary, "-m", str(models_dir / model.file)]
    if backend.mmproj:
        argv += ["--mmproj", str(models_dir / reg.models[backend.mmproj].file)]
    for name in backend.loras:
        argv += ["--lora", str(models_dir / reg.adapters[name].artifact.file)]
    if backend.loras:
        # --cache-ram 0: at b11580 the RAM prompt cache is keyed by tokens only, so a request could
        # resume a state computed under another LoRA scale (tools/extract_adapter/README.md).
        argv += ["--lora-init-without-apply", "--cache-ram", "0"]
    for key, value in backend.args.model_dump(exclude_none=True).items():
        if key == "jinja":
            argv.append("--jinja" if value else "--no-jinja")
        elif key == "mmproj_offload":
            argv.append("--mmproj-offload" if value else "--no-mmproj-offload")
        elif key == "override_tensor":
            for pattern in value:
                argv += ["-ot", pattern]
        elif key == "ngl":
            argv += ["-ngl", "999" if value == "all" else str(value)]
        elif key in FLAGS:
            argv += [FLAGS[key], str(value)]
        else:  # BackendArgs forbids unknown keys; this guards FLAGS drifting from the model
            raise ValueError(f"no llama-server flag mapping for {key!r}")
    return argv + ["--host", host, "--port", str(port), "--no-webui", "--metrics"]


def lora_paths(reg: Registry, backend: Backend, models_dir: Path) -> dict[str, str]:
    """adapter name -> the exact path string passed on the command line."""
    return {n: str(models_dir / reg.adapters[n].artifact.file) for n in backend.loras}


class BackendProcess:
    """One supervised llama-server. States: stopped, starting, ready, restarting, degraded."""

    def __init__(
        self,
        name: str,
        argv: list[str],
        url: str,
        cfg: SupervisorSettings,
        on_ready: Callable[[BackendProcess], Awaitable[None]] | None = None,
    ):
        self.name, self.argv, self.url, self.cfg, self.on_ready = name, argv, url, cfg, on_ready
        self.state = "stopped"
        self.proc: asyncio.subprocess.Process | None = None
        self.restarts = 0
        self.crashes: collections.deque[float] = collections.deque()
        self.last_error: str | None = None
        self.logs: collections.deque[str] = collections.deque(maxlen=200)
        self.ready_since: float | None = None
        self._task: asyncio.Task | None = None
        self._stopping = False
        self._log = logging.getLogger(f"brain.backend.{name}")

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    def info(self) -> dict:
        return {
            "state": self.state,
            "pid": self.proc.pid if self.proc and self.proc.returncode is None else None,
            "restarts": self.restarts,
            "last_error": self.last_error,
        }

    def start(self) -> None:
        self._stopping = False
        self._task = asyncio.create_task(self._run(), name=f"backend-{self.name}")

    async def stop(self) -> None:
        self._stopping = True
        await self._terminate()
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
        self.state = "stopped"

    async def _terminate(self) -> None:
        p = self.proc
        if p is None or p.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(p.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(p.wait(), self.cfg.stop_timeout_s)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(p.pid, signal.SIGKILL)
            await p.wait()

    async def _pump_logs(self, stream: asyncio.StreamReader) -> None:
        while line := await stream.readline():
            text = line.decode(errors="replace").rstrip()
            self.logs.append(text)
            self._log.info(text)

    async def _wait_ready(self) -> bool:
        deadline = time.monotonic() + self.cfg.ready_timeout_s
        async with httpx.AsyncClient(timeout=2) as client:
            while time.monotonic() < deadline:
                if self.proc.returncode is not None:
                    return False
                with contextlib.suppress(httpx.HTTPError):
                    if (await client.get(self.url + "/health")).status_code == 200:
                        return True
                await asyncio.sleep(self.cfg.health_poll_s)
        self.last_error = f"not healthy within {self.cfg.ready_timeout_s}s"
        return False

    async def _run(self) -> None:
        while not self._stopping:
            self.state = "starting"
            self.ready_since = None
            try:
                self.proc = await asyncio.create_subprocess_exec(
                    *self.argv,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    stdin=asyncio.subprocess.DEVNULL,
                    start_new_session=True,  # own process group, so stop() kills any children too
                )
            except OSError as e:
                self.last_error = f"spawn failed: {e}"
                self.proc = None
            if self.proc is not None:
                pump = asyncio.create_task(self._pump_logs(self.proc.stdout))
                if await self._wait_ready():
                    try:
                        if self.on_ready:
                            await self.on_ready(self)
                        self.state, self.ready_since, self.last_error = "ready", time.monotonic(), None
                        log.info("backend %s ready (pid %s)", self.name, self.proc.pid)
                    except Exception as e:  # e.g. the LoRA id map does not match
                        self.last_error = f"on_ready failed: {e}"
                        log.error("backend %s: %s", self.name, self.last_error)
                if self.state != "ready":
                    await self._terminate()
                rc = await self.proc.wait()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(pump, 5)
                if self._stopping:
                    break
                self.last_error = self.last_error or f"exited with code {rc}"
            if self._stopping:
                break
            now = time.monotonic()
            self.crashes.append(now)
            while self.crashes and now - self.crashes[0] > self.cfg.crash_loop_window_s:
                self.crashes.popleft()
            log.warning("backend %s down: %s (%d in window)", self.name, self.last_error, len(self.crashes))
            if len(self.crashes) >= self.cfg.crash_loop_max:
                self.state = "degraded"
                log.error("backend %s: crash loop, giving up until restart", self.name)
                return
            self.state = "restarting"
            backoff = self.cfg.restart_backoff_s
            await asyncio.sleep(backoff[min(len(self.crashes), len(backoff)) - 1])
            self.restarts += 1


class Supervisor:
    def __init__(
        self,
        reg: Registry,
        profile: Profile,
        settings: Settings,
        on_ready: Callable[[BackendProcess], Awaitable[None]] | None = None,
    ):
        self.backends: dict[str, BackendProcess] = {}
        for i, (name, be) in enumerate(profile.backends.items()):
            port = settings.port_for(name, i)
            argv = build_argv(
                reg, be, binary=settings.llama_server, models_dir=settings.models_dir,
                host=settings.backend_host, port=port,
            )
            url = f"http://{settings.backend_host}:{port}"
            self.backends[name] = BackendProcess(name, argv, url, settings.supervisor, on_ready)

    def start(self) -> None:
        for b in self.backends.values():
            b.start()

    async def stop(self) -> None:
        await asyncio.gather(*(b.stop() for b in self.backends.values()))

    @property
    def status(self) -> str:
        states = [b.state for b in self.backends.values()]
        if all(s == "ready" for s in states):
            return "ok"
        return "degraded" if "degraded" in states else "loading"
