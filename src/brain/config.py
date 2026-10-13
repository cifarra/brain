"""Gateway settings: env vars over an optional YAML file (BRAIN_CONFIG) over defaults."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SupervisorSettings(Strict):
    restart_backoff_s: list[float] = [1, 5, 30]
    crash_loop_max: int = 5  # crashes within the window => degraded, no more restarts
    crash_loop_window_s: float = 600
    ready_timeout_s: float = 600  # a backend that is not healthy by then counts as a crash
    health_poll_s: float = 0.25
    stop_timeout_s: float = 10


class Settings(Strict):
    profile: str = "ranger-12g"
    registry: Path = Path("registry")
    models_dir: Path = Path("/models")
    llama_server: str = "llama-server"  # binary on PATH, or an absolute path
    token: str | None = None  # single optional bearer token; None => no auth
    host: str = "0.0.0.0"
    port: int = 8790
    backend_host: str = "127.0.0.1"
    backend_port_base: int = 8791  # backends get base, base+1, ... in profile order
    backend_ports: dict[str, int] = Field(default_factory=dict)  # explicit per-backend override
    cache_dir: Path = Path.home() / ".cache" / "brain"
    log_level: str = "info"
    supervisor: SupervisorSettings = Field(default_factory=SupervisorSettings)

    def port_for(self, backend: str, index: int) -> int:
        return self.backend_ports.get(backend, self.backend_port_base + index)


ENV = {
    "BRAIN_PROFILE": "profile",
    "BRAIN_REGISTRY": "registry",
    "BRAIN_MODELS_DIR": "models_dir",
    "BRAIN_LLAMA_SERVER": "llama_server",
    "BRAIN_TOKEN": "token",
    "BRAIN_HOST": "host",
    "BRAIN_PORT": "port",
    "BRAIN_BACKEND_PORT_BASE": "backend_port_base",
    "BRAIN_CACHE_DIR": "cache_dir",
    "BRAIN_LOG_LEVEL": "log_level",
}


def load_settings(env: dict[str, str] | None = None, **overrides) -> Settings:
    env = dict(os.environ if env is None else env)
    data: dict = {}
    if path := env.get("BRAIN_CONFIG"):
        data = yaml.safe_load(Path(path).read_text()) or {}
    for var, key in ENV.items():
        if env.get(var):
            data[key] = env[var]
    data.update({k: v for k, v in overrides.items() if v is not None})
    return Settings.model_validate(data)
