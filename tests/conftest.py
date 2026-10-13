from __future__ import annotations

import shutil
import socket
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
import uvicorn
import yaml

from brain.config import Settings, SupervisorSettings
from brain.registry import load_registry

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "tests" / "fixtures" / "registry"  # frozen copy: the live registry/ evolves with real measurements
FAKE = Path(__file__).with_name("fake_llama_server.py")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def touch_model_files(reg_dir: Path, models_dir: Path) -> None:
    reg = load_registry(reg_dir)
    rels = [m.file for m in reg.models.values()] + [a.artifact.file for a in reg.adapters.values() if a.artifact.file]
    for rel in rels:
        (models_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (models_dir / rel).touch()


def fake_binary(tmp: Path, env: dict[str, str] | None = None) -> str:
    """An executable that behaves like `llama-server` (delegates to the fake)."""
    exports = "".join(f"export {k}={v}\n" for k, v in (env or {}).items())
    script = tmp / f"llama-server-{len(list(tmp.glob('llama-server-*')))}"
    script.write_text(f"#!/bin/sh\n{exports}exec {sys.executable} {FAKE} \"$@\"\n")
    script.chmod(0o755)
    return str(script)


def copy_registry(tmp: Path, adapters: dict[str, dict] | None = None, profile_patch=None,
                  profile: str = "ranger-dev") -> Path:
    """Copy the real registry, add adapter manifests, and optionally edit a profile dict in place."""
    dst = tmp / "registry"
    shutil.copytree(REGISTRY, dst)
    for name, manifest in (adapters or {}).items():
        (dst / "adapters" / f"{name}.yaml").write_text(yaml.safe_dump(manifest))
    if profile_patch:
        pf = dst / "profiles" / f"{profile}.yaml"
        data = yaml.safe_load(pf.read_text())
        profile_patch(data)
        pf.write_text(yaml.safe_dump(data))
    return dst


def gen_adapter(name: str, **kw) -> dict:
    """A generation adapter manifest compatible with the Qwen base, gate passed."""
    m = {
        "name": name, "version": 1, "kind": "generation",
        "base": {"model": "Qwen/Qwen3.5-9B", "hf_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
                 "quant_compat": ["Q4_K_M", "IQ4_XS", "Q8_0"]},
        "artifact": {"format": "gguf-lora", "arch": "qwen35", "file": f"adapters/{name}.gguf"},
        "rank": 8, "scale": {"default": 1.0, "min": 0.0, "max": 2.0},
        "roles": ["system2", "system2-fast"], "evals": {"gate": "passed"},
    }
    for k, v in kw.items():
        if isinstance(v, dict) and isinstance(m.get(k), dict):
            m[k] = {**m[k], **v}
        else:
            m[k] = v
    return m


def make_settings(tmp: Path, registry: Path = REGISTRY, binary: str | None = None, **kw) -> Settings:
    models = tmp / "models"
    models.mkdir(exist_ok=True)
    touch_model_files(registry, models)
    kw.setdefault("supervisor", SupervisorSettings(restart_backoff_s=[0.1, 0.2], crash_loop_max=3,
                                                   crash_loop_window_s=60, ready_timeout_s=30,
                                                   health_poll_s=0.05, stop_timeout_s=5))
    return Settings(
        profile="ranger-dev", registry=registry, models_dir=models,
        llama_server=binary or fake_binary(tmp), port=free_port(),
        backend_ports={"s1": free_port(), "s2": free_port()}, cache_dir=tmp / "cache", **kw,
    )


@contextmanager
def run_gateway(settings: Settings, wait_ok: bool = True, timeout: float = 30):
    """Run the gateway (and through it, the backends) with uvicorn in a thread."""
    from brain.app import create_app

    app = create_app(settings)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=settings.port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    url = f"http://127.0.0.1:{settings.port}"
    deadline = time.monotonic() + timeout
    while not server.started:
        assert time.monotonic() < deadline, "gateway did not start"
        time.sleep(0.02)
    if wait_ok:
        while True:
            r = httpx.get(url + "/health", headers={})
            if r.status_code == 200:
                break
            assert time.monotonic() < deadline, f"gateway not healthy: {r.text}"
            time.sleep(0.05)
    try:
        yield url, app
    finally:
        server.should_exit = True
        t.join(timeout=20)


@pytest.fixture(scope="module")
def module_tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("brain")
