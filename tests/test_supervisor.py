import asyncio
import os
import signal
import time
from pathlib import Path

import httpx
import pytest
from conftest import REGISTRY, fake_binary, make_settings

from brain.registry import Backend, load_registry
from brain.supervisor import Supervisor, build_argv


@pytest.mark.parametrize("profile", ["ranger-dev", "mac-dev"])
def test_argv_per_profile(profile):
    reg = load_registry(REGISTRY)
    p = reg.profile(profile)
    kw = dict(binary="llama-server", models_dir=Path("/models"), host="127.0.0.1")
    s1 = build_argv(reg, p.backends["s1"], port=8791, **kw)
    assert s1 == ["llama-server", "-m", "/models/gguf/clef-flash/Clef-Flash-Q4_K_M.gguf",
                  "-c", "4096", "-b", "4096", "-ub", "4096", "-np", "1", "-fa", "on", "-ngl", "999",
                  "--host", "127.0.0.1", "--port", "8791", "--no-webui", "--metrics"]
    s2 = build_argv(reg, p.backends["s2"], port=8792, **kw)
    assert s2 == ["llama-server", "-m", "/models/gguf/qwen3.5-9b/Qwen3.5-9B-Q4_K_M.gguf",
                  "--mmproj", "/models/gguf/qwen3.5-9b/mmproj-F16.gguf",
                  "--lora", "/models/adapters/abliteration-wangzhang-r1.gguf", "--lora-init-without-apply", "--cache-ram", "0",
                  "-c", "16384", "-b", "2048", "-ub", "512", "-np", "1", "-fa", "on", "-ngl", "999",
                  "--ctx-checkpoints", "4", "-ctk", "f16", "-ctv", "f16", "--reasoning-format", "deepseek",
                  "--jinja", "--host", "127.0.0.1", "--port", "8792", "--no-webui", "--metrics"]


def test_argv_misc_flags():
    reg = load_registry(REGISTRY)
    be = Backend(model="qwen3.5-9b-q8_0", args={"jinja": False, "ngl": 20, "reasoning_budget": 128})
    argv = build_argv(reg, be, binary="/opt/llama-server", models_dir=Path("/m"), host="127.0.0.1", port=1)
    assert "--no-jinja" in argv and argv[argv.index("-ngl") + 1] == "20"
    assert argv[argv.index("--reasoning-budget") + 1] == "128"
    assert "--lora-init-without-apply" not in argv and "--cache-ram" not in argv


def test_unknown_arg_rejected():
    with pytest.raises(ValueError, match="extra"):
        Backend(model="x", args={"ctx": 1, "rope_freq_base": 1})


async def wait_for(pred, timeout=20):
    deadline = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < deadline
        await asyncio.sleep(0.05)


def test_start_ready_restart_and_stop(tmp_path):
    settings = make_settings(tmp_path)
    reg = load_registry(REGISTRY)
    sup = Supervisor(reg, reg.profile("ranger-dev"), settings)

    async def run():
        sup.start()
        assert sup.status == "loading"
        await wait_for(lambda: sup.status == "ok")
        s2 = sup.backends["s2"]
        async with httpx.AsyncClient() as c:
            argv = (await c.get(s2.url + "/_fake/argv")).json()
        assert "--lora-init-without-apply" in argv
        pid = s2.proc.pid
        os.kill(pid, signal.SIGKILL)  # crash: restarted with backoff
        await wait_for(lambda: s2.state != "ready")
        assert sup.status in ("loading", "ok")
        await wait_for(lambda: s2.ready and s2.proc.pid != pid)
        assert s2.restarts == 1 and sup.backends["s1"].restarts == 0
        await sup.stop()
        assert all(b.state == "stopped" and b.proc.returncode is not None for b in sup.backends.values())

    asyncio.run(run())


def test_crash_loop_degrades(tmp_path):
    settings = make_settings(tmp_path, binary=fake_binary(tmp_path, {"FAKE_EXIT_CODE": "3"}))
    reg = load_registry(REGISTRY)
    sup = Supervisor(reg, reg.profile("ranger-dev"), settings)

    async def run():
        sup.start()
        await wait_for(lambda: sup.status == "degraded")
        b = sup.backends["s1"]
        await wait_for(lambda: b.state == "degraded")
        assert len(b.crashes) == settings.supervisor.crash_loop_max
        assert "code 3" in b.last_error and any("failing on purpose" in line for line in b.logs)
        await sup.stop()

    asyncio.run(run())


def test_missing_binary_degrades(tmp_path):
    settings = make_settings(tmp_path, binary=str(tmp_path / "does-not-exist"))
    reg = load_registry(REGISTRY)
    sup = Supervisor(reg, reg.profile("ranger-dev"), settings)

    async def run():
        sup.start()
        await wait_for(lambda: sup.status == "degraded")
        assert "spawn failed" in sup.backends["s2"].last_error
        await sup.stop()

    asyncio.run(run())
