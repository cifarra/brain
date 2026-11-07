"""residency: swap. One backend in memory at a time, switched on demand."""

import threading
import time

import httpx
import pytest
from conftest import copy_registry, make_settings, run_gateway

EXAMPLE = {"state": "a customer message", "questions": {
    "route": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": None, "shipping": None}}}}
MSG = [{"role": "user", "content": "hi"}]


def states(c):
    return {n: b["state"] for n, b in c.get("/health").json()["backends"].items()}


@pytest.fixture
def sw(tmp_path):
    reg = copy_registry(tmp_path, profile_patch=lambda d: d.update(residency="swap", initial_backend="s2"))
    settings = make_settings(tmp_path, registry=reg)
    with run_gateway(settings) as (url, _):
        with httpx.Client(base_url=url, timeout=60) as c:
            yield c


def test_boot_loads_only_the_initial_backend(sw):
    assert states(sw) == {"s1": "stopped", "s2": "ready"}
    h = sw.get("/health").json()
    assert h["kuma"] == "BRAIN_OK"
    assert h["residency"] == {"mode": "swap", "active": "s2", "switching": False, "switches": 0}


def test_switches_on_demand_and_reports_swap_ms(sw):
    r = sw.post("/v1/systemone", json=EXAMPLE)
    assert r.status_code == 200, r.text
    assert r.json()["x_brain"]["swap_ms"] > 0
    assert states(sw) == {"s1": "ready", "s2": "stopped"}
    r = sw.post("/v1/systemone", json=EXAMPLE)  # same backend again: no switch
    assert "swap_ms" not in r.json()["x_brain"]
    r = sw.post("/v1/chat/completions", json={"model": "system2", "messages": MSG})
    assert r.status_code == 200 and r.json()["x_brain"]["swap_ms"] > 0
    assert states(sw) == {"s1": "stopped", "s2": "ready"}
    # the LoRA id map is re-verified after every start, so adapters still apply
    assert r.json()["x_brain"]["adapters"] == {"abliteration": 1.0}


def test_switch_waits_for_inflight_stream_then_fifo(sw):
    """A systemone request waits for a running stream; a chat request that arrives after it
    queues behind it instead of jumping ahead and starving it."""
    order, out = [], {}

    def stream():
        body = {"model": "system2", "stream": True, "messages": [{"role": "user", "content": "SLOW"}]}
        with sw.stream("POST", "/v1/chat/completions", json=body) as r:
            text = "".join(r.iter_text())
        out["stream"] = text
        order.append("stream")

    def decide():
        out["s1"] = sw.post("/v1/systemone", json=EXAMPLE)
        order.append("s1")

    def chat():
        out["chat"] = sw.post("/v1/chat/completions", json={"model": "system2-fast", "messages": MSG})
        order.append("chat")

    threads = [threading.Thread(target=f) for f in (stream, decide, chat)]
    threads[0].start(); time.sleep(0.5)
    threads[1].start(); time.sleep(0.3)
    threads[2].start()
    for t in threads:
        t.join(60)
    assert order == ["stream", "s1", "chat"]
    assert "tok199" in out["stream"] and "[DONE]" in out["stream"]  # not cut off by the switch
    assert out["s1"].status_code == 200 and out["s1"].json()["x_brain"]["swap_ms"] > 0
    assert out["chat"].status_code == 200 and out["chat"].json()["x_brain"]["swap_ms"] > 0


def test_invalid_request_does_not_trigger_a_switch(sw):
    r = sw.post("/v1/systemone", json={"state": "x", "questions": {}})
    assert r.status_code == 400
    assert states(sw) == {"s1": "stopped", "s2": "ready"}
