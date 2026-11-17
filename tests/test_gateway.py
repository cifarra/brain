"""Integration: the real gateway + supervisor launching fake llama-servers via BRAIN_LLAMA_SERVER."""

import json
import os
import signal
import time

import httpx
import pytest
from conftest import copy_registry, fake_binary, gen_adapter, make_settings, run_gateway

TOKEN = "sekret"
H = {"Authorization": f"Bearer {TOKEN}"}
PNG = "data:image/png;base64,iVBORw0KGgo="
EXAMPLE = {
    "state": "Customer message: I was charged twice for my order last week and nobody has replied.",
    "questions": {
        "route": {"type": "choice", "instructions": "Which team should handle this?",
                  "criteria": {"billing": None, "shipping": None, "technical": None}},
        "angry": {"type": "noul", "instructions": "Is the customer angry?"},
        "urgency": {"type": "score", "instructions": "How urgent is this?",
                    "criteria": ["can wait", "this week", "today", "right now"]},
    },
}
MSG = [{"role": "user", "content": "hi"}]


@pytest.fixture(scope="module")
def gw(module_tmp):
    settings = make_settings(module_tmp, token=TOKEN)
    with run_gateway(settings) as (url, app):
        c = httpx.Client(base_url=url, headers=H, timeout=30)
        c.s1 = f"http://127.0.0.1:{settings.backend_ports['s1']}"
        c.s2 = f"http://127.0.0.1:{settings.backend_ports['s2']}"
        c.app = app
        yield c
        c.close()


def backend_requests(gw, which="s2"):
    return httpx.get(getattr(gw, which) + "/_fake/requests").json()


def last_body(gw, which="s2"):
    return backend_requests(gw, which)[-1]["body"]


def chat(gw, **body):
    body.setdefault("model", "system2")
    body.setdefault("messages", MSG)
    return gw.post("/v1/chat/completions", json=body)


def assert_error(r, status, code, reason=None):
    assert r.status_code == status, r.text
    err = r.json()["error"]
    assert err["code"] == code and {"message", "type"} <= set(err)
    if reason:
        assert err["reason"] == reason


# ------------------------------------------------------------ health and auth

def test_health_ok(gw):
    r = httpx.get(str(gw.base_url) + "/health")  # no token needed
    assert r.status_code == 200 and "BRAIN_OK" in r.text
    body = r.json()
    assert body["status"] == "ok" and body["profile"] == "ranger-dev"
    assert {b["state"] for b in body["backends"].values()} == {"ready"}


def test_auth(gw):
    base = str(gw.base_url)
    assert_error(httpx.get(base + "/v1/models"), 401, "unauthorized")
    assert_error(httpx.get(base + "/v1/models", headers={"Authorization": "Bearer nope"}), 401, "unauthorized")
    assert gw.get("/v1/models").status_code == 200


# ------------------------------------------------------------ systemone

def test_systemone_happy_path(gw):
    r = gw.post("/v1/systemone", json=EXAMPLE)
    assert r.status_code == 200, r.text
    data = r.json()
    assert len(data["answers"]) == 3
    assert abs(sum(data["answers"]["route"]["probabilities"].values()) - 1) < 1e-4
    xb = data["x_brain"]
    assert xb["profile"] == "ranger-dev" and xb["adapters"] == {"clef": 1.0} and xb["backend"] == "s1"
    assert xb["latency_ms"] >= 0 and xb["prompt_tokens"] > 0
    assert data["model"] == xb["backend_model"]  # the registry id, never the backend's file path
    sent = last_body(gw, "s1")
    assert set(sent) == {"state", "questions"}  # no model, no x_*


def test_systemone_rejects_generation_adapter(gw):
    n = len(backend_requests(gw, "s1"))
    r = gw.post("/v1/systemone", json={**EXAMPLE, "x_adapters": {"abliteration": 1}})
    assert_error(r, 400, "invalid_adapter", "role")
    assert gw.post("/v1/systemone", json={**EXAMPLE, "x_adapters": {"clef": 1}}).status_code == 200
    assert_error(gw.post("/v1/systemone", json={**EXAMPLE, "x_adapters": {"clef": 0}}), 400, "invalid_adapter")
    assert_error(gw.post("/v1/systemone", json={**EXAMPLE, "x_other": 1}), 400, "unsupported_parameter")
    assert len(backend_requests(gw, "s1")) == n + 1


def test_systemone_limits(gw):
    many = {"q": {"type": "choice", "instructions": "x", "criteria": {f"o{i}": None for i in range(256)}}}
    assert_error(gw.post("/v1/systemone", json={"state": "s", "questions": many}), 400, "invalid_request")
    eleven = {"q": {"type": "score", "instructions": "x", "criteria": [str(i) for i in range(11)]}}
    assert_error(gw.post("/v1/systemone", json={"state": "s", "questions": eleven}), 400, "invalid_request")
    assert_error(gw.post("/v1/systemone", json={**EXAMPLE, "images": [PNG]}), 501, "capability_unavailable")


def test_systemone_oversized_prompt(gw):
    r = gw.post("/v1/systemone", json={**EXAMPLE, "state": "OVERSIZE"})
    assert_error(r, 400, "context_length_exceeded")
    assert r.json()["error"]["count"] == 5000 and r.json()["error"]["limit"] == 4096


# ------------------------------------------------------------ chat: stacks -> lora

def test_chat_default_stack(gw):
    r = chat(gw)
    assert r.status_code == 200, r.text
    xb = r.json()["x_brain"]
    assert xb["adapters"] == {"abliteration": 1.0} and xb["stack_evaluated"] is False  # gate pending
    assert xb["backend_model"] == "qwen3.5-9b-q4km" and xb["prompt_tokens"] == 7
    sent = last_body(gw)
    assert sent["lora"] == [{"id": 0, "scale": 1.0}]
    assert sent["chat_template_kwargs"] == {"enable_thinking": True}
    assert not any(k.startswith("x_") for k in sent)


@pytest.mark.parametrize("body,adapters,lora", [
    ({"model": "system2+none"}, {}, 0.0),
    ({"model": "system2+abliteration@0.5"}, {"abliteration": 0.5}, 0.5),
    ({"x_adapters": {"abliteration": 0}}, {}, 0.0),
    ({"x_adapters": {"abliteration": 1.2}}, {"abliteration": 1.2}, 1.2),
    ({"x_adapters": {}, "x_adapters_mode": "replace"}, {}, 0.0),
])
def test_chat_stack_resolution(gw, body, adapters, lora):
    r = chat(gw, **body)
    assert r.status_code == 200, r.text
    assert r.json()["x_brain"]["adapters"] == adapters
    assert last_body(gw)["lora"] == [{"id": 0, "scale": lora}]  # explicit, even when 0


@pytest.mark.parametrize("body,reason", [
    ({"x_adapters": {"nope": 1}}, "unknown"),
    ({"x_adapters": {"clef": 1}}, "role"),
    ({"x_adapters": {"abliteration": 2.0}}, "scale_range"),
    ({"model": "system2+nope"}, "unknown"),
])
def test_chat_invalid_adapter_never_reaches_backend(gw, body, reason):
    n = len(backend_requests(gw))
    assert_error(chat(gw, **body), 400, "invalid_adapter", reason)
    assert len(backend_requests(gw)) == n


def test_chat_rejections(gw):
    assert_error(chat(gw, model="system2+none", x_adapters={"abliteration": 1}), 400, "ambiguous_adapters")
    for extra in ({"lora": [{"id": 0, "scale": 1}]}, {"tools": []}, {"n": 2}, {"x_foo": 1}):
        assert_error(chat(gw, **extra), 400, "unsupported_parameter")
    assert_error(chat(gw, model="system9"), 404, "model_not_found")
    assert_error(chat(gw, model="system1"), 400, "invalid_request")
    assert_error(chat(gw, x_task="nope"), 400, "invalid_request")
    assert_error(chat(gw, messages=[]), 400, "invalid_request")
    assert_error(gw.post("/v1/chat/completions", content=b"{nope"), 400, "invalid_request")


def test_chat_router(gw):
    r = chat(gw, model=None, x_task="translate")
    xb = r.json()["x_brain"]
    assert xb["role"] == "system2-fast" and xb["route"] == {"task": "translate", "source": "x_task"}
    sent = last_body(gw)
    assert sent["temperature"] == 0.3 and sent["chat_template_kwargs"] == {"enable_thinking": False}
    chat(gw, model=None, x_task="translate", temperature=0.9)
    assert last_body(gw)["temperature"] == 0.9  # explicit request fields win
    r = chat(gw, model=None)
    assert r.json()["x_brain"]["route"] == {"task": "default", "source": "default"}
    chat(gw, model=None, x_task="json")
    assert last_body(gw)["response_format"] == {"type": "json_object"}


def test_chat_thinking_controls(gw):
    chat(gw, x_thinking_budget=64)
    assert last_body(gw)["reasoning_budget_tokens"] == 64
    chat(gw, chat_template_kwargs={"enable_thinking": False})
    assert last_body(gw)["chat_template_kwargs"] == {"enable_thinking": False}
    chat(gw, model="system2-fast")
    assert last_body(gw)["chat_template_kwargs"] == {"enable_thinking": False}
    assert_error(chat(gw, x_thinking_budget="lots"), 400, "invalid_request")


def test_chat_passthrough_response_format_and_images(gw):
    rf = {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}}
    img = [{"role": "user", "content": [{"type": "text", "text": "what"},
                                        {"type": "image_url", "image_url": {"url": PNG}}]}]
    assert chat(gw, messages=img, response_format=rf).status_code == 200
    sent = last_body(gw)
    assert sent["response_format"] == rf and sent["messages"] == img
    bad = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,@@@"}}]}]
    assert_error(chat(gw, messages=bad), 400, "invalid_request")
    remote = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "file:///etc/passwd"}}]}]
    assert_error(chat(gw, messages=remote), 400, "invalid_request")


def test_chat_context_exceeded(gw):
    r = chat(gw, messages=[{"role": "user", "content": "OVERCTX"}])
    assert_error(r, 400, "context_length_exceeded")
    assert (r.json()["error"]["count"], r.json()["error"]["limit"]) == (20000, 16384)


# ------------------------------------------------------------ streaming

def sse(gw, **body):
    body.setdefault("model", "system2")
    body.setdefault("messages", MSG)
    with gw.stream("POST", "/v1/chat/completions", json={**body, "stream": True}) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        return [line[5:].strip() for line in r.iter_lines() if line.startswith("data:")]


def test_stream_thinking_then_content_then_done(gw):
    events = sse(gw)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    kinds = [k for c in chunks for ch in c["choices"] for k in ch["delta"]]
    assert kinds.index("reasoning_content") < kinds.index("content")
    final = chunks[-1]
    assert final["choices"] == [] and final["x_brain"]["adapters"] == {"abliteration": 1.0}
    assert final["x_brain"]["latency_ms"] >= 0 and final["id"] == "chatcmpl-fake"
    assert last_body(gw)["lora"] == [{"id": 0, "scale": 1.0}] and last_body(gw)["stream"] is True


def test_stream_fast_has_no_reasoning(gw):
    events = sse(gw, model="system2-fast+none")
    chunks = [json.loads(e) for e in events[:-1]]
    assert not any("reasoning_content" in ch["delta"] for c in chunks for ch in c["choices"])
    assert chunks[-1]["x_brain"]["adapters"] == {}


def test_stream_errors_before_first_byte(gw):
    body = {"model": "system2", "messages": [{"role": "user", "content": "OVERCTX"}], "stream": True}
    with gw.stream("POST", "/v1/chat/completions", json=body) as r:
        r.read()
        assert_error(r, 400, "context_length_exceeded")


def test_stream_client_disconnect_cancels_backend(gw):
    httpx.delete(gw.s2 + "/_fake/requests")
    with gw.stream("POST", "/v1/chat/completions",
                   json={"model": "system2", "messages": [{"role": "user", "content": "SLOW"}], "stream": True}) as r:
        for i, _ in enumerate(r.iter_lines()):
            if i > 4:
                break
    deadline = time.monotonic() + 10
    while True:
        rec = backend_requests(gw)[-1]
        if rec["cancelled"]:
            break
        assert time.monotonic() < deadline, rec
        time.sleep(0.1)
    assert rec["completed"] is False


# ------------------------------------------------------------ discovery

def test_models(gw):
    data = {m["id"]: m for m in gw.get("/v1/models").json()["data"]}
    assert set(data) == {"system1", "system2", "system2-fast"}
    assert data["system2"]["default_adapters"] == {"abliteration": 1.0}
    assert data["system2"]["model"]["source"]["hf_revision"].startswith("c202236")
    assert data["system2-fast"]["defaults"] == {"thinking": False, "reasoning_budget": -1}
    assert data["system1"]["endpoint"] == "systemone" and data["system2"]["mmproj"]["id"] == "qwen3.5-9b-mmproj-f16"


def test_adapters(gw):
    body = gw.get("/v1/adapters").json()
    ads = {a["name"]: a for a in body["data"]}
    a = ads["abliteration"]
    assert a["kind"] == "generation" and a["loaded"] == {"s1": False, "s2": True} and a["lora_ids"] == {"s2": 0}
    assert "clef" in a["compat"]["incompatible_with"] and a["gate"] == "pending" and a["stackable"]
    c = ads["clef"]
    assert c["kind"] == "decision" and c["loaded"] == {"s1": True, "s2": False} and not c["stackable"]
    assert body["approved_stacks"] == []


def test_router_listing(gw):
    tasks = gw.get("/v1/router").json()["tasks"]
    assert set(tasks) == {"default", "translate", "json"}
    assert tasks["translate"]["resolved"] == {"adapters": {"abliteration": 1.0}, "stack_evaluated": False}


def test_unknown_route_envelope(gw):
    assert_error(gw.get("/v1/nope"), 404, "not_found")


# ------------------------------------------------------------ crash isolation (keep last: restarts s2)

def test_s2_crash_isolated_and_recovers(gw):
    brain = gw.app.state.brain
    s2 = brain.supervisor.backends["s2"]
    os.kill(s2.proc.pid, signal.SIGKILL)
    deadline = time.monotonic() + 5
    while s2.ready:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    r = chat(gw)
    assert_error(r, 503, "brain_unavailable", "loading")
    assert r.headers["retry-after"] == "5"
    assert gw.post("/v1/systemone", json=EXAMPLE).status_code == 200  # system1 keeps serving
    assert httpx.get(str(gw.base_url) + "/health").status_code == 503
    deadline = time.monotonic() + 20
    while chat(gw).status_code != 200:
        assert time.monotonic() < deadline
        time.sleep(0.1)
    assert httpx.get(str(gw.base_url) + "/health").json()["backends"]["s2"]["restarts"] == 1


# ------------------------------------------------------------ other gateways

def test_loading_gateway(tmp_path):
    settings = make_settings(tmp_path, binary=fake_binary(tmp_path, {"FAKE_STARTUP_DELAY": "999"}))
    with run_gateway(settings, wait_ok=False) as (url, _):
        r = httpx.get(url + "/health")
        assert r.status_code == 503 and r.json()["status"] == "loading" and "BRAIN_DOWN:loading" in r.text
        assert "BRAIN_OK" not in r.text
        r = httpx.post(url + "/v1/chat/completions", json={"model": "system2", "messages": MSG})
        assert_error(r, 503, "brain_unavailable", "loading")
        # validation still happens first: no GPU time, no 503 for a bad request
        body = {"model": "system2", "messages": MSG, "x_adapters": {"x": 1}}
        r = httpx.post(url + "/v1/chat/completions", json=body)
        assert_error(r, 400, "invalid_adapter", "unknown")


def test_two_loras_explicit_zeros_and_unapproved_stack(tmp_path):
    reg = copy_registry(tmp_path, adapters={"style": gen_adapter("style")},
                        profile_patch=lambda d: d["backends"]["s2"]["loras"].append("style"))
    settings = make_settings(tmp_path, registry=reg)
    with run_gateway(settings) as (url, app):
        s2 = f"http://127.0.0.1:{settings.backend_ports['s2']}"
        assert app.state.brain.lora_ids["s2"] == {0: "abliteration", 1: "style"}

        def post(**b):
            return httpx.post(url + "/v1/chat/completions", json={"model": "system2", "messages": MSG, **b})

        def last():
            return httpx.get(s2 + "/_fake/requests").json()[-1]["body"]["lora"]

        assert post().status_code == 200
        assert last() == [{"id": 0, "scale": 1.0}, {"id": 1, "scale": 0.0}]
        r = post(x_adapters={"style": 0.5})
        assert r.json()["x_brain"] == {**r.json()["x_brain"], "adapters": {"abliteration": 1.0, "style": 0.5},
                                       "stack_evaluated": False}  # unapproved multi-adapter stack: allowed, flagged
        assert last() == [{"id": 0, "scale": 1.0}, {"id": 1, "scale": 0.5}]
        r = post(model="system2+none+style")
        assert r.json()["x_brain"]["adapters"] == {"style": 1.0} and r.json()["x_brain"]["stack_evaluated"] is True
        assert last() == [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 1.0}]
        post(model="system2+none")
        assert last() == [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 0.0}]


def test_serve_refuses_invalid_profile(tmp_path):
    from brain.app import create_app
    from brain.registry import RegistryError

    reg = copy_registry(tmp_path, profile_patch=lambda d: d["backends"]["s2"]["args"].update(batch=128))
    with pytest.raises(RegistryError, match="batch"):
        create_app(make_settings(tmp_path, registry=reg))
