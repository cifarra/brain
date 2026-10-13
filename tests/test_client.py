import json

import httpx
import pytest
from conftest import fake_binary, free_port, make_settings, run_gateway

from brain_client import BrainError, BrainUnavailable, Client

MSG = [{"role": "user", "content": "hi"}]


# ------------------------------------------------------------ failover rules (mock transport)

def mock_client(behaviour: dict[str, object], calls: list, **kw) -> Client:
    """behaviour: host -> status code, or an exception class to raise."""

    def handler(req: httpx.Request):
        calls.append(req.url.host)
        b = behaviour[req.url.host]
        if isinstance(b, type) and issubclass(b, Exception):
            raise b("boom", request=req)
        body = {"ok": True, "x_brain": {"profile": "p", "adapters": {}}} if b == 200 else \
            {"error": {"code": "x", "reason": "loading", "message": "m"}}
        return httpx.Response(b, json=body)

    return Client(["http://a", "http://b"], transport=httpx.MockTransport(handler), **kw)


@pytest.mark.parametrize("primary", [503, httpx.ConnectError, httpx.ConnectTimeout])
def test_failover_and_sticky(primary):
    calls = []
    c = mock_client({"a": primary, "b": 200}, calls)
    r = c.models()
    assert r.served_by == "http://b" and r.profile == "p" and calls == ["a", "b"]
    calls.clear()
    c.models()
    assert calls == ["b"]  # sticky
    c._sticky = (c._sticky[0], 0)  # stickiness expired: primary first again
    calls.clear()
    c.models()
    assert calls == ["a", "b"]


@pytest.mark.parametrize("status", [400, 401, 404, 429, 500, 504])
def test_no_failover_on_other_errors(status):
    calls = []
    c = mock_client({"a": status, "b": 200}, calls)
    with pytest.raises(BrainError) as ei:
        c.decide({"state": "s", "questions": {}})
    assert ei.value.status == status and ei.value.served_by == "http://a" and calls == ["a"]


def test_all_unavailable():
    c = mock_client({"a": 503, "b": httpx.ConnectError}, [])
    with pytest.raises(BrainUnavailable) as ei:
        c.models()
    assert [e for e, _ in ei.value.failures] == ["http://a", "http://b"]


def test_chat_body_mapping():
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        return httpx.Response(200, json={"choices": [], "x_brain": {"adapters": {"abliteration": 0.5}}})

    c = Client("http://a", token="t", transport=httpx.MockTransport(handler))
    r = c.chat(MSG, adapters={"abliteration": 0.5}, thinking=64, response_format={"type": "json_object"},
               temperature=0.1)
    assert seen[-1] == {"model": "system2", "messages": MSG, "temperature": 0.1,
                        "x_adapters": {"abliteration": 0.5}, "x_thinking_budget": 64,
                        "chat_template_kwargs": {"enable_thinking": True},
                        "response_format": {"type": "json_object"}}
    assert r.adapters == {"abliteration": 0.5}
    c.chat(MSG, task="translate", thinking=False)
    assert "model" not in seen[-1] and seen[-1]["x_task"] == "translate"
    assert seen[-1]["chat_template_kwargs"] == {"enable_thinking": False}


# ------------------------------------------------------------ two real gateways

@pytest.fixture(scope="module")
def gateways(tmp_path_factory):
    t1, t2 = tmp_path_factory.mktemp("lab"), tmp_path_factory.mktemp("local")
    lab = make_settings(t1, binary=fake_binary(t1, {"FAKE_STARTUP_DELAY": "999"}))  # stuck loading: 503
    local = make_settings(t2)
    with run_gateway(lab, wait_ok=False) as (lab_url, _), run_gateway(local) as (local_url, _):
        yield lab_url, local_url


def test_sdk_failover_between_gateways(gateways):
    lab, local = gateways
    dead = f"http://127.0.0.1:{free_port()}"
    with Client([dead, lab, local]) as c:
        r = c.decide({"state": "s", "questions": {"q": {"type": "noul", "instructions": "ok?"}}})
        assert r.served_by == local and r.profile == "ranger-dev" and "q" in r["answers"]
        r = c.chat(MSG, adapters={"abliteration": 0.5})
        assert r.served_by == local and r.adapters == {"abliteration": 0.5}
        stream = c.chat(MSG, stream=True, thinking=False)
        chunks = list(stream)
        assert stream.served_by == local and stream.adapters == {"abliteration": 1.0}
        assert any(ch["choices"] and ch["choices"][0]["delta"].get("content") for ch in chunks)
        health = c.health()
        assert health[dead]["status"] == "unreachable" and health[lab]["status"] == "loading"
        assert health[local]["status"] == "ok"


def test_sdk_client_error_not_failed_over(gateways):
    _, local = gateways
    with Client([local, "http://127.0.0.1:1"]) as c:
        with pytest.raises(BrainError) as ei:
            c.chat(MSG, adapters={"nope": 1})
        assert ei.value.status == 400 and ei.value.code == "invalid_adapter" and ei.value.reason == "unknown"
        assert ei.value.served_by == local
        assert {m["id"] for m in c.models()["data"]} == {"system1", "system2", "system2-fast"}
        assert {a["name"] for a in c.adapters()["data"]} == {"abliteration", "clef"}
