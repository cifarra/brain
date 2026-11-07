"""The web UI: a static page and the eval suites, served next to the API."""

import json

import httpx
import pytest
from conftest import copy_registry, make_settings, run_gateway

SUITE = {"name": "tiny", "title": "Tiny", "endpoint": "chat", "cases": [{"id": "a", "request": {}, "check": {"any": ["x"]}}]}


@pytest.fixture
def ui(tmp_path):
    reg = copy_registry(tmp_path)
    (reg / "evals").mkdir()
    (reg / "evals" / "tiny.json").write_text(json.dumps(SUITE))
    (reg / "evals" / "broken.json").write_text("{")
    with run_gateway(make_settings(tmp_path, registry=reg, token="s3cret")) as (url, _):
        with httpx.Client(base_url=url, timeout=30) as c:
            yield c


def test_page_is_public_and_the_api_still_needs_the_token(ui):
    r = ui.get("/", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/ui/"
    r = ui.get("/ui/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "<title>brain</title>" in r.text
    assert ui.get("/v1/models").status_code == 401


def test_eval_suites_are_listed_and_served(ui):
    data = ui.get("/ui/evals").json()["data"]
    assert data == [{"name": "tiny", "title": "Tiny", "description": None, "endpoint": "chat", "model": None,
                     "gate": None, "n": 1}]  # the unparsable file is skipped
    assert ui.get("/ui/evals/tiny").json() == SUITE
    for bad in ("nope", "..%2Frouter", "broken.json"):
        r = ui.get(f"/ui/evals/{bad}")
        assert r.status_code == 404, bad
        assert r.json()["error"]["type"] == "not_found_error"


def test_the_shipped_suites_are_well_formed():
    from pathlib import Path
    for f in sorted((Path(__file__).parent.parent / "registry" / "evals").glob("*.json")):
        suite = json.loads(f.read_text())
        assert suite["name"] == f.stem and suite["endpoint"] in ("systemone", "chat") and suite["cases"]
        for c in suite["cases"]:
            assert c["id"] and isinstance(c["request"], dict)
            assert ("ref" in c) if suite["endpoint"] == "systemone" else ("check" in c)
