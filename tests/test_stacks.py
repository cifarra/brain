import pytest
from conftest import REGISTRY, copy_registry, gen_adapter

from brain import stacks
from brain.registry import load_registry
from brain.stacks import AdapterError, StackContext, lora_payload, parse_model, resolve

EXTRA = {
    "style": gen_adapter("style"),
    "style2": gen_adapter("style2"),
    "p1": gen_adapter("p1", compat={"exclusive_group": "persona"}),
    "p2": gen_adapter("p2", compat={"exclusive_group": "persona"}),
    "needs": gen_adapter("needs", compat={"requires": ["abliteration"]}),
    "anti": gen_adapter("anti", compat={"incompatible_with": ["abliteration"]}),
    "broken": gen_adapter("broken", evals={"gate": "failed"}),
    "fresh": gen_adapter("fresh", evals={"gate": "pending"}),
    "oldbase": gen_adapter("oldbase", base={"hf_revision": "deadbeef"}),
    "s2only": gen_adapter("s2only", roles=["system2"]),
    "unloaded": gen_adapter("unloaded"),
}


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    loaded = [n for n in EXTRA if n != "unloaded"]
    reg_dir = copy_registry(
        tmp_path_factory.mktemp("stacks"), adapters=EXTRA,
        profile_patch=lambda d: (d["backends"]["s2"]["loras"].extend(loaded),
                                 d.update(approved_stacks=[{"adapters": ["style", "style2"]}])))
    reg = load_registry(reg_dir)
    return StackContext.from_profile(reg, reg.profile("ranger-dev"))  # validator skipped on purpose


def role(ctx, name="system2"):
    return ctx.profile.role(name)


# ------------------------------------------------------------ qualified names

@pytest.mark.parametrize("name,expected", [
    ("system2", ("system2", None, False)),
    ("system2+none", ("system2", {}, True)),
    ("system2+abliteration@0.5", ("system2", {"abliteration": 0.5}, False)),
    ("system2+abliteration+style", ("system2", {"abliteration": None, "style": None}, False)),
    ("system2-fast+none+style@1.2", ("system2-fast", {"style": 1.2}, True)),
])
def test_parse_model(name, expected):
    assert parse_model(name) == expected


@pytest.mark.parametrize("bad", ["", "+x", "system2+", "system2+a@", "system2+a@x", "system2++a"])
def test_parse_model_errors(bad):
    with pytest.raises(ValueError):
        parse_model(bad)


# ------------------------------------------------------------ resolution

def test_role_default(ctx):
    s = resolve(ctx, role(ctx))
    assert s.adapters == {"abliteration": 1.0}
    assert s.evaluated is False  # abliteration gate is pending


def test_inherited_role_default(ctx):
    r = role(ctx, "system2-fast")
    assert r.backend == "s2" and r.defaults.thinking is False and r.default_adapters == {"abliteration": 1.0}


def test_precedence_request_over_task_over_default(ctx):
    task = ctx.reg.router.tasks["translate"].model_copy(update={"x_adapters": {"abliteration": 0.7, "style": 0.5}})
    s = resolve(ctx, role(ctx), {"style": 0.9}, task=task)
    assert s.adapters == {"abliteration": 0.7, "style": 0.9}


def test_scale_zero_removes(ctx):
    assert resolve(ctx, role(ctx), {"abliteration": 0}).adapters == {}


def test_replace_and_none(ctx):
    assert resolve(ctx, role(ctx), {}, replace=True).adapters == {}
    assert resolve(ctx, role(ctx), {"style": 1.0}, replace=True).adapters == {"style": 1.0}
    _, ads, rep = parse_model("system2+none")
    assert resolve(ctx, role(ctx), ads, rep).adapters == {}


def test_task_replace_mode(ctx):
    task = ctx.reg.router.tasks["json"].model_copy(update={"x_adapters_mode": "replace"})
    assert resolve(ctx, role(ctx), task=task).adapters == {}


def test_qualified_without_scale_uses_manifest_default(ctx):
    _, ads, _ = parse_model("system2+style")
    assert resolve(ctx, role(ctx), ads).adapters == {"abliteration": 1.0, "style": 1.0}


def test_evaluated(ctx):
    assert resolve(ctx, role(ctx), {"style": 1}, replace=True).evaluated is True  # single, gate passed
    assert resolve(ctx, role(ctx), {"fresh": 1}, replace=True).evaluated is False  # gate pending
    assert resolve(ctx, role(ctx), {"style": 1, "style2": 1}, replace=True).evaluated is True  # approved
    s = resolve(ctx, role(ctx), {"style": 1, "p1": 1}, replace=True)  # unapproved: allowed, flagged
    assert s.adapters == {"style": 1.0, "p1": 1.0} and s.evaluated is False
    assert resolve(ctx, role(ctx), {}, replace=True).evaluated is True


# ------------------------------------------------------------ every reason code

@pytest.mark.parametrize("role_name,request_stack,reason", [
    ("system2", {"nope": 1}, "unknown"),
    ("system2", {"unloaded": 1}, "not_loaded"),
    ("system2", {"clef": 1}, "role"),
    ("system2-fast", {"s2only": 1}, "role"),
    ("system1", {"abliteration": 1}, "role"),
    ("system2", {"oldbase": 1}, "base_mismatch"),
    ("system2", {"abliteration": 2.0}, "scale_range"),
    ("system2", {"abliteration": -0.5}, "scale_range"),
    ("system2", {"abliteration": "1"}, "scale_range"),
    ("system2", {"anti": 1}, "incompatible"),
    ("system2", {"abliteration": 0, "needs": 1}, "missing_requirement"),
    ("system2", {"p1": 1, "p2": 1}, "exclusive_group"),
    ("system2", {"style": 1, "style2": 1, "p1": 1}, "stack_too_large"),
    ("system2", {"broken": 1}, "gate"),
])
def test_reason_codes(ctx, role_name, request_stack, reason):
    with pytest.raises(AdapterError) as ei:
        resolve(ctx, role(ctx, role_name), request_stack)
    assert ei.value.reason == reason


def test_kind_incompatibility(ctx):
    clef, abl = ctx.reg.adapters["clef"], ctx.reg.adapters["abliteration"]
    assert clef.conflicts_with(abl) and abl.conflicts_with(clef)
    with pytest.raises(AdapterError) as ei:
        stacks.check_stack_rules(ctx, {"clef": 1, "abliteration": 1})
    assert ei.value.reason == "incompatible"


def test_requires_satisfied(ctx):
    assert resolve(ctx, role(ctx), {"needs": 1}).adapters == {"abliteration": 1.0, "needs": 1.0}


def test_not_loaded_follows_runtime_map():
    reg = load_registry(REGISTRY)
    ctx = StackContext.from_profile(reg, reg.profile("ranger-dev"), loaded={"s1": {"clef"}, "s2": set()})
    with pytest.raises(AdapterError) as ei:
        resolve(ctx, ctx.profile.role("system2"))
    assert ei.value.reason == "not_loaded"


# ------------------------------------------------------------ llama.cpp mapping

def test_lora_payload_explicit_zeros():
    ids = {0: "abliteration", 1: "style", 2: None}  # id 2: a file the registry does not know
    assert lora_payload(ids, {"style": 0.5}) == [
        {"id": 0, "scale": 0.0}, {"id": 1, "scale": 0.5}, {"id": 2, "scale": 0.0}]
    assert lora_payload(ids, {}) == [{"id": i, "scale": 0.0} for i in range(3)]
    assert lora_payload({}, {}) == []


# ------------------------------------------------------------ router

def test_router_tasks(ctx):
    tasks = ctx.reg.router.tasks
    assert set(tasks) == {"default", "translate", "json"}
    tr = tasks["translate"]
    assert tr.role == "system2-fast" and tr.defaults.temperature == 0.3
    assert resolve(ctx, role(ctx, tr.role), task=tr).adapters == {"abliteration": 1.0}
    assert tasks["json"].defaults.response_format == {"type": "json_object"}
