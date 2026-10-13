import pytest
import yaml
from conftest import REGISTRY, copy_registry, gen_adapter, touch_model_files

from brain.config import load_settings
from brain.registry import AdapterManifest, BackendArgs, RegistryError, load_registry, validate_profile


def check(reg_dir, profile="ranger-dev", **kw):
    reg = load_registry(reg_dir)
    kw.setdefault("check_files", False)
    return validate_profile(reg, reg.profile(profile), **kw)


def test_real_registry_is_valid():
    reg = load_registry(REGISTRY)
    assert {"abliteration", "clef"} <= set(reg.adapters)
    for pid in ("ranger-dev", "mac-dev"):
        r = validate_profile(reg, reg.profile(pid), check_files=False)
        assert r.ok, r.errors
        assert any("measured_peak_gb" in w for w in r.warnings)  # unmeasured => warning, not error


def test_profile_by_path():
    reg = load_registry(REGISTRY)
    assert reg.profile("registry/profiles/ranger-dev.yaml").id == "ranger-dev"
    with pytest.raises(KeyError):
        reg.profile("nope")


def test_unknown_keys_are_errors(tmp_path):
    reg = copy_registry(tmp_path)
    p = reg / "adapters" / "abliteration.yaml"
    p.write_text(p.read_text() + "surprise: 1\n")
    with pytest.raises(RegistryError, match="surprise"):
        load_registry(reg)


def test_unknown_backend_arg_rejected(tmp_path):
    reg = copy_registry(tmp_path, profile_patch=lambda d: d["backends"]["s2"]["args"].update(rope_scale=2))
    with pytest.raises(RegistryError, match="rope_scale"):
        load_registry(reg)


def test_yaml_on_is_flash_attn_on():
    assert BackendArgs.model_validate(yaml.safe_load("flash_attn: on")).flash_attn == "on"


def test_sha256_format():
    m = gen_adapter("x")
    m["artifact"]["sha256"] = "unverified"
    AdapterManifest.model_validate(m)
    m["artifact"]["sha256"] = "abc"
    with pytest.raises(ValueError):
        AdapterManifest.model_validate(m)


def test_name_must_match_file(tmp_path):
    reg = copy_registry(tmp_path, adapters={"other": gen_adapter("different")})
    with pytest.raises(RegistryError, match="must equal the file name"):
        load_registry(reg)


def test_batch_lt_ubatch_names_both_values(tmp_path):
    reg = copy_registry(tmp_path, profile_patch=lambda d: d["backends"]["s2"]["args"].update(batch=256, ubatch=512))
    r = check(reg)
    assert any("batch (256) < ubatch (512)" in e for e in r.errors)


def test_max_prompt_tokens_le_ubatch(tmp_path):
    def patch(d):
        d["roles"]["system1"]["capabilities"]["max_prompt_tokens"] = 8192
    r = check(copy_registry(tmp_path, profile_patch=patch))
    assert any("max_prompt_tokens (8192) > ubatch (4096)" in e for e in r.errors)


def test_hf_revision_mismatch_names_both(tmp_path):
    bad = gen_adapter("oldbase", base={"hf_revision": "deadbeef"})
    reg = copy_registry(tmp_path, adapters={"oldbase": bad},
                        profile_patch=lambda d: d["backends"]["s2"]["loras"].append("oldbase"))
    r = check(reg)
    assert any("base_mismatch" in e and "deadbeef" in e and "c202236" in e for e in r.errors), r.errors


def test_weights_sha_mismatch(tmp_path):
    bad = gen_adapter("w", base={"weights_sha256": "a" * 64})
    reg = copy_registry(tmp_path, adapters={"w": bad}, profile_patch=lambda d: d["backends"]["s2"]["loras"].append("w"))
    mp = reg / "models" / "qwen3.5-9b-q4km.yaml"
    mp.write_text(mp.read_text().replace("weights_sha256: null", "weights_sha256: " + "b" * 64))
    r = check(reg)
    assert any("weights_sha256" in e and "a" * 64 in e and "b" * 64 in e for e in r.errors)


def test_quant_not_in_quant_compat(tmp_path):
    bad = gen_adapter("q", base={"quant_compat": ["Q8_0"]})
    reg = copy_registry(tmp_path, adapters={"q": bad}, profile_patch=lambda d: d["backends"]["s2"]["loras"].append("q"))
    r = check(reg)
    assert any("base_mismatch" in e and "Q4_K_M" in e for e in r.errors)


def test_decision_adapter_only_on_systemone_roles(tmp_path):
    reg = copy_registry(tmp_path)
    p = reg / "adapters" / "clef.yaml"
    p.write_text(p.read_text().replace("roles: [system1]", "roles: [system1, system2]"))
    r = check(reg)
    assert any("decision adapter lists non-systemone role system2" in e for e in r.errors)


def test_generation_adapter_not_on_systemone(tmp_path):
    reg = copy_registry(tmp_path)
    p = reg / "adapters" / "abliteration.yaml"
    p.write_text(p.read_text().replace("roles: [system2, system2-fast]", "roles: [system1, system2, system2-fast]"))
    assert any("generation adapter lists systemone role" in e for e in check(reg).errors)


def test_default_stack_requires(tmp_path):
    reg = copy_registry(
        tmp_path,
        adapters={"needs": gen_adapter("needs", compat={"requires": ["other"]}), "other": gen_adapter("other")},
        profile_patch=lambda d: (d["backends"]["s2"]["loras"].extend(["needs", "other"]),
                                 d["roles"]["system2"]["default_adapters"].update(needs=1.0)))
    assert any("default stack invalid: missing_requirement" in e for e in check(reg).errors)


def test_default_stack_incompatible(tmp_path):
    reg = copy_registry(
        tmp_path, adapters={"anti": gen_adapter("anti", compat={"incompatible_with": ["abliteration"]})},
        profile_patch=lambda d: (d["backends"]["s2"]["loras"].append("anti"),
                                 d["roles"]["system2"]["default_adapters"].update(anti=1.0)))
    assert any("default stack invalid: incompatible" in e for e in check(reg).errors)


def test_approved_stack_exclusive_group(tmp_path):
    reg = copy_registry(
        tmp_path,
        adapters={"p1": gen_adapter("p1", compat={"exclusive_group": "persona"}),
                  "p2": gen_adapter("p2", compat={"exclusive_group": "persona"})},
        profile_patch=lambda d: (d["backends"]["s2"]["loras"].extend(["p1", "p2"]),
                                 d.update(approved_stacks=[{"adapters": ["p1", "p2"]}])))
    assert any("approved_stacks[0]" in e and "exclusive_group" in e for e in check(reg).errors)


def test_lora_on_clef_backend_and_mmproj_family(tmp_path):
    def patch(d):
        d["backends"]["s1"]["loras"] = ["abliteration"]
        d["backends"]["s2"]["mmproj"] = "clef-flash-mmproj-q8_0"
    errs = check(copy_registry(tmp_path, profile_patch=patch)).errors
    assert any("clef backend cannot take generation LoRAs" in e for e in errs)
    assert any("mmproj family" in e for e in errs)


def test_router_task_unknown_role(tmp_path):
    reg = copy_registry(tmp_path)
    (reg / "router.yaml").write_text("tasks: {x: {role: system9}}\n")
    assert any("router task x" in e for e in check(reg).errors)


def test_budget(tmp_path):
    def measured(d):
        d["budget"] = {"peak_gb": 12.0}
        d["backends"]["s1"]["measured_peak_gb"] = 6.0
        d["backends"]["s2"]["measured_peak_gb"] = 7.0
    assert any("13.00 GB > peak_gb 12.0" in e for e in check(copy_registry(tmp_path, profile_patch=measured)).errors)
    r = check(copy_registry(tmp_path / "b", profile_patch=lambda d: d.update(budget={"peak_gb": 12.0})))
    assert r.ok and any("not checked" in w for w in r.warnings)


def test_files_exist_and_sha256(tmp_path):
    reg_dir = copy_registry(tmp_path)
    models = tmp_path / "models"
    r = check(reg_dir, models_dir=models, check_files=True)
    assert any("file not found" in e for e in r.errors)
    touch_model_files(reg_dir, models)
    assert check(reg_dir, models_dir=models, check_files=True).ok
    # sha256: an empty file's digest
    empty = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    mp = reg_dir / "models" / "qwen3.5-9b-q4km.yaml"
    mp.write_text(mp.read_text().replace("\nsha256: null\n", f"\nsha256: {empty}\n"))
    r = check(reg_dir, models_dir=models, check_files=True, sha256=True, cache_dir=tmp_path / "c")
    assert r.ok and (tmp_path / "c" / "sha256.json").exists()
    mp.write_text(mp.read_text().replace(empty, "f" * 64))
    r = check(reg_dir, models_dir=models, check_files=True, sha256=True, cache_dir=tmp_path / "c")
    assert any("sha256 mismatch" in e for e in r.errors)


def test_settings_env_and_yaml(tmp_path):
    cfg = tmp_path / "brain.yaml"
    cfg.write_text("port: 9000\nprofile: mac-dev\nsupervisor: {crash_loop_max: 2}\n")
    s = load_settings({"BRAIN_CONFIG": str(cfg), "BRAIN_PORT": "9100", "BRAIN_LLAMA_SERVER": "/opt/ls"})
    got = (s.port, s.profile, s.llama_server, s.supervisor.crash_loop_max, s.token)
    assert got == (9100, "mac-dev", "/opt/ls", 2, None)
    assert load_settings({}).llama_server == "llama-server"
