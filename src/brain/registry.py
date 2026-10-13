"""Registry: model, adapter and profile manifests plus router.yaml, and the validator.

Every manifest is a pydantic model with extra="forbid", so an unknown key is an error.
Paths in manifests are relative to the models dir (/models in the container).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import AfterValidator, Field, ValidationError, field_validator, model_validator

from .config import Strict

NAME = r"^[a-z0-9][a-z0-9._-]*$"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
CHAT, SYSTEMONE = "chat", "systemone"


def _sha(v: str | None) -> str | None:
    if v is None or v == "unverified" or SHA256.match(v):
        return v
    raise ValueError("sha256 must be null, 'unverified' or 64 lowercase hex chars")


Sha = Annotated[str | None, AfterValidator(_sha)]


# ---------------------------------------------------------------- models


class ModelSource(Strict):
    model: str
    hf_revision: str | None = None
    weights_sha256: Sha = None
    gguf_from: str | None = None  # repo id, or "built"
    gguf_revision: str | None = None


class DerivedFrom(Strict):
    model: str
    recipe: str


class ModelManifest(Strict):
    id: str = Field(pattern=NAME)
    kind: Literal["base", "mmproj", "decision-merged"]
    arch: str  # qwen35 | clef | clip (mmproj)
    family: str
    source: ModelSource
    quant: str
    file: str
    sha256: Sha = None
    size_gb: float | None = None
    derived_from: DerivedFrom | None = None
    license: str | None = None


# ---------------------------------------------------------------- adapters


class AdapterBase(Strict):
    model: str
    hf_revision: str | None = None
    weights_sha256: Sha = None
    quant_compat: list[str] = []


class Artifact(Strict):
    format: Literal["gguf-lora", "merged-gguf", "gguf-lora+head"]
    arch: str
    file: str | None = None  # gguf-lora: the LoRA file
    family: str | None = None  # merged-gguf: model family the adapter is baked into
    sha256: Sha = None
    size_mb: float | None = None

    @model_validator(mode="after")
    def _shape(self):
        if self.format == "gguf-lora" and not self.file:
            raise ValueError("gguf-lora artifacts need 'file'")
        if self.format == "merged-gguf" and not self.family:
            raise ValueError("merged-gguf artifacts need 'family'")
        if self.file and "," in self.file:
            raise ValueError("LoRA paths may not contain ',' (llama-server splits --lora on commas)")
        return self


class Scale(Strict):
    default: float = 1.0
    min: float = 0.0
    max: float = 1.0

    @model_validator(mode="after")
    def _order(self):
        if not self.min <= self.default <= self.max:
            raise ValueError("scale must satisfy min <= default <= max")
        return self


class Compat(Strict):
    incompatible_with: list[str] = []  # adapter names or "kind:<kind>"
    requires: list[str] = []
    exclusive_group: str | None = None


class Provenance(Strict):
    source: str | None = None
    method: str | None = None
    license: str | None = None
    created_at: str | None = None

    @field_validator("created_at", mode="before")
    @classmethod
    def _date(cls, v):
        return None if v is None else str(v)


class Evals(Strict):
    gate: Literal["passed", "failed", "pending"] = "pending"
    report: str | None = None
    metrics: dict[str, Any] = {}


class AdapterManifest(Strict):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    version: int = 1
    kind: Literal["generation", "decision"]
    description: str = ""
    router_description: str | None = None
    base: AdapterBase
    artifact: Artifact
    target_modules: list[str] = []
    rank: int | None = None
    scale: Scale = Field(default_factory=Scale)
    roles: list[str]
    compat: Compat = Field(default_factory=Compat)
    provenance: Provenance = Field(default_factory=Provenance)
    evals: Evals = Field(default_factory=Evals)

    @model_validator(mode="after")
    def _kind_format(self):
        ok = {"generation": {"gguf-lora"}, "decision": {"merged-gguf", "gguf-lora+head"}}
        if self.artifact.format not in ok[self.kind]:
            raise ValueError(f"{self.kind} adapters cannot use artifact format {self.artifact.format}")
        return self

    def conflicts_with(self, other: AdapterManifest) -> bool:
        return other.name in self.compat.incompatible_with or f"kind:{other.kind}" in self.compat.incompatible_with


# ---------------------------------------------------------------- profiles


class BackendArgs(Strict):
    """Only these keys become llama-server flags; anything else is a validation error."""

    ctx: int | None = None
    batch: int | None = None
    ubatch: int | None = None
    parallel: int | None = None
    flash_attn: Literal["on", "off", "auto"] | None = None
    ngl: int | Literal["all"] | None = None
    ctx_checkpoints: int | None = None
    cache_type_k: str | None = None
    cache_type_v: str | None = None
    reasoning_format: str | None = None
    reasoning_budget: int | None = None
    jinja: bool | None = None

    @field_validator("flash_attn", mode="before")
    @classmethod
    def _yaml_bool(cls, v):  # YAML 1.1 reads a bare `on` as True
        return {True: "on", False: "off"}.get(v, v) if isinstance(v, bool) else v

    @property
    def eff_batch(self) -> int:
        return self.batch or 2048  # llama-server defaults

    @property
    def eff_ubatch(self) -> int:
        return self.ubatch or 512


class Backend(Strict):
    model: str
    mmproj: str | None = None
    loras: list[str] = []  # adapter names; load order == llama.cpp id order
    args: BackendArgs = Field(default_factory=BackendArgs)
    measured_peak_gb: float | None = None


class RoleDefaults(Strict):
    """Request defaults supplied by a role or a router task; explicit request fields win."""

    thinking: bool | None = None
    reasoning_budget: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repeat_penalty: float | None = None
    max_tokens: int | None = None
    response_format: dict[str, Any] | None = None


class Capabilities(Strict):
    systemone: bool = False
    chat: bool = False
    thinking: bool = False
    json_schema: bool = False
    vision: bool = False
    images: bool = False  # images on /v1/systemone
    max_prompt_tokens: int | None = None
    max_context: int | None = None


class Role(Strict):
    backend: str | None = None
    endpoint: Literal["chat", "systemone"] | None = None
    inherits: str | None = None
    defaults: RoleDefaults = Field(default_factory=RoleDefaults)
    default_adapters: dict[str, float] | None = None
    capabilities: Capabilities | None = None


@dataclass
class ResolvedRole:
    name: str
    backend: str
    endpoint: str
    defaults: RoleDefaults
    default_adapters: dict[str, float]
    capabilities: Capabilities


class Budget(Strict):
    steady_gb: float | None = None
    peak_gb: float | None = None


class ApprovedStack(Strict):
    adapters: list[str]
    report: str | None = None


class Profile(Strict):
    id: str = Field(pattern=NAME)
    phase: Literal["A"] = "A"
    description: str = ""
    budget: Budget | None = None
    max_stack: int = 3
    backends: dict[str, Backend]
    roles: dict[str, Role]
    approved_stacks: list[ApprovedStack] = []

    def role(self, name: str) -> ResolvedRole:
        chain, cur = [], name
        while cur is not None:
            if cur in chain:
                raise ValueError(f"role inheritance cycle: {' -> '.join(chain + [cur])}")
            if cur not in self.roles:
                raise KeyError(cur)
            chain.append(cur)
            cur = self.roles[cur].inherits
        merged: dict[str, Any] = {"defaults": {}, "capabilities": {}}
        for r in reversed([self.roles[c] for c in chain]):  # base first, child overrides
            for k in ("backend", "endpoint", "default_adapters"):
                if getattr(r, k) is not None:
                    merged[k] = getattr(r, k)
            merged["defaults"].update(r.defaults.model_dump(exclude_unset=True))
            if r.capabilities:
                merged["capabilities"].update(r.capabilities.model_dump(exclude_unset=True))
        if not merged.get("backend") or not merged.get("endpoint"):
            raise ValueError(f"role {name}: backend and endpoint are required (directly or via inherits)")
        return ResolvedRole(
            name=name,
            backend=merged["backend"],
            endpoint=merged["endpoint"],
            defaults=RoleDefaults(**merged["defaults"]),
            default_adapters=dict(merged.get("default_adapters") or {}),
            capabilities=Capabilities(**merged["capabilities"]),
        )


# ---------------------------------------------------------------- router


class Task(Strict):
    role: str
    description: str = ""
    x_adapters: dict[str, float] = {}
    x_adapters_mode: Literal["merge", "replace"] = "merge"
    defaults: RoleDefaults = Field(default_factory=RoleDefaults)


class Router(Strict):
    tasks: dict[str, Task] = {}


# ---------------------------------------------------------------- loading


class RegistryError(Exception):
    def __init__(self, errors: list[str]):
        super().__init__("\n".join(errors))
        self.errors = errors


@dataclass
class Registry:
    root: Path
    models: dict[str, ModelManifest] = field(default_factory=dict)
    adapters: dict[str, AdapterManifest] = field(default_factory=dict)
    profiles: dict[str, Profile] = field(default_factory=dict)
    router: Router = field(default_factory=Router)

    def profile(self, ref: str) -> Profile:
        """A profile id, or a path to a profile YAML file inside this registry."""
        if ref in self.profiles:
            return self.profiles[ref]
        p = Path(ref)
        if p.suffix in (".yaml", ".yml") and p.stem in self.profiles:
            return self.profiles[p.stem]
        raise KeyError(f"unknown profile {ref!r} (known: {', '.join(sorted(self.profiles))})")

    def backend_model(self, profile: Profile, backend: str) -> ModelManifest:
        return self.models[profile.backends[backend].model]

    def configured_adapters(self, profile: Profile, backend: str) -> list[str]:
        """Adapters the profile makes available on a backend: its LoRAs, plus decision
        adapters merged into the backend's model family."""
        be = profile.backends[backend]
        model = self.models.get(be.model)
        names = [n for n in be.loras if n in self.adapters]
        for a in self.adapters.values():
            if a.artifact.format == "merged-gguf" and model and model.family == a.artifact.family:
                names.append(a.name)
        return names

    def adapter_backends(self, profile: Profile, name: str) -> list[str]:
        return [b for b in profile.backends if name in self.configured_adapters(profile, b)]


def _load_dir(dirpath: Path, cls, key: str, errors: list[str]) -> dict:
    out = {}
    for f in sorted(dirpath.glob("*.yaml")) if dirpath.is_dir() else []:
        try:
            obj = cls.model_validate(yaml.safe_load(f.read_text()) or {})
        except (ValidationError, yaml.YAMLError) as e:
            errors.append(f"{f}: {e}")
            continue
        ident = getattr(obj, key)
        if ident != f.stem:
            errors.append(f"{f}: {key} {ident!r} must equal the file name {f.stem!r}")
        out[ident] = obj
    return out


def load_registry(root: Path | str) -> Registry:
    root = Path(root)
    errors: list[str] = []
    if not root.is_dir():
        raise RegistryError([f"registry dir not found: {root}"])
    reg = Registry(
        root=root,
        models=_load_dir(root / "models", ModelManifest, "id", errors),
        adapters=_load_dir(root / "adapters", AdapterManifest, "name", errors),
        profiles=_load_dir(root / "profiles", Profile, "id", errors),
    )
    rpath = root / "router.yaml"
    if rpath.exists():
        try:
            reg.router = Router.model_validate(yaml.safe_load(rpath.read_text()) or {})
        except (ValidationError, yaml.YAMLError) as e:
            errors.append(f"{rpath}: {e}")
    if errors:
        raise RegistryError(errors)
    return reg


# ---------------------------------------------------------------- validator


def compat_errors(adapter: AdapterManifest, model: ModelManifest) -> list[str]:
    """Why `adapter` cannot be applied on a backend running `model` (empty list = compatible)."""
    errs = []
    if adapter.artifact.format == "merged-gguf":
        if model.family != adapter.artifact.family:
            errs.append(f"family {model.family!r} != adapter family {adapter.artifact.family!r}")
    else:
        if adapter.artifact.arch != model.arch:
            errs.append(f"arch: adapter {adapter.artifact.arch!r} != model {model.arch!r}")
        if adapter.base.model != model.source.model:
            errs.append(f"base model: adapter {adapter.base.model!r} != backend {model.source.model!r}")
        if adapter.base.hf_revision != model.source.hf_revision:
            errs.append(
                f"hf_revision: adapter {adapter.base.hf_revision!r} != backend {model.source.hf_revision!r}"
            )
        a, m = adapter.base.weights_sha256, model.source.weights_sha256
        if a and m and "unverified" not in (a, m) and a != m:
            errs.append(f"weights_sha256: adapter {a} != backend {m}")
    if model.quant not in adapter.base.quant_compat:
        errs.append(f"quant {model.quant!r} not in quant_compat {adapter.base.quant_compat}")
    return errs


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


class Sha256Cache:
    """sha256 of large files, cached by (path, size, mtime) in a JSON file."""

    def __init__(self, path: Path):
        self.path = path
        try:
            self.data = json.loads(path.read_text())
        except (OSError, ValueError):
            self.data = {}

    def digest(self, f: Path) -> str:
        st = f.stat()
        key = f"{f.resolve()}|{st.st_size}|{st.st_mtime_ns}"
        if key not in self.data:
            h = hashlib.sha256()
            with f.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 24), b""):
                    h.update(chunk)
            self.data[key] = h.hexdigest()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=1))
        return self.data[key]


def validate_profile(
    reg: Registry,
    profile: Profile,
    models_dir: Path | None = None,
    check_files: bool = True,
    sha256: bool = False,
    cache_dir: Path | None = None,
) -> Report:
    from . import stacks  # stacks depends on the types above

    r = Report()
    E, W = r.errors.append, r.warnings.append

    # --- references, files, args
    files: list[tuple[str, str, str | None]] = []  # (what, relpath, sha256)
    for bname, be in profile.backends.items():
        model = reg.models.get(be.model)
        if model is None:
            E(f"backend {bname}: unknown model {be.model!r}")
            continue
        if model.kind == "mmproj":
            E(f"backend {bname}: model {model.id} is an mmproj, not a base model")
        files.append((f"model {model.id}", model.file, model.sha256))
        if be.mmproj:
            mm = reg.models.get(be.mmproj)
            if mm is None:
                E(f"backend {bname}: unknown mmproj {be.mmproj!r}")
            elif mm.kind != "mmproj":
                E(f"backend {bname}: {mm.id} is not an mmproj (kind {mm.kind})")
            elif mm.family != model.family:
                E(f"backend {bname}: mmproj family {mm.family!r} != model family {model.family!r}")
            else:
                files.append((f"mmproj {mm.id}", mm.file, mm.sha256))
        a = be.args
        if a.eff_batch < a.eff_ubatch:
            E(f"backend {bname}: batch ({a.eff_batch}) < ubatch ({a.eff_ubatch})")
        if len(be.loras) != len(set(be.loras)):
            E(f"backend {bname}: duplicate loras")
        for name in be.loras:
            ad = reg.adapters.get(name)
            if ad is None:
                E(f"backend {bname}: unknown lora adapter {name!r}")
                continue
            if ad.artifact.format != "gguf-lora":
                E(f"backend {bname}: adapter {name} ({ad.kind}, {ad.artifact.format}) cannot be a --lora")
                continue
            for msg in compat_errors(ad, model):
                E(f"backend {bname}: adapter {name}: base_mismatch: {msg}")
            files.append((f"adapter {name}", ad.artifact.file, ad.artifact.sha256))
        if model.arch == "clef" and be.loras:
            E(f"backend {bname}: a clef backend cannot take generation LoRAs")

    # --- roles
    roles: dict[str, ResolvedRole] = {}
    for rname in profile.roles:
        try:
            role = roles[rname] = profile.role(rname)
        except (KeyError, ValueError) as e:
            E(f"role {rname}: {e}")
            continue
        if role.backend not in profile.backends:
            E(f"role {rname}: unknown backend {role.backend!r}")
            continue
        model = reg.models.get(profile.backends[role.backend].model)
        if model:
            is_clef = model.kind == "decision-merged"
            if role.endpoint == SYSTEMONE and not is_clef:
                E(f"role {rname}: systemone needs a decision model, backend {role.backend} runs {model.id}")
            if role.endpoint == CHAT and is_clef:
                E(f"role {rname}: a decision model ({model.id}) cannot serve chat")
        if role.endpoint == SYSTEMONE and role.capabilities.max_prompt_tokens:
            ub = profile.backends[role.backend].args.eff_ubatch
            if role.capabilities.max_prompt_tokens > ub:
                E(f"role {rname}: max_prompt_tokens ({role.capabilities.max_prompt_tokens}) > ubatch ({ub})")

    # --- adapter placement: decision adapters only on systemone roles, generation never there
    for ad in reg.adapters.values():
        for rname in ad.roles:
            role = roles.get(rname)
            if role is None:
                continue
            if ad.kind == "decision" and role.endpoint != SYSTEMONE:
                E(f"adapter {ad.name}: decision adapter lists non-systemone role {rname}")
            if ad.kind == "generation" and role.endpoint == SYSTEMONE:
                E(f"adapter {ad.name}: generation adapter lists systemone role {rname}")
        for dep in ad.compat.requires + [x for x in ad.compat.incompatible_with if not x.startswith("kind:")]:
            if dep not in reg.adapters:
                W(f"adapter {ad.name}: compat refers to unregistered adapter {dep!r}")

    # --- default stacks, approved stacks, router tasks hold the stack rules
    ctx = stacks.StackContext.from_profile(reg, profile)
    for rname, role in roles.items():
        try:
            st = stacks.resolve(ctx, role)
            if not st.evaluated:
                W(f"role {rname}: default stack {st.adapters} is not evaluated (gate pending or unapproved stack)")
        except stacks.AdapterError as e:
            E(f"role {rname}: default stack invalid: {e.reason}: {e}")
    for i, ap in enumerate(profile.approved_stacks):
        unknown = [n for n in ap.adapters if n not in reg.adapters]
        if unknown:
            E(f"approved_stacks[{i}]: unknown adapters {unknown}")
            continue
        try:
            stacks.check_stack_rules(ctx, {n: 1.0 for n in ap.adapters})
        except stacks.AdapterError as e:
            E(f"approved_stacks[{i}] {ap.adapters}: {e.reason}: {e}")
    for tname, task in reg.router.tasks.items():
        if task.role not in roles:
            E(f"router task {tname}: role {task.role!r} is not in profile {profile.id}")
            continue
        try:
            stacks.resolve(ctx, roles[task.role], task=task)
        except stacks.AdapterError as e:
            E(f"router task {tname}: {e.reason}: {e}")

    # --- budget (rev 5: the brain's own peak only)
    peaks = [be.measured_peak_gb for be in profile.backends.values()]
    if profile.budget and profile.budget.peak_gb is not None:
        if any(p is None for p in peaks):
            W(f"budget peak_gb={profile.budget.peak_gb} not checked: measured_peak_gb missing (Experiment 0)")
        elif sum(peaks) > profile.budget.peak_gb:
            E(f"budget: sum of measured peaks {sum(peaks):.2f} GB > peak_gb {profile.budget.peak_gb}")
    elif any(p is None for p in peaks):
        W("measured_peak_gb not set for every backend (Experiment 0 fills it in)")

    # --- files (and optionally sha256)
    if check_files and models_dir is not None:
        cache = Sha256Cache((cache_dir or Path.home() / ".cache" / "brain") / "sha256.json") if sha256 else None
        for what, rel, want in files:
            p = models_dir / rel
            if not p.is_file():
                E(f"{what}: file not found: {p}")
            elif cache:
                if want in (None, "unverified"):
                    W(f"{what}: sha256 unverified (manifest has none)")
                elif (got := cache.digest(p)) != want:
                    E(f"{what}: sha256 mismatch: manifest {want} != file {got}")
    if not sha256:
        n = sum(1 for _, _, s in files if s in (None, "unverified"))
        if n:
            W(f"{n} artifact(s) have no sha256 yet (unverified)")
    return r
