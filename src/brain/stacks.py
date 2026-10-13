"""Adapter stacks: resolution (role default ⊕ router task ⊕ request), validation, and the
mapping onto llama.cpp's per-request `lora` array.

Rev 5: an unapproved multi-adapter stack is allowed and flagged (`evaluated=False`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .registry import AdapterManifest, Profile, Registry, ResolvedRole, Task, compat_errors


class AdapterError(Exception):
    """400 invalid_adapter with a reason code from the spec table."""

    def __init__(self, reason: str, adapter: str | None, message: str):
        super().__init__(message)
        self.reason, self.adapter = reason, adapter


@dataclass
class StackContext:
    reg: Registry
    profile: Profile
    loaded: dict[str, set[str]]  # backend -> adapters usable on it right now
    approved: set[frozenset[str]] = field(default_factory=set)

    @classmethod
    def from_profile(cls, reg: Registry, profile: Profile, loaded: dict[str, set[str]] | None = None):
        if loaded is None:
            loaded = {b: set(reg.configured_adapters(profile, b)) for b in profile.backends}
        approved = {frozenset(s.adapters) for s in profile.approved_stacks}
        return cls(reg, profile, loaded, approved)


@dataclass
class Stack:
    adapters: dict[str, float]  # non-zero adapters only
    evaluated: bool


def parse_model(name: str) -> tuple[str, dict[str, float | None] | None, bool]:
    """`<role>(+<adapter>[@<scale>])*` -> (role, adapters or None if unqualified, replace).

    `+none` starts from an empty stack. A missing `@scale` means the manifest default (None).
    """
    role, *parts = name.split("+")
    if not role:
        raise ValueError(f"invalid model name {name!r}")
    if not parts:
        return role, None, False
    adapters: dict[str, float | None] = {}
    replace = False
    for p in parts:
        ad, at, scale = p.partition("@")
        if not ad or (at and not scale):
            raise ValueError(f"invalid model name {name!r}")
        if ad == "none" and not at:
            replace = True
            continue
        try:
            adapters[ad] = float(scale) if at else None
        except ValueError:
            raise ValueError(f"invalid scale {scale!r} in model name {name!r}") from None
    return role, adapters, replace


def _check_scale(name: str, v) -> None:
    if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)):
        raise AdapterError("scale_range", name, f"scale for {name!r} must be a finite number")


def check_adapter(ctx: StackContext, role: ResolvedRole, ad: AdapterManifest, scale: float) -> None:
    n = ad.name
    if role.name not in ad.roles:
        raise AdapterError("role", n, f"adapter {n!r} is not allowed on role {role.name!r} (roles: {ad.roles})")
    if n not in ctx.loaded.get(role.backend, set()):
        raise AdapterError("not_loaded", n, f"adapter {n!r} is not loaded on backend {role.backend!r}")
    model = ctx.reg.backend_model(ctx.profile, role.backend)
    if errs := compat_errors(ad, model):
        raise AdapterError("base_mismatch", n, f"adapter {n!r} does not match {model.id}: {'; '.join(errs)}")
    if not ad.scale.min <= scale <= ad.scale.max:
        raise AdapterError("scale_range", n, f"scale {scale} for {n!r} outside [{ad.scale.min}, {ad.scale.max}]")
    if ad.evals.gate == "failed":
        raise AdapterError("gate", n, f"adapter {n!r} failed its eval gate")


def check_stack_rules(ctx: StackContext, stack: dict[str, float]) -> None:
    ads = [ctx.reg.adapters[n] for n in stack]
    for a in ads:
        for b in ads:
            if a is not b and (a.conflicts_with(b) or b.conflicts_with(a)):
                raise AdapterError("incompatible", a.name, f"adapters {a.name!r} and {b.name!r} are incompatible")
    for a in ads:
        if missing := [r for r in a.compat.requires if r not in stack]:
            raise AdapterError("missing_requirement", a.name, f"adapter {a.name!r} requires {missing}")
    groups: dict[str, str] = {}
    for a in ads:
        if g := a.compat.exclusive_group:
            if g in groups:
                raise AdapterError(
                    "exclusive_group", a.name, f"{groups[g]!r} and {a.name!r} are both in exclusive group {g!r}"
                )
            groups[g] = a.name
    if len(stack) > ctx.profile.max_stack:
        raise AdapterError(
            "stack_too_large", None, f"{len(stack)} adapters exceed the profile's max_stack {ctx.profile.max_stack}"
        )


def is_evaluated(ctx: StackContext, stack: dict[str, float]) -> bool:
    if any(ctx.reg.adapters[n].evals.gate != "passed" for n in stack):
        return False
    return len(stack) <= 1 or frozenset(stack) in ctx.approved


def resolve(
    ctx: StackContext,
    role: ResolvedRole,
    x_adapters: dict[str, float | None] | None = None,
    replace: bool = False,
    task: Task | None = None,
) -> Stack:
    """Merge layers (later wins, scale 0 removes; replace starts empty), then validate."""
    merged: dict[str, float | None] = {}
    if not replace:
        merged.update(role.default_adapters)
        if task is not None:
            if task.x_adapters_mode == "replace":
                merged.clear()
            merged.update(task.x_adapters)
    merged.update(x_adapters or {})

    stack: dict[str, float] = {}
    for name, scale in merged.items():
        ad = ctx.reg.adapters.get(name)
        if ad is None:
            raise AdapterError("unknown", name, f"adapter {name!r} is not registered")
        _check_scale(name, scale)
        s = ad.scale.default if scale is None else float(scale)
        if s != 0:
            check_adapter(ctx, role, ad, s)
            stack[name] = s
    check_stack_rules(ctx, stack)
    return Stack(stack, is_evaluated(ctx, stack))


def lora_payload(id_map: dict[int, str | None], stack: dict[str, float]) -> list[dict]:
    """Explicit {id, scale} for EVERY adapter the backend has loaded, 0.0 when inactive.
    `id_map` is backend id -> adapter name (None for a file the registry does not know)."""
    return [{"id": i, "scale": float(stack.get(n, 0.0)) if n else 0.0} for i, n in sorted(id_map.items())]
