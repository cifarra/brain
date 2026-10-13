"""`brain serve | check | adapters list`."""

from __future__ import annotations

import argparse
import logging
import shlex
import sys
from pathlib import Path

from .config import load_settings
from .registry import RegistryError, load_registry, validate_profile
from .supervisor import build_argv


def _settings(args):
    overrides = {k: getattr(args, k, None) for k in ("profile", "registry", "models_dir")}
    ref = overrides.get("profile")
    if ref and ref.endswith((".yaml", ".yml")):  # a profile path implies its registry
        p = Path(ref)
        overrides["profile"] = p.stem
        if not overrides.get("registry") and p.parent.name == "profiles":
            overrides["registry"] = p.parent.parent
    return load_settings(**overrides)


def cmd_serve(args) -> int:
    import uvicorn

    from .app import create_app

    s = _settings(args)
    logging.basicConfig(level=s.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        app = create_app(s)
    except RegistryError as e:
        print("registry/profile invalid:\n  " + "\n  ".join(e.errors), file=sys.stderr)
        return 1
    uvicorn.run(app, host=args.host or s.host, port=args.port or s.port, log_level=s.log_level)
    return 0


def cmd_check(args) -> int:
    s = _settings(args)
    try:
        reg = load_registry(s.registry)
        profile = reg.profile(s.profile)
    except (RegistryError, KeyError) as e:
        print(f"ERROR {e}", file=sys.stderr)
        return 1
    print(f"registry {s.registry}: {len(reg.models)} models, {len(reg.adapters)} adapters, "
          f"{len(reg.profiles)} profiles, {len(reg.router.tasks)} router tasks")
    report = validate_profile(reg, profile, s.models_dir, check_files=not args.no_files,
                              sha256=args.sha256, cache_dir=s.cache_dir)
    for w in report.warnings:
        print(f"WARN  {w}")
    for e in report.errors:
        print(f"ERROR {e}")
    print(f"\nprofile {profile.id} (phase {profile.phase}); models dir {s.models_dir}")
    for i, (name, be) in enumerate(profile.backends.items()):
        try:
            argv = build_argv(reg, be, binary=s.llama_server, models_dir=s.models_dir,
                              host=s.backend_host, port=s.port_for(name, i))
            print(f"  {name}: {shlex.join(argv)}")
        except KeyError as e:
            print(f"  {name}: cannot render argv (missing {e})")
    print("\nOK" if report.ok else f"\nFAILED ({len(report.errors)} errors)")
    return 0 if report.ok else 1


def cmd_adapters_list(args) -> int:
    s = _settings(args)
    reg = load_registry(s.registry)
    profile = reg.profile(s.profile)
    rows = [("NAME", "KIND", "VER", "GATE", "SCALE", "ROLES", "LOADED ON", "FILE")]
    for a in reg.adapters.values():
        rows.append((
            a.name, a.kind, str(a.version), a.evals.gate,
            f"{a.scale.default} [{a.scale.min}-{a.scale.max}]", ",".join(a.roles),
            ",".join(reg.adapter_backends(profile, a.name)) or "-",
            a.artifact.file or f"(merged into {a.artifact.family})",
        ))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip())
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="brain", description="brain gateway (System 1 / System 2 over llama.cpp)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--registry", type=Path, help="registry dir (env BRAIN_REGISTRY, default ./registry)")
        p.add_argument("--models-dir", type=Path, help="models dir (env BRAIN_MODELS_DIR, default /models)")

    p = sub.add_parser("serve", help="run the gateway and its llama-server backends")
    p.add_argument("--profile", help="profile id or path (env BRAIN_PROFILE)")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    common(p)
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("check", help="validate the registry and a profile; print the backend argv")
    p.add_argument("profile", nargs="?", help="profile id or path to its YAML (env BRAIN_PROFILE)")
    p.add_argument("--sha256", action="store_true", help="verify file sha256 (slow; cached)")
    p.add_argument("--no-files", action="store_true", help="skip file-existence checks (e.g. on a laptop)")
    common(p)
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("adapters", help="adapter registry")
    asub = p.add_subparsers(dest="acmd", required=True)
    pl = asub.add_parser("list", help="list registered adapters")
    pl.add_argument("--profile", help="profile id or path (env BRAIN_PROFILE)")
    common(pl)
    pl.set_defaults(fn=cmd_adapters_list)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
