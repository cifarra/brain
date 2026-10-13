# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "safetensors>=0.4"]
# ///
"""Extract a LoRA adapter from the weight difference between two Hugging Face checkpoints.

Given a base model dir and a fine-tuned (or edited, e.g. abliterated) model dir with the same
architecture, this compares every tensor, and for each changed 2-D weight W it factors the delta
D = W_tuned - W_base with a truncated SVD into a low-rank product B @ A (rank <= --max-rank).
The result is a PEFT-format LoRA dir that llama.cpp's convert_lora_to_gguf.py can convert.

    uv run tools/extract_adapter/extract.py --base <hf dir> --tuned <hf dir> --out <peft dir> \\
        --max-rank 1 --ignore-missing '^mtp\\.'

Rank selection per matrix (smallest wins, at least 1):
  * --max-rank k        hard cap;
  * --energy e          smallest rank whose singular values hold >= e of ||D||_F^2;
  * --noise-mult c      stop once the residual ||D - B A||_F is <= c * the expected norm of the
                        tuned checkpoint's own storage rounding (sqrt(sum ulp(W_tuned)^2 / 12)):
                        what is left is indistinguishable from bf16 rounding of the merged weights.
All matrices are then zero-padded to one shared rank R, because llama.cpp scales every LoRA
tensor by lora_alpha / rank(tensor) with a single global alpha. lora_alpha = R, so scale 1.0
applies exactly the extracted delta.

Fails (exit 2) on any difference it cannot represent as a LoRA: a changed tensor that is not a
2-D ".weight", a shape change, or a tensor missing on one side (unless --ignore-missing /
--ignore-extra match it). Pure dtype re-casts (tuned == round(base, tuned dtype)) are reported
as artefacts and skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VERSION = "0.1.0"

DTYPES = {  # safetensors dtype -> (numpy storage dtype, explicit mantissa bits or None)
    "BF16": (np.uint16, 7),
    "F16": (np.float16, 10),
    "F32": (np.float32, 23),
    "F64": (np.float64, 52),
    "I64": (np.int64, None),
    "I32": (np.int32, None),
    "I16": (np.int16, None),
    "I8": (np.int8, None),
    "U8": (np.uint8, None),
    "BOOL": (np.bool_, None),
}
FLOAT_DTYPES = {"BF16", "F16", "F32", "F64"}


# ---------------------------------------------------------------------------- safetensors reading
@dataclass
class TensorRef:
    name: str
    file: Path
    dtype: str
    shape: tuple[int, ...]
    start: int  # absolute byte offset in file
    end: int


class Checkpoint:
    """Reads tensors of a (possibly sharded) safetensors checkpoint via memmap, no torch needed."""

    def __init__(self, root: Path):
        self.root = root
        idx = root / "model.safetensors.index.json"
        if idx.exists():
            files = sorted(set(json.loads(idx.read_text())["weight_map"].values()))
        else:
            files = sorted(p.name for p in root.glob("*.safetensors"))
        if not files:
            raise SystemExit(f"no safetensors in {root}")
        self.files = [root / f for f in files]
        self.tensors: dict[str, TensorRef] = {}
        self.header_sha256: dict[str, str] = {}
        self._mm: dict[Path, np.memmap] = {}
        for f in self.files:
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                raw = fh.read(n)
            self.header_sha256[f.name] = hashlib.sha256(raw).hexdigest()
            hdr = json.loads(raw)
            hdr.pop("__metadata__", None)
            for name, h in hdr.items():
                if name in self.tensors:
                    raise SystemExit(f"duplicate tensor {name} in {root}")
                s, e = h["data_offsets"]
                self.tensors[name] = TensorRef(name, f, h["dtype"], tuple(h["shape"]), 8 + n + s, 8 + n + e)

    def _map(self, f: Path) -> np.memmap:
        if f not in self._mm:
            self._mm[f] = np.memmap(f, dtype=np.uint8, mode="r")
        return self._mm[f]

    def raw(self, name: str) -> np.ndarray:
        t = self.tensors[name]
        return self._map(t.file)[t.start:t.end]

    def f32(self, name: str) -> np.ndarray:
        t = self.tensors[name]
        return to_f32(self.raw(name), t.dtype, t.shape)


def to_f32(raw: np.ndarray, dtype: str, shape) -> np.ndarray:
    if dtype == "BF16":
        u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return u.view(np.float32).reshape(shape)
    if dtype not in DTYPES:
        raise SystemExit(f"unsupported dtype {dtype}")
    return np.frombuffer(raw, dtype=DTYPES[dtype][0]).astype(np.float32).reshape(shape)


def round_to(x: np.ndarray, dtype: str) -> np.ndarray:
    """Round f32 values to `dtype` (round-to-nearest-even) and return them as f32."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if dtype == "BF16":
        u = x.view(np.uint32)
        u = (u + (((u >> 16) & 1) + 0x7FFF)) & 0xFFFF0000
        return u.astype(np.uint32).view(np.float32)
    if dtype == "F16":
        return x.astype(np.float16).astype(np.float32)
    return x


def ulp(x: np.ndarray, mant_bits: int) -> np.ndarray:
    """Spacing of the storage format around each value (for the rounding-noise estimate)."""
    _, e = np.frexp(np.abs(x).astype(np.float64))  # |x| = m * 2^e, m in [0.5, 1)
    return np.ldexp(1.0, e - 1 - mant_bits)


def bytes_equal(a: np.ndarray, b: np.ndarray, chunk: int = 1 << 28) -> bool:
    if a.shape != b.shape:
        return False
    for i in range(0, a.shape[0], chunk):
        if not np.array_equal(a[i:i + chunk], b[i:i + chunk]):
            return False
    return True


# ---------------------------------------------------------------------------- SVD
def top_svd(D: np.ndarray, k: int, full: bool, seed: int = 0):
    """Top-k singular triplets of D (f32). Randomized range finder with power iterations unless
    `full` or k is a large fraction of min(m, n)."""
    m, n = D.shape
    if full or k + 16 >= min(m, n) // 4:
        U, s, Vt = np.linalg.svd(D, full_matrices=False)
        return U[:, :k], s[:k], Vt[:k], s
    rng = np.random.default_rng(seed)
    p = k + 16
    Q = np.linalg.qr(D @ rng.standard_normal((n, p), dtype=np.float32))[0]
    for _ in range(6):  # power iterations sharpen the spectrum (Halko et al. 2011, alg. 4.4)
        Q = np.linalg.qr(D.T @ Q)[0]
        Q = np.linalg.qr(D @ Q)[0]
    Ub, s, Vt = np.linalg.svd(Q.T @ D, full_matrices=False)
    U = Q @ Ub
    return U[:, :k], s[:k], Vt[:k], s


# ---------------------------------------------------------------------------- main
def hf_revision(root: Path) -> str | None:
    meta = root / ".cache" / "huggingface" / "download" / "config.json.metadata"
    if meta.exists():
        return meta.read_text().split("\n")[0].strip() or None
    return None


def hf_etags(root: Path, files: list[Path]) -> dict[str, str]:
    """hf download stores each LFS file's sha256 as its etag in .cache/.../<file>.metadata."""
    out = {}
    for f in files:
        meta = root / ".cache" / "huggingface" / "download" / f"{f.name}.metadata"
        if meta.exists():
            lines = meta.read_text().split("\n")
            if len(lines) > 1 and re.fullmatch(r"[0-9a-f]{64}", lines[1].strip()):
                out[f.name] = lines[1].strip()
    return out


def module_of(weight_name: str) -> str:
    assert weight_name.endswith(".weight")
    return weight_name[: -len(".weight")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True, help="base HF model dir (safetensors)")
    ap.add_argument("--tuned", type=Path, required=True, help="fine-tuned / edited HF model dir")
    ap.add_argument("--out", type=Path, help="output PEFT adapter dir (omit with --check-only)")
    ap.add_argument("--max-rank", type=int, default=16)
    ap.add_argument("--energy", type=float, default=0.999, help="captured-energy target per matrix (1.0 = off)")
    ap.add_argument("--noise-mult", type=float, default=1.1,
                    help="stop when residual <= c * expected storage-rounding noise (0 = off)")
    ap.add_argument("--full-svd", action="store_true", help="exact SVD instead of randomized")
    ap.add_argument("--ignore-missing", action="append", default=[], metavar="REGEX",
                    help="base tensors allowed to be absent from tuned (e.g. '^mtp\\.')")
    ap.add_argument("--ignore-extra", action="append", default=[], metavar="REGEX",
                    help="tuned tensors allowed to be absent from base")
    ap.add_argument("--rename-tuned", action="append", default=[], metavar="REGEX=REPL",
                    help="rewrite tuned tensor names before matching (repeatable)")
    ap.add_argument("--exclude", action="append", default=[r"embed_tokens\.weight$", r"^lm_head\.weight$"],
                    metavar="REGEX", help="2-D weights that may not become LoRA targets (changed -> error)")
    ap.add_argument("--base-model-id", default=None, help="base_model_name_or_path for adapter_config.json")
    ap.add_argument("--base-revision", default=None, help="default: read from <base>/.cache/huggingface")
    ap.add_argument("--source-id", default=None, help="tuned model id (provenance only)")
    ap.add_argument("--source-revision", default=None, help="default: read from <tuned>/.cache/huggingface")
    ap.add_argument("--analyze-projection", action="store_true",
                    help="for rank-1 deltas, fit D = -lambda * u u^T W_base (directional ablation) and report")
    ap.add_argument("--check-only", action="store_true", help="classify tensors, no SVD, no output")
    args = ap.parse_args()
    if not args.check_only and args.out is None:
        ap.error("--out is required unless --check-only")

    t0 = time.time()
    base, tuned = Checkpoint(args.base), Checkpoint(args.tuned)
    renames = [tuple(r.split("=", 1)) for r in args.rename_tuned]
    tmap: dict[str, str] = {}
    for tn in tuned.tensors:
        n = tn
        for pat, rep in renames:
            n = re.sub(pat, rep, n)
        tmap[n] = tn

    errors: list[str] = []
    artefacts: list[dict] = []
    identical: list[str] = []
    changed: list[str] = []

    for name in sorted(set(base.tensors) | set(tmap)):
        if name not in tmap:
            if any(re.search(p, name) for p in args.ignore_missing):
                artefacts.append({"tensor": name, "kind": "missing_in_tuned"})
            else:
                errors.append(f"{name}: missing in tuned")
            continue
        if name not in base.tensors:
            if any(re.search(p, name) for p in args.ignore_extra):
                artefacts.append({"tensor": name, "kind": "extra_in_tuned"})
            else:
                errors.append(f"{name}: not in base")
            continue
        b, t = base.tensors[name], tuned.tensors[tmap[name]]
        if b.shape != t.shape:
            errors.append(f"{name}: shape {b.shape} -> {t.shape}")
            continue
        if b.dtype == t.dtype:
            if bytes_equal(base.raw(name), tuned.raw(tmap[name])):
                identical.append(name)
                continue
        else:
            if b.dtype not in FLOAT_DTYPES or t.dtype not in FLOAT_DTYPES:
                errors.append(f"{name}: dtype {b.dtype} -> {t.dtype}")
                continue
            bv, tv = base.f32(name), tuned.f32(tmap[name])
            if np.array_equal(round_to(bv, t.dtype), tv):
                artefacts.append({"tensor": name, "kind": "dtype_cast", "from": b.dtype, "to": t.dtype,
                                  "note": "tuned == base rounded to tuned dtype; not a learned change"})
                continue
            if np.array_equal(round_to(tv, b.dtype), bv):
                artefacts.append({"tensor": name, "kind": "dtype_cast", "from": b.dtype, "to": t.dtype,
                                  "note": "base == tuned rounded to base dtype; not a learned change"})
                continue
        is_weight = name.endswith(".weight") and len(b.shape) == 2 and b.dtype in FLOAT_DTYPES
        if not is_weight or any(re.search(p, name) for p in args.exclude):
            errors.append(f"{name}: changed but not a LoRA-able 2-D weight (shape {b.shape}, {b.dtype}->{t.dtype})")
            continue
        changed.append(name)

    print(f"tensors: base={len(base.tensors)} tuned={len(tuned.tensors)} identical={len(identical)} "
          f"changed_2d={len(changed)} artefacts={len(artefacts)} errors={len(errors)} ({time.time() - t0:.0f}s)")
    for a in artefacts:
        print(f"  artefact {a['kind']:16s} {a['tensor']} {a.get('from', '')}{'->' if 'to' in a else ''}{a.get('to', '')}")
    if errors:
        for e in errors:
            print(f"  ERROR {e}", file=sys.stderr)
        print(f"FAILED: {len(errors)} difference(s) cannot be represented as a LoRA", file=sys.stderr)
        return 2
    if args.check_only:
        for c in changed:
            print(f"  changed {c} {base.tensors[c].shape}")
        return 0
    if not changed:
        print("FAILED: no changed 2-D weights", file=sys.stderr)
        return 2

    # ---- per-matrix SVD
    rows: list[dict] = []
    factors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for i, name in enumerate(changed):
        t1 = time.time()
        W0 = base.f32(name)
        W1 = tuned.f32(tmap[name])
        D = (W1 - W0).astype(np.float32)
        m, n = D.shape
        d_norm2 = float(np.sum(D.astype(np.float64) ** 2))
        mant = DTYPES[tuned.tensors[tmap[name]].dtype][1]
        # Frobenius norm of the rounding error expected from storing W_tuned in its dtype
        # (uniform error in +-ulp/2 per element -> variance ulp^2 / 12).
        rounding = float(np.sqrt(np.sum(ulp(W1, mant) ** 2) / 12.0)) if mant else 0.0
        kq = min(args.max_rank + 1, min(m, n))
        U, s, Vt, s_all = top_svd(D, kq, args.full_svd)
        cum = np.cumsum(s.astype(np.float64) ** 2) / d_norm2
        k_energy = int(np.searchsorted(cum, args.energy) + 1) if args.energy < 1.0 else kq
        # residual after k components: ||D - D_k||^2 = ||D||^2 - sum_{i<=k} sigma_i^2
        resid = np.sqrt(np.maximum(d_norm2 - np.cumsum(s.astype(np.float64) ** 2), 0.0))
        if args.noise_mult > 0 and rounding > 0:
            below = np.nonzero(resid <= args.noise_mult * rounding)[0]
            k_noise = int(below[0] + 1) if len(below) else kq
        else:
            k_noise = kq
        k = max(1, min(args.max_rank, k_energy, k_noise))
        sq = np.sqrt(s[:k])
        B = (U[:, :k] * sq).astype(np.float32)          # (out, k)
        A = (sq[:, None] * Vt[:k]).astype(np.float32)   # (k, in)
        R = D - B @ A
        r_norm2 = float(np.sum(R.astype(np.float64) ** 2))
        rel_err = float(np.sqrt(r_norm2 / d_norm2))
        # Re-merge check: does round(W_base + B A) reproduce the tuned checkpoint's stored values?
        remerge_exact = float(np.mean(round_to(W0 + B @ A, tuned.tensors[tmap[name]].dtype) == W1))
        row = {
            "tensor": name, "shape": [m, n], "rank": k,
            "energy": float(cum[k - 1]),
            "rel_err": rel_err,                                   # ||D - BA|| / ||D||
            "rel_err_vs_weight": float(np.sqrt(np.sum(R.astype(np.float64) ** 2) / np.sum(W1.astype(np.float64) ** 2))),
            "delta_rel_norm": float(math.sqrt(d_norm2 / np.sum(W0.astype(np.float64) ** 2))),  # ||D|| / ||W0||
            "sigma": [float(x) for x in s[: min(len(s), k + 3)]],
            "next_sigma_over_top": float(s[k] / s[0]) if k < len(s) else None,
            "rounding_noise_norm": rounding,
            "resid_over_rounding": float(math.sqrt(r_norm2) / rounding) if rounding else None,
            "remerge_exact_frac": remerge_exact,
            "frac_changed": float(np.mean(W1 != W0)),
            "k_energy": k_energy, "k_noise": k_noise,
        }
        if args.analyze_projection:
            u = U[:, 0].astype(np.float64)
            P = -np.outer(u, u @ W0.astype(np.float64))  # directional ablation along u: -u u^T W0
            lam = float(np.sum(P * D) / np.sum(P * P))
            row["proj_lambda"] = lam
            row["proj_fit_rel_err"] = float(np.linalg.norm(D - lam * P) / math.sqrt(d_norm2))
            row["_u"] = u.astype(np.float32)
        rows.append(row)
        factors[name] = (A, B)
        print(f"[{i + 1:3d}/{len(changed)}] {name:60s} {m}x{n} k={k} energy={row['energy']:.4f} "
              f"rel_err={rel_err:.4f} resid/rounding={row['resid_over_rounding'] or 0:.3f} remerge_exact={remerge_exact:.4f}"
              + (f" lambda={row['proj_lambda']:.3f} fit={row['proj_fit_rel_err']:.4f}" if args.analyze_projection else "")
              + f" ({time.time() - t1:.1f}s)", flush=True)

    R_all = max(r["rank"] for r in rows)
    if args.analyze_projection:
        # Is the residual direction shared between the matrices of one layer?
        by_layer: dict[str, list[dict]] = {}
        for r in rows:
            mm = re.search(r"layers\.(\d+)\.", r["tensor"])
            by_layer.setdefault(mm.group(1) if mm else "?", []).append(r)
        shared = {}
        for L, rs in sorted(by_layer.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else -1):
            if len(rs) > 1:
                shared[L] = float(abs(np.dot(rs[0]["_u"], rs[1]["_u"])))
        print("layer direction |cos| between matrices in the same layer:",
              {k: round(v, 4) for k, v in shared.items()})
        for r in rows:
            r.pop("_u", None)
    else:
        shared = None

    # ---- write PEFT dir
    from safetensors.numpy import save_file

    args.out.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for name, (A, B) in factors.items():
        k = A.shape[0]
        if k < R_all:  # zero-pad: llama.cpp uses one alpha for all tensors (scale = alpha / rank)
            A = np.vstack([A, np.zeros((R_all - k, A.shape[1]), np.float32)])
            B = np.hstack([B, np.zeros((B.shape[0], R_all - k), np.float32)])
        mod = module_of(name)
        tensors[f"base_model.model.{mod}.lora_A.weight"] = np.ascontiguousarray(A)
        tensors[f"base_model.model.{mod}.lora_B.weight"] = np.ascontiguousarray(B)
    save_file(tensors, str(args.out / "adapter_model.safetensors"), metadata={"format": "pt"})

    base_rev = args.base_revision or hf_revision(args.base)
    src_rev = args.source_revision or hf_revision(args.tuned)
    modules = sorted(module_of(n) for n in changed)
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": args.base_model_id or str(args.base),
        "revision": base_rev,
        "r": R_all,
        "lora_alpha": R_all,  # scale = alpha / r = 1 -> the full extracted delta
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "use_rslora": False,
        "use_dora": False,
        "init_lora_weights": True,
        "inference_mode": True,
        "modules_to_save": None,
        "layers_to_transform": None,
        "rank_pattern": {},
        "alpha_pattern": {},
        "target_modules": modules,  # exact module names: only the changed matrices
    }
    (args.out / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")

    suffixes = sorted({re.sub(r"^.*layers\.\d+\.", "", m) for m in modules})
    report = {
        "tool": f"tools/extract_adapter v{VERSION}",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "params": {"max_rank": args.max_rank, "energy": args.energy, "noise_mult": args.noise_mult,
                   "full_svd": args.full_svd, "ignore_missing": args.ignore_missing},
        "base": {"dir": str(args.base), "id": args.base_model_id, "revision": base_rev,
                 "index_sha256": hashlib.sha256((args.base / "model.safetensors.index.json").read_bytes()).hexdigest()
                 if (args.base / "model.safetensors.index.json").exists() else None,
                 "file_sha256": hf_etags(args.base, base.files), "header_sha256": base.header_sha256},
        "source": {"dir": str(args.tuned), "id": args.source_id, "revision": src_rev,
                   "file_sha256": hf_etags(args.tuned, tuned.files), "header_sha256": tuned.header_sha256},
        "counts": {"base_tensors": len(base.tensors), "tuned_tensors": len(tuned.tensors),
                   "identical": len(identical), "changed_2d": len(changed), "artefacts": len(artefacts)},
        "rank": R_all,
        "target_modules": modules,
        "target_module_suffixes": suffixes,
        "summary": {
            "rank_hist": {str(k): sum(1 for r in rows if r["rank"] == k) for k in sorted({r["rank"] for r in rows})},
            "energy_min": min(r["energy"] for r in rows), "energy_mean": float(np.mean([r["energy"] for r in rows])),
            "rel_err_max": max(r["rel_err"] for r in rows), "rel_err_mean": float(np.mean([r["rel_err"] for r in rows])),
            "rel_err_vs_weight_max": max(r["rel_err_vs_weight"] for r in rows),
            "resid_over_rounding_max": max((r["resid_over_rounding"] or 0) for r in rows),
            "remerge_exact_frac_min": min(r["remerge_exact_frac"] for r in rows),
        },
        "layer_direction_cos": shared,
        "artefacts": artefacts,
        "matrices": rows,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (args.out / "extraction_report.json").write_text(json.dumps(report, indent=2) + "\n")
    s = report["summary"]
    print(f"wrote {args.out}: {len(changed)} matrices, r={R_all}, suffixes={suffixes}")
    print(f"energy min/mean {s['energy_min']:.4f}/{s['energy_mean']:.4f}  rel_err max/mean "
          f"{s['rel_err_max']:.4f}/{s['rel_err_mean']:.4f}  resid/rounding max {s['resid_over_rounding_max']:.3f}  "
          f"remerge exact min {s['remerge_exact_frac_min']:.4f} "
          f"({report['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
