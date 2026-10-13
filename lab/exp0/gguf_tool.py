"""GGUF helper for Experiment 0 (uses gguf-py from the pinned llama.cpp, commit 4413a19).

Run inside the pinned full image (it ships gguf-py at /app/gguf-py and in site-packages):
  docker run --rm -v ~/srv/models:/models -v $PWD:/lab --entrypoint python3 \
      ghcr.io/ggml-org/llama.cpp:full-cuda-b11580 /lab/gguf_tool.py inspect /models/.../X.gguf
  ... /lab/gguf_tool.py strip IN.gguf OUT.gguf output.weight [more tensor names]

inspect: per-type byte totals, and the type of every non-block tensor (decision.*, dec.blk.0.*, output, embd).
strip:   rewrite IN without the named tensors (metadata and other tensors are copied unchanged).
"""
import sys
from collections import Counter

import gguf


def inspect(path):
    r = gguf.GGUFReader(path)
    by_type, n_type = Counter(), Counter()
    head_types = Counter()
    for t in r.tensors:
        by_type[t.tensor_type.name] += int(t.n_bytes)
        n_type[t.tensor_type.name] += 1
        if t.name.startswith(("decision.", "dec.blk.")) and len(t.shape) > 1 or t.name.endswith("scorer_out.weight"):
            head_types[t.tensor_type.name] += 1
        if not t.name.startswith(("blk.", "dec.blk.")) or t.name.startswith("dec.blk.0."):
            print(f"{t.name:44s} {t.tensor_type.name:6s} {int(t.n_bytes)/2**20:9.2f} MiB")
    print("tensor bytes by type (GiB):", {k: round(v / 2**30, 3) for k, v in by_type.items()})
    print("tensor counts by type:", dict(n_type))
    print("decision-head weight matrix types:", dict(head_types))
    print("total tensor GiB:", round(sum(by_type.values()) / 2**30, 3))


def strip(src, dst, names):
    drop = set(names)
    r = gguf.GGUFReader(src)
    arch = r.fields[gguf.Keys.General.ARCHITECTURE].contents()
    w = gguf.GGUFWriter(dst, arch=arch, endianess=r.endianess)
    align = r.fields.get(gguf.Keys.General.ALIGNMENT)
    if align is not None:
        w.data_alignment = align.contents()
    for field in r.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        sub = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(field.name, field.contents(), vt, sub_type=sub)
    kept = [t for t in r.tensors if t.name not in drop]
    missing = drop - {t.name for t in r.tensors}
    if missing:
        sys.exit(f"tensors not found: {missing}")
    for t in kept:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in kept:
        w.write_tensor_data(t.data, tensor_endianess=r.endianess)
    w.close()
    print(f"wrote {dst}: dropped {sorted(drop)}, kept {len(kept)} tensors")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "inspect":
        for p in sys.argv[2:]:
            print("==", p)
            inspect(p)
    elif cmd == "strip":
        strip(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        sys.exit(__doc__)
