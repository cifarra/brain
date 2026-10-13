"""Check a converted LoRA GGUF against its PEFT source (Qwen3.5 / qwen35 naming).

Prints the GGUF metadata, tensor count/types/names, and the max relative error of every
B @ A product in the GGUF vs the PEFT one. For linear_attn.out_proj (GGUF ssm_out) the converter
reorders the input columns from grouped to tiled V-head order; this check applies the same
permutation, so a small error proves the reorder was applied to lora_a.

Run inside the llama.cpp full image (has torch/numpy + gguf-py) with the pinned source mounted:
  docker run --rm --entrypoint python3 -v <llama.cpp src>:/src:ro -v <this dir>:/tool:ro -v ... \\
    ghcr.io/ggml-org/llama.cpp:full-cuda-b11580 /tool/verify_gguf.py <adapter.gguf> <peft dir> <base config.json>
"""
import json
import re
import sys

import numpy as np

sys.path.insert(0, "/src/gguf-py")
import gguf  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402

HF = {"ssm_out": "linear_attn.out_proj", "ffn_down": "mlp.down_proj", "attn_output": "self_attn.o_proj",
      "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
      "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "attn_gate": "linear_attn.in_proj_z",
      "ssm_alpha": "linear_attn.in_proj_a", "ssm_beta": "linear_attn.in_proj_b"}


def main(gguf_path: str, peft_dir: str, config_path: str) -> int:
    r = gguf.GGUFReader(gguf_path)
    kv = {}
    for k, f in r.fields.items():
        if not k.startswith("GGUF."):
            try:
                kv[k] = f.contents()
            except Exception:
                kv[k] = "?"
    print(json.dumps(kv, default=str))
    names = [t.name for t in r.tensors]
    print("n_tensors", len(names), "types", sorted({t.tensor_type.name for t in r.tensors}))
    kinds: dict[str, set] = {}
    for n in names:
        m = re.match(r"blk\.(\d+)\.(\w+)\.weight\.lora_([ab])$", n)
        if not m:
            print("unexpected tensor name", n)
            return 1
        kinds.setdefault(m.group(2), set()).add(int(m.group(1)))
    print({k: sorted(v) for k, v in kinds.items()})

    peft = load_file(f"{peft_dir}/adapter_model.safetensors")
    cfg = json.load(open(config_path))
    cfg = cfg.get("text_config", cfg)
    nk, nv, hv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"], cfg["linear_value_head_dim"]
    idx = np.arange(nv * hv).reshape(nk, nv // nk, hv).transpose(1, 0, 2).reshape(-1)  # grouped -> tiled
    T = {t.name: t for t in r.tensors}
    worst = 0.0
    for n in names:
        if not n.endswith("lora_a"):
            continue
        L, kind = re.match(r"blk\.(\d+)\.(\w+)\.weight", n).groups()
        ta, tb = T[n], T[n[:-1] + "b"]
        a = np.array(ta.data, dtype=np.float32).reshape([int(x) for x in reversed(ta.shape)])  # (r, in)
        b = np.array(tb.data, dtype=np.float32).reshape([int(x) for x in reversed(tb.shape)])  # (out, r)
        key = f"base_model.model.model.language_model.layers.{L}.{HF[kind]}"
        A, B = peft[key + ".lora_A.weight"], peft[key + ".lora_B.weight"]
        ref = B @ (A[:, idx] if kind == "ssm_out" and nk != nv else A)
        worst = max(worst, float(np.linalg.norm(b @ a - ref) / np.linalg.norm(ref)))
    print(f"max rel err of B@A, GGUF vs PEFT (V-head column reorder applied on ssm_out): {worst:.2e}")
    peft_n = sum(1 for k in peft if k.endswith("lora_A.weight"))
    ok = worst < 1e-2 and len(names) == 2 * peft_n and kv.get("general.type") == "adapter"
    print("OK" if ok else "MISMATCH", f"(PEFT matrices {peft_n}, GGUF tensors {len(names)})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
