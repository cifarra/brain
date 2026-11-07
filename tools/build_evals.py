#!/usr/bin/env python3
"""Build the eval suites the web UI runs (registry/evals/*.json). Stdlib only.

  clef-calib.json  S1: the Experiment 0 calibration records with BF16 reference probabilities;
                   the UI applies the same gate as lab/exp0/analyze.py calib_compare.
  s2-smoke.json    S2: small checkable tasks (arithmetic, reasoning, translation, JSON, vision,
                   long-context recall) with pass/fail checks and timing.

Suite format: {name, title, description, endpoint: systemone|chat, gate?, cases: [{id, kind,
request, ref? | check?}]}. A chat check is {contains: [..]} (all, case-insensitive),
{any: [..]}, {regex: ".."} or {json: {key: value}}."""

import base64
import json
import struct
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP0 = ROOT / "lab/exp0"
OUT = ROOT / "registry/evals"
GATE = {"top1": 0.98, "p99": 0.03, "max": 0.10, "tie": 0.02}  # == lab/exp0/analyze.py


def load(p):
    return [json.loads(line) for line in open(p) if line.strip()]


def clef_calib():
    ref = {r["id"]: r for r in load(EXP0 / "results/calib/BF16.jsonl") if r.get("code") == 200}
    cases = []
    for rec in load(EXP0 / "data/clef_records.jsonl"):
        if rec["id"] in ref:
            r = ref[rec["id"]]
            cases.append({"id": rec["id"], "kind": rec["kind"], "tokens": r.get("input_tokens"),
                          "request": rec["request"], "ref": r["probs"]})
    return {"name": "clef-calib", "title": "S1 calibration vs BF16", "endpoint": "systemone", "model": "system1",
            "description": "Experiment 0 decision records; probabilities compared to the BF16 Clef reference. "
                           "Gate: top-1 agreement (ties excluded) ≥ 98%, p99 |Δp| ≤ 0.03, max |Δp| ≤ 0.10.",
            "gate": GATE, "cases": cases}


def png(w, h, pixel):
    raw = b"".join(b"\x00" + b"".join(bytes(pixel(x, y)) for x in range(w)) for y in range(h))
    ch = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    data = (b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + ch(b"IDAT", zlib.compress(raw, 9)) + ch(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(data).decode()


def user(text, image=None):
    if image is None:
        return [{"role": "user", "content": text}]
    return [{"role": "user", "content": [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": image}}]}]


def s2_smoke():
    fast, think = {"model": "system2-fast", "temperature": 0, "seed": 42}, {"model": "system2", "temperature": 0, "seed": 42}
    halves = png(256, 256, lambda x, y: (220, 30, 30) if x < 128 else (30, 60, 220))
    filler = ("Ledger note: the warehouse in sector {i} was inspected and found in order; "
              "pallets were counted, labels checked and the loading dock swept. ")
    hay = "".join(filler.format(i=i) for i in range(120))
    cut = len(hay) * 2 // 3
    hay = hay[:cut] + " The vault access code is 4817. " + hay[cut:]
    cases = [
        ("arith", "math", {**fast, "max_tokens": 16, "messages": user("What is 17 multiplied by 23? Reply with the number only.")},
         {"contains": ["391"]}),
        ("capital", "fact", {**fast, "max_tokens": 16, "messages": user("What is the capital of Australia? One word.")},
         {"contains": ["canberra"]}),
        ("pong", "instruct", {**fast, "max_tokens": 8, "messages": user("Reply with exactly the word pong and nothing else.")},
         {"regex": r"^\W*pong\W*$"}),
        ("translate", "translate", {**fast, "max_tokens": 48, "messages": user(
            "Translate to Spanish: 'The meeting has been moved to Thursday afternoon.'")}, {"contains": ["jueves"]}),
        ("json", "json", {**fast, "max_tokens": 64, "response_format": {"type": "json_object"}, "messages": user(
            "Extract the person's name and age as JSON with keys name and age: 'Maria is 34 years old and lives in Lima.'")},
         {"json": {"name": "Maria", "age": 34}}),
        ("bat-ball", "reasoning", {**think, "max_tokens": 2048, "messages": user(
            "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much is the ball?")},
         {"any": ["0.05", "5 cents", "five cents"]}),
        ("vision-colors", "vision", {**fast, "max_tokens": 24, "messages": user(
            "This image has two halves. Name the color of the left half, then the right half.", halves)},
         {"contains": ["red", "blue"]}),
        ("needle-2k", "long-context", {**fast, "max_tokens": 16, "messages": user(
            hay + "\n\nWhat is the vault access code? Reply with the number only.")}, {"contains": ["4817"]}),
    ]
    return {"name": "s2-smoke", "title": "S2 smoke", "endpoint": "chat",
            "description": "Short checkable S2 tasks: arithmetic, facts, instruction following, translation, "
                           "JSON mode, thinking, vision and 2k-token recall. Pass/fail plus speed.",
            "cases": [{"id": i, "kind": k, "request": r, "check": c} for i, k, r, c in cases]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for suite in (clef_calib(), s2_smoke()):
        p = OUT / f"{suite['name']}.json"
        p.write_text(json.dumps(suite, separators=(",", ":")) + "\n")
        print(f"{p.relative_to(ROOT)}: {len(suite['cases'])} cases, {p.stat().st_size // 1024} KiB")


if __name__ == "__main__":
    main()
