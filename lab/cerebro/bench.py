#!/usr/bin/env python3
"""cerebro (M4, 16 GB) benchmark: one llama-server at a time, stdlib only (macOS python3.9).

  bench.py s1 --model M [--flags "..."] [--limit N]    Clef: calibration records + latency + memory
  bench.py s2 --model M [--mmproj F] [--lora F] [--flags "..."]   Qwen: decode/prefill/vision/thinking
  bench.py s1 --url http://127.0.0.1:8790 --name N     the same S1 records through a running gateway

Memory is reported three ways: llama.cpp's own buffer sizes (from its log), peak RSS of the
server process, and the peak rise of system wired+active+compressed pages (vm_stat).
Results go to results/<name>.json (summary) and results/<name>.calib.jsonl (S1 probabilities,
same format as lab/exp0/results/calib, so lab/exp0/analyze.py can compare them to BF16)."""

import argparse
import base64
import contextlib
import json
import os
import re
import shlex
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
(HERE / "results").mkdir(exist_ok=True)
SERVER = os.path.expanduser(os.environ.get("LLAMA_SERVER", "~/srv/llama.cpp/b11580/llama-server"))
PORT = 8799
URL = f"http://127.0.0.1:{PORT}"


def post(path, body, timeout=900):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"content-type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, time.perf_counter() - t0, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, time.perf_counter() - t0, e.read().decode(errors="replace")


def vm_used_bytes():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1))
    get = lambda k: int(re.search(rf"{k}:\s+(\d+)", out).group(1))
    return (get("Pages wired down") + get("Pages active") + get("Pages occupied by compressor")) * page


class Sampler(threading.Thread):
    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid, self.stop, self.rss, self.vm = pid, False, 0, 0

    def run(self):
        while not self.stop:
            o = subprocess.run(["ps", "-o", "rss=", "-p", str(self.pid)], capture_output=True, text=True).stdout.strip()
            if o:
                self.rss = max(self.rss, int(o) * 1024)
            self.vm = max(self.vm, vm_used_bytes())
            time.sleep(0.2)


class Server:
    def __init__(self, argv, log):
        self.argv, self.log_path = argv, log

    def __enter__(self):
        subprocess.run(["sh", "-c", f"lsof -ti tcp:{PORT} | xargs kill 2>/dev/null"], check=False)
        time.sleep(0.5)
        self.vm0 = vm_used_bytes()
        self.log = open(self.log_path, "w")
        t0 = time.perf_counter()
        self.p = subprocess.Popen(self.argv + ["--host", "127.0.0.1", "--port", str(PORT), "--no-webui", "-lv", "4"],
                                  stdout=self.log, stderr=subprocess.STDOUT)
        self.s = Sampler(self.p.pid)
        self.s.start()
        while True:
            if self.p.poll() is not None:
                raise SystemExit(f"server exited {self.p.returncode}; see {self.log_path}")
            try:
                with urllib.request.urlopen(URL + "/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            time.sleep(0.2)
        self.load_s = time.perf_counter() - t0
        return self

    def __exit__(self, *exc):
        self.s.stop = True
        self.p.terminate()
        try:
            self.p.wait(30)
        except subprocess.TimeoutExpired:
            self.p.kill()
        self.log.close()

    def memory(self):
        text = Path(self.log_path).read_text(errors="replace")
        bufs = {}
        for m in re.finditer(r"(\w+?)(?:_Mapped)? (model|KV|compute|RS|output) buffer size =\s+([\d.]+) MiB", text):
            key = f"{m.group(1)} {m.group(2)}"
            bufs[key] = round(bufs.get(key, 0) + float(m.group(3)), 1)
        for m in re.finditer(r"clip_ctx:.*?(\w+) backend|CLIP using (\w+) backend", text):
            bufs["clip_backend"] = m.group(1) or m.group(2)
        for m in re.finditer(r"recommendedMaxWorkingSetSize\s*=\s*([\d.]+) MB", text):
            bufs["metal_recommended_max_mb"] = float(m.group(1))
        return {"buffers_mib": bufs, "peak_rss_gib": round(self.s.rss / 2**30, 2),
                "peak_vm_rise_gib": round((self.s.vm - self.vm0) / 2**30, 2), "load_s": round(self.load_s, 1)}


def png(w, h):
    raw = b"".join(b"\x00" + b"".join(bytes(((x * 7) % 256, (y * 5) % 256, 120)) for x in range(w)) for y in range(h))
    ch = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + ch(b"IDAT", zlib.compress(raw)) + ch(b"IEND", b"")


def cmd_s1(a):
    recs = [json.loads(l) for l in open(HERE.parent / "exp0/data/clef_records.jsonl")]
    if a.limit:
        recs = recs[: a.limit]
    global URL
    calib = HERE / "results" / f"{a.name}.calib.jsonl"
    lat = []
    if a.url:  # a running gateway: no server of our own, so no memory numbers
        URL, server = a.url.rstrip("/"), contextlib.nullcontext()
    else:
        server = Server([SERVER, "-m", os.path.expanduser(a.model)] + shlex.split(a.flags), HERE / "results" / f"{a.name}.log")
    with server as srv, open(calib, "w") as f:
        for rec in recs:
            code, dt, body = post("/v1/systemone", rec["request"])
            line = {"id": rec["id"], "kind": rec["kind"], "code": code, "latency_s": dt}
            if code == 200 and "x_brain" in body:
                line["swap_ms"] = body["x_brain"].get("swap_ms")
            if code == 200:
                line["input_tokens"] = body["usage"]["input_tokens"]
                line["probs"] = {q: (x["probabilities"] if "probabilities" in x else {"true": x["noul"], "false": 1 - x["noul"]})
                                 for q, x in body["answers"].items()}
                lat.append((line["input_tokens"], dt))
            else:
                line["error"] = str(body)[:300]
            f.write(json.dumps(line) + "\n")
            print(rec["id"], code, f"{dt:.2f}s", line.get("input_tokens"), flush=True)
        mem = srv.memory() if srv else {}
    lat.sort()
    buckets = {"<500": [], "500-1k": [], "1k-2k": [], "2k-4k": []}
    for n, dt in lat:
        buckets["<500" if n < 500 else "500-1k" if n < 1000 else "1k-2k" if n < 2000 else "2k-4k"].append(dt)
    med = lambda xs: round(sorted(xs)[len(xs) // 2], 2) if xs else None
    tps = [n / dt for n, dt in lat]
    out = {"name": a.name, "model": a.model, "flags": a.flags, **mem, "n_ok": len(lat), "n": len(recs),
           "latency_p50_s_by_tokens": {k: med(v) for k, v in buckets.items()},
           "prefill_tok_s_median": round(sorted(tps)[len(tps) // 2], 1) if tps else None}
    (HERE / "results" / f"{a.name}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def chat(body):
    code, dt, r = post("/v1/chat/completions", body)
    if code != 200:
        return {"code": code, "error": str(r)[:300]}
    t = r.get("timings", {})
    return {"code": code, "wall_s": round(dt, 2), "prompt_n": t.get("prompt_n"), "prompt_tok_s": round(t.get("prompt_per_second", 0), 1),
            "gen_n": t.get("predicted_n"), "gen_tok_s": round(t.get("predicted_per_second", 0), 1),
            "content": (r["choices"][0]["message"].get("content") or "")[:120]}


def cmd_s2(a):
    argv = [SERVER, "-m", os.path.expanduser(a.model)] + shlex.split(a.flags)
    if a.mmproj:
        argv += ["--mmproj", os.path.expanduser(a.mmproj)]
    if a.lora:
        argv += ["--lora", os.path.expanduser(a.lora), "--lora-init-without-apply", "--cache-ram", "0"]
    lora = lambda s: {"lora": [{"id": 0, "scale": s}]} if a.lora else {}
    off = {"chat_template_kwargs": {"enable_thinking": False}, "temperature": 0, "seed": 42}
    long_prompt = " ".join(["The quick brown fox jumps over the lazy dog near the riverbank at dawn."] * 140)
    res = {}
    with Server(argv, HERE / "results" / f"{a.name}.log") as srv:
        chat({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8, **off, **lora(1)})  # warm-up
        for s in ([0.0, 1.0] if a.lora else [None]):
            key = f"scale{s}" if s is not None else "nolora"
            res[f"decode_{key}"] = chat({"messages": [{"role": "user", "content": "Write a 300-word story about a lighthouse keeper."}],
                                         "max_tokens": 256, "ignore_eos": True, **off, **lora(s or 0)})
        res["prefill_2k"] = chat({"messages": [{"role": "user", "content": long_prompt + "\nSummarize in one line."}],
                                  "max_tokens": 16, **off, **lora(1)})
        res["thinking"] = chat({"messages": [{"role": "user", "content": "A bat and a ball cost $1.10; the bat costs $1 more than the ball. Price of the ball?"}],
                                "max_tokens": 1024, "temperature": 0, "seed": 42, **lora(1)})
        if a.mmproj:
            img = "data:image/png;base64," + base64.b64encode(png(448, 448)).decode()
            res["vision_448"] = chat({"messages": [{"role": "user", "content": [
                {"type": "text", "text": "Describe this image in one sentence."},
                {"type": "image_url", "image_url": {"url": img}}]}], "max_tokens": 40, **off, **lora(1)})
        mem = srv.memory()
    out = {"name": a.name, "model": a.model, "flags": a.flags, "lora": a.lora, **mem, "results": res}
    (HERE / "results" / f"{a.name}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("s1", "s2"):
        p = sub.add_parser(n)
        p.add_argument("--name", required=True)
        p.add_argument("--model", required=(n == "s2"))
        p.add_argument("--flags", default="")
        if n == "s1":
            p.add_argument("--limit", type=int, default=0)
            p.add_argument("--url", help="drive a running gateway instead of starting llama-server")
        else:
            p.add_argument("--mmproj")
            p.add_argument("--lora")
    a = ap.parse_args()
    (cmd_s1 if a.cmd == "s1" else cmd_s2)(a)


if __name__ == "__main__":
    main()
