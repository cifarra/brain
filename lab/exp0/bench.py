# /// script
# requires-python = ">=3.11"
# dependencies = ["nvidia-ml-py>=12", "requests>=2.31"]
# ///
"""Experiment 0 harness: start pinned llama-server containers, sample per-PID VRAM, run workloads.

Runs on ranger's host (not in a container): `uv run bench.py <cmd> ...`. Every configuration takes the
shared GPU lock (~/srv/brain-lab/gpu.lock, flock(2), same as `flock(1)`) only while it runs.

Commands (one JSONL line per configuration is appended to --out):
  s1     Clef server: load time (cold + warm), steady/peak VRAM, systemone latency per size bucket,
         optional --calib to dump per-record probabilities (results/calib/<tag>.jsonl).
  s2     Qwen server: steady/peak VRAM over a short chat, an image (if mmproj) and a long prefill
         near the ctx limit + decode; pp/tg t/s from server timings. --np 2 runs two long prompts at once.
  s2q    Qwen next-token distributions (n_probs) under teacher forcing on data/s2_prompts.jsonl.
  pair   S1 + S2 containers at once; S1 and S2 workloads run concurrently; combined per-PID peak.

VRAM: NVML (the same counters nvidia-smi --query-compute-apps reads), per host PID of the container's
processes (docker top), sampled every --interval seconds (default 0.1).
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import random
import statistics
import struct
import subprocess
import threading
import time
import zlib
from pathlib import Path

import pynvml
import requests

IMAGE = "ghcr.io/ggml-org/llama.cpp:server-cuda-b11580@sha256:c4348afb91499b10fad12f92626c7a8ca8b69d6475d10c793b810cb61237ce00"
BUILD = "b11580 (4413a1924)"
HOME = Path.home()
MODELS = HOME / "srv/models"
LAB = HOME / "srv/brain-lab"
WORK = LAB / "exp0"
LOCK = LAB / "gpu.lock"
HERE = Path(__file__).resolve().parent
MiB = 2**20


# ---------------------------------------------------------------- utilities
def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


_sha_cache = {}


def sha256_file(p):
    """sha256 of a model file, cached next to it in WORK/sha256.json (keyed by path+size+mtime)."""
    p = Path(p)
    st = p.stat()
    key = f"{p}:{st.st_size}:{int(st.st_mtime)}"
    cache_f = WORK / "sha256.json"
    if not _sha_cache and cache_f.exists():
        _sha_cache.update(json.loads(cache_f.read_text()))
    if key not in _sha_cache:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(16 * MiB), b""):
                h.update(chunk)
        _sha_cache[key] = h.hexdigest()
        cache_f.write_text(json.dumps(_sha_cache, indent=0))
    return _sha_cache[key]


def evict_page_cache(p):
    """Drop a file from the page cache (POSIX_FADV_DONTNEED, no root needed) for a cold-load measurement."""
    fd = os.open(p, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


@contextlib.contextmanager
def gpu_lock():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK, "a") as f:
        t0 = time.time()
        fcntl.flock(f, fcntl.LOCK_EX)
        waited = time.time() - t0
        if waited > 1:
            print(f"[lock] waited {waited:.0f}s for {LOCK}", flush=True)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ---------------------------------------------------------------- NVML sampling
class Gpu:
    def __init__(self):
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(0)

    def mem(self):
        m = pynvml.nvmlDeviceGetMemoryInfo(self.h)
        return {"total_mib": m.total // MiB, "used_mib": m.used // MiB, "free_mib": m.free // MiB}

    def procs(self):
        out = {}
        for p in pynvml.nvmlDeviceGetComputeRunningProcesses(self.h):
            if p.usedGpuMemory is not None:
                out[p.pid] = p.usedGpuMemory // MiB
        return out


class Sampler(threading.Thread):
    """Samples per-group VRAM (sum over the group's host PIDs) every `interval` seconds.

    groups: {name: Server}. Phases ('load', 'steady', 'work', ...) are tagged by set_phase().
    """

    def __init__(self, gpu, groups, interval=0.1):
        super().__init__(daemon=True)
        self.gpu, self.groups, self.interval = gpu, groups, interval
        self.phase = "load"
        self.samples = []  # (t, phase, {group: mib}, device_used)
        self._halt = threading.Event()

    def set_phase(self, ph):
        self.phase = ph

    def run(self):
        while not self._halt.is_set():
            t = time.time()
            try:
                procs = self.gpu.procs()
                dev = self.gpu.mem()["used_mib"]
            except pynvml.NVMLError:
                time.sleep(self.interval)
                continue
            vals = {}
            for name, srv in self.groups.items():
                pids = srv.host_pids()
                vals[name] = sum(procs.get(pid, 0) for pid in pids)
            self.samples.append((t, self.phase, vals, dev))
            time.sleep(max(0.0, self.interval - (time.time() - t)))

    def stop(self):
        self._halt.set()
        self.join(timeout=5)

    def stats(self, phases=None, group=None):
        """max/median of a group (or the sum of all groups) over the given phases."""
        xs = []
        for _, ph, vals, _ in self.samples:
            if phases and ph not in phases:
                continue
            xs.append(vals.get(group, 0) if group else sum(vals.values()))
        if not xs:
            return {"max_mib": None, "median_mib": None, "n": 0}
        return {"max_mib": max(xs), "median_mib": statistics.median(xs), "n": len(xs)}


# ---------------------------------------------------------------- server container
class Server:
    def __init__(self, name, port, args, tag):
        self.name, self.port, self.args, self.tag = name, port, args, tag
        self._pids = []
        self._pids_t = 0
        self.url = f"http://127.0.0.1:{port}"

    def start(self):
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
        cmd = ["docker", "run", "-d", "--name", self.name, "--gpus", "all",
               "-v", f"{MODELS}:/models:ro", "-p", f"127.0.0.1:{self.port}:8080", IMAGE,
               *self.args, "--host", "0.0.0.0", "--port", "8080"]
        self.cmd = cmd
        self.t_start = time.time()
        subprocess.run(cmd, check=True, capture_output=True)

    def host_pids(self):
        # refresh at most every 2 s; llama-server is one process but be generic
        if time.time() - self._pids_t > 2 or not self._pids:
            r = subprocess.run(["docker", "top", self.name, "-eo", "pid"], capture_output=True, text=True)
            self._pids = [int(x) for x in r.stdout.split()[1:] if x.isdigit()]
            self._pids_t = time.time()
        return self._pids

    def running(self):
        r = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", self.name], capture_output=True, text=True)
        return r.stdout.strip() == "true"

    def wait_healthy(self, timeout=900):
        while time.time() - self.t_start < timeout:
            try:
                r = requests.get(self.url + "/health", timeout=2)
                if r.status_code == 200:
                    self.load_s = time.time() - self.t_start
                    return True
            except requests.RequestException:
                pass
            if not self.running():
                return False
            time.sleep(0.05)
        return False

    def logs(self):
        r = subprocess.run(["docker", "logs", self.name], capture_output=True, text=True)
        return r.stdout + r.stderr

    def stop(self, save_log=True):
        if save_log:
            d = WORK / "logs"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{self.tag}.log").write_text(self.logs())
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)


def log_facts(text):
    """Pull a few facts from the llama-server log."""
    facts = {}
    for line in text.splitlines():
        low = line.lower()
        if "flash_attn" in low and "=" in line and "flash_attn" not in facts:
            facts["flash_attn"] = line.split("=")[-1].strip()
        if "offloaded" in low and "layers to gpu" in low:
            facts["offloaded"] = line.split(":", 1)[-1].strip()
        if "cuda0 model buffer size" in low:
            facts.setdefault("model_buffer", []).append(line.split("=")[-1].strip())
        if "cuda0 compute buffer size" in low:
            facts.setdefault("compute_buffer", []).append(line.split("=")[-1].strip())
        if "cuda0 kv buffer size" in low or ("cuda0" in low and "rs buffer size" in low):
            facts.setdefault("kv_rs_buffer", []).append(line.split(":", 1)[-1].strip())
        if "out of memory" in low or "failed to allocate" in low:
            facts.setdefault("errors", []).append(line.strip()[:200])
    return facts


# ---------------------------------------------------------------- S1 (clef)
def s1_args(a):
    m = f"/models/{a.model}"
    args = ["-m", m, "-c", str(a.ctx), "-b", str(a.ctx), "-ub", str(a.ctx), "-np", "1",
            "-ngl", str(a.ngl), "-fit", "off", "-fa", a.fa, "--no-webui"]
    if a.output_cpu:
        args += ["-ot", r"^output\.weight$=CPU"]
    if a.mmproj:
        args += ["--mmproj", f"/models/{a.mmproj}"]
    args += a.extra
    return args


def systemone(url, req, timeout=600):
    t0 = time.time()
    r = requests.post(url + "/v1/systemone", json=req, timeout=timeout)
    dt = time.time() - t0
    try:
        body = r.json()
    except ValueError:
        body = {"raw": r.text[:300]}
    return r.status_code, dt, body


def run_s1_workload(url, workload, passes, sampler, out):
    """First request (cold after load), then `passes` warm passes per bucket."""
    res = {"first_request": None, "buckets": {}}
    sampler.set_phase("work")
    first = True
    for bucket in ("500", "2k", "6k"):
        reqs = [w for w in workload if w["bucket"] == bucket]
        lat, toks, errs = [], [], []
        for p in range(passes + 1):
            for w in reqs:
                code, dt, body = systemone(url, w["request"])
                if code != 200:
                    errs.append({"id": w["id"], "code": code, "err": str(body.get("error", body))[:200]})
                    continue
                if first:
                    res["first_request"] = {"id": w["id"], "latency_s": dt, "input_tokens": body["usage"]["input_tokens"]}
                    first = False
                    continue
                if p == 0:
                    toks.append(body["usage"]["input_tokens"])
                    continue  # pass 0 = per-request warmup (graph shapes vary with prompt size)
                lat.append(dt)
            if errs and not lat and p == 0 and len(errs) == len(reqs):
                break
        res["buckets"][bucket] = {
            "n": len(lat), "p50_s": pct(lat, 0.5), "p95_s": pct(lat, 0.95), "max_s": max(lat) if lat else None,
            "input_tokens_mean": statistics.mean(toks) if toks else None,
            "input_tokens_max": max(toks) if toks else None,
            "errors": errs[:3], "n_errors": len(errs),
        }
    return res


def run_calib(url, records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with open(path, "w") as f:
        for rec in records:
            code, dt, body = systemone(url, rec["request"])
            line = {"id": rec["id"], "kind": rec["kind"], "code": code, "latency_s": dt}
            if code == 200:
                n_ok += 1
                line["input_tokens"] = body["usage"]["input_tokens"]
                ans = {}
                for qid, a in body["answers"].items():
                    if "probabilities" in a:
                        ans[qid] = a["probabilities"]
                    else:
                        ans[qid] = {"true": a["noul"], "false": 1.0 - a["noul"]}
                line["probs"] = ans
            else:
                line["error"] = str(body)[:300]
            f.write(json.dumps(line) + "\n")
    return n_ok


def cmd_s1(a):
    workload = [json.loads(l) for l in open(HERE / "data/s1_workload.jsonl")]
    model_path = MODELS / a.model
    tag = a.tag or f"s1-{Path(a.model).stem}-c{a.ctx}" + ("-ocpu" if a.output_cpu else "") + (f"-ngl{a.ngl}" if a.ngl != "all" else "")
    rec = {"kind": "s1", "tag": tag, "time": now(), "build": BUILD, "model": a.model, "model_bytes": model_path.stat().st_size,
           "model_sha256": sha256_file(model_path), "ctx": a.ctx, "ub": a.ctx, "b": a.ctx, "ngl": a.ngl,
           "output_cpu": a.output_cpu, "mmproj": a.mmproj, "fa": a.fa, "extra": a.extra}
    gpu = Gpu()
    with gpu_lock():
        rec["free_mib_at_start"] = gpu.mem()["free_mib"]
        rec["device_used_mib_at_start"] = gpu.mem()["used_mib"]
        srv = Server("exp0-s1", a.port, s1_args(a), tag)
        # cold load (page cache evicted), then a warm load used for the measurements
        loads = {}
        if a.cold:
            evict_page_cache(model_path)
            srv.start()
            ok = srv.wait_healthy()
            loads["cold_s"] = srv.load_s if ok else None
            if not ok:
                rec["error"] = "cold load failed"
                rec["log_facts"] = log_facts(srv.logs())
                srv.stop()
                append_jsonl(a.out, rec)
                print(json.dumps(rec)[:1000])
                return
            srv.stop(save_log=False)
        srv.start()
        sampler = Sampler(gpu, {"s1": srv}, a.interval)
        sampler.start()
        ok = srv.wait_healthy()
        loads["warm_s"] = srv.load_s if ok else None
        rec["load"] = loads
        if not ok:
            sampler.stop()
            rec["error"] = "load failed"
            rec["log_facts"] = log_facts(srv.logs())
            srv.stop()
            append_jsonl(a.out, rec)
            print(json.dumps(rec)[:1500])
            return
        sampler.set_phase("steady")
        time.sleep(2.0)
        try:
            if not a.no_workload:
                rec["workload"] = run_s1_workload(srv.url, workload, a.passes, sampler, a.out)
            if a.calib:
                sampler.set_phase("calib")
                recs = [json.loads(l) for l in open(HERE / "data/clef_records.jsonl")]
                if a.calib_limit:
                    recs = recs[: a.calib_limit]
                cpath = Path(a.out).parent / "calib" / f"{a.calib_name or tag}.jsonl"
                t0 = time.time()
                rec["calib"] = {"path": str(cpath), "n_ok": run_calib(srv.url, recs, cpath), "n": len(recs), "seconds": time.time() - t0}
        finally:
            sampler.set_phase("post")
            time.sleep(0.5)
            sampler.stop()
            rec["vram"] = {
                "load_peak_mib": sampler.stats(["load"])["max_mib"],
                "steady_mib": sampler.stats(["steady"])["median_mib"],
                "work_peak_mib": sampler.stats(["work", "calib"])["max_mib"],
                "peak_mib": sampler.stats()["max_mib"],
                "samples": len(sampler.samples),
            }
            rec["log_facts"] = log_facts(srv.logs())
            srv.stop()
    append_jsonl(a.out, rec)
    print(json.dumps({k: rec[k] for k in rec if k not in ("workload",)})[:1500])
    if "workload" in rec:
        for b, v in rec["workload"]["buckets"].items():
            print(f"  {b}: n={v['n']} p50={v['p50_s']} p95={v['p95_s']} tok={v['input_tokens_mean']} err={v['n_errors']}")
        print("  first:", rec["workload"]["first_request"])


# ---------------------------------------------------------------- S2 (qwen)
def s2_args(a):
    ctx_total = a.ctx * a.np
    args = ["-m", f"/models/{a.model}", "-c", str(ctx_total), "-np", str(a.np), "-ub", str(a.ub), "-b", str(max(2048, a.ub)),
            "-fa", "on", "-ctk", a.kv, "-ctv", a.kv, "-ngl", "all", "-fit", "off", "--jinja",
            "--reasoning-format", "deepseek", "--ctx-checkpoints", str(a.checkpoints), "--no-webui", "--metrics"]
    if a.np > 1:
        args += ["--no-kv-unified"]
    if a.mmproj:
        args += ["--mmproj", f"/models/{a.mmproj}"]
        if getattr(a, "mmproj_cpu", False):
            args += ["--no-mmproj-offload"]
    else:
        args += ["--no-mmproj"]
    for l in a.lora:
        args += ["--lora", f"/models/{l}"]
    if a.lora:
        args += ["--lora-init-without-apply"]
    args += a.extra
    return args


def png_bytes(w, h, seed=0):
    """A synthetic RGB PNG (colored blocks) without PIL."""
    rng = random.Random(seed)
    blocks = [(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(16)]
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        for x in range(w):
            raw += bytes(blocks[(y * 4 // h) * 4 + (x * 4 // w)])
    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b"")


_long_text = None


def long_text():
    global _long_text
    if _long_text is None:
        recs = [json.loads(l) for l in open(HERE / "data/clef_records.jsonl")]
        _long_text = "\n\n".join(r["request"]["state"] for r in recs)
    return _long_text


def tokenize(url, text):
    r = requests.post(url + "/tokenize", json={"content": text, "add_special": False}, timeout=600)
    r.raise_for_status()
    return r.json()["tokens"]


def lora_field(a, scale=1.0):
    return [{"id": i, "scale": scale} for i in range(len(a.lora))] if a.lora else None


def completion(url, prompt, n_predict, extra=None, timeout=1800):
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0, "cache_prompt": False, "ignore_eos": True}
    body.update(extra or {})
    t0 = time.time()
    r = requests.post(url + "/completion", json=body, timeout=timeout)
    dt = time.time() - t0
    j = r.json()
    if r.status_code != 200:
        raise RuntimeError(f"/completion {r.status_code}: {str(j)[:300]}")
    return j, dt


def run_s2_workload(url, a, sampler, n_long=1):
    res = {}
    lf = {"lora": lora_field(a)} if a.lora else {}
    sampler.set_phase("work")
    # 1) short chat (thinking off) -> tg t/s at short context
    t0 = time.time()
    r = requests.post(url + "/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Write a 250-word story about a lighthouse keeper."}],
        "max_tokens": 256, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}, **lf}, timeout=600)
    j = r.json()
    res["short"] = {"code": r.status_code, "latency_s": time.time() - t0, "timings": j.get("timings"), "usage": j.get("usage")}
    # 2) image (if mmproj) -> vision sanity + encoder peak
    if a.mmproj:
        img = "data:image/png;base64," + __import__("base64").b64encode(png_bytes(768, 768)).decode()
        t0 = time.time()
        r = requests.post(url + "/v1/chat/completions", json={
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": img}},
                                                      {"type": "text", "text": "Describe this image in one sentence. How many colored blocks are there roughly?"}]}],
            "max_tokens": 64, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}, **lf}, timeout=600)
        j = r.json()
        content = (j.get("choices") or [{}])[0].get("message", {}).get("content")
        res["image"] = {"code": r.status_code, "latency_s": time.time() - t0, "timings": j.get("timings"),
                        "usage": j.get("usage"), "answer": (content or str(j))[:200]}
    # 3) long prefill near the ctx limit + decode
    n_pred = 128
    target = a.ctx - n_pred - 64
    toks = tokenize(url, long_text())
    while len(toks) < target:
        toks = toks + toks
    prompt = toks[:target]
    out = []
    def one(i):
        p = prompt[i * 7:] + prompt[: i * 7]  # make concurrent prompts differ
        j, dt = completion(url, p, n_pred, {"id_slot": i if n_long > 1 else -1, **lf})
        out.append({"slot": i, "latency_s": dt, "timings": j.get("timings"), "tokens_evaluated": j.get("tokens_evaluated")})
    ths = [threading.Thread(target=one, args=(i,)) for i in range(n_long)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    res["long"] = sorted(out, key=lambda x: x["slot"])
    res["long_prompt_tokens"] = target
    return res


def cmd_s2(a):
    model_path = MODELS / a.model
    lora_tag = "-abl" if a.lora else ""
    tag = a.tag or f"s2-{Path(a.model).stem}-c{a.ctx}-{a.kv}-np{a.np}" + ("" if a.mmproj else "-nommproj") + ("-mmcpu" if a.mmproj_cpu else "") + lora_tag
    rec = {"kind": "s2", "tag": tag, "time": now(), "build": BUILD, "model": a.model, "model_bytes": model_path.stat().st_size,
           "model_sha256": sha256_file(model_path), "ctx_per_slot": a.ctx, "np": a.np, "ub": a.ub, "kv": a.kv,
           "mmproj": a.mmproj, "mmproj_cpu": a.mmproj_cpu, "lora": a.lora, "checkpoints": a.checkpoints, "extra": a.extra}
    gpu = Gpu()
    with gpu_lock():
        rec["free_mib_at_start"] = gpu.mem()["free_mib"]
        srv = Server("exp0-s2", a.port, s2_args(a), tag)
        srv.start()
        sampler = Sampler(gpu, {"s2": srv}, a.interval)
        sampler.start()
        ok = srv.wait_healthy()
        rec["load"] = {"warm_s": srv.load_s if ok else None}
        try:
            if not ok:
                rec["error"] = "load failed"
            else:
                sampler.set_phase("steady")
                time.sleep(2.0)
                rec["workload"] = run_s2_workload(srv.url, a, sampler, n_long=a.np)
        except Exception as e:  # keep the VRAM numbers even if a request fails (e.g. OOM)
            rec["error"] = repr(e)[:500]
        finally:
            sampler.set_phase("post")
            time.sleep(0.5)
            sampler.stop()
            rec["vram"] = {"load_peak_mib": sampler.stats(["load"])["max_mib"], "steady_mib": sampler.stats(["steady"])["median_mib"],
                           "work_peak_mib": sampler.stats(["work"])["max_mib"], "peak_mib": sampler.stats()["max_mib"],
                           "samples": len(sampler.samples)}
            rec["log_facts"] = log_facts(srv.logs())
            srv.stop()
    append_jsonl(a.out, rec)
    summarize_s2(rec)


def summarize_s2(rec):
    print(rec["tag"], "free@start", rec.get("free_mib_at_start"), "load", rec.get("load"), "vram", rec.get("vram"), "err", rec.get("error"))
    w = rec.get("workload") or {}
    if "short" in w and w["short"].get("timings"):
        t = w["short"]["timings"]
        print(f"  short: pp {t.get('prompt_per_second'):.0f} t/s, tg {t.get('predicted_per_second'):.1f} t/s")
    if "image" in w:
        print("  image:", w["image"]["code"], (w["image"].get("answer") or "")[:120])
    for l in w.get("long", []):
        t = l["timings"]
        print(f"  long slot{l['slot']}: n={t.get('prompt_n')} pp {t.get('prompt_per_second'):.0f} t/s, tg {t.get('predicted_per_second'):.1f} t/s")


# ---------------------------------------------------------------- S2 quality proxy
def chat_prompt(p):
    return f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def cmd_s2q(a):
    prompts = [json.loads(l) for l in open(HERE / "data/s2_prompts.jsonl")]
    ref_path = Path(a.out).parent / "s2q_ref_continuations.json"
    tag = a.tag or f"s2q-{Path(a.model).stem}"
    a.ctx, a.np, a.ub, a.kv, a.mmproj, a.checkpoints, a.mmproj_cpu = 4096, 1, 512, "f16", None, 4, False
    gpu = Gpu()
    out_path = Path(a.out).parent / "s2q" / f"{tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lf = {"lora": lora_field(a, a.lora_scale)} if a.lora else {}
    with gpu_lock():
        srv = Server("exp0-s2", a.port, s2_args(a), tag)
        srv.start()
        try:
            if not srv.wait_healthy():
                raise RuntimeError("load failed: " + srv.logs()[-2000:])
            refs = json.loads(ref_path.read_text()) if ref_path.exists() else None
            make_ref = refs is None
            if make_ref:
                refs = {}
            with open(out_path, "w") as f:
                for p in prompts:
                    ptoks = tokenize(srv.url, chat_prompt(p["prompt"]))
                    if make_ref:
                        j, _ = completion(srv.url, ptoks, a.n_tokens, {"ignore_eos": False, **lf})
                        # tokenize the generated text back (robust across server versions)
                        refs[p["id"]] = tokenize(srv.url, j["content"])[: a.n_tokens]
                    cont = refs[p["id"]]
                    pos = []
                    for i in range(len(cont)):
                        j, _ = completion(srv.url, ptoks + cont[:i], 1, {"n_probs": a.n_probs, "cache_prompt": True, **lf})
                        cp = j["completion_probabilities"][0]
                        top = cp.get("top_logprobs") or cp.get("top_probs")
                        pos.append({str(t["id"]): (t["logprob"] if "logprob" in t else __import__("math").log(max(t["prob"], 1e-30))) for t in top})
                    f.write(json.dumps({"id": p["id"], "n": len(cont), "top_logprobs": pos}) + "\n")
            if make_ref:
                ref_path.write_text(json.dumps(refs))
        finally:
            srv.stop()
    print("wrote", out_path)


# ---------------------------------------------------------------- pairs
def cmd_pair(a):
    workload = [json.loads(l) for l in open(HERE / "data/s1_workload.jsonl")]
    tag = a.tag
    s1 = argparse.Namespace(model=a.s1_model, ctx=a.s1_ctx, ngl=a.s1_ngl, fa="on", output_cpu=a.s1_output_cpu, mmproj=None, extra=[])
    s2 = argparse.Namespace(model=a.s2_model, ctx=a.s2_ctx, np=a.s2_np, ub=512, kv=a.s2_kv, mmproj=a.s2_mmproj, lora=a.lora,
                            checkpoints=4, extra=[], mmproj_cpu=a.s2_mmproj_cpu)
    rec = {"kind": "pair", "tag": tag, "time": now(), "build": BUILD,
           "s1": {"model": a.s1_model, "ctx": a.s1_ctx, "ngl": a.s1_ngl, "output_cpu": a.s1_output_cpu, "sha256": sha256_file(MODELS / a.s1_model)},
           "s2": {"model": a.s2_model, "ctx_per_slot": a.s2_ctx, "np": a.s2_np, "kv": a.s2_kv, "mmproj": a.s2_mmproj,
                  "mmproj_cpu": a.s2_mmproj_cpu, "lora": a.lora,
                  "sha256": sha256_file(MODELS / a.s2_model)}}
    gpu = Gpu()
    with gpu_lock():
        rec["free_mib_at_start"] = gpu.mem()["free_mib"]
        srv1 = Server("exp0-s1", a.port, s1_args(s1), tag + "-s1")
        srv2 = Server("exp0-s2", a.port + 1, s2_args(s2), tag + "-s2")
        srv1.start()
        srv2.start()
        sampler = Sampler(gpu, {"s1": srv1, "s2": srv2}, a.interval)
        sampler.start()
        ok = srv1.wait_healthy() and srv2.wait_healthy()
        rec["load"] = {"s1_s": getattr(srv1, "load_s", None), "s2_s": getattr(srv2, "load_s", None), "note": "loaded concurrently"}
        try:
            if not ok:
                rec["error"] = "load failed"
            else:
                sampler.set_phase("steady")
                time.sleep(2.0)
                sampler.set_phase("work")
                s1_res = {}
                stop = threading.Event()

                def s1_loop():
                    lat = []
                    big = [w for w in workload if w["bucket"] == ("6k" if a.s1_ctx >= 8192 else "2k")]
                    small = [w for w in workload if w["bucket"] == "500"]
                    while not stop.is_set():
                        for w in big + small:
                            code, dt, body = systemone(srv1.url, w["request"])
                            if code == 200:
                                lat.append(dt)
                            if stop.is_set():
                                break
                    s1_res.update({"n": len(lat), "p50_s": pct(lat, 0.5), "p95_s": pct(lat, 0.95)})

                th = threading.Thread(target=s1_loop)
                th.start()
                rec["s2_workload"] = run_s2_workload(srv2.url, s2, sampler, n_long=a.s2_np)
                stop.set()
                th.join()
                rec["s1_workload"] = s1_res
        except Exception as e:
            rec["error"] = repr(e)[:500]
        finally:
            sampler.set_phase("post")
            time.sleep(0.5)
            sampler.stop()
            # combined = max over samples of (s1 + s2) at the same instant
            rec["vram"] = {
                "combined_peak_mib": sampler.stats(["work"])["max_mib"],
                "combined_peak_all_mib": sampler.stats()["max_mib"],
                "combined_steady_mib": sampler.stats(["steady"])["median_mib"],
                "s1_peak_mib": sampler.stats(["work"], "s1")["max_mib"],
                "s2_peak_mib": sampler.stats(["work"], "s2")["max_mib"],
                "s1_steady_mib": sampler.stats(["steady"], "s1")["median_mib"],
                "s2_steady_mib": sampler.stats(["steady"], "s2")["median_mib"],
                "samples": len(sampler.samples),
            }
            srv1.stop()
            srv2.stop()
    append_jsonl(a.out, rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "s2_workload"}))
    summarize_s2({"tag": tag, "workload": rec.get("s2_workload")})


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default=str(WORK / "results/runs.jsonl"))
    common.add_argument("--tag")
    common.add_argument("--interval", type=float, default=0.1)
    common.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra llama-server args (must be last)")

    p = sub.add_parser("s1", parents=[common])
    p.add_argument("--model", required=True, help="path under ~/srv/models")
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--ngl", default="all")
    p.add_argument("--fa", default="on")
    p.add_argument("--output-cpu", action="store_true", help="-ot output.weight=CPU")
    p.add_argument("--mmproj")
    p.add_argument("--cold", action="store_true", help="also measure a cold load (page cache evicted)")
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--no-workload", action="store_true")
    p.add_argument("--calib", action="store_true")
    p.add_argument("--calib-name")
    p.add_argument("--calib-limit", type=int)
    p.add_argument("--port", type=int, default=18180)
    p.set_defaults(fn=cmd_s1)

    p = sub.add_parser("s2", parents=[common])
    p.add_argument("--model", required=True)
    p.add_argument("--ctx", type=int, default=16384, help="per slot")
    p.add_argument("--np", type=int, default=1)
    p.add_argument("--ub", type=int, default=512)
    p.add_argument("--kv", default="f16")
    p.add_argument("--mmproj")
    p.add_argument("--mmproj-cpu", action="store_true", help="--no-mmproj-offload (vision encoder on the CPU)")
    p.add_argument("--lora", action="append", default=[])
    p.add_argument("--checkpoints", type=int, default=4)
    p.add_argument("--port", type=int, default=18181)
    p.set_defaults(fn=cmd_s2)

    p = sub.add_parser("s2q", parents=[common])
    p.add_argument("--model", required=True)
    p.add_argument("--lora", action="append", default=[])
    p.add_argument("--lora-scale", type=float, default=0.0)
    p.add_argument("--n-tokens", type=int, default=24)
    p.add_argument("--n-probs", type=int, default=20)
    p.add_argument("--port", type=int, default=18181)
    p.set_defaults(fn=cmd_s2q)

    p = sub.add_parser("pair", parents=[common])
    p.add_argument("--s1-model", required=True)
    p.add_argument("--s1-ctx", type=int, default=4096)
    p.add_argument("--s1-output-cpu", action="store_true")
    p.add_argument("--s1-ngl", default="all")
    p.add_argument("--s2-model", required=True)
    p.add_argument("--s2-ctx", type=int, default=16384)
    p.add_argument("--s2-np", type=int, default=1)
    p.add_argument("--s2-kv", default="f16")
    p.add_argument("--s2-mmproj")
    p.add_argument("--s2-mmproj-cpu", action="store_true")
    p.add_argument("--lora", action="append", default=[])
    p.add_argument("--port", type=int, default=18180)
    p.set_defaults(fn=cmd_pair)

    a = ap.parse_args()
    if a.tag is None and a.cmd == "pair":
        ap.error("pair needs --tag")
    a.fn(a)


if __name__ == "__main__":
    main()
