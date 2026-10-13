# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Experiment 0 analysis: calibration gate, S1/S2/pair tables, S2 quality proxy.

Usage: python3 analyze.py [results_dir]   (default: lab/exp0/results)
Writes <results>/tables.md and <results>/summary.json; prints the tables.
"""
import glob
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
GATE = {"top1": 0.98, "p99": 0.03, "max": 0.10, "tie": 0.02}
EPS = 1e-9


def load(p):
    p = Path(p)
    return [json.loads(l) for l in open(p)] if p.exists() else []


def pct(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def gib(mib):
    return "—" if mib is None else f"{mib / 1024:.2f}"


def f(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


# ------------------------------------------------------------------ S1 calibration
def tok_bucket(n):
    return "<500" if n < 500 else "500-1.5k" if n < 1500 else "1.5k-3k" if n < 3000 else ">=3k"


def calib_compare(ref, cand, recs_meta):
    ref = {r["id"]: r for r in ref if r.get("code") == 200}
    out = {"n_q": 0, "n_q_scored": 0, "agree": 0, "dps": [], "kls": [], "records": [], "fail_by": defaultdict(lambda: [0, 0])}
    for c in cand:
        if c.get("code") != 200 or c["id"] not in ref:
            continue
        r = ref[c["id"]]
        meta = recs_meta[c["id"]]
        rec_max, rec_ok, rec_ties = 0.0, True, 0
        for qid, pr in r["probs"].items():
            pc = c["probs"][qid]
            keys = list(pr)
            dps = [abs(pr[k] - pc[k]) for k in keys]
            out["dps"] += dps
            rec_max = max(rec_max, max(dps))
            kl = sum(pr[k] * math.log((pr[k] + EPS) / (pc[k] + EPS)) for k in keys)
            out["kls"].append(kl)
            srt = sorted(pr.values(), reverse=True)
            margin = srt[0] - (srt[1] if len(srt) > 1 else 0)
            out["n_q"] += 1
            if margin < GATE["tie"]:
                rec_ties += 1
                continue
            out["n_q_scored"] += 1
            same = max(pr, key=pr.get) == max(pc, key=pc.get)
            out["agree"] += same
            rec_ok &= same
        rec_pass = rec_ok and rec_max <= GATE["p99"]
        nopt = max(len(p) for p in r["probs"].values())
        out["records"].append({"id": c["id"], "kind": c["kind"], "tokens": r.get("input_tokens"), "n_opt": nopt,
                               "max_dp": rec_max, "top1_ok": rec_ok, "pass": rec_pass})
        for key in (f"kind={c['kind']}", f"tokens={tok_bucket(r.get('input_tokens') or 0)}",
                    f"options={'2-3' if nopt <= 3 else '4-7' if nopt <= 7 else '8-15'}"):
            out["fail_by"][key][0] += 1
            out["fail_by"][key][1] += 0 if rec_pass else 1
    dps = out["dps"]
    s = {
        "n_records": len(out["records"]), "n_questions": out["n_q"], "n_scored": out["n_q_scored"],
        "top1": out["agree"] / max(1, out["n_q_scored"]),
        "p99_dp": pct(dps, 0.99), "max_dp": max(dps) if dps else float("nan"), "mean_dp": sum(dps) / max(1, len(dps)),
        "mean_kl": sum(out["kls"]) / max(1, len(out["kls"])),
        "records_pass": sum(r["pass"] for r in out["records"]),
        "records_top1_fail": sum(not r["top1_ok"] for r in out["records"]),
        "fail_by": {k: v for k, v in sorted(out["fail_by"].items())},
        "worst": sorted(out["records"], key=lambda r: -r["max_dp"])[:5],
    }
    s["gate"] = s["top1"] >= GATE["top1"] and s["p99_dp"] <= GATE["p99"] and s["max_dp"] <= GATE["max"]
    return s


def calibration():
    meta = {r["id"]: r for r in load(HERE / "data/clef_records.jsonl")}
    ref = load(RES / "calib/BF16.jsonl")
    if not ref:
        return {}, "no BF16 reference yet\n"
    rows, out = {}, []
    names = sorted(Path(p).stem for p in glob.glob(str(RES / "calib/*.jsonl")) if Path(p).stem != "BF16")
    order = ["Q8_0", "Q6_K-hbf16", "Q5_K_M-hbf16", "Q4_K_M", "Q4_K_M-hbf16", "IQ4_XS-hbf16", "Q4_K_M-ocpu", "Q4_K_M-nooutput"]
    names = [n for n in order if n in names] + [n for n in names if n not in order]
    out.append("| S1 quant | top-1 (excl. ties) | p99 \\|Δp\\| | max \\|Δp\\| | mean \\|Δp\\| | mean KL | records passing | gate |")
    out.append("|---|---|---|---|---|---|---|---|")
    for n in names:
        s = calib_compare(ref, load(RES / f"calib/{n}.jsonl"), meta)
        rows[n] = s
        out.append(f"| {n} | {s['top1']*100:.1f}% ({s['n_scored']} q) | {s['p99_dp']:.4f} | {s['max_dp']:.4f} | {s['mean_dp']:.4f} | "
                   f"{s['mean_kl']:.5f} | {s['records_pass']}/{s['n_records']} | {'PASS' if s['gate'] else 'fail'} |")
    out.append("")
    out.append("Records failing (record passes = top-1 agrees on every non-tie question and its max |Δp| ≤ 0.03), by slice (failing/total):")
    out.append("")
    keys = sorted({k for s in rows.values() for k in s["fail_by"]})
    out.append("| slice | " + " | ".join(rows) + " |")
    out.append("|---|" + "---|" * len(rows))
    for k in keys:
        out.append(f"| {k} | " + " | ".join(f"{rows[n]['fail_by'].get(k, [0, 0])[1]}/{rows[n]['fail_by'].get(k, [0, 0])[0]}" for n in rows) + " |")
    out.append("")
    for n, s in rows.items():
        w = ", ".join(f"{r['id']}({r['kind']},{r['tokens']}t,{r['n_opt']}opt,{r['max_dp']:.3f})" for r in s["worst"][:3])
        out.append(f"- worst {n}: {w}")
    ref_tokens = [r.get("input_tokens", 0) for r in ref if r.get("code") == 200]
    out.append(f"\nReference: BF16, {len(ref_tokens)} records, {GATE['tie']} near-tie margin; prompt tokens {min(ref_tokens)}-{max(ref_tokens)}.")
    return rows, "\n".join(out) + "\n"


# ------------------------------------------------------------------ S1 sweep
def s1_table():
    rows = [r for r in load(RES / "s1.jsonl")]
    out = ["| S1 model | ctx=b=ub | output.weight | file GiB | load cold/warm s | steady GiB | peak GiB | free@start GiB | 500 p50/p95 s | 2k p50/p95 s | 6k p50/p95 s | first req s (tok) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("ngl") not in (None, "all"):
            note = f" ngl{r['ngl']}"
        else:
            note = ""
        name = Path(r["model"]).stem.replace("Clef-Flash-", "") + note
        ld = r.get("load") or {}
        w = (r.get("workload") or {}).get("buckets", {})
        def bk(b):
            v = w.get(b)
            if not v:
                return "—"
            if not v.get("n"):
                return "doesn't fit" if v.get("n_errors") else "—"
            return f"{v['p50_s']:.3f}/{v['p95_s']:.3f}"
        fr = (r.get("workload") or {}).get("first_request") or {}
        vr = r.get("vram") or {}
        out.append(f"| {name} | {r['ctx']} | {'CPU' if r.get('output_cpu') else 'GPU'} | {r['model_bytes']/2**30:.2f} | "
                   f"{f(ld.get('cold_s'),1)}/{f(ld.get('warm_s'),1)} | {gib(vr.get('steady_mib'))} | {gib(vr.get('peak_mib'))} | "
                   f"{gib(r.get('free_mib_at_start'))} | {bk('500')} | {bk('2k')} | {bk('6k')} | "
                   f"{f(fr.get('latency_s'))} ({fr.get('input_tokens','—')}) |" + (f" ERROR {r['error']}" if r.get("error") else ""))
    return rows, "\n".join(out) + "\n"


# ------------------------------------------------------------------ S2
def s2_table():
    rows = [r for r in load(RES / "s2.jsonl") if r.get("kind") == "s2"]
    out = ["| S2 quant | ctx/slot | np | KV | mmproj | adapter | file GiB | load s | steady GiB | peak GiB | free@start GiB | short pp/tg t/s | long prompt tok | long pp / tg t/s (per slot) | image (latency) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        w = r.get("workload") or {}
        st = (w.get("short") or {}).get("timings") or {}
        lg = [l.get("timings") or {} for l in w.get("long", [])]
        lpp = "+".join(f"{t.get('prompt_per_second', 0):.0f}" for t in lg) or "—"
        ltg = "+".join(f"{t.get('predicted_per_second', 0):.1f}" for t in lg) or "—"
        img = w.get("image")
        img_s = "—" if not img else (f"ok {img['latency_s']:.1f}s" if img.get("code") == 200 else f"HTTP {img.get('code')}")
        vr = r.get("vram") or {}
        name = Path(r["model"]).stem.replace("Qwen3.5-9B-", "")
        out.append(f"| {name} | {r['ctx_per_slot']} | {r['np']} | {r['kv']} | {('F16 CPU' if r.get('mmproj_cpu') else 'F16') if r.get('mmproj') else 'no'} | {'abl' if r.get('lora') else '—'} | "
                   f"{r['model_bytes']/2**30:.2f} | {f((r.get('load') or {}).get('warm_s'),1)} | {gib(vr.get('steady_mib'))} | {gib(vr.get('peak_mib'))} | "
                   f"{gib(r.get('free_mib_at_start'))} | {f(st.get('prompt_per_second'),0)}/{f(st.get('predicted_per_second'),1)} | "
                   f"{w.get('long_prompt_tokens','—')} | {lpp}/{ltg} | {img_s} |" + (f" ERROR {r['error'][:120]}" if r.get("error") else ""))
    return rows, "\n".join(out) + "\n"


def s2_quality():
    ref = {r["id"]: r for r in load(RES / "s2q/s2q-Q8_0.jsonl")}
    if not ref:
        return {}, "no S2 quality reference yet\n"
    out = ["| S2 quant | positions | top-1 agreement vs Q8_0 | mean KL (top-20 approx) | p99 \\|Δp\\| of ref top-1 |", "|---|---|---|---|---|"]
    rows = {}
    for p in sorted(glob.glob(str(RES / "s2q/*.jsonl"))):
        name = Path(p).stem.replace("s2q-", "")
        if name == "Q8_0":
            continue
        n = agree = 0
        kls, dps = [], []
        for c in load(p):
            r = ref.get(c["id"])
            if not r:
                continue
            for lr, lc in zip(r["top_logprobs"], c["top_logprobs"]):
                pr = {k: math.exp(v) for k, v in lr.items()}
                pc = {k: math.exp(v) for k, v in lc.items()}
                n += 1
                t_r = max(pr, key=pr.get)
                agree += t_r == max(pc, key=pc.get)
                floor_c = min(pc.values()) if pc else EPS
                kl = 0.0
                for k, v in pr.items():
                    q = pc.get(k, min(floor_c, max(EPS, 1 - sum(pc.values()))))
                    kl += v * math.log((v + EPS) / (q + EPS))
                rest_r, rest_c = max(EPS, 1 - sum(pr.values())), max(EPS, 1 - sum(pc.get(k, 0) for k in pr))
                kl += rest_r * math.log(rest_r / rest_c)
                kls.append(max(0.0, kl))
                dps.append(abs(pr[t_r] - pc.get(t_r, 0.0)))
        rows[name] = {"positions": n, "top1": agree / max(1, n), "mean_kl": sum(kls) / max(1, len(kls)), "p99_dp": pct(dps, 0.99)}
        out.append(f"| {name} | {n} | {rows[name]['top1']*100:.1f}% | {rows[name]['mean_kl']:.4f} | {rows[name]['p99_dp']:.3f} |")
    return rows, "\n".join(out) + "\n"


def pair_table():
    rows = [r for r in load(RES / "pairs.jsonl")]
    out = ["| pair | S1 | S2 | combined peak GiB | combined steady GiB | S1 peak | S2 peak | free@start GiB | S2 long pp/tg t/s | S1 p50/p95 s under load |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        vr = r.get("vram") or {}
        s1, s2 = r["s1"], r["s2"]
        lg = [l.get("timings") or {} for l in (r.get("s2_workload") or {}).get("long", [])]
        lpp = "+".join(f"{t.get('prompt_per_second', 0):.0f}" for t in lg) or "—"
        ltg = "+".join(f"{t.get('predicted_per_second', 0):.1f}" for t in lg) or "—"
        w1 = r.get("s1_workload") or {}
        out.append(f"| {r['tag']} | {Path(s1['model']).stem.replace('Clef-Flash-','')} ub{s1['ctx']} ngl {s1.get('ngl', 'all')}{' ocpu' if s1.get('output_cpu') else ''} | "
                   f"{Path(s2['model']).stem.replace('Qwen3.5-9B-','')} c{s2['ctx_per_slot']} np{s2['np']} {s2['kv']}{(' mmproj-CPU' if s2.get('mmproj_cpu') else ' mmproj') if s2.get('mmproj') else ''}{' abl' if s2.get('lora') else ''} | "
                   f"**{gib(vr.get('combined_peak_mib'))}** | {gib(vr.get('combined_steady_mib'))} | {gib(vr.get('s1_peak_mib'))} | {gib(vr.get('s2_peak_mib'))} | "
                   f"{gib(r.get('free_mib_at_start'))} | {lpp}/{ltg} | {f(w1.get('p50_s'),3)}/{f(w1.get('p95_s'),3)} (n={w1.get('n')}) |"
                   + (f" ERROR {r['error'][:120]}" if r.get("error") else ""))
    return rows, "\n".join(out) + "\n"


def main():
    cal_rows, cal_md = calibration()
    _, s1_md = s1_table()
    _, s2_md = s2_table()
    q_rows, q_md = s2_quality()
    _, p_md = pair_table()
    md = "## S1 sweep\n\n" + s1_md + "\n## S1 calibration vs BF16\n\n" + cal_md + "\n## S2 sweep\n\n" + s2_md + \
         "\n## S2 quality proxy vs Q8_0\n\n" + q_md + "\n## Pairs\n\n" + p_md
    (RES / "tables.md").write_text(md)
    (RES / "summary.json").write_text(json.dumps({"calibration": cal_rows, "s2_quality": q_rows}, indent=1, default=str))
    print(md)


if __name__ == "__main__":
    main()
