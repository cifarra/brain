# brain on cerebro: operations

cerebro is a Mac mini M4 with 16 GB, at 192.168.1.204 (`ssh cerebro`, which jumps through perez). The
gateway runs natively, not in Docker. llama-server uses Metal through the pinned b11580 release.
Residency is **swap**: one model is in memory at a time. A request for the other backend waits for
in-flight requests, swaps the models (seconds) and reports `x_brain.swap_ms`. S2 starts first.

Clients reach it at `https://brain.hx` (Caddy on hx, tailnet) or `http://192.168.1.204:8790` on the
LAN. The web UI is at `/ui/` and has Chat, Decide, Evals and Status tabs.

| What | Where |
|---|---|
| source + venv | `~/srv/brain-src` (`tools/sync-cerebro.sh` from the laptop) |
| entry point | `deploy/cerebro/run.sh` (env overrides in `~/srv/brain/env`) |
| llama.cpp | `~/srv/llama.cpp/b11580/` (pinned: calibration was measured on it) |
| models + adapters | `~/srv/models` (Clef Q8_0-abf16 rebuilt locally, sha256 matches ranger) |
| logs | `~/srv/brain/logs/gateway.log` (newsyslog keeps 5 × 20 MB) |
| eval suites for the UI | `registry/evals/*.json` (`tools/build_evals.py`) |

## One-time root setup
```bash
ssh -t cerebro 'sudo ~/srv/brain-src/deploy/cerebro/install.sh'      # --remove undoes it
```
This installs two LaunchDaemons:
- `local.brain.gpu-memory-limit` sets `iogpu.wired_limit_mb=14336` at boot. The macOS default of
  about 10.7 GB is too small for the calibrated Clef, whose weights alone are 11.2 GB.
- `local.brain.gateway` runs `run.sh` as cucostudio at boot and keeps it alive. There is no GUI
  login, so a LaunchAgent would never start.

## Update
```bash
tools/sync-cerebro.sh                                    # code/registry -> cerebro, uv sync
ssh cerebro 'pkill -f brain-src/.venv/bin/brain'         # launchd restarts it within ~10 s
curl -s http://192.168.1.204:8790/health
```
Don't use `-ot output.weight=CPU` on Metal. It adds a 1 GB CPU_REPACK copy, while the mapped
buffer still covers the whole file.

## hx: Caddy route
`brain.hx` points to `192.168.1.204:8790`. The block's details are in `deploy/ranger/ops.md`.
Backups: `Caddyfile.bak-brain-cerebro-20261106` (it pointed to ranger before this change).
