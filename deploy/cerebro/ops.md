# brain on a 16 GB Mac mini: operations

Profile `cerebro`: Mac mini M4, 16 GB. The gateway runs natively, not in Docker. llama-server uses Metal
through the pinned b11580 release. Residency is **swap**: one model is in memory at a time. A request for
the other backend waits for in-flight requests, swaps the models (about 5 s) and reports `x_brain.swap_ms`.
S2 starts first.

| What | Where |
|---|---|
| source + venv | `~/srv/brain-src` (`tools/sync-cerebro.sh` copies it from a dev machine) |
| entry point | `deploy/cerebro/run.sh` (env overrides in `~/srv/brain/env`) |
| llama.cpp | `~/srv/llama.cpp/b11580/` (pinned: calibration was measured on it) |
| models + adapters | `~/srv/models` (paths as in `registry/models/*.yaml`) |
| logs | `~/srv/brain/logs/gateway.log` (newsyslog keeps 5 × 20 MB) |
| eval suites for the UI | `registry/evals/*.json` (`tools/build_evals.py`) |

## One-time root setup
```bash
sudo ~/srv/brain-src/deploy/cerebro/install.sh            # --remove undoes it
```
This installs two LaunchDaemons:
- `local.brain.gpu-memory-limit` sets `iogpu.wired_limit_mb=14336` at boot. The macOS default of
  about 10.7 GB is too small for the calibrated Clef, whose weights alone are 11.2 GB.
- `local.brain.gateway` runs `run.sh` as the user who ran `sudo`, at boot, and keeps it alive.
  A headless Mac has no GUI login, so a LaunchAgent would never start.

`run.sh` listens on `0.0.0.0:8790`. Put `BRAIN_TOKEN=...` in `~/srv/brain/env` unless the network is trusted.

## Update
```bash
tools/sync-cerebro.sh                            # code/registry -> ~/srv/brain-src, uv sync
pkill -f brain-src/.venv/bin/brain               # on the Mac: launchd restarts it within ~10 s
curl -s http://<mac>:8790/health
```
Don't use `-ot output.weight=CPU` on Metal. It adds a 1 GB CPU_REPACK copy, while the mapped
buffer still covers the whole file.
