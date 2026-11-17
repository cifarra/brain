# brain on a CUDA host: operations

The brain is one container (`brain`). It runs the gateway on :8790, which owns two llama-server
children: `s1` (Clef decisions) and `s2` (Qwen3.5-9B chat + adapters). The `ranger-*` profiles were
measured on an RTX 4090 shared with other services (see `lab/exp0/REPORT.md`).

| What | Where |
|---|---|
| source | `$BRAIN_SRC` (`tools/sync-ranger.sh` copies it from a dev machine) |
| compose stack + `.env` | any directory: `compose.yaml` + `.env` from `.env.example` |
| model weights + adapters | `$BRAIN_MODELS`, mounted read-only at `/models` |
| registry (models, adapters, profiles, router) | `$BRAIN_SRC/registry`, mounted read-only |

## Install / update
```bash
mkdir -p ~/brain && cd ~/brain
cp "$BRAIN_SRC/deploy/ranger/compose.yaml" .
[ -f .env ] || { cp "$BRAIN_SRC/deploy/ranger/.env.example" .env; chmod 600 .env; }   # then edit it
docker compose build && docker compose up -d
curl -s localhost:8790/health
```
The port is published on all interfaces. Set `BRAIN_TOKEN`, or firewall it to trusted hosts.

## Swapping models / adding an adapter (no rebuild)
1. Edit the registry (model manifest, adapter manifest, profile, router).
2. `uv run brain check <profile> --no-files` on the dev machine (the weights live on the host).
3. `tools/sync-ranger.sh registry`
4. On the host: `docker compose restart brain` (or change `BRAIN_PROFILE` in `.env`, then `docker compose up -d`).

A gateway restart reloads both models, which takes seconds.

## Upgrading llama.cpp
The build is pinned by digest in the Dockerfile, on purpose: Clef calibration and the adapter gates
were measured on this build. To bump it:
1. Change the tag and digest.
2. Re-run the Experiment 0 calibration subset (`lab/exp0`) and the adapter gate (`tools/extract_adapter`).
3. Rebuild.

## Behind a reverse proxy
Disable response buffering so SSE tokens stream, and allow long responses. Caddy:
```
brain.example {
  reverse_proxy <host>:8790 {
    flush_interval -1
    transport http {
      response_header_timeout 15m
    }
  }
}
```

## Monitoring and logs
`GET /health` returns 200 with the literal `BRAIN_OK` when ready. It answers `loading` for about a
minute after a restart.
```bash
docker logs -f brain          # gateway + both llama-server children (prefixed s1/s2)
```
