# brain on ranger: operations

The brain is one container (`brain`) on ranger. It runs a Python gateway on :8790 that owns two
llama-server children: `s1` (Clef decisions) and `s2` (Qwen3.5-9B chat + adapters). Clients reach
it as `https://brain.hx` (Caddy on hx, tailnet) or `http://192.168.1.215:8790` on the LAN.

| What | Where |
|---|---|
| source (synced from the laptop) | `ranger:~/srv/brain-src` (`tools/sync-ranger.sh`) |
| compose stack + `.env` | `ranger:~/srv/docker/apps/brain/` |
| model weights + adapters | `ranger:~/srv/models` (`download.sh` re-creates them at pinned revisions; not backed up) |
| registry (models, adapters, profiles, router) | `~/srv/brain-src/registry`, mounted read-only |

## Install / update
The laptop has no direct route to ranger, so everything goes through hx.
```bash
# laptop
tools/sync-ranger.sh                       # repo -> ranger:~/srv/brain-src
# ranger (ssh hx, then ssh ranger)
mkdir -p ~/srv/docker/apps/brain && cd ~/srv/docker/apps/brain
cp ~/srv/brain-src/deploy/ranger/compose.yaml .
[ -f .env ] || { cp ~/srv/brain-src/deploy/ranger/.env.example .env; chmod 600 .env; }
docker compose build && docker compose up -d
curl -s localhost:8790/health
```

## Swapping models / adding an adapter (no rebuild)
1. Edit the registry in the repo (model manifest, adapter manifest, profile, router).
2. `uv run brain check` on the laptop, with `--no-files` because the weights live on ranger.
3. `tools/sync-ranger.sh registry`
4. On ranger: `docker compose restart brain` (or change `BRAIN_PROFILE` in `.env`, then `docker compose up -d`).

A gateway restart reloads both models. That takes seconds, and PCIe gen 1 makes it slower than it
would otherwise be.

## Upgrading llama.cpp
The build is pinned by digest in the Dockerfile, on purpose: Clef calibration and the adapter gates
were measured on this build. To bump it:
1. Change the tag and digest.
2. Re-run the Experiment 0 calibration subset (`lab/exp0`) and the adapter gate (`tools/extract_adapter`).
3. Rebuild.

`update-containers` skips `brain:local`, and watchtower is told to ignore the container.

## hx: Caddy route (installed 2026-10-13)
The block lives in `hx:~/srv/docker/apps/caddy/Caddyfile`; the backup from before it was added is
`Caddyfile.bak-brain-20261013`. After editing, run `docker restart caddy`. Restart, never reload: the
Caddyfile is a single-file bind mount. TLS comes from the global `local_certs`.
```
brain.hx {
  reverse_proxy 192.168.1.215:8790 {
    flush_interval -1          # stream SSE tokens immediately
    transport http {
      response_header_timeout 15m
    }
  }
}
```
Then add `100.86.82.61 brain.hx` to `/etc/hosts` on each client. This is the `.hx` DNS trap: no
network DNS serves `.hx` names.

## hx: Uptime Kuma
Add an HTTP keyword monitor on `http://192.168.1.215:8790/health` with keyword `BRAIN_OK`. While the
models are loading, `/health` answers `loading`, which is expected for about a minute after a restart.

## Logs
```bash
docker logs -f brain          # gateway + both llama-server children (prefixed s1/s2)
```
