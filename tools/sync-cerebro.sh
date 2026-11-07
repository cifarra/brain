#!/usr/bin/env bash
# Copy this repo's working tree to cerebro (~/srv/brain-src) and refresh the venv.
# Usage: tools/sync-cerebro.sh [subdir ...]   (default: what the gateway needs)
# Then, if code changed: ssh cerebro 'pkill -f brain-src/.venv/bin/brain' (launchd restarts it, no sudo)
set -euo pipefail
cd "$(dirname "$0")/.."
paths=("${@:-src registry deploy pyproject.toml uv.lock README.md}")
# shellcheck disable=SC2068
COPYFILE_DISABLE=1 tar -c --no-xattrs --exclude __pycache__ --exclude .pytest_cache ${paths[@]} |
  ssh -o BatchMode=yes cerebro 'mkdir -p ~/srv/brain-src && cd ~/srv/brain-src && tar -x && ~/.local/bin/uv sync --no-dev -q'
echo "synced ${paths[*]} -> cerebro:~/srv/brain-src"
