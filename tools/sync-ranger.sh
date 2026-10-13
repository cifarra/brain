#!/usr/bin/env bash
# Copy this repo's working tree to ranger (~/srv/brain-src) through hx.
# The laptop has no direct route/key to ranger, so we pipe a tar stream: laptop -> hx -> ranger.
# Usage: tools/sync-ranger.sh [subdir ...]   (default: whole repo, minus .git and caches)
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=${BRAIN_SRC_DIR:-srv/brain-src}
paths=("${@:-.}")
COPYFILE_DISABLE=1 tar -c --no-xattrs --exclude .git --exclude .venv --exclude __pycache__ --exclude .pytest_cache \
  --exclude '*.gguf' --exclude '*.safetensors' "${paths[@]}" |
  ssh -o BatchMode=yes hx "ssh -o BatchMode=yes ranger 'mkdir -p ~/$DEST && tar -x -C ~/$DEST'"
echo "synced ${paths[*]} -> ranger:~/$DEST"
