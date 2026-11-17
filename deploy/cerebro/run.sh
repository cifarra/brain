#!/bin/bash
# Gateway entry point on a Mac (launchd runs this; also fine by hand). Overrides go in $B/brain/env,
# e.g. BRAIN_TOKEN=... (set one: this listens on the LAN).
set -euo pipefail
B=${BRAIN_HOME:-$HOME/srv}
export BRAIN_PROFILE=cerebro BRAIN_HOST=0.0.0.0 BRAIN_REGISTRY=$B/brain-src/registry BRAIN_MODELS_DIR=$B/models \
       BRAIN_LLAMA_SERVER=$B/llama.cpp/b11580/llama-server BRAIN_CACHE_DIR=$B/brain/cache
[ -f $B/brain/env ] && . $B/brain/env
cd $B/brain-src
exec .venv/bin/brain serve
