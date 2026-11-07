#!/bin/bash
# Gateway entry point on cerebro (launchd runs this; also fine by hand). Overrides go in ~/srv/brain/env.
set -euo pipefail
B=/Users/cucostudio/srv
export BRAIN_PROFILE=cerebro BRAIN_REGISTRY=$B/brain-src/registry BRAIN_MODELS_DIR=$B/models \
       BRAIN_LLAMA_SERVER=$B/llama.cpp/b11580/llama-server BRAIN_CACHE_DIR=$B/brain/cache
[ -f $B/brain/env ] && . $B/brain/env
cd $B/brain-src
exec .venv/bin/brain serve
