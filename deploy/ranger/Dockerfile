# brain: Python gateway + llama-server backends in one container.
# Base = the pinned llama.cpp CUDA build (b11580, commit 4413a19, CUDA 12.8, Ubuntu 24.04, python3.12 present).
# Bumping it means re-running the Experiment 0 calibration subset and the adapter gates (see ops.md).
FROM ghcr.io/ggml-org/llama.cpp:server-cuda-b11580@sha256:c4348afb91499b10fad12f92626c7a8ca8b69d6475d10c793b810cb61237ce00

COPY --from=ghcr.io/astral-sh/uv:0.12.24 /uv /usr/local/bin/uv

WORKDIR /opt/brain
COPY pyproject.toml uv.lock* README.md ./
COPY src ./src
RUN uv venv --python /usr/bin/python3 /opt/brain/.venv \
 && uv pip install --python /opt/brain/.venv/bin/python --no-cache . \
 && rm /usr/local/bin/uv
COPY registry ./registry

ENV PATH=/opt/brain/.venv/bin:$PATH \
    BRAIN_LLAMA_SERVER=/app/llama-server \
    BRAIN_REGISTRY=/opt/brain/registry \
    BRAIN_MODELS_DIR=/models \
    BRAIN_HOST=0.0.0.0 \
    BRAIN_PORT=8790

USER 1000:1000
EXPOSE 8790
ENTRYPOINT ["brain", "serve"]
