# syntax=docker/dockerfile:1
#
# Build (from repo root, so pyproject.toml/uv.lock are in the build context):
#   docker build -f src/Dockerfile -t mj-biped-train .
#
# Run (GPU, checkpoints persisted on the host):
#   docker run --rm --gpus all -v $(pwd)/logs:/workspace/logs mj-biped-train \
#     train Mjlab-Tinker --env.scene.num-envs 1024
#
# CPU fallback (no --gpus flag; mjlab picks CPU automatically):
#   docker run --rm -v $(pwd)/logs:/workspace/logs mj-biped-train \
#     train Mjlab-Tinker --env.scene.num-envs 64

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    MUJOCO_GL=egl \
    PATH="/opt/venv/bin:${PATH}"

# libgl1/libegl1/libosmesa6: headless MuJoCo rendering (video recording, viewer).
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 \
      libglfw3 \
      libegl1 \
      libosmesa6 \
      git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /workspace

# Install third-party deps first so this layer stays cached across source edits.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

# Now install the local packages (mj_biped, optim) on top.
COPY src ./src
RUN uv sync --locked --no-dev

# uv.lock resolves plain PyPI `torch`, which on Linux now bundles a CUDA 13
# runtime (nvidia-*-cu13 wheels) — that needs a driver newer than most hosts
# have. Swap torch/torchvision for the CUDA 12.8 build from PyTorch's own
# index instead. Must run *after* the sync above so it isn't reverted by
# lockfile hash verification. mujoco-warp/warp-lang talk to the driver
# directly (no bundled CUDA runtime of their own), so they're unaffected.
#
# Pinned to the exact "+cu128" local version: download.pytorch.org/whl/cu128
# tops out at torch 2.11.0 (no 2.12.0 build exists there yet), and without
# the "+cu128" suffix "torch==2.12.0" silently matches the plain PyPI (CUDA
# 13) wheel via --extra-index-url instead of failing loudly.
RUN uv pip install --python /opt/venv/bin/python \
      --index-url https://download.pytorch.org/whl/cu128 \
      --extra-index-url https://pypi.org/simple \
      "torch==2.11.0+cu128" "torchvision==0.26.0+cu128"

ENTRYPOINT ["uv", "run"]
CMD ["train", "Mjlab-Tinker"]
