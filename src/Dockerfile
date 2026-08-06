# GPU training image for the `mj_biped` / tinker task. Plain pip, no `uv`
# binary needed on the server (uv is only used as this repo's *local* dev
# tool via pyproject.toml's build-system; pip builds it in an isolated env
# automatically).
#
# IMPORTANT: don't just `pip install mjlab mujoco` + COPY src + PYTHONPATH.
# `train`/`play` discover tasks via `importlib.metadata.entry_points(group=
# "mjlab.tasks")` (see mjlab/__init__.py:_import_registered_packages), which
# only sees packages that are actually *installed* as a distribution (i.e.
# have a .dist-info/entry_points.txt). A raw file copy on PYTHONPATH is
# invisible to that lookup, so "Mjlab-Tinker" would silently fail to
# register. `pip install .` on this repo's own pyproject.toml is what wires
# `[project.entry-points."mjlab.tasks"] mj-biped = "mj_biped.tasks"` in
# properly.
#
# Build from the REPO ROOT (pyproject.toml lives there, not under src/):
#
#   docker build -f src/Dockerfile -t mj-biped-train .
#
# Train (needs an NVIDIA driver + nvidia-container-toolkit on the host):
#
#   docker run --rm --gpus all -v "$(pwd)/logs:/app/logs" mj-biped-train \
#     train Mjlab-Tinker --env.scene.num-envs 1024
#
# Play a checkpoint:
#
#   docker run --rm --gpus all -v "$(pwd)/logs:/app/logs" mj-biped-train \
#     play Mjlab-Tinker --checkpoint-file logs/rsl_rl/tinker/<run-dir>/model_0.pt

FROM python:3.12-slim

WORKDIR /app

# rsl_rl's logger does `import git` (GitPython) at module load time to tag
# runs with the current commit; GitPython raises ImportError if no `git`
# binary is on PATH, which takes down `train`/`play` entirely.
#
# libgl1/libegl1/libosmesa6: headless MuJoCo rendering, needed if you ever
# pass `--video` to `train`/`play` (VideoRecorder). Drop these (and set
# MUJOCO_GL=disable instead of egl) if you never render.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      libgl1 \
      libglfw3 \
      libegl1 \
      libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

ENV MUJOCO_GL=egl \
    PYTHONUNBUFFERED=1

# Only pyproject.toml/README.md here, NOT src/ yet: Docker invalidates a
# layer -- and every layer after it -- the moment its COPY'd content
# changes. Since we edit src/ constantly but the dependency list almost
# never changes, the expensive installs below must happen BEFORE src/ is
# copied in, or every source edit forces a full torch/mjlab re-download.
COPY pyproject.toml README.md ./

# Default PyPI `torch` now bundles a CUDA 13 runtime, which needs an NVIDIA
# driver >=580.65.06. mjlab's own pyproject routes its `cu128` extra to
# https://download.pytorch.org/whl/cu128 via `[tool.uv.sources]` -- but that
# is a uv-only mechanism that plain pip ignores. Reproduce it manually here:
# install the CUDA 12.8 torch build first so the later install sees the
# `torch>=2.7.0` requirement already satisfied and never reaches for the
# CUDA 13 default. Match this to whatever `nvidia-smi` reports on the host
# (swap cu128 for cu121/cu124/cu126 if it reports an older CUDA version).
RUN pip install --no-cache-dir "torch==2.11.0+cu128" \
      --index-url https://download.pytorch.org/whl/cu128

# Install the rest of the heavy deps up front too, without needing src/ yet.
# Keep this in sync with [project.dependencies] in pyproject.toml.
RUN pip install --no-cache-dir "mjlab>=1.4.0" "mujoco>=3.8.1"

# Only now bring in the actual source -- this and the next layer are cheap
# and rebuild in seconds since the big downloads above are already cached.
COPY src/ src/
RUN pip install --no-cache-dir --no-deps .

CMD ["bash"]
