#!/usr/bin/env bash
# Cosmos 3 environment on a cluster without root (Slurm + Apptainer). Run on a compute node, not on a login node.
#   bash setup_cosmos.sh <work_dir> [<hf_repo> <revision> <folder_name>]
# Same steps and versions as the lab setup (uv, Python 3.13, cosmos-framework c23e51f, cu128 wheels), inside an
# Ubuntu 24.04 container image. Everything goes under <work_dir>; nothing is written to the home folder on purpose.
# Safe to run again: finished steps are skipped. Log: <work_dir>/setup_cosmos.log
set -euo pipefail
W="$(realpath "$1")"
REPO="${2:-nvidia/Cosmos3-Super}"; REV="${3:-fp8}"; NAME="${4:-Cosmos3-Super-fp8}"
IMG="$W/cosmos_ubuntu2404.sif"
TOOLS="$W/tools"

if [ "${5:-}" != "inside" ]; then
  # ---- part 1, on the node itself: build the image, then run part 2 inside it
  mkdir -p "$W" "$TOOLS/bin" "$TOOLS/cosmos3_models"
  exec > >(tee -a "$W/setup_cosmos.log") 2>&1
  echo "=== setup start $(date '+%F %T') on $(hostname)"
  module purge; module load apptainer
  export APPTAINER_CACHEDIR="$W/apptainer_cache"; mkdir -p "$APPTAINER_CACHEDIR"
  if [ ! -f "$IMG" ]; then
    cat > "$W/cosmos_ubuntu2404.def" <<'EOF'
Bootstrap: docker
From: ubuntu:24.04
%post
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends ffmpeg git git-lfs libx11-dev build-essential curl ca-certificates
    rm -rf /var/lib/apt/lists/*
EOF
    echo "=== build image -> $IMG"
    apptainer build "$IMG" "$W/cosmos_ubuntu2404.def"
  fi
  echo "=== run part 2 inside the image"
  apptainer exec --nv --bind "$W:$W" "$IMG" bash "$(realpath "$0")" "$W" "$REPO" "$REV" "$NAME" inside
  echo "=== setup done $(date '+%F %T')"
  exit 0
fi

# ---- part 2, inside the image
export PATH="$TOOLS/bin:$PATH"
export UV_INSTALL_DIR="$TOOLS/bin" UV_NO_MODIFY_PATH=1 UV_LINK_MODE=copy
export UV_CACHE_DIR="$TOOLS/uv_cache" UV_PYTHON_INSTALL_DIR="$TOOLS/uv_python"
export UV_TOOL_DIR="$TOOLS/uv_tools" UV_TOOL_BIN_DIR="$TOOLS/bin"
export HF_HOME="$TOOLS/cosmos3_models/hf_cache" XDG_CACHE_HOME="$TOOLS/cache"

# 1. uv and the hf command
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
command -v hf >/dev/null 2>&1 || uv tool install huggingface_hub
hf version

# 2. cosmos-framework, pinned to the commit of the lab setup
FW="$TOOLS/cosmos-framework"
[ -d "$FW/.git" ] || git clone https://github.com/NVIDIA/cosmos-framework.git "$FW"
git -C "$FW" checkout -q c23e51f2f157ae3e51cfcd86ebfb5464850894f2
cd "$FW"
git rev-parse HEAD | tee COMMIT.txt

# 3. Python 3.13 venv with the CUDA 12.8 wheels
echo "=== uv sync (cu128-train)"
uv sync --all-extras --group=cu128-train
source .venv/bin/activate
# transformer-engine and torchcodec need the CUDA libraries of the venv; the driver libraries of the host
# come from apptainer --nv under /.singularity.d/libs and must stay on the path.
NV="$FW/.venv/lib/python3.13/site-packages/nvidia"
export NVRTC_HOME="$NV/cuda_nvrtc" CURAND_HOME="$NV/curand" CUDNN_HOME="$NV/cudnn"
export LD_LIBRARY_PATH="$(ls -d "$NV"/*/lib | tr '\n' ':')/.singularity.d/libs"

# 4. checkpoint (example media of the repo are skipped)
CK="$TOOLS/cosmos3_models/$NAME"
echo "=== checkpoint $REPO @ $REV -> $CK"
hf download "$REPO" --revision "$REV" --local-dir "$CK" --exclude "assets/*" "images/*"
du -sh "$CK"

# 5. checks: wheels import, GPUs visible, inference command loads
python - <<'EOF'
import torch, flash_attn, transformer_engine
from torchcodec.decoders import VideoDecoder
n = torch.cuda.device_count()
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| gpus", n)
for i in range(n):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i), torch.cuda.get_device_properties(i).total_memory // 2**20, "MiB")
EOF
python -m cosmos_framework.scripts.inference --help > "$W/cosmos_help.txt" && head -3 "$W/cosmos_help.txt" && echo "=== inference command ok"
