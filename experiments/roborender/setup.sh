#!/usr/bin/env bash
# RoboRender (depth-conditioned Wan2.1 1.3B, robo-render.github.io) for a test against our Cosmos videos.
# Runs INSIDE the container (root). Touches no existing env. Everything goes under /workspace/tools:
#   tools/roborender/src                       the RoboRender code (git clone, commit written to the log)
#   tools/roborender/.venv                     uv venv, Python 3.10, the pinned requirements of the repo (torch 2.9.1 cu128)
#   tools/roborender_models/Wan2.1-Fun-V1.1-1.3B-Control   base model, 19.8 GB (alibaba-pai)
#   tools/roborender_models/ckpt/roborender_lora.safetensors  the RoboRender LoRA, 0.7 GB
#   tools/roborender_models/hf_cache           HF_HOME
# Steps (safe to re-run, finished steps are skipped):
#   bash experiments/roborender/setup.sh dry-run    # clone + venv, then only SHOW what pip would install
#   bash experiments/roborender/setup.sh install    # install the packages
#   bash experiments/roborender/setup.sh download   # the two model downloads (20.5 GB)
#   bash experiments/roborender/setup.sh check      # import torch in the venv, list the model files
# Log: setup.log next to this script (appended). Afterwards: source experiments/roborender/env.sh
set -euo pipefail
cd /workspace/SoftMimicGen
TOOLS=/workspace/tools
HERE="$(cd "$(dirname "$0")" && pwd)"   # the log goes next to this script
SRC="$TOOLS/roborender/src"
VENV="$TOOLS/roborender/.venv"
M="$TOOLS/roborender_models"
REPO=https://github.com/robo-render/RoboRender-Video-Model
mkdir -p "$HERE" "$TOOLS/roborender" "$M"
exec > >(tee -a "$HERE/setup.log") 2>&1
step="${1:-dry-run}"
echo "=== roborender setup $step start $(date '+%F %T')"

export PATH="$TOOLS/bin:$PATH"
export UV_CACHE_DIR="$TOOLS/uv_cache" UV_PYTHON_INSTALL_DIR="$TOOLS/uv_python" UV_TOOL_DIR="$TOOLS/uv_tools" UV_TOOL_BIN_DIR="$TOOLS/bin"
export HF_HOME="$M/hf_cache"
command -v uv >/dev/null || { echo "uv missing: run experiments/cosmos3/setup.sh first"; exit 1; }
command -v hf >/dev/null || { echo "hf missing: run experiments/cosmos3/setup.sh first"; exit 1; }

clone_and_venv() {
  if [ ! -d "$SRC/.git" ]; then
    git clone --depth 1 "$REPO" "$SRC"
  fi
  echo "=== code: $(git -C "$SRC" rev-parse --short HEAD) $(git -C "$SRC" log -1 --format=%cd)"
  [ -d "$VENV" ] || uv venv --python 3.10 "$VENV"
  echo "=== venv: $("$VENV/bin/python" --version)"
}

case "$step" in
  dry-run)
    clone_and_venv
    echo "=== pip dry run (nothing is installed):"
    uv pip install --python "$VENV/bin/python" --dry-run -r "$SRC/requirements.txt"
    ;;
  install)
    clone_and_venv
    uv pip install --python "$VENV/bin/python" -r "$SRC/requirements.txt"
    "$VENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
    ;;
  download)
    echo "=== alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control -> $M/Wan2.1-Fun-V1.1-1.3B-Control (19.8 GB)"
    # one --include per pattern: with one --include and several patterns hf skipped the first file (seen 2026-10-12)
    hf download alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control --local-dir "$M/Wan2.1-Fun-V1.1-1.3B-Control" \
      --include "diffusion_pytorch_model.safetensors" --include "Wan2.1_VAE.pth" --include "models_t5_umt5-xxl-enc-bf16.pth" \
      --include "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" --include "google/umt5-xxl/*" --include "config.json"
    [ -f "$M/Wan2.1-Fun-V1.1-1.3B-Control/diffusion_pytorch_model.safetensors" ] || { echo "transformer weights missing"; exit 1; }
    echo "=== RoboRender/roborender-video -> $M/ckpt (0.7 GB)"
    hf download RoboRender/roborender-video roborender_lora.safetensors --local-dir "$M/ckpt"
    du -sh "$M"/*
    ;;
  check)
    "$VENV/bin/python" -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"
    ls -l "$M/Wan2.1-Fun-V1.1-1.3B-Control" "$M/ckpt"
    ;;
  *) echo "unknown step $step"; exit 1 ;;
esac
echo "=== roborender setup $step done $(date '+%F %T')"
