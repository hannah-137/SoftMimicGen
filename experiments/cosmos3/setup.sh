#!/usr/bin/env bash
# Cosmos 3 environment for the separate Cosmos experiment (experiments/cosmos3). Runs INSIDE the container (root).
#   bash experiments/cosmos3/setup.sh        # install everything; safe to re-run, finished steps are skipped
# Touches no existing file and no existing conda env. Everything goes under /workspace/tools:
#   tools/bin/uv, tools/bin/hf              uv + Hugging Face CLI
#   tools/uv_cache, tools/uv_python         uv wheel cache, uv-managed Python 3.13 (cosmos-framework pins 3.13)
#   tools/cosmos-framework (+ .venv)        NVIDIA cosmos-framework, the inference code the nvidia/cosmos cookbook calls
#   tools/cosmos3_models/Cosmos3-Nano-fp8   nvidia/Cosmos3-Nano, branch fp8 (ModelOpt fp8, ~21 GB)
#   tools/cosmos3_models/hf_cache           HF_HOME for the small files fetched at run time (Qwen3-VL processor)
# GPU: docker/gpu.sh (GPU 1 only). The install itself needs no GPU; only the last check looks at it.
# Log: experiments/cosmos3/setup.log (appended). Afterwards: source experiments/cosmos3/env.sh
set -euo pipefail
cd /workspace/SoftMimicGen
source docker/gpu.sh
TOOLS=/workspace/tools
HERE=experiments/cosmos3
mkdir -p "$HERE" "$TOOLS/bin" "$TOOLS/cosmos3_models"
exec > >(tee -a "$HERE/setup.log") 2>&1
echo "=== cosmos3 setup start $(date '+%F %T')"

export PATH="$TOOLS/bin:$PATH"
export UV_INSTALL_DIR="$TOOLS/bin" UV_CACHE_DIR="$TOOLS/uv_cache" UV_PYTHON_INSTALL_DIR="$TOOLS/uv_python"
export UV_TOOL_DIR="$TOOLS/uv_tools" UV_TOOL_BIN_DIR="$TOOLS/bin"
export HF_HOME="$TOOLS/cosmos3_models/hf_cache"
export LD_LIBRARY_PATH=

# 1. apt packages cosmos-framework asks for (curl/git/wget are already in the container)
need=""
for p in ffmpeg git-lfs libx11-dev; do dpkg -s "$p" >/dev/null 2>&1 || need="$need $p"; done
if [ -n "$need" ]; then
  echo "=== apt install:$need"
  apt-get update && apt-get install -y --no-install-recommends $need
fi

# 2. uv (installer respects UV_INSTALL_DIR) and the hf CLI
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
command -v hf >/dev/null 2>&1 || uv tool install huggingface_hub
hf version

# 3. cosmos-framework source, pinned to the commit used on smg (2026-09-18); change the hash here to upgrade
FW="$TOOLS/cosmos-framework"
[ -d "$FW/.git" ] || git clone https://github.com/NVIDIA/cosmos-framework.git "$FW"
git -C "$FW" checkout -q c23e51f2f157ae3e51cfcd86ebfb5464850894f2
cd "$FW"
git rev-parse HEAD | tee COMMIT.txt

# 4. Python 3.13 venv with the CUDA 12.8 wheels (torch 2.10, flash-attn 2, natten, transformer-engine)
echo "=== uv sync (cu128-train)"
uv sync --all-extras --group=cu128-train
source .venv/bin/activate
export LD_LIBRARY_PATH=
# transformer-engine (cu128 wheel) looks for CUDA libs in a system toolkit first; this container has none, and its pip-package
# fallback expects an "nvidia/cudart" folder that the cu12 pip packages do not have (they ship "nvidia/cuda_runtime").
# Pointing the *_HOME variables at the venv's pip CUDA libs makes TE take the "system" path and import cleanly (verified 2026-09-19).
NV="$FW/.venv/lib/python3.13/site-packages/nvidia"
export NVRTC_HOME="$NV/cuda_nvrtc" CURAND_HOME="$NV/curand" CUDNN_HOME="$NV/cudnn"
# torchcodec (video decode) and other extensions link the pip-installed CUDA libs (npp, nvrtc, cudart) by soname, so the venv's
# nvidia/*/lib dirs must be on LD_LIBRARY_PATH. Only venv paths are added; host CUDA libraries stay out (verified 2026-09-19).
export LD_LIBRARY_PATH="$(ls -d "$NV"/*/lib | tr '\n' ':')"

# 5. Cosmos3-Nano fp8 checkpoint (not gated, no token). Example videos/images of the repo are skipped.
CK="$TOOLS/cosmos3_models/Cosmos3-Nano-fp8"
echo "=== checkpoint -> $CK"
hf download nvidia/Cosmos3-Nano --revision fp8 --local-dir "$CK" --exclude "assets/*" "images/*"
du -sh "$CK"

# 6. checks: wheels import, GPU 1 visible as cuda:0, inference CLI loads
python - <<'EOF'
import torch, flash_attn, transformer_engine
from torchcodec.decoders import VideoDecoder   # needs the venv nvidia libs on LD_LIBRARY_PATH
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0), "| cap", torch.cuda.get_device_capability(0))
EOF
python -m cosmos_framework.scripts.inference --help > /tmp/cosmos3_help.txt && head -3 /tmp/cosmos3_help.txt && echo "=== inference CLI ok"
cd /workspace/SoftMimicGen
chown -R --reference=experiments/wan_canny "$HERE" 2>/dev/null || true   # container is root; give the folder to the repo owner
echo "=== cosmos3 setup done $(date '+%F %T')"
