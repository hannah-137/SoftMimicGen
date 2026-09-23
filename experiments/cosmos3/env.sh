# Source INSIDE the container before any Cosmos 3 command:   source experiments/cosmos3/env.sh
# GPU 1 only (docker/gpu.sh), cosmos-framework venv, model and cache paths. Installed by experiments/cosmos3/setup.sh.
cd /workspace/SoftMimicGen
source docker/gpu.sh
TOOLS=/workspace/tools
export PATH="$TOOLS/bin:$PATH"
export UV_CACHE_DIR="$TOOLS/uv_cache" UV_PYTHON_INSTALL_DIR="$TOOLS/uv_python" UV_TOOL_DIR="$TOOLS/uv_tools" UV_TOOL_BIN_DIR="$TOOLS/bin"
export HF_HOME="$TOOLS/cosmos3_models/hf_cache"
export COSMOS_FW="$TOOLS/cosmos-framework"
export COSMOS3_CKPT="$TOOLS/cosmos3_models/Cosmos3-Nano-fp8"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$COSMOS_FW/.venv/bin/activate"
export LD_LIBRARY_PATH=
# transformer-engine (cu128 wheel) looks for CUDA libs in a system toolkit first; this container has none, and its pip-package
# fallback expects an "nvidia/cudart" folder that the cu12 pip packages do not have (they ship "nvidia/cuda_runtime").
# Pointing the *_HOME variables at the venv's pip CUDA libs makes TE take the "system" path and import cleanly (verified 2026-09-19).
NV="$COSMOS_FW/.venv/lib/python3.13/site-packages/nvidia"
export NVRTC_HOME="$NV/cuda_nvrtc" CURAND_HOME="$NV/curand" CUDNN_HOME="$NV/cudnn"
# torchcodec (video decode) and other extensions link the pip-installed CUDA libs (npp, nvrtc, cudart) by soname, so the venv's
# nvidia/*/lib dirs must be on LD_LIBRARY_PATH. Only venv paths are added; host CUDA libraries stay out (verified 2026-09-19).
export LD_LIBRARY_PATH="$(ls -d "$NV"/*/lib | tr '\n' ':')"
