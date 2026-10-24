# Source INSIDE the container before any RoboRender command:   source experiments/roborender/env.sh
# Paths of experiments/roborender/setup.sh. The GPU is chosen per run with CUDA_VISIBLE_DEVICES.
cd /workspace/SoftMimicGen
TOOLS=/workspace/tools
export PATH="$TOOLS/bin:$PATH"
export UV_CACHE_DIR="$TOOLS/uv_cache" UV_PYTHON_INSTALL_DIR="$TOOLS/uv_python" UV_TOOL_DIR="$TOOLS/uv_tools" UV_TOOL_BIN_DIR="$TOOLS/bin"
export HF_HOME="$TOOLS/roborender_models/hf_cache"
export ROBORENDER_SRC="$TOOLS/roborender/src"
export ROBORENDER_MODEL="$TOOLS/roborender_models/Wan2.1-Fun-V1.1-1.3B-Control"
export ROBORENDER_LORA="$TOOLS/roborender_models/ckpt/roborender_lora.safetensors"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$TOOLS/roborender/.venv/bin/activate"
