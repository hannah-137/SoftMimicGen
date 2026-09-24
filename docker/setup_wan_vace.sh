#!/bin/bash
# Wan2.2-VACE-Fun-A14B for ComfyUI (run inside the container, after setup_comfyui.sh)
# Same MoE base as the Wan 2.2 Fun-Control we use (high/low-noise 14B experts), but with the VACE control path:
# control video + optional reference image go through the native WanVaceToVideo node (present in ComfyUI 1d48d9cf).
# Text encoder and VAE are the Fun-Control ones (umt5_xxl fp8, wan_2.1_vae), already installed by setup_comfyui.sh.
# Files (~35 GB): Comfy-Org repackaged fp8 experts, 17.35 GB each. bf16 (34.7 GB each) is available in the same repo
# if a full-precision run is wanted later: pass --bf16.
# Usage: bash docker/setup_wan_vace.sh [--bf16]
set -e

TOOLS=/workspace/tools
MODELS=$TOOLS/comfyui_models
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate comfyui

# 1. models (one file per call, see setup_comfyui.sh)
mkdir -p $MODELS && cd $MODELS
files="split_files/diffusion_models/wan2.2_fun_vace_high_noise_14B_fp8_scaled.safetensors
       split_files/diffusion_models/wan2.2_fun_vace_low_noise_14B_fp8_scaled.safetensors"
[ "${1:-}" = "--bf16" ] && files="$files
       split_files/diffusion_models/wan2.2_fun_vace_high_noise_14B_bf16.safetensors
       split_files/diffusion_models/wan2.2_fun_vace_low_noise_14B_bf16.safetensors"
for f in $files; do
  hf download Comfy-Org/Wan_2.2_ComfyUI_Repackaged "$f" --local-dir .
done

# 2. symlink into ComfyUI
C=$TOOLS/ComfyUI/models
for f in $files; do
  ln -sf $MODELS/$f $C/diffusion_models/
done

echo "=== Setup done ==="
ls -la $C/diffusion_models/wan2.2_fun_vace_*.safetensors
echo "Graph: WanVaceToVideo (control_video, reference_image) -> KSamplerAdvanced high/low -> TrimVideoLatent -> VAEDecode"
