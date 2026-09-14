#!/bin/bash
# Qwen-Image-Edit-2511 for ComfyUI (run inside the hyeon_smg container, after setup_comfyui.sh)
# Used by experiments/wan_canny/make_refs.py to turn the simulator frame (images/<prefix>_ref_sim.png) into
# photorealistic reference images for Wan. Same ComfyUI server / conda env as Wan, models next to the Wan models.
# Files (~31 GB): Comfy-Org repackaged fp8 diffusion model (20.5 GB), Qwen2.5-VL 7B fp8 text encoder (9.4 GB),
# Qwen-Image VAE (0.25 GB), lightx2v Lightning 4-step LoRA (0.85 GB, optional fast path, --lightning in make_refs.py).
# Tested: 2026-09-13, RTX 4090 24GB, ComfyUI 0.35.0 (has TextEncodeQwenImageEditPlus).
# Usage: bash docker/setup_qwen_image_edit.sh
set -e

TOOLS=/workspace/tools
MODELS=$TOOLS/comfyui_models
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate comfyui

# 1. models (one file per call, see setup_comfyui.sh)
mkdir -p $MODELS/loras && cd $MODELS
hf download Comfy-Org/Qwen-Image-Edit_ComfyUI split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors --local-dir .
hf download Comfy-Org/Qwen-Image_ComfyUI split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors --local-dir .
hf download Comfy-Org/Qwen-Image_ComfyUI split_files/vae/qwen_image_vae.safetensors --local-dir .
hf download lightx2v/Qwen-Image-Edit-2511-Lightning Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors --local-dir loras

# 2. symlink into ComfyUI
C=$TOOLS/ComfyUI/models
ln -sf $MODELS/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors $C/diffusion_models/
ln -sf $MODELS/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors $C/text_encoders/
ln -sf $MODELS/split_files/vae/qwen_image_vae.safetensors $C/vae/
ln -sf $MODELS/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors $C/loras/

echo "=== Setup done ==="
ls -la $C/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors $C/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors \
      $C/vae/qwen_image_vae.safetensors $C/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors
echo "Web UI template: Templates -> 'Qwen-Image-Edit 2511'.  Pipeline: python experiments/wan_canny/make_refs.py <run_dir> --n_ref N"
