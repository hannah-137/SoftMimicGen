#!/bin/bash
# Wan Fun Control + ComfyUI setup (run inside the hyeon_smg container)
# Models: Wan2.2 Fun Control 14B fp8 + Wan2.1 Fun Control 1.3B (CRAFT baseline)
# Tested: 2026-09-11, RTX 4090 24GB, driver 550.127.05
# Usage: bash setup_comfyui.sh
set -e

TOOLS=/workspace/tools
MODELS=$TOOLS/comfyui_models
CACHE=/workspace/SoftMimicGen/.pip_cache

# 1. conda env (python 3.12 = ComfyUI supported version)
source /opt/miniconda3/etc/profile.d/conda.sh
conda env list | grep -q "^comfyui " || conda create -n comfyui python=3.12 -y
conda activate comfyui

# 2. ComfyUI
mkdir -p $TOOLS && cd $TOOLS
[ -d ComfyUI ] || git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# 3. torch cu128 first (ComfyUI needs newer torch than cu124 build; cu128 works on driver 550)
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --cache-dir $CACHE
pip install -r requirements.txt --cache-dir $CACHE

# 4. Wan2.2 Fun Control models (~34GB total)
# NOTE: one file per download call; passing multiple paths after --include silently drops files
mkdir -p $MODELS && cd $MODELS
for f in \
  split_files/diffusion_models/wan2.2_fun_control_high_noise_14B_fp8_scaled.safetensors \
  split_files/diffusion_models/wan2.2_fun_control_low_noise_14B_fp8_scaled.safetensors \
  split_files/vae/wan_2.1_vae.safetensors \
  split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors ; do
  hf download Comfy-Org/Wan_2.2_ComfyUI_Repackaged "$f" --local-dir .
done

# 4b. Wan2.1 Fun Control 1.3B (~3GB) — same model CRAFT used, for comparison
hf download Comfy-Org/Wan_2.1_ComfyUI_repackaged \
  split_files/diffusion_models/wan2.1_fun_control_1.3B_bf16.safetensors --local-dir .

# 5. symlink models into ComfyUI
C=$TOOLS/ComfyUI/models
ln -sf $MODELS/split_files/diffusion_models/*.safetensors $C/diffusion_models/
ln -sf $MODELS/split_files/vae/wan_2.1_vae.safetensors $C/vae/
ln -sf $MODELS/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors $C/text_encoders/

echo "=== Setup done ==="
echo "Start server:  conda activate comfyui && cd $TOOLS/ComfyUI && nohup python main.py --listen 0.0.0.0 --port 8188 > comfyui.log 2>&1 &"
echo "Tunnel (mac):  ssh -L 8188:<container-ip>:8188 hyeon@<server-ip>   # container ip: docker inspect hyeon_smg | grep IPAddress"
echo "Browser:       http://localhost:8188  ->  Templates -> 'Wan 2.2 14B Fun Control' or 'Wan 2.1 Fun Control'"