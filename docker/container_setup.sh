#!/usr/bin/env bash
# Runs INSIDE the container. Called by create_container.sh.
# Reproduces the manually verified setup: libs -> Vulkan/EGL -> GPU check -> miniconda -> softmimicgen.sh
set -eo pipefail
export DEBIAN_FRONTEND=noninteractive

REPO="/workspace/${REPO_NAME:-SoftMimicGen}"
[ -d "$REPO" ] || { echo "[!] repo not found at $REPO"; exit 1; }

echo "=== [1/6] system libraries ==="
apt-get update
apt-get install -y --no-install-recommends \
  curl wget git ca-certificates unzip build-essential cmake \
  libatomic1 libegl1 libgl1 libglu1-mesa libglx0 libgomp1 \
  libsm6 libxi6 libxrandr2 libxt6 libglib2.0-0 libnghttp2-14 vulkan-tools
rm -rf /var/lib/apt/lists/*

echo "=== [2/6] NVIDIA Vulkan / EGL registration ==="
mkdir -p /usr/share/glvnd/egl_vendor.d /etc/vulkan/icd.d /etc/vulkan/implicit_layer.d
printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libEGL_nvidia.so.0"\n    }\n}\n' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libGLX_nvidia.so.0",\n        "api_version" : "1.3.194"\n    }\n}\n' > /etc/vulkan/icd.d/nvidia_icd.json
printf '{\n    "file_format_version" : "1.0.0",\n    "layer": {\n        "name": "VK_LAYER_NV_optimus",\n        "type": "INSTANCE",\n        "library_path": "libGLX_nvidia.so.0",\n        "api_version" : "1.3.194",\n        "implementation_version" : "1",\n        "description" : "NVIDIA Optimus layer",\n        "functions": {\n            "vkGetInstanceProcAddr": "vk_optimusGetInstanceProcAddr",\n            "vkGetDeviceProcAddr": "vk_optimusGetDeviceProcAddr"\n        },\n        "enable_environment": {\n            "__NV_PRIME_RENDER_OFFLOAD": "1"\n        },\n        "disable_environment": {\n            "DISABLE_LAYER_NV_OPTIMUS_1": ""\n        }\n    }\n}\n' > /etc/vulkan/implicit_layer.d/nvidia_layers.json
export VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json

echo "=== [3/6] Vulkan GPU check ==="
if vulkaninfo --summary 2>&1 | grep -i deviceName | grep -q NVIDIA; then
  echo "OK: NVIDIA GPU detected by Vulkan"
else
  echo "FAIL: Vulkan cannot see NVIDIA GPU. Check host driver / nvidia-container-toolkit. Aborting."
  exit 1
fi

echo "=== [4/6] miniconda ==="
if [ -x /opt/miniconda3/bin/conda ]; then
  echo "already installed, skipping"
else
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p /opt/miniconda3
  rm /tmp/miniconda.sh
fi
export PATH=/opt/miniconda3/bin:$PATH
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

echo "=== [5/6] shell auto-setup (.bashrc) ==="
if grep -q "SOFTMIMICGEN_SETUP" /root/.bashrc 2>/dev/null; then
  echo "already configured, skipping"
else
  cat >> /root/.bashrc << EOF
# SOFTMIMICGEN_SETUP
export PATH=/opt/miniconda3/bin:\$PATH
export VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json
source /opt/miniconda3/etc/profile.d/conda.sh
EOF
fi

echo "=== [6/6] SoftMimicGen install (softmimicgen.sh) ==="
export PIP_CACHE_DIR="$REPO/.pip_cache"   # reuse downloads on re-run
cd "$REPO"
bash softmimicgen.sh

echo "=== setup complete ==="