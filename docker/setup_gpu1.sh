#!/bin/bash
# GPU 1 only (run inside the hyeon_smg container). The shared lab server gives this project GPU 1; GPU 0 belongs to other
# users (2026-09-14). This installs conda activation hooks so that every `conda activate` of the project envs sets
# CUDA_VISIBLE_DEVICES=1: anything started from an activated env (Isaac Sim, torch, ComfyUI, soft-edge detectors) sees only
# GPU 1, as cuda:0, even when it is run by hand outside docker/gen.sh. Deactivation restores the previous value.
# The pipeline scripts also pin GPU 1 on their own (docker/gpu.sh). Processes started with an env's python by full path,
# without `conda activate`, do not run the hook (make_wan.py starts ComfyUI that way, with --cuda-device 1).
# Usage: bash docker/setup_gpu1.sh      (idempotent)
set -e
ENVS="/opt/miniconda3/envs/softmimicgen /opt/miniconda3/envs/comfyui /workspace/tools/envs/rgb_edge"
for env in $ENVS; do
  [ -d "$env" ] || { echo "skip (missing): $env"; continue; }
  mkdir -p "$env/etc/conda/activate.d" "$env/etc/conda/deactivate.d"
  cat > "$env/etc/conda/activate.d/smg_gpu1.sh" <<'EOF'
# SoftMimicGen: this project uses GPU 1 only (docker/setup_gpu1.sh)
if [ -z "${_SMG_GPU1_ACTIVE:-}" ]; then
  export _SMG_OLD_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-__unset__}"
  export _SMG_GPU1_ACTIVE=1
fi
export CUDA_VISIBLE_DEVICES=1
EOF
  cat > "$env/etc/conda/deactivate.d/smg_gpu1.sh" <<'EOF'
# SoftMimicGen: restore CUDA_VISIBLE_DEVICES from before activation (docker/setup_gpu1.sh)
if [ -n "${_SMG_GPU1_ACTIVE:-}" ]; then
  if [ "${_SMG_OLD_CUDA_VISIBLE_DEVICES:-}" = "__unset__" ]; then unset CUDA_VISIBLE_DEVICES; else export CUDA_VISIBLE_DEVICES="$_SMG_OLD_CUDA_VISIBLE_DEVICES"; fi
  unset _SMG_OLD_CUDA_VISIBLE_DEVICES _SMG_GPU1_ACTIVE
fi
EOF
  echo "installed GPU 1 hook: $env"
done
echo "=== Setup done: conda activate <env> now sets CUDA_VISIBLE_DEVICES=1"
