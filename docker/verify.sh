#!/usr/bin/env bash
# Runs INSIDE the container. Verifies the environment end-to-end:
# generate 2 rope demos -> convert to mp4 -> check output files.
# Usage: bash docker/verify.sh
set -eo pipefail

REPO="/workspace/${REPO_NAME:-SoftMimicGen}"
OUT="$REPO/datasets/generated_dataset/verify_rope.hdf5"
VID="$REPO/videos/verify"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate softmimicgen
cd "$REPO"
source docker/gpu.sh
stop_comfy_on_smg_gpu
mkdir -p datasets/generated_dataset logs

echo "=== [1/3] generate 2 rope demos (GPU $SMG_GPU) ==="
echo Yes | python scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  "${ISAAC_ARGS[@]}" --num_envs 1 \
  --generation_num_trials 2 \
  --input_file ./datasets/annotated_dataset/annotated_dataset_franka_rope.hdf5 \
  --output_file "$OUT" \
  --enable_cameras --headless \
  2>&1 | tee logs/verify.txt
  [ -s "$OUT" ] || { echo "FAIL: generation produced no output"; exit 1; }

echo "=== [2/3] convert to mp4 ==="
python docker/make_video.py "$OUT" "$VID"

echo "=== [3/3] check ==="
[ -s "$OUT" ] || { echo "FAIL: no output hdf5"; exit 1; }
ls "$VID"/*.mp4 > /dev/null 2>&1 || { echo "FAIL: no mp4"; exit 1; }
echo "OK. videos in: $VID"