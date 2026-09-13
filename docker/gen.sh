#!/usr/bin/env bash
# Runs INSIDE the container (softmimicgen env). Interactive: asks robot/task/tag/gpu/n, then generates one hdf5.
#   -> datasets/generated_dataset/<robot>_<task>_<tag>.hdf5 , log in logs/gen_<robot>_<task>_<tag>.txt
set -eo pipefail
cd /workspace/SoftMimicGen
mkdir -p logs datasets/generated_dataset

ROBOTS="franka humanoid surgical yam"
tasks_for() { case "$1" in
  franka)   echo "rope towel jenga stack";;
  humanoid) echo "teddy towel";;
  surgical) echo "tissue threading";;
  yam)      echo "towel bag";;
esac; }

ask_choice() {  # ask_choice "prompt" "opt1 opt2 ..."
  local ans
  while true; do
    read -rp "$1 [${2// //}]: " ans
    for o in $2; do [ "$ans" = "$o" ] && { echo "$ans"; return; }; done
    echo "  -> choose one of: $2" >&2
  done
}

ROBOT=$(ask_choice "Robot" "$ROBOTS")
TASK=$(ask_choice "Task for $ROBOT" "$(tasks_for "$ROBOT")")
while [ -z "$TAG" ]; do read -rp "Tag (e.g. 512_bg): " TAG; done
read -rp "GPU [0]: " GPU; GPU="${GPU:-0}"
read -rp "Number of demos [1]: " N; N="${N:-1}"

IN="datasets/annotated_dataset/annotated_dataset_${ROBOT}_${TASK}.hdf5"
OUT="datasets/generated_dataset/${ROBOT}_${TASK}_${TAG}.hdf5"
[ -f "$IN" ] || { echo "input not found: $IN"; exit 1; }
[ -f "$OUT" ] && echo "note: $OUT exists and will be overwritten"

read -rp "Generate $ROBOT/$TASK -> $(basename "$OUT") on GPU $GPU, $N demo(s)? [Y/n] " go
[[ "${go:-Y}" =~ ^[Yy]$ ]] || { echo "cancelled"; exit 0; }

echo Yes | PYTHONUNBUFFERED=1 python scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --device "cuda:$GPU" --num_envs 1 --generation_num_trials "$N" \
  --input_file "$IN" --output_file "$OUT" --enable_cameras --headless \
  --kit_args "--/renderer/multiGpu/enabled=false --/renderer/activeGpu=$GPU" \
  2>&1 | tee "logs/gen_${ROBOT}_${TASK}_${TAG}.txt"
echo "output: $OUT"

# Canny + source video
PREFIX="${ROBOT}_${TASK}_${TAG}"
python experiments/wan_canny/make_videos.py "$OUT"
rm -rf "experiments/wan_canny/inputs_${PREFIX}"
mv experiments/wan_canny/inputs "experiments/wan_canny/inputs_${PREFIX}"
echo "videos: experiments/wan_canny/inputs_${PREFIX}/"