#!/usr/bin/env bash
# One RoboRender run on one sample folder (made by make_inputs.py). Runs INSIDE the container.
#   bash experiments/roborender/run.sh <sample folder> <output folder> <gpu> [extra infer.py flags]
# Example:
#   bash experiments/roborender/run.sh /workspace/scratch/roborender/samples/demo_003 \
#        /workspace/scratch/roborender/outputs/demo_003 0 --teacache
# Views: room (agentview) and wrist (robot0_eye_in_hand), stacked in this order. Output 416x240 per view, 20 fps
# (our simulator rate), every frame of the input; clips of 81 frames with 1 frame overlap. Log: <output>/run.log
set -euo pipefail
sample="$1"; out="$2"; gpu="$3"; shift 3
source "$(cd "$(dirname "$0")" && pwd)/env.sh"   # env.sh next to this script
mkdir -p "$out"
export CUDA_VISIBLE_DEVICES="$gpu"
cd "$ROBORENDER_SRC"
echo "=== roborender run start $(date '+%F %T') gpu $gpu sample $sample" | tee -a "$out/run.log"
python infer.py --sample_dir "$sample" --views room wrist --output_dir "$out" \
  --model_dir "$ROBORENDER_MODEL" --lora_path "$ROBORENDER_LORA" --height 240 --width 416 --fps 20 "$@" 2>&1 | tee -a "$out/run.log"
echo "=== roborender run done $(date '+%F %T')" | tee -a "$out/run.log"
ls -l "$out"
