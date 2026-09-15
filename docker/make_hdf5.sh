#!/usr/bin/env bash
# Stage 1 of the Wan pipeline (docker/gen.sh chains all stages). Runs INSIDE the container, softmimicgen env.
# Generates one hdf5 with Isaac Lab Mimic from the annotated source dataset into the run folder.
#   bash docker/make_hdf5.sh <task> <tag> [n_demos=1] [--inspect]   # no arguments -> asks interactively
# <task> is a key of experiments/wan_canny/scripts/tasks.py (e.g. franka_towel). The run folder
# experiments/wan_canny/runs/<tag>/<task>_<tag>/ receives <task>_<tag>.hdf5 (+ _failed.hdf5) and <task>_<tag>_gen.log.
# GPU: always GPU 1 (docker/gpu.sh; GPU 0 belongs to other users). A ComfyUI on GPU 1 is stopped first.
set -eo pipefail
cd /workspace/SoftMimicGen
source docker/gpu.sh
TASKS=$(cd experiments/wan_canny/scripts && python -c "import tasks; print(' '.join(tasks.TASKS))")
is_task() { for t in $TASKS; do [ "$1" = "$t" ] && return 0; done; return 1; }

# --inspect anywhere in the arguments: also record the inspection channels (<cam>_shaded, ...) in the hdf5
INSPECT=0; ARGS=()
for a in "$@"; do if [ "$a" = "--inspect" ]; then INSPECT=1; else ARGS+=("$a"); fi; done
set -- "${ARGS[@]}"
TASK="${1:-}"; TAG="${2:-}"; N="${3:-1}"
[ -n "${4:-}" ] && [ "$4" != "$SMG_GPU" ] && { echo "GPU $4 is not allowed: this project uses GPU $SMG_GPU only (docker/gpu.sh)"; exit 1; }
if [ $# -eq 0 ]; then
  until is_task "$TASK"; do read -rp "Task [${TASKS// //}]: " TASK; done
  while [ -z "$TAG" ]; do read -rp "Tag (e.g. v6): " TAG; done
  read -rp "Number of demos [1]: " n; N="${n:-1}"
fi
is_task "$TASK" || { echo "unknown task: '$TASK' (one of: $TASKS)"; exit 1; }
[ -n "$TAG" ] || { echo "usage: bash docker/make_hdf5.sh <task> <tag> [n_demos=1] [--inspect]"; exit 1; }

ROBOT="${TASK%%_*}"; SUBTASK="${TASK#*_}"
PREFIX="${TASK}_${TAG}"
RUN_DIR="experiments/wan_canny/runs/${TAG}/${PREFIX}"
IN="datasets/annotated_dataset/annotated_dataset_${ROBOT}_${SUBTASK}.hdf5"
OUT="${RUN_DIR}/${PREFIX}.hdf5"
LOG="${RUN_DIR}/${PREFIX}_gen.log"
[ -f "$IN" ] || { echo "input not found: $IN"; exit 1; }
EXTRA=""; [ "$ROBOT" = humanoid ] && EXTRA="--enable_pinocchio"
mkdir -p "$RUN_DIR"
[ -f "$OUT" ] && echo "note: $OUT exists and will be overwritten"
if [ $# -eq 0 ]; then
  read -rp "Generate $TASK -> $OUT on GPU $SMG_GPU, $N demo(s)? [Y/n] " go
  [[ "${go:-Y}" =~ ^[Yy]$ ]] || { echo "cancelled"; exit 0; }
fi

[ "$INSPECT" = 1 ] && export WAN_INSPECT=1 && echo "hdf5: --inspect: recording the inspection channels too"
stop_comfy_on_smg_gpu
echo "hdf5: $TASK -> $OUT on GPU $SMG_GPU, $N demo(s)  (log: $LOG)"
echo Yes | PYTHONUNBUFFERED=1 python scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  "${ISAAC_ARGS[@]}" --num_envs 1 --generation_num_trials "$N" $EXTRA \
  --input_file "$IN" --output_file "$OUT" --enable_cameras --headless \
  2>&1 | tee "$LOG" || true
chown -R --reference=experiments/wan_canny "$RUN_DIR" 2>/dev/null || true  # container is root; give the folder to the repo owner
[ -f "$OUT" ] || { echo "hdf5: generation failed, $OUT not written (see $LOG)"; exit 1; }
echo "hdf5: $OUT"
