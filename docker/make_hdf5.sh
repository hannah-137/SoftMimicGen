#!/usr/bin/env bash
# Stage 1 of the Wan pipeline (docker/gen.sh chains all stages). Runs INSIDE the container, softmimicgen env.
# Generates one hdf5 with Isaac Lab Mimic from the annotated source dataset into the run folder.
#   bash docker/make_hdf5.sh <task> <tag> [n_demos=1] [gpu=0]      # no arguments -> asks interactively
# <task> is a key of experiments/wan_canny/tasks.py (e.g. franka_towel). The run folder
# experiments/wan_canny/runs/<task>_<tag>/ receives <task>_<tag>.hdf5 (+ _failed.hdf5) and <task>_<tag>_gen.log.
# GPU: use 0. Camera rendering on cuda:1 hangs at the first sim.reset() in this container (checked 2026-09-13).
set -eo pipefail
cd /workspace/SoftMimicGen
TASKS=$(cd experiments/wan_canny && python -c "import tasks; print(' '.join(tasks.TASKS))")
is_task() { for t in $TASKS; do [ "$1" = "$t" ] && return 0; done; return 1; }

TASK="${1:-}"; TAG="${2:-}"; N="${3:-1}"; GPU="${4:-0}"
if [ $# -eq 0 ]; then
  until is_task "$TASK"; do read -rp "Task [${TASKS// //}]: " TASK; done
  while [ -z "$TAG" ]; do read -rp "Tag (e.g. v6): " TAG; done
  read -rp "Number of demos [1]: " n; N="${n:-1}"
  read -rp "GPU [0]: " g; GPU="${g:-0}"
fi
is_task "$TASK" || { echo "unknown task: '$TASK' (one of: $TASKS)"; exit 1; }
[ -n "$TAG" ] || { echo "usage: bash docker/make_hdf5.sh <task> <tag> [n_demos=1] [gpu=0]"; exit 1; }

ROBOT="${TASK%%_*}"; SUBTASK="${TASK#*_}"
PREFIX="${TASK}_${TAG}"
RUN_DIR="experiments/wan_canny/runs/${PREFIX}"
IN="datasets/annotated_dataset/annotated_dataset_${ROBOT}_${SUBTASK}.hdf5"
OUT="${RUN_DIR}/${PREFIX}.hdf5"
LOG="${RUN_DIR}/${PREFIX}_gen.log"
[ -f "$IN" ] || { echo "input not found: $IN"; exit 1; }
EXTRA=""; [ "$ROBOT" = humanoid ] && EXTRA="--enable_pinocchio"
mkdir -p "$RUN_DIR"
[ -f "$OUT" ] && echo "note: $OUT exists and will be overwritten"
if [ $# -eq 0 ]; then
  read -rp "Generate $TASK -> $OUT on GPU $GPU, $N demo(s)? [Y/n] " go
  [[ "${go:-Y}" =~ ^[Yy]$ ]] || { echo "cancelled"; exit 0; }
fi

echo "hdf5: $TASK -> $OUT on GPU $GPU, $N demo(s)  (log: $LOG)"
echo Yes | PYTHONUNBUFFERED=1 python scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --device "cuda:$GPU" --num_envs 1 --generation_num_trials "$N" $EXTRA \
  --input_file "$IN" --output_file "$OUT" --enable_cameras --headless \
  --kit_args "--/renderer/multiGpu/enabled=false --/renderer/activeGpu=$GPU" \
  2>&1 | tee "$LOG" || true
[ -f "$OUT" ] || { echo "hdf5: generation failed, $OUT not written (see $LOG)"; exit 1; }
echo "hdf5: $OUT"
