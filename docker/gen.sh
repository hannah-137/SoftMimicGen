#!/usr/bin/env bash
# Wan pipeline, all stages for one or more tasks. Runs INSIDE the container (activates the env each stage needs).
#   bash docker/gen.sh <task[,task...]|all> <tag> [--n N] [--from K] [--only K] [--n_ref N] [--n_obj M] [--n_var K]
#                      [--lightning] [--controls a,b] [--ref PNG] [--force] [--inspect]
#     task      key(s) of experiments/wan_canny/scripts/tasks.py, e.g. franka_towel or franka_towel,yam_bag; all = every entry
#     --n       demos per task for stage 1 (default 1)          GPU: always GPU 1 (docker/gpu.sh), GPU 0 is never used
#     --from K  start at stage K (earlier outputs must exist)   --only K run stage K only
#     --n_ref N  stage 4 first generates N reference images with Qwen-Image-Edit (make_refs.py, images/_ref_ai_01..NN.png:
#                01 faithful, 02.. varied); without it stage 4 expects user-made reference images
#     --n_obj M  M object colour/material videos per control (empty reference made by make_refs.py --empty + prompt, _oNN)
#     --n_var K  K background/table/lighting videos per control (no reference, prompt only, _pNN)
#     --lightning  fast 4-step LoRA for make_refs.py
#     --controls, --ref, --force  passed to make_wan.py (stage 4; --force also to make_refs.py)
#     --inspect  stage 1 also records the inspection channels (<cam>_shaded, ...) -> extra _shaded.mp4 etc. in stage 3
# Stages, all writing into experiments/wan_canny/runs/<tag>/<task>_<tag>/  (prefix = <task>_<tag>): hdf5 + logs at the root,
#   videos in sources/ edges/ wans/, reference images in images/
#   1 hdf5      docker/make_hdf5.sh                                   (softmimicgen)  <prefix>.hdf5, _gen.log
#   2 source    make_source.py                                        (softmimicgen)  _source.mp4, _ref_sim.png
#   3 edges     make_rgb_canny.py make_shaded_canny.py make_geo_edge.py (softmimicgen)  _canny _shadedcanny _geoedge
#               make_shaded_canny_depth.py make_union.py           (softmimicgen)  _shadedcanny_depth _union
#               make_shaded.py make_geo_inputs.py  (--inspect only) (softmimicgen)  _shaded _depth _normals _instance
#               make_learned_edges.py                                 (rgb_edge)      _hed _pidinet _teed _lineart
#   4 refs      make_refs.py (--n_ref only; ComfyUI on GPU 1)         (softmimicgen) images/_ref_ai_01..NN.png + .json
#     wan       make_wan.py  (one queue for all tasks, ComfyUI on GPU 1)  (softmimicgen) _wan_<control><ref>.mp4 + .json
# Order: stages 1-3 run for every task first (Isaac Sim), then stage 4 runs once for the whole batch: the Wan videos of all
# tasks go into one queue on the ComfyUI of GPU 1 (8188), so the 14B model loads once. Everything runs on GPU 1
# (docker/gpu.sh); a ComfyUI on GPU 1 is stopped before stage 1. Batch log: runs/<tag>/<tag>_wan.log.
# Stage 4 needs reference images in images/: <prefix>_ref_ai.png and/or <prefix>_ref_ai_01.png, _02.png, ... (photorealistic
# versions of <prefix>_ref_sim.png, the stage 2 frame, same composition). One Wan video per (reference, control), the
# reference suffix appended (_wan_canny_01.mp4). If there is none the task ends as NEED_REF after stage 3; drop the
# files in and rerun with --from 4 (existing videos are skipped). Pass --ref sim to use the simulator frame on purpose.
# Everything printed is also appended to <prefix>_pipeline.log. A failing task does not stop the others; the exit
# code is non-zero if any task failed (NEED_REF is not a failure). Each stage script can be run on its own.
set -uo pipefail
cd /workspace/SoftMimicGen
source /opt/miniconda3/etc/profile.d/conda.sh
WC=experiments/wan_canny
RGB_EDGE_ENV=/workspace/tools/envs/rgb_edge
export PYTHONUNBUFFERED=1 HF_HOME=/workspace/tools/rgb_edge_models
source docker/gpu.sh

usage() { sed -n '2,8p' "$0"; exit 1; }
[ $# -ge 2 ] || usage
TASK_ARG="$1"; TAG="$2"; shift 2
N=1; FROM=1; ONLY=""; WAN_ARGS=(); REF=""; INSPECT=""; N_REF=0; N_OBJ=0; N_VAR=0; REF_ARGS=()
while [ $# -gt 0 ]; do case "$1" in
  --n) N="$2"; shift 2;;
  --gpu) [ "$2" = "$SMG_GPU" ] || { echo "GPU $2 is not allowed: this project uses GPU $SMG_GPU only (docker/gpu.sh)"; exit 1; }; shift 2;;
  --from) FROM="$2"; shift 2;;      --only) ONLY="$2"; shift 2;;
  --controls) WAN_ARGS+=(--controls "$2"); shift 2;;
  --ref) REF="$2"; WAN_ARGS+=(--ref "$2"); shift 2;;
  --force) WAN_ARGS+=(--force); REF_ARGS+=(--force); shift;;
  --n_ref) N_REF="$2"; shift 2;;         --lightning) REF_ARGS+=(--lightning); shift;;
  --n_obj) N_OBJ="$2"; WAN_ARGS+=(--n_obj "$2"); REF_ARGS+=(--empty); shift 2;;
  --n_var) N_VAR="$2"; WAN_ARGS+=(--n_var "$2"); shift 2;;
  --inspect) INSPECT="--inspect"; shift;;
  *) echo "unknown option: $1"; usage;;
esac; done

conda activate softmimicgen
ALL_TASKS=$(cd $WC/scripts && python -c "import tasks; print(' '.join(tasks.TASKS))")
if [ "$TASK_ARG" = all ]; then TASKS="$ALL_TASKS"; else TASKS="${TASK_ARG//,/ }"; fi
for t in $TASKS; do case " $ALL_TASKS " in *" $t "*) ;; *) echo "unknown task: $t (one of: $ALL_TASKS)"; exit 1;; esac; done

want() { if [ -n "$ONLY" ]; then [ "$ONLY" = "$1" ]; else [ "$1" -ge "$FROM" ]; fi; }

# stage <n> <label> <command...>: run in the current env, tee to the task log, return the command's status
stage() {
  local n="$1" label="$2"; shift 2
  echo "=== [$PREFIX] stage $n: $label  ($(date '+%F %T'))" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || echo "=== [$PREFIX] stage $n FAILED (exit $rc)" | tee -a "$LOG"
  return "$rc"
}

# Stage 4 ComfyUI: one server on SMG_GPU (port 8188, the same server the web UI / docker/comfy.sh use).
COMFY_URLS="http://127.0.0.1:8188"; COMFY_GPUS="$SMG_GPU"; MAIN_URL="$COMFY_URLS"; MAIN_GPU="$SMG_GPU"

declare -A RESULT
# ---------------------------------------------------------------- stages 1-3: every task first (Isaac Sim on GPU $SMG_GPU)
if want 1 || want 2 || want 3; then
  want 1 && stop_comfy_on_smg_gpu
  for TASK in $TASKS; do
    PREFIX="${TASK}_${TAG}"; RUN="$WC/runs/$TAG/$PREFIX"; LOG="$RUN/${PREFIX}_pipeline.log"
    mkdir -p "$RUN"
    echo "##### $PREFIX stages 1-3 -> $RUN/  ($(date '+%F %T'))" | tee -a "$LOG"
    T0=$(date +%s); ok=1
    conda activate softmimicgen
    if [ $ok = 1 ] && want 1; then stage 1 hdf5 bash docker/make_hdf5.sh "$TASK" "$TAG" "$N" $INSPECT || ok=0; fi
    if [ $ok = 1 ] && want 2; then stage 2 source python $WC/scripts/make_source.py "$RUN" || ok=0; fi
    if [ $ok = 1 ] && want 3; then
      stage 3 "rgb canny" python $WC/scripts/make_rgb_canny.py "$RUN" || ok=0
      stage 3 "shaded canny" python $WC/scripts/make_shaded_canny.py "$RUN" || ok=0
      stage 3 "geo edge" python $WC/scripts/make_geo_edge.py "$RUN" || ok=0
      stage 3 "shaded canny depth" python $WC/scripts/make_shaded_canny_depth.py "$RUN" || ok=0
      stage 3 "union" python $WC/scripts/make_union.py "$RUN" || ok=0
      stage 3 "shaded (inspection, only with --inspect)" python $WC/scripts/make_shaded.py "$RUN" || ok=0
      stage 3 "geo inputs (inspection, only with --inspect)" python $WC/scripts/make_geo_inputs.py "$RUN" || ok=0
      conda activate "$RGB_EDGE_ENV"
      stage 3 "learned edges (rgb_edge env)" python $WC/scripts/make_learned_edges.py "$RUN" || ok=0
      conda activate softmimicgen
    fi
    [ $ok = 1 ] || RESULT[$TASK]=FAILED
    chown -R --reference="$WC" "$RUN" 2>/dev/null || true
    echo "##### $PREFIX stages 1-3 $([ $ok = 1 ] && echo OK || echo FAILED) in $(( ($(date +%s) - T0) / 60 )) min" | tee -a "$LOG"
  done
fi

# ---------------------------------------------------------------- stage 4: references per task, then one Wan queue on every GPU
if want 4; then
  conda activate softmimicgen
  READY=()
  for TASK in $TASKS; do
    [ "${RESULT[$TASK]:-}" = FAILED ] && continue
    PREFIX="${TASK}_${TAG}"; RUN="$WC/runs/$TAG/$PREFIX"; LOG="$RUN/${PREFIX}_pipeline.log"
    if [ "$N_REF" -gt 0 ] || [ "$N_OBJ" -gt 0 ]; then
      stage 4 "refs (Qwen-Image-Edit, $N_REF images$([ "$N_OBJ" -gt 0 ] && echo ' + empty'))" \
        python $WC/scripts/make_refs.py "$RUN" --n_ref "$N_REF" "${REF_ARGS[@]}" --comfy_url "$MAIN_URL" --comfy_gpu "$MAIN_GPU" \
        || { RESULT[$TASK]=FAILED; continue; }
    fi
    if [ -z "$REF" ] && [ "$N_OBJ" = 0 ] && [ "$N_VAR" = 0 ] && ! compgen -G "$RUN/images/${PREFIX}_ref_ai*.png" > /dev/null; then
      RESULT[$TASK]=NEED_REF
      echo "=== [$PREFIX] stage 4 skipped: NEED_REF. Make photorealistic versions of $RUN/images/${PREFIX}_ref_sim.png (same composition)," | tee -a "$LOG"
      echo "    save them as $RUN/images/${PREFIX}_ref_ai.png or ${PREFIX}_ref_ai_01.png, _02.png, ..., then: bash docker/gen.sh $TASK $TAG --from 4" | tee -a "$LOG"
      continue
    fi
    READY+=("$RUN")
  done
  if [ ${#READY[@]} -gt 0 ]; then
    BATCH_LOG="$WC/runs/$TAG/${TAG}_wan.log"
    echo "=== stage 4 wan: ${#READY[@]} task(s) in one queue on $COMFY_URLS (GPU $COMFY_GPUS), log $BATCH_LOG  ($(date '+%F %T'))" | tee -a "$BATCH_LOG"
    python $WC/scripts/make_wan.py "${READY[@]}" "${WAN_ARGS[@]}" --comfy_url "$COMFY_URLS" --comfy_gpu "$COMFY_GPUS" 2>&1 | tee -a "$BATCH_LOG"
    for RUN in "${READY[@]}"; do
      PREFIX=$(basename "$RUN"); TASK=${PREFIX%_"$TAG"}
      grep -h -- " ${PREFIX}:" "$BATCH_LOG" >> "$RUN/${PREFIX}_pipeline.log" 2>/dev/null  # this task's lines of the batch log
      if grep -q "wan: .* ${PREFIX}:[^ ]*: FAILED" "$BATCH_LOG"; then RESULT[$TASK]=FAILED; else RESULT[$TASK]=OK; fi
      chown -R --reference="$WC" "$RUN" 2>/dev/null || true
    done
  fi
fi

echo; echo "summary (tag $TAG):"; fail=0
for TASK in $TASKS; do r=${RESULT[$TASK]:-OK}; printf "  %-24s %s\n" "$TASK" "$r"; [ "$r" = FAILED ] && fail=1; done
exit $fail
