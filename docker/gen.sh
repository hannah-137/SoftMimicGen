#!/usr/bin/env bash
# Wan pipeline, all stages for one or more tasks. Runs INSIDE the container (activates the env each stage needs).
#   bash docker/gen.sh <task[,task...]|all> <tag> [--n N] [--gpu N] [--from K] [--only K] [--controls a,b] [--ref PNG] [--force]
#     task      key(s) of experiments/wan_canny/tasks.py, e.g. franka_towel or franka_towel,yam_bag; all = every entry
#     --n       demos per task for stage 1 (default 1)          --gpu    GPU for Isaac Sim (default 0; GPU 1 hangs)
#     --from K  start at stage K (earlier outputs must exist)   --only K run stage K only
#     --controls, --ref, --force  passed to make_wan.py (stage 4)
# Stages, all writing flat into experiments/wan_canny/runs/<task>_<tag>/  (prefix = <task>_<tag>):
#   1 hdf5      docker/make_hdf5.sh                                   (softmimicgen)  <prefix>.hdf5, _gen.log
#   2 source    make_source.py                                        (softmimicgen)  _source.mp4, _ref_sim.png
#   3 edges     make_rgb_canny.py make_shaded_canny.py make_geo_edge.py (softmimicgen)  _canny _shadedcanny _geoedge
#               make_learned_edges.py                                 (rgb_edge)      _hed _pidinet _teed _lineart
#   4 wan       make_wan.py  (ComfyUI on GPU 1, auto-started)         (softmimicgen)  _wan_<control>.mp4 + .json
# Everything printed is also appended to <prefix>_pipeline.log. A failing task does not stop the others; the exit
# code is non-zero if any task failed. Each stage script can be run on its own with the same run folder.
set -uo pipefail
cd /workspace/SoftMimicGen
source /opt/miniconda3/etc/profile.d/conda.sh
WC=experiments/wan_canny
RGB_EDGE_ENV=/workspace/tools/envs/rgb_edge
export PYTHONUNBUFFERED=1 HF_HOME=/workspace/tools/rgb_edge_models

usage() { sed -n '2,8p' "$0"; exit 1; }
[ $# -ge 2 ] || usage
TASK_ARG="$1"; TAG="$2"; shift 2
N=1; GPU=0; FROM=1; ONLY=""; WAN_ARGS=()
while [ $# -gt 0 ]; do case "$1" in
  --n) N="$2"; shift 2;;            --gpu) GPU="$2"; shift 2;;
  --from) FROM="$2"; shift 2;;      --only) ONLY="$2"; shift 2;;
  --controls) WAN_ARGS+=(--controls "$2"); shift 2;;
  --ref) WAN_ARGS+=(--ref "$2"); shift 2;;
  --force) WAN_ARGS+=(--force); shift;;
  *) echo "unknown option: $1"; usage;;
esac; done

conda activate softmimicgen
ALL_TASKS=$(cd $WC && python -c "import tasks; print(' '.join(tasks.TASKS))")
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

declare -A RESULT
for TASK in $TASKS; do
  PREFIX="${TASK}_${TAG}"; RUN="$WC/runs/$PREFIX"; LOG="$RUN/${PREFIX}_pipeline.log"
  mkdir -p "$RUN"
  echo "##### $PREFIX -> $RUN/  ($(date '+%F %T'))" | tee -a "$LOG"
  T0=$(date +%s); ok=1
  conda activate softmimicgen
  if [ $ok = 1 ] && want 1; then stage 1 hdf5 bash docker/make_hdf5.sh "$TASK" "$TAG" "$N" "$GPU" || ok=0; fi
  if [ $ok = 1 ] && want 2; then stage 2 source python $WC/make_source.py "$RUN" || ok=0; fi
  if [ $ok = 1 ] && want 3; then
    stage 3 "rgb canny" python $WC/make_rgb_canny.py "$RUN" || ok=0
    stage 3 "shaded canny" python $WC/make_shaded_canny.py "$RUN" || ok=0
    stage 3 "geo edge" python $WC/make_geo_edge.py "$RUN" || ok=0
    conda activate "$RGB_EDGE_ENV"
    stage 3 "learned edges (rgb_edge env)" python $WC/make_learned_edges.py "$RUN" || ok=0
    conda activate softmimicgen
  fi
  if [ $ok = 1 ] && want 4; then stage 4 wan python $WC/make_wan.py "$RUN" "${WAN_ARGS[@]}" || ok=0; fi
  RESULT[$TASK]=$([ $ok = 1 ] && echo OK || echo FAILED)
  echo "##### $PREFIX ${RESULT[$TASK]} in $(( ($(date +%s) - T0) / 60 )) min -> $RUN/" | tee -a "$LOG"
done

echo; echo "summary (tag $TAG):"; fail=0
for TASK in $TASKS; do printf "  %-24s %s\n" "$TASK" "${RESULT[$TASK]}"; [ "${RESULT[$TASK]}" = OK ] || fail=1; done
exit $fail
