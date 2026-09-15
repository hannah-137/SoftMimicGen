# Sourced by every container-side script (gen.sh, make_hdf5.sh, verify.sh). The shared lab server gives this project
# GPU 1 only; GPU 0 belongs to other users (2026-09-14). Nothing here may touch GPU 0.
SMG_GPU=1
# every CUDA program started from these scripts sees only SMG_GPU, as cuda:0
export CUDA_VISIBLE_DEVICES=$SMG_GPU
# Isaac Sim: CUDA sees only SMG_GPU (cuda:0), while the Vulkan renderer still lists every GPU, so activeGpu takes the
# physical index. With both GPUs visible to CUDA, "--device cuda:1 activeGpu=1" hung at the first render (a stray CUDA
# context on GPU 0); with CUDA_VISIBLE_DEVICES=1 the same run finishes (2026-09-14, franka_towel, 57 s).
ISAAC_ARGS=(--device cuda:0 --kit_args "--/renderer/multiGpu/enabled=false --/renderer/activeGpu=$SMG_GPU")
# stop pipeline ComfyUI servers on SMG_GPU (Wan 14B holds 10-20 GB, Isaac Sim cannot start next to it)
stop_comfy_on_smg_gpu() {
  local pids; pids=$(ps -eo pid,args | awk -v g="$SMG_GPU" '/[m]ain\.py --listen/ && $0 ~ ("--cuda-device " g "( |$)") {print $1}')
  [ -z "$pids" ] && return 0
  echo "=== stopping ComfyUI on GPU $SMG_GPU (pid $pids) before Isaac Sim (the web UI on 8188 goes down too; make_wan.py restarts it)"
  kill $pids 2>/dev/null; for _ in $(seq 1 30); do ps -p ${pids// /,} >/dev/null 2>&1 || return 0; sleep 1; done
  kill -9 $pids 2>/dev/null || true
}
