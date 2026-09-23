#!/bin/bash
# Runs on the HOST. Starts a Jupyter server inside the hyeon_smg container (skips if already running) and prints the
# URL to paste into VSCode ("Select Kernel" -> "Existing Jupyter Server"). Kernel = softmimicgen env, GPU 1 only.
# The container has no published ports; the host reaches it directly at its bridge IP (172.17.0.x), like ComfyUI.
# Requires docker/setup_jupyter.sh once.
# Usage: bash docker/jupyter.sh [root_dir]      root_dir inside the container, default /workspace (kernels start there)
#        bash docker/jupyter.sh stop
PORT=8888
PY=/opt/miniconda3/envs/softmimicgen/bin
IP=$(docker inspect hyeon_smg -f '{{.NetworkSettings.IPAddress}}')

if [ "$1" = "stop" ]; then
  docker exec hyeon_smg pkill -f "jupyter server --ip" && echo "Jupyter stopped." || echo "Jupyter was not running."
  exit 0
fi

ROOT=${1:-/workspace}
if curl -s -o /dev/null --max-time 3 "http://$IP:$PORT/api"; then
  echo "Jupyter already running."
else
  docker exec -d hyeon_smg bash -c "cd $ROOT && CUDA_VISIBLE_DEVICES=1 $PY/jupyter server --ip 0.0.0.0 --port $PORT --no-browser --allow-root \
    --ServerApp.root_dir=$ROOT --ServerApp.allow_remote_access=True >> /workspace/SoftMimicGen/logs/jupyter.log 2>&1"
  echo "Starting Jupyter (root $ROOT)..."
  for _ in $(seq 1 20); do curl -s -o /dev/null --max-time 2 "http://$IP:$PORT/api" && break; sleep 1; done
fi

TOKEN=$(docker exec hyeon_smg $PY/jupyter server list 2>/dev/null | grep -o 'token=[0-9a-f]*' | head -1)
echo "URL for VSCode (Existing Jupyter Server): http://$IP:$PORT/?$TOKEN"
