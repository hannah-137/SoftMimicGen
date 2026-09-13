#!/bin/bash
# Runs on the HOST. Starts ComfyUI inside the hyeon_smg container (skips if already running).
# GPU 1 on purpose: Isaac Sim only renders on GPU 0 in this container and Wan 14B peaks at ~20 GB, so the two must not
# share a GPU. experiments/wan_canny/make_wan.py starts the same server itself (inside the container) when it is down.
if docker exec hyeon_smg curl -s -o /dev/null http://localhost:8188; then
  echo "ComfyUI already running."
else
  docker exec -d hyeon_smg bash -c "cd /workspace/tools/ComfyUI && /opt/miniconda3/envs/comfyui/bin/python main.py --listen 0.0.0.0 --port 8188 --cuda-device 1 >> /workspace/SoftMimicGen/logs/comfyui.log 2>&1"
  echo "Starting ComfyUI..."; sleep 8
fi
echo "Container IP: $(docker inspect hyeon_smg -f '{{.NetworkSettings.IPAddress}}')"
echo "Open http://localhost:8188 (port forward must point to the IP above)"
