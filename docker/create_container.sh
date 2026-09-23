#!/usr/bin/env bash
set -e

cat << 'EOF'
##############################################################
#                                                            #
#   SoftMimicGen - create container                          #
#                                                            #
#   Run:  bash docker/create_container.sh                    #
#                                                            #
#   You will be asked for:                                   #
#     mount_dir  host dir mounted to /workspace              #
#                (this repo must be inside it)               #
#                e.g. /data/hyeon                            #
#     name       container name, use <user>_smg              #
#                e.g. hyeon_smg                              #
#                                                            #
##############################################################
EOF

read -rp "mount_dir: " MOUNT_INPUT
read -rp "name:      " NAME

if [ -z "$MOUNT_INPUT" ] || [ -z "$NAME" ]; then
  echo "[!] both values are required."
  exit 1
fi

DATA_DIR="$(cd "$MOUNT_INPUT" && pwd)"         # host dir mounted to /workspace
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"   # this repo
REPO_NAME="$(basename "$REPO_DIR")"            # repo folder name

# repo must be inside mount dir
case "$REPO_DIR" in
  "$DATA_DIR"/*) ;;
  *) echo "[!] repo ($REPO_DIR) must be inside mount dir ($DATA_DIR)"; exit 1 ;;
esac

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "[!] container '$NAME' already exists. remove with: docker rm -f $NAME"
  exit 1
fi

echo "[1/3] creating container: $NAME  ($DATA_DIR -> /workspace)"
mkdir -p "$DATA_DIR/tools/.tmp"   # TMPDIR inside the container; must exist before the first apt/dpkg run
docker run -d --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e ACCEPT_EULA=Y \
  -e TMPDIR=/workspace/tools/.tmp \
  -e REPO_NAME="$REPO_NAME" \
  --name "$NAME" \
  -v "$DATA_DIR":/workspace \
  ubuntu:24.04 sleep infinity

echo "[2/3] running setup inside container (15-20 min)"
set -o pipefail
docker exec "$NAME" bash "/workspace/$REPO_NAME/docker/container_setup.sh" 2>&1 | tee "$DATA_DIR/setup_log_${NAME}.txt"

echo "[3/3] done."
echo "      enter:  docker exec -it $NAME bash"
echo "      verify: bash docker/verify.sh"