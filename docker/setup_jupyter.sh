#!/bin/bash
# Jupyter server for notebooks (run inside the hyeon_smg container). Adds only the server + kernel packages to the
# existing softmimicgen env (torch 2.7 cu128, sklearn, matplotlib already there); no new env. VSCode on the host
# connects to it as an "Existing Jupyter Server" (docker/jupyter.sh prints the URL). GPU 1 only, as everywhere.
# Tested: 2026-09-20, python 3.11.16
# Usage: bash docker/setup_jupyter.sh      (idempotent)
set -e
ENV=/opt/miniconda3/envs/softmimicgen
CACHE=/workspace/tools/.pip_cache
export TMPDIR=/workspace/tools/.tmp      # pip unpacks wheels here, not on the container overlay
mkdir -p $CACHE $TMPDIR

if $ENV/bin/python -c "import jupyter_server, ipykernel" 2>/dev/null; then
  echo "jupyter already installed in $ENV"
else
  $ENV/bin/pip install "jupyter-server==2.21.1" "ipykernel==7.3.0" --cache-dir $CACHE
fi

# register the env as a kernel named "softmimicgen" (idempotent)
$ENV/bin/python -m ipykernel install --user --name softmimicgen --display-name "softmimicgen (GPU 1)"

$ENV/bin/python -c "import jupyter_server, ipykernel; print('jupyter_server', jupyter_server.__version__, 'ipykernel', ipykernel.__version__)"
echo "=== Setup done: start the server with  bash docker/jupyter.sh  (on the host)"
