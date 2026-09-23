#!/usr/bin/env bash
# Bigger Cosmos 3 checkpoints for experiments/cosmos3, after setup.sh. Runs INSIDE the container (root).
#   bash experiments/cosmos3/setup_super.sh          # Cosmos3-Super fp8 (70 GB) + Cosmos3-Nano bf16 (35 GB)
#   bash experiments/cosmos3/setup_super.sh --bf16   # also Cosmos3-Super bf16 (133 GB; needs 4 x 48 GB at run time)
# Sizes on Hugging Face (2026-09-23): nvidia/Cosmos3-Super main = bf16 133 GB, branch fp8 = ModelOpt fp8 70 GB (same recipe
# as the Nano fp8 we use), nvidia/Cosmos3-Nano main = bf16 35 GB. None is gated. The framework picks the model configuration
# from the repo id stamped in modular_model_index.json, so the folder names below are free to choose.
# Everything goes under tools/cosmos3_models/<name>; safe to re-run (finished files are skipped). Log: experiments/cosmos3/setup_super.log
set -euo pipefail
cd /workspace/SoftMimicGen
source experiments/cosmos3/env.sh      # hf on PATH, HF_HOME, TOOLS; no GPU needed here
HERE=experiments/cosmos3
M="$TOOLS/cosmos3_models"
exec > >(tee -a "$HERE/setup_super.log") 2>&1
echo "=== cosmos3 setup_super start $(date '+%F %T')"
df -h /workspace | tail -1

dl() {  # dl <repo> <revision> <folder>: download (example media skipped) and check the checkpoint folder
  echo "=== $1 @ $2 -> $M/$3"
  hf download "$1" --revision "$2" --local-dir "$M/$3" --exclude "assets/*" "images/*"
  du -sh "$M/$3"
  python - "$M/$3" <<'EOF'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
idx = json.loads((d / "modular_model_index.json").read_text())
repos = {c[2]["pretrained_model_name_or_path"] for c in idx.values() if isinstance(c, list) and len(c) == 3}
shards = sorted((d / "transformer").glob("*.safetensors"))
print(f"ok: repo {repos}, transformer shards {len(shards)}, quantized: {(d / 'hf_quant_config.json').is_file()}")
EOF
}

dl nvidia/Cosmos3-Super fp8  Cosmos3-Super-fp8
dl nvidia/Cosmos3-Nano  main Cosmos3-Nano-bf16
[ "${1:-}" = "--bf16" ] && dl nvidia/Cosmos3-Super main Cosmos3-Super-bf16
du -sh "$M"
echo "=== cosmos3 setup_super done $(date '+%F %T')"
