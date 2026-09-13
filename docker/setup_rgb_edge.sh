#!/bin/bash
# Learned edge/line detectors (HED, PiDiNet, TEED, LineArt) for RGB control videos — run inside the hyeon_smg container.
# Separate conda env under /workspace/tools; softmimicgen and comfyui envs are untouched.
# Everything (env, conda/pip caches, weights) stays under /workspace/tools: the container root overlay has 0 B free.
# Tested: 2026-09-13, RTX 4090 24GB, driver 550.127.05
# Usage: bash docker/setup_rgb_edge.sh
set -e

TOOLS=/workspace/tools
ENV=$TOOLS/envs/rgb_edge           # conda env prefix
WEIGHTS=$TOOLS/rgb_edge_models     # HF cache holding the annotator weights (HF_HOME)
CACHE=$TOOLS/.pip_cache
export CONDA_PKGS_DIRS=$TOOLS/.conda_pkgs
export TMPDIR=$TOOLS/.tmp          # pip unpacks wheels here (/tmp is on the full overlay)
export HF_HOME=$WEIGHTS
mkdir -p $TOOLS/envs $WEIGHTS $CACHE $CONDA_PKGS_DIRS $TMPDIR

# 1. conda env (python 3.11, same as softmimicgen)
source /opt/miniconda3/etc/profile.d/conda.sh
[ -x $ENV/bin/python ] || conda create -p $ENV python=3.11 pip -y
conda activate $ENV

# 2. torch cu128 (same build as softmimicgen; works on driver 550)
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128 --cache-dir $CACHE

# 3. controlnet-aux (HED / PiDiNet / TEED / LineArt wrappers) + what the video scripts need
pip install "controlnet-aux==0.0.10" "timm==1.0.15" "scikit-image==0.25.2" "huggingface_hub[cli]>=0.34" \
            h5py "imageio[ffmpeg]" scipy --cache-dir $CACHE

# 4. annotator weights (~70 MB) into $HF_HOME so the detectors load offline afterwards
#    HED: ControlNetHED.pth   PiDiNet: table5_pidinet.pth   LineArt: sk_model.pth (realistic) + sk_model2.pth (coarse)
for f in ControlNetHED.pth table5_pidinet.pth sk_model.pth sk_model2.pth; do
  hf download lllyasviel/Annotators "$f"
done
hf download fal-ai/teed 5_model.pth          # TEED

# 5. smoke test: load the four detectors and run one synthetic 512x512 frame through each
python - <<'EOF'
import numpy as np, torch
from controlnet_aux import HEDdetector, PidiNetDetector, TEEDdetector, LineartDetector
dev = "cuda" if torch.cuda.is_available() else "cpu"
img = np.zeros((512, 512, 3), np.uint8); img[128:384, 128:384] = 255
dets = [("hed", HEDdetector.from_pretrained("lllyasviel/Annotators")),
        ("pidinet", PidiNetDetector.from_pretrained("lllyasviel/Annotators")),
        ("teed", TEEDdetector.from_pretrained("fal-ai/teed", filename="5_model.pth")),
        ("lineart", LineartDetector.from_pretrained("lllyasviel/Annotators"))]
for name, det in dets:
    kw = {"detect_resolution": 512, "output_type": "np"}
    if name != "teed":  # TEEDdetector.__call__ has no image_resolution argument
        kw["image_resolution"] = 512
    out = np.asarray(det.to(dev)(img, **kw))
    print(f"{name:8s} ok  shape={out.shape} dtype={out.dtype} device={dev}")
EOF

echo "=== Setup done ==="
echo "Use:  source /opt/miniconda3/etc/profile.d/conda.sh && conda activate $ENV && export HF_HOME=$WEIGHTS"
