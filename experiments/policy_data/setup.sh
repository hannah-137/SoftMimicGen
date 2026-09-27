#!/usr/bin/env bash
# Extra packages for this folder, on top of the SoftMimicGen environment (installed as in its README).
#
#   conda activate softmimicgen
#   bash experiments/policy_data/setup.sh
#
# openai: the image edit API that make_references.py uses for the reference images.
set -eo pipefail
pip install "openai==3.19.2"
python -c "import openai; print('openai', openai.__version__)"
