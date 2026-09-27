#!/usr/bin/env bash
# Extra packages for this folder, on top of the SoftMimicGen environment (installed as in its README).
#
#   conda activate softmimicgen
#   bash experiments/policy_data/setup.sh
#
# openai: the image edit API that make_references.py uses for the reference images.
set -eo pipefail
# --no-deps: install exactly these four and change nothing else. The rest of what openai needs (httpx, anyio,
# pydantic, typing_extensions) is already in the SoftMimicGen environment. openai 2.x and 3.x would upgrade
# typing_extensions, which Isaac Sim pins.
pip install --no-deps "openai==1.109.1" "distro==1.9.0" "jiter==0.17.0" "sniffio==1.3.1"
python -c "import openai; print('openai', openai.__version__)"
