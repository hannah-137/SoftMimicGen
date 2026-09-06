# Docker setup

Tested on Ubuntu 24.04 host, RTX 4090 x2, NVIDIA driver 550.127.05, nvidia-container-toolkit.

## Usage

    git clone https://github.com/hannah-137/SoftMimicGen.git
    cd SoftMimicGen
    bash docker/create_container.sh        # asks for mount dir and container name, then installs (15-20 min)
    docker exec -it <name> bash            # enter, then: conda activate softmimicgen
    bash docker/verify.sh                  # generate 2 rope demos and convert to mp4

## Files

- `create_container.sh` – run on host. Creates the container and calls `container_setup.sh`.
- `container_setup.sh` – runs inside the container. Installs libs, Vulkan/EGL config, miniconda, then `softmimicgen.sh`.
- `verify.sh` – runs inside the container. Generates 2 rope demos on GPU 0 and writes mp4 to `docker/videos/`. Use this to check the environment works.
- `make_video.py` – converts camera images in an hdf5 to mp4. Used by `verify.sh`.

## Comparison with upstream README

Same
- Isaac Lab 2.2.1, Isaac Sim 5.1.0, Python 3.11 (pinned by `softmimicgen.sh`)
- Same install script, same `generate_dataset.py`

Different
- Runs inside Docker (`ubuntu:24.04`), not directly on the host
- conda is installed by `container_setup.sh`; upstream assumes it exists
- Vulkan/EGL config added for the NVIDIA driver; not needed on a host install, required in a container
- Run options added: `--headless` (no display), `--device cuda:0`, `--kit_args "--/renderer/multiGpu/enabled=false --/renderer/activeGpu=0"` (multi-GPU renderer hangs without it)
- NVIDIA driver 550.127.05. Not mentioned in upstream README; Isaac Sim 5.1 officially requires >= 570.169. Works, not validated

## Notes

- Mount dir must contain this repo. Code and datasets stay on the host; deleting the container loses nothing.
- `verify.sh` uses GPU 0. Check `nvidia-smi` first; if GPU 0 is busy, edit the script.