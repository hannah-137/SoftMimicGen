# Docker setup

Tested on Ubuntu 24.04 host, RTX 4090 x2, NVIDIA driver 550.127.05, nvidia-container-toolkit.

## Usage

    git clone https://github.com/hannah-137/SoftMimicGen.git
    cd SoftMimicGen
    bash docker/create_container.sh        # asks for mount dir and container name, then installs (15-20 min)
    docker exec -it <name> bash            # enter, then: conda activate softmimicgen
    bash docker/verify.sh                  # generate 2 rope demos and convert to mp4

## Wan pipeline (inside the container)

    bash docker/gen.sh franka_towel v7        # hdf5 -> source video + reference image -> 7 control videos -> 7 Wan videos
    bash docker/gen.sh all v7 --from 2        # every task in experiments/wan_canny/scripts/tasks.py, skipping the hdf5 stage
    bash docker/gen.sh all v7 --n_ref 10      # unattended: 10 Qwen-Image-Edit reference images per task, then Wan for each

Everything for one run lands in `experiments/wan_canny/runs/<tag>/<task>_<tag>/` (one folder per tag): the hdf5 and logs at the root, then
`sources/` (source and inspection videos), `edges/` (control videos), `images/` (reference images) and `wans/`
(Wan videos + json). Stages can be run alone
(`make_hdf5.sh <task> <tag>`, then the `experiments/wan_canny/make_*.py <run_dir>` scripts). GPU split: Isaac Sim
renders on GPU 0 only (GPU 1 hangs at the first render on this machine), ComfyUI/Wan runs on GPU 1 and is started
automatically by `make_wan.py` when it is not up. See `CLAUDE.md` for the layout and the task table.

## Files

- `create_container.sh` – run on host. Creates the container and calls `container_setup.sh`.
- `container_setup.sh` – runs inside the container. Installs libs, Vulkan/EGL config, miniconda, then `softmimicgen.sh`.
- `verify.sh` – runs inside the container. Generates 2 rope demos on GPU 0 and writes mp4 to `videos/verify/`. Use this to check the environment works.
- `make_video.py` – converts camera images in an hdf5 to mp4. Used by `verify.sh`.
- `gen.sh` – runs inside the container. Whole Wan pipeline for one or more tasks (stages 1-4, per-task log and summary).
- `make_hdf5.sh` – runs inside the container. Stage 1 only: one hdf5 into the run folder (interactive without arguments).
- `comfy.sh` – run on host. Starts ComfyUI (port 8188, GPU 1) inside the container if it is not running; the web UI shares that server with the pipeline.
- `setup_comfyui.sh` – runs inside the container. Installs ComfyUI (conda env `comfyui`) and the Wan 2.2 / 2.1 Fun-Control models under `/workspace/tools`.
- `setup_rgb_edge.sh` – runs inside the container. Installs the soft-edge detector env (`/workspace/tools/envs/rgb_edge`: HED, PiDiNet, TEED, LineArt) and their weights.
- `setup_gpu1.sh` – runs inside the container. Conda activation hooks that set `CUDA_VISIBLE_DEVICES=1` for the
  softmimicgen, comfyui and rgb_edge envs (this project may only use GPU 1 on the shared server).
- `setup_qwen_image_edit.sh` – runs inside the container after `setup_comfyui.sh`. Qwen-Image-Edit-2511 models (~31 GB) for
  `experiments/wan_canny/scripts/make_refs.py` (automatic photorealistic reference images).
- `setup_jupyter.sh` – runs inside the container. Adds `jupyter-server` + `ipykernel` to the softmimicgen env and registers
  the kernel `softmimicgen (GPU 1)`; no new env.
- `jupyter.sh` – run on host. Starts a Jupyter server (port 8888, GPU 1) inside the container if it is not running and prints
  the URL for VSCode "Existing Jupyter Server" (`bash docker/jupyter.sh [root_dir]`, `bash docker/jupyter.sh stop`).

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