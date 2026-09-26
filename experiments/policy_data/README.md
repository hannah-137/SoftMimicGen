# policy_data

Demos of SoftMimicGen with a room camera and a wide wrist camera, their edge videos, and real-looking videos
made with Cosmos3. Needs only upstream SoftMimicGen and this folder.

## What it changes

- Both cameras render at 512 x 512. The wrist camera focal length is 12 mm (upstream: 24 mm). Nothing else in
  the scene changes.
- Extra observations per camera in the hdf5: `*_geoedge` (edges), `*_depth_raw`, `*_normals_raw`,
  `*_instance_raw` (raw renderer outputs, float32 / int32). The instance id table is saved as json.
- `--seed` sets the generation seed (default 1, the upstream value). Same seed = same generation choices.
- Size: about 600 MB per demo with the raw data (50 demos = 30 GB). Write to a data disk. `--no_raw` gives
  about 100 MB per demo.

## Run

    conda activate softmimicgen
    bash experiments/policy_data/make_demos.sh franka_towel 50 --gpu 1

This writes a new folder `experiments/policy_data/runs/franka_towel_n50_seed1_<date>_<time>`. It holds the hdf5
and, per demo, the videos in `videos/`: `source` (room | wrist RGB), `geoedge` (control
video, lossless, 1 px line between the views), `depth`, `normals`, `instance` (views of the raw data) and
`ref_sim/demo_NNN.png` (frame 0). Also `sheet_first.png`, `sheet_last.png`, `summary.csv`, `summary.txt`.
Videos are 81 frames at 16 fps. `--all_frames` keeps every step.

Real-looking videos with Cosmos3. First put reference images (made from `ref_sim`, aspect 2:1) in
`<run_dir>/refs/` as `demo_NNN_<tag>.png` (several per demo, one video per image) or `demo_NNN.png`. Then check
them, then run:

    python experiments/policy_data/check_references.py <run_dir>
    python experiments/policy_data/run_cosmos.py <run_dir> --framework <cosmos-framework> \
        --checkpoint <Cosmos3-Super-fp8> --hf_home <hf cache> --gpus 2,3 --cp 2

The check resizes 2:1 images to 1024 x 512, rejects other aspects, and scores the layout (robot and towel where
the simulator has them). `failed_references.txt` lists the images to make again. `run_cosmos.py` only starts when
every reference passed. Its output goes to `<run_dir>/cosmos/<checkpoint name>_<date>_<time>/`.

## Names

Folder names are fixed by the scripts. Do not rename them or add words.

- Demo run: `runs/<task>_n<demos>_seed<seed>_<YYYYMMDD>_<HHMM>/`. Other settings (cameras, raw data) are in its
  `run_config.json`.
- Cosmos run: `<run dir>/cosmos/<checkpoint name>_<YYYYMMDD>_<HHMM>/`. Prompt, seed and steps are in its
  `run_config.json`.
- Reference images: `refs/demo_<NNN>_<TT>.png`, NNN = demo index, TT = two-digit number (01, 02, ...).
- Videos: `<cosmos run>/demo_<NNN>_<TT>/vision.mp4`, the same name as the reference image.

## Files

- `make_demos.sh` - runs generation and videos.
- `generate_demos.py` - upstream generation plus the camera settings and observations above.
- `observations.py` - `GeoEdgeImage` (copied from the fork) and `RawCameraImage`.
- `make_videos.py` - videos, sheets, summary (no GPU).
- `check_references.py` - size and layout check of the reference images.
- `run_cosmos.py`, `cosmos_launch.py` - Cosmos3 video2video with the edge control video.

## Needs

SoftMimicGen installed as in its README (Isaac Sim, Isaac Lab, annotated datasets). For videos: h5py, numpy,
opencv-python, imageio, imageio-ffmpeg, ffmpeg. For Cosmos3: cosmos-framework with its `.venv` and a checkpoint.
