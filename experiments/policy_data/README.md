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
    bash experiments/policy_data/make_demos.sh franka_towel 50 experiments/policy_data/runs/franka_towel_n50 --gpu 1

This makes the hdf5 and, per demo, the videos in `videos/`: `source` (room | wrist RGB), `geoedge` (control
video, lossless, 1 px line between the views), `depth`, `normals`, `instance` (views of the raw data) and
`ref_sim/demo_NNN.png` (frame 0). Also `sheet_first.png`, `sheet_last.png`, `summary.csv`, `summary.txt`.
Videos are 81 frames at 16 fps. `--all_frames` keeps every step.

Real-looking videos with Cosmos3 (one reference image per demo, made from `ref_sim`):

    python experiments/policy_data/run_cosmos.py <run_dir> --demos 0 1 --refs ref_000.png ref_001.png \
        --framework <cosmos-framework> --checkpoint <Cosmos3-Super-fp8> --gpus 2,3 --cp 2

## Files

- `make_demos.sh` - runs generation and videos.
- `generate_demos.py` - upstream generation plus the camera settings and observations above.
- `observations.py` - `GeoEdgeImage` (copied from the fork) and `RawCameraImage`.
- `make_videos.py` - videos, sheets, summary (no GPU).
- `run_cosmos.py`, `cosmos_launch.py` - Cosmos3 video2video with the edge control video.

## Needs

SoftMimicGen installed as in its README (Isaac Sim, Isaac Lab, annotated datasets). For videos: h5py, numpy,
opencv-python, imageio, imageio-ffmpeg, ffmpeg. For Cosmos3: cosmos-framework with its `.venv` and a checkpoint.
