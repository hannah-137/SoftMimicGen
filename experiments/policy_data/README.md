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

This writes a new folder. Videos are 81 frames at 16 fps; `--all_frames` keeps every step.

    runs/franka_towel_n50_seed1_<date>_<time>/
      franka_towel_n50_seed1.hdf5                the demos (Isaac Lab Mimic format, see below)
      franka_towel_n50_seed1_failed.hdf5         failed attempts (only when there were any)
      franka_towel_n50_seed1_instance_ids.json   instance id -> object
      run_config.json, summary.txt, gen.log, videos.log
      videos/   per demo: demo_NNN_source.mp4 (room | wrist RGB), demo_NNN_geoedge.mp4 (edges, the control
                video, 1 px line between the views), demo_NNN_depth.mp4, demo_NNN_normals.mp4,
                demo_NNN_instance.mp4 (views of the raw data), demo_NNN_ref_sim.png (frame 0);
                sheet_first.png, sheet_last.png, summary.csv
      refs/     your reference images demo_NNN_TT.png; check_references.csv (+ failed_references.txt and
                check_<name>.png only when something fails)
      cosmos/<checkpoint>_<date>_<time>/   per reference: <name>.mp4 (the video) and <name>.json (the settings
                the framework used); run_config.json, run.log, debug.log, benchmark.json

Each demo in the hdf5 (`data/demo_N`) has `actions` (T, 7), `obs/agentview_image` and
`obs/robot0_eye_in_hand_image` (T, 512, 512, 3) uint8, the robot state observations of the task, `states`,
`initial_state`, `obs/agentview_geoedge` and `obs/robot0_eye_in_hand_geoedge` (T, 512, 512, 1) uint8 0 / 255,
and the raw renderer data `*_depth_raw` (float32 m), `*_normals_raw` (float32), `*_instance_raw` (int32).
T is the number of control steps (20 Hz for the Franka towel task).

Real-looking videos with Cosmos3. First put reference images (made from `ref_sim`, aspect 2:1) in
`<run_dir>/refs/` as `demo_NNN_<tag>.png` (several per demo, one video per image) or `demo_NNN.png`. Then check
them, then run:

    python experiments/policy_data/check_references.py <run_dir>
    python experiments/policy_data/run_cosmos.py <run_dir> --framework <cosmos-framework> \
        --checkpoint <Cosmos3-Super-fp8> --hf_home <hf cache> --gpus 2,3 --cp 2

The check rejects images that are not 2:1 and scores the layout (robot and towel where the simulator has them).
When something fails, `failed_references.txt` lists the images to make again. `run_cosmos.py` only starts when
every reference passed; it resizes the images to 1024 x 512 itself. Its output goes to
`<run_dir>/cosmos/<checkpoint name>_<date>_<time>/`.

## Names

Folder names are fixed by the scripts. Do not rename them or add words.

- Demo run: `runs/<task>_n<demos>_seed<seed>_<YYYYMMDD>_<HHMM>/`. Other settings (cameras, raw data) are in its
  `run_config.json`.
- Cosmos run: `<run dir>/cosmos/<checkpoint name>_<YYYYMMDD>_<HHMM>/`. Prompt, seed and steps are in its
  `run_config.json`.
- Reference images: `refs/demo_<NNN>_<TT>.png`, NNN = demo index, TT = two-digit number (01, 02, ...).
- Videos: `<cosmos run>/demo_<NNN>_<TT>.mp4`, the same name as the reference image.

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
