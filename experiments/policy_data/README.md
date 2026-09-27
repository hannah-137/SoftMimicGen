# policy_data

Demos of SoftMimicGen with a room camera and a wide wrist camera, their edge videos, and real-looking videos
made with Cosmos3. Needs only upstream SoftMimicGen and this folder.

## What it changes

- Both cameras render at 512 x 512. The wrist camera focal length is 12 mm (upstream: 24 mm). Nothing else in
  the scene changes.
- Extra observations per camera in the hdf5: `*_geoedge` (edges), `*_depth_raw`, `*_normals_raw`,
  `*_instance_raw` (raw renderer outputs, float32 / int32). The instance id table is saved as json.
- `--seed` sets the generation seed (default 1, the upstream value). The same seed does not give the same hdf5
  file: the GPU physics differs slightly between runs.
- `--camera_noise_pos 0.05 --camera_noise_rot 5` moves the room camera by a random offset once per demo (uniform,
  +-5 cm on each axis and +-5 degrees on each axis). A draw is kept only when the whole towel is inside the image
  at the start (a small margin); otherwise the generator draws again. Default 0 = fixed camera. The wrist camera
  never changes. The room camera pose of every step is in `obs/agentview_camera_pose` (x, y, z, qw, qx, qy, qz).
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
      sheet_first.png, sheet_last.png (room frame 0 / last frame of every demo), summary.csv (steps per demo)
      demos/000-049/demo_NNN/   demo_NNN_source.mp4 (room | wrist RGB), demo_NNN_geoedge.mp4 (edges, the control
                video, 1 px line between the views), demo_NNN_depth.mp4, demo_NNN_normals.mp4,
                demo_NNN_instance.mp4 (views of the raw data)
      ref_sim/000-049/   demo_NNN_ref_sim.png (frame 0 of every demo, the base for the reference images)
      refs/000-049/   reference images demo_NNN_TT.png, in the folder of their demo; refs/references.csv (how
                each image was made), refs/check_references.csv (+ failed_references.txt, and check_<name>.png
                next to the image, only when something fails)
      refs_rejected/   images replaced by make_references.py --retry
      cosmos/<checkpoint>_<date>_<time>/000-049/   per reference: <name>.mp4 (the video) and <name>.json (the
                settings the framework used); run_config.json, run.log, debug.log, benchmark.json at the top

Every folder with per-demo files holds 50 demos: 000-049, 050-099, ... (layout.py). The rule is the same for
2 demos and for 750.

Each demo in the hdf5 (`data/demo_N`) has `actions` (T, 7), `obs/agentview_image` and
`obs/robot0_eye_in_hand_image` (T, 512, 512, 3) uint8, the robot state observations of the task, `states`,
`initial_state`, `obs/agentview_geoedge` and `obs/robot0_eye_in_hand_geoedge` (T, 512, 512, 1) uint8 0 / 255,
and the raw renderer data `*_depth_raw` (float32 m), `*_normals_raw` (float32), `*_instance_raw` (int32).
T is the number of control steps (20 Hz for the Franka towel task).

Real-looking videos with Cosmos3. First make one reference image per demo. It is a realistic version of `ref_sim`
with the same layout, aspect 2:1. `make_references.py` makes it with the OpenAI image edit API (gpt-image-2). Every
image changes 7 axes at once: place, table, lighting, robot wear, towel color, towel material and towel pattern.
The lists and the rules are in `variations.py`. The place type comes first. It is exact in every group of 50 demos
(000-049, ...): 10 outdoor (20%) and 4 of each of the 10 indoor types (8% each). An outdoor place gets outdoor
lighting. 2 of every 50 images get strong red, green, blue or yellow light (indoor only, as in CRAFT). No two
images in a run share a combination. `refs/references.csv` records the axes, the prompt, the tokens and the cost of
every image. You can also put your own images in
`<run_dir>/refs/000-049/` (the folder of the demo) as `demo_NNN_<tag>.png` (several per demo, one video per image)
or `demo_NNN.png`. Then check them, then run:

    export OPENAI_API_KEY=...
    python experiments/policy_data/make_references.py <run_dir>            # --dry_run: prompts only, no API call
    python experiments/policy_data/checks/check_references.py <run_dir>
    python experiments/policy_data/make_references.py <run_dir> --retry    # failed and missing images, new combination
    python experiments/policy_data/run_cosmos.py <run_dir> --framework <cosmos-framework> \
        --checkpoint <Cosmos3-Super-fp8> --hf_home <hf cache> --gpus 2,3 --cp 2

The check rejects images that are not 2:1 and scores the layout (robot and towel where the simulator has them).
When something fails, `failed_references.txt` lists the images to make again. `run_cosmos.py` only starts when
every reference passed; it resizes the images to 1024 x 512 itself. Its output goes to
`<run_dir>/cosmos/<checkpoint name>_<date>_<time>/000-049/`.

## Names

Folder names are fixed by the scripts. Do not rename them or add words.

- Demo run: `runs/<task>_n<demos>_seed<seed>_<YYYYMMDD>_<HHMM>/`. Other settings (cameras, raw data) are in its
  `run_config.json`.
- Cosmos run: `<run dir>/cosmos/<checkpoint name>_<YYYYMMDD>_<HHMM>/`. Prompt, seed and steps are in its
  `run_config.json`.
- Demo folders: `demos/<AAA>-<BBB>/demo_<NNN>/`, 50 demos per group folder (000-049, 050-099, ...).
- Frame-0 images: `ref_sim/<AAA>-<BBB>/demo_<NNN>_ref_sim.png`.
- Reference images: `refs/<AAA>-<BBB>/demo_<NNN>_<TT>.png`, NNN = demo index, TT = two-digit number (01, 02, ...).
- Videos: `<cosmos run>/<AAA>-<BBB>/demo_<NNN>_<TT>.mp4`, the same name as the reference image.

## Files

- `make_demos.sh` - runs generation and videos.
- `setup.sh` - extra packages on top of the SoftMimicGen environment (openai).
- `generate_demos.py` - upstream generation plus the camera settings and observations above.
- `observations.py` - `GeoEdgeImage` (copied from the fork), `RawCameraImage` and `camera_pose`.
- `events.py` - the random camera move at reset.
- `make_videos.py` - videos, sheets, summary (no GPU).
- `layout.py` - the folder rules (50 demos per folder).
- `make_references.py`, `variations.py` - reference images with the OpenAI image API; the 7 axes, their lists and
  the rules.
- `checks/check_references.py` - size and layout check of the reference images.
- `checks/check_towel_stuck.py` - finds demos where the towel still hangs on the gripper at the last frame.
- `checks/check_towel_in_view.py` - finds demos where the towel touches the image border in any frame.
- `run_cosmos.py`, `cosmos_launch.py` - Cosmos3 video2video with the edge control video.

## Needs

SoftMimicGen installed as in its README (Isaac Sim, Isaac Lab, annotated datasets). For videos: h5py, numpy,
opencv-python, imageio, imageio-ffmpeg, ffmpeg. For reference images: the `openai` package
(`bash experiments/policy_data/setup.sh`) and an OpenAI API key in `OPENAI_API_KEY` (paid, about $0.05 per image at
quality medium). For Cosmos3: cosmos-framework with its `.venv` and a checkpoint.
