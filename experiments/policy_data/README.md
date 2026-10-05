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
- `--table clean_top` uses the table of the task without its metal parts (the beam and the post beside the table,
  the handles, the bolts) and with the bolt holes in the top closed. The top itself is the upstream one: same
  material, size and place. Default `upstream` = the table as it is. `run_config.json` records the choice.
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
      refs_rejected/000-049/   images replaced by make_references.py --retry (<name>_attempt<N>.png), and
                refs_rejected/rejected.csv (their scores and the reason)
      cosmos/<checkpoint>_<date>_<time>/000-049/   per reference: <name>.mp4 (the video) and <name>.json (the
                settings the framework used, and policy_data: source demo, reference image and its sha1, seed);
                run_config.json, run.log, debug.log, benchmark.json at the top

Every folder with per-demo files holds 50 demos: 000-049, 050-099, ... (layout.py). The rule is the same for
2 demos and for 750.

Each demo in the hdf5 (`data/demo_N`) has `actions` (T, 7), `obs/agentview_image` and
`obs/robot0_eye_in_hand_image` (T, 512, 512, 3) uint8, the robot state observations of the task, `states`,
`initial_state`, `obs/agentview_geoedge` and `obs/robot0_eye_in_hand_geoedge` (T, 512, 512, 1) uint8 0 / 255,
and the raw renderer data `*_depth_raw` (float32 m), `*_normals_raw` (float32), `*_instance_raw` (int32).
T is the number of control steps (20 Hz for the Franka towel task).

Real-looking videos with Cosmos3. First make one reference image per demo. It is a realistic version of `ref_sim`
with the same layout, aspect 2:1. `make_references.py` makes it with the OpenAI image edit API
(v2: gpt-image-2.5-sunburst, quality medium). The lists, the prompts, the image model and the Cosmos prompt of each
spec version are in one file, `specs/<version>.json` (see `SPEC.md`, change history); `--spec v3` picks another
version, and the default is v2. In v2 every image changes 7 axes at once: place, table, lighting, robot wear, towel
color, towel material and towel pattern. The rules are in `variations.py` and are the same for every version. The
place type comes first. It is exact in every group of 50 demos (000-049, ...): 10 outdoor (20%) and 4 of each of the
10 indoor types (8% each). An outdoor place gets outdoor lighting. No light comes from the left or the right: the
image model would draw it the same way in both views. 2 of every 50 images get strong red, green, blue or yellow
light (indoor only, as in CRAFT). No two images in a run share a combination. `refs/references.csv` records the
axes, the prompt, the tokens, the cost and the spec version of every image. For a test of other lists or prompts,
copy a spec file, give it a new version name and pass its path: `--spec <file.json>` (in a dataset folder only with
`--refs <other folder>`). You can also put your own images in
`<run_dir>/refs/000-049/` (the folder of the demo) as `demo_NNN_<tag>.png` (several per demo, one video per image)
or `demo_NNN.png`. Then check them, then run:

    export OPENAI_API_KEY=...
    python experiments/policy_data/make_references.py <run_dir>            # --dry_run: prompts only, no API call
    python experiments/policy_data/checks/check_references.py <run_dir>
    python experiments/policy_data/make_references.py <run_dir> --retry    # failed images, new combination
    python experiments/policy_data/run_cosmos.py <run_dir> --framework <cosmos-framework> \
        --checkpoint <Cosmos3-Super-fp8> --hf_home <hf cache> --gpus 2,3 --cp 2

The check rejects images that are not 2:1 and scores the layout (robot and towel where the simulator has them).
When something fails, `failed_references.txt` lists the images to make again. `--retry` makes at most 3 images per
demo (`--max_attempts`). Images that were never made (for example after an API error) are made by the same command
without `--retry`. In a run folder, `run_cosmos.py` only starts when every reference passed; in a dataset folder
it takes the demos that are ready (see step 6 below). It resizes the images to 1024 x 512 itself. Its output goes
to `<run_dir>/cosmos/<checkpoint name>_<date>_<time>/000-049/`.

## Make the dataset (750 demos)

The spec is in `SPEC.md`. Run the commands from the repository root in the SoftMimicGen environment.
`make_references.py`, `checks/check_references.py`, `run_cosmos.py` (in a dataset folder) and `status.py` can run
again: finished work is skipped. `make_demos.sh` and `make_dataset.py` (`--take`, `--replace`, `--add`) do the
work again every time: run them once per step. (`--replace` without `--reason` only finishes a replace that
`status.py` lists as stopped halfway.)
`status.py` prints the state and the next commands.

1. Simulator demos. Make more than needed: demos with the towel out of view are not used, and some demos get
   replaced later. About 12 hours for 390 demos on one GPU, videos included.

       bash experiments/policy_data/make_demos.sh franka_towel 100 --seed 1 --gpu 0
       bash experiments/policy_data/make_demos.sh franka_towel 390 --seed 2 --gpu 1 \
           --camera_noise_pos 0.05 --camera_noise_rot 5
       bash experiments/policy_data/make_demos.sh franka_towel 390 --seed 3 --gpu 2 \
           --camera_noise_pos 0.05 --camera_noise_rot 5

2. Towel-in-view check, once per run:

       python experiments/policy_data/checks/check_towel_in_view.py <run>/<name>.hdf5 \
           --out <run>/check_towel_in_view.csv

3. One dataset folder with the first demos of each run that pass the check (see "Dataset folder" below). The
   hdf5 copy needs about 350 MB per demo (750 demos: about 260 GB); the script stops first when the disk has less.

       python experiments/policy_data/make_dataset.py --take <seed 1 run>:50 <seed 2 run>:350 <seed 3 run>:350

   More demos later: `--add` puts the next unused demos of a run after the last number. The folder keeps its
   name. Then do steps 4 to 7 for the new numbers.

       python experiments/policy_data/make_dataset.py <dataset> --add <run>:<N> [<run>:<N> ...]

4. Look at the simulator videos on the review page (see step 7), tab Simulator. After you reject a demo that looks
   wrong, run `status.py`: it prints the command that puts another demo at its number:

       python experiments/policy_data/review.py <dataset> --sim
       python experiments/policy_data/make_dataset.py <dataset> --replace <demo> --reason "<why>"

   It takes the next unused demo of the same run. When a run has no spare left, make more demos with a new
   `--seed` and the same other settings, run the check on them, and add `--from <that run>`. When a replace stops
   halfway, `status.py` says so and prints the command that finishes it; the other scripts skip the number until
   then.

5. Reference images, in batches (one group of 50 is easy to follow; any size works, the ratios do not change).
   About $0.014 per image; 50 images take about 10 minutes (3 calls in parallel). Check again after every retry.

       export OPENAI_API_KEY=...
       python experiments/policy_data/make_references.py <dataset> --demos 0-49
       python experiments/policy_data/checks/check_references.py <dataset> --demos 0-49
       python experiments/policy_data/make_references.py <dataset> --retry --demos 0-49

   `--spec v3` makes the images with spec version v3 (`specs/v3.json`); the default is v2. `--retry` without
   `--spec` keeps the version of each image. To make passed images again (for example with a new spec version),
   use `--redo --demos <numbers> --reason "<why>"`. It skips demos that have a Cosmos video: reject those videos on
   the review page with "reference image problem".

6. Cosmos videos, about 12 minutes per video on 2 GPUs (about 20 with the depth control of spec v7; about 39
   with the color guide of v8, which makes each video in two passes, see `SPEC.md`). The script takes only the
   demos that are ready (state `needs_video`: the image passed its check, no video yet) and skips the rest, so
   the same command can run again.
   With 4 GPUs, run two commands at the same time: each takes other demos (`--max` splits the batch).

       python experiments/policy_data/run_cosmos.py <dataset> --demos 0-49 --max 25 --gpus 0,1 --cp 2 \
           --framework <cosmos-framework> --checkpoint <Cosmos3-Super-fp8> --hf_home <hf cache>
       python experiments/policy_data/run_cosmos.py <dataset> --demos 0-49 --max 25 --gpus 2,3 --cp 2 \
           --framework <cosmos-framework> --checkpoint <Cosmos3-Super-fp8> --hf_home <hf cache>

   A demo that a running job makes a video for cannot be replaced until that job ends. The prompt comes from the
   spec version of each reference image (v2: a short prompt with the towel sentence of the image, for example "The
   towel is pewter bamboo fiber, plain, one solid color."), so a video always follows the version of its image.
   `--spec v3` takes only the images of that version.

   Steps 5 and 6 for many groups: `run_queue.py` does them group by group (images, check, retry, Cosmos). It makes
   the images of the next group while Cosmos runs, so the GPUs do not wait between groups, and it waits for a
   Cosmos job that already runs on the same GPUs. It does not wait for the review. Give the run_cosmos.py options
   after `--`. `--count N` makes only the next N demos that still need a video (for example 100 = two groups),
   then stops. `--spec v3` makes the images and videos with version v3; an image of another version that has no
   video yet is made again as v3 first. Change the version at the start of a group of 50. Stop the queue after the
   running job with `touch <dataset>/.queue_stop`, or at once with Ctrl-C.

       export OPENAI_API_KEY=...
       python experiments/policy_data/run_queue.py <dataset> --demos 150-749 [--spec v2] -- --gpus 2,3 --cp 2 \
           --framework <cosmos-framework> --checkpoint <Cosmos3-Super-fp8> --hf_home <hf cache> --tools <tools>

7. Review the videos on the review page: the simulator video and the Cosmos video play together, next to the
   reference image. Mark each video Approve (O), Weak (a triangle: usable but weak) or Reject (X); the keys are A,
   W and R. Weak counts as done, like Approve, and review.csv keeps the word weak for later analysis. With Weak or
   Reject, tick one or more reasons (towel look, towel shape, doubled towel, towel, extra object, robot, background,
   wrist view background, lighting, table, image quality, room and wrist views do not match, reference image problem) or
   write the reason in the line. A weak or rejected video without a reason is saved but counts as not reviewed until it has one. With
   Reject, tick "reference image problem" when the image is the cause: the video is then made again with a new
   image. The filter bar shows all videos, only O, only Weak, only X, or only the ones not reviewed. Submit saves
   the marks (videos without a verdict are not saved); submit again to change them, and the page shows the saved
   marks when it opens again.

       python experiments/policy_data/review.py <dataset> --group 000-049

   The script prints the address with a secret token (kept in `<dataset>/.review_token`, so it stays the same when
   the script starts again). From another machine, run the ssh command it prints on that machine, then open the
   address. With VSCode Remote-SSH, run the ssh command in a VSCode terminal and forward the port in the Ports panel
   (VSCode often does it by itself). Making rejected videos again (`run_cosmos.py --redo_bad`) comes in the next
   version; a demo whose video fails 3 times gets replaced.

8. The state at any time:

       python experiments/policy_data/status.py <dataset>

## Dataset folder

`make_dataset.py --take` copies the demos into one hdf5 and numbers them 0, 1, 2, ... The videos and frame-0 images
are hard links to the runs (no extra disk space). The runs stay as they are: `--replace` takes spare demos there,
and `--add` puts unused demos after the last number.

    runs/<task>_n<demos>_seeds<seeds>_<date>_<time>/
      <task>_n<demos>_seeds<seeds>.hdf5   data/demo_N as in a run, plus the attributes source_run, source_demo and
                                          instance_ids (the instance id table of its run; runs can number the ids
                                          differently, so there is no _instance_ids.json in a dataset)
      demos/, ref_sim/, refs/, refs_rejected/, cosmos/   as in a run folder
      sim_rejected/000-049/demo_NNN_r<k>/  everything of a replaced demo (videos, frame-0 image, reference images,
                                          Cosmos videos) and replaced.json
      run_config.json                     the takes, every run (relative path and settings), and the fixed values
                                          of the reference images: variation_seed, tag, max_attempts

The list files. Each file has one writer. Every write happens under the dataset lock (`.lists.lock`, see
`status.py`), so scripts can run at the same time. (The run_config.json of a running Cosmos job is written only by
its own job.)

| File | Writer | Content |
|---|---|---|
| `sources.csv` | make_dataset.py | demo number -> source run and demo, seed, room camera noise, steps |
| `replacements.csv` | make_dataset.py | one row per `--replace`: old and new source demo, reason, folder |
| `refs/references.csv` | make_references.py | how each reference image was made (axes, prompt, source demo, cost, spec version) |
| `refs/check_references.csv`, `refs/failed_references.txt` | checks/check_references.py | check result per image, with the sha1 of the checked file |
| `refs_rejected/rejected.csv` | make_references.py | images replaced by `--retry`: scores, reason |
| `cosmos/<run>/run_config.json` | run_cosmos.py | settings of the Cosmos job, the demos it takes and the spec versions |
| `cosmos/<run>/<group>/<name>.json` | run_cosmos.py | framework settings of one video, and `policy_data`: source demo, reference image and its sha1, seed, prompt, spec version |
| `review.csv` | review.py | the current verdict of each video (key: `<cosmos run>/<name>`): `approved`, `weak` or `rejected`, reference image problem, reasons (ids joined with `;`, the list is `REASONS` in `status.py`), one line |
| `sim_review.csv` | review.py | the verdict of each simulator demo (demo number and source demo) |
| `review_history.csv` | review.py | every submitted verdict, with the time |
| `dataset.csv`, `videos.csv` | status.py | state of every number and every video, with the spec version of each image and video, made new on every run |

States in `dataset.csv`: `conflict` (two images or videos for one number, or a file in the wrong folder: fix by
hand), `needs_replace`, `needs_ref`, `needs_check` (no check result for the file that is there now), `ref_failed`,
`needs_video`, `needs_review`, `video_rejected`, `approved`. `approved` includes the videos marked weak (with a
reason); the column `review` still says weak. The docstring of `status.py` explains each state. The column `note`
says "replace running", "replace stopped halfway" or "cosmos running (<run>)" when a job works on the number, and
"no reason" for a video marked weak or rejected without a reason.

When all 750 videos are approved (O or weak), the hdf5 files of the runs are not needed any more (the dataset has its
own copy and the hard-linked videos stay).

## Moving to another machine

Nothing in the code names a host, a user or an absolute path.
- Set up the environment with the setup scripts (SoftMimicGen README, `setup.sh`). Do not copy an environment.
- Copy the dataset folder. It works alone: the hdf5, videos, images and lists are inside, and the lists use paths
  relative to the folder. A copy of the dataset alone turns the hard links into normal files (about 20 MB per demo).
- Copy the source runs too while demos can still be replaced. Put them next to the dataset, or give
  `make_dataset.py --replace ... --runs_dir <folder>`.
- Set `OPENAI_API_KEY`. For Cosmos: the framework with its `.venv`, the checkpoint and the Hugging Face cache.
- Keep the dataset on a local disk: the locks use flock.

## Troubleshooting

- "waiting: another script holds ...lock": another script works on the same dataset. It goes on when that one is
  done.
- `make_dataset.py --take` stops with "differ in" or "different env_args": the runs were made with other settings
  and cannot be in one dataset.
- Isaac Sim stops at the first frame with "Vulkan device lost": use another GPU (`--gpu`). This happened on one GPU
  of a 4-GPU machine; the other GPUs worked.
- In a container, NVML errors or torch sees no GPU while the host is fine: restart the container.
- `make_references.py` stops with "no credit left" (error type insufficient_quota, code credit_balance_exhausted):
  add credit to the OpenAI account, then run the same command again.
- Before you install a package into the Isaac Sim environment, run `pip install --dry-run` and read the list.
  A normal install of openai changed typing_extensions and idna, which Isaac Sim pins; `setup.sh` uses `--no-deps`
  for this reason.
- Cosmos3-Super fp8 needs 2 GPUs (`--gpus a,b --cp 2`); 3 GPUs did not work. `run_cosmos.py` takes the first free
  torchrun port from 29511, so two jobs on the same dataset need no `--port`. Jobs on other datasets or run folders
  that start at the same moment need their own `--port`.
- Stop a Cosmos job with Ctrl-C or `kill <pid>`: it stops the framework and keeps the finished videos. Run the same
  command again for the rest. After `kill -9` the framework may go on for a while: `status.py` shows the job as
  running until it ends; the next `run_cosmos.py` then takes over its finished videos.
- h5py "unable to lock file": another process has the hdf5 open for writing (`make_dataset.py --replace` or
  `--add`). Run the command again when it is done.
- The same `--seed` does not give the same hdf5 file: the GPU physics differs slightly between runs.
- `status.py` lists problems (for example a file in the wrong group folder): fix them by hand, then run it again.
- "--replace stopped halfway": run `make_dataset.py <dataset> --replace <demo>` (no `--reason`). It finishes the
  plan in `sim_rejected/<group>/demo_NNN_r<k>/replaced.json`. To drop the plan instead, add `--cancel` (possible
  while the new demo is not linked yet; the old files move back). "replace running" in `dataset.csv` means a
  `--replace` works on that number now: wait.
- Do not run `make_videos.py` on a run again after `--take`: the dataset shares its video files (hard links), so
  the dataset videos would change too.

## Names

Folder names are fixed by the scripts. Do not rename them or add words.

- Demo run: `runs/<task>_n<demos>_seed<seed>_<YYYYMMDD>_<HHMM>/`. Other settings (cameras, raw data) are in its
  `run_config.json`.
- Dataset: `runs/<task>_n<demos>_seeds<seeds>_<YYYYMMDD>_<HHMM>/`, `<seeds>` = the seeds of its runs written one
  after the other (seeds 1, 2 and 3: `seeds123`). Replaced demos: `sim_rejected/<AAA>-<BBB>/demo_<NNN>_r<k>/`.
- Cosmos run: `<run dir>/cosmos/<checkpoint name>_<YYYYMMDD>_<HHMM>/` (`_2`, `_3`, ... when two jobs start in the
  same minute). Prompt, seed, steps and demos are in its `run_config.json`.
- Demo folders: `demos/<AAA>-<BBB>/demo_<NNN>/`, 50 demos per group folder (000-049, 050-099, ...).
- Frame-0 images: `ref_sim/<AAA>-<BBB>/demo_<NNN>_ref_sim.png`.
- Reference images: `refs/<AAA>-<BBB>/demo_<NNN>_<TT>.png`, NNN = demo index, TT = two-digit number (01, 02, ...).
- Videos: `<cosmos run>/<AAA>-<BBB>/demo_<NNN>_<TT>.mp4`, the same name as the reference image.

## Files

- `make_demos.sh` - runs generation and videos.
- `setup.sh` - extra packages on top of the SoftMimicGen environment (openai).
- `generate_demos.py` - upstream generation plus the camera settings and observations above.
- `table.py` - the table with a clean top for `--table clean_top`.
- `observations.py` - `GeoEdgeImage` (copied from the fork), `RawCameraImage` and `camera_pose`.
- `events.py` - the random camera move at reset.
- `make_videos.py` - videos, sheets, summary (no GPU).
- `layout.py` - the folder rules (50 demos per folder).
- `make_references.py`, `variations.py` - reference images with the OpenAI image API; the rules of the variations
  (place type shares, strong light) and the loader of the spec files.
- `specs/<version>.json` - one file per spec version: the axis lists, the image prompt and model, the Cosmos prompt.
- `make_dataset.py` - one dataset folder from several runs (`--take`), another demo at a number (`--replace`),
  more demos after the last number (`--add`).
- `status.py` - state of a dataset (`dataset.csv`, `videos.csv`) and the next commands; the lock and csv helpers.
- `review.py` - review page (web server, standard library only): mark videos O, weak or X with reasons, and check
  the simulator demos.
- `SPEC.md` - the dataset spec: demos, ratios, reference images, Cosmos settings.
- `checks/check_references.py` - size and layout check of the reference images.
- `checks/check_towel_stuck.py` - finds demos where the towel still hangs on the gripper at the last frame.
- `checks/check_towel_in_view.py` - finds demos where the towel touches the image border in any frame.
- `run_cosmos.py`, `cosmos_launch.py` - Cosmos3 video2video with the edge control video (from spec v7 also a
  depth control video, from v8 also a color guide made from a first pass).
- `run_queue.py` - reference images and Cosmos videos group by group, with no wait on the GPUs between groups.

## Needs

SoftMimicGen installed as in its README (Isaac Sim, Isaac Lab, annotated datasets). For videos: h5py, numpy,
opencv-python, imageio, imageio-ffmpeg, ffmpeg. For reference images: the `openai` package
(`bash experiments/policy_data/setup.sh`) and an OpenAI API key in `OPENAI_API_KEY` (paid: about $0.014 per image
with gpt-image-2.5-sunburst at quality medium; 50 images cost $0.70 on 2026-09-27). For Cosmos3: cosmos-framework
with its `.venv` and a checkpoint.
