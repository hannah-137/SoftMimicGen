# Franka towel dataset spec

Two training sets for a diffusion policy:
- Set A: 750 simulator demos.
- Set B: set A plus 750 real-looking Cosmos videos, one per demo.

The steps and commands are in README.md. The lists of the reference image axes, the image prompt and the Cosmos
prompt of each version are in `specs/<version>.json`; the rules that do not change between versions (place type
shares, strong light slots) are in `variations.py`.

## Simulator demos

| Item | Value |
|---|---|
| Task | `franka_towel`, SoftMimicGen generation and success rule as upstream |
| Cameras | room 512x512; wrist 512x512 with a 12 mm lens (upstream: 24 mm) |
| Seed 1 | 50 demos, fixed room camera (dataset demos 000-049) |
| Seeds 2 and 3 | 350 demos each, room camera moved once per demo: uniform, +-5 cm and +-5 degrees on each axis, towel fully in view at the start |
| Towel in view | the towel stays inside the room image in every frame (`checks/check_towel_in_view.py`); other demos are not used |
| Spares | more demos than needed per seed (for example 390); `make_dataset.py --replace` takes the next spare |
| Recorded | RGB, geoedge, raw depth, normals and instance ids for both cameras; the room camera pose of every step |

## Dataset folder

- One folder with demo numbers 000-749, in groups of 50 (000-049, 050-099, ...).
- Each number has one reference image and one video. Nothing is dropped: a number is done when its video is
  approved or marked weak (usable but weak; review.csv keeps the mark).
- A number gets another simulator demo (`make_dataset.py --replace`) when the demo looks wrong in the simulator,
  its reference image failed 3 times, or its video was rejected 3 times. The number keeps its place type and its
  strong light slot, so the ratios below stay exact.

## Reference images

| Item | Value |
|---|---|
| Input | frame 0 of the demo, room and wrist side by side, 1024x512 |
| Output | realistic image with the same layout, 1024x512 PNG (made at 2048x1024, then resized) |
| Model | gpt-image-2.5-sunburst, image edit (OpenAI API), quality medium |
| Count | one per demo |
| Place | first a place type, then a place inside it. Outdoor 20%, each of the 10 indoor types 8%, exact in every group of 50 demos |
| Lighting | indoor lighting for an indoor place, outdoor lighting for an outdoor place; no light from the left, right or front (the image model draws it the same way in the room and the wrist view, which is wrong for the wrist camera) |
| Strong colored light | 2 of every 50 images (4%), indoor only: red, green, blue or yellow light over the whole scene |
| Other axes | every item has the same chance |
| No repeat | no two images in the dataset share the same 7 values |
| Checks | missing image; wrong folder; aspect 2:1; layout score 0.70 or more in both views (share of the simulator robot and towel outline within 5 px of an image edge) |
| On failure | the image is made again with another combination, at most 3 images per demo |

Prompt (the same for every image; `{variation}` is the only part that changes):

> Turn this simulator image into a realistic photo. It is a split screen: left the room camera, right the camera on
> the robot gripper. Keep the composition exactly the same in both halves: the same camera viewpoint and framing,
> and the same size and position of the robot, the towel, the table and the background. Keep the same aspect ratio.
> Do not crop, zoom or pad. The robot keeps the same pose. The towel keeps exactly the same position, shape, size and
> orientation. The table keeps the same position, size and shape; only its top surface changes. Do not add, remove,
> move, fold, bend or reshape anything. There is exactly one towel, flat on the table. {variation} The towel has the
> same color and texture on both sides. Change only the materials, textures, lighting and the look of the
> background. No text, no logos, no watermark.

Axes (every image changes all 7 at once):

| Axis | Sentence | Count |
|---|---|---|
| Place | The scene is in {place}. | 180 (indoor 150 in 10 types, outdoor 30) |
| Table | The table top is {table}. | 60 |
| Lighting | The lighting is {lighting}. | indoor 72 (source 6 x color 4 x level 3), strong colored 4, outdoor 8 |
| Robot | {robot} | 5 (surface wear of the white Franka) |
| Towel color, material, pattern | The towel is {color} {material}, {pattern}. | 100, 8, 2 (plain or a subtle herringbone weave: stripes, checks, dots and borders faded to plain in the Cosmos videos when the towel folds) |

## Cosmos videos

| Item | Value |
|---|---|
| Model | Cosmos3-Super fp8, video2video on 2 GPUs (context parallel 2) |
| Control | the geoedge video of the demo (room and wrist side by side) |
| Reference | the reference image as frame 0 |
| Output | 81 frames, 16 fps, 1024x512 |
| Settings | 35 steps, guidance 3, control guidance 3, shift 5, seed 0 (a new seed when a video is made again) |
| Review | a person marks every video approved, weak or rejected; weak and rejected need at least one reason; at most 3 videos per demo |

## Change history

Each version is one file, `specs/<version>.json`. A new version is a new file; the files of the old versions stay
as they are. `make_references.py`, `run_cosmos.py` and `run_queue.py` take `--spec <version>` (default `SPEC` in
`status.py`, now v2). The scripts write the version with every reference image (column `spec` of
`refs/references.csv`) and every Cosmos video (`policy_data.spec` in the json of each video, and `spec` in
`run_config.json`). A video always follows the version of its reference image. `dataset.csv` and `videos.csv` have
the version, `status.py` prints the approved count per version, and the review page shows it on each video. Change
the version at the start of a group of 50 demos.

| Version | Date | Commit | Changes | Reference images | Cosmos videos |
|---|---|---|---|---|---|
| v1 | 2026-09-27 | 1910d53 | First spec. Lighting may come from the left, the right or the front. Towel patterns: plain, subtle herringbone, thin stripes, small checks, small dots, darker border. The Cosmos prompt has no towel sentence. | demos 0-49, first images: moved to `refs_rejected/`, except 11, 25, 35, 41 and 45, which follow the v2 rules and stay in use | demos 0-9, run `Cosmos3-Super-fp8_20260928_2118` (moved to `cosmos_archive/`) |
| v2 | 2026-09-28 | 1ffe58d | No light from the left, the right or the front: the image model drew it the same way in both halves. Towel patterns: plain and subtle herringbone only: the prints faded to plain when the towel folded. The Cosmos prompt has the towel sentence of the reference image. File: `specs/v2.json`. | demos 0-149 | demos 0-9 (run `Cosmos3-Super-fp8_20260929_0100`), 10-49 (`..._20260929_0347`), 50-99 (`..._20260929_1413`) |
