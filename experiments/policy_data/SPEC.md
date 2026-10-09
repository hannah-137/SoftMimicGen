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
| Table | the upstream table (demos 000-749). Option `--table clean_top` for new demos: the same table without its metal parts (beam and post beside the table, handles, bolts) and with the bolt holes in the top closed, because Cosmos and the image model drew these parts in wrong ways. Option `--table wide_top`: `clean_top` with a wider top (0.60 m more toward the room camera, 0.65 m more on each side, nothing behind the robot), so the wrist camera sees only the table: the area beside the table took up to 30% of the wrist image, and Cosmos had to invent it |
| Ground | the grid floor of the task (demos 000-749). Option `--ground plain` for new demos: a flat floor of one plain gray color at the same height, no grid lines, because the image model drew the grid lines as tiles, rails and pipes and Cosmos kept them |
| Seed 1 | 50 demos, fixed room camera (dataset demos 000-049) |
| Seeds 2 and 3 | 350 demos each, room camera moved once per demo: uniform, +-5 cm and +-5 degrees on each axis, towel fully in view at the start |
| Towel in view | the towel stays inside the room image in every frame (`checks/check_towel_in_view.py`); other demos are not used |
| Spares | more demos than needed per seed (for example 390); `make_dataset.py --replace` takes the next spare, `--add` puts unused demos after the last number of the dataset |
| Recorded | RGB, geoedge, raw depth, normals and instance ids for both cameras; the room camera pose of every step |

## Dataset folder

- One folder with demo numbers 000-749, in groups of 50 (000-049, 050-099, ...).
- More demos can follow after the last number (`make_dataset.py --add`, unused demos of the runs). The folder
  keeps its name. The place type shares are exact in every full group of 50.
- Each number has one reference image and one video. Nothing is dropped: a number is done when its video is
  approved or marked weak (usable but weak; review.csv keeps the mark).
- A number gets another simulator demo (`make_dataset.py --replace`) when the demo looks wrong in the simulator,
  its reference image failed 5 times, or its video was rejected 5 times. The number keeps its place type and its
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
| Checks | missing image; wrong folder; aspect 2:1; layout against simulator frame 0: towel and table (both views) and robot (wrist view) within 8 px, no missing table edge, no other surface on more than 5 % of the table top or the towel, the same table look in both views |
| On failure | the image is made again with another combination, at most 5 images per demo |

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
| Control | the geoedge video of the demo (room and wrist side by side); from v7 also a depth video made from the raw depth of the demo (1 / depth as gray, near is white); from v8 also a color guide (the video of a first pass with the edge video only, with the fabric painted in the one color that it has in the reference image; Cosmos gets it as its blur control). All controls have the same weight |
| Reference | the reference image as frame 0 |
| Output | 81 frames, 16 fps, 1024x512. With `run_cosmos.py --all_steps`: every step of the demo (131 to 133 frames for the Franka towel task), made in one chunk and stored at the control rate of the simulator (20 fps) |
| Settings | 35 steps, guidance 3, control guidance 3, shift 5, seed 0 (a new seed when a video is made again) |
| Review | a person marks every video approved, weak or rejected; weak and rejected need at least one reason; at most 5 videos per demo |

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
| v3 | 2026-10-01 | the commit that adds `specs/v3.json` | The word fabric instead of towel in the image prompt and the Cosmos prompt: the word towel gave a terry look that turned into a worn cloth when folded. Fabric axes instead of the towel axes: color (100), material (15), thickness (thin, medium or heavy; only the ones that fit the material) and condition (new, washed soft, old and worn). No pattern: every fabric is plain, because the herringbone weave was hardly visible in the image and did not stay on the folded part. Place, table, lighting and robot as in v2. File: `specs/v3.json`. | none yet | none yet |
| v4 | 2026-10-02 | the commit that adds `specs/v4.json` | v2 without the pattern: every towel is plain, one solid color (the herringbone weave was hardly visible in the image and did not stay on the folded part). Adds thickness (thin, medium or heavy; only the ones that fit the material) and condition (new, washed soft, old and worn). The word towel, the v2 towel materials, place, table, lighting and robot as in v2. File: `specs/v4.json`. | none yet | none yet |
| v5 | 2026-10-02 | the commit that adds `specs/v5.json` | v3 with the word towel instead of fabric (image prompt of v2). The fabric materials, thickness, condition and sentence structure are the ones of v3, so v3 and v5 differ only in this word. File: `specs/v5.json`. | none yet | none yet |
| v6 | 2026-10-03 | the commit that adds `specs/v6.json` | v3 (the word fabric) with the floor and fewer materials. Floor: one that fits the place (180 places, 1 or 2 floors each), named in the image prompt and in the Cosmos prompt. Before, the wrist view showed a black or wrong area beside the table when the gripper moved (about 27% of the wrist view; the reference image and the edges say nothing about it). Materials: cotton poplin, crepe, cotton muslin, cotton flannel, wool flannel and fleece. They kept their look when folded in 70% of their v3 videos (all 533 videos of v2-v5 looked at: v3 55% good, v5 48%, v2 15%, v4 14%). Dropped: fabrics whose back side came out as another fabric (waffle-weave cotton, cotton terry, microfiber, linen, cotton canvas, chambray) and fabrics that turned into clothes with hems and ribs (cotton jersey, gabardine, rayon challis). Five color names got a clear color word (sand beige, stone gray, mushroom brown, grape purple, rose pink). Condition in short words: brand new, clean, old and worn. The v3 words named a surface look (crisp, faded, pilled) next to "plain, one solid color"; with "old and worn, evenly faded and pilled all over" the texture changed during the video in 18 of 88 videos of v3 and v5 (4 of 159 with the other two conditions). File: `specs/v6.json`. | none yet | none yet |
| v7 | 2026-10-04 | the commit that adds `specs/v7.json` | v6 with a second control video for Cosmos: depth, next to the edge video (key `cosmos.controls`). The images and both prompts are the ones of v6. The edge video gives only lines, so Cosmos had to guess the shape of the lifted and folded fabric and often drew a thick roll; the depth video shows that it is a thin sheet. In a test on 4 demos that already had a video the folded part became cleaner in 1 demo clearly and in 2 demos a little; the rest of the scene did not change. A video takes about 20 minutes on 2 GPUs (edge only: about 12). File: `specs/v7.json`. | none yet | none yet |
| v8 | 2026-10-05 | the commit that adds `specs/v8.json` | v7 with three more materials and a color guide for Cosmos. Materials: microfiber (thin), cotton twill (medium) and oxford cloth (thin) are new, next to the six of v7. Color guide: a third control video, next to the edge and the depth video (key cosmos.controls, name color). Each video is made in two passes. Pass 1 uses the edge video only. The color guide is the pass-1 video with the fabric repainted in one color: the color that the fabric has in the reference image. Pass 2 uses the edge video, the depth video and the color guide, and makes the final video. Before, the color and the look of the fabric often changed on the lifted and folded part. In tests on demos that already had a video, the fabric kept its color on the folded part with the guide. The three new materials were tested on one demo each: they kept their color and weave on the turned side. Strong weave patterns (waffle weave, herringbone) still changed, so they are not in the list. Two limits stay. A flaw of the pass-1 video outside the fabric stays in the final video. The lifted fabric can look thicker than the sheet in the simulator: Cosmos draws shallow creases as deep folds, and a depth video with stronger small differences did not change this. A video takes about 39 minutes on 2 GPUs (pass 1 about 12, pass 2 about 27). File: `specs/v8.json`. | none yet | none yet |
