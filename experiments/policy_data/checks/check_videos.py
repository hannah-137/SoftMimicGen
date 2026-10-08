"""Check the Cosmos videos of a dataset against the simulator, frame by frame. No GPU needed.

  python experiments/policy_data/checks/check_videos.py <dataset_dir> --out <folder> [--demos 0-49] [--workers 4]
                                                        [--sim] [--marks <csv>]

For every current video in <dataset_dir>/videos.csv the script reads the video and, from the dataset hdf5, the
simulator data of the same steps (RGB, raw instance ids, geometry edges). A video with every step (run_cosmos.py
--all_steps) has one frame per step. A video with 81 frames shows the steps of make_videos.sample_idx.

The check looks for problems that grow with the number of frames. It measures 10 items in every frame, in the room
view and in the wrist view. Each item gets a score from 0 to 100. 100: like the start of the video, no problem.
0: the "clearly bad" level of the item (the BAD_* constants below, fixed before the first run).
The fixed region of a view has the same content in every frame: in the room view the pixels that the robot and the
towel never cover, in the wrist view the pixels that are robot in every frame (the gripper).

   1 sharpness   mean |Laplacian| in the fixed region, as a ratio to the first frames
   2 color       mean Lab color of the fixed region, difference to frame 0
   3 look        towel and table inside their simulator masks, difference to the first frames: color a, b; for
                 the table in the room view also the lightness L and the fine texture (the wrist camera moves, so
                 light and texture scale change there; a folded towel has creases, so its texture is written only)
   4 background  room view: share of the pixels outside robot and towel that changed since they were first seen
                 (the dark share is written too); wrist view: color a, b of the floor, difference to its first frames
   5 flicker     mean difference between two frames in the fixed region, minus the same difference in the
                 simulator frames (light on the gripper changes in the simulator too)
   6 jump        frame difference of the whole view against the frame difference of the simulator: a cut or a
                 sudden change is a jump (the correlation of the two is written too)
   7 period      size of a pattern with a period of 4 frames (the video model packs 4 frames into one) in the
                 lightness and in the frame difference of the fixed region (its share of the series is written too)
   8 control     share of the simulator outlines of robot and towel that lie within 5 px of an edge of the video
  11 views       table (L, a, b; fine texture in the first frames) and towel (a, b): difference between the room
                 view and the wrist view, in units of the thresholds of the reference image check
  12 end         drop of the scores of items 1, 2 and 4 in the last frames against the frames before them
Items 9 (ghosting, smearing) and 10 (wrong shapes, extra objects) cannot be measured. A person looks at the
strip images in <folder>/strips/ (a frame every 5 steps) and writes a csv with the columns demo, ghosting, shape
(Y or N). --marks <csv> adds these two items (N: 100, Y: 0).

The score of an item for a video is its worst part: the lowest mean of about 5 steps in a row, in the worse view.
The final score of a video is the mean of its item scores. --sim measures the simulator frames in place of the
video: the base line of every item.

Output in <folder>: check_videos.csv (one row per video: item scores, the worst raw numbers, the number of items
below 50, the final score), check_videos_frames.csv (one row per video, view and frame) and strips/.
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool

import cv2
import imageio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import check_references  # noqa: E402
import layout  # noqa: E402
import make_videos  # noqa: E402
import reference_layout  # noqa: E402
import status  # noqa: E402

VIEWS = reference_layout.VIEWS
ITEMS = ("sharpness", "color", "look", "background", "flicker", "jump", "period", "control", "views", "end")
EYE_ITEMS = ("ghosting", "shape")

# The level of every item that scores 0 ("clearly bad"). GOOD_* scores 100 where that is not 0.
BAD_SHARP = 0.5       # sharpness ratio to the first frames (1.0 scores 100)
BAD_COLOR = 20.0      # Lab distance of the fixed region to frame 0
BAD_LOOK = 20.0       # Lab distance of towel, table or wrist floor to the first frames (see items 3 and 4)
BAD_TEXTURE = 1.0     # difference of the fine texture of the table (log scale) to the first frames, room view
BAD_CHANGED = 0.30    # room view: share of the background that changed since it was first seen
BAD_FLICKER = 4.0     # gray levels (0..255) between two frames in the fixed region
BAD_JUMP = 10.0       # frame difference against the simulator (1.0 scores 100)
BAD_PERIOD_LIGHT = 1.0    # size of the 4-frame pattern in the lightness L (0..100)
BAD_PERIOD_FLICKER = 2.0  # size of the 4-frame pattern in the frame difference (gray levels)
GOOD_EDGE, BAD_EDGE = 0.9, 0.5      # share of the simulator outlines found in the video
BAD_VIEWS = 2.0       # view difference in units of the reference check thresholds (LOOK_DL, LOOK_DC, LOOK_DTEX)
BAD_END = 50.0        # score points lost in the last frames

GROW = 6              # px around robot and towel that count as moving
CHANGED_DE = 25.0     # Lab distance that counts as a changed background pixel
DARK_L = 18.0         # L (0..100) below this is a very dark pixel
MIN_PX = reference_layout.LOOK_MIN_PX  # pixels a region needs for a measure
EDGE_TOL = 5          # px between a simulator outline and a video edge
EDGE_MIN = 80         # simulator outline pixels needed in a frame
STRIP_STEPS = 5       # the strip images show a frame every this many steps
WINDOW_STEPS = 5      # the worst part of a video: this many steps in a row
START_STEPS = 10      # "the first frames": this many steps
PERIOD = 4            # frames: the temporal packing of the video model
BELOW = 50.0          # an item below this score counts as failed


def linear(x, good: float, bad: float):
    """Score 100 at good, 0 at bad, linear between, clipped. nan stays nan."""
    return np.clip(100.0 * (np.asarray(x, np.float64) - bad) / (good - bad), 0.0, 100.0)


def lowest(values) -> float:
    """Smallest number that is not nan (nan when there is none). values: numbers or arrays."""
    v = np.concatenate([np.atleast_1d(np.asarray(x, np.float64)).ravel() for x in values]) if len(values) else np.zeros(0)
    v = v[np.isfinite(v)]
    return float(v.min()) if len(v) else float("nan")


def highest(values) -> float:
    """Largest number that is not nan (nan when there is none)."""
    return -lowest([-np.asarray(x, np.float64) for x in values])


def lab_of(bgr: np.ndarray) -> np.ndarray:
    """(H, W, 3) BGR uint8 -> Lab float32 with L in 0..100 and a, b around 0, lightly blurred."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab = cv2.GaussianBlur(lab, (0, 0), 2.0)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return lab


def grow(mask: np.ndarray, px: int) -> np.ndarray:
    return cv2.dilate(mask.astype(np.uint8), np.ones((2 * px + 1, 2 * px + 1), np.uint8)) > 0


def shrink(mask: np.ndarray, px: int) -> np.ndarray:
    return cv2.erode(mask.astype(np.uint8), np.ones((2 * px + 1, 2 * px + 1), np.uint8)) > 0


def sim_steps(obs, cams: tuple, tables: dict, steps: np.ndarray) -> dict:
    """Simulator data of the given steps: per view the BGR frames, the masks of robot, towel and table, the pixels
    with no instance and the geometry edges. The table is the instance that the towel lies on at the first step
    (the same rule as check_references.sim_views)."""
    out, prim = {}, None
    for view, cam in zip(VIEWS, cams):
        name = check_references.prefix(cam)
        ids = obs[f"{name}_instance_raw"][steps][..., 0]
        groups = check_references.group_ids(ids, tables[cam])
        if prim is None:  # the room view comes first
            ring = grow(groups[0] == 2, 1) & (groups[0] == 3)
            vals, counts = np.unique(ids[0][ring], return_counts=True)
            prim = tables[cam].get(str(int(vals[counts.argmax()]))) if len(vals) else None
            if prim is None:
                raise ValueError("no towel on a table in the room view at the first step")
        table = np.isin(ids, [int(k) for k, v in tables[cam].items() if v == prim])
        out[view] = {"sim": np.ascontiguousarray(obs[cam][steps][..., ::-1]), "robot": groups == 1,
                     "towel": groups == 2, "table": table, "void": ids == 0,
                     "edges": obs[f"{name}_geoedge"][steps][..., 0] > 127}
    return out


def worst(scores: np.ndarray, window: int) -> float:
    """Lowest mean of `window` frames in a row. Frames without a value are left out."""
    s = np.asarray(scores, np.float64)
    if not np.isfinite(s).any():
        return float("nan")
    best = np.inf
    for i in range(0, max(1, len(s) - window + 1)):
        part = s[i:i + window]
        if np.isfinite(part).any():
            best = min(best, float(np.nanmean(part)))
    return best


def periodic(series: np.ndarray) -> tuple:
    """The pattern with a period of PERIOD frames in a series, after the slow trend is taken out:
    (its share of the power of the series, 0..1; its size in the unit of the series)."""
    x = np.asarray(series, np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 8 * PERIOD:
        return float("nan"), float("nan")
    kernel = np.ones(9) / 9.0
    z = x - np.convolve(x, kernel, "same") / np.convolve(np.ones(len(x)), kernel, "same")
    power = np.abs(np.fft.rfft(z * np.hanning(len(z)))) ** 2
    if power[1:].sum() < 1e-12:
        return 0.0, 0.0
    near = np.abs(np.fft.rfftfreq(len(z)) - 1.0 / PERIOD) <= 1.5 / len(z)
    size = 2.0 / len(z) * abs(np.sum(z * np.exp(-2j * np.pi * np.arange(len(z)) / PERIOD)))
    return float(power[near].sum() / power[1:].sum()), float(size)


def view_measures(frames: np.ndarray, sim: dict, view: str, start: int) -> dict:
    """Raw numbers of one view for every frame. frames: (F, H, W, 3) BGR. start: frames that count as the start."""
    F = len(frames)
    nan = np.full(F, np.nan)
    m = {k: nan.copy() for k in ("sharp", "sharp_ratio", "light", "color_de", "towel_dc", "table_de", "towel_tex",
                                 "table_tex", "changed", "dark", "floor_dc", "flicker", "flicker_video", "motion",
                                 "sim_motion", "edge")}
    moving = np.stack([grow(sim["robot"][i] | sim["towel"][i], GROW) for i in range(F)])
    fixed = ~moving.any(axis=0) if view == "room" else shrink(sim["robot"].all(axis=0), 4)
    has_fixed = int(fixed.sum()) >= MIN_PX
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames])
    sim_gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in sim["sim"]])
    labs = [lab_of(f) for f in frames]

    # items 1, 2, 5: the fixed region
    if has_fixed:
        for i in range(F):
            m["sharp"][i] = float(np.abs(cv2.Laplacian(gray[i], cv2.CV_32F, ksize=3))[fixed].mean())
            mean = labs[i][fixed].mean(axis=0)
            m["light"][i] = float(mean[0])
            if i == 0:
                first = mean
            m["color_de"][i] = float(np.linalg.norm(mean - first))
            if i:
                m["flicker_video"][i] = float(np.abs(gray[i] - gray[i - 1])[fixed].mean())
                m["flicker"][i] = max(0.0, m["flicker_video"][i] - float(np.abs(sim_gray[i] - sim_gray[i - 1])[fixed].mean()))
        m["sharp_ratio"] = m["sharp"] / max(float(m["sharp"][:start].mean()), 1e-6)

    # item 6: motion of the whole view, video and simulator
    m["motion"][1:] = np.abs(gray[1:] - gray[:-1]).mean(axis=(1, 2))
    m["sim_motion"][1:] = np.abs(sim_gray[1:] - sim_gray[:-1]).mean(axis=(1, 2))

    # item 3: towel and table inside their simulator masks; item 4, wrist view: the floor
    looks = {"towel": [], "table": [], "floor": []}
    for i in range(F):
        robot, towel, table = sim["robot"][i], sim["towel"][i], sim["table"][i]
        regions = {"towel": shrink(towel, 8) & ~grow(robot, 8), "table": shrink(table, 8) & ~grow(robot | towel, 8),
                   "floor": shrink(~(robot | towel | table), 8)}
        for name, region in regions.items():
            looks[name].append(reference_layout.look(frames[i], region) if int(region.sum()) >= MIN_PX else None)
    for name, key, with_light in (("towel", "towel_dc", False), ("table", "table_de", view == "room"),
                                  ("floor", "floor_dc", False)):
        seen = [x for x in looks[name] if x is not None][:max(1, start // 2)]
        if not seen:
            continue
        base = np.median(np.array(seen), axis=0)  # L, a, b, texture of the first frames that show the region
        for i, x in enumerate(looks[name]):
            if x is None:
                continue
            d = np.array(x) - base
            m[key][i] = float(np.linalg.norm(d[:3] if with_light else d[1:3]))
            if name != "floor":
                m[f"{name}_tex"][i] = abs(float(d[3]))
    if view == "wrist":  # the wrist camera changes its distance to the table: no texture measure
        m["towel_tex"][:], m["table_tex"][:] = np.nan, np.nan
    else:
        m["floor_dc"][:] = np.nan  # the room view has the per-pixel measure below

    # item 4, room view: background pixels against the frame that first showed them
    if view == "room":
        ref = np.zeros_like(labs[0])
        seen = np.zeros(fixed.shape, bool)
        dark0 = None
        for i in range(F):
            back = ~moving[i]
            old = back & seen
            if int(old.sum()) >= MIN_PX:
                m["changed"][i] = float((np.linalg.norm(labs[i] - ref, axis=2)[old] > CHANGED_DE).mean())
            new = back & ~seen
            ref[new] = labs[i][new]
            seen |= new
            share = float((labs[i][..., 0][back] < DARK_L).mean()) if back.any() else np.nan
            dark0 = share if dark0 is None else dark0
            m["dark"][i] = share - dark0

    # item 8: the simulator outlines of robot and towel against the edges of the video
    for i in range(F):
        robot, towel = sim["robot"][i], sim["towel"][i]
        target = (robot & ~shrink(robot, 1)) | (towel & ~shrink(towel, 1))
        if int(target.sum()) >= EDGE_MIN:
            edges = cv2.Canny(cv2.GaussianBlur(gray[i].astype(np.uint8), (5, 5), 0), 40, 120) > 0
            m["edge"][i] = float((target & grow(edges, EDGE_TOL)).sum() / target.sum())
    return m


def view_difference(frames: np.ndarray, sims: dict, start: int) -> np.ndarray:
    """Item 11 for every frame: difference of table and towel between the two views, in units of the thresholds
    of the reference check. The fine texture counts in the first frames only."""
    W = frames.shape[2] // 2
    out = np.full(len(frames), np.nan)
    for i in range(len(frames)):
        worst_ratio = np.nan
        for name in ("table", "towel"):
            look = {}
            for v, view in enumerate(VIEWS):
                sim = sims[view]
                others = sim["robot"][i] | (sim["towel"][i] if name == "table" else False)
                region = shrink(sim[name][i], 8) & ~grow(others, 8)
                if int(region.sum()) >= MIN_PX:
                    look[view] = reference_layout.look(np.ascontiguousarray(frames[i][:, v * W:(v + 1) * W]), region)
            if len(look) < 2:
                continue
            (l0, a0, b0, t0), (l1, a1, b1, t1) = look["room"], look["wrist"]
            ratios = [float(np.hypot(a0 - a1, b0 - b1)) / reference_layout.LOOK_DC]
            if name == "table":
                ratios.append(abs(l0 - l1) / reference_layout.LOOK_DL)
                if i < start:
                    ratios.append(abs(t0 - t1) / reference_layout.LOOK_DTEX)
            worst_ratio = np.nanmax([worst_ratio, max(ratios)])
        out[i] = worst_ratio
    return out


def frame_scores(m: dict) -> dict:
    """Scores of one view for every frame, from its raw numbers."""
    med = float(np.nanmedian(m["motion"]))
    sim_med = float(np.nanmedian(m["sim_motion"]))
    rel = m["motion"] / max(med, 1e-6)
    sim_rel = np.maximum(m["sim_motion"] / max(sim_med, 1e-6), 1.0)
    look = np.fmin(np.fmin(linear(m["towel_dc"], 0, BAD_LOOK), linear(m["table_de"], 0, BAD_LOOK)),
                   linear(m["table_tex"], 0, BAD_TEXTURE))
    return {"sharpness": linear(m["sharp_ratio"], 1.0, BAD_SHARP), "color": linear(m["color_de"], 0, BAD_COLOR),
            "look": look,
            "background": np.fmin(linear(m["changed"], 0, BAD_CHANGED), linear(m["floor_dc"], 0, BAD_LOOK)),
            "flicker": linear(m["flicker"], 0, BAD_FLICKER), "jump": linear(rel / sim_rel, 1.0, BAD_JUMP),
            "control": linear(m["edge"], GOOD_EDGE, BAD_EDGE)}


def check_video(job: tuple) -> tuple:
    """One video in a worker process -> (row of check_videos.csv, rows of check_videos_frames.csv)."""
    ds, h5path, run_tables, video, out, use_sim = job
    idx, name = int(video["demo"]), video["name"]
    row = {"demo": idx, "name": name, "file": video["file"], "note": ""}
    try:
        frames = np.stack(imageio.mimread(f"{ds}/{video['file']}", memtest=False))[..., :3][..., ::-1]
        with status.open_hdf5(h5path) as h5:
            demo = h5["data"][f"demo_{idx}"]
            obs = demo["obs"]
            cams = check_references.camera_keys(obs)
            T = obs[cams[0]].shape[0]
            F = len(frames)
            if F == T:
                steps = np.arange(T)
            elif F == make_videos.N_FRAMES:
                steps = make_videos.sample_idx(T)
            else:
                raise ValueError(f"{F} frames for {T} steps: not every step and not {make_videos.N_FRAMES} frames")
            sims = sim_steps(obs, cams, check_references.demo_tables(demo, run_tables), steps)
        if use_sim:  # the base line: the simulator frames in place of the video
            frames = np.concatenate([sims[v]["sim"] for v in VIEWS], axis=2)
        frames = np.ascontiguousarray(frames)
        W = frames.shape[2] // 2
        per_step = F / T
        start = max(3, round(START_STEPS * per_step))
        window = max(3, round(WINDOW_STEPS * per_step))
        row.update(frames=F, steps=T, all_steps=F == T)
        raw, scores = {}, {}
        for v, view in enumerate(VIEWS):
            raw[view] = view_measures(np.ascontiguousarray(frames[:, :, v * W:(v + 1) * W]), sims[view], view, start)
            scores[view] = frame_scores(raw[view])
        diff = view_difference(frames, sims, start)
        views_score = linear(diff, 0.0, BAD_VIEWS)

        item = {}
        for k in ("sharpness", "color", "look", "background", "flicker"):
            item[k] = lowest([worst(scores[v][k], window) for v in VIEWS])
        item["jump"] = lowest([scores[v]["jump"] for v in VIEWS])  # one frame is enough for a jump
        pattern = {k: [periodic(raw[v][k]) for v in VIEWS] for k in ("light", "flicker_video")}
        share = highest([x[0] for xs in pattern.values() for x in xs])
        sizes = {k: highest([x[1] for x in xs]) for k, xs in pattern.items()}
        item["period"] = lowest([linear(sizes["light"], 0, BAD_PERIOD_LIGHT), linear(sizes["flicker_video"], 0, BAD_PERIOD_FLICKER)])
        item["control"] = lowest([worst(scores[v]["control"], window) for v in VIEWS])
        item["views"] = worst(views_score, window)
        n_end, n_before = max(3, round(0.04 * F)), round(0.15 * F)
        end_drop = highest([lowest([np.mean(x[-n_end - n_before:-n_end][np.isfinite(x[-n_end - n_before:-n_end])])])
                            - lowest([np.mean(x[-n_end:][np.isfinite(x[-n_end:])])])
                            for x in (scores[v][k] for v in VIEWS for k in ("sharpness", "color", "background"))
                            if np.isfinite(x[-n_end:]).any() and np.isfinite(x[-n_end - n_before:-n_end]).any()])
        item["end"] = float(linear(max(end_drop, 0.0), 0.0, BAD_END)) if np.isfinite(end_drop) else float("nan")

        row.update({k: round(float(item[k]), 1) for k in ITEMS})
        low = lambda key: lowest([raw[v][key] for v in VIEWS])  # noqa: E731
        high = lambda key: highest([raw[v][key] for v in VIEWS])  # noqa: E731
        corr = [np.corrcoef(raw[v]["motion"][1:], raw[v]["sim_motion"][1:])[0, 1] for v in VIEWS]
        numbers = dict(sharp_min_ratio=low("sharp_ratio"), color_max_de=high("color_de"), towel_max_dc=high("towel_dc"),
                       table_max_de=high("table_de"), texture_max=highest([high("towel_tex"), high("table_tex")]),
                       changed_max=high("changed"), dark_max=high("dark"), floor_max_dc=high("floor_dc"),
                       flicker_max=high("flicker"), motion_corr=lowest(corr), period_share=share,
                       period_light=sizes["light"], period_flicker=sizes["flicker_video"], edge_min=low("edge"),
                       views_max=highest([diff]), end_drop=end_drop)
        row.update({k: round(x, 3) if np.isfinite(x) else "" for k, x in numbers.items()})

        frame_rows = []
        for view in VIEWS:
            for i in range(F):
                r = {"demo": idx, "name": name, "view": view, "frame": i, "step": int(steps[i])}
                r.update({k: round(float(x[i]), 4) for k, x in raw[view].items() if np.isfinite(x[i])})
                r.update({f"score_{k}": round(float(x[i]), 1) for k, x in scores[view].items() if np.isfinite(x[i])})
                if np.isfinite(diff[i]):
                    r.update(views=round(float(diff[i]), 3), score_views=round(float(views_score[i]), 1))
                frame_rows.append(r)
        strips(frames, steps, f"{out}/strips/{layout.demo_name(idx)}", name)
        return row, frame_rows
    except Exception as e:  # noqa: BLE001  one broken video must not stop the others: it gets a note and no scores
        row["note"] = f"{type(e).__name__}: {e}"
        return row, []


def strips(frames: np.ndarray, steps: np.ndarray, folder: str, name: str) -> None:
    """Images for the check by eye: a frame every STRIP_STEPS steps, four frames per image, with the step number."""
    os.makedirs(folder, exist_ok=True)
    picks = sorted({int(np.argmin(np.abs(steps - s))) for s in range(0, int(steps[-1]) + 1, STRIP_STEPS)})
    for n, at in enumerate(range(0, len(picks), 4)):
        tiles = []
        for i in picks[at:at + 4]:
            tile = frames[i].copy()
            text = f"{name}  step {int(steps[i])}  frame {i}"
            cv2.putText(tile, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(tile, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(tile)
        while len(tiles) < 4:
            tiles.append(np.zeros_like(frames[0]))
        sheet = np.concatenate([np.concatenate(tiles[:2], axis=1), np.concatenate(tiles[2:], axis=1)], axis=0)
        cv2.imwrite(f"{folder}/{name}_strip_{n:02d}.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir")
    ap.add_argument("--out", required=True, help="folder for the csv files and the strip images")
    ap.add_argument("--demos", nargs="*", default=None, help="demo numbers and ranges, e.g. 0-49 (default: all)")
    ap.add_argument("--workers", type=int, default=4, help="videos checked at the same time")
    ap.add_argument("--sim", action="store_true", help="measure the simulator frames in place of the videos")
    ap.add_argument("--marks", default=None, help="csv with demo, ghosting, shape (Y or N): the check by eye")
    args = ap.parse_args()
    ds = os.path.abspath(args.dataset_dir)
    if not status.is_dataset(ds):
        sys.exit(f"{ds} is not a dataset folder (no sources.csv)")
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    h5path = f"{ds}/{status.dataset_config(ds)['hdf5']}"
    table_path = h5path[: -len(".hdf5")] + "_instance_ids.json"
    run_tables = json.load(open(table_path)) if os.path.isfile(table_path) else {}
    wanted = status.parse_demos(args.demos)
    videos = [v for v in status.read_csv(f"{ds}/videos.csv")
              if v["where"] == "current" and (wanted is None or int(v["demo"]) in set(wanted))]
    videos.sort(key=lambda v: int(v["demo"]))
    if not videos:
        sys.exit("no videos to check")
    marks = {}
    for r in status.read_csv(args.marks) if args.marks else []:
        marks[int(r["demo"])] = {k: 0.0 if r.get(k, "").strip().upper() == "Y" else 100.0 for k in EYE_ITEMS}

    rows, frame_rows = [], []
    jobs = [(ds, h5path, run_tables, v, out, args.sim) for v in videos]
    with Pool(max(1, args.workers)) as pool:
        for row, frows in pool.imap(check_video, jobs, chunksize=1):
            if row.get("note"):
                print(f"{row['name']}: no result ({row['note']})", flush=True)
            else:
                row.update(marks.get(row["demo"], {}))
                got = {k: row[k] for k in ITEMS + EYE_ITEMS if k in row and np.isfinite(row[k])}
                row["below"] = sum(x < BELOW for x in got.values())
                row["score"] = round(float(np.mean(list(got.values()))), 1) if got else ""
                print(f"{row['name']}: score {row['score']}, {row['below']} items below {BELOW:.0f}  "
                      + " ".join(f"{k} {x:.0f}" for k, x in got.items()), flush=True)
                row.update({k: "" for k in ITEMS if not np.isfinite(row[k])})
            rows.append(row)
            frame_rows += frows
    columns = (["demo", "name", "file", "frames", "steps", "all_steps", "score", "below"] + list(ITEMS) + list(EYE_ITEMS)
               + ["sharp_min_ratio", "color_max_de", "towel_max_dc", "table_max_de", "texture_max", "changed_max",
                  "dark_max", "floor_max_dc", "flicker_max", "motion_corr", "period_share", "period_light",
                  "period_flicker", "edge_min",
                  "views_max", "end_drop", "note"])
    status.write_csv(f"{out}/check_videos.csv", rows, columns)
    frame_columns = ["demo", "name", "view", "frame", "step"]
    for r in frame_rows:
        frame_columns += [k for k in r if k not in frame_columns]
    status.write_csv(f"{out}/check_videos_frames.csv", frame_rows, frame_columns)
    done = [r for r in rows if not r.get("note")]
    if done:
        print(f"{len(done)}/{len(rows)} videos checked, median score {np.median([r['score'] for r in done]):.1f} "
              f"-> {out}/check_videos.csv")
    else:
        print(f"0/{len(rows)} videos checked")


if __name__ == "__main__":
    main()
