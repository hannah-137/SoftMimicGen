"""Graphs and a summary table from the results of check_videos.py, for one, two or three sets of videos.

  python experiments/policy_data/checks/plot_check_videos.py --set "81 frames=<folder>" --set "all steps=<folder>"
                                                             --out <folder> [--mark_frame 93]

Each --set is a name and a folder with check_videos.csv and check_videos_frames.csv. Output in --out:
  item_<name>.png   one graph per item with a value in every frame: the score along the demo (0 % = first step,
                    100 % = last step). Thin lines: the videos. Thick line: the median of the set. The score of a
                    frame is the lower one of the two views.
  summary.png       the scores of all videos per item and the final score, one column of dots per set, with the
                    median as a short line
  summary.csv       per set and item: number of videos, median, lowest score, videos below 50
--mark_frame N draws a line where frame N lies in the videos that have every step (for example 93, the default
chunk length of the video model). Needs matplotlib.
"""

import argparse
import collections
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import check_videos  # noqa: E402
import status  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # one fixed color per set, in the order of --set
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
FRAME_ITEMS = ("sharpness", "color", "look", "background", "flicker", "jump", "control", "views")
TITLES = {"sharpness": "1 Sharpness", "color": "2 Color and brightness", "look": "3 Look of towel and table",
          "background": "4 Background", "flicker": "5 Flicker", "jump": "6 Jumps", "period": "7 Pattern of 4 frames",
          "control": "8 Control (outlines of robot and towel)", "views": "11 Room view against wrist view",
          "end": "12 Last frames", "ghosting": "9 Ghosting (by eye)", "shape": "10 Shapes (by eye)",
          "score": "Final score"}
GRID_X = np.linspace(0.0, 100.0, 101)


def number(text) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def load(folder: str) -> tuple:
    """-> (rows of check_videos.csv with scores, {demo: {item: score per GRID_X point}}, {demo: steps})"""
    videos = [r for r in status.read_csv(f"{folder}/check_videos.csv") if not r.get("note")]
    steps = {int(r["demo"]): int(r["steps"]) for r in videos}
    by = collections.defaultdict(lambda: collections.defaultdict(dict))  # demo -> item -> step -> lowest view score
    for r in status.read_csv(f"{folder}/check_videos_frames.csv"):
        demo, step = int(r["demo"]), int(r["step"])
        for item in FRAME_ITEMS:
            x = number(r.get(f"score_{item}"))
            if np.isfinite(x):
                by[demo][item][step] = min(x, by[demo][item].get(step, np.inf))
    curves = {}
    for demo, items in by.items():
        curves[demo] = {}
        for item, values in items.items():
            at = np.array(sorted(values))
            if len(at) >= 2:
                progress = 100.0 * at / max(steps[demo] - 1, 1)
                y = np.interp(GRID_X, progress, [values[s] for s in at], left=np.nan, right=np.nan)
                curves[demo][item] = y
    return videos, curves, steps


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(True, axis="y", color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)


def item_graph(item: str, sets: list, out: str, mark: float | None, mark_frame: int | None) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.6), dpi=130, facecolor=SURFACE)
    style(ax)
    for (name, _, curves, _), color in zip(sets, COLORS):
        lines = [c[item] for c in curves.values() if item in c]
        if not lines:
            continue
        for y in lines:
            ax.plot(GRID_X, y, color=color, linewidth=0.8, alpha=0.28, solid_capstyle="round")
        stack = np.array(lines)
        seen = np.isfinite(stack).any(axis=0)  # points where at least one video has a value
        median = np.full(len(GRID_X), np.nan)
        median[seen] = np.nanmedian(stack[:, seen], axis=0)
        ax.plot(GRID_X, median, color=color, linewidth=2.0, solid_capstyle="round", solid_joinstyle="round",
                label=f"{name}: median of {len(lines)} videos")
    if mark is not None:
        ax.axvline(mark, color=MUTED, linewidth=1.0)
        ax.text(mark + 0.8, 4, f"frame {mark_frame} of the videos\nwith every step", color=MUTED, fontsize=8, va="bottom")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 103)
    ax.set_xlabel("progress of the demo (%)", color=MUTED, fontsize=9)
    ax.set_ylabel("score (100: no problem, 0: clearly bad)", color=MUTED, fontsize=9)
    ax.set_title(f"{TITLES[item]}: score in every frame (thin lines: single videos)", color=INK, fontsize=11, loc="left")
    legend = ax.legend(loc="lower left", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK)
    fig.tight_layout()
    fig.savefig(f"{out}/item_{item}.png", facecolor=SURFACE)
    plt.close(fig)


def summary(sets: list, out: str) -> None:
    items = [k for k in list(check_videos.ITEMS) + list(check_videos.EYE_ITEMS) + ["score"]
             if any(np.isfinite(number(r.get(k))) for _, videos, _, _ in sets for r in videos)]
    rows = []
    fig, ax = plt.subplots(figsize=(max(9.0, 1.05 * len(items) + 2), 5.0), dpi=130, facecolor=SURFACE)
    style(ax)
    width = 0.8 / len(sets)
    rng = np.random.default_rng(0)
    for s, ((name, videos, _, _), color) in enumerate(zip(sets, COLORS)):
        for i, item in enumerate(items):
            y = np.array([number(r.get(item)) for r in videos])
            y = y[np.isfinite(y)]
            if not len(y):
                continue
            x0 = i - 0.4 + width * (s + 0.5)
            ax.scatter(x0 + rng.uniform(-0.3, 0.3, len(y)) * width, y, s=14, color=color, alpha=0.55, linewidths=0,
                       label=name if i == 0 else None)
            ax.plot([x0 - 0.42 * width, x0 + 0.42 * width], [np.median(y)] * 2, color=INK, linewidth=2.0, solid_capstyle="round")
            rows.append({"set": name, "item": item, "videos": len(y), "median": round(float(np.median(y)), 1),
                         "lowest": round(float(y.min()), 1), "below_50": int((y < check_videos.BELOW).sum())})
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([TITLES[k].replace(" (", "\n(") for k in items], fontsize=8, color=MUTED, rotation=30, ha="right")
    ax.set_ylim(0, 103)
    ax.set_ylabel("score of a video (100: no problem, 0: clearly bad)", color=MUTED, fontsize=9)
    ax.set_title("Scores per item and final score: one dot per video, black line = median of the set", color=INK,
                 fontsize=11, loc="left")
    legend = ax.legend(loc="lower left", frameon=False, fontsize=9, markerscale=1.8)
    for text in legend.get_texts():
        text.set_color(INK)
    fig.tight_layout()
    fig.savefig(f"{out}/summary.png", facecolor=SURFACE)
    plt.close(fig)
    status.write_csv(f"{out}/summary.csv", rows, ["set", "item", "videos", "median", "lowest", "below_50"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", required=True, metavar="NAME=FOLDER", help="a name and a result folder")
    ap.add_argument("--out", required=True, help="folder for the graphs and summary.csv")
    ap.add_argument("--mark_frame", type=int, default=None, help="mark this frame of the videos that have every step")
    args = ap.parse_args()
    if len(args.set) > len(COLORS):
        sys.exit(f"at most {len(COLORS)} sets")
    sets = []
    for text in args.set:
        name, _, folder = text.partition("=")
        if not os.path.isfile(f"{folder}/check_videos.csv"):
            sys.exit(f"no check_videos.csv in {folder}")
        sets.append((name, *load(folder)))
    os.makedirs(args.out, exist_ok=True)
    mark = None
    if args.mark_frame is not None:  # where this frame lies in the all-step videos, as progress of the demo
        steps = [int(r["steps"]) for _, videos, _, _ in sets for r in videos if r.get("all_steps") == "True"]
        mark = 100.0 * args.mark_frame / (np.median(steps) - 1) if steps else None
    for item in FRAME_ITEMS:
        item_graph(item, sets, args.out, mark, args.mark_frame)
    summary(sets, args.out)
    print(f"{len(FRAME_ITEMS)} item graphs, summary.png and summary.csv -> {args.out}")


if __name__ == "__main__":
    main()
