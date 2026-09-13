"""Task table for the Wan pipeline: everything that differs between tasks lives here, the stage scripts stay generic.

A pipeline run is one folder  experiments/wan_canny/runs/<task>_<tag>/  holding the hdf5, the source video and
reference image, the control videos (<prefix>_<control>.mp4) and the Wan videos (<prefix>_wan_<control>.mp4), all
flat, where <prefix> = <task>_<tag>.

To add a task: register the two edge channels (<edges>_shadedcanny, <edges>_geoedge) in its env_cfg.py PolicyCfg,
then add one entry to TASKS below.
"""
import os

RUNS_DIR = "experiments/wan_canny/runs"

# control-video suffixes in pipeline order; make_wan.py makes one Wan video per suffix that exists in the run dir
CONTROLS = ["canny", "shadedcanny", "geoedge", "hed", "pidinet", "teed", "lineart"]

# ComfyUI Wan 2.2 Fun-Control template default negative prompt
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，"
    "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)

TASKS = {
    "franka_towel": dict(
        robot="franka",
        task="towel",
        camera="agentview_image",  # rgb obs key
        edges="agentview",  # obs key prefix of the simulator edge channels: <edges>_shadedcanny, <edges>_geoedge
        roi="hsv_blue",  # learned_edges ROI for the lower threshold (blue towel); "none" for other objects
        prompt="A Franka robot arm folds a blue towel on a dark table, laboratory scene, realistic video, static camera",
    ),
    "franka_rope": dict(
        robot="franka",
        task="rope",
        camera="agentview_image",
        edges="agentview",
        roi="none",
        prompt="A Franka robot arm manipulates a rope on a dark table, laboratory scene, realistic video, static camera",
    ),
    "franka_jenga": dict(
        robot="franka",
        task="jenga",
        camera="agentview_image",
        edges="agentview",
        roi="none",
        prompt="A Franka robot arm pulls a rope to move a jenga block on a dark table, laboratory scene, realistic video, "
               "static camera",
    ),
}


def run_dir(task: str, tag: str) -> str:
    return os.path.join(RUNS_DIR, f"{task}_{tag}")


def task_of(name: str) -> str:
    """Task key for a run dir or prefix such as 'franka_towel_v6' (longest key that prefixes the name)."""
    base = os.path.basename(os.path.normpath(name))
    hits = [k for k in TASKS if base == k or base.startswith(k + "_")]
    if not hits:
        raise KeyError(f"no task in tasks.TASKS matches '{base}' (known: {', '.join(TASKS)})")
    return max(hits, key=len)
