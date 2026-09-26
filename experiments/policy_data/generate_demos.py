# Based on scripts/imitation_learning/isaaclab_mimic/generate_dataset.py (Isaac Lab Project Developers, Apache-2.0).
"""Generate SoftMimicGen demos with a room camera and a wide wrist camera.

The generation itself is the upstream one (source dataset, subtask configs, success check, retry until N successful
demos, failed demos in <output>_failed.hdf5). This script only changes:

  --image_size    both cameras render at this square size (default 512)
  --wrist_focal   focal length of the wrist camera in mm (default 12; upstream Franka value is 24)
  --seed          generation seed (default 1 = upstream). Same seed gives the same generation choices
                  (start poses, noise). The physics on the GPU is not bit-exact, so pixels can differ a little.
  --far_clip      far clipping plane of both cameras in m (default: keep the task's value, 2 m for Franka)

and it records extra observations for both cameras (see observations.py):

  <camera>_geoedge        (H, W, 1) uint8, 0 or 255      geometry edges
  <camera>_depth_raw      (H, W, 1) float32, meters      raw depth (skip with --no_raw)
  <camera>_normals_raw    (H, W, 3) float32              raw surface normals
  <camera>_instance_raw   (H, W, 1) int32                raw instance id

<camera> is the camera name without "_image". The instance id table is saved next to the hdf5 as
<output>_instance_ids.json. Run inside the SoftMimicGen environment from the repository root. make_demos.sh
wraps this script.
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate demos with a room camera and a wide wrist camera.")
parser.add_argument("--task", type=str, default=None, help="Task name (default: read from the input file).")
parser.add_argument("--generation_num_trials", type=int, default=None, help="Number of successful demos.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--input_file", type=str, required=True, help="Annotated source dataset (hdf5).")
parser.add_argument("--output_file", type=str, default="./datasets/output_dataset.hdf5", help="Output dataset (hdf5).")
parser.add_argument("--enable_pinocchio", action="store_true", default=False, help="Enable Pinocchio (humanoid tasks).")
parser.add_argument("--seed", type=int, default=1, help="Generation seed.")
parser.add_argument("--image_size", type=int, default=512, help="Camera image size in pixels (square).")
parser.add_argument("--wrist_focal", type=float, default=12.0, help="Wrist camera focal length in mm.")
parser.add_argument("--room_camera", type=str, default="agentview_image", help="Scene name of the room camera.")
parser.add_argument("--wrist_camera", type=str, default="robot0_eye_in_hand_image", help="Scene name of the wrist camera.")
parser.add_argument("--no_raw", action="store_true", default=False, help="Do not record the raw depth, normals and instance id.")
parser.add_argument("--far_clip", type=float, default=None, help="Far clipping plane of both cameras in m (default: task value).")
parser.add_argument("--depth_jump", type=float, default=0.02, help="Edge rule: depth difference in m.")
parser.add_argument("--normal_angle", type=float, default=25.0, help="Edge rule: surface normal angle in degrees.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401  (must come before the app, same as the upstream script)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import asyncio  # noqa: E402
import inspect  # noqa: E402
import random  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import omni  # noqa: E402

from isaaclab.envs import ManagerBasedRLMimicEnv  # noqa: E402
from isaaclab.managers import ObservationTermCfg, SceneEntityCfg  # noqa: E402

import softmimicgen.envs  # noqa: F401, E402

if args_cli.enable_pinocchio:
    import softmimicgen.envs.pinocchio_envs  # noqa: F401
from softmimicgen.datagen.generation import env_loop, setup_async_generation, setup_env_config  # noqa: E402
from softmimicgen.datagen.utils import get_env_name_from_dataset, setup_output_paths  # noqa: E402

import softmimicgen_tasks  # noqa: F401, E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import observations  # noqa: E402


def term_prefix(camera: str) -> str:
    """agentview_image -> agentview"""
    return camera[: -len("_image")] if camera.endswith("_image") else camera


def configure(env_cfg, args) -> None:
    """Set the cameras and add the observation terms. Nothing else in the scene changes."""
    for name in (args.room_camera, args.wrist_camera):
        cam = getattr(env_cfg.scene, name, None)
        if cam is None:
            raise SystemExit(f"[policy_data] this task has no camera named {name!r}. Use --room_camera / --wrist_camera.")
        print(f"[policy_data] {name}: {cam.width}x{cam.height} -> {args.image_size}x{args.image_size}")
        cam.height = cam.width = args.image_size
        if args.far_clip is not None:
            print(f"[policy_data] {name}: far clip {cam.spawn.clipping_range[1]} -> {args.far_clip} m")
            cam.spawn.clipping_range = (cam.spawn.clipping_range[0], args.far_clip)
    wrist = getattr(env_cfg.scene, args.wrist_camera)
    print(f"[policy_data] {args.wrist_camera}: focal length {wrist.spawn.focal_length} -> {args.wrist_focal} mm")
    wrist.spawn.focal_length = args.wrist_focal

    policy = env_cfg.observations.policy
    for name in (args.room_camera, args.wrist_camera):
        prefix = term_prefix(name)
        sensor = SceneEntityCfg(name)
        terms = {
            f"{prefix}_geoedge": ObservationTermCfg(
                func=observations.GeoEdgeImage,
                params={"sensor_cfg": sensor, "depth_jump": args.depth_jump, "normal_angle_deg": args.normal_angle},
            )
        }
        if not args.no_raw:
            for kind in ("depth", "normals", "instance"):
                terms[f"{prefix}_{kind}_raw"] = ObservationTermCfg(
                    func=observations.RawCameraImage, params={"sensor_cfg": sensor, "kind": kind}
                )
        for term_name, term in terms.items():
            setattr(policy, term_name, term)
        print(f"[policy_data] {name}: observations {', '.join(terms)}")
    print(f"[policy_data] seed {args.seed}")


def main():
    output_dir, output_file_name = setup_output_paths(args_cli.output_file)
    task_name = args_cli.task.split(":")[-1] if args_cli.task else None
    env_name = task_name or get_env_name_from_dataset(args_cli.input_file)

    env_cfg, success_term = setup_env_config(
        env_name=env_name,
        output_dir=output_dir,
        output_file_name=output_file_name,
        num_envs=args_cli.num_envs,
        device=args_cli.device,
        generation_num_trials=args_cli.generation_num_trials,
    )
    env_cfg.datagen_config.seed = args_cli.seed
    configure(env_cfg, args_cli)

    env = gym.make(env_name, cfg=env_cfg).unwrapped
    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")
    if "action_noise_dict" not in inspect.signature(env.target_eef_pose_to_action).parameters:
        omni.log.warn(
            f'The "noise" parameter in the "{env_name}" environment\'s mimic API "target_eef_pose_to_action" '
            "is deprecated. Please update the API to take action_noise_dict instead."
        )

    random.seed(env.cfg.datagen_config.seed)
    np.random.seed(env.cfg.datagen_config.seed)
    torch.manual_seed(env.cfg.datagen_config.seed)
    env.reset()

    async_components = setup_async_generation(
        env=env,
        num_envs=args_cli.num_envs,
        input_file=args_cli.input_file,
        success_term=success_term,
        pause_subtask=False,
    )
    try:
        data_gen_tasks = asyncio.ensure_future(asyncio.gather(*async_components["tasks"]))
        env_loop(
            env,
            async_components["reset_queue"],
            async_components["action_queue"],
            async_components["info_pool"],
            async_components["event_loop"],
        )
    except asyncio.CancelledError:
        print("Tasks were cancelled.")
    finally:
        data_gen_tasks.cancel()
        try:
            async_components["event_loop"].run_until_complete(data_gen_tasks)
        except asyncio.CancelledError:
            print("Remaining async tasks cancelled and cleaned up.")
        except Exception as e:  # noqa: BLE001
            print(f"Error cancelling remaining async tasks: {e}")
        if observations.INSTANCE_LABELS:  # also after an interrupt
            path = os.path.join(output_dir, output_file_name + "_instance_ids.json")
            with open(path, "w") as f:
                json.dump(observations.INSTANCE_LABELS, f, indent=1, sort_keys=True)
            print(f"[policy_data] instance id table: {path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    simulation_app.close()
