# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul
from softmimicgen_assets import SOFTMIMICGEN_ASSETS_DATA_DIR
import json
import torch
import math
from . import mdp

##
# Scene definition
##


# Camera information
width, height = 640, 480

intrinsic_matrix = [
    392.1729, 0, 324.6466,
    0, 391.6392, 244.5516,
    0, 0, 1
]

# Wan pipeline: the top camera renders 512x512 (control videos are square). Same focal length, centred principal
# point = the original 640x480 view cropped to a square (64 px cut on each side, 16 px gained top and bottom).
top_width, top_height = 512, 512
top_intrinsic_matrix = [
    392.1729, 0, 256.0,
    0, 391.6392, 256.0,
    0, 0, 1
]


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """ Scene configuration."""

    # robots: will be populated by agent env cfg
    robot_1: ArticulationCfg = MISSING
    robot_2: ArticulationCfg = MISSING

    # YAM workstation
    workstation = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Workstation",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{SOFTMIMICGEN_ASSETS_DATA_DIR}/Robots/yam/workstation/workstation.usd",
            scale=(1.0, 1.02, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.06, 0.0)),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # top camera
    top_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/TopCamera",
        update_period=0.0,
        update_latest_camera_pose=True,
        height=top_height,
        width=top_width,
        data_types=[
            "rgb",
            "distance_to_image_plane",
            "semantic_segmentation",
        ],
        colorize_semantic_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=top_intrinsic_matrix,
            width=top_width,
            height=top_height,
            f_stop=0.0,
            projection_type="pinhole",
            lock_camera=False,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.086, -0.009, 1.7043),
            rot=(0.67842, 0.18981, -0.19408, -0.68268),
            convention="opengl"
        ),
    )

    # left wrist camera
    left_wrist_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot_1/arm/camera_d405/wrist_camera",
        update_period=0.0,
        update_latest_camera_pose=True,
        height=height,
        width=width,
        data_types=[
            "rgb",
        ],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=intrinsic_matrix,
            width=width,
            height=height,
            f_stop=0.0,
            projection_type="pinhole",
            lock_camera=False,
        ),
        offset=CameraCfg.OffsetCfg(
            rot=quat_from_euler_xyz(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([torch.pi])).squeeze(0).cpu().numpy(),
        ),
    )
    
    # right wrist camera
    right_wrist_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot_2/arm/camera_d405/wrist_camera",
        update_period=0.0,
        update_latest_camera_pose=True,
        height=height,
        width=width,
        data_types=[
            "rgb",
        ],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=intrinsic_matrix,
            width=width,
            height=height,
            f_stop=0.0,
            projection_type="pinhole",
            lock_camera=False,
        ),
        offset=CameraCfg.OffsetCfg(
            rot=quat_from_euler_xyz(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([torch.pi])).squeeze(0).cpu().numpy(),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_1_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_1_action: mdp.BinaryJointPositionActionCfg = MISSING
    arm_2_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_2_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        actions = ObsTerm(func=mdp.last_action)

        actions_ee = ObsTerm(func=mdp.actions_ee)

        joint_1_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot_1")})
        joint_2_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot_2")})

        robot0_eef_pos = ObsTerm(func=mdp.ee_frame_pos, params={"ee_frame_cfg": SceneEntityCfg("ee_1_frame")})
        robot1_eef_pos = ObsTerm(func=mdp.ee_frame_pos, params={"ee_frame_cfg": SceneEntityCfg("ee_2_frame")})

        robot0_eef_quat = ObsTerm(func=mdp.ee_frame_quat, params={"ee_frame_cfg": SceneEntityCfg("ee_1_frame")})
        robot1_eef_quat = ObsTerm(func=mdp.ee_frame_quat, params={"ee_frame_cfg": SceneEntityCfg("ee_2_frame")})

        robot0_gripper_qpos = ObsTerm(func=mdp.gripper_pos, params={"robot_cfg": SceneEntityCfg("robot_1")})
        robot1_gripper_qpos = ObsTerm(func=mdp.gripper_pos, params={"robot_cfg": SceneEntityCfg("robot_2")})

        state = ObsTerm(func=mdp.state)

        # Wan pipeline: native 512x512 (no crop/resize) so the rgb obs and the edge channels below are pixel-aligned
        top = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("top_camera"), "data_type": "rgb", "normalize": False},
        )
        top_shadedcanny = ObsTerm(
            func=mdp.ShadedCannyImage,
            params={"sensor_cfg": SceneEntityCfg("top_camera"), "canny_low": 10, "canny_high": 100},
        )
        top_geoedge = ObsTerm(
            func=mdp.GeoEdgeImage,
            params={"sensor_cfg": SceneEntityCfg("top_camera"), "depth_jump": 0.02, "normal_angle_deg": 25.0},
        )
        # shaded Canny OR depth-discontinuity edges (same annotators as the two terms above)
        top_shadedcanny_depth = ObsTerm(
            func=mdp.ShadedCannyDepthImage,
            params={"sensor_cfg": SceneEntityCfg("top_camera"), "canny_low": 10, "canny_high": 100, "depth_jump": 0.02},
        )
        # the shaded instance-id segmentation itself (input of the shaded Canny), RGB, for inspection
        top_shaded = ObsTerm(func=mdp.ShadedSegImage, params={"sensor_cfg": SceneEntityCfg("top_camera")})
        # the three GeoEdgeImage inputs as 8-bit images, for inspection (depth 0..far plane -> 0..255)
        top_depth = ObsTerm(func=mdp.GeoInputImage, params={"sensor_cfg": SceneEntityCfg("top_camera"), "kind": "depth"})
        top_normals = ObsTerm(func=mdp.GeoInputImage, params={"sensor_cfg": SceneEntityCfg("top_camera"), "kind": "normals"})
        top_instance = ObsTerm(func=mdp.GeoInputImage, params={"sensor_cfg": SceneEntityCfg("top_camera"), "kind": "instance"})

        left = ObsTerm(
            func=mdp.image_cropped,
            params={
                "sensor_cfg": SceneEntityCfg("left_wrist_camera"),
                "data_type": "rgb",
                "normalize": False,
                "target_height": 84,
                "target_width": 84,
            },
        )

        right = ObsTerm(
            func=mdp.image_cropped,
            params={
                "sensor_cfg": SceneEntityCfg("right_wrist_camera"),
                "data_type": "rgb",
                "normalize": False,
                "target_height": 84,
                "target_width": 84,
            },
        )


        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False
            if os.environ.get("WAN_INSPECT") != "1":  # inspection channels only with --inspect (gen.sh/make_hdf5.sh)
                self.top_shaded = None
                self.top_depth = None
                self.top_normals = None
                self.top_instance = None

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    init_robot_1_pose = EventTerm(
        func=mdp.set_default_joint_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_1"),
            "default_pose": [
                0.00,
                1.047,
                1.047,
                0.00,
                0.00,
                0.00,
                0.04,
                -0.04,
            ],
        },
    )

    randomize_robot_1_joint_state = EventTerm(
        func=mdp.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot_1"),
        },
    )

    init_robot_2_pose = EventTerm(
        func=mdp.set_default_joint_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_2"),
            "default_pose": [
                0.00,
                1.047,
                1.047,
                0.00,
                0.00,
                0.00,
                0.04,
                -0.04,
            ],
        },
    )

    randomize_robot_2_joint_state = EventTerm(
        func=mdp.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot_2"),
        },
    )

    reset_bag = EventTerm(
        func=mdp.reset_nodal_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-math.pi/9, math.pi/18),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("bag"),
        },
    )

    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.05), 
                "y": (-0.1, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-math.pi/9, math.pi/9),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("banana", body_names="link_iter_2_canonical_mesh_obj"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.7, "asset_cfg": SceneEntityCfg("bag")}
    )

    success = DoneTerm(
        func=mdp.object_in_bag,
        params={
            "banana_cfg": SceneEntityCfg("banana"),
            "bag_cfg": SceneEntityCfg("bag"),
            "robot_1_cfg": SceneEntityCfg("robot_1"),
            "robot_2_cfg": SceneEntityCfg("robot_2"),
            "containment_radius": 0.22,
            "velocity_threshold": 0.2,
        },
    )


##
# Environment configuration
##


@configclass
class EnvCfg(ManagerBasedRLEnvCfg):
    """Base environment configuration."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    commands = None
    rewards = None
    curriculum = None


    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 100.0
        self.seed = 101
        # simulation settings
        self.sim.dt = 1/120  # 120Hz
        self.sim.render_interval = 4

        self.viewer.eye = (0, 0.0, 1.6)
        self.viewer.lookat = (0.65, 0.0, 0.75)

        # set settings for camera rendering
        self.rerender_on_reset = True
        self.sim.render.antialiasing_mode = "OFF"  # disable dlss

        # list of image observations in policy observations
        self.image_obs_list = ["top", "left", "right"]
