# Copyright (c) 2024-2025, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDevice, OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import GripperRetargeterCfg
from isaaclab.devices.openxr.retargeters.manipulator.se3_rel_retargeter import Se3RelRetargeterCfg
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
from softmimicgen_assets import SOFTMIMICGEN_ASSETS_DATA_DIR
import math
import torch

from . import mdp

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """ Scene configuration."""

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.457)),
        spawn=UsdFileCfg(usd_path=f"{SOFTMIMICGEN_ASSETS_DATA_DIR}/Props/Table/Table.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -0.95)),
        spawn=GroundPlaneCfg(),
    )

    # backdrop wall for Canny contrast (same colour as franka_towel); the agentview camera at (-0.03, 0.15, 0.05) looks
    # horizontally along -y, so the wall lies along x 1.65 m ahead of it. BackdropCfg renders but casts no shadows.
    backdrop = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Backdrop",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, -1.5, 1.0]),
        spawn=mdp.BackdropCfg(
            size=(6.0, 0.05, 6.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.18, 0.15)),
        ),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    agentview_image = CameraCfg(
        prim_path="{ENV_REGEX_NS}/agentview_image",
        update_period=0.0,
        height=512,  # Wan pipeline control videos are 512x512 (was 256)
        width=512,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=22, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2)
        ),
        offset=CameraCfg.OffsetCfg(pos=(-0.03, 0.15, 0.05), rot=(0.0, 0.0, 0.70711, 0.70711), convention="opengl"),
    )

    robot0_eye_in_hand_image = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/psm_tool_tip_link/robot0_eye_in_hand_image",
        update_period=0.0,
        height=84,
        width=84,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=5, focus_distance=400, horizontal_aperture=20.955, clipping_range=(0.000001, 2)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.01, 0.017), rot=(0.0, 0.0, -0.15643, -0.9876), convention="opengl"),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    body_joint_pos: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    finger_joint_pos: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)

        robot0_joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        robot0_joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        robot0_eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        robot0_eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        robot0_gripper_qpos = ObsTerm(func=mdp.gripper_pos)

        object_nodal_pos = ObsTerm(func=mdp.object_nodal_pos)

        agentview_image = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("agentview_image"), "data_type": "rgb", "normalize": False}
        )
        robot0_eye_in_hand_image = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("robot0_eye_in_hand_image"), "data_type": "rgb", "normalize": False}
        )
        # Wan pipeline edge channels (see experiments/wan_canny/tasks.py): CosmosWriter-style Canny and geometry edges
        agentview_shadedcanny = ObsTerm(
            func=mdp.ShadedCannyImage,
            params={"sensor_cfg": SceneEntityCfg("agentview_image"), "canny_low": 10, "canny_high": 100},
        )
        agentview_geoedge = ObsTerm(
            func=mdp.GeoEdgeImage,
            params={"sensor_cfg": SceneEntityCfg("agentview_image"), "depth_jump": 0.02, "normal_angle_deg": 25.0},
        )
        # shaded Canny OR depth-discontinuity edges (same annotators as the two terms above)
        agentview_shadedcanny_depth = ObsTerm(
            func=mdp.ShadedCannyDepthImage,
            params={"sensor_cfg": SceneEntityCfg("agentview_image"), "canny_low": 10, "canny_high": 100, "depth_jump": 0.02},
        )
        # the shaded instance-id segmentation itself (input of the shaded Canny), RGB, for inspection
        agentview_shaded = ObsTerm(func=mdp.ShadedSegImage, params={"sensor_cfg": SceneEntityCfg("agentview_image")})
        # the three GeoEdgeImage inputs as 8-bit images, for inspection (depth 0..far plane -> 0..255)
        agentview_depth = ObsTerm(func=mdp.GeoInputImage, params={"sensor_cfg": SceneEntityCfg("agentview_image"), "kind": "depth"})
        agentview_normals = ObsTerm(func=mdp.GeoInputImage, params={"sensor_cfg": SceneEntityCfg("agentview_image"), "kind": "normals"})
        agentview_instance = ObsTerm(func=mdp.GeoInputImage, params={"sensor_cfg": SceneEntityCfg("agentview_image"), "kind": "instance"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False
            if os.environ.get("WAN_INSPECT") != "1":  # inspection channels only with --inspect (gen.sh/make_hdf5.sh)
                self.agentview_shaded = None
                self.agentview_depth = None
                self.agentview_normals = None
                self.agentview_instance = None

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        grasp = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("object"),
                "gripper_open_val": torch.tensor([0.5]),
                "gripper_threshold": 0.3,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    init_psm_pose = EventTerm(
        func=mdp.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.2,  0.0,  0.1, 1.6,  0.2, 0.0, -0.5000,  0.5000],
        },
    )

    randomize_psm_joint_state = EventTerm(
        func=mdp.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.005,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    
    reset_object_position = EventTerm(
        func=mdp.reset_nodal_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.0, 0.0),
                "y": (-0.025, 0.025),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-math.pi/4, math.pi/4),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
        )

    reset_ring_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0), 
                "y": (-0.025, 0.025), 
                "z": (0.0, 0.0), 
                "roll": (0.0, 0.0), 
                "pitch": (0.0, 0.0), 
                "yaw": (-math.pi/4, math.pi/4),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("ring", body_names="Ring"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=False)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )

    success = DoneTerm(func=mdp.object_reached_goal)


##
# Environment configuration
##


@configclass
class EnvCfg(ManagerBasedRLEnvCfg):
    """Base environment configuration."""

    settling_steps: int = 10

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

    # Position of the XR anchor in the world frame
    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.25, -1.0),
        anchor_rot=(0.0, 0.0, 0.0, 1.0),
    )

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 5
        self.sim.render_interval = 2
        self.episode_length_s = 10.0
        self.seed = 7
        # simulation settings
        self.sim.dt = 0.01  # 100Hz

        self.viewer.eye = (0.2, 0.2, 0.1)
        self.viewer.lookat = (0.0, 0.0, 0.04)

        # Set settings for camera rendering
        self.rerender_on_reset = True
        self.sim.render.antialiasing_mode = "OFF"  # disable dlss

        # List of image observations in policy observations
        self.image_obs_list = ["agentview_image", "robot0_eye_in_hand_image"]

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        Se3RelRetargeterCfg(
                            bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT,
                            zero_out_xy_rotation=True,
                            use_wrist_rotation=True,
                            use_wrist_position=True,
                            delta_pos_scale_factor=1.0,
                            delta_rot_scale_factor=1.0,
                            alpha_pos=0.5,
                            alpha_rot=0.5,
                            sim_device=self.sim.device,
                        ),
                        GripperRetargeterCfg(
                            bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT, sim_device=self.sim.device
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.005,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )
