# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared observation functions for SoftMimicGen tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import numpy as np
from pxr import UsdGeom
from isaaclab.assets import Articulation, DeformableObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.array import convert_to_torch
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_cached_usd_mesh_points: dict[str, torch.Tensor] = {}


def ee_frame_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]

    return ee_frame_pos


def ee_frame_quat(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_quat = ee_frame.data.target_quat_w[:, 0, :]

    return ee_frame_quat


def gripper_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    finger_joint_1 = robot.data.joint_pos[:, -1].clone().unsqueeze(1)
    finger_joint_2 = -1 * robot.data.joint_pos[:, -2].clone().unsqueeze(1)

    return torch.cat((finger_joint_1, finger_joint_2), dim=1)


def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.015,
    gripper_open_val: torch.tensor = torch.tensor([0.04]),
    gripper_threshold: float = 0.005,
) -> torch.Tensor:
    """Check if the deformable object is grasped by the robot.

    Args:
        env: The environment.
        robot_cfg: Robot articulation configuration.
        ee_frame_cfg: End effector frame configuration.
        object_cfg: Deformable object configuration.
        diff_threshold: Max distance between EE and closest node for grasp.
        gripper_open_val: Joint position when gripper is fully open.
        gripper_threshold: Min deviation from open position to count as closed.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: DeformableObject = env.scene[object_cfg.name]

    nodal_pos_w = object.data.nodal_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.norm(nodal_pos_w - end_effector_pos.unsqueeze(1), dim=-1)
    min_distance = pose_diff.min(dim=1).values

    grasped = torch.logical_and(
        min_distance < diff_threshold,
        torch.abs(robot.data.joint_pos[:, -1] - gripper_open_val.to(env.device)) > gripper_threshold,
    )
    grasped = torch.logical_and(
        grasped, torch.abs(robot.data.joint_pos[:, -2] - gripper_open_val.to(env.device)) > gripper_threshold
    )

    return grasped


def object_nodal_pos(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    object: DeformableObject = env.scene[object_cfg.name]

    nodal_pos_w = object.data.nodal_pos_w.clone()

    return nodal_pos_w


def _find_mesh_prim(prim):
    """Recursively find the first UsdGeom.Mesh under a given prim."""
    if prim.IsA(UsdGeom.Mesh):
        return prim
    for child in prim.GetChildren():
        result = _find_mesh_prim(child)
        if result is not None:
            return result
    return None


def _get_local_mesh_points(env, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Read mesh vertices from USD.

    Applies the mesh prim's local xform relative to the object root prim so that
    returned points are in the object's local frame with correct scale.
    """
    cache_key = object_cfg.name
    if cache_key not in _cached_usd_mesh_points:
        obj = env.scene[object_cfg.name]
        env0_path = env.scene.env_prim_paths[0]
        prim_path = obj.cfg.prim_path.replace(env.scene.env_regex_ns, env0_path)

        stage = env.sim.stage
        root_prim = stage.GetPrimAtPath(prim_path)
        if not root_prim.IsValid():
            raise ValueError(f"USD prim not found at {prim_path}")

        mesh_prim = _find_mesh_prim(root_prim)
        if mesh_prim is None:
            raise ValueError(f"No UsdGeom.Mesh found under {prim_path}")

        mesh = UsdGeom.Mesh(mesh_prim)
        points = mesh.GetPointsAttr().Get()

        xform_cache = UsdGeom.XformCache()
        mesh_world_xform = xform_cache.GetLocalToWorldTransform(mesh_prim)
        root_world_xform = xform_cache.GetLocalToWorldTransform(root_prim)
        mesh_to_root = mesh_world_xform * root_world_xform.GetInverse()

        pts_np = np.array(points, dtype=np.float64)
        xform_np = np.array(mesh_to_root).T  # transpose for row-vector multiplication (pts @ xform)
        ones = np.ones((pts_np.shape[0], 1), dtype=np.float64)
        pts_h = np.concatenate([pts_np, ones], axis=1)  # (N, 4) homogeneous
        transformed_np = (pts_h @ xform_np)[:, :3]

        local_pts = torch.tensor(transformed_np, dtype=torch.float32, device=env.device)
        _cached_usd_mesh_points[cache_key] = local_pts

    return _cached_usd_mesh_points[cache_key]


def usd_mesh_points_w(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Get world-space surface mesh vertices for a rigid body object.

    Args:
        env: The environment instance.
        object_cfg: Scene entity config identifying the rigid object.

    Returns:
        World-space vertex positions of shape ``(num_envs, num_vertices, 3)``.
    """
    obj = env.scene[object_cfg.name]
    local_pts = _get_local_mesh_points(env, object_cfg)

    root_pos = obj.data.root_pos_w     # (num_envs, 3)

    num_envs = root_pos.shape[0]
    num_pts = local_pts.shape[0]

    if hasattr(obj.data, "root_quat_w"):
        root_quat = obj.data.root_quat_w   # (num_envs, 4)
        rotated = quat_apply(
            root_quat.unsqueeze(1).expand(-1, num_pts, -1),
            local_pts.unsqueeze(0).expand(num_envs, -1, -1),
        )
        world_pts = rotated + root_pos.unsqueeze(1)
    else:
        world_pts = local_pts.unsqueeze(0).expand(num_envs, -1, -1) + root_pos.unsqueeze(1)

    return world_pts


class ShadedCannyImage(ManagerTermBase):
    """Canny edges of the shaded instance-id segmentation, identical to Replicator's CosmosWriter "edges" modality.

    Builds the same annotator graph as ``CosmosWriter`` in instance-id mode: a Canny augmentation over
    ``shaded_instance_id_segmentation`` (default thresholds 10/100), whose colours come from the render product's
    ``instance_id_segmentation_fast`` node switched to ``colorize=True``. The graph is attached to the render
    products of an existing camera sensor, and the edges are returned as (num_envs, H, W, 1) uint8.
    Attach happens lazily on the first call so the camera's render products already exist.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._edge_annots = None
        self._seg_annots = None
        self._warned_empty = False

    def _attach(self, sensor, canny_low: int, canny_high: int):
        import omni.replicator.core as rep

        device = "cuda" if "cuda" in sensor.device else "cpu"
        self._edge_annots = []
        self._seg_annots = []
        for rp in sensor.render_product_paths:
            # Same construction and attach order as CosmosWriter: the Canny-augmented annotator first (creates the
            # shade node and its instance_id_segmentation_fast input with defaults), then a second annotator that
            # sets colorize=True on that shared input node. useCandyColours stays at its default (False) so the
            # shade kernel decodes the colorize palette exactly like the writer.
            edge = rep.AnnotatorRegistry.get_annotator("shaded_instance_id_segmentation", device=device).augment(
                "Canny", thresholdLow=canny_low, thresholdHigh=canny_high, name="shaded_canny"
            )
            seg = rep.AnnotatorRegistry.get_annotator(
                "instance_id_segmentation_fast", init_params={"colorize": True}, device=device
            )
            edge.attach(rp)
            seg.attach(rp)
            self._edge_annots.append(edge)
            self._seg_annots.append(seg)

    def __call__(
        self, env, sensor_cfg: SceneEntityCfg, canny_low: int = 10, canny_high: int = 100, inspect: bool = False
    ) -> torch.Tensor:
        sensor = env.scene.sensors[sensor_cfg.name]
        if self._edge_annots is None:
            self._attach(sensor, canny_low, canny_high)
        height, width = sensor.image_shape
        out = torch.zeros((env.num_envs, height, width, 1), dtype=torch.uint8, device=env.device)
        for i, annot in enumerate(self._edge_annots):
            data = annot.get_data()
            if data is None or getattr(data, "size", 0) == 0:
                if not self._warned_empty:
                    print("[ShadedCannyImage] annotator returned no data yet; emitting zeros for this call")
                    self._warned_empty = True
                continue
            t = convert_to_torch(data, device=env.device)
            t = t[..., :1] if t.ndim == 3 else t[..., None]
            out[i] = t.to(torch.uint8)
        return out
