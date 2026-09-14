# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared observation functions for SoftMimicGen tasks."""

from __future__ import annotations

import math
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


class GeoEdgeImage(ManagerTermBase):
    """Geometry edges from the camera's own render product, no RGB involved: a pixel is an edge when its right or
    lower neighbour differs in depth (> depth_jump metres), surface normal (angle > normal_angle_deg) or instance id.

    Attaches three annotators (distance_to_image_plane, normals, instance_id_segmentation_fast) to the render
    products of an existing camera sensor on the first call and returns (num_envs, H, W, 1) uint8 {0, 255}.
    Comparing only right/lower neighbours gives 1 px lines; the 1 px image border is always 0. Depth is read as
    the annotator returns it (inf where nothing is hit -> replaced by the largest finite value); the edge test
    runs entirely in torch on the env device (annotator buffers are shared with torch, no CPU round trip).
    """

    ANNOTATORS = ("distance_to_image_plane", "normals", "instance_id_segmentation_fast")

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._annots = None
        self._warned_empty = False

    def _attach(self, sensor):
        import omni.replicator.core as rep

        device = "cuda" if "cuda" in sensor.device else "cpu"
        self._annots = []
        for rp in sensor.render_product_paths:
            # No init_params on instance_id_segmentation_fast: the node is shared per render product and
            # ShadedCannyImage sets colorize=True on it; passing colorize here would overwrite that. The id test
            # below works on raw uint32 ids and on colorized RGBA alike (different id -> different value).
            trio = tuple(rep.AnnotatorRegistry.get_annotator(name, device=device) for name in self.ANNOTATORS)
            for annot in trio:
                annot.attach(rp)
            self._annots.append(trio)

    def __call__(
        self,
        env,
        sensor_cfg: SceneEntityCfg,
        depth_jump: float = 0.02,
        normal_angle_deg: float = 25.0,
        inspect: bool = False,
    ) -> torch.Tensor:
        sensor = env.scene.sensors[sensor_cfg.name]
        if self._annots is None:
            self._attach(sensor)
        height, width = sensor.image_shape
        out = torch.zeros((env.num_envs, height, width, 1), dtype=torch.uint8, device=env.device)
        cos_thr = math.cos(math.radians(normal_angle_deg))
        for i, (a_depth, a_normal, a_seg) in enumerate(self._annots):
            raw = [a.get_data() for a in (a_depth, a_normal, a_seg)]
            # segmentation annotators return {"data": buffer, "info": {...}} (see isaaclab Camera._process_annotator_output)
            raw = [r["data"] if isinstance(r, dict) else r for r in raw]
            if any(r is None or getattr(r, "size", 0) == 0 for r in raw):
                if not self._warned_empty:
                    print("[GeoEdgeImage] annotators returned no data yet; emitting zeros for this call")
                    self._warned_empty = True
                continue
            d = convert_to_torch(raw[0], device=env.device).reshape(height, width).float()
            n = convert_to_torch(raw[1], device=env.device).reshape(height, width, -1)[..., :3].float()
            s = convert_to_torch(raw[2], device=env.device)
            if hasattr(torch, "uint32") and s.dtype == torch.uint32:
                s = s.view(torch.int32)  # uint32 has few ops in torch; bit-identical view is enough for !=
            s = s.reshape(height, width, -1)

            finite = torch.isfinite(d)
            if not bool(finite.all()):
                fill = d[finite].max() if bool(finite.any()) else torch.zeros((), device=d.device)
                d = torch.where(finite, d, fill)
            n_len = n.norm(dim=-1)
            valid = n_len > 1e-6  # background pixels have zero normals: excluded from the angle test
            n = n / n_len.clamp_min(1e-6).unsqueeze(-1)

            edge = torch.zeros((height, width), dtype=torch.bool, device=env.device)
            # right neighbour
            edge[:, :-1] |= (d[:, 1:] - d[:, :-1]).abs() > depth_jump
            edge[:, :-1] |= ((n[:, 1:] * n[:, :-1]).sum(-1) < cos_thr) & valid[:, 1:] & valid[:, :-1]
            edge[:, :-1] |= (s[:, 1:] != s[:, :-1]).any(-1)
            # lower neighbour
            edge[:-1, :] |= (d[1:, :] - d[:-1, :]).abs() > depth_jump
            edge[:-1, :] |= ((n[1:, :] * n[:-1, :]).sum(-1) < cos_thr) & valid[1:, :] & valid[:-1, :]
            edge[:-1, :] |= (s[1:, :] != s[:-1, :]).any(-1)
            edge[0, :] = False
            edge[-1, :] = False
            edge[:, 0] = False
            edge[:, -1] = False
            out[i, ..., 0] = edge.to(torch.uint8) * 255
        return out


class ShadedCannyDepthImage(ManagerTermBase):
    """Shaded-Canny edges OR depth-discontinuity edges, (num_envs, H, W, 1) uint8 {0, 255}.

    The Canny half is exactly :class:`ShadedCannyImage` (same annotator graph, same thresholds); the depth half is
    the depth rule of :class:`GeoEdgeImage` alone (right/lower neighbour differs by more than ``depth_jump`` metres;
    normals and instance ids are not used). The annotators are the same templates the other two terms attach to
    the same render product, so their shared nodes are reused untouched: the Canny-augmented annotator gets no
    ``init_params`` (they would land on the Canny node) and ``instance_id_segmentation_fast`` is attached with
    ``colorize=True``, the value ShadedCannyImage sets too. If ShadedCannyImage is registered on the same camera,
    give both terms the same canny thresholds (the shared Canny node keeps the first ones it was created with).
    Attach happens lazily on the first call so the camera's render products already exist.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._annots = None  # per render product: (canny-edge annotator, depth annotator)
        self._warned_empty = False

    def _attach(self, sensor, canny_low: int, canny_high: int):
        import omni.replicator.core as rep

        device = "cuda" if "cuda" in sensor.device else "cpu"
        self._annots = []
        for rp in sensor.render_product_paths:
            edge = rep.AnnotatorRegistry.get_annotator("shaded_instance_id_segmentation", device=device).augment(
                "Canny", thresholdLow=canny_low, thresholdHigh=canny_high, name="shaded_canny"
            )
            seg = rep.AnnotatorRegistry.get_annotator(
                "instance_id_segmentation_fast", init_params={"colorize": True}, device=device
            )
            depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane", device=device)
            edge.attach(rp)
            seg.attach(rp)
            depth.attach(rp)
            self._annots.append((edge, depth))

    def __call__(
        self,
        env,
        sensor_cfg: SceneEntityCfg,
        canny_low: int = 10,
        canny_high: int = 100,
        depth_jump: float = 0.02,
        inspect: bool = False,
    ) -> torch.Tensor:
        sensor = env.scene.sensors[sensor_cfg.name]
        if self._annots is None:
            self._attach(sensor, canny_low, canny_high)
        height, width = sensor.image_shape
        out = torch.zeros((env.num_envs, height, width, 1), dtype=torch.uint8, device=env.device)
        for i, (a_edge, a_depth) in enumerate(self._annots):
            raw = [a.get_data() for a in (a_edge, a_depth)]
            raw = [r["data"] if isinstance(r, dict) else r for r in raw]  # segmentation-type annotators return dicts
            if any(r is None or getattr(r, "size", 0) == 0 for r in raw):
                if not self._warned_empty:
                    print("[ShadedCannyDepthImage] annotators returned no data yet; emitting zeros for this call")
                    self._warned_empty = True
                continue
            canny = convert_to_torch(raw[0], device=env.device)
            canny = canny[..., 0] if canny.ndim == 3 else canny
            canny = canny.reshape(height, width) > 0
            d = convert_to_torch(raw[1], device=env.device).reshape(height, width).float()
            finite = torch.isfinite(d)
            if not bool(finite.all()):
                fill = d[finite].max() if bool(finite.any()) else torch.zeros((), device=d.device)
                d = torch.where(finite, d, fill)
            edge = torch.zeros((height, width), dtype=torch.bool, device=env.device)
            edge[:, :-1] |= (d[:, 1:] - d[:, :-1]).abs() > depth_jump  # right neighbour
            edge[:-1, :] |= (d[1:, :] - d[:-1, :]).abs() > depth_jump  # lower neighbour
            edge[0, :] = False
            edge[-1, :] = False
            edge[:, 0] = False
            edge[:, -1] = False
            out[i, ..., 0] = (canny | edge).to(torch.uint8) * 255
        return out


class ShadedSegImage(ManagerTermBase):
    """The colourised shaded instance-id segmentation itself, the image ShadedCannyImage runs Canny on,
    as (num_envs, H, W, 3) uint8 RGB. For inspection and reports, not a control signal.

    Attaches the same shared nodes as :class:`ShadedCannyImage` (``shaded_instance_id_segmentation`` fed by
    ``instance_id_segmentation_fast`` with ``colorize=True``) but reads the shade node's output directly, before
    the Canny augmentation. No ``init_params`` are given to the shade node, so nothing shared is re-initialised.
    Attach happens lazily on the first call so the camera's render products already exist.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._annots = None
        self._warned_empty = False

    def _attach(self, sensor):
        import omni.replicator.core as rep

        device = "cuda" if "cuda" in sensor.device else "cpu"
        self._annots = []
        for rp in sensor.render_product_paths:
            shade = rep.AnnotatorRegistry.get_annotator("shaded_instance_id_segmentation", device=device)
            seg = rep.AnnotatorRegistry.get_annotator(
                "instance_id_segmentation_fast", init_params={"colorize": True}, device=device
            )
            shade.attach(rp)
            seg.attach(rp)
            self._annots.append(shade)

    def __call__(self, env, sensor_cfg: SceneEntityCfg, inspect: bool = False) -> torch.Tensor:
        sensor = env.scene.sensors[sensor_cfg.name]
        if self._annots is None:
            self._attach(sensor)
        height, width = sensor.image_shape
        out = torch.zeros((env.num_envs, height, width, 3), dtype=torch.uint8, device=env.device)
        for i, annot in enumerate(self._annots):
            data = annot.get_data()
            data = data["data"] if isinstance(data, dict) else data
            if data is None or getattr(data, "size", 0) == 0:
                if not self._warned_empty:
                    print("[ShadedSegImage] annotator returned no data yet; emitting zeros for this call")
                    self._warned_empty = True
                continue
            t = convert_to_torch(data, device=env.device)
            if t.ndim == 2:  # packed RGBA in one 32-bit value per pixel
                t = t.view(torch.uint8).reshape(height, width, -1)
            out[i] = t.reshape(height, width, -1)[..., :3].to(torch.uint8)
        return out


class GeoInputImage(ManagerTermBase):
    """One input of :class:`GeoEdgeImage` as an 8-bit image, for inspection (not a control signal).

    ``kind="depth"``: (num_envs, H, W, 1), distance_to_image_plane mapped linearly from 0..``max_depth`` metres to
    0..255 (``max_depth`` defaults to the camera's far clipping plane; inf and anything beyond -> 255).
    ``kind="normals"``: (num_envs, H, W, 3), normal xyz in -1..1 mapped to 0..255 (pixels without a surface -> 0).
    ``kind="instance"``: (num_envs, H, W, 3), the colourised instance-id segmentation as RGB.
    Attaches the same annotator template GeoEdgeImage uses for that input, so the shared node is reused with no
    init_params. Attach happens lazily on the first call.
    """

    ANNOTATORS = {"depth": "distance_to_image_plane", "normals": "normals", "instance": "instance_id_segmentation_fast"}

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._annots = None
        self._warned_empty = False

    def _attach(self, sensor, kind: str):
        import omni.replicator.core as rep

        device = "cuda" if "cuda" in sensor.device else "cpu"
        self._annots = []
        for rp in sensor.render_product_paths:
            annot = rep.AnnotatorRegistry.get_annotator(self.ANNOTATORS[kind], device=device)
            annot.attach(rp)
            self._annots.append(annot)

    def __call__(
        self, env, sensor_cfg: SceneEntityCfg, kind: str = "depth", max_depth: float | None = None, inspect: bool = False
    ) -> torch.Tensor:
        if kind not in self.ANNOTATORS:
            raise ValueError(f"kind must be one of {tuple(self.ANNOTATORS)}, got {kind!r}")
        sensor = env.scene.sensors[sensor_cfg.name]
        if self._annots is None:
            self._attach(sensor, kind)
        height, width = sensor.image_shape
        channels = 1 if kind == "depth" else 3
        out = torch.zeros((env.num_envs, height, width, channels), dtype=torch.uint8, device=env.device)
        for i, annot in enumerate(self._annots):
            data = annot.get_data()
            data = data["data"] if isinstance(data, dict) else data
            if data is None or getattr(data, "size", 0) == 0:
                if not self._warned_empty:
                    print(f"[GeoInputImage:{kind}] annotator returned no data yet; emitting zeros for this call")
                    self._warned_empty = True
                continue
            t = convert_to_torch(data, device=env.device)
            if kind == "depth":
                d = t.reshape(height, width).float()
                far = float(max_depth) if max_depth else float(sensor.cfg.spawn.clipping_range[1])
                d = torch.where(torch.isfinite(d), d, torch.full_like(d, far))
                out[i, ..., 0] = ((d / far).clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
            elif kind == "normals":
                n = t.reshape(height, width, -1)[..., :3].float()
                out[i] = ((n.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
            else:  # instance: colourised RGBA (packed 32-bit when the buffer comes as one value per pixel)
                if t.ndim == 2:
                    t = t.view(torch.uint8).reshape(height, width, -1)
                out[i] = t.reshape(height, width, -1)[..., :3].to(torch.uint8)
        return out
