"""Reset events for the demo generator. One event: a small random move of a fixed camera at every reset."""

from __future__ import annotations

import math

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_from_euler_xyz, quat_mul


def project_points(points_w: torch.Tensor, cam_pos: torch.Tensor, cam_quat: torch.Tensor, intrinsic: torch.Tensor) -> torch.Tensor:
    """World points (M, 3) -> pixels (M, 2) for a camera pose in the OpenGL convention (forward -Z, up +Y).
    cam_pos (3,), cam_quat (4,) w x y z, intrinsic (3, 3). Points behind the camera get pixels far outside."""
    rot = matrix_from_quat(cam_quat.view(1, 4))[0]  # camera axes in the world frame
    p = (points_w - cam_pos.view(1, 3)) @ rot  # world -> camera frame
    depth = -p[:, 2]
    ok = depth > 1e-6
    depth = torch.where(ok, depth, torch.ones_like(depth))
    u = intrinsic[0, 0] * p[:, 0] / depth + intrinsic[0, 2]
    v = intrinsic[1, 2] - intrinsic[1, 1] * p[:, 1] / depth
    far = torch.full_like(u, -1e6)
    return torch.stack([torch.where(ok, u, far), torch.where(ok, v, far)], dim=-1)


def points_in_view(points_w, cam_pos, cam_quat, intrinsic, width: int, height: int, margin_px: float) -> bool:
    """True when every point projects inside the image, at least margin_px away from the border."""
    uv = project_points(points_w, cam_pos, cam_quat, intrinsic)
    inside = (uv[:, 0] >= margin_px) & (uv[:, 0] < width - margin_px) & (uv[:, 1] >= margin_px) & (uv[:, 1] < height - margin_px)
    return bool(inside.all())


def nodal_positions(obj) -> torch.Tensor:
    """(num_envs, nodes, 3) node positions of a deformable object, read from the physics view so that the values
    written by an earlier reset event in the same reset are seen."""
    view = obj.root_physx_view
    for name in ("get_simulation_nodal_positions", "get_sim_nodal_positions"):
        if hasattr(view, name):
            return getattr(view, name)()
    return obj.data.nodal_pos_w


def randomize_camera_pose(
    env,
    env_ids: torch.Tensor,
    sensor_cfg: SceneEntityCfg,
    pos_range_m: float,
    rot_range_deg: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    margin_px: float = 16.0,
    max_tries: int = 20,
):
    """Move a fixed camera by a random offset at every reset. The pose stays the same inside the episode.

    The base pose is the camera offset of the config plus the environment origin. The noise is uniform:
    +-pos_range_m on x, y and z, and +-rot_range_deg on roll, pitch and yaw around the camera's own axes.
    A draw is kept only when every node of the deformable object projects inside the image with margin_px to
    spare (the object is not cut at the start of the demo). The node cloud is about 10 px smaller than the
    rendered towel, so 16 px on the nodes is about 6 px on the picture. After max_tries draws the base pose is
    used.
    The draws use the torch random generator, so they depend on the generation seed. Turning the noise on
    changes the random draws of the resets that follow (for example the object start pose).
    This event must come after the event that places the object.
    """
    cam = env.scene[sensor_cfg.name]
    obj = env.scene[object_cfg.name]
    dev = env.device
    nodes_all = nodal_positions(obj).to(dev)  # (num_envs, nodes, 3), the positions just written by the object reset
    width, height = cam.cfg.width, cam.cfg.height
    base_pos_all = torch.tensor(cam.cfg.offset.pos, device=dev, dtype=torch.float32).view(1, 3) + env.scene.env_origins
    base_rot = torch.tensor(cam.cfg.offset.rot, device=dev, dtype=torch.float32)  # (w, x, y, z)
    rot_range = math.radians(rot_range_deg)
    pos_out, rot_out = [], []
    for i in env_ids.tolist():
        base_pos = base_pos_all[i]
        pos, rot = base_pos, base_rot
        for _ in range(max_tries):
            d_pos = (torch.rand(3, device=dev) * 2.0 - 1.0) * pos_range_m
            ang = (torch.rand(3, device=dev) * 2.0 - 1.0) * rot_range
            d_rot = quat_from_euler_xyz(ang[0:1], ang[1:2], ang[2:3])[0]
            cand_pos, cand_rot = base_pos + d_pos, quat_mul(base_rot.view(1, 4), d_rot.view(1, 4))[0]
            if points_in_view(nodes_all[i], cand_pos, cand_rot, cam.data.intrinsic_matrices[i], width, height, margin_px):
                pos, rot = cand_pos, cand_rot
                break
        else:
            print(f"[policy_data] camera noise: no pose keeps the object in view after {max_tries} tries, using the base pose", flush=True)
        pos_out.append(pos)
        rot_out.append(rot)
    cam.set_world_poses(torch.stack(pos_out), torch.stack(rot_out), env_ids=env_ids, convention=cam.cfg.offset.convention)
