# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spawners shared by the Wan-pipeline scenes."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.shapes import shapes
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass
from pxr import Sdf, Usd, UsdGeom

__all__ = ["BackdropCfg", "spawn_backdrop"]


@clone
def spawn_backdrop(
    prim_path: str,
    cfg: BackdropCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """A cuboid the cameras see but that does not block light.

    The scenes are lit by a dome light only, so a large wall placed in front of a camera darkens the whole table
    (franka_rope: table brightness 109 -> 65 with a plain cuboid). ``primvars:doNotCastShadows`` makes RTX skip the
    prim for shadow rays, so the wall only paints the background. The attribute is set on the root and on every
    gprim below it.
    """
    prim = shapes.spawn_cuboid.__wrapped__(prim_path, cfg, translation, orientation, **kwargs)
    for p in Usd.PrimRange(prim):
        if p == prim or p.IsA(UsdGeom.Gprim):
            p.CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
    return prim


@configclass
class BackdropCfg(sim_utils.CuboidCfg):
    """Cuboid backdrop that renders but casts no shadows. Same fields as :class:`isaaclab.sim.CuboidCfg`."""

    func = spawn_backdrop
