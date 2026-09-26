"""Observation terms for the camera images: geometry edges and raw renderer outputs.

GeoEdgeImage is copied from the SoftMimicGen fork (source/softmimicgen/softmimicgen/mdp/observations.py, 2026-09)
without changes. RawCameraImage is new. Both read the camera render product with Replicator annotators.
"""

from __future__ import annotations

import math

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.array import convert_to_torch

# camera name -> {instance id: prim path}. RawCameraImage fills it. generate_demos.py saves it as json.
INSTANCE_LABELS: dict[str, dict] = {}


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


class RawCameraImage(ManagerTermBase):
    """Raw renderer output of one camera, saved without change.

    kind = "depth":    (num_envs, H, W, 1) float32, meters, inf where nothing is hit
    kind = "normals":  (num_envs, H, W, 3) float32, xyz in -1..1
    kind = "instance": (num_envs, H, W, 1) int32, the uint32 instance id stored as int32 (same bits)

    These are the three inputs of GeoEdgeImage. The instance id -> prim path table goes to INSTANCE_LABELS
    (merged over all frames, because one frame lists only the ids it shows).
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

    def __call__(self, env, sensor_cfg: SceneEntityCfg, kind: str = "depth") -> torch.Tensor:
        if kind not in self.ANNOTATORS:
            raise ValueError(f"kind must be one of {list(self.ANNOTATORS)}, got {kind!r}")
        sensor = env.scene.sensors[sensor_cfg.name]
        if self._annots is None:
            self._attach(sensor, kind)
        height, width = sensor.image_shape
        channels = 3 if kind == "normals" else 1
        dtype = torch.int32 if kind == "instance" else torch.float32
        out = torch.zeros((env.num_envs, height, width, channels), dtype=dtype, device=env.device)
        for i, annot in enumerate(self._annots):
            raw = annot.get_data()
            info = raw.get("info") if isinstance(raw, dict) else None
            data = raw["data"] if isinstance(raw, dict) else raw
            if data is None or getattr(data, "size", 0) == 0:
                if not self._warned_empty:
                    print(f"[RawCameraImage:{kind}] annotator returned no data yet; emitting zeros for this call")
                    self._warned_empty = True
                continue
            t = convert_to_torch(data, device=env.device)  # uint32 ids arrive as int32 (same bits)
            if kind == "instance":
                if t.dtype != torch.int32:
                    raise ValueError(f"[RawCameraImage] expected raw int32 instance ids, got {t.dtype} (colorized ids?)")
                out[i] = t.reshape(height, width, 1)
                if info and "idToLabels" in info:  # only the ids visible in this frame: merge over the run
                    INSTANCE_LABELS.setdefault(sensor_cfg.name, {}).update({str(k): str(v) for k, v in info["idToLabels"].items()})
            elif kind == "normals":
                out[i] = t.reshape(height, width, -1)[..., :3].float()
            else:
                out[i] = t.reshape(height, width, 1).float()
        return out
