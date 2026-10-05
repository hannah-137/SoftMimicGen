"""The table of the Franka tasks with a clean top: no beam, no handles, no bolts and no bolt holes.

generate_demos.py --table clean_top calls build(). The upstream table asset (SeattleLabTable) has metal parts that
show in the camera images: a beam and a post beside the table, handles at both ends, and bolts in the top. build()
writes a small USD file. The file references the upstream asset and changes only the table mesh:

1. It leaves out the faces of the metal parts (material table_parts). The wheels under the body stay, so the table
   does not float.
2. It closes the small holes that the bolts leave in the flat faces of the top. Each hole gets a fan of triangles
   in the plane of the face. The new faces have the material of the body, and their texture coordinates continue
   the ones around the hole.

The table body with its material and textures, its size and place, and the collision box are the ones of the
upstream asset. The upstream file is not changed. The image still shows faint ring marks at the old holes: they are
painted in the base color texture of the asset.

build() needs the USD Python modules (pxr): call it after the simulation app has started.
"""

import collections

import numpy as np

METAL, BODY = "table_parts", "table_base"  # material names in the upstream asset
MESH = "Visuals/TableGeom"  # the table mesh, under the default prim of the asset
HOLE_RADIUS = 0.02  # m: a hole in the top with a smaller radius is closed
UNDER = 0.02  # a metal part stays when its highest point is in the lowest 2% of the body height (wheels, feet)


def _weld(points: np.ndarray) -> np.ndarray:
    """One id per place: points at the same place get the same id."""
    return np.unique(np.round(points, 5), axis=0, return_inverse=True)[1].reshape(-1)


def _parts(tri: np.ndarray, weld: np.ndarray) -> list:
    """Connected parts of the mesh -> list of face index arrays. Faces that share a place of a point are joined."""
    parent = np.arange(weld.max() + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for f in range(len(tri)):
        v = weld[tri[f]]
        r = find(v[0])
        for x in v[1:]:
            rx = find(x)
            if rx != r:
                parent[rx] = r
    parts = collections.defaultdict(list)
    for f in range(len(tri)):
        parts[find(weld[tri[f, 0]])].append(f)
    return [np.array(faces) for faces in parts.values()]


def _top_levels(tri: np.ndarray, w: np.ndarray, is_body: np.ndarray, up: int) -> list:
    """Groups of body faces that are flat, face up, and lie at the height of the top or up to 5 mm above it. The top
    is the height with the largest area of such faces."""
    a, b, c = w[tri[:, 0]], w[tri[:, 1]], w[tri[:, 2]]
    normal = np.cross(b - a, c - a)
    area = np.linalg.norm(normal, axis=1) / 2
    normal = normal / np.maximum(2 * area[:, None], 1e-15)
    height = (a[:, up] + b[:, up] + c[:, up]) / 3
    flat_up = is_body & (normal[:, up] > 0.999)
    by_height = collections.Counter()
    for h, ar in zip(np.round(height[flat_up], 4), area[flat_up]):
        by_height[float(h)] += ar
    top = max(by_height, key=by_height.get)
    hs = sorted(h for h in by_height if top - 5e-4 <= h <= top + 5e-3)
    groups = [[hs[0]]]
    for h in hs[1:]:
        if h - groups[-1][-1] < 3e-4:
            groups[-1].append(h)
        else:
            groups.append([h])
    return [np.nonzero(flat_up & (height >= g[0] - 1e-4) & (height <= g[-1] + 1e-4))[0] for g in groups]


def _loops(tri: np.ndarray, w: np.ndarray, weld: np.ndarray, faces: np.ndarray, flat: list) -> list:
    """Boundary loops of a set of faces -> [(corners, center, radius)]. corners: one (face, corner) per loop point."""
    edges, corner_of = collections.Counter(), {}
    for f in faces:
        v = weld[tri[f]]
        for j in range(3):
            corner_of.setdefault(int(v[j]), (int(f), j))
            edges[tuple(sorted((int(v[j]), int(v[(j + 1) % 3]))))] += 1
    near = collections.defaultdict(list)
    for (x, y), k in edges.items():
        if k == 1:  # an edge of one face only: it lies on a boundary
            near[x].append(y)
            near[y].append(x)
    seen, out = set(), []
    for start in near:
        if start in seen:
            continue
        stack, loop = [start], []
        seen.add(start)
        while stack:
            x = stack.pop()
            loop.append(x)
            for y in near[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        corners = [corner_of[k] for k in loop]
        p = w[[tri[f, j] for f, j in corners]]
        center = p.mean(0)
        out.append((corners, center, float(np.linalg.norm((p - center)[:, flat], axis=1).max())))
    return out


def _covered(point: np.ndarray, tri: np.ndarray, w: np.ndarray, faces: np.ndarray, flat: list) -> bool:
    """True when the point, seen from above, lies inside one of the faces. Then a small loop is an island, no hole."""
    a, b, c = (w[tri[faces, k]][:, flat] for k in range(3))
    p = point[flat]

    def side(u, v):
        return (v[:, 0] - u[:, 0]) * (p[1] - u[:, 1]) - (v[:, 1] - u[:, 1]) * (p[0] - u[:, 0])

    s1, s2, s3 = side(a, b), side(b, c), side(c, a)
    return bool((((s1 >= 0) & (s2 >= 0) & (s3 >= 0)) | ((s1 <= 0) & (s2 <= 0) & (s3 <= 0))).any())


def _holes(tri: np.ndarray, w: np.ndarray, weld: np.ndarray, is_body: np.ndarray, up: int) -> list:
    """The small holes in the flat faces of the top -> [(corners of the rim, center, radius)]."""
    flat = [i for i in range(3) if i != up]
    out = []
    for faces in _top_levels(tri, w, is_body, up):
        for corners, center, radius in _loops(tri, w, weld, faces, flat):
            if radius < HOLE_RADIUS and len(corners) >= 3 and not _covered(center, tri, w, faces, flat):
                out.append((corners, center, radius))
    return out


def build(source: str, out: str) -> dict:
    """Write the clean table to `out` (a USD file). `source` is the upstream table asset. -> counts for the log:
    metal_parts, metal_faces (left out), holes, hole_faces (added), holes_left (0 when every small hole is closed)."""
    from pxr import Usd, UsdGeom, UsdShade

    stage = Usd.Stage.Open(source)
    root = stage.GetDefaultPrim() if stage else None
    prim = stage.GetPrimAtPath(root.GetPath().AppendPath(MESH)) if root else None
    if not prim or not prim.IsA(UsdGeom.Mesh):
        raise SystemExit(f"{source}: no mesh {MESH}. --table clean_top works only with the SeattleLabTable asset.")
    mesh = UsdGeom.Mesh(prim)
    points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    if not (np.array(mesh.GetFaceVertexCountsAttr().Get()) == 3).all():
        raise SystemExit(f"{source}: the table mesh must have triangles only")
    tri = np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    n_face = len(tri)
    up = {"Y": 1, "Z": 2}[UsdGeom.GetStageUpAxis(stage)]
    matrix = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))

    def in_asset_frame(p):
        return (np.c_[p, np.ones(len(p))] @ matrix)[:, :3]

    subsets = UsdGeom.Subset.GetAllGeomSubsets(mesh)
    material_of = {}
    face_material = np.full(n_face, "", dtype=object)
    for s in subsets:
        m = UsdShade.MaterialBindingAPI(s.GetPrim()).ComputeBoundMaterial()[0]
        material_of[s.GetPrim().GetName()] = m.GetPath().name if m else ""
        face_material[np.array(s.GetIndicesAttr().Get())] = material_of[s.GetPrim().GetName()]
    if set(face_material) != {METAL, BODY}:
        raise SystemExit(f"{source}: the table mesh must have the materials {METAL} and {BODY}, found "
                         f"{sorted(set(face_material))}")

    # 1. The metal parts: connected parts with the metal material that are not under the body.
    w, weld = in_asset_frame(points), _weld(points)
    parts = []
    for faces in _parts(tri, weld):
        materials = set(face_material[faces])
        if len(materials) != 1:
            raise SystemExit(f"{source}: a connected part of the table mesh has two materials")
        p = w[np.unique(tri[faces])]
        parts.append((faces, materials.pop(), p.min(0)[up], p.max(0)[up]))
    _, _, low, high = max((p for p in parts if p[1] == BODY), key=lambda p: len(p[0]))
    keep = np.ones(n_face, bool)
    metal_parts = 0
    for faces, material, _, part_high in parts:
        if material == METAL and part_high >= low + UNDER * (high - low):
            keep[faces] = False
            metal_parts += 1

    # 2. The holes: a fan of triangles from the points of each rim to a new point at its center.
    corner_values = {}  # name -> (attribute, values per face corner (faces, 3, dim))
    normals = mesh.GetNormalsAttr()
    if not normals.HasValue() or mesh.GetNormalsInterpolation() != UsdGeom.Tokens.faceVarying:
        raise SystemExit(f"{source}: the table mesh must have faceVarying normals")
    corner_values["normals"] = (normals, np.array(normals.Get(), dtype=np.float64).reshape(n_face, 3, -1))
    for pv in UsdGeom.PrimvarsAPI(prim).GetPrimvars():
        if pv.HasValue():
            if pv.GetInterpolation() != UsdGeom.Tokens.faceVarying or pv.IsIndexed():
                raise SystemExit(f"{source}: primvar {pv.GetName()} of the table mesh must be faceVarying, not indexed")
            corner_values[pv.GetName()] = (pv.GetAttr(), np.array(pv.Get(), dtype=np.float64).reshape(n_face, 3, -1))
    new_points, new_tri, new_values = [], [], {name: [] for name in corner_values}
    holes = _holes(tri, w, weld, keep & (face_material == BODY), up)
    for corners, _, _ in holes:
        index = np.array([tri[f, j] for f, j in corners])
        rim = points[index]
        center = rim.mean(0)
        f0 = corners[0][0]
        normal = np.cross(points[tri[f0, 1]] - points[tri[f0, 0]], points[tri[f0, 2]] - points[tri[f0, 0]])
        normal = normal / np.linalg.norm(normal)
        u = rim[0] - center
        u = u - (u @ normal) * normal
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        d = rim - center
        order = np.argsort(np.arctan2(d @ v, d @ u))  # counter-clockwise around the normal of the face
        ring = [(int(index[i]), corners[i]) for i in order]
        plane = np.c_[d @ u, d @ v, np.ones(len(d))][order]
        at_center = {}
        for name, (_, values) in corner_values.items():
            at_rim = np.array([values[f, j] for _, (f, j) in ring])
            if values.shape[2] == 2:  # texture coordinates: the plane that fits the rim, read at the center
                fit = np.linalg.lstsq(plane, at_rim, rcond=None)[0]
                at_center[name] = fit[2] if np.abs(plane @ fit - at_rim).max() < 0.01 else at_rim.mean(0)
            else:  # normals and tangents: the mean of the rim, with length 1 when the rim values have length 1
                c = at_rim.mean(0)
                if np.allclose(np.linalg.norm(at_rim, axis=1), 1.0, atol=1e-3) and np.linalg.norm(c) > 1e-6:
                    c = c / np.linalg.norm(c)
                at_center[name] = c
        center_index = len(points) + len(new_points)
        new_points.append(center)
        for i in range(len(ring)):
            j = (i + 1) % len(ring)
            new_tri.append([center_index, ring[i][0], ring[j][0]])
            for name, (_, values) in corner_values.items():
                new_values[name].append([at_center[name], values[ring[i][1][0], ring[i][1][1]],
                                         values[ring[j][1][0], ring[j][1][1]]])
    n_keep, n_new = int(keep.sum()), len(new_tri)
    points_out = np.r_[points, np.array(new_points).reshape(-1, 3)]
    tri_out = np.r_[tri[keep], np.array(new_tri, dtype=np.int64).reshape(-1, 3)]
    body_out = np.r_[(face_material == BODY)[keep], np.ones(n_new, bool)]
    holes_left = len(_holes(tri_out, in_asset_frame(points_out), _weld(points_out), body_out, up))

    def put(target, attr, values):
        old = attr.Get()
        target.CreateAttribute(attr.GetName(), attr.GetTypeName()).Set(
            type(old).FromNumpy(np.ascontiguousarray(values, dtype=np.array(old).dtype)))

    layer = Usd.Stage.CreateNew(out)
    UsdGeom.SetStageUpAxis(layer, UsdGeom.GetStageUpAxis(stage))
    UsdGeom.SetStageMetersPerUnit(layer, UsdGeom.GetStageMetersPerUnit(stage))
    table = layer.DefinePrim("/Table", "Xform")
    layer.SetDefaultPrim(table)
    table.GetReferences().AddReference(source)
    layer.OverridePrim(f"/Table/{MESH.split('/')[0]}").SetInstanceable(False)  # an instance cannot take the changes
    over = layer.OverridePrim(f"/Table/{MESH}")
    put(over, mesh.GetPointsAttr(), points_out)
    put(over, mesh.GetFaceVertexCountsAttr(), np.full(len(tri_out), 3))
    put(over, mesh.GetFaceVertexIndicesAttr(), tri_out.reshape(-1))
    for name, (attr, values) in corner_values.items():
        dim = values.shape[2]
        added = np.array(new_values[name], dtype=np.float64).reshape(-1, dim)
        put(over, attr, np.r_[values[keep].reshape(-1, dim), added])
    new_index = np.cumsum(keep) - 1
    for s in subsets:
        old = np.array(s.GetIndicesAttr().Get())
        faces = new_index[old[keep[old]]]
        if material_of[s.GetPrim().GetName()] == BODY:
            faces = np.r_[faces, n_keep + np.arange(n_new)]
        put(layer.OverridePrim(f"/Table/{MESH}/{s.GetPrim().GetName()}"), s.GetIndicesAttr(), faces)
    used = points_out[np.unique(tri_out)]
    extent = prim.GetAttribute("extent")
    over.CreateAttribute("extent", extent.GetTypeName()).Set([tuple(used.min(0)), tuple(used.max(0))])
    layer.Save()
    return {"metal_parts": metal_parts, "metal_faces": int((~keep).sum()), "holes": len(holes), "hole_faces": n_new,
            "holes_left": holes_left}
