"""Small modelling kit for the build scripts: viewer-space transforms, primitive
parts merged per material, generated tiling textures and glTF export.

Everything is given in the viewer's axes and in metres: x right, y up, z
forward (three.js). Blender is Z-up, so positions map (x, y, z) -> (x, -z, y);
the glTF exporter maps them back. Rotations keep their angles: a viewer pitch
about x is a Blender rotation about x, a viewer yaw about y is a Blender
rotation about z.
"""

from __future__ import annotations

import math

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector


def V(x, y, z) -> Vector:
    return Vector((x, -z, y))


def M(x=0.0, y=0.0, z=0.0, rx=0.0, ry=0.0, rz=0.0, sx=1.0, sy=1.0, sz=1.0) -> Matrix:
    """Viewer-space transform: scale, then roll (rz), pitch (rx), yaw (ry), then move."""
    R = Matrix.Rotation(ry, 4, "Z") @ Matrix.Rotation(rx, 4, "X") @ Matrix.Rotation(-rz, 4, "Y")
    S = Matrix.Diagonal((sx, sz, sy, 1.0))
    return Matrix.Translation(V(x, y, z)) @ R @ S


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


# ------------------------------------------------------------------ textures
def _wrap_noise(h, w, cells, rng):
    """Tileable value noise: a random grid of `cells`, wrapped, bilinear."""
    g = rng.random((cells, cells))
    y = np.arange(h) * cells / h
    x = np.arange(w) * cells / w
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    ty, tx = (y - y0)[:, None], (x - x0)[None, :]
    ty, tx = ty * ty * (3 - 2 * ty), tx * tx * (3 - 2 * tx)
    y1, x1 = (y0 + 1) % cells, (x0 + 1) % cells
    a, b = g[y0][:, x0], g[y0][:, x1]
    c, d = g[y1][:, x0], g[y1][:, x1]
    return (a + (b - a) * tx) * (1 - ty) + (c + (d - c) * tx) * ty


def fbm(h, w, base=4, octaves=5, seed=0):
    rng = np.random.default_rng(seed)
    out, amp, tot = np.zeros((h, w)), 1.0, 0.0
    for o in range(octaves):
        out += _wrap_noise(h, w, base * 2**o, rng) * amp
        tot += amp
        amp *= 0.5
    return out / tot


def image(name: str, rgba: np.ndarray) -> bpy.types.Image:
    """rgba: H x W x 4 floats in 0..1, sRGB, row 0 at the top."""
    h, w = rgba.shape[:2]
    img = bpy.data.images.new(name, w, h, alpha=True)
    img.pixels.foreach_set(np.ascontiguousarray(np.flipud(np.clip(rgba, 0, 1))).astype(np.float32).ravel())
    img.pack()
    return img


def rgba(rgb: np.ndarray, alpha=None) -> np.ndarray:
    a = np.ones(rgb.shape[:2]) if alpha is None else alpha
    return np.dstack([rgb, a])


def tint(gray: np.ndarray, color) -> np.ndarray:
    c = np.array(color, dtype=float)
    return gray[..., None] * c[None, None, :]


def hexc(s: str):
    s = s.lstrip("#")
    return tuple(int(s[i:i + 2], 16) / 255 for i in (0, 2, 4))


def lin(c):
    return tuple(((v + 0.055) / 1.055) ** 2.4 if v > 0.04045 else v / 12.92 for v in c)


# ------------------------------------------------------------------ materials
def material(name, color="#ffffff", rough=0.8, metal=0.0, tex=None, alpha=False, emit=None, strength=1.0, tile=2.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    bsdf.inputs["Base Color"].default_value = (*lin(hexc(color)), 1.0)
    if tex is not None:
        node = nt.nodes.new("ShaderNodeTexImage")
        node.image = tex
        nt.links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
        if alpha:
            nt.links.new(node.outputs["Alpha"], bsdf.inputs["Alpha"])
    if emit:
        bsdf.inputs["Emission Color"].default_value = (*lin(hexc(emit)), 1.0)
        bsdf.inputs["Emission Strength"].default_value = strength
    m["tile"] = tile  # metres per texture repeat, for the box projection
    return m


# ------------------------------------------------------------------ the kit
class Kit:
    """Collects parts per material and builds one object per material."""

    def __init__(self):
        self.parts: dict[bpy.types.Material, bmesh.types.BMesh] = {}

    def bm(self, mat):
        if mat not in self.parts:
            self.parts[mat] = bmesh.new()
        return self.parts[mat]

    @staticmethod
    def _smooth(ret, flag=True):
        for v in ret["verts"]:
            for f in v.link_faces:
                f.smooth = flag

    def box(self, mat, w, h, d, x, y, z, rx=0, ry=0, rz=0):
        bmesh.ops.create_cube(self.bm(mat), size=1.0, matrix=M(x, y, z, rx, ry, rz, w, h, d))

    def cyl(self, mat, r1, r2, h, x, y, z, rx=0, ry=0, rz=0, seg=12, caps=True, smooth=True):
        """A cylinder or cone along the viewer's y axis, centred on (x, y, z)."""
        bm = self.bm(mat)
        ret = bmesh.ops.create_cone(bm, cap_ends=caps, cap_tris=False, segments=seg, radius1=r1, radius2=r2,
                                    depth=h, matrix=M(x, y, z, rx, ry, rz))
        if smooth:
            for v in ret["verts"]:
                for f in v.link_faces:
                    f.smooth = len(f.verts) == 4  # sides smooth, caps flat

    def sphere(self, mat, r, x, y, z, sx=1, sy=1, sz=1, u=16, v=10, smooth=True):
        ret = bmesh.ops.create_uvsphere(self.bm(mat), u_segments=u, v_segments=v, radius=r,
                                        matrix=M(x, y, z, sx=sx, sy=sy, sz=sz))
        self._smooth(ret, smooth)

    def rod(self, mat, a, b, r, seg=8):
        """A cylinder from viewer point a to b."""
        a, b = V(*a), V(*b)
        d = b - a
        L = d.length
        rot = d.to_track_quat("Z", "Y").to_matrix().to_4x4()
        bm = self.bm(mat)
        ret = bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=seg, radius1=r, radius2=r, depth=L,
                                    matrix=Matrix.Translation((a + b) / 2) @ rot)
        for v in ret["verts"]:
            for f in v.link_faces:
                f.smooth = len(f.verts) == 4

    def mesh(self, mat, verts, faces, smooth=False, uvs=None):
        """verts in viewer coordinates; faces as index lists; optional per-face-corner uvs."""
        bm = self.bm(mat)
        vs = [bm.verts.new(V(*p)) for p in verts]
        out = []
        for fi, f in enumerate(faces):
            try:
                face = bm.faces.new([vs[i] for i in f])
            except ValueError:
                continue
            face.smooth = smooth
            out.append((face, fi))
        if uvs is not None:
            layer = bm.loops.layers.uv.verify()
            for face, fi in out:
                for loop, uv in zip(face.loops, uvs[fi]):
                    loop[layer].uv = uv
        return vs

    def gable(self, mat, w, h, d, x, y, z, ry=0.0, eave=0.5, thick=0.12):
        """A pitched roof, ridge along the viewer's z (rotate with ry), w across, h tall, d long,
        with eaves overhanging by `eave` and a sheet thickness."""
        W, D = w / 2 + eave, d / 2 + eave
        pts = []
        for zz in (-D, D):
            pts += [(-W, -eave * h / (w / 2), zz), (0, h, zz), (W, -eave * h / (w / 2), zz)]
        top = [(px, py + thick, pz) for px, py, pz in pts]
        verts = pts + top
        faces = [[0, 1, 4, 3], [1, 2, 5, 4], [6, 9, 10, 7], [7, 10, 11, 8], [0, 3, 9, 6], [2, 8, 11, 5],
                 [0, 6, 7, 1], [1, 7, 8, 2], [3, 4, 10, 9], [4, 5, 11, 10]]
        c, s = math.cos(ry), math.sin(ry)
        verts = [(x + px * c + pz * s, y + py, z - px * s + pz * c) for px, py, pz in verts]
        self.mesh(mat, verts, faces)

    def gable_ends(self, mat, w, h, d, x, y, z, ry=0.0):
        """The triangular wall under each end of a gable roof."""
        c, s = math.cos(ry), math.sin(ry)
        for zz in (-d / 2, d / 2):
            tri = [(-w / 2, 0, zz), (w / 2, 0, zz), (0, h, zz)]
            self.mesh(mat, [(x + px * c + pz * s, y + py, z - px * s + pz * c) for px, py, pz in tri], [[0, 1, 2]])

    def build(self, name, smooth_all=False):
        objs = []
        for mat, bm in self.parts.items():
            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            if float(mat.get("tile", 2.0)) > 0:  # 0: the parts carry their own uvs (leaf cards)
                box_uv(bm, float(mat.get("tile", 2.0)))
            me = bpy.data.meshes.new(f"{name}_{mat.name}")
            bm.to_mesh(me)
            bm.free()
            me.materials.append(mat)
            if smooth_all:
                for p in me.polygons:
                    p.use_smooth = True
            obj = bpy.data.objects.new(f"{name}_{mat.name}" if len(self.parts) > 1 else name, me)
            bpy.context.scene.collection.objects.link(obj)
            objs.append(obj)
        self.parts = {}
        return objs


def box_uv(bm, tile):
    """Project uvs from each face's dominant axis, in metres / tile, so textures keep their real size."""
    layer = bm.loops.layers.uv.verify()
    for f in bm.faces:
        n = f.normal
        ax = max(range(3), key=lambda i: abs(n[i]))
        for loop in f.loops:
            co = loop.vert.co
            if ax == 2:
                uv = (co.x, co.y)
            elif ax == 0:
                uv = (co.y, co.z)
            else:
                uv = (co.x, co.z)
            loop[layer].uv = (uv[0] / tile, uv[1] / tile)


def group(name, objs, pivot=(0, 0, 0)):
    """Join objects into one, with its origin at a viewer-space pivot (for things that move)."""
    for o in objs:
        o.data.transform(Matrix.Translation(-V(*pivot)))
    if len(objs) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for o in objs:
            o.select_set(True)
        bpy.context.view_layer.objects.active = objs[0]
        bpy.ops.object.join()
    obj = objs[0]
    obj.name = name
    obj.location = V(*pivot)
    return obj


def export(path):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", use_selection=True, export_apply=True,
                              export_yup=True, export_cameras=False, export_lights=False)
