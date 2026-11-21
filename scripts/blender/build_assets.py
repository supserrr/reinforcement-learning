"""Build the 3D viewer's models in Blender and export them as glTF.

    blender -b --factory-startup -P scripts/blender/build_assets.py

Writes environment/web/models/: drone.glb, hub.glb, clinic.glb, houses.glb and
trees.glb. Everything is modelled in metres in the viewer's axes (see kit.py);
the textures are generated here, tile at their real size and are packed into
the files, so the viewer needs nothing else. Named nodes are what the viewer
moves or lights: the drone's propellers, lights and cargo pods, the launcher's
carriage, the windsocks and the charging dock's lights.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kit import Kit, export, fbm, group, hexc, image, material, reset, rgba, tint  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "environment", "web", "models")
DEG = math.pi / 180


# ================================================================== textures
def tex_plaster(name, color, seed, size=256):
    n = fbm(size, size, 4, 5, seed)
    stain = fbm(size, size, 2, 4, seed + 7)
    g = 0.92 + 0.12 * (n - 0.5) - 0.18 * np.clip(stain - 0.58, 0, 1)
    return image(name, rgba(tint(g, hexc(color))))


def tex_corrugated(name, color, seed, ribs=8, rust=0.0, horizontal=False):
    size = 256
    x = np.arange(size) / size
    prof = 0.74 + 0.26 * np.sin(x * 2 * np.pi * ribs) ** 2
    g = np.tile(prof, (size, 1))
    if horizontal:
        g = g.T
    g = g * (0.93 + 0.1 * (fbm(size, size, 8, 4, seed) - 0.5))
    rgb = tint(g, hexc(color))
    if rust:
        r = np.clip((fbm(size, size, 4, 5, seed + 3) - (1 - rust)) * 4, 0, 1)[..., None]
        rgb = rgb * (1 - r) + tint(g, hexc("#8a4a2b")) * r
    return image(name, rgba(rgb))


def tex_asphalt(seed):
    size = 512
    rng = np.random.default_rng(seed)
    g = 0.30 + 0.07 * (fbm(size, size, 8, 5, seed) - 0.5) + 0.06 * (rng.random((size, size)) - 0.5)
    g -= 0.06 * np.clip(fbm(size, size, 3, 4, seed + 1) - 0.6, 0, 1) * 3  # oil and tyre stains
    return image("asphalt", rgba(tint(g, (1.0, 0.99, 0.96))))


def tex_concrete(seed):
    size = 512
    rng = np.random.default_rng(seed)
    g = 0.66 + 0.09 * (fbm(size, size, 6, 5, seed) - 0.5) + 0.03 * (rng.random((size, size)) - 0.5)
    g[::128, :] -= 0.12  # expansion joints every tile
    g[:, ::128] -= 0.12
    return image("concrete", rgba(tint(g, (0.98, 0.97, 0.94))))


def tex_gravel(seed):
    size = 256
    rng = np.random.default_rng(seed)
    g = 0.55 + 0.1 * (fbm(size, size, 16, 4, seed) - 0.5) + 0.12 * (rng.random((size, size)) - 0.5)
    return image("gravel", rgba(tint(g, hexc("#b4a68f"))))


def tex_bark(name, color, seed, peel=False):
    n = fbm(64, 256, 4, 5, seed)               # wide rows, then stretched tall: vertical fissures
    n = np.repeat(n, 4, axis=0)
    g = 0.72 + 0.5 * (n - 0.5)
    rgb = tint(g, hexc(color))
    if peel:  # eucalyptus sheds in long strips: cream patches over grey-brown
        p = np.repeat(fbm(32, 256, 3, 4, seed + 5), 8, axis=0)
        m = np.clip((p - 0.55) * 6, 0, 1)[..., None]
        rgb = rgb * (1 - m) + tint(g * 1.15, hexc("#d8cfb8")) * m
    return image(name, rgba(rgb))


def tex_leaves(name, palette, seed, count=240, narrow=False, size=512):
    """A cluster of leaves with an alpha mask, for leaf cards."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / size
    rgb = np.zeros((size, size, 3))
    alpha = np.zeros((size, size))
    cols = [np.array(hexc(c)) for c in palette]
    for _ in range(count):
        a = rng.random() * 2 * np.pi
        r = np.sqrt(rng.random()) * 0.4
        cx, cy = 0.5 + np.cos(a) * r, 0.5 + np.sin(a) * r
        ang = rng.random() * np.pi
        L = (0.07 if narrow else 0.055) * (0.7 + 0.6 * rng.random())
        W = L * (0.18 if narrow else 0.45)
        x0, x1 = int(max((cx - L) * size, 0)), int(min((cx + L) * size, size))
        y0, y1 = int(max((cy - L) * size, 0)), int(min((cy + L) * size, size))
        if x1 <= x0 or y1 <= y0:
            continue
        dx, dy = xx[y0:y1, x0:x1] - cx, yy[y0:y1, x0:x1] - cy
        u = dx * np.cos(ang) + dy * np.sin(ang)
        v = -dx * np.sin(ang) + dy * np.cos(ang)
        inside = (u / L) ** 2 + (v / W) ** 2 < 1
        col = cols[rng.integers(len(cols))] * (0.75 + 0.35 * rng.random())
        light = 1 + 0.25 * (v / W)  # light falls across the blade
        view = rgb[y0:y1, x0:x1]   # one leaf over another: the newest wins
        view[inside] = col[None, :] * light[inside][:, None]
        view[inside & (np.abs(v) < W * 0.08)] *= 1.18  # the midrib
        alpha[y0:y1, x0:x1][inside] = 1
    return image(name, rgba(np.clip(rgb, 0, 1), alpha))


def tex_banana(seed):
    rng = np.random.default_rng(seed)
    h, w = 1024, 256
    v, u = np.mgrid[0:h, 0:w].astype(float)
    v, u = v / h, u / w
    half = 0.48 * np.sin(np.pi * np.clip(v * 1.02, 0, 1)) ** 0.6  # the blade's half-width along its length
    d = np.abs(u - 0.5)
    alpha = (d < half).astype(float)
    # tears from the edge toward the midrib, as wind shreds banana leaves
    for _ in range(26):
        row = rng.integers(60, h - 30)
        depth = rng.uniform(0.15, 0.42)
        side = rng.choice([-1, 1])
        band = (np.abs(v * h - row) < 1.2) & (np.sign(u - 0.5) == side) & (d > half * (1 - depth))
        alpha[band] = 0
    g = 0.78 + 0.25 * (fbm(h // 4, w, 4, 4, seed) - 0.5).repeat(4, axis=0)
    veins = 0.93 + 0.07 * np.sin(v * h * 0.9) ** 2
    rgb = tint(g * veins, hexc("#6f9a3a"))
    rgb[d < 0.025] = np.array(hexc("#c9cf8a"))  # pale midrib
    edge = np.clip((d / np.maximum(half, 1e-3) - 0.85) * 6, 0, 1)[..., None]  # dry brown edges
    rgb = rgb * (1 - edge * 0.6) + np.array(hexc("#8a7a46")) * edge * 0.6
    return image("banana_leaf", rgba(rgb, alpha))


def tex_solar():
    size = 256
    g = np.full((size, size, 3), hexc("#1a2a44"))
    g[:: size // 6, :] = hexc("#c7ccd2")
    g[:, :: size // 10] = hexc("#c7ccd2")
    g *= (0.9 + 0.2 * fbm(size, size, 4, 3, 3))[..., None]
    return image("solar", rgba(np.clip(g, 0, 1)))


def tex_fence():
    size = 128
    yy, xx = np.mgrid[0:size, 0:size] / size * 8
    wire = (np.abs(((xx + yy) % 1) - 0.5) > 0.44) | (np.abs(((xx - yy) % 1) - 0.5) > 0.44)
    return image("fence", rgba(np.full((size, size, 3), hexc("#8d9196")), wire.astype(float)))


def tex_stripes(name, a, b, n=5, diagonal=False):
    size = 128
    yy, xx = np.mgrid[0:size, 0:size] / size
    k = ((xx + yy) if diagonal else xx) * n
    m = (np.floor(k) % 2 == 0)[..., None]
    return image(name, rgba(np.where(m, hexc(a), hexc(b)) * np.ones((size, size, 3))))


# ================================================================== the drone
def airfoil(chord, thick, camber=0.02, n=18):
    pts = []
    for k in range(2 * n + 1):
        up = k <= n
        t = 1 - k / n if up else (k - n) / n
        x = (1 - math.cos(t * math.pi)) / 2
        yt = 5 * thick * (0.2969 * math.sqrt(x) - 0.126 * x - 0.3516 * x * x + 0.2843 * x**3 - 0.1036 * x**4)
        yc = camber * ((2 * 0.4 * x - x * x) / 0.16 if x < 0.4 else (1 - 0.8 + 2 * 0.4 * x - x * x) / 0.36)
        pts.append((-x * chord, (yc + yt if up else yc - yt) * chord))
    pts.pop()
    return pts


def surface(k, mat, span, root, tip, thick, axis, normal, origin, sweep=0.0, steps=10, camber=0.015):
    """A tapered, swept lifting surface lofted from airfoil sections along `axis`."""
    stations = []
    for i in range(steps + 1):
        t = i / steps
        rnd = 0.55 if i == steps else 1
        chord = (root + (tip - root) * t) * rnd
        stations.append([(u - sweep * t - (1 - rnd) * tip * 0.2, v * (0.6 if i == steps else 1))
                         for u, v in airfoil(chord, thick, camber)])
    m = len(stations[0])
    verts = []
    for i, loop in enumerate(stations):
        for u, v in loop:
            verts.append(tuple(origin[j] + axis[j] * span * i / steps + normal[j] * v + (u if j == 2 else 0) for j in range(3)))
    faces = []
    for i in range(steps):
        for j in range(m):
            a, b = i * m + j, i * m + (j + 1) % m
            faces.append([a, b, b + m, a + m])
    faces.append(list(range(m))[::-1])
    faces.append([steps * m + j for j in range(m)])
    k.mesh(mat, verts, faces, smooth=True)


def lathe(k, mat, profile, seg=40, squash=0.92, x=0, y=0, z=0):
    """Revolve (radius, z) pairs around the viewer z axis."""
    verts, faces = [], []
    for r, zz in profile:
        for s in range(seg):
            a = 2 * math.pi * s / seg
            verts.append((x + math.cos(a) * r, y + math.sin(a) * r * squash, z + zz))
    n = len(profile)
    for i in range(n - 1):
        for s in range(seg):
            a, b = i * seg + s, i * seg + (s + 1) % seg
            faces.append([a, b, b + seg, a + seg])
    k.mesh(mat, verts, faces, smooth=True)


def build_drone():
    reset()
    body = material("drone_body", "#eceeeb", 0.3, 0.0)
    red = material("drone_red", "#c8241c", 0.35)
    grey = material("drone_grey", "#2a2d30", 0.45, 0.4)
    nacm = material("drone_nacelle", "#d9dcdc", 0.35, 0.1)
    glass = material("drone_glass", "#0e1316", 0.05, 0.1)
    k = Kit()
    L, R = 1.5, 0.2
    prof = []
    for i in range(41):
        t = i / 40
        prof.append((max(R * (t**1.05 * (1 - t) ** 0.5) / 0.372, 1e-4), -L / 2 + t * L))
    lathe(k, body, prof)
    k.sphere(glass, 0.075, 0, 0.14, 0.5, sx=1, sy=0.6, sz=1.5)            # sensor dome
    k.sphere(red, 0.03, 0, -0.17, 0.14, sx=1, sy=1, sz=3)                  # release under the belly
    k.box(red, 0.006, 0.08, 0.7, 0.2 * 0.92, 0.0, 0.05)                    # livery stripe, both flanks
    k.box(red, 0.006, 0.08, 0.7, -0.2 * 0.92, 0.0, 0.05)
    wingY, wingZ, dih = 0.18, 0.15, 3.5 * DEG
    for side in (1, -1):
        surface(k, body, 1.65, 0.32, 0.17, 0.13, (side * math.cos(dih), math.sin(dih), 0), (0, 1, 0), (0, wingY, wingZ), sweep=0.04, steps=14)
        tx = side * 1.6
        k.box(red, 0.12, 0.014, 0.16, tx, wingY + math.sin(dih) * 1.6 + 0.012, wingZ - 0.1)
        # winglet
        surface(k, body, 0.12, 0.14, 0.08, 0.1, (side * 0.15, 1, 0), (1, 0, 0), (side * 1.64, wingY + math.sin(dih) * 1.64, wingZ - 0.02), sweep=0.05, steps=3, camber=0)
    boomLen = 1.2
    k.rod(body, (0, 0.06, -0.55), (0, 0.06, -0.55 - boomLen), 0.019, 12)
    tailZ = -0.55 - boomLen
    surface(k, body, 0.38, 0.27, 0.15, 0.1, (0, 1, 0), (1, 0, 0), (0, 0.06, tailZ + 0.3), sweep=0.12, steps=6, camber=0)
    k.box(red, 0.02, 0.06, 0.12, 0, 0.06 + 0.37, tailZ + 0.12)
    for side in (1, -1):
        surface(k, body, 0.32, 0.18, 0.1, 0.1, (side, 0, 0), (0, 1, 0), (0, 0.06, tailZ + 0.2), sweep=0.05, steps=5, camber=0)
    k.rod(grey, (0, 0.06, tailZ + 0.06), (0, -0.07, tailZ + 0.03), 0.008, 6)  # tail skid
    surface(k, body, 0.3, 0.2, 0.16, 0.14, (0, 1, 0), (1, 0, 0), (0, wingY + 0.03, wingZ + 0.03), sweep=0.02, steps=4, camber=0)
    nacY, nacZ = wingY + 0.33, wingZ - 0.05
    lathe(k, nacm, [(1e-4, -0.24), (0.03, -0.22), (0.045, -0.16), (0.047, 0.0), (0.045, 0.16), (0.03, 0.22), (1e-4, 0.24)], seg=24, squash=1, y=nacY, z=nacZ)
    k.build("drone_airframe")

    def prop(name, z, d):
        pk = Kit()
        pk.cyl(grey, 0.032, 0.004, 0.07, 0, nacY, z + d * 0.03, rx=d * math.pi / 2, seg=16)  # spinner
        for a in (0, math.pi):
            verts, faces = [], []
            st = 8
            for i in range(st + 1):  # a tapered, twisted blade from the hub out
                t = i / st
                rad = 0.03 + t * 0.25
                chord = 0.045 - 0.025 * t
                tw = d * (28 - 18 * t) * DEG
                for cc in (-chord / 2, chord / 2):
                    px, py, pz = rad, cc * math.cos(tw), cc * math.sin(tw)
                    verts.append((px * math.cos(a) - py * math.sin(a), nacY + px * math.sin(a) + py * math.cos(a), z + pz))
            for i in range(st):
                faces.append([2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1])
            pk.mesh(grey, verts, faces)
        return group(name, pk.build(name), pivot=(0, nacY, z))

    prop("prop_front", nacZ + 0.25, 1)
    prop("prop_rear", nacZ - 0.25, -1)
    for i, x in enumerate((-0.065, 0.065)):
        ck = Kit()
        ck.sphere(red, 0.06, x, -0.16, -0.06, sx=1.1, sy=0.7, sz=1.9)
        group(f"cargo_{i}", ck.build(f"cargo_{i}"), pivot=(x, -0.16, -0.06))
    tipY = wingY + math.sin(dih) * 1.65
    for name, color, pos in (("light_port", "#ff2a20", (1.67, tipY, wingZ - 0.06)), ("light_starboard", "#20ff70", (-1.67, tipY, wingZ - 0.06)),
                             ("strobe", "#ffffff", (0, 0.45, tailZ + 0.13))):
        lk = Kit()
        lk.sphere(material(name, color, 0.2, emit=color, strength=30), 0.025, *pos, u=10, v=6)
        group(name, lk.build(name), pivot=pos)
    export(os.path.join(OUT, "drone.glb"))


# ================================================================== the hub
def arch(k, mat, x0, x1, cz, y0, r, seg=24):
    """An arched hangar roof along x: a half cylinder over the walls, with its end walls."""
    verts, faces = [], []
    for x in (x0, x1):
        for s in range(seg + 1):
            a = math.pi * s / seg
            verts.append((x, y0 + math.sin(a) * r, cz + math.cos(a) * r))
    n = seg + 1
    for s in range(seg):
        faces.append([s, s + 1, n + s + 1, n + s])
    k.mesh(mat, verts, faces, smooth=True)
    for x, flip in ((x0, True), (x1, False)):  # the half-disc ends
        ring = [(x, y0 + math.sin(math.pi * s / seg) * r, cz + math.cos(math.pi * s / seg) * r) for s in range(seg + 1)]
        k.mesh(mat, ring, [list(range(seg + 1))[::-1] if flip else list(range(seg + 1))])


def windows(k, frame, glass, x0, x1, y, z, h=1.2, w=1.4, every=3.0, normal=1, along="x"):
    """A row of framed windows on a wall facing +/- the given axis."""
    n = max(1, int((x1 - x0) // every))
    for i in range(n):
        c = x0 + (i + 0.5) * (x1 - x0) / n
        if along == "x":
            k.box(frame, w + 0.16, h + 0.16, 0.06, c, y, z + normal * 0.03)
            k.box(glass, w, h, 0.07, c, y, z + normal * 0.05)
        else:
            k.box(frame, 0.06, h + 0.16, w + 0.16, z + normal * 0.03, y, c)
            k.box(glass, 0.07, h, w, z + normal * 0.05, y, c)


def build_hub():
    reset()
    asphalt = material("asphalt", tex=tex_asphalt(1), rough=0.95, tile=6)
    concrete = material("concrete", tex=tex_concrete(2), rough=0.9, tile=4)
    yellowp = material("paint_yellow", "#e0b12a", 0.6)
    whitep = material("paint_white", "#efede6", 0.6)
    hang = material("hangar_metal", tex=tex_corrugated("corr_hangar", "#9aa4ab", 3, ribs=10, rust=0.08), rough=0.45, metal=0.5, tile=1.5)
    door = material("hangar_door", tex=tex_corrugated("corr_door", "#5d646b", 4, ribs=16, horizontal=True), rough=0.5, metal=0.4, tile=2)
    plaster = material("office_walls", tex=tex_plaster("plaster_office", "#e7e4dc", 5), rough=0.9, tile=3)
    roofc = material("office_roof", tex=tex_concrete(6), rough=0.9, tile=4)
    frame = material("window_frame", "#e9e9e6", 0.5, 0.2)
    glass = material("window_glass", "#18232b", 0.05, 0.1)
    cont_w = material("container_white", tex=tex_corrugated("corr_cw", "#dcdfdc", 7, ribs=12, rust=0.05), rough=0.6, metal=0.3, tile=1.2)
    cont_o = material("container_orange", tex=tex_corrugated("corr_co", "#c9622e", 8, ribs=12, rust=0.12), rough=0.6, metal=0.3, tile=1.2)
    steel = material("steel", "#b7bcc1", 0.4, 0.75)
    dark = material("dark_steel", "#2a2e32", 0.55, 0.5)
    rubber = material("rubber", "#1b1c1d", 0.9)
    hazard = material("hazard", tex=tex_stripes("hazard", "#e0b12a", "#1d1d1d", 6, diagonal=True), rough=0.6, tile=1)
    cabinet = material("cabinet", "#e8e8e4", 0.45, 0.2)
    panel = material("solar_panel", tex=tex_solar(), rough=0.25, metal=0.4, tile=1.6)
    fence = material("alpha_fence", tex=tex_fence(), rough=0.6, metal=0.6, alpha=True, tile=2)
    red = material("red_light", "#ff3020", 0.3, emit="#ff3020", strength=20)

    k = Kit()
    k.box(asphalt, 120, 0.4, 100, 0, 0.2, 0)
    k.box(concrete, 14, 0.04, 10, -30, 0.42, -16)
    k.box(concrete, 8, 0.04, 22, 26, 0.42, -1)
    for z in range(-6, 24, 3):  # dashed taxi line from the dock to the launcher
        k.box(yellowp, 0.25, 0.02, 1.6, 26 - 7.5, 0.43, z)
    for x in range(-24, 20, 3):
        k.box(yellowp, 1.6, 0.02, 0.25, x, 0.43, -10)
    for bz in (8, 14, 20):  # the recovery lane under the wire
        k.box(whitep, 22, 0.02, 0.4, -32, 0.42, bz)
    for bx in (-43, -21):
        k.box(whitep, 0.4, 0.02, 22, bx, 0.42, 14)
    # hangar: corrugated walls, an arched roof, a roller door and a band of windows
    k.box(hang, 34, 6, 18, -30, 3.4, -38)
    arch(k, hang, -47, -13, -38, 6.4, 9.0)
    k.box(door, 15, 7, 0.25, -30, 3.9, -28.9)
    windows(k, frame, glass, -46, -14, 5.0, -47.0, h=0.8, w=2.2, every=4, normal=-1)
    # offices: plaster, a flat roof with a parapet, windows, a door and air conditioners
    k.box(plaster, 18, 5.5, 10, 6, 3.15, -40)
    k.box(roofc, 18.4, 0.3, 10.4, 6, 6.05, -40)
    for px, pz, pw, pd in ((6, -34.9, 18.4, 0.25), (6, -45.1, 18.4, 0.25), (-3.1, -40, 0.25, 10.4), (15.1, -40, 0.25, 10.4)):
        k.box(plaster, pw, 0.6, pd, px, 6.5, pz)
    windows(k, frame, glass, -2, 14, 3.6, -35.0, normal=1)
    k.box(glass, 1.4, 2.3, 0.08, 12, 1.55, -34.95)
    for ax in (2, 9):
        k.box(cabinet, 1.4, 0.9, 1.0, ax, 6.65, -42)
    # containers with door bars
    for cx, cz, L, m in ((32, -38, 12.2, cont_w), (32, -34.5, 12.2, cont_o), (29, -31, 6.1, cont_w)):
        k.box(m, L, 2.6, 2.44, cx, 1.7, cz)
        for b in (-0.6, -0.2, 0.2, 0.6):
            k.rod(steel, (cx + L / 2 + 0.03, 0.6, cz + b), (cx + L / 2 + 0.03, 2.8, cz + b), 0.025, 6)
    # charging dock: plinth, cradle arms with rubber pads, bollards, the charger cabinet and its cable
    k.box(dark, 5, 0.6, 3.6, -30, 0.7, -16)
    for ax in (-31.6, -28.4):
        k.box(steel, 0.3, 0.45, 2.6, ax, 1.22, -16)
        k.box(rubber, 0.34, 0.06, 2.4, ax, 1.47, -16)
    for bx, bz in ((-33, -18.5), (-27, -18.5), (-33, -13.5), (-27, -13.5)):
        k.cyl(yellowp, 0.12, 0.12, 1.0, bx, 0.9, bz, seg=10)
    k.box(cabinet, 1.3, 2.1, 0.8, -25.5, 1.45, -19.5)
    k.box(dark, 1.0, 0.25, 0.06, -25.5, 1.25, -19.08)
    k.rod(rubber, (-25.5, 0.6, -19.1), (-27.0, 0.45, -17.9), 0.04, 6)
    k.rod(rubber, (-27.0, 0.45, -17.9), (-29.2, 1.05, -16.8), 0.04, 6)
    for x0, z0, x1, z1 in ((-35, -20, -25, -20), (-35, -12, -25, -12), (-35, -20, -35, -12), (-25, -20, -25, -12)):
        k.box(yellowp, abs(x1 - x0) + 0.15, 0.02, abs(z1 - z0) + 0.15, (x0 + x1) / 2, 0.44, (z0 + z1) / 2)
    # catapult launcher: base, twin rails on two A-frames, the actuator, end stop and control box
    LX, up12, RL = 26.0, 12 * DEG, 15.0
    r0y, r0z = 1.4, -8.0
    r1y, r1z = r0y + math.sin(up12) * RL, r0z + math.cos(up12) * RL
    k.box(dark, 4, 1, 5, LX, 0.9, -9)
    for off in (-0.22, 0.22):
        k.box(steel, 0.1, 0.16, RL, LX + off, (r0y + r1y) / 2, (r0z + r1z) / 2, rx=-up12)
    k.box(dark, 0.5, 0.12, RL, LX, (r0y + r1y) / 2 - 0.15, (r0z + r1z) / 2, rx=-up12)
    k.cyl(steel, 0.11, 0.11, RL * 0.8, LX, (r0y + r1y) / 2 - 0.38, (r0z + r1z) / 2 - 0.8, rx=math.pi / 2 - up12, seg=14)
    for t in (0.45, 0.9):
        hy, hz = r0y + math.sin(up12) * RL * t, r0z + math.cos(up12) * RL * t
        for side in (-1, 1):
            k.rod(steel, (LX + side * 1.3, 0.4, hz), (LX + side * 0.25, hy - 0.2, hz), 0.06, 8)
        k.rod(steel, (LX - 0.9, hy * 0.45, hz), (LX + 0.9, hy * 0.45, hz), 0.04, 6)
    k.box(hazard, 0.9, 0.5, 0.5, LX, r1y + 0.1, r1z)
    k.box(cabinet, 0.8, 1.2, 0.5, LX + 2.6, 1.0, -9.5)
    # recovery: tapered masts with guy wires, warning lights, and the arresting wire
    WX, WZ = -32.0, 24.0
    for mx in (WX - 12, WX + 12):
        k.cyl(steel, 0.32, 0.18, 15, mx, 7.9, WZ, seg=12)
        k.box(yellowp, 0.8, 0.5, 0.8, mx, 15.5, WZ)
        k.sphere(red, 0.15, mx, 15.9, WZ, u=8, v=6)
        for gx, gz in ((-6, 0), (6, 0), (0, -6), (0, 6)):
            if (mx < WX and gx > 0) or (mx > WX and gx < 0):
                continue
            k.rod(dark, (mx, 12.5, WZ), (mx + gx, 0.45, WZ + gz), 0.012, 4)
    # solar array on frames
    for r in range(3):
        for c in range(4):
            x, z = -8 + c * 10.2, -56 - r * 6
            k.box(panel, 9.6, 0.08, 3.4, x, 1.4, z, rx=-0.26)
            for lx in (-4.2, 4.2):
                k.rod(steel, (x + lx, 0.4, z - 1.2), (x + lx, 1.0, z - 1.2), 0.04, 6)
                k.rod(steel, (x + lx, 0.4, z + 1.2), (x + lx, 1.8, z + 1.2), 0.04, 6)
    # perimeter fence: posts and chain-link
    for (fx0, fz0, fx1, fz1) in ((-62, -62, 62, -62), (62, -62, 62, 52), (62, 52, 4, 52), (-4, 52, -62, 52), (-62, 52, -62, -62)):
        L = math.hypot(fx1 - fx0, fz1 - fz0)
        n = max(1, int(L // 4))
        for i in range(n + 1):
            t = i / n
            k.rod(steel, (fx0 + (fx1 - fx0) * t, 0.3, fz0 + (fz1 - fz0) * t), (fx0 + (fx1 - fx0) * t, 2.4, fz0 + (fz1 - fz0) * t), 0.04, 6)
        k.box(fence, abs(fx1 - fx0) + 0.02, 2.0, abs(fz1 - fz0) + 0.02, (fx0 + fx1) / 2, 1.4, (fz0 + fz1) / 2)
    # light poles and the windsock pole
    for lx, lz in ((-58, -58), (58, -58), (-58, 48), (58, 48)):
        k.cyl(dark, 0.1, 0.07, 9, lx, 4.9, lz, seg=8)
        k.box(dark, 1.2, 0.2, 0.5, lx, 9.4, lz)
    k.cyl(steel, 0.07, 0.07, 6, 52, 3.4, 42, seg=8)
    k.build("hub")

    # things the viewer moves or lights
    ck = Kit()
    ck.box(dark, 1.3, 0.3, 1.1, LX, r0y + 0.3, r0z, rx=-up12)
    ck.box(hazard, 1.32, 0.08, 0.3, LX, r0y + 0.48, r0z - 0.4, rx=-up12)
    group("carriage", ck.build("carriage"), pivot=(LX, r0y + 0.3, r0z))
    sk = Kit()
    sk.cyl(material("sock", tex=tex_stripes("sock", "#e2622a", "#f2f0ea", 5), rough=0.8, tile=0.55), 0.45, 0.18, 2.6, 52 + 1.4, 5.8, 42, rz=-math.pi / 2, seg=14, caps=False)
    group("windsock", sk.build("windsock"), pivot=(52, 5.8, 42))
    lk = Kit()
    lk.box(material("charge_light", "#1d3326", 0.4, emit="#46ff8a", strength=1), 0.9, 0.5, 0.05, -25.5, 1.9, -19.08)
    group("charge_light", lk.build("charge_light"), pivot=(-25.5, 1.9, -19.08))
    dk = Kit()
    dk.box(material("dock_strip", "#1d3326", 0.4, emit="#46ff8a", strength=1), 4.6, 0.08, 0.08, -30, 1.02, -14.15)
    group("dock_strip", dk.build("dock_strip"), pivot=(-30, 1.02, -14.15))
    wk = Kit()  # the arresting wire: the viewer pulls it into a V when the hook takes it
    wk.rod(rubber, (WX - 12, 12, WZ), (WX + 12, 12, WZ), 0.035, 6)
    group("wire", wk.build("wire"), pivot=(WX, 12, WZ))
    export(os.path.join(OUT, "hub.glb"))


# ================================================================== a health centre
def building(k, walls, plinth, roof, frame, glass, door, w, h, d, x, z, ridge_x=True, rh=2.0, veranda=None):
    k.box(walls, w, h, d, x, h / 2, z)
    k.box(plinth, w + 0.06, 0.6, d + 0.06, x, 0.3, z)
    if ridge_x:
        k.gable(roof, d, rh, w, x, h, z, ry=math.pi / 2, eave=0.6)
        k.gable_ends(walls, d, rh, w, x, h, z, ry=math.pi / 2)
        windows(k, frame, glass, x - w / 2 + 1, x + w / 2 - 1, h * 0.55, z + d / 2, normal=1)
        windows(k, frame, glass, x - w / 2 + 1, x + w / 2 - 1, h * 0.55, z - d / 2, normal=-1)
        k.box(door, 1.1, 2.2, 0.08, x + w * 0.25, 1.1, z + d / 2 + 0.05)
    else:
        k.gable(roof, w, rh, d, x, h, z, eave=0.6)
        k.gable_ends(walls, w, rh, d, x, h, z)
        windows(k, frame, glass, z - d / 2 + 1, z + d / 2 - 1, h * 0.55, x + w / 2, normal=1, along="z")
        windows(k, frame, glass, z - d / 2 + 1, z + d / 2 - 1, h * 0.55, x - w / 2, normal=-1, along="z")
    if veranda:  # a lean-to on posts along the front
        vd = veranda
        k.box(roof, w, 0.1, vd, x, h - 0.5, z + d / 2 + vd / 2, rx=0.12)
        for px in np.linspace(x - w / 2 + 0.4, x + w / 2 - 0.4, max(2, int(w // 4))):
            k.cyl(frame, 0.09, 0.09, h - 0.5, float(px), (h - 0.5) / 2, z + d / 2 + vd - 0.3, seg=8)
        k.box(plinth, w, 0.3, vd, x, 0.15, z + d / 2 + vd / 2)


def build_clinic():
    reset()
    yard = material("gravel", tex=tex_gravel(11), rough=0.95, tile=3)
    walls = material("clinic_walls", tex=tex_plaster("plaster_clinic", "#ece4d2", 12), rough=0.9, tile=3)
    plinth = material("clinic_plinth", tex=tex_plaster("plaster_plinth", "#7d8a6e", 13), rough=0.9, tile=3)
    roof = material("clinic_roof", tex=tex_corrugated("corr_clinic", "#9aa1a6", 14, ribs=10, rust=0.1), rough=0.5, metal=0.4, tile=1.2)
    frame = material("window_frame", "#eeeeea", 0.5, 0.2)
    glass = material("window_glass", "#18232b", 0.05, 0.1)
    door = material("door_paint", "#2e5f7a", 0.5)
    wallM = material("compound_wall", tex=tex_plaster("plaster_wall", "#d3c7ae", 15), rough=0.9, tile=3)
    tank = material("water_tank", "#2f5674", 0.5)
    steel = material("steel", "#9ea4a9", 0.45, 0.6)
    white = material("paint_white", "#f2f0ea", 0.6)
    grass = material("mown_grass", tex=image("mown", rgba(tint(0.85 + 0.2 * (fbm(128, 128, 8, 4, 16) - 0.5), (0.47, 0.6, 0.3)))), rough=0.95, tile=4)
    orange = material("cone_orange", "#e0662a", 0.6)
    k = Kit()
    k.box(yard, 64, 0.12, 44, 0, 0.06, -10)
    for wx, wz, ww, wd in ((0, -32, 64, 0.3), (-32, -10, 0.3, 44), (32, -10, 0.3, 44), (-18, 12, 28, 0.3), (18, 12, 28, 0.3)):
        k.box(wallM, ww, 1.8, wd, wx, 0.9, wz)
        k.box(wallM, ww + 0.15, 0.1, wd + 0.15, wx, 1.85, wz)
    for gx in (-4.2, 4.2):
        k.box(wallM, 0.6, 2.4, 0.6, gx, 1.2, 12)
    k.box(steel, 3.8, 1.6, 0.05, -2.1, 1.0, 13.2, ry=-0.9)  # the gate, standing open
    building(k, walls, plinth, roof, frame, glass, door, 30, 4.2, 10, -4, -22, veranda=2.4)
    building(k, walls, plinth, roof, frame, glass, door, 10, 4.2, 16, 18, -12, ridge_x=False)
    building(k, walls, plinth, roof, frame, glass, door, 16, 3.8, 8, -14, 2, veranda=1.8)
    building(k, walls, plinth, roof, frame, glass, door, 6, 3.0, 6, 22, 4, rh=1.3)
    for px, pz in ((24, -2), (27, -2), (24, 1), (27, 1)):
        k.rod(steel, (px, 0.1, pz), (px, 4.0, pz), 0.07, 6)
    k.box(steel, 3.4, 0.12, 3.4, 25.5, 4.0, -0.5)
    k.cyl(tank, 1.5, 1.5, 2.5, 25.5, 5.3, -0.5, seg=20)
    # sign at the gate: two posts and a board
    for sx in (5.2, 8.0):
        k.rod(steel, (sx, 0.1, 13.5), (sx, 2.6, 13.5), 0.05, 6)
    k.box(white, 3.2, 1.0, 0.06, 6.6, 2.1, 13.5)
    k.box(material("sign_green", "#2f7a4a", 0.6), 3.2, 0.25, 0.07, 6.6, 1.7, 13.52)
    # the drop zone: a mown disc, a painted ring and cross, and marker cones
    k.cyl(grass, 12, 12, 0.06, 0, 0.03, 46, seg=40, smooth=False)
    ring_v, ring_f = [], []
    for s in range(48):
        a = 2 * math.pi * s / 48
        ring_v += [(math.cos(a) * 8.4, 0.08, 46 + math.sin(a) * 8.4), (math.cos(a) * 9.2, 0.08, 46 + math.sin(a) * 9.2)]
    for s in range(48):
        a0, a1 = 2 * s, 2 * ((s + 1) % 48)
        ring_f.append([a0, a0 + 1, a1 + 1, a1])
    k.mesh(white, ring_v, ring_f)
    k.box(white, 10, 0.03, 0.7, 0, 0.075, 46)
    k.box(white, 0.7, 0.03, 10, 0, 0.075, 46)
    for a in range(4):
        ang = a * math.pi / 2 + math.pi / 4
        k.cyl(orange, 0.18, 0.02, 0.6, math.cos(ang) * 11, 0.3, 46 + math.sin(ang) * 11, seg=10)
    k.cyl(steel, 0.06, 0.06, 6, 13, 3, 46, seg=8)
    k.build("clinic")
    sk = Kit()
    sk.cyl(material("sock", tex=tex_stripes("sock", "#e2622a", "#f2f0ea", 5), rough=0.8, tile=0.55), 0.45, 0.18, 2.6, 13 + 1.4, 5.8, 46, rz=-math.pi / 2, seg=14, caps=False)
    group("windsock", sk.build("windsock"), pivot=(13, 5.8, 46))
    export(os.path.join(OUT, "clinic.glb"))


# ================================================================== houses
def build_houses():
    reset()
    walls = material("house_walls", tex=tex_plaster("plaster_house", "#f4f1ea", 21), rough=0.9, tile=2.5)
    roof = material("house_roof", tex=tex_corrugated("corr_house", "#e8e8e8", 22, ribs=9, rust=0.15), rough=0.5, metal=0.35, tile=1.2)
    trim = material("house_trim", "#2b2620", 0.7)

    def done(name, k):
        for o in k.build(name):
            role = o.data.materials[0].name.split("_")[-1]
            o.name = f"{name}_{role}"  # house_a_walls, house_a_roof, house_a_trim

    # a: the common rectangular house, gable roof, front door, two windows and a small porch
    k = Kit()
    k.box(walls, 8, 2.9, 6, 0, 1.45, 0)
    k.gable(roof, 6, 1.5, 8, 0, 2.9, 0, ry=math.pi / 2, eave=0.45)
    k.gable_ends(walls, 6, 1.5, 8, 0, 2.9, 0, ry=math.pi / 2)
    k.box(trim, 0.95, 2.0, 0.06, 0, 1.0, 3.03)
    for wx in (-2.4, 2.4):
        k.box(trim, 1.0, 1.0, 0.06, wx, 1.7, 3.03)
        k.box(trim, 1.0, 1.0, 0.06, wx, 1.7, -3.03)
    k.box(roof, 3.0, 0.06, 1.6, 0, 2.5, 3.8, rx=0.2)
    for px in (-1.3, 1.3):
        k.rod(trim, (px, 0, 4.4), (px, 2.4, 4.4), 0.05, 6)
    done("house_a", k)
    # b: an L-shaped house
    k = Kit()
    k.box(walls, 9, 3.0, 5, 0, 1.5, -1.5)
    k.gable(roof, 5, 1.5, 9, 0, 3.0, -1.5, ry=math.pi / 2, eave=0.45)
    k.gable_ends(walls, 5, 1.5, 9, 0, 3.0, -1.5, ry=math.pi / 2)
    k.box(walls, 4.5, 3.0, 5, 2.25, 1.5, 3.5)
    k.gable(roof, 4.5, 1.4, 5, 2.25, 3.0, 3.5, eave=0.45)
    k.gable_ends(walls, 4.5, 1.4, 5, 2.25, 3.0, 3.5)
    k.box(trim, 0.95, 2.0, 0.06, -2.5, 1.0, 1.03)
    for wx in (-3.6, -1.2):
        k.box(trim, 1.0, 1.0, 0.06, wx, 1.8, -4.03)
    k.box(trim, 0.06, 1.0, 1.0, 4.53, 1.8, 3.5)
    done("house_b", k)
    # c: a small house with a lean-to roof and a separate cooking hut
    k = Kit()
    k.box(walls, 6, 2.7, 5, 0, 1.35, 0)
    k.box(roof, 6.8, 0.08, 5.9, 0, 3.05, 0, rx=-0.12)
    k.box(trim, 0.9, 1.9, 0.06, 1.2, 0.95, 2.53)
    k.box(trim, 0.9, 0.9, 0.06, -1.5, 1.6, 2.53)
    k.box(walls, 2.4, 2.0, 2.4, -1.5, 1.0, -4.5)
    k.box(roof, 3.0, 0.08, 3.0, -1.5, 2.15, -4.5, rx=0.15)
    done("house_c", k)
    # d: a two-storey town building with a shop front, flat roof, parapet and balcony
    k = Kit()
    k.box(walls, 11, 6.6, 9, 0, 3.3, 0)
    k.box(roof, 11.3, 0.25, 9.3, 0, 6.7, 0)
    for px, pz, pw, pd in ((0, 4.55, 11.3, 0.2), (0, -4.55, 11.3, 0.2), (-5.55, 0, 0.2, 9.3), (5.55, 0, 0.2, 9.3)):
        k.box(walls, pw, 0.7, pd, px, 7.1, pz)
    for sx in (-3.5, 0, 3.5):
        k.box(trim, 2.6, 2.4, 0.06, sx, 1.3, 4.53)
        k.box(trim, 1.3, 1.3, 0.06, sx, 4.6, 4.53)
    k.box(walls, 11, 0.15, 1.4, 0, 3.4, 5.2)
    k.box(trim, 11, 0.9, 0.05, 0, 3.95, 5.9)
    done("house_d", k)
    export(os.path.join(OUT, "houses.glb"))


# ================================================================== trees
def cards(k, mat, centres, size_range, rng, normal_out=True):
    """Leaf cards: quads with the whole leaf texture, scattered and turned at random."""
    verts, faces, uvs = [], [], []
    for (cx, cy, cz) in centres:
        s = rng.uniform(*size_range) / 2
        a, b = rng.uniform(0, 2 * np.pi), rng.uniform(-0.9, 0.9)
        # two in-plane axes from a random orientation
        nx, ny, nz = np.cos(a) * np.cos(b), np.sin(b), np.sin(a) * np.cos(b)
        n = np.array([nx, ny, nz])
        t1 = np.cross(n, [0, 1, 0]) if abs(ny) < 0.95 else np.cross(n, [1, 0, 0])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        base = len(verts)
        for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            p = np.array([cx, cy, cz]) + (t1 * du + t2 * dv) * s
            verts.append(tuple(p))
        faces.append([base, base + 1, base + 2, base + 3])
        uvs.append([(0, 0), (1, 0), (1, 1), (0, 1)])
    k.mesh(mat, verts, faces, uvs=uvs)


def trunk(k, mat, pts, r0, r1, seg=8):
    for i in range(len(pts) - 1):
        t = i / (len(pts) - 1)
        k.rod(mat, pts[i], pts[i + 1], r0 + (r1 - r0) * t, seg)


def build_trees():
    reset()
    rng = np.random.default_rng(31)
    bark_euc = material("bark_euc", tex=tex_bark("bark_euc", "#8d8270", 41, peel=True), rough=0.85, tile=1.2)
    bark = material("bark", tex=tex_bark("bark_brown", "#5b4634", 42), rough=0.95, tile=1.0)
    leaf_euc = material("leaf_euc", tex=tex_leaves("leaves_euc", ["#5f7a63", "#4f6a55", "#748a6a", "#6a7f5a"], 43, count=420, narrow=True), rough=0.75, alpha=True, tile=0)
    leaf_broad = material("leaf_broad", tex=tex_leaves("leaves_broad", ["#3f6a2e", "#4d7a34", "#355c27", "#5b8a3c"], 44, count=260), rough=0.7, alpha=True, tile=0)
    leaf_cyp = material("leaf_cypress", tex=tex_leaves("leaves_cyp", ["#34502f", "#2c4629", "#3f5c36"], 45, count=500, narrow=True), rough=0.8, alpha=True, tile=0)
    leaf_ban = material("leaf_banana", tex=tex_banana(46), rough=0.6, alpha=True, tile=0)
    stem = material("banana_stem", "#7f8a45", 0.7)

    def done(name, k):
        for o in k.build(name):
            mname = o.data.materials[0].name
            o.name = f"{name}_{'leaves' if mname.startswith('leaf') else 'bark'}"

    # eucalyptus: a tall leaning trunk, high branches, an open crown of narrow leaves
    k = Kit()
    lean = rng.uniform(-0.04, 0.04, 2)
    spine = [(lean[0] * y * y / 18, y, lean[1] * y * y / 18) for y in np.linspace(0, 18, 7)]
    trunk(k, bark_euc, spine, 0.26, 0.08)
    tips = []
    for i in range(7):
        y = rng.uniform(9, 16)
        a = rng.uniform(0, 2 * np.pi)
        L = rng.uniform(2, 3.2)
        b0 = (lean[0] * y * y / 18, y, lean[1] * y * y / 18)
        b1 = (b0[0] + np.cos(a) * L, y + L * 0.8, b0[2] + np.sin(a) * L)
        k.rod(bark_euc, b0, b1, 0.06, 6)
        tips.append(b1)
    centres = []
    for _ in range(80):
        tp = tips[rng.integers(len(tips))] if rng.random() < 0.6 else (rng.uniform(-1, 1), rng.uniform(13, 19), rng.uniform(-1, 1))
        centres.append((tp[0] + rng.normal(0, 1.1), tp[1] + rng.normal(0, 1.0), tp[2] + rng.normal(0, 1.1)))
    cards(k, leaf_euc, centres, (1.6, 2.6), rng)
    done("euc", k)
    # broadleaf (avocado, mango): a short trunk, spreading limbs and a dense round crown
    k = Kit()
    trunk(k, bark, [(0, 0, 0), (0.1, 1.6, 0), (0.05, 3.0, 0.1)], 0.32, 0.22)
    limbs = []
    for i in range(5):
        a = i / 5 * 2 * np.pi + rng.uniform(-0.3, 0.3)
        end = (np.cos(a) * 2.4, rng.uniform(4.6, 6.2), np.sin(a) * 2.4)
        k.rod(bark, (0.05, 2.8, 0.1), end, 0.11, 6)
        limbs.append(end)
    centres = []
    for _ in range(150):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        r = rng.uniform(1.6, 3.6)
        centres.append((u[0] * r * 1.1, 6.0 + u[1] * r * 0.8, u[2] * r * 1.1))
    cards(k, leaf_broad, centres, (1.4, 2.4), rng)
    done("broad", k)
    # cypress: a narrow cone, used along compound walls and roads
    k = Kit()
    trunk(k, bark, [(0, 0, 0), (0, 9, 0)], 0.2, 0.06)
    centres = []
    for _ in range(140):
        y = rng.uniform(1.2, 10.5)
        r = (1 - (y - 1.2) / 9.5) * 1.8 + 0.2
        a = rng.uniform(0, 2 * np.pi)
        rr = r * np.sqrt(rng.random())
        centres.append((np.cos(a) * rr, y, np.sin(a) * rr))
    cards(k, leaf_cyp, centres, (0.9, 1.6), rng)
    done("cypress", k)
    # banana: a pseudo-stem and long arching blades, with a sucker beside it
    k = Kit()
    k.cyl(stem, 0.16, 0.12, 2.2, 0, 1.1, 0, seg=10)
    k.cyl(stem, 0.09, 0.07, 1.0, 0.5, 0.5, 0.3, seg=8)
    verts, faces, uvs = [], [], []
    for i in range(9):
        a = i / 9 * 2 * np.pi + rng.uniform(-0.25, 0.25)
        lift = rng.uniform(0.5, 1.0)
        L = rng.uniform(2.2, 3.0)
        base = len(verts)
        st = 8
        for j in range(st + 1):
            t = j / st
            out = t * L
            y = 2.1 + out * np.sin(lift) * (1 - t * 0.9) - t * t * 0.9
            for side in (-1, 1):
                w = 0.32 * side
                px = np.cos(a) * out - np.sin(a) * w
                pz = np.sin(a) * out + np.cos(a) * w
                verts.append((px, y - abs(w) * 0.25, pz))
        for j in range(st):
            faces.append([base + 2 * j, base + 2 * j + 1, base + 2 * j + 3, base + 2 * j + 2])
            uvs.append([(0, j / st), (1, j / st), (1, (j + 1) / st), (0, (j + 1) / st)])
    k.mesh(leaf_ban, verts, faces, uvs=uvs, smooth=True)
    done("banana", k)
    export(os.path.join(OUT, "trees.glb"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    only = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    for name, fn in (("drone", build_drone), ("hub", build_hub), ("clinic", build_clinic), ("houses", build_houses), ("trees", build_trees)):
        if not only or name in only:
            fn()
            print(f"built {name}.glb", flush=True)
