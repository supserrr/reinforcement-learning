# /// script
# requires-python = ">=3.10"
# dependencies = ["osmium>=3.7", "numpy", "pillow"]
# ///
"""Bake the real buildings and woodland around Muhanga from OpenStreetMap.

Writes environment/web/map/buildings.bin (every building footprint inside the
inner map as a rotated rectangle) and environment/web/map/woods.png (where
OpenStreetMap maps forest or woodland), and adds both to map.json. The viewer
draws these buildings instead of made-up villages, so the houses stand on the
roofs the satellite image shows.

    uv run scripts/bake_buildings.py     # the Rwanda extract is cached in .map-cache/osm/

Data (c) OpenStreetMap contributors, ODbL 1.0, via the Geofabrik Rwanda extract.

buildings.bin is little-endian, 8 bytes a building:
    uint16 x, z    metres east and south of the map's west/north edge (x + 32000)
    uint8  w, d    footprint length and width in half metres (w >= d)
    uint8  yaw     direction of the long side, 0..255 = 0..pi from east toward south
    uint8  kind    low 3 bits storeys (1..7), high bits 0 house, 1 big building, 2 church,
                   3 school or health, 4 shed or industry
"""

from __future__ import annotations

import json
import math
import urllib.request
from pathlib import Path

import numpy as np
import osmium
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
MAP = ROOT / "environment" / "web" / "map"
PBF = ROOT / ".map-cache" / "osm" / "rwanda-latest.osm.pbf"
URL = "https://download.geofabrik.de/africa/rwanda-latest.osm.pbf"

LAT0, LON0 = -2.0845, 29.7563  # as scripts/bake_map.py
KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON = 111.320 * math.cos(math.radians(LAT0))
WOODS_PX = 1024
KINDS = {"church": 2, "cathedral": 2, "chapel": 2, "mosque": 2, "school": 3, "hospital": 3, "clinic": 3,
         "university": 3, "college": 3, "industrial": 4, "warehouse": 4, "shed": 4, "farm_auxiliary": 4,
         "barn": 4, "greenhouse": 4, "commercial": 1, "retail": 1, "office": 1, "apartments": 1, "hotel": 1}


def local(lon: float, lat: float) -> tuple[float, float]:
    """Metres east and south of the hub."""
    return (lon - LON0) * KM_PER_DEG_LON * 1000, -(lat - LAT0) * KM_PER_DEG_LAT * 1000


def min_rect(pts: np.ndarray) -> tuple[float, float, float, float, float]:
    """Smallest rotated rectangle around a footprint: centre x, z, long side, short side, its angle."""
    from_hull = hull(pts)
    best = None
    for i in range(len(from_hull)):
        e = from_hull[(i + 1) % len(from_hull)] - from_hull[i]
        n = math.hypot(*e)
        if n < 1e-6:
            continue
        u = e / n
        v = np.array([-u[1], u[0]])
        a, b = from_hull @ u, from_hull @ v
        area = (a.max() - a.min()) * (b.max() - b.min())
        if best is None or area < best[0]:
            c = u * (a.max() + a.min()) / 2 + v * (b.max() + b.min()) / 2
            best = (area, c, a.max() - a.min(), b.max() - b.min(), math.atan2(u[1], u[0]))
    _, c, la, lb, ang = best
    if lb > la:
        la, lb, ang = lb, la, ang + math.pi / 2
    return c[0], c[1], la, lb, ang % math.pi


def hull(pts: np.ndarray) -> np.ndarray:
    p = sorted(map(tuple, pts))
    if len(p) < 3:
        return np.array(p)
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for q in p:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], q) <= 0:
            lo.pop()
        lo.append(q)
    for q in reversed(p):
        while len(up) >= 2 and cross(up[-2], up[-1], q) <= 0:
            up.pop()
        up.append(q)
    return np.array(lo[:-1] + up[:-1])


def main() -> None:
    meta = json.loads((MAP / "map.json").read_text())
    half_m = meta["layers"]["inner"]["half_km"] * 1000
    if not PBF.exists():
        PBF.parent.mkdir(parents=True, exist_ok=True)
        print("downloading", URL)
        req = urllib.request.Request(URL, headers={"User-Agent": "drone-delivery-rl-map-bake/1.0"})
        with urllib.request.urlopen(req, timeout=600) as r:
            PBF.write_bytes(r.read())

    rows, woods = [], []
    fp = osmium.FileProcessor(str(PBF)).with_locations().with_areas()
    for obj in fp:
        if not obj.is_area():
            continue
        tags = obj.tags
        b = tags.get("building")
        wood = tags.get("landuse") == "forest" or tags.get("natural") == "wood"
        if not (b and b != "no") and not wood:
            continue
        for ring in obj.outer_rings():
            try:
                pts = np.array([local(n.lon, n.lat) for n in ring])
            except osmium.InvalidLocationError:
                continue
            if len(pts) < 4 or np.abs(pts).max() > half_m + 2000:
                continue
            if wood:
                woods.append(pts)
                continue
            x, z, w, d, yaw = min_rect(pts[:-1])
            if abs(x) > half_m - 50 or abs(z) > half_m - 50 or w * d < 6 or w > 127 or d < 1.5:
                continue
            try:
                levels = int(float(tags.get("building:levels", "0")))
            except ValueError:
                levels = 0
            kind = KINDS.get(b, KINDS.get(tags.get("amenity", ""), 0))
            if kind == 0 and w * d > 220:
                kind = 1
            if levels <= 0:
                levels = 2 if kind == 1 and w * d > 400 else 1
            rows.append((round(x + half_m), round(z + half_m), min(round(w * 2), 255), min(round(d * 2), 255),
                         round(yaw / math.pi * 255) % 256, (kind << 3) | min(levels, 7)))
            break  # one rectangle a building, from its outer ring

    arr = np.array(rows, dtype=np.uint16)
    rec = np.zeros(len(arr), dtype=[("x", "<u2"), ("z", "<u2"), ("w", "u1"), ("d", "u1"), ("yaw", "u1"), ("kind", "u1")])
    for i, k in enumerate(("x", "z", "w", "d", "yaw", "kind")):
        rec[k] = arr[:, i]
    rec.sort(order=["z", "x"])
    (MAP / "buildings.bin").write_bytes(rec.tobytes())

    # woodland as a mask over the inner map, row 0 at the north edge
    img = Image.new("L", (WOODS_PX, WOODS_PX), 0)
    draw = ImageDraw.Draw(img)
    s = WOODS_PX / (2 * half_m)
    for pts in woods:
        draw.polygon([((x + half_m) * s, (z + half_m) * s) for x, z in pts], fill=255)
    img.save(MAP / "woods.png", optimize=True)

    meta["layers"]["inner"]["buildings"] = {"file": "buildings.bin", "count": int(len(rec)), "origin_m": half_m}
    meta["layers"]["inner"]["woods"] = {"mask": "woods.png", "px": WOODS_PX}
    if "OpenStreetMap" not in meta["attribution"]:
        meta["attribution"] += " Buildings and woodland: (c) OpenStreetMap contributors, ODbL."
    (MAP / "map.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{len(rec)} buildings, {len(woods)} woodland areas, woodland covers "
          f"{np.asarray(img).astype(bool).mean() * 100:.1f}% of the map")


if __name__ == "__main__":
    main()
