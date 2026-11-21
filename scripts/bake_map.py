"""Bake the viewer's map of Muhanga from open satellite imagery and elevation.

Writes environment/web/map/: two imagery textures (inner, around the sites,
and outer, the surrounding hills to Lake Kivu), two heightmaps and map.json.
The viewer loads these files, so it needs no network at run time.

    uv run python scripts/bake_map.py            # tiles are cached in .map-cache/

Sources (see environment/web/map/README.md for licences):
    imagery    Sentinel-2 cloudless 2025 by EOX (CC BY-NC-SA 4.0)
    elevation  AWS Terrain Tiles, terrarium encoding (SRTM and others)
"""

from __future__ import annotations

import io
import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "environment" / "web" / "map"
CACHE = ROOT / ".map-cache"

# Muhanga (Gitarama) town: the hub. The env's site positions are kilometres from here.
LAT0, LON0 = -2.0845, 29.7563
KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON = 111.320 * math.cos(math.radians(LAT0))

IMAGERY = "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2025_3857/default/g/{z}/{y}/{x}.jpg"
TERRAIN = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"

# name: (half width in km, imagery zoom, imagery px, terrain zoom, heightmap px)
LAYERS = {
    "inner": (32.0, 14, 6144, 13, 2048),   # 31 m a height sample: SRTM's own resolution
    "outer": (100.0, 11, 2048, 10, 1024),
}


def merc(lat: float, lon: float, z: int) -> tuple[float, float]:
    """Global pixel coordinates of a point in the 256 px Web Mercator grid at zoom z."""
    n = 256 * 2**z
    s = math.sin(math.radians(lat))
    return (lon + 180) / 360 * n, (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * n


def fetch(url: str, path: Path) -> bytes:
    if path.exists():
        return path.read_bytes()
    req = urllib.request.Request(url, headers={"User-Agent": "drone-delivery-rl-map-bake/1.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return data
        except Exception:  # noqa: BLE001 - retry any network error
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(url)


def mosaic(template: str, kind: str, z: int, half_km: float, mode: str) -> tuple[Image.Image, tuple[float, float, float, float]]:
    """Stitch every tile covering the square, return it and the square's pixel box inside it."""
    dlat, dlon = half_km / KM_PER_DEG_LAT, half_km / KM_PER_DEG_LON
    x0, y0 = merc(LAT0 + dlat, LON0 - dlon, z)  # north-west corner
    x1, y1 = merc(LAT0 - dlat, LON0 + dlon, z)  # south-east corner
    tx0, ty0, tx1, ty1 = int(x0 // 256), int(y0 // 256), int(x1 // 256), int(y1 // 256)
    tiles = [(tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]
    print(f"  {kind} z{z}: {len(tiles)} tiles")

    def get(t):
        tx, ty = t
        ext = template.rsplit(".", 1)[1]
        return t, fetch(template.format(z=z, x=tx, y=ty), CACHE / kind / str(z) / f"{tx}_{ty}.{ext}")

    canvas = Image.new(mode, ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256))
    with ThreadPoolExecutor(6) as pool:
        for (tx, ty), data in pool.map(get, tiles):
            canvas.paste(Image.open(io.BytesIO(data)).convert(mode), ((tx - tx0) * 256, (ty - ty0) * 256))
    return canvas, (x0 - tx0 * 256, y0 - ty0 * 256, x1 - tx0 * 256, y1 - ty0 * 256)


def fill_voids(h: np.ndarray) -> np.ndarray:
    """SRTM has no data over some water; grow the valid neighbours into those holes."""
    void = h < 300
    while void.any():
        total = np.zeros_like(h)
        count = np.zeros_like(h)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ok = np.roll(~void, (dy, dx), (0, 1))
            total += np.where(ok, np.roll(h, (dy, dx), (0, 1)), 0)
            count += ok
        grow = void & (count > 0)
        h[grow] = total[grow] / count[grow]
        void &= ~grow
    return h


def bake(name: str, half_km: float, img_z: int, img_px: int, dem_z: int, dem_px: int) -> dict:
    print(name)
    img, box = mosaic(IMAGERY, "imagery", img_z, half_km, "RGB")
    img = img.resize((img_px, img_px), Image.LANCZOS, box=box)
    img.save(OUT / f"{name}.jpg", quality=84, optimize=True, progressive=True)

    dem, box = mosaic(TERRAIN, "terrain", dem_z, half_km, "RGB")
    a = np.asarray(dem, dtype=np.float32)
    metres = fill_voids(a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768)
    h = np.asarray(Image.fromarray(metres, mode="F").resize((dem_px, dem_px), Image.BILINEAR, box=box))
    h = np.clip(h, 0, 65535).round().astype("<u2")
    (OUT / f"{name}-height.bin").write_bytes(h.tobytes())
    print(f"  elevation {h.min()}–{h.max()} m")
    return {"half_km": half_km, "image": f"{name}.jpg", "image_px": img_px,
            "height": f"{name}-height.bin", "height_px": dem_px,
            "min_m": int(h.min()), "max_m": int(h.max())}


# Lake Kivu's surface and a point well inside it (km east, km north of the hub)
KIVU_M, KIVU_SEED = 1462, (-67.0, 9.0)


def bake_lake(outer: dict) -> dict:
    """Mark the ground that is Lake Kivu's surface: the connected patch of near-lake-level
    ground around a point in the lake. Low valleys elsewhere stay dry, and the river
    gorge out of its south end stops the fill where it drops below the lake."""
    n, half = outer["height_px"], outer["half_km"]
    h = np.frombuffer((OUT / outer["height"]).read_bytes(), dtype="<u2").reshape(n, n).astype(np.int32)
    ok = (h >= KIVU_M - 12) & (h <= KIVU_M + 6)
    px = lambda km: int(round((km + half) / (2 * half) * (n - 1)))
    seed = (px(-KIVU_SEED[1]), px(KIVU_SEED[0]))  # row from the north edge, column from the west
    if not ok[seed]:
        raise SystemExit(f"lake seed {KIVU_SEED} is not at lake level ({h[seed]} m)")
    lake = np.zeros_like(ok)
    lake[seed] = True
    while True:  # grow the seed through level ground
        grown = lake.copy()
        grown[1:] |= lake[:-1]; grown[:-1] |= lake[1:]; grown[:, 1:] |= lake[:, :-1]; grown[:, :-1] |= lake[:, 1:]
        grown &= ok
        if (grown == lake).all():
            break
        lake = grown
    Image.fromarray((lake * 255).astype(np.uint8), mode="L").save(OUT / "outer-lake.png", optimize=True)
    print(f"lake: {lake.sum() * (2 * half / n) ** 2:.0f} km² at {KIVU_M} m")
    return {"mask": "outer-lake.png", "surface_m": KIVU_M}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    layers = {name: bake(name, *spec) for name, spec in LAYERS.items()}
    layers["outer"]["lake"] = bake_lake(layers["outer"])
    meta = {
        "center": {"lat": LAT0, "lon": LON0},
        "note": "x east and y north in km from the center; heightmaps are uint16 metres, row 0 at the north edge",
        "layers": layers,
        "attribution": "Imagery: Sentinel-2 cloudless 2025 by EOX (contains modified Copernicus Sentinel data 2025), "
                       "CC BY-NC-SA 4.0. Elevation: AWS Terrain Tiles (SRTM, NASA).",
    }
    (OUT / "map.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
