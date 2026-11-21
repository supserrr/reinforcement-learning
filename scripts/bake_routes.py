"""Bake the environment's terrain numbers from the viewer's real elevation map.

For every site: its ground elevation. For every ordered pair of sites: how many
metres the drone climbs on the way, flying the route the viewer flies: a
terrain-following cruise about 120 m above the ground (it climbs over each
ridge ahead of time and holds height across narrow valleys), leaving from the
ground at the hub or from a 90 m loiter at a centre.

Prints the arrays for environment/custom_env.py (SITE_ELEVATION_M and
ROUTE_CLIMB_M). Needs environment/web/map/, written by scripts/bake_map.py.

    uv run python scripts/bake_routes.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from environment.custom_env import HUB, SITE_NAMES, SITE_XY  # noqa: E402

MAP = ROOT / "environment" / "web" / "map"
STEP_KM = 0.02
CRUISE_AGL, LOITER_AGL = 120.0, 90.0
WINDOW_KM, SMOOTH_KM = 0.6, 0.4  # look-ahead over ridges, then smoothing


def load_inner():
    meta = json.loads((MAP / "map.json").read_text())["layers"]["inner"]
    n = meta["height_px"]
    h = np.fromfile(MAP / meta["height"], dtype="<u2").astype(np.float64).reshape(n, n)
    return h, meta["half_km"]


def sampler(h: np.ndarray, half: float):
    n = h.shape[0]
    s = (n - 1) / (2 * half)

    def at(x_east: np.ndarray, y_north: np.ndarray) -> np.ndarray:
        fx = np.clip((x_east + half) * s, 0, n - 1.001)
        fy = np.clip((-y_north + half) * s, 0, n - 1.001)  # row 0 is the north edge
        ix, iy = np.floor(fx).astype(int), np.floor(fy).astype(int)
        tx, ty = fx - ix, fy - iy
        a, b = h[iy, ix], h[iy, ix + 1]
        c, d = h[iy + 1, ix], h[iy + 1, ix + 1]
        return (a + (b - a) * tx) * (1 - ty) + (c + (d - c) * tx) * ty

    return at


def window(a: np.ndarray, k: int, fn) -> np.ndarray:
    pad = np.pad(a, k, mode="edge")
    return np.array([fn(pad[i:i + 2 * k + 1]) for i in range(len(a))])


def route_climb(at, src: int, dst: int) -> float:
    p0, p1 = SITE_XY[src], SITE_XY[dst]
    d = float(np.linalg.norm(p1 - p0))
    n = max(int(d / STEP_KM), 2)
    t = np.linspace(0, 1, n + 1)
    g = at(p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t)
    cruise = window(g, int(WINDOW_KM / STEP_KM), np.max) + CRUISE_AGL
    cruise = window(cruise, int(SMOOTH_KM / STEP_KM), np.mean)
    cruise = np.maximum(cruise, g + 60.0)
    start = g[0] + (0.0 if src == HUB else LOITER_AGL)
    profile = np.concatenate([[start], cruise, [g[-1] + (12.0 if dst == HUB else 35.0)]])
    return float(np.clip(np.diff(profile), 0, None).sum())


def main():
    h, half = load_inner()
    at = sampler(h, half)
    elev = at(SITE_XY[:, 0], SITE_XY[:, 1])
    n = len(SITE_NAMES)
    climb = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                climb[i, j] = route_climb(at, i, j)
    np.set_printoptions(suppress=True)
    print("SITE_ELEVATION_M = np.array([" + ", ".join(f"{e:.0f}.0" for e in elev) + "])")
    print("ROUTE_CLIMB_M = np.array([")
    for row in climb:
        print("    [" + ", ".join(f"{v:.0f}.0" for v in row) + "],")
    print("])")
    for i, name in enumerate(SITE_NAMES):
        print(f"# {name:<15} {elev[i]:6.0f} m")


if __name__ == "__main__":
    main()
