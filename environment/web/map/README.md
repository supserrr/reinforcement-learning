# Map of Muhanga

The viewer's terrain, baked by `scripts/bake_map.py` (`uv run python scripts/bake_map.py`). It is centred on Muhanga town (2.0845° S, 29.7563° E), the hub; the environment's site positions are kilometres east and north of it.

| File | What |
|---|---|
| `inner.jpg` | imagery, 64 × 64 km around the hub, 6144 px (≈10 m a pixel) |
| `outer.jpg` | imagery, 200 × 200 km out to Lake Kivu, 2048 px |
| `inner-height.bin` | elevation, 2048 × 2048 (31 m, SRTM's own resolution) little-endian uint16 metres, row 0 at the north edge |
| `outer-height.bin` | elevation, 1024 × 1024, same format |
| `outer-lake.png` | Lake Kivu: the level ground joined to the lake, where the viewer draws water |
| `buildings.bin` | every OpenStreetMap building in the inner map as a rotated rectangle, 8 bytes each (see `scripts/bake_buildings.py`) |
| `woods.png` | where OpenStreetMap maps forest or woodland, 1024 × 1024 over the inner map |
| `map.json` | extents, sizes and the attribution line |

## Sources and licences

- **Imagery:** Sentinel-2 cloudless 2025 by EOX IT Services GmbH, <https://cloudless.eox.at> (contains modified Copernicus Sentinel data 2025), licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). `inner.jpg` and `outer.jpg` are resampled crops of it and are shared under the same licence. Commercial use needs a licence from EOX.
- **Elevation:** AWS Terrain Tiles (Mapzen terrarium encoding), built from NASA SRTM data, public domain.
- **Buildings and woodland:** © OpenStreetMap contributors, [ODbL 1.0](https://opendatacommons.org/licenses/odbl/), from the Geofabrik Rwanda extract (`uv run scripts/bake_buildings.py`). `buildings.bin` and `woods.png` are derived databases under the same licence.

The viewer shows the attribution in its corner.
