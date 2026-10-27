# Map packs

## launchpad_s2.mbtiles - launch pad satellite map

**Sentinel-2 cloudless 2023 by EOX IT Services GmbH (contains modified Copernicus Sentinel data 2023).**

The map window shows this credit whenever the pack is on screen, as:

> Sentinel-2 cloudless - https://s2maps.eu by EOX IT Services GmbH (Contains modified Copernicus Sentinel data 2023)

| | |
|---|---|
| Licence | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) - **non-commercial use only**, attribution required, share alike |
| Bounds | W 84.2542, E 84.3549, S 26.6751, N 26.7650 (10 km x 10 km) |
| Zoom | 8 to 14 (about 10 m per pixel at 14; the map scales up to 17) |
| Launch pad | 26.720028 N, 84.304583 E (26° 43' 12.1" N, 84° 18' 16.5" E) |
| Format | MBTiles, 49 JPEG tiles, 864,256 bytes |

Rebuild it with:

```bash
python tools/fetch_tiles.py \
    --bbox 84.2542 26.6751 84.3549 26.7650 \
    --zoom 8 14 \
    --source "Sentinel-2 cloudless 2023 (EOX)" \
    --name "Launch pad 10 km - Sentinel-2 cloudless 2023" \
    --out assets/maps/launchpad_s2.mbtiles
```

## india_overview.osmpack.json

Country outline, state boundaries and major cities from Natural Earth (public
domain), drawn by the application as a vector pack.
