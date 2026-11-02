#!/usr/bin/env python3
"""Build a 3D KML flight path from a ground-station CSV log.
Height = barometric altitude (altitude_m, zeroed at the pad) drawn relativeToGround.
Usage: python3 tools/log_to_3d_kml.py logs/Flight_xxx.csv [out.kml]
"""
import csv, sys, os
from xml.sax.saxutils import escape

COLORS = {  # KML aabbggrr
    "BOOT": "ff9e9e9e", "LAUNCH_PAD": "ff9e9e9e", "ASCENT": "ff3c3cff",
    "DEPLOY": "ff00d7ff", "DESCENT": "ff00a5ff", "AEROBRAKE_RELEASE": "ffffb400",
}
def num(v):
    try:
        x = float(v); return x if x == x else None
    except Exception: return None

src = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + "_3D.kml"
rows = []
for r in csv.DictReader(open(src, newline="")):
    lat, lon, h = num(r["lat"]), num(r["lon"]), num(r["altitude_m"])
    if None in (lat, lon, h) or (lat == 0 and lon == 0) or r.get("checksum_valid", "1") != "1":
        continue
    rows.append(dict(t=float(r["gs_recv_epoch"]), iso=r["gs_recv_iso"][:23] + "Z", lat=lat, lon=lon,
                     h=max(h, 0.0), g=num(r["nav_alt_m"]), st=r["fsm_state_name"], pc=r["packet_count"]))
rows.sort(key=lambda x: x["t"])
if not rows: sys.exit("no usable fixes")

def c(p): return f"{p['lon']:.7f},{p['lat']:.7f},{p['h']:.1f}"

# contiguous segments by FSM state
segs, cur = [], None
for p in rows:
    if cur is None or p["st"] != cur[0]:
        if cur: cur[1].append(p)         # join segments so the line is continuous
        cur = (p["st"], [p]); segs.append(cur)
    else: cur[1].append(p)

apo = max(rows, key=lambda p: p["h"])
first_asc = next((p for p in rows if p["st"] == "ASCENT"), rows[0])
dep = next((p for p in rows if p["st"] in ("DEPLOY", "DESCENT", "AEROBRAKE_RELEASE")), None)
last = rows[-1]

K = ['<?xml version="1.0" encoding="UTF-8"?>',
     '<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2"><Document>',
     f'<name>{escape(os.path.basename(out))}</name>',
     f'<description>3D path from {escape(os.path.basename(src))}. Height = barometric altitude above the pad '
     f'(relativeToGround). {len(rows)} fixes. Apogee {apo["h"]:.0f} m AGL.</description>']
for st, col in COLORS.items():
    K.append(f'<Style id="s_{st}"><LineStyle><color>{col}</color><width>4</width></LineStyle>'
             f'<PolyStyle><color>{"40" + col[2:]}</color></PolyStyle></Style>')
K.append('<Style id="gnss"><LineStyle><color>9900ffff</color><width>2</width></LineStyle></Style>')
K.append('<Style id="pin"><IconStyle><scale>1.1</scale><Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon></IconStyle></Style>')

K.append('<Folder><name>Flight path (by state)</name>')
for st, pts in segs:
    K.append(f'<Placemark><name>{escape(st)} ({pts[0]["h"]:.0f} to {pts[-1]["h"]:.0f} m)</name><styleUrl>#s_{st if st in COLORS else "ASCENT"}</styleUrl>'
             '<LineString><extrude>1</extrude><tessellate>0</tessellate><altitudeMode>relativeToGround</altitudeMode><coordinates>'
             + " ".join(c(p) for p in pts) + '</coordinates></LineString></Placemark>')
K.append('</Folder>')

K.append('<Folder><name>Markers</name>')
for name, p in [("Launch", first_asc), ("Apogee", apo), ("Deploy / release", dep), ("Last fix", last)]:
    if p:
        K.append(f'<Placemark><name>{name}: {p["h"]:.0f} m</name><description>{p["iso"]} | {escape(p["st"])} | packet {p["pc"]}</description>'
                 f'<styleUrl>#pin</styleUrl><Point><extrude>1</extrude><altitudeMode>relativeToGround</altitudeMode><coordinates>{c(p)}</coordinates></Point></Placemark>')
K.append('</Folder>')

K.append('<Placemark><name>Replay (time slider)</name><styleUrl>#s_ASCENT</styleUrl><gx:Track><altitudeMode>relativeToGround</altitudeMode>')
K += [f'<when>{p["iso"]}</when>' for p in rows]
K += [f'<gx:coord>{p["lon"]:.7f} {p["lat"]:.7f} {p["h"]:.1f}</gx:coord>' for p in rows]
K.append('</gx:Track></Placemark>')

g = [p for p in rows if p["g"] is not None]
if g:
    K.append('<Placemark><name>GNSS altitude (MSL, comparison)</name><visibility>0</visibility><styleUrl>#gnss</styleUrl>'
             '<LineString><altitudeMode>absolute</altitudeMode><coordinates>'
             + " ".join(f'{p["lon"]:.7f},{p["lat"]:.7f},{p["g"]:.1f}' for p in g) + '</coordinates></LineString></Placemark>')
K.append('</Document></kml>')
open(out, "w").write("\n".join(K))
print(f"wrote {out}: {len(rows)} fixes, {len(segs)} segments, apogee {apo['h']:.1f} m, last fix {last['h']:.1f} m ({last['st']})")
