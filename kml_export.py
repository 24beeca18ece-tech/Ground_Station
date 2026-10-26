"""
kml_export.py
=============

Turns a session's tracks into a KML 2.2 document Google Earth can replay.

PURE, ON PURPOSE
----------------
No Qt, no file I/O, no clock: :func:`build_kml` takes tracks and options and
returns a string. That makes the whole output testable without a display or a
radio, which matters because a KML is read long after the flight, by people
who were not there, and a silent error in it is not discoverable at the time.

Everything is written through ``xml.etree.ElementTree``, so a vehicle name or
team ID containing ``&``, ``<`` or a quote is escaped by the serialiser rather
than by hand.

WHAT GOOGLE EARTH GETS
----------------------
Per vehicle, one Folder containing:

* a ``gx:Track`` -- paired ``<when>`` / ``<gx:coord>`` elements, which is what
  drives the time slider and lets the flight be replayed rather than merely
  looked at. ``ExtendedData`` arrays carry fsm_state, sats and packet_count so
  the balloon shows telemetry at the cursor's moment.
* a ``LineString`` of the same path, extruded to the ground, as the static
  shape that is visible with the time slider parked.
* three Placemarks: Launch, Apogee and Last known / Landing.

ALTITUDE
--------
``nav_alt_m`` -- GNSS altitude above mean sea level -- with
``altitudeMode absolute``. The barometric ``altitude_m`` is deliberately NOT
used: it is referenced to a fixed 1013.25 hPa sea level with no ground
zero-set, so it carries a constant offset that would place the whole track at
the wrong height. A track with no GNSS altitude is written ``clampToGround``
and says so in the document description, rather than being drawn at a height
nobody measured.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

from track_store import Track, TrackPoint, DEFAULT_JUMP_M, DEFAULT_JUMP_WINDOW_S

KML_NS = "http://www.opengis.net/kml/2.2"
GX_NS = "http://www.google.com/kml/ext/2.2"

#: Per-vehicle line styles. KML colours are aabbggrr -- alpha, then BGR, the
#: reverse of the usual rrggbb -- which is a classic source of silently wrong
#: colours, so each is written next to the rrggbb it corresponds to.
VEHICLE_STYLES: Dict[str, Dict[str, str]] = {
    # #ff1f2e crimson
    "ROCKET": {"line": "ff2e1fff", "poly": "552e1fff", "icon": "ff2e1fff"},
    # #00b4d8 cyan
    "CANSAT": {"line": "ffd8b400", "poly": "55d8b400", "icon": "ffd8b400"},
    # #ffb000 amber
    "GENERIC": {"line": "ff00b0ff", "poly": "5500b0ff", "icon": "ff00b0ff"},
}

#: Placemark pin colours, same aabbggrr convention.
PIN_LAUNCH = "ff00c000"      # green
PIN_APOGEE = "ff00d4ff"      # amber
PIN_LAST = "ff2020ff"        # red


def _iso(epoch: float) -> str:
    """UTC ISO-8601 with a trailing Z, which is what KML's <when> expects."""
    if not (isinstance(epoch, (int, float)) and math.isfinite(epoch)) or epoch <= 0:
        return ""
    return (datetime.fromtimestamp(float(epoch), tz=timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ"))


def _coord(point: TrackPoint, use_alt: bool) -> str:
    """``lon,lat,alt`` -- KML's order, which is NOT lat,lon."""
    alt = point.alt_m if (use_alt and point.has_valid_alt) else 0.0
    return "%.7f,%.7f,%.2f" % (point.lon, point.lat, alt)


def _sub(parent: ET.Element, tag: str, text: Optional[str] = None,
         **attrs: str) -> ET.Element:
    el = ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})
    if text is not None:
        el.text = str(text)
    return el


def _describe_point(point: TrackPoint, label: str) -> str:
    """An HTML table for a Placemark balloon.

    Built as text rather than markup elements because KML descriptions are
    CDATA/HTML; the ElementTree serialiser still escapes it on the way out,
    so Google Earth receives it as displayable text and nothing here can
    break the document if a field ever contains a stray angle bracket.
    """
    alt = ("%.1f m MSL" % point.alt_m) if point.has_valid_alt else "not reported"
    rows = [
        ("Event", label),
        ("Time (UTC)", _iso(point.epoch) or "unknown"),
        ("Mission time", "T+%.1f s" % point.mission_time_s),
        ("Latitude", "%.7f" % point.lat),
        ("Longitude", "%.7f" % point.lon),
        ("GNSS altitude", alt),
        ("Satellites", str(point.sats)),
        ("Flight state", point.fsm_state or "not reported"),
        ("Packet", str(point.packet_count)),
    ]
    body = "".join("<tr><td><b>%s</b></td><td>%s</td></tr>" % (k, v)
                   for k, v in rows)
    return "<table>%s</table>" % body


def _add_styles(doc: ET.Element) -> None:
    for vehicle, colours in VEHICLE_STYLES.items():
        style = _sub(doc, "Style", id="track-%s" % vehicle.lower())
        line = _sub(style, "LineStyle")
        _sub(line, "color", colours["line"])
        _sub(line, "width", "3")
        poly = _sub(style, "PolyStyle")
        _sub(poly, "color", colours["poly"])
        icon = _sub(style, "IconStyle")
        _sub(icon, "color", colours["icon"])
        _sub(icon, "scale", "0.9")

    for name, colour in (("launch", PIN_LAUNCH), ("apogee", PIN_APOGEE),
                         ("last", PIN_LAST)):
        style = _sub(doc, "Style", id="pin-%s" % name)
        icon = _sub(style, "IconStyle")
        _sub(icon, "color", colour)
        _sub(icon, "scale", "1.1")
        icon_el = _sub(icon, "Icon")
        _sub(icon_el, "href",
             "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png")


def _add_placemark(folder: ET.Element, name: str, point: TrackPoint,
                   style_id: str, use_alt: bool, label: str) -> None:
    mark = _sub(folder, "Placemark")
    _sub(mark, "name", name)
    _sub(mark, "description", _describe_point(point, label))
    _sub(mark, "styleUrl", "#%s" % style_id)
    when = _iso(point.epoch)
    if when:
        stamp = _sub(mark, "TimeStamp")
        _sub(stamp, "when", when)
    geom = _sub(mark, "Point")
    _sub(geom, "altitudeMode", "absolute" if use_alt else "clampToGround")
    if use_alt:
        _sub(geom, "extrude", "1")
    _sub(geom, "coordinates", _coord(point, use_alt))


def _add_track_folder(parent: ET.Element, track: Track) -> None:
    """One vehicle: gx:Track, LineString and the three event Placemarks."""
    points = track.points
    use_alt = track.has_altitude
    style_id = "track-%s" % track.vehicle.lower()
    if track.vehicle not in VEHICLE_STYLES:
        style_id = "track-generic"

    folder = _sub(parent, "Folder")
    _sub(folder, "name", track.vehicle)
    _sub(folder, "description",
         "%d fixes. Altitude: %s." % (
             len(points),
             "GNSS MSL (absolute)" if use_alt
             else "not transmitted on this link; path clamped to ground"))

    # --- gx:Track: this is what the time slider replays -------------------
    replay = _sub(folder, "Placemark")
    _sub(replay, "name", "%s — replay" % track.vehicle)
    _sub(replay, "styleUrl", "#%s" % style_id)
    gx_track = _sub(replay, "gx:Track")
    _sub(gx_track, "altitudeMode", "absolute" if use_alt else "clampToGround")
    # Every <when> is followed by its <gx:coord>; Google Earth pairs them by
    # position, so the two element counts must match exactly.
    for point in points:
        _sub(gx_track, "when", _iso(point.epoch))
    for point in points:
        _sub(gx_track, "gx:coord",
             _coord(point, use_alt).replace(",", " "))

    extended = _sub(gx_track, "ExtendedData")
    for field_name, values in (
        ("fsm_state", [p.fsm_state for p in points]),
        ("sats", [str(p.sats) for p in points]),
        ("packet_count", [str(p.packet_count) for p in points]),
    ):
        array = _sub(extended, "gx:SimpleArrayData", name=field_name)
        for value in values:
            _sub(array, "gx:value", value)

    # --- LineString: the static shape, visible with the slider parked -----
    path = _sub(folder, "Placemark")
    _sub(path, "name", "%s — path" % track.vehicle)
    _sub(path, "styleUrl", "#%s" % style_id)
    line = _sub(path, "LineString")
    _sub(line, "altitudeMode", "absolute" if use_alt else "clampToGround")
    _sub(line, "extrude", "1" if use_alt else "0")
    _sub(line, "tessellate", "1")
    _sub(line, "coordinates",
         "\n".join(_coord(p, use_alt) for p in points))

    # --- the three events -------------------------------------------------
    first, last = track.first, track.last
    if first is not None:
        _add_placemark(folder, "Launch — %s" % track.vehicle, first,
                       "pin-launch", use_alt, "First fix")
    apogee = track.apogee
    if apogee is not None and apogee is not first and apogee is not last:
        _add_placemark(folder, "Apogee — %s (%.1f m)"
                       % (track.vehicle, apogee.alt_m), apogee,
                       "pin-apogee", use_alt, "Highest GNSS altitude")
    if last is not None and last is not first:
        _add_placemark(folder, "Last known — %s" % track.vehicle, last,
                       "pin-last", use_alt, "Last fix received")


def build_kml(tracks: Sequence[Track], options: Optional[Dict[str, Any]] = None) -> str:
    """Return a complete KML 2.2 document for *tracks*.

    Parameters
    ----------
    tracks:
        :class:`track_store.Track` objects. Empty tracks are skipped; if none
        has any point the document is still valid, and says so.
    options:
        ``team_id``, ``session_date`` (text for the document name),
        ``jump_m`` / ``jump_window_s`` (reported in the description, since the
        tracks arrive already filtered), and ``source`` (free text naming
        where the data came from, e.g. a CSV path).
    """
    options = dict(options or {})
    team_id = str(options.get("team_id") or "UNKNOWN")
    session_date = str(options.get("session_date")
                       or datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    ET.register_namespace("", KML_NS)
    ET.register_namespace("gx", GX_NS)
    root = ET.Element("kml", {"xmlns": KML_NS, "xmlns:gx": GX_NS})
    doc = _sub(root, "Document")
    _sub(doc, "name", "Flight %s — %s" % (team_id, session_date))

    usable = [t for t in tracks if t.points]
    total = sum(len(t.points) for t in usable)
    jumps = sum(t.jumps_dropped for t in tracks)
    dupes = sum(t.duplicates_dropped for t in tracks)
    stationary = sum(getattr(t, "stationary_dropped", 0) for t in tracks)
    thinned = sum(getattr(t, "thinned", 0) for t in tracks)
    no_alt = [t.vehicle for t in usable if not t.has_altitude]

    lines = [
        "Ground station export for team %s." % team_id,
        "Session: %s." % session_date,
        "%d position fixes across %d vehicle(s): %s."
        % (total, len(usable),
           ", ".join("%s %d" % (t.vehicle, len(t.points)) for t in usable)
           or "none"),
        "Altitude is GNSS height above mean sea level, not barometric.",
        "Discarded before export: %d GPS jumps (over %.0f m in under %.1f s), "
        "%d duplicate packet counts. Fixes with no lock, and exact 0,0, were "
        "never recorded."
        % (jumps, float(options.get("jump_m", DEFAULT_JUMP_M)),
           float(options.get("jump_window_s", DEFAULT_JUMP_WINDOW_S)), dupes),
    ]
    if stationary:
        lines.append(
            "%d fixes were not stored because the vehicle had not moved more "
            "than 1 m for over a minute; one fix every 10 s was kept while it "
            "was stationary." % stationary)
    if thinned:
        lines.append(
            "%d older fixes were thinned 2:1 after the track reached the "
            "300,000-point cap. The most recent half is at full resolution."
            % thinned)
    if no_alt:
        lines.append(
            "No GNSS altitude was transmitted for: %s. Those paths are "
            "clamped to the ground rather than drawn at a height that was "
            "never measured." % ", ".join(no_alt))
    if options.get("source"):
        lines.append("Source: %s" % options["source"])
    if not usable:
        lines.append("WARNING: this session recorded no usable position fix.")
    _sub(doc, "description", "\n".join(lines))

    _add_styles(doc)
    for track in usable:
        _add_track_folder(doc, track)

    body = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + body + "\n"


def kml_filename(team_id: str, when: Optional[float] = None) -> str:
    """``Flight_<team>_<YYYY-MM-DD_HHMMSS>.kml``, matching the CSV pattern."""
    from telemetry_packet import safe_filename
    stamp = datetime.fromtimestamp(when) if when else datetime.now()
    return "Flight_%s_%s.kml" % (safe_filename(team_id),
                                 stamp.strftime("%Y-%m-%d_%H%M%S"))
