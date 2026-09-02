"""Solve for four PM line colours that stay distinguishable in daylight.

Four lines share the particulate chart, which is the tightest colour problem in
the dashboard: every one has to clear 3:1 against the plot ground *and* differ
from all three others by at least 1.35:1 in lightness, so they remain separable
when sun glare flattens saturation and for red-green colour vision deficiency,
where hue alone would not be enough.

Run this after changing any PM colour:  python tools/pm_colours.py
"""
import colorsys
import sys

#: Chart interior these are drawn on (dashboard_ui.COL_PLOT_BG).
BG = "#e9edf3"
#: Minimum contrast for a graphical element against its background.
MIN_VS_BG = 3.0
#: Minimum lightness separation between any two lines on the same chart.
MIN_PAIR = 1.35


def _lin(c):
    c /= 255.0
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def luminance(h):
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def ratio(a, b):
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def solve(hue, sat, target):
    """Darkest-to-lightest sweep at a fixed hue; closest to *target* vs BG."""
    best = None
    for i in range(2, 250):
        r, g, b = colorsys.hsv_to_rgb(hue, sat, i / 255.0)
        h = "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))
        d = abs(ratio(h, BG) - target)
        if best is None or d < best[0]:
            best = (d, h)
    return best[1]


# Four rungs, each ~1.37x apart, with the lightest just clearing 3:1. Hues are
# teal / green / ochre / red so the order also reads as a natural progression
# from fine to coarse particulate.
PLAN = [
    ("PM1.0", 0.515, 0.95, 3.10),   # teal
    ("PM2.5", 0.365, 0.85, 4.25),   # green
    ("PM4.0", 0.075, 1.00, 5.85),   # ochre
    ("PM10", 0.985, 0.90, 8.10),    # dark red
]

cols = [(name, solve(hue, sat, target)) for name, hue, sat, target in PLAN]

print("=" * 62)
print("PM CHART COLOURS  (plot ground %s)" % BG)
print("=" * 62)
ok = True
for name, c in cols:
    r = ratio(c, BG)
    good = r >= MIN_VS_BG
    ok &= good
    print("  %-6s %-9s vs ground %5.2f:1  %s"
          % (name, c, r, "OK" if good else "TOO LIGHT"))

print()
print("pairwise lightness separation (floor %.2f:1):" % MIN_PAIR)
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        r = ratio(cols[i][1], cols[j][1])
        good = r >= MIN_PAIR
        ok &= good
        print("  %-6s / %-6s %5.2f:1  %s"
              % (cols[i][0], cols[j][0], r, "OK" if good else "TOO CLOSE"))

print()
print("COL_PM1  = %r" % cols[0][1])
print("COL_PM25 = %r" % cols[1][1])
print("COL_PM4  = %r" % cols[2][1])
print("COL_PM10 = %r" % cols[3][1])
print()
print("ALL PASS" if ok else "CONSTRAINTS NOT MET")
sys.exit(0 if ok else 1)
