#!/usr/bin/env python3
"""Generate all flight-data graphs from a Ground Station CSV log."""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else (
    "/Users/brijnandandogra/Desktop/Ground_Station/logs/Flight_2025ASI001_2026-11-01_115959.csv"
)

OUT_PDF = CSV_PATH.replace(".csv", "_graphs.pdf")

# ---------- load & clean ----------
df = pd.read_csv(CSV_PATH)
df = df.sort_values("mission_time_s").reset_index(drop=True)

t = df["mission_time_s"].values
alt = df["altitude_m"].values
pres = df["pressure_hpa"].values
temp = df["temp_c"].values
voltage = df["voltage_v"].values

acc_x = df["acc_x"].values
acc_y = df["acc_y"].values
acc_z = df["acc_z"].values
acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

gyro_x = df["gyro_x"].values
gyro_y = df["gyro_y"].values
gyro_z = df["gyro_z"].values
gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)

lat = df["lat"].values
lon = df["lon"].values
nav_alt = df["nav_alt_m"].values
sats = df["sats"].values

pm1  = df["pm1_0_ugm3"].values
pm25 = df["pm2_5_ugm3"].values
pm4  = df["pm4_0_ugm3"].values
pm10 = df["pm10_ugm3"].values

fsm_name = df["fsm_state_name"].values
radio = df["radio"].values

# vertical speed via finite difference (m/s)
dt = np.diff(t)
dt = np.where(dt == 0, np.nan, dt)
vspeed = np.concatenate([[np.nan], np.diff(alt) / dt])

# FSM colour map
STATE_COLORS = {
    "BOOT":             "#9e9e9e",
    "LAUNCH_PAD":       "#2196F3",
    "ASCENT":           "#4CAF50",
    "DEPLOY":           "#FF9800",
    "DESCENT":          "#F44336",
    "AEROBRAKE_RELEASE":"#9C27B0",
}

def state_color(s):
    return STATE_COLORS.get(s, "#000000")

# time spans per FSM state (for shading)
state_spans = []
prev_s, prev_t = fsm_name[0], t[0]
for i in range(1, len(fsm_name)):
    if fsm_name[i] != prev_s:
        state_spans.append((prev_s, prev_t, t[i]))
        prev_s, prev_t = fsm_name[i], t[i]
state_spans.append((prev_s, prev_t, t[-1]))

def shade_states(ax, ymin=None, ymax=None, alpha=0.10):
    yl = ax.get_ylim()
    y0 = yl[0] if ymin is None else ymin
    y1 = yl[1] if ymax is None else ymax
    for s, t0, t1 in state_spans:
        ax.axvspan(t0, t1, color=state_color(s), alpha=alpha, linewidth=0)

def legend_states(ax):
    patches = [mpatches.Patch(color=v, label=k) for k, v in STATE_COLORS.items()
               if k in set(fsm_name)]
    ax.legend(handles=patches, fontsize=7, loc="upper right",
              ncol=2, framealpha=0.7)

# ------------------------------------------------------------------ helpers
def new_fig(title, nrows=1, ncols=1, figsize=None):
    if figsize is None:
        figsize = (14, 4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.subplots_adjust(hspace=0.45, wspace=0.35, top=0.88)
    return fig, axes

TIGHT = dict(pad=1.5)

# ================================================================== figures
with PdfPages(OUT_PDF) as pdf:

    # ---- 1. Flight overview: altitude + FSM timeline --------------------
    fig, axes = new_fig("Flight Overview — Altitude & FSM State", nrows=2, figsize=(14, 8))

    ax = axes[0]
    ax.plot(t, alt, color="#1565C0", lw=1.2, label="Baro alt")
    ax.plot(t, nav_alt, color="#EF6C00", lw=0.9, alpha=0.7, label="GPS alt")
    shade_states(ax)
    ax.set_xlabel("Mission time (s)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title("Barometric vs GPS Altitude")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    state_nums = {s: i for i, s in enumerate(STATE_COLORS)}
    sn = np.array([state_nums.get(s, 0) for s in fsm_name])
    scatter = ax.scatter(t, sn, c=[state_color(s) for s in fsm_name],
                         s=4, marker="|", linewidths=1)
    ax.set_yticks(list(state_nums.values()))
    ax.set_yticklabels(list(state_nums.keys()), fontsize=8)
    ax.set_xlabel("Mission time (s)")
    ax.set_title("FSM State Timeline")
    ax.grid(True, alpha=0.3, axis="x")
    legend_states(ax)

    fig.tight_layout(**TIGHT)
    pdf.savefig(fig)
    plt.close(fig)

    # ---- 2. Pressure & temperature -------------------------------------
    fig, axes = new_fig("Atmospheric Sensors", nrows=2, figsize=(14, 8))

    ax = axes[0]
    ax.plot(t, pres, color="#6A1B9A", lw=1.2)
    shade_states(ax)
    ax.set_xlabel("Mission time (s)")
    ax.set_ylabel("Pressure (hPa)")
    ax.set_title("Atmospheric Pressure vs Time")
    ax.grid(True, alpha=0.3)
    legend_states(ax)

    ax = axes[1]
    ax.plot(t, temp, color="#C62828", lw=1.2)
    shade_states(ax)
    ax.set_xlabel("Mission time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Temperature vs Time")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(**TIGHT)
    pdf.savefig(fig)
    plt.close(fig)

    # ---- 3. Pressure vs Altitude (hypsometric) -------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(alt, pres, c=t, cmap="viridis", s=3, alpha=0.7)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Mission time (s)")
    ax.set_xlabel("Altitude (m)")
    ax.set_ylabel("Pressure (hPa)")
    ax.set_title("Pressure vs Altitude (coloured by time)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(**TIGHT)
    pdf.savefig(fig)
    plt.close(fig)

    # ---- 4. Vertical speed ---------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5))
    vs_smooth = pd.Series(vspeed).rolling(10, center=True).mean().values
    ax.plot(t, vspeed, color="#90A4AE", lw=0.6, alpha=0.5, label="raw")
    ax.plot(t, vs_smooth, color="#1565C0", lw=1.4, label="10-pt rolling avg")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    shade_states(ax)
    ax.set_xlabel("Mission time (s)")
    ax.set_ylabel("Vertical speed (m/s)")
    ax.set_title("Derived Vertical Speed")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    legend_states(ax)
    fig.tight_layout(**TIGHT)
    pdf.savefig(fig)
    plt.close(fig)

    # ---- 5. Accelerometer XYZ ------------------------------------------
    fig, axes = new_fig("Accelerometer", nrows=2, figsize=(14, 8))

    ax = axes[0]
    ax.plot(t, acc_x, color="#E53935", lw=0.8, label="X")
    ax.plot(t, acc_y, color="#43A047", lw=0.8, label="Y")
    ax.plot(t, acc_z, color="#1E88E5", lw=0.8, label="Z")
    shade_states(ax)
    ax.set_xlabel("Mission time (s)")
    ax.set_ylabel("Acceleration (m/s²  or  g)")
    ax.set_title("Accelerometer — X, Y, Z Components")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, acc_mag, color="#6D4C41", lw=1.0, label="|a|")
    ax.axhline(9.81, color="gray", lw=0.8, ls="--", label="1 g")
    shade_states(ax)
    ax.set_xlabel("Mission time (s)")
    ax.set_ylabel("|Acceleration|")
    ax.set_title("Total Acceleration Magnitude")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    legend_states(ax)

    fig.tight_layout(**TIGHT)
    pdf.savefig(fig)
    plt.close(fig)

    # ---- 6. Gyroscope XYZ ----------------------------------------------
    fig, axes = new_fig("Gyroscope", nrows=2, figsize=(14, 8))

    ax = axes[0]
    ax.plot(t, gyro_x, color="#E53935", lw=0.8, label="X")
    ax.plot(t, gyro_y, color="#43A047", lw=0.8, label="Y")
    ax.plot(t, gyro_z, color="#1E88E5", lw=0.8, label="Z")
    shade_states(ax)
    ax.set_xlabel("Mission time (s)")
    ax.set_ylabel("Angular rate (°/s)")
    ax.set_title("Gyroscope — X, Y, Z Components")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, gyro_mag, color="#4A148C", lw=1.0)
    shade_states(ax)
    ax.set_xlabel("Mission time (s)")
    ax.set_ylabel("|ω| (°/s)")
    ax.set_title("Total Rotation Rate Magnitude")
    ax.grid(True, alpha=0.3)
    legend_states(ax)

    fig.tight_layout(**TIGHT)
    pdf.savefig(fig)
    plt.close(fig)

    # ---- 7. IMU during ascent peak (zoom) ------------------------------
    # find ASCENT time window
    ascent_mask = fsm_name == "ASCENT"
    if ascent_mask.any():
        t_asc = t[ascent_mask]
        t0_a, t1_a = t_asc.min(), t_asc.max()
        fig, axes = new_fig("IMU — Ascent Phase Zoom", nrows=2, figsize=(14, 8))

        mask = (t >= t0_a) & (t <= t1_a)
        ax = axes[0]
        ax.plot(t[mask], acc_x[mask], color="#E53935", lw=0.9, label="X")
        ax.plot(t[mask], acc_y[mask], color="#43A047", lw=0.9, label="Y")
        ax.plot(t[mask], acc_z[mask], color="#1E88E5", lw=0.9, label="Z")
        ax.plot(t[mask], acc_mag[mask], color="#000", lw=1.2, ls="--", label="|a|")
        ax.set_xlabel("Mission time (s)")
        ax.set_ylabel("Acceleration")
        ax.set_title("Acceleration during Ascent")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(t[mask], gyro_x[mask], color="#E53935", lw=0.9, label="X")
        ax.plot(t[mask], gyro_y[mask], color="#43A047", lw=0.9, label="Y")
        ax.plot(t[mask], gyro_z[mask], color="#1E88E5", lw=0.9, label="Z")
        ax.set_xlabel("Mission time (s)")
        ax.set_ylabel("Angular rate (°/s)")
        ax.set_title("Gyroscope during Ascent")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.tight_layout(**TIGHT)
        pdf.savefig(fig)
        plt.close(fig)

    # ---- 8. GPS track (lat/lon) ----------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(lon, lat, c=t, cmap="plasma", s=5, alpha=0.8)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Mission time (s)")
    # mark start/end
    ax.plot(lon[0], lat[0], "g^", ms=10, label="Start", zorder=5)
    ax.plot(lon[-1], lat[-1], "rs", ms=10, label="End", zorder=5)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("GPS Ground Track (coloured by time)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    # equal aspect for geographic accuracy
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout(**TIGHT)
    pdf.savefig(fig)
    plt.close(fig)

    # ---- 9. GPS satellite count ----------------------------------------
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, sats, color="#00838F", lw=1.0, drawstyle="steps-post")
    shade_states(ax)
    ax.set_xlabel("Mission time (s)")
    ax.set_ylabel("Satellites locked")
    ax.set_title("GPS Satellite Count")
    ax.set_ylim(0, sats.max() + 2)
    ax.grid(True, alpha=0.3)
    legend_states(ax)
    fig.tight_layout(**TIGHT)
    pdf.savefig(fig)
    plt.close(fig)

    # ---- 10. Particulate matter vs time --------------------------------
    fig, axes = new_fig("Particulate Matter vs Time", nrows=2, figsize=(14, 8))

    ax = axes[0]
    ax.plot(t, pm1,  color="#FFEB3B", lw=0.9, label="PM1.0")
    ax.plot(t, pm25, color="#FF9800", lw=0.9, label="PM2.5")
    ax.plot(t, pm4,  color="#F44336", lw=0.9, label="PM4.0")
    ax.plot(t, pm10, color="#9C27B0", lw=0.9, label="PM10")
    shade_states(ax)
    ax.set_xlabel("Mission time (s)")
    ax.set_ylabel("Concentration (μg/m³)")
    ax.set_title("All PM Channels vs Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, pm25, color="#FF9800", lw=1.2, label="PM2.5")
    shade_states(ax)
    ax.set_xlabel("Mission time (s)")
    ax.set_ylabel("PM2.5 (μg/m³)")
    ax.set_title("PM2.5 Detail vs Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    legend_states(ax)

    fig.tight_layout(**TIGHT)
    pdf.savefig(fig)
    plt.close(fig)

    # ---- 11. PM vs Altitude (scatter) ----------------------------------
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(alt, pm1,  s=3, alpha=0.5, color="#CDDC39",  label="PM1.0")
    ax.scatter(alt, pm25, s=3, alpha=0.5, color="#FF9800",  label="PM2.5")
    ax.scatter(alt, pm4,  s=3, alpha=0.5, color="#F44336",  label="PM4.0")
    ax.scatter(alt, pm10, s=3, alpha=0.5, color="#9C27B0",  label="PM10")
    ax.set_xlabel("Altitude (m)")
    ax.set_ylabel("Concentration (μg/m³)")
    ax.set_title("Particulate Matter vs Altitude")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(**TIGHT)
    pdf.savefig(fig)
    plt.close(fig)

    # ---- 12. PM2.5 vs Altitude (ascent vs descent) ---------------------
    fig, ax = plt.subplots(figsize=(10, 7))
    asc_m = fsm_name == "ASCENT"
    desc_m = np.isin(fsm_name, ["DESCENT", "AEROBRAKE_RELEASE", "DEPLOY"])
    ax.scatter(alt[asc_m],  pm25[asc_m],  s=5, alpha=0.6, color="#4CAF50", label="Ascent")
    ax.scatter(alt[desc_m], pm25[desc_m], s=5, alpha=0.6, color="#F44336", label="Descent/Release")
    rest_m = ~asc_m & ~desc_m
    ax.scatter(alt[rest_m], pm25[rest_m], s=5, alpha=0.4, color="#9E9E9E", label="Other")
    ax.set_xlabel("Altitude (m)")
    ax.set_ylabel("PM2.5 (μg/m³)")
    ax.set_title("PM2.5 vs Altitude — Ascent vs Descent")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(**TIGHT)
    pdf.savefig(fig)
    plt.close(fig)

    # ---- 13. Temperature vs Altitude -----------------------------------
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(alt, temp, c=t, cmap="coolwarm", s=4, alpha=0.7)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Mission time (s)")
    ax.set_xlabel("Altitude (m)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Temperature vs Altitude (coloured by time)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(**TIGHT)
    pdf.savefig(fig)
    plt.close(fig)

    # ---- 14. Acceleration vector norm vs altitude ----------------------
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(alt, acc_mag, c=t, cmap="plasma", s=4, alpha=0.7)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Mission time (s)")
    ax.axhline(9.81, color="gray", lw=1, ls="--", label="1 g")
    ax.set_xlabel("Altitude (m)")
    ax.set_ylabel("|Acceleration| (m/s²)")
    ax.set_title("|Acceleration| vs Altitude")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(**TIGHT)
    pdf.savefig(fig)
    plt.close(fig)

    # ---- 15. Radio source over time ------------------------------------
    fig, ax = plt.subplots(figsize=(14, 3))
    rx_colors = {"RX1": "#1565C0", "RX2": "#E53935"}
    for r, col in rx_colors.items():
        mask_r = radio == r
        ax.scatter(t[mask_r], np.ones(mask_r.sum()) * (1 if r == "RX1" else 2),
                   color=col, s=3, label=r, marker="|")
    ax.set_yticks([1, 2])
    ax.set_yticklabels(["RX1", "RX2"])
    ax.set_xlabel("Mission time (s)")
    ax.set_title("Active Radio Receiver vs Time")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout(**TIGHT)
    pdf.savefig(fig)
    plt.close(fig)

    # ---- 16. Full dashboard summary (multi-panel) ----------------------
    fig = plt.figure(figsize=(18, 22))
    fig.suptitle("Flight Dashboard — 2025ASI001", fontsize=15, fontweight="bold", y=0.995)
    gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.55, wspace=0.35,
                           top=0.975, bottom=0.04, left=0.07, right=0.97)

    panel_specs = [
        (0, 0, t, alt,    "Altitude (m)",          "Barometric Altitude",   "#1565C0"),
        (0, 1, t, pres,   "Pressure (hPa)",         "Atmospheric Pressure",  "#6A1B9A"),
        (1, 0, t, temp,   "Temperature (°C)",       "Temperature",           "#C62828"),
        (1, 1, t, vs_smooth, "Vert. speed (m/s)",   "Vertical Speed (smoothed)", "#00695C"),
        (2, 0, t, acc_mag, "|Accel| (m/s²)",        "Acceleration Magnitude", "#4E342E"),
        (2, 1, t, gyro_mag, "|ω| (°/s)",            "Rotation Rate Magnitude","#4A148C"),
        (3, 0, t, pm25,   "PM2.5 (μg/m³)",          "PM2.5",                 "#FF9800"),
        (3, 1, t, pm10,   "PM10 (μg/m³)",           "PM10",                  "#9C27B0"),
        (4, 0, t, sats,   "Satellites",             "GPS Satellites",         "#00838F"),
        (4, 1, t, nav_alt, "GPS Alt (m)",            "GPS Altitude",          "#EF6C00"),
    ]

    for row, col, x, y, ylabel, title, color in panel_specs:
        ax = fig.add_subplot(gs[row, col])
        ax.plot(x, y, color=color, lw=0.8)
        shade_states(ax)
        ax.set_xlabel("Mission time (s)", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    # state legend at bottom
    patches = [mpatches.Patch(color=v, label=k) for k, v in STATE_COLORS.items()
               if k in set(fsm_name)]
    fig.legend(handles=patches, loc="lower center", ncol=len(patches),
               fontsize=8, bbox_to_anchor=(0.5, 0.005), framealpha=0.8,
               title="FSM State shading")

    pdf.savefig(fig)
    plt.close(fig)

print(f"Saved {OUT_PDF}")
