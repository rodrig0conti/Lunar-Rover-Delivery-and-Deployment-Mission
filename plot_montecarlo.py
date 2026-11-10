# figures for the presentation, read straight from montecarlo_results.csv.
# no re-running: everything is post-processed from the saved rows.

import csv
import numpy as np
import matplotlib.pyplot as plt

CSV_IN = "montecarlo_results.csv"
TARGET_LAT = np.deg2rad(-89.0)
TARGET_LON = np.deg2rad(0.0)
R_MOON = 1737.4
V_LIMIT = 4.0

# ---- load -------------------------------------------------------------

def load(path):
    lat, lon, v_vert, dist, soft = [], [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            lat.append(float(row["lat"]))
            lon.append(float(row["lon"]))
            v_vert.append(float(row["v_vert"]))
            dist.append(float(row["dist"]))
            soft.append(row["soft"] == "True")
    return (np.deg2rad(lat), np.deg2rad(lon), np.array(v_vert),
            np.array(dist), np.array(soft))


lat, lon, v_vert, dist, soft = load(CSV_IN)
n = len(dist)

# ---- figure 1: success probability vs zone radius ---------------------

radii = np.linspace(0.5, 200.0, 400)
p_success = np.array([(soft & (dist < R)).mean() for R in radii])

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(radii, p_success * 100, color="#1f5fa8", lw=2.2)
ax.axhline(90, color="#888", ls="--", lw=1, label="soft-landing ceiling (90%)")
cep = np.percentile(dist[soft], 50)
ax.axvline(cep, color="#c0392b", ls=":", lw=1.2, label=f"CEP = {cep:.0f} km")
ax.set_xlabel("landing-zone radius  [km]")
ax.set_ylabel("P(success)  [%]")
ax.set_title(f"landing success probability vs zone radius  (N={n})")
ax.set_xlim(0, 200); ax.set_ylim(0, 100)
ax.grid(alpha=0.25); ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("fig_success_vs_radius.png", dpi=150)

# ---- figure 2: landing dispersion, target-centered --------------------
# azimuthal-equidistant about the target: exact distance, azimuth = bearing.

dlon = lon - TARGET_LON
az = np.arctan2(np.sin(dlon) * np.cos(lat),
                np.cos(TARGET_LAT) * np.sin(lat)
                - np.sin(TARGET_LAT) * np.cos(lat) * np.cos(dlon))
x = dist * np.sin(az)      # east [km]
y = dist * np.cos(az)      # north [km]

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(x[soft], y[soft], s=14, color="#1f5fa8", alpha=0.6, label="soft")
ax.scatter(x[~soft], y[~soft], s=20, color="#c0392b", marker="x", label="hard / no land")
ax.plot(0, 0, "*", color="#f1c40f", ms=18, mec="k", label="target")
th = np.linspace(0, 2 * np.pi, 200)
ax.plot(cep * np.cos(th), cep * np.sin(th), color="#c0392b", ls=":", lw=1.2,
        label=f"CEP {cep:.0f} km")
ax.set_xlabel("east  [km]"); ax.set_ylabel("north  [km]")
ax.set_title(f"landing dispersion  (N={n})")
ax.set_aspect("equal"); ax.grid(alpha=0.25); ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig("fig_dispersion.png", dpi=150)

# ---- figure 3: touchdown vertical velocity ----------------------------

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(v_vert, bins=40, color="#1f5fa8", alpha=0.8)
ax.axvline(V_LIMIT, color="#c0392b", ls="--", lw=1.5, label=f"limit {V_LIMIT:.0f} m/s")
ax.set_xlabel("touchdown vertical velocity  [m/s]")
ax.set_ylabel("count")
ax.set_title(f"touchdown v_vert distribution  (N={n})")
ax.grid(alpha=0.25); ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("fig_vvert_hist.png", dpi=150)

print("saved: fig_success_vs_radius.png, fig_dispersion.png, fig_vvert_hist.png")
plt.show()