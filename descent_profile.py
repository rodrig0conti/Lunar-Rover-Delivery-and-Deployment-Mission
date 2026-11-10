# descent profile figure for the report: altitude vs time, phases marked.
# run after a nominal descent to have h_lander and guidance in scope,
# or adapt to read from a saved run.

import numpy as np
import matplotlib.pyplot as plt

from config import OrbitConfig, PropulsionConfig
from moon_environment import Moon
from spacecraft import Spacecraft
from mission import run_mission

moon = Moon()
relay = Spacecraft("Relay", mass=250.0)
lander = Spacecraft("Lander", mass=PropulsionConfig.MASS_WET)
for c in (relay, lander):
    c.set_orbit(moon.mu, OrbitConfig.SMA, OrbitConfig.ECC,
                OrbitConfig.INC, OrbitConfig.RAAN,
                OrbitConfig.AOP, OrbitConfig.TA)
lander.set_propulsion(PropulsionConfig.MASS_DRY, PropulsionConfig.THRUST_MAX,
                      PropulsionConfig.ISP, PropulsionConfig.THROTTLE_MIN)

_, _, h, guidance, t_sep = run_mission(
    moon, relay, lander,
    n_orbits_before_sep=OrbitConfig.N_ORBITS_BEFORE_SEP, verbose=False)

# time since separation, in minutes
t = (h.t - t_sep) / 60.0
alt = h.altitude

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(t, alt, color="#1f5fa8", lw=2)

# mark phase transitions from guidance events
colors = {"deorbit": "#888", "coast": "#2a9d8f", "braking": "#e76f51",
          "terminal": "#9b5de5", "landed": "#222"}
for e in guidance.events:
    tt = (e["t"] - t_sep) / 60.0
    ax.axvline(tt, color=colors.get(e["phase"], "#aaa"), ls="--", lw=1, alpha=0.7)
    ax.text(tt, alt.max()*0.95, e["phase"], rotation=90,
            va="top", ha="right", fontsize=8, color=colors.get(e["phase"], "#aaa"))

ax.set_xlabel("time since separation [min]")
ax.set_ylabel("altitude [km]")
ax.set_title("Descent profile: altitude vs time")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("fig_descent_profile.png", dpi=150)
print("saved fig_descent_profile.png")
plt.show()