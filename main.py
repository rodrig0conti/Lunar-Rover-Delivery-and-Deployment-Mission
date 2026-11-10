# Initial conditions and output selection (animation, plots, data)

import numpy as np
import vtk

from config import OrbitConfig, PropulsionConfig, DisplayConfig
from moon_environment import Moon
from spacecraft import Spacecraft
from rcs import standard_layout
from control import AttitudeController
from inertia import build_descent_module
from mission import run_mission
from visualization import show_full_mission
from calculations import mission_timeline


vtk.vtkObject.GlobalWarningDisplayOff()


# Model

moon = Moon()

relay = Spacecraft("Relay", mass=250.0)
lander = Spacecraft("Lander", mass=PropulsionConfig.MASS_WET)

# Both start attached, so they share one orbit
for craft in (relay, lander):
    craft.set_orbit(moon.mu, OrbitConfig.SMA, OrbitConfig.ECC,
                    OrbitConfig.INC, OrbitConfig.RAAN,
                    OrbitConfig.AOP, OrbitConfig.TA)

lander.set_propulsion(PropulsionConfig.MASS_DRY, PropulsionConfig.THRUST_MAX,
                      PropulsionConfig.ISP, PropulsionConfig.THROTTLE_MIN)

# attitude, after set_propulsion
inertia = build_descent_module(PropulsionConfig.MASS_WET
                               - PropulsionConfig.MASS_DRY).inertia()
rcs = standard_layout()
lander.set_attitude_control(
    inertia, rcs, AttitudeController(rcs, inertia, deadband_deg=0.2))


# Run

h_stack, h_relay, h_lander, guidance, t_sep = run_mission(
    moon, relay, lander,
    h_pdi=12.0,
    n_orbits_before_sep=OrbitConfig.N_ORBITS_BEFORE_SEP)
mission_timeline(moon, h_stack, h_relay, h_lander, guidance, t_sep)

lat, lon, alt = moon.geodetic(h_lander.r[-1], h_lander.t[-1])
print(f"\ntouchdown:  lat {np.rad2deg(lat):+8.4f} deg"
      f"  lon {np.rad2deg(lon):+8.4f} deg"
      f"  alt {alt*1e3:+6.2f} m")


# Visualise

show_full_mission(moon, h_stack, h_relay, h_lander, guidance, t_sep,
                  site_latlon=(-89.0, 0.0),
                  texture="moon_texture.jpg",
                  relay_mesh="relay.obj",
                  lander_mesh="lander.obj",
                  display_size=DisplayConfig.SPACECRAFT_DISPLAY_SIZE,
                  offset_factor=DisplayConfig.ATTACHED_OFFSET_FACTOR,
                  frame_scale=2.0,
                  time_scale=180.0)