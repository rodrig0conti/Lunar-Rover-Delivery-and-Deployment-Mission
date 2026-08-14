"""
config.py
=========
Central repository of physical constants, environment parameters and mission
configuration. Nothing in this file computes anything: it only declares values.

Unit convention for the whole simulator
---------------------------------------
    length      km
    time        s
    velocity    km/s
    mass        kg
    angle       rad (degrees only at the user interface level)

Rationale: with a lunar radius of ~1737 km and orbital velocities of ~1.6 km/s,
working in kilometres keeps every state variable within a few orders of
magnitude of unity, which is numerically better conditioned than SI metres.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Universal constants
# ---------------------------------------------------------------------------

G = 6.67430e-20  # universal gravitational constant [km^3 / (kg s^2)]

# ---------------------------------------------------------------------------
# Moon: physical and rotational parameters
# ---------------------------------------------------------------------------


class MoonConstants:
    """Physical parameters of the Moon (values from JPL DE440 / GRAIL)."""

    # Gravity
    MU = 4902.800118          # gravitational parameter [km^3/s^2]
    MASS = MU / G             # mass [kg]

    # Shape
    RADIUS_MEAN = 1737.4      # volumetric mean radius [km]
    RADIUS_EQ = 1738.1        # equatorial radius [km]
    RADIUS_POL = 1736.0       # polar radius [km]
    FLATTENING = (RADIUS_EQ - RADIUS_POL) / RADIUS_EQ

    # Zonal harmonics (unnormalised). J2 is the dominant non-spherical term.
    J2 = 2.0330530e-4
    J3 = 8.4759000e-6

    # Rotation. The Moon is in 1:1 spin-orbit resonance with the Earth, so its
    # sidereal rotation period equals its orbital period around the Earth.
    SIDEREAL_PERIOD = 27.321661 * 86400.0        # [s]
    OMEGA = 2.0 * np.pi / SIDEREAL_PERIOD        # spin rate [rad/s]

    # Obliquity of the lunar equator to the ecliptic. Kept here for later
    # illumination modelling; not used by the current dynamics.
    OBLIQUITY = np.deg2rad(1.54)

    # Surface gravity magnitude at the mean radius [km/s^2]
    G_SURFACE = MU / RADIUS_MEAN**2


# ---------------------------------------------------------------------------
# Mission configuration: initial orbit (Phase 1 of the mission phase breakdown)
# ---------------------------------------------------------------------------


class OrbitConfig:
    """Initial parking orbit, per the mission definition."""

    ALTITUDE = 100.0                                  # [km]
    SMA = MoonConstants.RADIUS_MEAN + ALTITUDE        # semi-major axis [km]
    ECC = 0.0                                         # circular
    INC = np.deg2rad(86.0)                            # frozen-ish polar orbit
    RAAN = np.deg2rad(0.0)                            # right ascension of AN
    AOP = np.deg2rad(0.0)                             # argument of perilune
    TA = np.deg2rad(0.0)                              # true anomaly at t0


# ---------------------------------------------------------------------------
# Mission configuration: target landing site
# ---------------------------------------------------------------------------


class SiteConfig:
    """Designated landing region, expressed in the body-fixed frame."""

    NAME = "Shackleton rim (south pole, high-illumination terrain)"
    LATITUDE = np.deg2rad(-89.0)
    LONGITUDE = np.deg2rad(0.0)
    ALTITUDE = 0.0  # relative to the mean radius [km]


# ---------------------------------------------------------------------------
# Spacecraft configuration
# ---------------------------------------------------------------------------


class SpacecraftConfig:
    """
    Mass and propulsion baseline for the descent module.

    These are Phase A placeholder values, deliberately round. They exist so the
    simulator can be exercised end to end; they are not a mass budget.
    """

    MASS_WET = 170.0        # total mass in lunar orbit [kg]
    MASS_DRY = 88.0         # mass after all propellant is spent [kg]

    THRUST_MAX = 0.800      # maximum thrust [kN]  (= 800 N)
    THROTTLE_MIN = 0.30     # deep-throttling limit [-]
    ISP = 300.0             # specific impulse [s]
    G0 = 9.80665e-3         # standard gravity for Isp definition [km/s^2]

    # Reference area and inertia are placeholders for later phases.
    AREA_REF = 2.0          # [m^2]
    INERTIA = np.diag([60.0, 60.0, 45.0])  # [kg m^2]


# ---------------------------------------------------------------------------
# Numerical / simulation settings
# ---------------------------------------------------------------------------


class SimConfig:
    """Integration and logging settings."""

    DT = 1.0                    # nominal integration step [s]
    T_FINAL = 3.0 * 7200.0      # default run duration [s] (~3 orbits)
    INTEGRATOR = "rk4"          # "rk4" | "rk45"
    RTOL = 1e-9                 # relative tolerance (adaptive schemes only)
    ATOL = 1e-12                # absolute tolerance (adaptive schemes only)

    # Environment model switches. Turning J2 off recovers pure two-body motion,
    # which is the case with a closed-form solution and therefore the natural
    # validation baseline.
    ENABLE_J2 = True
    ENABLE_THIRD_BODY = False   # Earth/Sun perturbations, not yet implemented

    LOG_EVERY = 1               # store one sample every N steps
