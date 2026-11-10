
#File for physical constant, parameters and mission configuration


import numpy as np


class MoonConstants:
    MU = 4902.800118                       # gravitational parameter [km^3/s^2]
    RADIUS_MEAN = 1737.4                   # mean radius [km]

    SIDEREAL_PERIOD = 27.321661 * 86400.0  # rotation period [s] (~27.3 days)
    OMEGA = 2 * np.pi / SIDEREAL_PERIOD    # spin rate [rad/s]
    J2 = 2.034e-4                          # second zonal harmonic (GRAIL)

class DisplayConfig:
    # Display only, never used in physics.
    # TIME_SCALE = mission seconds per wall-clock second (1 = real time).
    
    SPACECRAFT_DISPLAY_SIZE = 70.0    # [km] largest dimension on screen, inflated
    TIME_SCALE = 1.0
    FPS = 30
    # Gap between the attached vehicles, as a multiple of their display size.
    ATTACHED_OFFSET_FACTOR = 0.3


class OrbitConfig:
    ALTITUDE = 100.0
    SMA = MoonConstants.RADIUS_MEAN + ALTITUDE
    ECC = 0.0
    INC = np.deg2rad(89.5)          # reaches the 89 S site; 86 deg cannot
    AOP = 0.0
    TA = np.deg2rad(84)           #108.0 - 23.6 to reach -89 with J2

    # Orbits flown attached before the lander is released.
    N_ORBITS_BEFORE_SEP = 1.0

    # RAAN pre-biased by the Moon rotation during the orbits flown before
    # separation.
    PERIOD = 2 * np.pi * np.sqrt(SMA**3 / MoonConstants.MU)
    RAAN = (np.deg2rad(31.64)      # was 30.525, +2.7 to trim lon to ~0e
            + MoonConstants.OMEGA * N_ORBITS_BEFORE_SEP * PERIOD)
class SiteConfig:
    LATITUDE = np.deg2rad(-89.0)
    LONGITUDE = np.deg2rad(0.0)


class SolarConstants:
    FLUX_1AU = 1361.0
    YEAR = 365.25 * 86400.0
    OMEGA = 2.0 * np.pi / YEAR
    OBLIQUITY = np.deg2rad(1.54)

    # Solar longitude at t = 0, chosen to put the Sun as close as possible to
    # the orbit normal (beta = 88.2 deg, no eclipse). 1.8 deg of residual is
    # unavoidable: the orbit normal is fixed by the landing-site phasing.
    RA0 = np.deg2rad(121.619)


class PropulsionConfig:
    MASS_WET = 465.0        # [kg]
    MASS_DRY = 240.0        # [kg]
    THRUST_MAX = 2.59       # [kN]  T/W = 3.50 at PDI
    THROTTLE_MIN = 0.30
    ISP = 300.0
    G0 = 9.80665e-3
    CR_SRP = 1.3           # radiation-pressure coefficient
    AREA_SRP = 7.0         # m^2, sun-facing area

def descent_module_inertia(propellant_mass=None):
    # Inertia tensor [kg m^2] for a given propellant load. See inertia.py.
    from inertia import build_descent_module
    if propellant_mass is None:
        propellant_mass = PropulsionConfig.MASS_WET - PropulsionConfig.MASS_DRY
    return build_descent_module(propellant_mass).inertia()