# Illumination geometry for power-system sizing. Run: python power_geometry.py

#     1. eclipse per orbit, now and over the coming weeks  -> battery capacity
#     2. solar aspect during the descent                   -> array placement
#     3. energy drawn from battery while off-sun           -> battery capacity
#     4. surface illumination at the landing site          -> lander array size

# No spacecraft attitude is assumed except where written out, since array
# output means nothing without it.

import numpy as np

from config import (MoonConstants, OrbitConfig, PropulsionConfig,
                    SolarConstants)
from moon_environment import Moon
from spacecraft import Spacecraft
from mission import run_mission

# Loads. Replace with the values from the power budget appendix.

LOADS = {
    "relay_peak": 0.91,       # kW
    "relay_avg": 0.37,        # kW
    "lander_peak": 0.64,      # kW
    "lander_avg": 0.11,       # kW
    "lander_descent": 0.64,   # kW, assumed at peak throughout the descent
}

SITE_LAT = np.deg2rad(-89.0)
SITE_LON = np.deg2rad(0.0)


def rule(c="=", n=72):
    print(c * n)


def beta_angle(moon, h_hat, t):
    # Angle between the orbit plane and the Sun. 90 deg puts the Sun on the orbit
    # normal and removes eclipse entirely; 0 gives maximum eclipse.
    s = moon.sun_direction_mci(t)
    return np.pi / 2.0 - np.arccos(np.clip(abs(np.dot(s, h_hat)), -1.0, 1.0))


def eclipse_fraction(moon, r_orbit, beta):
    # Fraction of a circular orbit spent in shadow. Cylindrical umbra, so the
    # penumbra (a couple of seconds per pass) is ignored.
    x = np.sqrt(r_orbit**2 - moon.radius**2) / (r_orbit * np.cos(beta))
    if x >= 1.0:
        return 0.0
    return np.arccos(x) / np.pi


def main():
    moon = Moon()
    period = OrbitConfig.PERIOD
    r_orbit = OrbitConfig.SMA

    i, raan = OrbitConfig.INC, OrbitConfig.RAAN
    h_hat = np.array([np.sin(i) * np.sin(raan),
                      -np.sin(i) * np.cos(raan),
                      np.cos(i)])

    rule()
    print(" POWER SYSTEM — ILLUMINATION GEOMETRY")
    rule()
    print(f"\n   orbit            {OrbitConfig.ALTITUDE:.0f} km circular, "
          f"i = {np.rad2deg(i):.2f} deg, RAAN = {np.rad2deg(raan):.2f} deg")
    print(f"   period           {period/60:.2f} min")
    print(f"   orbit normal     [{h_hat[0]:+.4f} {h_hat[1]:+.4f} {h_hat[2]:+.4f}]")

    print()
    rule("-")
    print(" 1. ECLIPSE AND ITS EVOLUTION")
    rule("-")

    b0 = beta_angle(moon, h_hat, 0.0)
    print(f"\n   beta at epoch    {np.rad2deg(b0):.3f} deg")
    print(f"   eclipse-free requires |beta| > "
          f"{np.rad2deg(np.arcsin(np.sqrt(r_orbit**2 - moon.radius**2)/r_orbit)):.2f} deg\n")

    print(f"   {'days':>7}{'beta [deg]':>13}{'eclipse frac':>15}"
          f"{'eclipse [min]':>15}{'sunlit [min]':>14}")
    print("   " + "-" * 62)
    for d in (0, 7, 14, 30, 60, 90, 120, 150, 180):
        t = d * 86400.0
        b = beta_angle(moon, h_hat, t)
        f = eclipse_fraction(moon, r_orbit, b)
        print(f"   {d:7d}{np.rad2deg(b):13.3f}{f:15.4f}"
              f"{f*period/60:15.2f}{(1-f)*period/60:14.2f}")

    # worst case over a full year
    ts = np.linspace(0.0, SolarConstants.YEAR, 4000)
    fracs = np.array([eclipse_fraction(moon, r_orbit,
                                       beta_angle(moon, h_hat, t)) for t in ts])
    fmax = fracs.max()
    print(f"\n   worst case in a year: {fmax:.4f} -> {fmax*period/60:.2f} min "
          f"per orbit")
    print(f"   battery for worst-case eclipse at relay average load: "
          f"{LOADS['relay_avg']*fmax*period/3600:.3f} kWh")
    print(f"   battery for worst-case eclipse at relay peak load:    "
          f"{LOADS['relay_peak']*fmax*period/3600:.3f} kWh")
    print("\n   The orbit plane is fixed in inertial space while the Sun moves")
    print("   about 1 deg per day, so beta drifts and eclipse eventually")
    print("   returns. The epoch is chosen, not permanent.")

    print()
    rule("-")
    print(" 2. DESCENT: SOLAR ASPECT AND ARRAY PLACEMENT")
    rule("-")

    relay = Spacecraft("Relay", mass=250.0)
    lander = Spacecraft("Lander", mass=PropulsionConfig.MASS_WET)
    for c in (relay, lander):
        c.set_orbit(moon.mu, OrbitConfig.SMA, OrbitConfig.ECC,
                    OrbitConfig.INC, OrbitConfig.RAAN,
                    OrbitConfig.AOP, OrbitConfig.TA)
    lander.set_propulsion(PropulsionConfig.MASS_DRY,
                          PropulsionConfig.THRUST_MAX,
                          PropulsionConfig.ISP,
                          PropulsionConfig.THROTTLE_MIN)

    h_stack, h_relay, h_lander, guidance, t_sep = run_mission(
        moon, relay, lander,
        n_orbits_before_sep=OrbitConfig.N_ORBITS_BEFORE_SEP, verbose=False)

    def sunlit_fraction(hist):
        lit = np.array([moon.is_sunlit(hist.r[k], hist.t[k])
                        for k in range(len(hist.t))])
        return lit.mean()

    f_stack = sunlit_fraction(h_stack)
    f_desc = sunlit_fraction(h_lander)
    dur = h_lander.t[-1] - h_lander.t[0]

    print(f"\n   attached orbit    sunlit {f_stack:6.2%}  "
          f"({f_stack*period/60:.2f} of {period/60:.2f} min)")
    print(f"   descent           sunlit {f_desc:6.2%}  "
          f"({dur/60:.2f} min total)")

    # Angle between the Sun and the thrust axis, sampled along the descent
    ang = []
    for k in range(len(h_lander.t)):
        v = h_lander.v[k]
        axis = -v / np.linalg.norm(v)              # thrust points anti-velocity
        s = moon.sun_direction_mci(h_lander.t[k])
        ang.append(np.arccos(np.clip(np.dot(axis, s), -1.0, 1.0)))
    ang = np.rad2deg(np.array(ang))

    print(f"\n   Sun to thrust axis: {ang.min():.1f} to {ang.max():.1f} deg")
    print("""
   The Sun is normal to the orbit plane by choice of epoch, and thrust always
   lies inside that plane, so the two are perpendicular throughout. This is a
   configuration requirement, not a preference:

     - arrays whose normal lies along +/- the orbit normal see full Sun
       continuously, whatever attitude manoeuvres happen in-plane
     - arrays whose normal lies along the thrust axis see essentially nothing

   Body-mounted arrays should therefore face the orbit normal.""")

    print()
    rule("-")
    print(" 3. BATTERY DEMAND DURING DESCENT")
    rule("-")

    e_descent = LOADS["lander_descent"] * dur / 3600.0
    print(f"\n   descent duration        {dur/60:8.2f} min")
    print(f"   assumed load            {LOADS['lander_descent']:8.2f} kW")
    print(f"   energy required         {e_descent:8.3f} kWh")
    print("""
   Whether this comes from the battery depends on array placement. With arrays
   facing the orbit normal the vehicle stays powered throughout and this is not
   a battery case. With any other placement, attitude is set by guidance rather
   than by Sun pointing and the whole descent runs on battery.""")

    print()
    rule("-")
    print(" 4. SURFACE ILLUMINATION AT THE LANDING SITE")
    rule("-")

    synodic = 29.530589 * 86400.0
    ts = np.linspace(0.0, synodic, 600)
    elev = np.array([moon.solar_elevation(SITE_LAT, SITE_LON, t) for t in ts])
    elev_deg = np.rad2deg(elev)

    horiz = np.clip(np.sin(elev), 0.0, None)      # horizontal panel
    vert = np.abs(np.cos(elev))                   # vertical, azimuth-tracking

    print(f"\n   over one synodic month ({synodic/86400:.2f} days), at "
          f"lat {np.rad2deg(SITE_LAT):.1f} deg:\n")
    print(f"   solar elevation      {elev_deg.min():+7.2f} to "
          f"{elev_deg.max():+7.2f} deg")
    print(f"   Sun above horizon    {(elev_deg > 0).mean():7.1%} of the month")
    print(f"   horizontal panel     mean yield {horiz.mean():7.2%}, "
          f"peak {horiz.max():.2%}")
    print(f"   vertical panel       mean yield {vert.mean():7.2%}, "
          f"peak {vert.max():.2%}")

    flux = SolarConstants.FLUX_1AU
    print(f"\n   at {flux:.0f} W/m2 and 30% cell efficiency:")
    print(f"     horizontal 1 m2 -> mean {flux*0.30*horiz.mean():7.1f} W")
    print(f"     vertical   1 m2 -> mean {flux*0.30*vert.mean():7.1f} W")
    print(f"""
   The Sun never rises far above the horizon at 89 deg latitude, so a
   horizontal array is close to useless. Arrays must stand vertical, facing
   the horizon, and track in azimuth: the Sun circles the sky once per
   synodic month.

   This assumes a spherical Moon with a flat horizon. Real terrain at the
   Shackleton rim raises the horizon in some directions and lowers it in
   others, which is the difference between near-permanent illumination on a
   ridge and permanent darkness in a crater floor a few km away. Quantifying
   that needs the LOLA elevation map, and is the outstanding work behind any
   illumination percentage quoted in the report.""")

    print()
    rule()
    print(" SUMMARY FOR THE POWER BUDGET")
    rule()
    print(f"""
   Relay
     eclipse at epoch            {eclipse_fraction(moon, r_orbit, b0)*period/60:.2f} min per orbit
     eclipse worst case in a year{fmax*period/60:7.2f} min per orbit
     battery, worst case at avg  {LOADS['relay_avg']*fmax*period/3600:.3f} kWh
     array must recharge within  {(1-fmax)*period/60:.2f} min of sunlight

   Lander, in flight
     descent duration            {dur/60:.2f} min
     energy at peak load         {e_descent:.3f} kWh
     arrays face the orbit normal for continuous Sun

   Lander, on the surface
     vertical array mean yield   {vert.mean():.1%}
     horizontal array mean yield {horiz.mean():.1%}
     azimuth tracking required over the {synodic/86400:.1f} day synodic month
""")
    rule()


if __name__ == "__main__":
    main()