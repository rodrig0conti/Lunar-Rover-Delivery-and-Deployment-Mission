# standalone check that the j2 term is correct: magnitude, direction, and
# the secular nodal regression against the closed-form rate.

import numpy as np

from config import MoonConstants, OrbitConfig
from moon_environment import Moon
from dynamics import Dynamics
from simulation import Simulation


def raan_of(r, v):
    # right ascension of the ascending node from the state vector
    h = np.cross(r, v)
    n = np.cross([0.0, 0.0, 1.0], h)       # node vector
    return np.arctan2(n[1], n[0])


def main():
    moon = Moon()
    mu, R, J2 = moon.mu, moon.radius, MoonConstants.J2

    # ---- 1. magnitude and direction, at pdi altitude on the equator -----
    dyn = Dynamics(moon, None, use_j2=True)
    r_eq = np.array([R + 12.0, 0.0, 0.0])          # z = 0, equatorial
    a_j2 = dyn._a_j2(r_eq)
    print(f" 1. magnitude  |a_j2| = {np.linalg.norm(a_j2):.3e} km/s^2 "
          f"(expect ~1e-7)")
    print(f"    direction  a_j2 = [{a_j2[0]:+.3e} {a_j2[1]:+.3e} {a_j2[2]:+.3e}]")
    print(f"    at z=0 it must point radially inward (-x here): "
          f"{'ok' if a_j2[0] < 0 and abs(a_j2[1]) < 1e-20 else 'WRONG'}")

    # ---- 2. secular nodal regression, i = 45 deg ------------------------
    # the polar mission orbit barely precesses (cos 89.5 ~ 0), so the test
    # uses 45 deg where the effect is large enough to measure cleanly.
    a = OrbitConfig.SMA
    inc = np.deg2rad(45.0)
    v_circ = np.sqrt(mu / a)
    r0 = np.array([a, 0.0, 0.0])
    v0 = v_circ * np.array([0.0, np.cos(inc), np.sin(inc)])
    state0 = np.array([*r0, *v0, 1.0])             # dummy mass, engine off

    period = 2.0 * np.pi * np.sqrt(a**3 / mu)
    n_orbits = 15
    sim = Simulation(moon, dyn, dt=1.0, log_every=50)
    h = sim.run(state0, t_final=n_orbits * period, verbose=False)

    raan = np.unwrap([raan_of(h.r[k], h.v[k]) for k in range(len(h.t))])
    slope = np.polyfit(h.t, raan, 1)[0]            # rad/s, measured

    n_mean = np.sqrt(mu / a**3)
    analytic = -1.5 * J2 * (R / a)**2 * n_mean * np.cos(inc)   # rad/s

    print(f"\n 2. nodal regression at i = 45 deg over {n_orbits} orbits")
    print(f"    measured  dRAAN/dt = {np.rad2deg(slope)*86400:+.4f} deg/day")
    print(f"    analytic  dRAAN/dt = {np.rad2deg(analytic)*86400:+.4f} deg/day")
    err = abs(slope - analytic) / abs(analytic)
    print(f"    relative error = {err:.2%}  "
          f"{'ok' if err < 0.02 else 'CHECK'}")
    
      # ---- 3. srp magnitude, direction, eclipse ---------------------------
    dyn_srp = Dynamics(moon, None, use_j2=False, use_srp=True)
    t = 0.0
    s = moon.sun_direction_mci(t)
    r_test = (R + 100.0) * s          # put the test point on the sunlit side
    mass = 465.0
    print(f"\n 3. srp check")
    if moon.is_sunlit(r_test, t):
        a = dyn_srp._a_srp(r_test, t, mass)
        print(f"    |a_srp| = {np.linalg.norm(a):.3e} km/s^2 (expect ~1e-10)")
        print(f"    aligned with sun: {'ok' if np.dot(a, s) > 0 else 'WRONG'}")
    else:
        print(f"    still in eclipse — check is_sunlit / sun_direction")


if __name__ == "__main__":
    main()