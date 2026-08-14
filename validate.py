"""
validate.py
===========
Verification of the propagator against cases with known answers.

This exists because "the animation looks right" is not evidence. In a systems
engineering context, any requirement verified by analysis is only as credible
as the tool used, so the tool itself needs a documented verification record.
Run this, put the table in the report appendix, and the simulator becomes an
admissible verification method rather than a nice picture.

    python validate.py
"""

import numpy as np

from config import OrbitConfig
from moon import Moon
from spacecraft import Spacecraft
from dynamics import Dynamics
from simulation import Simulation
from mathutils import elements_to_state, state_to_elements

RESULTS = []


def record(name, value, expected, tol, unit=""):
    ok = abs(value - expected) <= tol
    RESULTS.append((name, value, expected, tol, unit, ok))
    return ok


def _propagate(enable_j2, n_orbits=3.0, dt=1.0):
    moon = Moon(enable_j2=enable_j2)
    craft = Spacecraft()
    dyn = Dynamics(moon, craft)
    r0, v0 = elements_to_state(moon.mu, OrbitConfig.SMA, OrbitConfig.ECC,
                               OrbitConfig.INC, OrbitConfig.RAAN,
                               OrbitConfig.AOP, OrbitConfig.TA)
    sim = Simulation(moon, craft, dyn, dt=dt)
    period = moon.orbital_period(OrbitConfig.SMA)
    hist = sim.run(craft.initial_state(r0, v0), n_orbits * period,
                   verbose=False)
    return moon, hist, period


# ---------------------------------------------------------------------------
# Test 1: element round-trip
# ---------------------------------------------------------------------------

def test_element_roundtrip():
    """Elements -> state -> elements must return the original elements."""
    mu = Moon().mu
    sma, ecc, inc = 2200.0, 0.15, np.deg2rad(75.0)
    raan, aop, ta = np.deg2rad(40.0), np.deg2rad(25.0), np.deg2rad(130.0)

    r, v = elements_to_state(mu, sma, ecc, inc, raan, aop, ta)
    el = state_to_elements(mu, r, v)

    record("Round-trip: SMA", el["sma"], sma, 1e-6, "km")
    record("Round-trip: ECC", el["ecc"], ecc, 1e-9, "-")
    record("Round-trip: INC", np.rad2deg(el["inc"]), 75.0, 1e-7, "deg")
    record("Round-trip: RAAN", np.rad2deg(el["raan"]), 40.0, 1e-7, "deg")
    record("Round-trip: AOP", np.rad2deg(el["aop"]), 25.0, 1e-7, "deg")
    record("Round-trip: TA", np.rad2deg(el["ta"]), 130.0, 1e-7, "deg")


# ---------------------------------------------------------------------------
# Test 2: two-body conservation
# ---------------------------------------------------------------------------

def test_two_body_conservation():
    """
    With J2 off, specific energy, angular momentum and every element except
    true anomaly are exact invariants. Their drift is pure integration error.
    """
    moon, hist, _ = _propagate(enable_j2=False)

    record("Two-body: max |dE/E0|", hist.energy_error.max(), 0.0, 1e-11, "-")

    h = np.cross(hist.r, hist.v)
    h_mag = np.linalg.norm(h, axis=1)
    record("Two-body: |dh/h0|",
           abs(h_mag[-1] - h_mag[0]) / h_mag[0], 0.0, 1e-12, "-")

    el0, elf = hist.elements_at(0), hist.elements_at(-1)
    record("Two-body: SMA drift", abs(elf["sma"] - el0["sma"]), 0.0, 1e-6, "km")
    record("Two-body: INC drift",
           np.rad2deg(abs(elf["inc"] - el0["inc"])), 0.0, 1e-8, "deg")
    record("Two-body: RAAN drift",
           np.rad2deg(abs(elf["raan"] - el0["raan"])), 0.0, 1e-8, "deg")


# ---------------------------------------------------------------------------
# Test 3: circular orbit period
# ---------------------------------------------------------------------------

def test_orbital_period():
    """
    Time to return to the initial position must match the Keplerian period.
    Detected by looking for the sign change in the radial component of the
    position relative to its start direction.
    """
    moon, hist, period = _propagate(enable_j2=False, n_orbits=1.2, dt=0.5)

    r0_hat = hist.r[0] / np.linalg.norm(hist.r[0])
    proj = hist.r @ r0_hat

    # First maximum of the projection after the start, i.e. one full lap
    start = int(0.5 * period / (hist.t[1] - hist.t[0]))
    k = start + int(np.argmax(proj[start:]))
    record("Circular period", hist.t[k], period, 1.0, "s")

    record("Circular speed", hist.speed.mean(),
           moon.circular_velocity(OrbitConfig.SMA), 1e-9, "km/s")
    record("Circular altitude spread",
           hist.altitude.max() - hist.altitude.min(), 0.0, 1e-6, "km")


# ---------------------------------------------------------------------------
# Test 4: J2 nodal regression against the secular analytic rate
# ---------------------------------------------------------------------------

def test_j2_nodal_regression():
    """
    The classical first-order secular rate is

        dRAAN/dt = -1.5 * J2 * n * (R/p)^2 * cos(i)

    Agreement to better than a per cent is a strong check that the J2 term is
    correctly signed, scaled and oriented.
    """
    moon, hist, period = _propagate(enable_j2=True, n_orbits=5.0)

    n = 2.0 * np.pi / period
    p = OrbitConfig.SMA * (1.0 - OrbitConfig.ECC**2)
    rate_analytic = (-1.5 * moon.j2 * n * (moon.radius / p) ** 2
                     * np.cos(OrbitConfig.INC))

    el0, elf = hist.elements_at(0), hist.elements_at(-1)
    draan = np.arctan2(np.sin(elf["raan"] - el0["raan"]),
                       np.cos(elf["raan"] - el0["raan"]))
    rate_sim = draan / (hist.t[-1] - hist.t[0])

    rel_err = abs(rate_sim - rate_analytic) / abs(rate_analytic)
    record("J2 nodal rate (analytic)", rate_analytic * 1e9, rate_analytic * 1e9,
           np.inf, "nrad/s")
    record("J2 nodal rate (simulated)", rate_sim * 1e9, rate_analytic * 1e9,
           abs(rate_analytic) * 1e9 * 0.02, "nrad/s")
    record("J2 nodal rate: rel. error", rel_err, 0.0, 0.02, "-")


# ---------------------------------------------------------------------------
# Test 5: frame transformations
# ---------------------------------------------------------------------------

def test_frames():
    """MCI <-> MCMF must be an exact inverse pair, and preserve length."""
    moon = Moon()
    rng = np.random.default_rng(7)

    max_rt, max_len = 0.0, 0.0
    for _ in range(200):
        t = rng.uniform(0.0, 3.0 * 86400.0)
        v = rng.normal(size=3) * 2000.0
        back = moon.to_mci(moon.to_mcmf(v, t), t)
        max_rt = max(max_rt, np.linalg.norm(back - v))
        max_len = max(max_len, abs(np.linalg.norm(moon.to_mcmf(v, t))
                                   - np.linalg.norm(v)))

    record("Frame round-trip error", max_rt, 0.0, 1e-9, "km")
    record("Frame norm preservation", max_len, 0.0, 1e-9, "km")

    # A surface point must keep a constant body-fixed position at all times.
    lat, lon = np.deg2rad(-89.0), np.deg2rad(0.0)
    p_bf = moon.surface_point_mcmf(lat, lon)
    err = max(np.linalg.norm(moon.to_mcmf(moon.surface_point_mci(lat, lon, t), t)
                             - p_bf)
              for t in np.linspace(0.0, 5.0 * 86400.0, 50))
    record("Surface point fixed in MCMF", err, 0.0, 1e-9, "km")


# ---------------------------------------------------------------------------

def main():
    for test in (test_element_roundtrip, test_two_body_conservation,
                 test_orbital_period, test_j2_nodal_regression, test_frames):
        test()

    width = 78
    print("=" * width)
    print(" PROPAGATOR VERIFICATION REPORT")
    print("=" * width)
    print(f" {'Check':<34}{'Value':>14}{'Tolerance':>14}{'Unit':>8}  {'':<4}")
    print("-" * width)

    n_fail = 0
    for name, value, expected, tol, unit, ok in RESULTS:
        n_fail += (not ok)
        tol_s = "-" if not np.isfinite(tol) else f"{tol:.2e}"
        print(f" {name:<34}{value:>14.6g}{tol_s:>14}{unit:>8}  "
              f"{'PASS' if ok else 'FAIL':<4}")

    print("-" * width)
    print(f" {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed.")
    print("=" * width)
    return n_fail


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
