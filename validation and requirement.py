# Checks that confirm requirements and verify the model functions

import numpy as np
from moon_environment import Moon


def check_frame_roundtrip(moon, tol=1e-9):
    # MCMF -> MCI -> MCMF must return the original vector.
    p = np.array([1000.0, 500.0, -300.0])
    worst = 0.0
    for t in [0.0, 1.0e4, 1.0e5, 1.0e6]:
        back = moon.to_mcmf(moon.to_mci(p, t), t)
        worst = max(worst, np.linalg.norm(back - p))
    return "frame round-trip", worst, tol


def check_surface_point_is_fixed(moon, tol=1e-9):
    # A surface point must keep constant MCMF coordinates. If this fails, one of
    # the two rotation matrices is inverted.
    lat, lon = np.deg2rad(-89.0), 0.0
    p_bf = moon.surface_point_mcmf(lat, lon)
    worst = 0.0
    for t in np.linspace(0.0, 5.0 * 86400.0, 50):
        back = moon.to_mcmf(moon.surface_point_mci(lat, lon, t), t)
        worst = max(worst, np.linalg.norm(back - p_bf))
    return "surface point fixed in MCMF", worst, tol


def check_full_revolution(moon, tol=1e-9):
    # After one sidereal period the DCM must be the identity again.
    from config import MoonConstants
    dcm = moon.dcm_mcmf_to_mci(MoonConstants.SIDEREAL_PERIOD)
    return "full revolution -> identity", np.max(np.abs(dcm - np.eye(3))), tol


def check_geodetic(moon, tol=1e-9):
    # surface_point -> geodetic must return the latitude/longitude asked for.
    lat, lon, alt = np.deg2rad(-89.0), np.deg2rad(45.0), 100.0
    t = 3.0e5
    r_mci = moon.surface_point_mci(lat, lon, t, alt)
    lat_b, lon_b, alt_b = moon.geodetic(r_mci, t)
    err = max(abs(lat_b - lat), abs(lon_b - lon), abs(alt_b - alt))
    return "geodetic inverse", err, tol

def check_attitude(tol=1e-10):
    # Torque-free rigid body: h and E are exact invariants.
    from attitude import RigidBody, propagate_attitude
    body = RigidBody([60.0, 60.0, 45.0])
    y0 = body.initial_state(w_body=[0.02, 0.05, 0.03])
    r = propagate_attitude(body, y0, 600.0, dt=0.01, log_every=100)
    dh = np.max(np.linalg.norm(r["h_inertial"] - r["h_inertial"][0], axis=1))
    dh /= np.linalg.norm(r["h_inertial"][0])
    de = np.max(np.abs(r["energy"] - r["energy"][0])) / r["energy"][0]
    return "attitude: |dh|/|h| and |dE|/E", max(dh, de), tol


def check_gravity(moon, tol=1e-9):
    # Surface gravity should be about 1.62 m/s^2, pointing inwards.
    r = np.array([moon.radius, 0.0, 0.0])
    g = moon.gravity(r)
    g_expected = 1.62e-3          # km/s^2
    err = abs(np.linalg.norm(g) - g_expected)
    if g[0] >= 0.0:               # must point towards the centre
        err = np.inf
    return "surface gravity magnitude", err, 5e-6


def run_all():
    moon = Moon()
    checks = [check_frame_roundtrip, check_surface_point_is_fixed,
              check_full_revolution, check_geodetic, check_gravity,check_attitude]

    print("=" * 58)
    print(" MOON ENVIRONMENT VERIFICATION")
    print("=" * 58)
    n_fail = 0
    for check in checks:
        name, value, tol = check(moon)
        ok = value <= tol
        n_fail += (not ok)
        print(f" {name:<32}{value:>12.3e}   {'PASS' if ok else 'FAIL'}")
    print("-" * 58)
    print(f" {len(checks) - n_fail}/{len(checks)} passed")
    print("=" * 58)


if __name__ == "__main__":
    run_all()

