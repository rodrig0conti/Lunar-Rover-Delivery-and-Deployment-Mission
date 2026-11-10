# Math helpers: orbital elements, rotations, RK4

import numpy as np


def perifocal_to_inertial(inc, raan, aop):
    # Rotation matrix perifocal -> MCI.
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(aop), np.sin(aop)
    return np.array([
        [cO*cw - sO*sw*ci, -cO*sw - sO*cw*ci,  sO*si],
        [sO*cw + cO*sw*ci, -sO*sw + cO*cw*ci, -cO*si],
        [sw*si,             cw*si,             ci],
    ])


def elements_to_state(mu, sma, ecc, inc, raan, aop, ta):
    # Keplerian elements -> inertial r, v. Angles in rad, sma in km, mu in km^3/s^2.
    p = sma * (1.0 - ecc**2)
    r = p / (1.0 + ecc * np.cos(ta))

    r_pf = np.array([r * np.cos(ta), r * np.sin(ta), 0.0])
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(ta), ecc + np.cos(ta), 0.0])

    q = perifocal_to_inertial(inc, raan, aop)
    return q @ r_pf, q @ v_pf


def mean_motion(mu, sma):
    # Mean motion [rad/s].
    return np.sqrt(mu / sma**3)


def orbital_period(mu, sma):
    # Keplerian period [s].
    return 2.0 * np.pi / mean_motion(mu, sma)

def rk4_step(f, t, y, dt):
    # One classical RK4 step. f(t, y) -> dy/dt, fixed step, four evaluations.
    # Fourth order: halving dt cuts the error by about sixteen.
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def osculating_elements(mu, r_vec, v_vec):
    # Keplerian elements the vehicle would keep if all forces but gravity stopped
    # now. Used as a burn cut-off: stop once the current ellipse has the perilune
    # wanted. Returns sma, ecc, r_peri, r_apo [km].
    r = np.linalg.norm(r_vec)
    h_vec = np.cross(r_vec, v_vec)
    e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r
    ecc = np.linalg.norm(e_vec)
    sma = 1.0 / (2.0 / r - np.dot(v_vec, v_vec) / mu)
    return {"sma": sma, "ecc": ecc,
            "r_peri": sma * (1.0 - ecc),
            "r_apo": sma * (1.0 + ecc)}


def perilune_radius(mu, r_vec, v_vec):
    # Osculating perilune radius [km].
    return osculating_elements(mu, r_vec, v_vec)["r_peri"]