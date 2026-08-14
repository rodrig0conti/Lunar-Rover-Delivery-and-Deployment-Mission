"""
mathutils.py
============
Pure numerical machinery: integrators and coordinate/element conversions.
Nothing in this module knows anything about the Moon or the spacecraft; it
operates on generic state vectors and callables. Keeping it physics-agnostic
means it can be unit-tested against problems with known analytic solutions.
"""

import numpy as np

# ===========================================================================
# Integrators
# ===========================================================================


def rk4_step(f, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """
    One step of the classical fourth-order Runge-Kutta scheme.

    Parameters
    ----------
    f  : callable f(t, y) -> dy/dt
    t  : current time
    y  : current state vector
    dt : step size

    Fourth-order accurate, fixed step, four function evaluations. It is the
    right default here: cheap, well understood, and with a 1 s step the energy
    drift over a lunar orbit is negligible compared with modelling error.
    """
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk45_step(f, t: float, y: np.ndarray, dt: float,
              rtol: float = 1e-9, atol: float = 1e-12):
    """
    One step of Runge-Kutta-Fehlberg 4(5) with an embedded error estimate.

    Returns
    -------
    y_new     : state advanced by dt (fifth-order solution)
    err       : estimated local error norm, scaled by the tolerances
    dt_next   : suggested next step size
    """
    a = [0.0, 1 / 4, 3 / 8, 12 / 13, 1.0, 1 / 2]
    b = [
        [],
        [1 / 4],
        [3 / 32, 9 / 32],
        [1932 / 2197, -7200 / 2197, 7296 / 2197],
        [439 / 216, -8.0, 3680 / 513, -845 / 4104],
        [-8 / 27, 2.0, -3544 / 2565, 1859 / 4104, -11 / 40],
    ]
    c4 = np.array([25 / 216, 0.0, 1408 / 2565, 2197 / 4104, -1 / 5, 0.0])
    c5 = np.array([16 / 135, 0.0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55])

    k = []
    for i in range(6):
        yi = y.copy()
        for j, bij in enumerate(b[i]):
            yi = yi + dt * bij * k[j]
        k.append(f(t + a[i] * dt, yi))
    k = np.array(k)

    y4 = y + dt * np.tensordot(c4, k, axes=(0, 0))
    y5 = y + dt * np.tensordot(c5, k, axes=(0, 0))

    scale = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
    err = np.sqrt(np.mean(((y5 - y4) / scale) ** 2))

    if err == 0.0:
        factor = 5.0
    else:
        factor = 0.9 * err ** (-0.2)
    dt_next = dt * min(5.0, max(0.2, factor))

    return y5, err, dt_next


# ===========================================================================
# Orbital elements
# ===========================================================================


def elements_to_state(mu: float, sma: float, ecc: float, inc: float,
                      raan: float, aop: float, ta: float):
    """
    Classical Keplerian elements -> inertial position and velocity.

    Angles in rad, sma in km, mu in km^3/s^2.
    Returns (r_vec, v_vec) in km and km/s.
    """
    p = sma * (1.0 - ecc**2)
    r = p / (1.0 + ecc * np.cos(ta))

    # Perifocal frame
    r_pf = np.array([r * np.cos(ta), r * np.sin(ta), 0.0])
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(ta), ecc + np.cos(ta), 0.0])

    q = _perifocal_to_inertial(inc, raan, aop)
    return q @ r_pf, q @ v_pf


def _perifocal_to_inertial(inc: float, raan: float, aop: float) -> np.ndarray:
    """Rotation matrix from the perifocal frame to the inertial frame."""
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(aop), np.sin(aop)

    return np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si, cw * si, ci],
    ])


def state_to_elements(mu: float, r_vec: np.ndarray, v_vec: np.ndarray) -> dict:
    """
    Inertial position and velocity -> classical Keplerian elements.

    Returns a dict with sma, ecc, inc, raan, aop, ta (angles in rad).
    Used as a diagnostic: for an unperturbed orbit these must stay constant,
    which is a direct check on the integrator.
    """
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)

    n_vec = np.cross([0.0, 0.0, 1.0], h_vec)
    n = np.linalg.norm(n_vec)

    e_vec = (np.cross(v_vec, h_vec) / mu) - r_vec / r
    ecc = np.linalg.norm(e_vec)

    energy = 0.5 * v**2 - mu / r
    sma = -mu / (2.0 * energy)

    inc = np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0))

    if n > 1e-12:
        raan = np.arccos(np.clip(n_vec[0] / n, -1.0, 1.0))
        if n_vec[1] < 0.0:
            raan = 2.0 * np.pi - raan
    else:
        raan = 0.0

    if n > 1e-12 and ecc > 1e-12:
        aop = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * ecc), -1.0, 1.0))
        if e_vec[2] < 0.0:
            aop = 2.0 * np.pi - aop
    else:
        aop = 0.0

    if ecc > 1e-12:
        ta = np.arccos(np.clip(np.dot(e_vec, r_vec) / (ecc * r), -1.0, 1.0))
        if np.dot(r_vec, v_vec) < 0.0:
            ta = 2.0 * np.pi - ta
    else:
        # Circular orbit: fall back to argument of latitude.
        ta = np.arctan2(np.dot(r_vec, np.cross(h_vec, n_vec)) / h,
                        np.dot(r_vec, n_vec)) if n > 1e-12 else 0.0

    return {"sma": sma, "ecc": ecc, "inc": inc,
            "raan": raan, "aop": aop, "ta": ta}


# ===========================================================================
# Small helpers
# ===========================================================================


def unit(v: np.ndarray) -> np.ndarray:
    """Unit vector along v. Returns the zero vector if v is degenerate."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-15 else np.zeros_like(v)


def wrap_pi(angle):
    """Wrap an angle to (-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def specific_energy(mu: float, r_vec: np.ndarray, v_vec: np.ndarray) -> float:
    """Specific orbital energy [km^2/s^2]: conserved under point-mass gravity."""
    return 0.5 * np.dot(v_vec, v_vec) - mu / np.linalg.norm(r_vec)
