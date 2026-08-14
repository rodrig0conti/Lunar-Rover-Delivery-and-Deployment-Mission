"""
dynamics.py
===========
The equations of motion. This module is the single place where accelerations
are summed, which keeps the force model auditable: if an acceleration is not
listed in `derivative`, it is not acting on the vehicle.

Model currently active
----------------------
    - Central-body gravity (point mass)
    - J2 oblateness              (switchable in config)
    - Engine thrust              (present, commanded to zero in Step 1)

Not yet modelled
----------------
    - Earth and Sun third-body perturbations
    - Higher-order lunar gravity field (mascons)
    - Solar radiation pressure
    - Attitude dynamics: the vehicle is treated as a point mass and the thrust
      direction is prescribed rather than achieved through torques.
"""

import numpy as np


class Dynamics:
    """Assembles dy/dt for the translational + mass state vector."""

    def __init__(self, moon, spacecraft):
        self.moon = moon
        self.spacecraft = spacecraft

        # Diagnostic store: last computed acceleration contributions, so the
        # visualiser can show their relative magnitudes without recomputing.
        self.last_accelerations = {}

    def derivative(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Time derivative of the state vector.

        y    = [r (3), v (3), m (1)]
        dydt = [v (3), a (3), -mdot (1)]
        """
        r_vec = y[0:3]
        v_vec = y[3:6]
        mass = y[6]

        a_grav = self.moon.gravity(r_vec)
        a_thrust = self.spacecraft.thrust_acceleration(mass)
        a_total = a_grav + a_thrust

        mdot = self.spacecraft.mass_flow_rate(mass)

        self.last_accelerations = {
            "gravity": a_grav,
            "thrust": a_thrust,
            "total": a_total,
        }

        dydt = np.empty_like(y)
        dydt[0:3] = v_vec
        dydt[3:6] = a_total
        dydt[6] = -mdot
        return dydt

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    def energy(self, state: np.ndarray) -> float:
        """
        Specific mechanical energy [km^2/s^2].

        Under point-mass gravity with the engine off this is exactly conserved,
        so its drift over a run is a direct measure of integration error. It is
        the cheapest validation available and worth logging always.
        """
        r_vec = state[0:3]
        v_vec = state[3:6]
        return 0.5 * np.dot(v_vec, v_vec) + self.moon.gravity_potential(r_vec)

    def angular_momentum(self, state: np.ndarray) -> np.ndarray:
        """Specific angular momentum vector [km^2/s]. Conserved when J2 is off."""
        return np.cross(state[0:3], state[3:6])
