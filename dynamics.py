# Equations of motion shared by every vehicle (relay, lander, rover)

import numpy as np


class Dynamics:
    # Equations of motion. Every acceleration and torque acting on a vehicle is
    # summed in derivative(), so the force model can be read off in one place.

    # Two state layouts:
    #     7   [r, v, m]              thrust direction set directly by guidance
    #     14  [r, v, m, q, omega]    thrust follows the body axis, so the vehicle
    #                                has to rotate itself first

    # Modelled: point-mass gravity, main engine, rigid-body rotation under RCS
    # torque. Not modelled: mascons, third-body, , gravity-gradient
    # torque, slosh, and the inertia change as the tanks drain within one run.

    def __init__(self, moon, spacecraft=None, use_j2=True, use_srp=True,
                 accel_bias=None):        
        self.moon = moon
        self.spacecraft = spacecraft
        self.use_j2 = use_j2
        self.use_srp = use_srp
        if use_j2:
            from config import MoonConstants
            self._j2 = MoonConstants.J2
        if use_srp:
            from config import PropulsionConfig, SolarConstants
            # P = flux / c, in N/m^2
            self._srp_p = SolarConstants.FLUX_1AU / 299792458.0
            self._cr = PropulsionConfig.CR_SRP
            self._area = PropulsionConfig.AREA_SRP
            self.accel_bias = (np.asarray(accel_bias, dtype=float)
                           if accel_bias is not None else None)

    def derivative(self, t, y):
        # dydt = [v, a, -mdot] or [v, a, -mdot, q_dot, omega_dot].
        from attitude import quat_derivative

        r_vec = y[0:3]
        v_vec = y[3:6]
        mass = y[6]

        craft = self.spacecraft
        coupled = (len(y) == 14 and craft is not None
                   and getattr(craft, "has_attitude", False))

        a_grav = self.moon.gravity(r_vec)
        if self.use_j2:
            a_grav = a_grav + self._a_j2(r_vec)
        if self.use_srp:
            a_grav = a_grav + self._a_srp(r_vec, t, mass)
        if self.accel_bias is not None:
            a_grav = a_grav + self.accel_bias
        a_thrust = np.zeros(3)
        mdot = 0.0

        if craft is not None and hasattr(craft, "throttle"):
            if coupled:
                # thrust along the body axis, wherever it now is
                f = craft.thrust_magnitude(mass)
                if f > 0.0:
                    a_thrust = (f / mass) * craft.thrust_direction_from_attitude(y[7:11])
                mdot = craft.mass_flow_rate(mass)
            else:
                a_thrust = craft.thrust_acceleration(mass)
                mdot = craft.mass_flow_rate(mass)

        dydt = np.empty_like(y)
        dydt[0:3] = v_vec
        dydt[3:6] = a_grav + a_thrust
        dydt[6] = -mdot

        if coupled:
            q = y[7:11]
            w = y[11:14]
            dydt[7:11] = quat_derivative(q, w)
            # Euler equations in body axes
            dydt[11:14] = craft.inertia_inv @ (craft.torque
                                               - np.cross(w, craft.inertia @ w))

        return dydt
    
    def _a_j2(self, r_vec):
        # j2 zonal perturbation in the inertial frame (z = moon spin axis).
        # axially symmetric, so it needs no body-fixed rotation.
        mu = self.moon.mu
        R = self.moon.radius
        x, y, z = r_vec
        r = np.linalg.norm(r_vec)
        f = 1.5 * self._j2 * mu * R * R / r**5
        zz = 5.0 * z * z / (r * r)
        return f * np.array([x * (zz - 1.0),
                             y * (zz - 1.0),
                             z * (zz - 3.0)])

    def _a_srp(self, r_vec, t, mass):
        # solar radiation pressure, pushing along sun -> spacecraft.
        # zero in eclipse. P*A is in newtons; /mass gives m/s^2; *1e-3 -> km/s^2.
        if not self.moon.is_sunlit(r_vec, t):
            return np.zeros(3)
        s = self.moon.sun_direction_mci(t)
        a_mag = self._srp_p * self._cr * self._area / mass * 1.0e-3
        return a_mag * s
    
    def specific_energy(self, y):
        # Specific mechanical energy [km^2/s^2]. Exactly conserved with the engine off,
        # so its drift measures integration error.
        r_vec = y[0:3]
        v_vec = y[3:6]
        return 0.5 * np.dot(v_vec, v_vec) - self.moon.mu / np.linalg.norm(r_vec)

    def angular_momentum(self, y):
        # Specific orbital angular momentum [km^2/s]. Conserved for a central force.
        return np.cross(y[0:3], y[3:6])

    def rotational_energy(self, y):
        # Rotational kinetic energy [J], or None if attitude is not in the state.
        # Conserved without torque, so it checks the integration the same way.
        if len(y) != 14 or self.spacecraft is None:
            return None
        w = y[11:14]
        return 0.5 * float(w @ (self.spacecraft.inertia @ w))