# Spacecraft class (relay and lunar lander) and its state

import numpy as np
from mathutils import elements_to_state, mean_motion


class Spacecraft:
    # A vehicle in orbit. State in MCI:
    #     y = [rx, ry, rz, vx, vy, vz, m]      km, km/s, kg

    # Mass is inside the integrated state: during a burn the acceleration depends
    # on the mass and the mass flow on the thrust, so both must advance together.

    # With set_attitude_control the state grows to fourteen elements,
    #     y = [r, v, m, qw, qx, qy, qz, wx, wy, wz]
    # q body-to-inertial, omega in body axes. Dynamics picks the layout by length.

    def __init__(self, name, mass, mesh_path=None, colour="#cccccc"):
        self.name = name
        self.mass = mass
        self.mesh_path = mesh_path
        self.colour = colour
        self.state = np.zeros(7)

    def set_orbit(self, mu, sma, ecc, inc, raan, aop, ta):
        # Initialise the state from Keplerian elements.
        r, v = elements_to_state(mu, sma, ecc, inc, raan, aop, ta)
        self.state = np.concatenate([r, v, [self.mass]])
        self._mu, self._sma, self._ta0 = mu, sma, ta
        self._inc, self._raan, self._aop, self._ecc = inc, raan, aop, ecc
        return self.state

    def state_at(self, t):
        # Closed-form state at time t, exact for ecc = 0. Reference for the integrator.
        ta = self._ta0 + mean_motion(self._mu, self._sma) * t
        r, v = elements_to_state(self._mu, self._sma, self._ecc,
                                 self._inc, self._raan, self._aop, ta)
        return np.concatenate([r, v, [self.mass]])

    @property
    def position(self):
        return self.state[0:3]

    @property
    def velocity(self):
        return self.state[3:6]

    def __repr__(self):
        return f"<Spacecraft '{self.name}': m={self.mass:.1f} kg>"
    
    # Propulsion

    def set_propulsion(self, mass_dry, thrust_max, isp,
                       throttle_min=0.30):
        # Attach an engine. Units: kg, kN, s.
        from config import PropulsionConfig
        self.mass_dry = mass_dry
        self.thrust_max = thrust_max
        self.isp = isp
        self.throttle_min = throttle_min
        self.v_exhaust = isp * PropulsionConfig.G0    # [km/s]

        # commanded by guidance each step, zero means coasting
        self.throttle = 0.0
        self.thrust_direction = np.array([1.0, 0.0, 0.0])

    def thrust_magnitude(self, mass):
        # Delivered thrust [kN]. Zero with dry tanks or below the deep-throttling
        # limit, where a real engine cannot be commanded.
        if mass <= self.mass_dry + 1e-9 or self.throttle <= 0.0:
            return 0.0
        return float(np.clip(self.throttle, self.throttle_min, 1.0)) * self.thrust_max

    def thrust_acceleration(self, mass):
        # Engine acceleration in MCI [km/s^2].
        f = self.thrust_magnitude(mass)
        if f == 0.0:
            return np.zeros(3)
        d = np.asarray(self.thrust_direction, dtype=float)
        n = np.linalg.norm(d)
        return (f / mass) * (d / n) if n > 1e-12 else np.zeros(3)

    def mass_flow_rate(self, mass):
        # Propellant consumption [kg/s], positive. mdot = F / (Isp g0).
        f = self.thrust_magnitude(mass)
        return f / self.v_exhaust if f > 0.0 else 0.0

    def delta_v_capability(self, mass_current=None):
        # Remaining delta-v [km/s] from Tsiolkovsky: dv = ve ln(m0/m_dry).
        m0 = self.mass if mass_current is None else mass_current
        return self.v_exhaust * np.log(m0 / self.mass_dry)

    def thrust_to_weight(self, mass, g_local):
        # Thrust over local weight. Must exceed 1 to decelerate; braking needs more
        # than that, or gravity losses grow.
        return self.thrust_magnitude(mass) / (mass * g_local)

    def set_command(self, throttle, direction):
        # Apply a guidance command.
        self.throttle = float(np.clip(throttle, 0.0, 1.0))
        self.thrust_direction = np.asarray(direction, dtype=float)

    def coast(self):
        # Shut the engine down.
        self.throttle = 0.0

    # Attitude coupling, optional. Without it the state is 7 elements and
    # guidance assigns the thrust direction; with it the state is 14 and the
    # thrust direction is read off the body attitude.

    has_attitude = False        # class default, so getattr checks are safe

    def set_attitude_control(self, inertia, rcs, controller,
                             body_thrust_axis=(0.0, 0.0, 1.0)):
        # Attach rotational dynamics and a controller.
        #     inertia          3x3 tensor in body axes [kg m^2], from inertia.py
        #     rcs              ThrusterCluster from rcs.py
        #     controller       AttitudeController from control.py
        #     body_thrust_axis engine push direction, +Z here (engine at -Z)
        self.inertia = np.asarray(inertia, dtype=float)
        self.inertia_inv = np.linalg.inv(self.inertia)
        self.rcs = rcs
        self.controller = controller
        self.body_thrust_axis = np.asarray(body_thrust_axis, dtype=float)
        self.torque = np.zeros(3)       # body axes [N m], set each step
        self.has_attitude = True

    def initial_state_full(self, q=None, omega=None):
        # Assemble [r, v, m, q, omega] from the orbit already set by set_orbit.
        from attitude import quat_identity, quat_normalise
        q = quat_identity() if q is None else quat_normalise(
            np.asarray(q, dtype=float))
        w = np.zeros(3) if omega is None else np.asarray(omega, dtype=float)
        return np.concatenate([self.state[0:7], q, w])

    def thrust_direction_from_attitude(self, q):
        # Where the engine actually points, in inertial axes.
        from attitude import rotate_body_to_inertial
        return rotate_body_to_inertial(q, self.body_thrust_axis)