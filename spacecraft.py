"""
spacecraft.py
=============
The vehicle: its state, its mass properties and its propulsion model.

State vector convention (7 elements, MCI frame)
-----------------------------------------------
    y = [ rx, ry, rz, vx, vy, vz, m ]
        [ km  km  km  km/s km/s km/s  kg ]

Mass is carried inside the integrated state rather than updated separately,
because during a burn the mass flow and the acceleration are coupled: the
same integrator step must see both. In Step 1 no engine is fired, so mass is
constant and the seventh element simply stays put - but the slot is already
there, which means Phase 2 (deorbit burn) needs no structural change.
"""

import numpy as np

from config import SpacecraftConfig


class Spacecraft:
    """Point-mass vehicle with a throttleable, constant-Isp engine."""

    # Index slices into the state vector, so no other module hard-codes them.
    IDX_POS = slice(0, 3)
    IDX_VEL = slice(3, 6)
    IDX_MASS = 6
    STATE_DIM = 7

    def __init__(self,
                 mass_wet: float = SpacecraftConfig.MASS_WET,
                 mass_dry: float = SpacecraftConfig.MASS_DRY,
                 thrust_max: float = SpacecraftConfig.THRUST_MAX,
                 isp: float = SpacecraftConfig.ISP,
                 throttle_min: float = SpacecraftConfig.THROTTLE_MIN,
                 name: str = "Descent Module"):

        self.name = name
        self.mass_wet = mass_wet
        self.mass_dry = mass_dry
        self.thrust_max = thrust_max      # [kN]
        self.isp = isp                    # [s]
        self.throttle_min = throttle_min
        self.g0 = SpacecraftConfig.G0     # [km/s^2]

        # Effective exhaust velocity [km/s]
        self.v_exhaust = self.isp * self.g0

        # Commanded control, refreshed each step by the guidance layer.
        # In Step 1 both stay at their defaults and the vehicle coasts.
        self.throttle = 0.0                                  # [0, 1]
        self.thrust_direction = np.array([1.0, 0.0, 0.0])    # unit, MCI

        # Bookkeeping accumulated over a run.
        self.delta_v_used = 0.0     # [km/s]
        self.propellant_used = 0.0  # [kg]

    # -----------------------------------------------------------------
    # State construction and access
    # -----------------------------------------------------------------

    def initial_state(self, r_vec: np.ndarray, v_vec: np.ndarray,
                      mass: float = None) -> np.ndarray:
        """Assemble a state vector from position, velocity and mass."""
        m = self.mass_wet if mass is None else mass
        return np.concatenate([np.asarray(r_vec, dtype=float),
                               np.asarray(v_vec, dtype=float),
                               [m]])

    @staticmethod
    def position(state: np.ndarray) -> np.ndarray:
        return state[Spacecraft.IDX_POS]

    @staticmethod
    def velocity(state: np.ndarray) -> np.ndarray:
        return state[Spacecraft.IDX_VEL]

    @staticmethod
    def mass(state: np.ndarray) -> float:
        return state[Spacecraft.IDX_MASS]

    # -----------------------------------------------------------------
    # Propulsion
    # -----------------------------------------------------------------

    @property
    def propellant_mass(self) -> float:
        """Usable propellant at the start of the mission [kg]."""
        return self.mass_wet - self.mass_dry

    def thrust_magnitude(self, mass: float) -> float:
        """
        Delivered thrust for the current throttle command [kN].

        Returns zero if the tanks are empty or if the command sits below the
        deep-throttling limit, since a real engine cannot be commanded there.
        """
        if mass <= self.mass_dry + 1e-9:
            return 0.0
        if self.throttle <= 0.0:
            return 0.0

        cmd = np.clip(self.throttle, self.throttle_min, 1.0)
        return cmd * self.thrust_max

    def thrust_acceleration(self, mass: float) -> np.ndarray:
        """Acceleration produced by the engine, in MCI [km/s^2]."""
        f = self.thrust_magnitude(mass)
        if f == 0.0:
            return np.zeros(3)

        direction = self.thrust_direction
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            return np.zeros(3)

        return (f / mass) * (direction / norm)

    def mass_flow_rate(self, mass: float) -> float:
        """Propellant consumption rate [kg/s], returned as a positive number."""
        f = self.thrust_magnitude(mass)
        return f / self.v_exhaust if f > 0.0 else 0.0

    # -----------------------------------------------------------------
    # Performance figures
    # -----------------------------------------------------------------

    def delta_v_capability(self, mass_current: float = None) -> float:
        """
        Remaining delta-v from the Tsiolkovsky equation [km/s].

        This is the number that decides whether the descent strategy closes,
        so it is worth having available at every instant of the simulation.
        """
        m0 = self.mass_wet if mass_current is None else mass_current
        return self.v_exhaust * np.log(m0 / self.mass_dry)

    def thrust_to_weight(self, mass: float, g_local: float) -> float:
        """Thrust-to-weight ratio against the local gravity. Dimensionless."""
        return self.thrust_magnitude(mass) / (mass * g_local)

    def set_command(self, throttle: float, direction: np.ndarray) -> None:
        """Apply a guidance command. Called by the phase/guidance layer."""
        self.throttle = float(np.clip(throttle, 0.0, 1.0))
        self.thrust_direction = np.asarray(direction, dtype=float)

    def coast(self) -> None:
        """Shut the engine down."""
        self.throttle = 0.0

    def __repr__(self) -> str:
        return (f"<Spacecraft '{self.name}': "
                f"m_wet={self.mass_wet:.1f} kg, m_dry={self.mass_dry:.1f} kg, "
                f"Isp={self.isp:.0f} s, F_max={self.thrust_max * 1e3:.0f} N, "
                f"dv_max={self.delta_v_capability() * 1e3:.0f} m/s>")
