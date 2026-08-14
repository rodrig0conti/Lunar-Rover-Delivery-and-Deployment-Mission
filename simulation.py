"""
simulation.py
=============
The propagation loop and the results container.

The `Simulation` class owns the time loop and nothing else: it does not know
what a gravity field is, only that `Dynamics` can hand it a derivative. The
`History` class owns the recorded data and the derived quantities computed
from it. Keeping them apart means the same history object can be produced by
a different integrator, or loaded from disk, and the visualiser will not care.
"""

import time as _time

import numpy as np

from config import SimConfig
from mathutils import rk4_step, rk45_step, state_to_elements


class History:
    """Time series of the run, plus the quantities derived from it."""

    def __init__(self, moon, spacecraft):
        self.moon = moon
        self.spacecraft = spacecraft

        self.t = []
        self.states = []
        self.energy = []

    def append(self, t: float, state: np.ndarray, energy: float) -> None:
        self.t.append(t)
        self.states.append(np.array(state, copy=True))
        self.energy.append(energy)

    def finalise(self) -> None:
        """Freeze the lists into arrays and precompute derived series."""
        self.t = np.asarray(self.t)
        self.states = np.asarray(self.states)
        self.energy = np.asarray(self.energy)

        self.r = self.states[:, 0:3]
        self.v = self.states[:, 3:6]
        self.mass = self.states[:, 6]

        self.radius = np.linalg.norm(self.r, axis=1)
        self.speed = np.linalg.norm(self.v, axis=1)
        self.altitude = self.radius - self.moon.radius

        # Ground track and body-fixed trajectory
        n = len(self.t)
        self.lat = np.empty(n)
        self.lon = np.empty(n)
        self.r_mcmf = np.empty((n, 3))

        for i in range(n):
            lat, lon, _ = self.moon.geodetic(self.r[i], self.t[i])
            self.lat[i] = lat
            self.lon[i] = lon
            self.r_mcmf[i] = self.moon.to_mcmf(self.r[i], self.t[i])

        # Energy conservation diagnostic, relative to the initial value
        e0 = self.energy[0]
        self.energy_error = np.abs((self.energy - e0) / e0)

        # Propellant expended
        self.propellant_used = self.mass[0] - self.mass[-1]

    def elements_at(self, index: int) -> dict:
        """Osculating Keplerian elements at a given sample."""
        return state_to_elements(self.moon.mu, self.r[index], self.v[index])

    def summary(self) -> str:
        """Human-readable run summary."""
        el0 = self.elements_at(0)
        elf = self.elements_at(-1)

        lines = [
            "=" * 62,
            " SIMULATION SUMMARY",
            "=" * 62,
            f" Duration                 : {self.t[-1] / 3600.0:>10.3f} h "
            f"({len(self.t)} samples)",
            f" Altitude   initial/final : {self.altitude[0]:>10.3f} / "
            f"{self.altitude[-1]:.3f} km",
            f" Altitude   min/max       : {self.altitude.min():>10.3f} / "
            f"{self.altitude.max():.3f} km",
            f" Speed      initial/final : {self.speed[0]:>10.4f} / "
            f"{self.speed[-1]:.4f} km/s",
            "-" * 62,
            f" SMA        initial/final : {el0['sma']:>10.3f} / "
            f"{elf['sma']:.3f} km",
            f" Ecc        initial/final : {el0['ecc']:>10.3e} / "
            f"{elf['ecc']:.3e}",
            f" Inc        initial/final : {np.rad2deg(el0['inc']):>10.4f} / "
            f"{np.rad2deg(elf['inc']):.4f} deg",
            f" RAAN       initial/final : {np.rad2deg(el0['raan']):>10.4f} / "
            f"{np.rad2deg(elf['raan']):.4f} deg",
            "-" * 62,
            f" Propellant used          : {self.propellant_used:>10.4f} kg",
            f" Max relative energy error: {self.energy_error.max():>10.3e}",
            "=" * 62,
        ]
        return "\n".join(lines)


class Simulation:
    """Fixed- or adaptive-step propagation of the vehicle state."""

    def __init__(self, moon, spacecraft, dynamics,
                 dt: float = SimConfig.DT,
                 integrator: str = SimConfig.INTEGRATOR,
                 log_every: int = SimConfig.LOG_EVERY):

        self.moon = moon
        self.spacecraft = spacecraft
        self.dynamics = dynamics
        self.dt = dt
        self.integrator = integrator.lower()
        self.log_every = log_every

        # Termination conditions, evaluated once per step.
        # Each entry is (name, callable(t, state) -> bool).
        self.stop_conditions = []
        self.stop_reason = None

        self.add_stop_condition(
            "surface impact",
            lambda t, s: np.linalg.norm(s[0:3]) <= self.moon.radius,
        )

    def add_stop_condition(self, name: str, condition) -> None:
        """Register a termination condition."""
        self.stop_conditions.append((name, condition))

    def _check_stop(self, t: float, state: np.ndarray):
        for name, cond in self.stop_conditions:
            if cond(t, state):
                return name
        return None

    def run(self, state0: np.ndarray, t_final: float,
            t0: float = 0.0, verbose: bool = True) -> History:
        """
        Propagate from `state0` at `t0` until `t_final` or a stop condition.

        Returns the populated (and finalised) History.
        """
        history = History(self.moon, self.spacecraft)

        t = t0
        state = np.array(state0, dtype=float)
        dt = self.dt
        step = 0
        self.stop_reason = None

        history.append(t, state, self.dynamics.energy(state))
        wall_start = _time.perf_counter()

        while t < t_final - 1e-12:
            dt_step = min(dt, t_final - t)

            if self.integrator == "rk4":
                state = rk4_step(self.dynamics.derivative, t, state, dt_step)
                t += dt_step
            elif self.integrator == "rk45":
                state_new, err, dt_next = rk45_step(
                    self.dynamics.derivative, t, state, dt_step,
                    SimConfig.RTOL, SimConfig.ATOL)
                if err > 1.0:
                    dt = max(dt_next, 1e-3)
                    continue  # reject the step and retry with a smaller one
                state = state_new
                t += dt_step
                dt = dt_next
            else:
                raise ValueError(f"Unknown integrator: '{self.integrator}'")

            step += 1

            reason = self._check_stop(t, state)
            if reason is not None:
                self.stop_reason = reason
                history.append(t, state, self.dynamics.energy(state))
                break

            if step % self.log_every == 0:
                history.append(t, state, self.dynamics.energy(state))

        # Guarantee the final state is recorded.
        if history.t[-1] < t:
            history.append(t, state, self.dynamics.energy(state))

        history.finalise()
        wall = _time.perf_counter() - wall_start

        if verbose:
            print(f"Propagated {t - t0:.1f} s of mission time in "
                  f"{wall:.2f} s wall clock ({step} steps, "
                  f"integrator '{self.integrator}').")
            if self.stop_reason:
                print(f"Stopped early: {self.stop_reason}.")

        return history
