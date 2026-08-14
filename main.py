"""
main.py
=======
Entry point. Assembles the model, propagates it and produces the outputs.

Step 1 scope
------------
Central body with rotation and gravity (J2 optional), spacecraft state
propagated in the inertial frame, no engine firing. The purpose of this step
is to establish the frames, the integration loop and the visualisation, and to
verify them against conserved quantities before any manoeuvre is introduced.

Usage
-----
    python main.py                      # 3 orbits, plots to screen
    python main.py --orbits 5           # longer run
    python main.py --animate            # animation window
    python main.py --animate --save out.gif
    python main.py --no-j2              # pure two-body (validation case)
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from config import MoonConstants, OrbitConfig, SimConfig, SiteConfig
from moon import Moon
from spacecraft import Spacecraft
from dynamics import Dynamics
from simulation import Simulation
from mathutils import elements_to_state
import visualization as viz


def build_model(enable_j2: bool = True):
    """Instantiate the Moon, the vehicle and the force model."""
    moon = Moon(enable_j2=enable_j2)
    craft = Spacecraft()
    dyn = Dynamics(moon, craft)
    return moon, craft, dyn


def initial_conditions(moon, craft):
    """
    Build the initial state from the parking-orbit definition in config.

    The state is generated from Keplerian elements rather than hard-coded
    vectors, so changing the orbit is a one-line edit in config.py.
    """
    r0, v0 = elements_to_state(
        mu=moon.mu,
        sma=OrbitConfig.SMA,
        ecc=OrbitConfig.ECC,
        inc=OrbitConfig.INC,
        raan=OrbitConfig.RAAN,
        aop=OrbitConfig.AOP,
        ta=OrbitConfig.TA,
    )
    return craft.initial_state(r0, v0)


def print_header(moon, craft):
    period = moon.orbital_period(OrbitConfig.SMA)
    v_circ = moon.circular_velocity(OrbitConfig.SMA)

    print("=" * 62)
    print(" LUNAR DELIVERY SYSTEM — DESCENT SIMULATOR")
    print(" Step 1: central body dynamics and orbital propagation")
    print("=" * 62)
    print(f" Central body   : Moon, mu = {moon.mu:.3f} km^3/s^2, "
          f"R = {moon.radius:.1f} km")
    print(f" Spin period    : {MoonConstants.SIDEREAL_PERIOD / 86400.0:.4f} "
          f"days  (omega = {moon.omega:.4e} rad/s)")
    print(f" J2 oblateness  : {'ON' if moon.enable_j2 else 'OFF'}")
    print("-" * 62)
    print(f" Parking orbit  : {OrbitConfig.ALTITUDE:.1f} km circular, "
          f"i = {np.rad2deg(OrbitConfig.INC):.1f} deg")
    print(f" Orbital period : {period / 3600.0:.4f} h ({period:.1f} s)")
    print(f" Orbital speed  : {v_circ:.4f} km/s")
    print(f" Landing site   : {SiteConfig.NAME}")
    print(f"                  lat {np.rad2deg(SiteConfig.LATITUDE):.2f} deg, "
          f"lon {np.rad2deg(SiteConfig.LONGITUDE):.2f} deg")
    print("-" * 62)
    print(f" Vehicle        : {craft!r}")
    print("=" * 62)
    return period


def main():
    parser = argparse.ArgumentParser(description="Lunar descent simulator")
    parser.add_argument("--orbits", type=float, default=3.0,
                        help="run duration in orbital periods")
    parser.add_argument("--dt", type=float, default=SimConfig.DT,
                        help="integration step [s]")
    parser.add_argument("--integrator", default=SimConfig.INTEGRATOR,
                        choices=["rk4", "rk45"])
    parser.add_argument("--no-j2", action="store_true",
                        help="disable J2 (pure two-body validation case)")
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--save", default=None,
                        help="save the animation to this GIF path")
    parser.add_argument("--spin-exaggeration", type=float, default=1.0,
                        help="display-only multiplier on the lunar spin rate")
    parser.add_argument("--no-show", action="store_true",
                        help="do not open figure windows")
    args = parser.parse_args()

    # --- Model -------------------------------------------------------
    moon, craft, dyn = build_model(enable_j2=not args.no_j2)
    period = print_header(moon, craft)

    # --- Initial conditions and propagation --------------------------
    state0 = initial_conditions(moon, craft)
    sim = Simulation(moon, craft, dyn, dt=args.dt,
                     integrator=args.integrator)

    history = sim.run(state0, t_final=args.orbits * period)
    print()
    print(history.summary())

    # --- Output ------------------------------------------------------
    viz.plot_trajectory_3d(history, moon)
    viz.plot_groundtrack(history)
    viz.plot_diagnostics(history)

    anim = None
    if args.animate or args.save:
        _, anim = viz.animate_orbit(
            history, moon,
            spin_exaggeration=args.spin_exaggeration,
            filename=args.save,
        )

    if not args.no_show:
        plt.show()

    return history, anim


if __name__ == "__main__":
    main()
