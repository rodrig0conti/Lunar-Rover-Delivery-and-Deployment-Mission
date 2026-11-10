# Mission sequence: joint orbit, separation, then divergent trajectories.

import numpy as np

from mathutils import orbital_period
from simulation import Simulation
from dynamics import Dynamics
from descent import DescentGuidance


def run_mission(moon, relay, lander, h_pdi=12.0, n_orbits_before_sep=1.0,
                dt_coast=1.0, dt_descent=0.05, n_orbits_after=3.0,
                verbose=True, sep_dp=None, sep_dv=None, accel_bias=None):
    # Three propagations, because separation is a discontinuity in the state.

    # 1. STACK   relay and lander attached and coasting, mass is the sum, no engine
    # 2. RELAY   from the separation state, no thrust, relay mass only
    # 3. LANDER  from the same separation state, lander mass, flying the guidance

    # Both vehicles share the separation state, so what each one does afterwards is
    # the only reason their trajectories differ.

    # If the lander carries attitude its propagation runs on fourteen elements
    # while the stack and relay stay on seven; Dynamics decides by length.

    # Returns (h_stack, h_relay, h_lander, guidance, t_sep).
    r0 = np.linalg.norm(relay.state[0:3])
    period = orbital_period(moon.mu, r0)
    t_sep = n_orbits_before_sep * period

    # --- 1. stack ---------------------------------------------------
    stack_mass = relay.mass + lander.mass
    stack_state = relay.state.copy()
    stack_state[6] = stack_mass
    sim_stack = Simulation(moon, Dynamics(moon, None), dt=dt_coast)
    if t_sep > 0.0:
        h_stack = sim_stack.run(stack_state, t_final=t_sep, verbose=False)
        sep_state = h_stack.states[-1].copy()
    else:
        h_stack = None
        sep_state = stack_state

        # monte-carlo dispersions: perturb the shared separation state
    if sep_dp is not None:
        sep_state = sep_state.copy()
        sep_state[0:3] = sep_state[0:3] + np.asarray(sep_dp)
    if sep_dv is not None:
        sep_state[3:6] = sep_state[3:6] + np.asarray(sep_dv)

    if verbose:
        print(f"separation at t = {t_sep:.1f} s "
              f"({n_orbits_before_sep:.2f} orbits), "
              f"stack mass {stack_mass:.1f} kg")

    # --- 2. relay ---------------------------------------------------
    relay_state = sep_state.copy()
    relay_state[6] = relay.mass
    sim_relay = Simulation(moon, Dynamics(moon, None), dt=dt_coast)
    h_relay = sim_relay.run(relay_state,
                            t_final=t_sep + n_orbits_after * period,
                            t0=t_sep, verbose=False)

    # --- 3. lander --------------------------------------------------
    lander_state = sep_state.copy()
    lander_state[6] = lander.mass
    lander.state = lander_state

    # Coupled: fourteen-element state, released already pointing retrograde
    # so the deorbit burn can begin at once.
    if getattr(lander, "has_attitude", False):
        from control import attitude_for_pointing
        v0 = lander_state[3:6]
        q0 = attitude_for_pointing(-v0 / np.linalg.norm(v0))
        lander_state = lander.initial_state_full(q=q0)

    sim_l = Simulation(moon, Dynamics(moon, lander, accel_bias=accel_bias),
                       dt=dt_descent, log_every=20)
    # dt_hint must match dt: the controller integrates thruster on-time
    guidance = DescentGuidance(moon, lander, h_pdi=h_pdi,
                               dt_hint=dt_descent)
    sim_l.guidance = guidance
    sim_l.add_stop_condition(
        "landed", lambda t, s: guidance.phase == DescentGuidance.LANDED)
    h_lander = sim_l.run(lander_state, t_final=t_sep + 6000.0,
                         t0=t_sep, verbose=False)

    if verbose:
        print(f"relay  : parking orbit maintained, "
              f"altitude {h_relay.altitude.min():.3f} to "
              f"{h_relay.altitude.max():.3f} km")
        print(f"lander : {guidance.phase}, "
              f"{h_lander.t[-1] - t_sep:.0f} s after separation")
        if getattr(lander, "has_attitude", False):
            print(f"         RCS propellant {lander.rcs.propellant_used:.3f} kg, "
                  f"pointing error {np.rad2deg(lander.controller.last_error_angle):.4f} deg")

    return h_stack, h_relay, h_lander, guidance, t_sep