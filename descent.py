# Parking orbit to touchdown.

# DescentGuidance is a phase state machine. Simulation hands it the time and
# state on every step; it picks the phase and issues a throttle and a pointing
# requirement. It owns no physics, so the same guidance can later be flown
# against a different force model without changing it.

# With a 7-element state the thrust direction is assigned outright. With a
# 14-element state guidance can only ask the controller to point the body, and
# the thrust goes wherever the body ends up. The gap between the two is the
# cost of pretending attitude is free.

# Phases:
#     DEORBIT   full throttle retrograde until the osculating perilune reaches
#               the target altitude
#     COAST     engine off, ballistic arc to perilune. Attitude held inertially
#               and slewed to retrograde only near the end
#     BRAKING   full throttle retrograde. No pitch profile is scheduled: as
#               speed bleeds off gravity rotates the velocity vector and the
#               thrust follows it, so the gravity turn emerges on its own
#     TERMINAL  thrust along the local vertical, throttle tracking a descent
#               rate that tapers towards the ground
#     LANDED    engine off

# Transitions are recorded in events, which is what turns a run into a timeline.

import numpy as np

from mathutils import perilune_radius, osculating_elements, orbital_period


class DescentGuidance:
    # Phase state machine and guidance law for the powered descent.

    DEORBIT = "deorbit"
    COAST = "coast"
    BRAKING = "braking"
    TERMINAL = "terminal"
    LANDED = "landed"

    def __init__(self, moon, spacecraft,
                 h_pdi=12.0,
                 v_horz_pitchover=3.0e-3,
                 touchdown_alt=5.0e-4,
                 descent_rate_gain=1.2,
                 descent_rate_k=0.04,
                 v_touchdown=1.0e-3,
                 start_phase=DEORBIT,
                 dt_hint=0.05,
                 slew_lead_time=400.0,
                 perilune_ceiling=40.0):
        # Parameters in km, km/s, s.
        #     v_horz_pitchover  horizontal speed below which braking is complete
        #     descent_rate_k    shapes the terminal profile,
        #                       v_target = -(v_touchdown + k sqrt(altitude)). Square
        #                       root is the shape of a constant-deceleration stop,
        #                       softened near the ground by the v_touchdown floor
        #     dt_hint           the integration step, needed because the controller
        #                       integrates thruster on-time. Must match Simulation.dt
        #     slew_lead_time    how long before perilune to start pointing retrograde
        self.moon = moon
        self.craft = spacecraft
        self.h_pdi = h_pdi
        self.r_pdi = moon.radius + h_pdi
        self.v_horz_pitchover = v_horz_pitchover
        self.touchdown_alt = touchdown_alt
        self.gain = descent_rate_gain
        self.k = descent_rate_k
        self.v_touchdown = v_touchdown
        self.dt_hint = dt_hint
        self.slew_lead_time = slew_lead_time
        self.h_pdi_ceiling = perilune_ceiling


        self.phase = start_phase
        self._hold_quat = None
        self.events = []
        self._logged = set()

    def _log(self, t, state, note=""):
        if self.phase in self._logged:
            return
        self._logged.add(self.phase)
        r = np.linalg.norm(state[0:3])
        self.events.append({
            "t": t, "phase": self.phase,
            "altitude": r - self.moon.radius,
            "speed": np.linalg.norm(state[3:6]),
            "mass": state[6],
            "note": note,
        })

    def _set_phase(self, phase, t, state, note=""):
        self.phase = phase
        self._log(t, state, note)

    def _time_to_pdi(self, r_vec, v_vec):
        # Rough time to perilune from the osculating orbit, through the eccentric
        # anomaly. Only decides when to start the slew, so it need not be exact.
        mu = self.moon.mu
        el = osculating_elements(mu, r_vec, v_vec)
        a, e = el["sma"], el["ecc"]
        if a <= 0.0:
            return None
        r = np.linalg.norm(r_vec)
        cos_E = np.clip((1.0 - r / a) / max(e, 1e-12), -1.0, 1.0)
        E = np.arccos(cos_E)
        if np.dot(r_vec, v_vec) < 0.0:          # descending: E in (pi, 2pi)
            E = 2.0 * np.pi - E
        M = E - e * np.sin(E)
        n = np.sqrt(mu / a**3)
        return (2.0 * np.pi - M) / n

    

    def __call__(self, t, state):
        # Called by Simulation before every integration step.
        r_vec, v_vec, mass = state[0:3], state[3:6], state[6]
        r = np.linalg.norm(r_vec)
        up = r_vec / r
        v_vert = float(v_vec @ up)
        v_horz = float(np.linalg.norm(v_vec - v_vert * up))
        alt = r - self.moon.radius

        self._log(t, state)

        # Coupled: guidance only states a pointing requirement, the thrust
        # goes wherever the body actually is.
        coupled = (len(state) == 14
                   and getattr(self.craft, "has_attitude", False))

        def aim(direction, throttle):
            self.craft.throttle = float(np.clip(throttle, 0.0, 1.0))
            if coupled:
                self.craft.controller.set_target_direction(direction)
                tau, _ = self.craft.controller.update(state[7:11],
                                                      state[11:14],
                                                      self.dt_hint)
                self.craft.torque = tau
                self.craft.rcs.step(self.dt_hint)
            else:
                self.craft.thrust_direction = np.asarray(direction, dtype=float)

        def hold_inertial():
            # Engine off, attitude actively held at a fixed inertial target.
            # Zeroing the torque would only freeze the applied torque, not the motion: the
            # body would keep its residual rate and drift round for the whole coast. The
            # target is captured once, so the controller damps the rate and parks there.
            self.craft.throttle = 0.0
            if not coupled:
                return
            if self._hold_quat is None:
                self._hold_quat = state[7:11].copy()
            self.craft.controller.set_target_quat(self._hold_quat)
            tau, _ = self.craft.controller.update(state[7:11], state[11:14],
                                                  self.dt_hint)
            self.craft.torque = tau
            self.craft.rcs.step(self.dt_hint)

        if self.phase == self.DEORBIT:
            if perilune_radius(self.moon.mu, r_vec, v_vec) <= self.r_pdi:
                self.craft.coast()
                self._set_phase(self.COAST, t, state,
                                "target perilune reached, engine off")
            else:
                aim(-v_vec / np.linalg.norm(v_vec), 1.0)
            return

        if self.phase == self.COAST:

            # Held inertially, slew to retrograde only shortly before PDI.
            # Held inertially, slew to retrograde only shortly before PDI.
            # Tracking retrograde for the whole coast measured 62 kg of RCS
            # propellant against under 1 kg for a single slew.
            if coupled:
                t_to_pdi = self._time_to_pdi(r_vec, v_vec)
                if t_to_pdi is not None and t_to_pdi < self.slew_lead_time:
                    self._hold_quat = None          # release the hold target
                    aim(-v_vec / np.linalg.norm(v_vec), 0.0)
                else:
                    hold_inertial()
            else:
                self.craft.coast()

            # PDI on arrival at the target altitude while descending
            # pdi at the target altitude while descending. if j2 or later the
            # monte-carlo dispersions raise the real perilune above h_pdi, the
            # altitude test never trips, so fall back to perilune passage:
            # brake at the lowest point, where the velocity is horizontal
            # anyway. the ceiling keeps this from firing at the 100 km burn
            # point, where v_vert is also ~0.
            reached_pdi_alt = alt <= self.h_pdi and v_vert < 0.0
            past_perilune = v_vert >= 0.0 and alt < self.h_pdi_ceiling
            if reached_pdi_alt or past_perilune:
                self._set_phase(self.BRAKING, t, state, "PDI")
            return

        if self.phase == self.BRAKING:
            if v_horz < self.v_horz_pitchover:
                self._set_phase(self.TERMINAL, t, state,
                                "horizontal velocity nulled")
            else:
                aim(-v_vec / np.linalg.norm(v_vec), 1.0)
                return

        if self.phase == self.TERMINAL:
            if alt <= self.touchdown_alt:
                self.craft.coast()
                self._set_phase(self.LANDED, t, state, "touchdown")
                if coupled:
                    self.craft.torque = np.zeros(3)
                    self.craft.rcs.all_off()
                return

            g = self.moon.mu / r**2
            a_max = self.craft.thrust_max / mass
            v_target = -(self.v_touchdown + self.k * np.sqrt(max(alt, 0.0)))
            # cancel weight plus a proportional correction on descent rate.
            # Enough to size the phase, not a landing autopilot.
            throttle = (g + self.gain * (v_target - v_vert)) / a_max
            aim(up, throttle)
            return

        if self.phase == self.LANDED:
            self.craft.coast()
            if coupled:
                self.craft.torque = np.zeros(3)
                self.craft.rcs.all_off()

    def phase_time(self, phase):
        # Start time of a phase, or None if it never began.
        for e in self.events:
            if e["phase"] == phase:
                return e["t"]
        return None

    def timeline(self):
        # Formatted event timeline.
        lines = [f"   {'t [s]':>9}{'phase':>10}{'alt [km]':>11}"
                 f"{'speed [m/s]':>13}{'mass [kg]':>11}  note",
                 "   " + "-" * 68]
        for e in self.events:
            lines.append(f"   {e['t']:9.2f}{e['phase']:>10}"
                         f"{e['altitude']:11.3f}{e['speed']*1e3:13.1f}"
                         f"{e['mass']:11.2f}  {e['note']}")
        return "\n".join(lines)


# Convenience runner

def run_descent(moon, craft, dynamics, simulation,
                h_pdi=12.0, t_max=6000.0, verbose=True,
                initial_state=None):
    # Fly a complete descent from the parking orbit to touchdown.
    # craft needs set_orbit and set_propulsion first; if it also has attitude
    # control, pass the 14-element state through initial_state.
    # Returns (history, guidance).
    guidance = DescentGuidance(moon, craft, h_pdi=h_pdi,
                               dt_hint=simulation.dt)
    simulation.guidance = guidance
    simulation.add_stop_condition(
        "landed", lambda t, s: guidance.phase == DescentGuidance.LANDED)

    y0 = craft.state if initial_state is None else initial_state
    history = simulation.run(y0, t_final=t_max, verbose=verbose)
    return history, guidance


def descent_report(moon, craft, history, guidance, m0):
    # Printable summary of a descent run.
    m_end = history.mass[-1]
    prop = m0 - m_end
    dv = craft.v_exhaust * np.log(m0 / m_end)

    r_vec, v_vec = history.r[-1], history.v[-1]
    r = np.linalg.norm(r_vec)
    up = r_vec / r
    v_vert = float(v_vec @ up)
    v_horz = float(np.linalg.norm(v_vec - v_vert * up))

    lat, lon, alt = moon.geodetic(r_vec, history.t[-1])

    # downrange: central angle between the start and end position vectors
    r0 = history.r[0] / np.linalg.norm(history.r[0])
    ang = np.arccos(np.clip(np.dot(r0, up), -1.0, 1.0))

    out = ["", "=" * 72, " DESCENT RUN", "=" * 72, "",
           guidance.timeline(), "",
           "   " + "-" * 68,
           f"   total duration        {history.t[-1]:10.2f} s "
           f"({history.t[-1]/60:.2f} min)",
           f"   propellant used       {prop:10.2f} kg",
           f"   delta-v expended      {dv*1e3:10.1f} m/s",
           f"   propellant fraction   {prop/m0:10.2%}",
           f"   mass at touchdown     {m_end:10.2f} kg",
           f"   dv still available    {craft.delta_v_capability(m_end)*1e3:10.1f} m/s",
           "   " + "-" * 68,
           f"   touchdown altitude    {alt*1e3:10.2f} m",
           f"   vertical velocity     {v_vert*1e3:10.2f} m/s",
           f"   horizontal velocity   {v_horz*1e3:10.2f} m/s",
           f"   latitude              {np.rad2deg(lat):10.3f} deg",
           f"   longitude             {np.rad2deg(lon):10.3f} deg",
           f"   downrange from start  {ang*moon.radius:10.1f} km "
           f"({np.rad2deg(ang):.2f} deg)"]

    if getattr(craft, "has_attitude", False):
        out += ["   " + "-" * 68,
                f"   RCS propellant        {craft.rcs.propellant_used:10.3f} kg",
                f"   RCS firing time       {craft.controller.firing_time:10.1f} s",
                f"   final pointing error  "
                f"{np.rad2deg(craft.controller.last_error_angle):10.4f} deg"]

    out.append("=" * 72)
    return "\n".join(out)


if __name__ == "__main__":
    from config import OrbitConfig, PropulsionConfig
    from moon_environment import Moon
    from spacecraft import Spacecraft
    from dynamics import Dynamics
    from simulation import Simulation
    from rcs import standard_layout
    from control import AttitudeController, attitude_for_pointing
    from inertia import build_descent_module

    moon = Moon()
    craft = Spacecraft("Descent Module", mass=PropulsionConfig.MASS_WET)
    craft.set_orbit(moon.mu, OrbitConfig.SMA, OrbitConfig.ECC,
                    OrbitConfig.INC, OrbitConfig.RAAN,
                    OrbitConfig.AOP, OrbitConfig.TA)
    craft.set_propulsion(PropulsionConfig.MASS_DRY,
                         PropulsionConfig.THRUST_MAX,
                         PropulsionConfig.ISP,
                         PropulsionConfig.THROTTLE_MIN)

    # attitude: inertia from the breakdown, RCS from the standard layout
    inertia = build_descent_module(PropulsionConfig.MASS_WET
                                   - PropulsionConfig.MASS_DRY).inertia()
    rcs = standard_layout()
    controller = AttitudeController(rcs, inertia, deadband_deg=0.2)
    craft.set_attitude_control(inertia, rcs, controller,
                               body_thrust_axis=(0.0, 0.0, 1.0))

    # start pointing retrograde so the deorbit burn begins at once
    v0 = craft.state[3:6]
    q0 = attitude_for_pointing(-v0 / np.linalg.norm(v0))
    y0 = craft.initial_state_full(q=q0)

    dyn = Dynamics(moon, craft)
    sim = Simulation(moon, dyn, dt=0.05, log_every=20)

    m0 = craft.state[6]
    hist, guid = run_descent(moon, craft, dyn, sim, h_pdi=12.0,
                             initial_state=y0)
    print(descent_report(moon, craft, hist, guid, m0))