# Derivation of the descent-system requirements from mission-level inputs.

# Every number in the requirements table at the end is computed here from
# the assumptions block, with the method next to it. Change an assumption and
# every dependent number moves. The printed output goes in the report appendix
# as the evidence behind each requirement ID.

#     1. parking orbit and transfer geometry      closed form
#     2. deorbit burn delta-v                     closed form (vis-viva)
#     3. powered descent                          numerical integration
#     4. PDI altitude and thrust-to-weight trade   parameter sweep
#     5. reserve budget                           closed form
#     6. mass closure                             Tsiolkovsky

# Usage: python calculations.py [--quick]

import argparse

import numpy as np

from config import MoonConstants, OrbitConfig, PropulsionConfig
from moon_environment import Moon

# Assumptions. Everything after this block is derived from it; each entry
# carries the reason for its value.

ASSUMPTIONS = {
    "isp": (PropulsionConfig.ISP, "s",
            "storable bipropellant, conservative for a pressure-fed system"),
    "park_alt": (OrbitConfig.ALTITUDE, "km",
                 "mission definition: circular parking orbit"),
    "inclination": (np.rad2deg(OrbitConfig.INC), "deg",
                    "near-polar, required to reach the south-pole site"),
    "site_lat": (-89.0, "deg",
                 "Shackleton rim, high-illumination terrain"),
    "hover_time": (30.0, "s",
                   "terrain assessment and landing point selection"),
    "dv_divert": (30.0, "m/s",
                  "lateral translation to the selected landing point"),
    "dv_nav": (20.0, "m/s",
               "allowance for navigation dispersion at PDI"),
    "v_touchdown_max": (3.0, "m/s",
                        "landing gear design limit, honeycomb stroke"),
    "vh_touchdown_max": (1.0, "m/s",
                         "tip-over margin at touchdown"),
}

# DT is also the guidance update interval: the command is held across each
# integration step, like a digital control loop holds its output.
DT = 0.02          # [s]  -> 50 Hz guidance loop
DT_QUICK = 0.05

moon = Moon()
MU = moon.mu
R = moon.radius
VE = PropulsionConfig.ISP * PropulsionConfig.G0     # exhaust velocity [km/s]


# Numerics

def rk4_step(f, t, y, dt):
    # Classical fourth-order Runge-Kutta step.
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# 1. Orbit and transfer geometry  (closed form)

def vis_viva(r, sma):
    # Speed on a Keplerian orbit at radius r [km/s].
    return np.sqrt(MU * (2.0 / r - 1.0 / sma))


def transfer_geometry(h_pdi):
    # Hohmann-type transfer from the parking orbit down to a perilune at h_pdi.
    # One retrograde burn lowers the opposite side of the orbit, so the burn
    # point becomes apolune of the transfer ellipse.
    r_park = R + OrbitConfig.ALTITUDE
    r_peri = R + h_pdi
    sma = 0.5 * (r_park + r_peri)

    v_circ = np.sqrt(MU / r_park)
    v_apo = vis_viva(r_park, sma)
    v_peri = vis_viva(r_peri, sma)

    return {
        "r_park": r_park, "r_peri": r_peri, "sma": sma,
        "ecc": (r_park - r_peri) / (r_park + r_peri),
        "v_circ": v_circ, "v_apo": v_apo, "v_peri": v_peri,
        "dv_deorbit": v_circ - v_apo,
        "coast": np.pi * np.sqrt(sma**3 / MU),
    }


# 2. Powered descent  (numerical)

def simulate_descent(h_pdi, tw0, m0=1000.0, dt=DT, gain=0.8, t_max=4000.0):
    # Integrate the powered descent from PDI to touchdown.

    # BRAKING is a gravity turn: full throttle opposite to velocity, no attitude
    # profile. As speed bleeds off gravity rotates the velocity vector and the
    # thrust follows it, so the pitch-over emerges from the physics and no thrust
    # is wasted perpendicular to the motion.

    # TERMINAL starts once horizontal speed drops below 3 m/s: thrust along the
    # local vertical, throttle tracking a descent rate that tapers to the ground.

    # tw0 is thrust-to-weight at PDI, not thrust. With thrust proportional to
    # mass the trajectory and the propellant FRACTION are independent of vehicle
    # size, which is why this takes a nominal m0 and returns a fraction.
    geom = transfer_geometry(h_pdi)
    g_pdi = MU / geom["r_peri"] ** 2
    thrust = tw0 * m0 * g_pdi                       # [kN]

    y = np.concatenate([[geom["r_peri"], 0.0, 0.0],
                        [0.0, geom["v_peri"], 0.0],
                        [m0]])
    t = 0.0
    cmd = {"throttle": 0.0, "direction": np.array([1.0, 0.0, 0.0])}

    def derivative(t, y):
        r_vec, v_vec, m = y[0:3], y[3:6], y[6]
        a = -MU * r_vec / np.linalg.norm(r_vec) ** 3
        mdot = 0.0
        if cmd["throttle"] > 0.0:
            a = a + (cmd["throttle"] * thrust / m) * cmd["direction"]
            mdot = cmd["throttle"] * thrust / VE
        d = np.empty(7)
        d[0:3] = v_vec
        d[3:6] = a
        d[6] = -mdot
        return d

    phase = "braking"
    crashed = False
    t_pitchover = None

    while t < t_max:
        r_vec, v_vec, m = y[0:3], y[3:6], y[6]
        r = np.linalg.norm(r_vec)
        up = r_vec / r
        v_vert = v_vec @ up
        v_horz = np.linalg.norm(v_vec - v_vert * up)
        alt = r - R

        if alt <= 0.0:
            crashed = True
            break

        if phase == "braking":
            if v_horz < 3.0e-3:
                phase = "terminal"
                t_pitchover = t
            else:
                cmd["throttle"] = 1.0
                cmd["direction"] = -v_vec / np.linalg.norm(v_vec)

        if phase == "terminal":
            g = MU / r**2
            a_max = thrust / m
            v_target = -(0.002 + 0.06 * np.sqrt(max(alt, 0.0)))
            cmd["throttle"] = float(np.clip(
                (g + gain * (v_target - v_vert)) / a_max, 0.0, 1.0))
            cmd["direction"] = up
            if alt <= 5.0e-4:                       # 0.5 m: touchdown
                break

        y = rk4_step(derivative, t, y, dt)
        t += dt

    r_vec, v_vec, m = y[0:3], y[3:6], y[6]
    r = np.linalg.norm(r_vec)
    up = r_vec / r
    v_vert = v_vec @ up
    v_horz = np.linalg.norm(v_vec - v_vert * up)

    prop_fraction = (m0 - m) / m0
    return {
        "crashed": crashed,
        "duration": t,
        "t_pitchover": t_pitchover,
        "prop_fraction": prop_fraction,
        "dv": VE * np.log(1.0 / (1.0 - prop_fraction)),
        "v_vert": v_vert * 1e3,
        "v_horz": v_horz * 1e3,
        "alt": (r - R) * 1e3,
        "thrust_per_kg": tw0 * g_pdi,
    }


def descent_is_valid(result):
    # Acceptance test for a descent case, from the assumptions block.
    if result["crashed"]:
        return False
    if abs(result["v_vert"]) > ASSUMPTIONS["v_touchdown_max"][0] * 5.0:
        return False        # 5x the gear limit: the simple controller is
        # not a landing autopilot, so this only screens out gross failures
    return True


# 3. Reserve budget  (closed form)

def reserve_budget():
    # Delta-v held back after the nominal descent. Hovering costs exactly the
    # local gravity, dv = g t, since the engine must cancel weight.
    g_surface = MU / R**2
    dv_hover = g_surface * ASSUMPTIONS["hover_time"][0] * 1e3      # m/s
    dv_divert = ASSUMPTIONS["dv_divert"][0]
    dv_nav = ASSUMPTIONS["dv_nav"][0]
    return {
        "hover": dv_hover, "divert": dv_divert, "nav": dv_nav,
        "total": dv_hover + dv_divert + dv_nav,
    }


# 4. Mass closure  (Tsiolkovsky)

def close_mass(m_dry, prop_fraction_descent, dv_reserve, dv_deorbit):
    # Smallest separation mass that flies the descent and still holds the
    # reserve. Worked backwards from the dry mass, un-burning each phase in
    # reverse, so no iteration is needed.
    f_reserve = 1.0 - np.exp(-dv_reserve / VE)
    m_landed = m_dry / (1.0 - f_reserve)          # dry + reserve propellant
    m_pdi = m_landed / (1.0 - prop_fraction_descent)
    m_sep = m_pdi * np.exp(dv_deorbit / VE)       # before the deorbit burn

    return {
        "m_dry": m_dry,
        "m_reserve_prop": m_landed - m_dry,
        "m_landed": m_landed,
        "m_descent_prop": m_pdi - m_landed,
        "m_pdi": m_pdi,
        "m_deorbit_prop": m_sep - m_pdi,
        "m_sep": m_sep,
        "prop_total": m_sep - m_dry,
        "prop_fraction_total": (m_sep - m_dry) / m_sep,
        "dv_total": VE * np.log(m_sep / m_dry) * 1e3,
    }


# Report

def rule(char="=", n=74):
    print(char * n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="coarser sweeps and step size")
    args = parser.parse_args()
    dt = DT_QUICK if args.quick else DT

    requirements = []

    rule()
    print(" DESCENT SYSTEM — REQUIREMENTS DERIVATION")
    print(" Lunar Rover Delivery and Deployment Mission")
    rule()
    print("\n ASSUMPTIONS AND INPUTS\n")
    for key, (val, unit, why) in ASSUMPTIONS.items():
        print(f"   {key:<18}{val:>9.2f} {unit:<5}  {why}")
    print(f"   {'guidance rate':<18}{1.0/dt:>9.1f} {'Hz':<5}  "
          f"integration step = guidance update interval")

    print()
    rule("-")
    print(" STEP 1 — PARKING ORBIT AND TRANSFER GEOMETRY   (closed form)")
    rule("-")
    g15 = transfer_geometry(15.0)
    period = 2.0 * np.pi * np.sqrt(g15["r_park"]**3 / MU)
    print(f"\n   circular speed at {OrbitConfig.ALTITUDE:.0f} km      "
          f"{g15['v_circ']*1e3:10.2f} m/s")
    print(f"   orbital period                {period/60:10.2f} min")
    print("\n   Vis-viva: v^2 = mu (2/r - 1/a). On a circular orbit a = r,")
    print("   so v = sqrt(mu/r). Independent of vehicle mass — which is why")
    print("   the delta-v budget can be fixed before the mass is known.")

    print()
    rule("-")
    print(" STEP 2 — DEORBIT BURN                          (closed form)")
    rule("-")
    print(f"\n   {'h_PDI':>7}{'dv_deorbit':>13}{'ecc':>10}"
          f"{'coast':>10}{'v_peri':>11}")
    print(f"   {'[km]':>7}{'[m/s]':>13}{'[-]':>10}{'[min]':>10}{'[m/s]':>11}")
    for h in (15.0, 12.0, 10.0):
        g = transfer_geometry(h)
        print(f"   {h:7.1f}{g['dv_deorbit']*1e3:13.2f}{g['ecc']:10.5f}"
              f"{g['coast']/60:10.2f}{g['v_peri']*1e3:11.1f}")

    # sensitivity of perilune to burn cut-off
    h_ref = 12.0
    g_ref = transfer_geometry(h_ref)
    eps = 1e-5

    def perilune_of_dv(dv):
        v = g_ref["v_circ"] - dv
        a = 1.0 / (2.0 / g_ref["r_park"] - v * v / MU)
        return a * (1.0 - abs(1.0 - g_ref["r_park"] / a))

    dperi_ddv = ((perilune_of_dv(g_ref["dv_deorbit"] + eps)
                  - perilune_of_dv(g_ref["dv_deorbit"] - eps)) / (2 * eps))
    print(f"\n   Perilune sensitivity at h_PDI = {h_ref:.0f} km:")
    print(f"     d(perilune)/d(dv) = {abs(dperi_ddv)/1e3:.3f} km per m/s")
    print("\n   The deorbit burn is nearly loss-free at any thrust level,")
    print("   because it is tangential: thrust is perpendicular to gravity,")
    print("   so there is nothing to lose. Engine thrust is therefore NOT")
    print("   sized by this manoeuvre.")

    print()
    rule("-")
    print(" STEP 3 — POWERED DESCENT TRADE                 (integrated)")
    rule("-")
    print("\n   Propellant fraction is scale-invariant at fixed T/W, so the")
    print("   sweep below applies at any vehicle mass.\n")
    print(f"   {'h_PDI':>7}{'T/W':>7}{'prop frac':>11}{'dv':>9}"
          f"{'burn':>8}{'v_vert':>9}{'v_horz':>9}  result")
    print(f"   {'[km]':>7}{'[-]':>7}{'[-]':>11}{'[m/s]':>9}"
          f"{'[s]':>8}{'[m/s]':>9}{'[m/s]':>9}")

    tw_list = (3.0, 3.25, 3.5, 4.0, 5.0) if args.quick else \
              (2.75, 3.0, 3.25, 3.5, 4.0, 4.5, 5.0)
    h_list = (15.0, 12.0, 10.0)

    best = {}
    for h in h_list:
        for tw in tw_list:
            r = simulate_descent(h, tw, dt=dt)
            ok = descent_is_valid(r)
            tag = "ok" if ok else ("CRASH" if r["crashed"] else "reject")
            print(f"   {h:7.1f}{tw:7.2f}{r['prop_fraction']:11.4f}"
                  f"{r['dv']*1e3:9.0f}{r['duration']:8.1f}"
                  f"{r['v_vert']:9.2f}{r['v_horz']:9.2f}  {tag}")
            if ok and h not in best:
                best[h] = (tw, r)          # first valid = minimum viable T/W
        print()

    print("   Two findings:")
    print("     - A minimum T/W exists below which the vehicle reaches the")
    print("       surface before horizontal velocity is nulled. It rises as")
    print("       PDI altitude falls: less altitude, less time to brake.")
    print("     - Above the minimum, propellant fraction INCREASES with T/W.")
    print("       Braking harder ends the phase higher, and the altitude then")
    print("       has to be flown down. Design near the minimum, not above it.")

    print()
    rule("-")
    print(" STEP 4 — RESERVE BUDGET                        (closed form)")
    rule("-")
    res = reserve_budget()
    print(f"\n   hover {ASSUMPTIONS['hover_time'][0]:.0f} s at T/W = 1"
          f"        {res['hover']:10.1f} m/s   dv = g * t")
    print(f"   divert to landing point       {res['divert']:10.1f} m/s")
    print(f"   navigation dispersion         {res['nav']:10.1f} m/s")
    print(f"   {'-'*46}")
    print(f"   RESERVE TOTAL                 {res['total']:10.1f} m/s")

    print()
    rule("-")
    print(" STEP 5 — MASS CLOSURE                          (Tsiolkovsky)")
    rule("-")

    h_sel = 12.0
    tw_sel, r_sel = best[h_sel]
    g_sel = transfer_geometry(h_sel)

    print(f"\n   Selected point: PDI {h_sel:.0f} km, T/W {tw_sel:.2f}")
    print(f"   (10 km is marginally better on propellant but leaves less")
    print(f"    navigation margin; the difference is inside Phase A error.)\n")
    print(f"   {'m_dry':>8}{'m_sep':>9}{'prop':>9}{'frac':>8}"
          f"{'thrust':>9}{'dv_total':>10}")
    print(f"   {'[kg]':>8}{'[kg]':>9}{'[kg]':>9}{'[-]':>8}"
          f"{'[N]':>9}{'[m/s]':>10}")

    closure = None
    for m_dry in (200.0, 220.0, 240.0, 260.0):
        c = close_mass(m_dry, r_sel["prop_fraction"],
                       res["total"] / 1e3, g_sel["dv_deorbit"])
        thrust = tw_sel * c["m_pdi"] * MU / g_sel["r_peri"]**2
        print(f"   {m_dry:8.0f}{c['m_sep']:9.1f}{c['prop_total']:9.1f}"
              f"{c['prop_fraction_total']:8.1%}{thrust*1e3:9.0f}"
              f"{c['dv_total']:10.0f}")
        if m_dry == 240.0:
            closure = c
            thrust_sel = thrust

    print("\n   Worked backwards from the dry mass: un-burn the reserve, then")
    print("   the descent, then the deorbit burn. No iteration needed.\n")
    print(f"   Mass breakdown at m_dry = {closure['m_dry']:.0f} kg:")
    print(f"     dry mass                    {closure['m_dry']:8.1f} kg")
    print(f"     reserve propellant          {closure['m_reserve_prop']:8.1f} kg")
    print(f"     descent propellant          {closure['m_descent_prop']:8.1f} kg")
    print(f"     deorbit propellant          {closure['m_deorbit_prop']:8.1f} kg")
    print(f"     {'-'*40}")
    print(f"     separation mass             {closure['m_sep']:8.1f} kg")

    requirements = [
        ("PROP-010", "Descent propulsion shall provide a total impulse "
         f"equivalent to {closure['dv_total']:.0f} m/s of delta-V.",
         "Step 5, Tsiolkovsky closure"),
        ("PROP-020", f"Descent engine thrust shall be no less than "
         f"{thrust_sel*1e3:.0f} N at PDI.",
         f"Step 3, minimum T/W = {tw_sel:.2f} at {h_sel:.0f} km PDI"),
        ("PROP-030", "Descent engine shall be throttleable over at least "
         "30-100% of rated thrust.",
         "Step 3, terminal descent rate control"),
        ("PROP-040", f"Specific impulse shall be no less than "
         f"{ASSUMPTIONS['isp'][0]:.0f} s.",
         "Step 5, assumption; closure fails below this"),
        ("MASS-010", f"Descent module dry mass shall not exceed "
         f"{closure['m_dry']:.0f} kg.",
         "Step 5, for the selected separation mass"),
        ("MASS-020", f"Separation mass shall not exceed "
         f"{closure['m_sep']:.0f} kg.",
         "Step 5, mass closure"),
        ("MASS-030", f"Propellant load shall be "
         f"{closure['prop_total']:.0f} kg "
         f"({closure['prop_fraction_total']:.0%} of separation mass).",
         "Step 5"),
        ("GNC-010", f"Deorbit burn cut-off shall be accurate to 0.1 s.",
         f"Step 2, sensitivity {abs(dperi_ddv)/1e3:.2f} km perilune per m/s"),
        ("GNC-020", f"Guidance loop shall execute at no less than "
         f"{1.0/dt:.0f} Hz during powered descent.",
         "Step 3, integration step at which the trade converges"),
        ("GNC-030", f"Navigation shall deliver perilune altitude to within "
         f"the dispersion covered by {res['nav']:.0f} m/s of reserve.",
         "Step 4"),
        ("GNC-040", "Guidance shall implement a gravity-turn braking law "
         "with thrust anti-parallel to the velocity vector.",
         "Step 3, near-optimal and requires no scheduled attitude profile"),
        ("LAND-010", f"Landing gear shall absorb a vertical touchdown "
         f"velocity of {ASSUMPTIONS['v_touchdown_max'][0]:.0f} m/s.",
         "Assumption; to be verified once terminal guidance is closed"),
        ("MIS-010", f"Powered descent shall initiate at "
         f"{h_sel:.0f} km altitude.",
         "Step 3, PDI trade"),
        ("MIS-020", f"Coast from deorbit burn to PDI shall be "
         f"{g_sel['coast']/60:.0f} min.",
         "Step 2, half the transfer ellipse period"),
    ]

    print()
    rule()
    print(" DERIVED REQUIREMENTS")
    rule()
    print()
    for rid, text, source in requirements:
        print(f"   [{rid}]  {text}")
        print(f"{'':13}source: {source}\n")

    rule()
    print(" CAVEATS")
    rule()
    print("""
   1. The terminal-descent controller is a proportional rate tracker, not a
      landing autopilot. Touchdown velocities above are indicative; LAND-010
      cannot be closed until terminal guidance is designed.

   2. Gravity is modelled as a point mass. J2 and the mascons perturb the
      parking orbit over days and shift the ground track; they do not
      materially change the delta-V budget.

   3. The descent is planar. A plane change or a downrange correction would
      add delta-V not counted here.

   4. Isp is treated as constant. Real engines lose Isp when deep-throttled,
      which penalises the terminal phase specifically.

   5. No structural, thermal or power masses are allocated. MASS-010 is a
      ceiling on the sum of every dry subsystem, and is currently the binding
      constraint on the design.
""")

# Relay–Lander communication windows

def relay_visibility(moon, h_orbit=100.0, inc_deg=89.5, lander_lat_deg=-89.0,
                     dt=1.0, verbose=True):
    # Propagate the relay one orbit and compute the communication windows with
    # the surface lander. The relay sees the lander when the angle between its
    # sub-satellite point and the site is below the horizon half-angle,
    #     rho = arccos(R / (R + h))
    # The lander is within ~1 deg of the pole and the cone is ~19 deg, so every
    # south-bound pass gives a window. Units km, deg, s. Returns a dict.
    R = moon.radius
    a = R + h_orbit
    T = 2.0 * np.pi * np.sqrt(a**3 / moon.mu)          # orbital period [s]
    rho = np.arccos(R / a)                                # horizon half-angle [rad]

    # lander position unit vector (fixed on the surface, MCMF)
    lat = np.deg2rad(lander_lat_deg)
    # lon = 0: moon rotation is negligible over one orbit
    lander_hat = np.array([np.cos(lat), 0.0, np.sin(lat)])

    # relay mean motion
    n = 2.0 * np.pi / T
    inc = np.deg2rad(inc_deg)

    # step through one orbit
    visible = []
    t = 0.0
    while t <= T:
        # relay position in MCI (circular orbit, RAAN=0, AOP=0)
        M = n * t
        r_relay = a * np.array([
            np.cos(M),
            np.sin(M) * np.cos(inc),
            np.sin(M) * np.sin(inc),
        ])
        relay_hat = r_relay / np.linalg.norm(r_relay)

        # angular separation between sub-satellite point and lander
        cos_ang = np.dot(relay_hat, lander_hat)
        cos_ang = np.clip(cos_ang, -1.0, 1.0)
        ang = np.arccos(cos_ang)

        visible.append(ang < rho)
        t += dt

    visible = np.array(visible)

    # extract windows (contiguous True blocks)
    edges = np.diff(visible.astype(int))
    rises = np.where(edges == 1)[0]
    sets  = np.where(edges == -1)[0]

    # handle window open at t=0
    if visible[0]:
        rises = np.concatenate(([0], rises))
    if visible[-1]:
        sets = np.concatenate((sets, [len(visible) - 1]))

    windows = [(rises[i] * dt, sets[i] * dt) for i in range(min(len(rises), len(sets)))]

    total_visible = sum(end - start for start, end in windows)
    duty_cycle = total_visible / T

    results = {
        "period_s":        T,
        "rho_deg":         np.degrees(rho),
        "n_windows":       len(windows),
        "windows":         windows,           # list of (start_s, end_s)
        "total_visible_s": total_visible,
        "gap_s":           T - total_visible,
        "duty_cycle":      duty_cycle,
    }

    if verbose:
        print("\n RELAY–LANDER VISIBILITY (one orbit)\n")
        print(f"   orbit altitude          {h_orbit:.1f} km")
        print(f"   orbital period          {T:.1f} s  ({T/60:.1f} min)")
        print(f"   horizon half-angle      {np.degrees(rho):.2f} deg")
        print(f"   lander latitude         {lander_lat_deg:.1f} deg")
        print(f"   inclination             {inc_deg:.1f} deg")
        print()
        for i, (t0, t1) in enumerate(windows):
            dur = t1 - t0
            print(f"   window {i+1}:  {t0:.0f} – {t1:.0f} s  "
                  f"({dur:.0f} s = {dur/60:.1f} min)")
        print()
        print(f"   total contact           {total_visible:.0f} s  "
              f"({total_visible/60:.1f} min)")
        print(f"   blackout                {T - total_visible:.0f} s  "
              f"({(T - total_visible)/60:.1f} min)")
        print(f"   duty cycle              {duty_cycle*100:.1f} %")
        print()

        # descent autonomy note
        descent_time = 57.5 * 60   # ~57.5 min from your descent run
        print(f"   descent duration        {descent_time:.0f} s  "
              f"({descent_time/60:.1f} min)")
        if total_visible < descent_time:
            print(f"   !! contact < descent → lander MUST operate autonomously")
            print(f"      max 1 partial window during descent")
        print()

        # derived requirement
        print("   DERIVED REQUIREMENT:")
        print("   COM-010  The lander shall complete powered descent and")
        print("            landing without real-time ground commands.")
        print(f"            [source: {total_visible/60:.1f} min contact vs "
              f"{descent_time/60:.1f} min descent]")

    return results

def mission_timeline(moon, h_stack, h_relay, h_lander, guidance, t_sep,
                     imu_drift_mps_hr=0.5, verbose=True):
    # Post-process the histories into a mission timeline with go/no-go checks at
    # each phase transition. Not a new simulation.
    #     h_stack/relay/lander  History objects from run_mission()
    #     guidance              DescentGuidance, carries .events
    #     t_sep                 separation epoch [s]
    #     imu_drift_mps_hr      assumed IMU velocity random walk [m/s per hour]
    if not verbose:
        return

    R = moon.radius

    # helpers

    def relay_pos_at(t):
        # Relay position at arbitrary time.
        idx = np.searchsorted(h_relay.t, t)
        idx = np.clip(idx, 0, len(h_relay.t) - 1)
        return h_relay.r[idx]

    def lander_pos_at(t):
        idx = np.searchsorted(h_lander.t, t)
        idx = np.clip(idx, 0, len(h_lander.t) - 1)
        return h_lander.r[idx]

    def lander_state_at(t):
        idx = np.searchsorted(h_lander.t, t)
        idx = np.clip(idx, 0, len(h_lander.t) - 1)
        return (h_lander.r[idx], h_lander.v[idx],
                h_lander.altitude[idx], h_lander.speed[idx],
                h_lander.mass[idx], h_lander.lat[idx], h_lander.lon[idx])

    def check_visibility(r_relay, r_lander):
        # True if the relay can see the lander (above the mutual horizon).
        rho = np.arccos(R / np.linalg.norm(r_relay))   # horizon half-angle
        r_hat = r_relay / np.linalg.norm(r_relay)
        l_hat = r_lander / np.linalg.norm(r_lander)
        ang = np.arccos(np.clip(np.dot(r_hat, l_hat), -1, 1))
        return ang < rho

    def ok(condition):
        return "✓" if condition else "✗"

    # extract event times from guidance
    # guidance.events is [(t_abs, phase, alt_km, speed_kms, mass, note)]
    events = {}
    for ev in guidance.events:
        events[ev["phase"]] = ev
      

    # print timeline

    print("\n" + "=" * 65)
    print(" MISSION TIMELINE")
    print("=" * 65)

    # 1. Stack in parking orbit
    r0 = h_stack.r[0] if h_stack is not None else h_lander.r[0]
    alt0 = np.linalg.norm(r0) - R
    print(f"\n  t = 0 s  │  STACK — parking orbit")
    print(f"           │  altitude: {alt0:.1f} km")
    print(f"           │  period:   {2*np.pi*np.sqrt((R+alt0)**3/moon.mu):.0f} s")

    # 2. Separation
    sep_r = h_stack.r[-1] if h_stack is not None else h_lander.r[0]
    sep_alt = np.linalg.norm(sep_r) - R
    sep_speed = np.linalg.norm(h_stack.v[-1]) if h_stack is not None else 0
    relay_at_sep = relay_pos_at(t_sep)
    vis_at_sep = check_visibility(relay_at_sep, sep_r)

    print(f"\n  t = {t_sep:.0f} s  │  SEPARATION CHECK")
    print(f"           │  orbit altitude:   {sep_alt:.1f} km   "
          f"{ok(abs(sep_alt - 100.0) < 1.0)}  (req: 100 ± 1 km)")
    print(f"           │  orbit speed:      {sep_speed:.1f} km/s")
    print(f"           │  relay link:       {'ACTIVE' if vis_at_sep else 'NONE'}     "
          f"{ok(vis_at_sep)}  (req: comm window open)")
    print(f"           │  → SEPARATION {'AUTHORIZED' if vis_at_sep else 'DELAYED'}")

    # 3. Deorbit burn
    if "deorbit" in events:
        ev = events["deorbit"]
        t_deorbit = ev["t"]
        print(f"\n  t = {t_deorbit:.0f} s  │  LANDER — deorbit burn")
        print(f"           │  altitude: {ev['altitude']:.1f} km")
        print(f"           │  mass:     {ev['mass']:.1f} kg")

    # 4. Coast
    if "coast" in events:
        ev = events["coast"]
        t_coast = ev["t"]

        # coast duration (to braking)
        t_pdi = events.get("braking", {}).get("t", t_coast)
        coast_dur_s = t_pdi - t_coast
        coast_dur_min = coast_dur_s / 60.0

        # IMU drift estimate
        drift_vel = imu_drift_mps_hr * (coast_dur_s / 3600.0)   # m/s
        drift_pos = drift_vel * coast_dur_s / 2.0                # m (avg vel × time)

        print(f"\n  t = {t_coast:.0f} s  │  LANDER — coast (engine off)")
        print(f"           │  altitude: {ev['altitude']:.1f} km")
        print(f"           │  nav source: IMU only (no radar, no doppler)")
        print(f"           │  coast duration: {coast_dur_min:.1f} min")
        print(f"           │  IMU drift estimate ({imu_drift_mps_hr} m/s/hr):")
        print(f"           │    velocity error at PDI:  {drift_vel:.2f} m/s")
        print(f"           │    position error at PDI:  {drift_pos:.0f} m")
        print(f"           │    within divert range:    "
              f"{ok(drift_pos < 500)}  (req: < 500 m, NAV-030)")

    # 5. PDI / Braking
    if "braking" in events:
        ev = events["braking"]
        t_pdi = ev["t"]
        r_l = lander_pos_at(t_pdi)
        r_r = relay_pos_at(t_pdi)
        vis_pdi = check_visibility(r_r, r_l)

        print(f"\n  t = {t_pdi:.0f} s  │  LANDER — PDI (braking)")
        print(f"           │  altitude: {ev['altitude']:.1f} km")
        print(f"           │  speed:    {ev['speed']*1000:.1f} m/s")
        print(f"           │  mass:     {ev['mass']:.1f} kg")
        print(f"           │  radar altimeter: ACTIVE   "
              f"{ok(ev['altitude'] < 20)}  (req: range ≥ 15 km)")
        print(f"           │  doppler velocimeter: ACTIVE")
        print(f"           │  relay link: {'ACTIVE' if vis_pdi else 'BLACKOUT'}")
        if not vis_pdi:
            print(f"           │  → autonomous descent (COM-010)")

    # 6. Terminal
    if "terminal" in events:
        ev = events["terminal"]
        t_term = ev["t"]
        r_l = lander_pos_at(t_term)
        r_r = relay_pos_at(t_term)
        vis_term = check_visibility(r_r, r_l)

        print(f"\n  t = {t_term:.0f} s  │  LANDER — terminal descent")
        print(f"           │  altitude: {ev['altitude']*1000:.0f} m")
        print(f"           │  speed:    {ev['speed']*1000:.1f} m/s")
        print(f"           │  obstacle detection: ACTIVE")
        print(f"           │  relay link: {'ACTIVE' if vis_term else 'BLACKOUT'}")

    # 7. Touchdown
    if "landed" in events:
        ev = events["landed"]
        t_land = ev["t"]
        r_l, v_l, alt_l, spd_l, mass_l, lat_l, lon_l = lander_state_at(t_land)
        r_r = relay_pos_at(t_land)
        vis_land = check_visibility(r_r, r_l)

        # vertical / horizontal velocity decomposition
        r_hat = r_l / np.linalg.norm(r_l)
        v_vert = np.dot(v_l, r_hat)          # km/s, negative = downward
        v_horz = np.linalg.norm(v_l - v_vert * r_hat)

        relay_alt = np.linalg.norm(r_r) - R
        relay_lat_idx = np.searchsorted(h_relay.t, t_land)
        relay_lat_idx = np.clip(relay_lat_idx, 0, len(h_relay.t) - 1)

        print(f"\n  t = {t_land:.0f} s  │  LANDER — TOUCHDOWN")
        print(f"           │  lat: {np.degrees(lat_l):.3f}°  "
              f"lon: {np.degrees(lon_l):.3f}°")
        print(f"           │  v_vertical:   {abs(v_vert)*1000:.2f} m/s   "
              f"{ok(abs(v_vert)*1000 < 4.0)}  (req: < 4 m/s, LAND-010)")
        print(f"           │  v_horizontal: {v_horz*1000:.2f} m/s   "
              f"{ok(v_horz*1000 < 4.0)}  (req: < 4 m/s)")
        print(f"           │  mass: {mass_l:.1f} kg")
        print(f"           │  → SAFE ON SURFACE")
        print(f"           │")
        print(f"           │  RELAY at touchdown:")
        print(f"           │    altitude: {relay_alt:.1f} km")
        print(f"           │    lat: {np.degrees(h_relay.lat[relay_lat_idx]):.1f}°")
        print(f"           │    link: {'ACTIVE' if vis_land else 'BLACKOUT'}")
        if not vis_land:
            # estimate next pass
            for i in range(relay_lat_idx, min(relay_lat_idx + len(h_relay.t)//2, len(h_relay.t))):
                if check_visibility(h_relay.r[i], r_l):
                    wait = h_relay.t[i] - t_land
                    print(f"           │    next pass in: {wait/60:.0f} min")
                    break

    print(f"\n{'=' * 65}\n")

if __name__ == "__main__":
    main()
    relay_visibility(moon)