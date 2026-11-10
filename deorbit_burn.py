# Integrated deorbit burn: real propellant, real burn duration.

import numpy as np
from config import OrbitConfig, PropulsionConfig
from moon_environment import Moon
from spacecraft import Spacecraft
from dynamics import Dynamics
from simulation import Simulation
from mathutils import perilune_radius, orbital_period

H_PERI_TARGET = 15.0                       # [km]

moon = Moon()
r_peri_target = moon.radius + H_PERI_TARGET

craft = Spacecraft("Descent Module", mass=PropulsionConfig.MASS_WET)
craft.set_orbit(moon.mu, OrbitConfig.SMA, OrbitConfig.ECC, OrbitConfig.INC,
                OrbitConfig.RAAN, OrbitConfig.AOP, OrbitConfig.TA)
craft.set_propulsion(PropulsionConfig.MASS_DRY, PropulsionConfig.THRUST_MAX,
                     PropulsionConfig.ISP, PropulsionConfig.THROTTLE_MIN)

dyn = Dynamics(moon, craft)
sim = Simulation(moon, dyn, dt=0.05)


def retrograde_burn(t, state):
    # Full throttle opposite to velocity, which removes energy fastest.
    v = state[3:6]
    craft.set_command(1.0, -v / np.linalg.norm(v))


sim.guidance = retrograde_burn
sim.add_stop_condition(
    "target perilune reached",
    lambda t, s: perilune_radius(moon.mu, s[0:3], s[3:6]) <= r_peri_target)

m0 = craft.state[6]
hist = sim.run(craft.state, t_final=600.0)

m1 = hist.mass[-1]
prop = m0 - m1
dv_engine = craft.v_exhaust * np.log(m0 / m1)      # Tsiolkovsky, actual

# ideal impulsive reference
r_park = OrbitConfig.SMA
a_t = 0.5 * (r_park + r_peri_target)
dv_ideal = (np.sqrt(moon.mu / r_park)
            - np.sqrt(moon.mu * (2.0 / r_park - 1.0 / a_t)))

el_r, el_v = hist.r[-1], hist.v[-1]
from mathutils import osculating_elements
el = osculating_elements(moon.mu, el_r, el_v)

print("\n" + "=" * 58)
print(" DEORBIT BURN  (integrated)")
print("=" * 58)
print(f" thrust                {PropulsionConfig.THRUST_MAX*1e3:10.0f} N")
print(f" initial accel         {PropulsionConfig.THRUST_MAX/m0*1e3:10.3f} m/s2")
print(f" burn duration         {hist.t[-1]:10.2f} s")
print(f" propellant used       {prop:10.3f} kg")
print("-" * 58)
print(f" dv imparted (engine)  {dv_engine*1e3:10.2f} m/s")
print(f" dv ideal (impulsive)  {dv_ideal*1e3:10.2f} m/s")
print(f" finite-burn loss      {(dv_engine-dv_ideal)/dv_ideal:10.2%}")
print("-" * 58)
print(f" resulting perilune    {el['r_peri']-moon.radius:10.3f} km "
      f"(target {H_PERI_TARGET:.1f})")
print(f" resulting apolune     {el['r_apo']-moon.radius:10.3f} km")
print(f" eccentricity          {el['ecc']:10.6f}")
print(f" coast to perilune     {np.pi*np.sqrt(el['sma']**3/moon.mu)/60:10.2f} min")
print(f" dv still available    {craft.delta_v_capability(m1)*1e3:10.1f} m/s")
print("=" * 58)