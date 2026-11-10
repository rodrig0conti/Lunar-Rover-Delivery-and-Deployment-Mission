# Propagate the parking orbit with RK4 and compare against the closed form.

import numpy as np
from config import OrbitConfig
from moon_environment import Moon
from spacecraft import Spacecraft
from dynamics import Dynamics
from mathutils import rk4_step, orbital_period

moon = Moon()
craft = Spacecraft("Test", mass=170.0)
craft.set_orbit(moon.mu, OrbitConfig.SMA, OrbitConfig.ECC, OrbitConfig.INC,
                OrbitConfig.RAAN, OrbitConfig.AOP, OrbitConfig.TA)
dyn = Dynamics(moon, craft)

period = orbital_period(moon.mu, OrbitConfig.SMA)
print(f"orbital period : {period:.3f} s  ({period/3600:.4f} h)\n")

print(f"{'dt [s]':>8}{'pos error [m]':>18}{'|dE/E0|':>14}")
print("-" * 40)

for dt in (10.0, 5.0, 1.0):
    y = craft.state.copy()
    e0 = dyn.specific_energy(y)
    t = 0.0
    for _ in range(int(round(3.0 * period / dt))):
        y = rk4_step(dyn.derivative, t, y, dt)
        t += dt

    ref = craft.state_at(t)                 # closed form at the same instant
    err_m = np.linalg.norm(y[0:3] - ref[0:3]) * 1e3
    de = abs((dyn.specific_energy(y) - e0) / e0)
    print(f"{dt:8.1f}{err_m:18.3e}{de:14.3e}")

print("\nHalving dt should cut the error by about 16 (fourth order).")


from simulation import Simulation
from dynamics import Dynamics

print()
sim = Simulation(moon, dyn, dt=1.0)
hist = sim.run(craft.state, 3.0 * period)
print(hist.summary())

# A deliberately sub-orbital state: it must stop on impact, not run to t_final
print()
bad = np.concatenate([[moon.radius + 50.0, 0, 0], [0, 0.5, 0], [170.0]])
Simulation(moon, dyn, dt=1.0).run(bad, 20000.0)