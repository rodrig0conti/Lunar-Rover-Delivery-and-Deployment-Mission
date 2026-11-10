# Analytic delta-v budget for the descent strategy. No integration.

import numpy as np
from config import MoonConstants, OrbitConfig, PropulsionConfig
from moon_environment import Moon

moon = Moon()
mu = moon.mu

r_park = OrbitConfig.SMA                      # 100 km circular
h_peri = 15.0                                 # target perilune [km]
r_peri = moon.radius + h_peri
a_t = 0.5 * (r_park + r_peri)                 # transfer ellipse

v_circ = np.sqrt(mu / r_park)
v_apo = np.sqrt(mu * (2.0 / r_park - 1.0 / a_t))
v_per = np.sqrt(mu * (2.0 / r_peri - 1.0 / a_t))

dv_deorbit = v_circ - v_apo
dv_brake = v_per                              # ideal impulsive, zero at surface
coast = np.pi * np.sqrt(a_t**3 / mu)

print("=" * 58)
print(" DELTA-V BUDGET  (ideal, impulsive)")
print("=" * 58)
print(f" parking orbit        {r_park - moon.radius:8.1f} km  "
      f"v = {v_circ*1e3:8.1f} m/s")
print(f" transfer perilune    {h_peri:8.1f} km  "
      f"ecc = {(r_park - r_peri)/(r_park + r_peri):.6f}")
print(f" coast to perilune    {coast/60:8.2f} min")
print("-" * 58)
print(f" deorbit burn         {dv_deorbit*1e3:8.1f} m/s")
print(f" powered braking      {dv_brake*1e3:8.1f} m/s   (ideal)")
print(f" TOTAL ideal          {(dv_deorbit + dv_brake)*1e3:8.1f} m/s")
print("=" * 58)

print("\n Propellant required (Tsiolkovsky), ideal dv only:\n")
print(f" {'Isp [s]':>8}{'m_wet [kg]':>12}{'m_prop [kg]':>13}{'fraction':>10}")
print("-" * 58)
dv = dv_deorbit + dv_brake
for isp in (220.0, 300.0, 320.0):
    ve = isp * PropulsionConfig.G0
    for m0 in (300.0, 500.0):
        mp = m0 * (1.0 - np.exp(-dv / ve))
        print(f" {isp:8.0f}{m0:12.0f}{mp:13.1f}{mp/m0:10.1%}")

print("\n Ideal means impulsive. Real braking is a finite burn against gravity,")
print(" so add gravity losses, plus hover, divert and margin. Those need the")
print(" integrated simulation, not this script.")
