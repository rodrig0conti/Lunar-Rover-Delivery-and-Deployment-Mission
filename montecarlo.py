# monte-carlo dispersion of the landing, for a success-probability estimate.
# one batch of runs; the success-vs-zone-radius curve is post-processed from
# the landing distances, so no re-running per radius.

import csv
import numpy as np

from config import OrbitConfig, PropulsionConfig, SiteConfig
from moon_environment import Moon
from spacecraft import Spacecraft
from mission import run_mission
from descent import DescentGuidance

# ---- campaign settings ------------------------------------------------

N_RUNS = 1            # start with 10 to validate, then raise
DT_DESCENT = 0.05        # coarser than nominal 0.05 for speed; nominal check below
SEED = 12345
USE_TCM = False          # mid-course correction on/off
CSV_OUT = "montecarlo_tcm.csv" if USE_TCM else "montecarlo_results.csv"
R_ZONA_SWEEP = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100]

# 1-sigma dispersions, in physical units
SIG = {
    "sep_pos_km":       0.5,      # separation position (orbit determination)
    "sep_vel_ms":       0.5,      # separation velocity
    "deorbit_mag_frac": 0.01,     # deorbit dv magnitude
    "deorbit_point_deg": 0.5,     # deorbit pointing
    "thrust_frac":      0.02,     # main engine thrust
    "isp_frac":         0.01,     # isp
    "prop_kg":          1.0,      # propellant load
    "grav_bias_ms2":    1.0e-4,   # unmodelled gravity (mascons beyond j2)
}
DEORBIT_DV_MS = 20.0    # nominal deorbit dv, to turn the fractions into m/s

TARGET_LAT = SiteConfig.LATITUDE
TARGET_LON = SiteConfig.LONGITUDE
V_LIMIT = 4.0           # m/s, touchdown limit on both components


# ---- helpers ----------------------------------------------------------

def great_circle(lat1, lon1, lat2, lon2, R):
    # surface distance between two lat/lon points [km]
    d = np.arccos(np.clip(
        np.sin(lat1) * np.sin(lat2)
        + np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2), -1.0, 1.0))
    return R * d


def nominal_params():
    return {"dp": np.zeros(3), "dv": np.zeros(3), "accel_bias": np.zeros(3),
            "thrust": PropulsionConfig.THRUST_MAX,
            "isp": PropulsionConfig.ISP,
            "mass_wet": PropulsionConfig.MASS_WET}


def sample_params(rng):
    # position and velocity dispersion at separation, in km and km/s
    dp = rng.normal(0.0, SIG["sep_pos_km"], 3)

    dv_od = rng.normal(0.0, SIG["sep_vel_ms"], 3)
    # deorbit execution folded into an isotropic velocity error: magnitude
    # (fraction of dv) and pointing (angle on dv) combined in quadrature
    sig_burn = np.hypot(SIG["deorbit_mag_frac"] * DEORBIT_DV_MS,
                        DEORBIT_DV_MS * np.sin(np.radians(SIG["deorbit_point_deg"])))
    dv_burn = rng.normal(0.0, sig_burn, 3)
    dv = (dv_od + dv_burn) * 1.0e-3            # m/s -> km/s

    # unmodelled gravity: random-direction bias, per-component so the vector
    # rms magnitude matches grav_bias. m/s^2 -> km/s^2
    sig_g = SIG["grav_bias_ms2"] / np.sqrt(3.0) * 1.0e-3
    accel_bias = rng.normal(0.0, sig_g, 3)

    return {"dp": dp, "dv": dv, "accel_bias": accel_bias,
            "thrust": PropulsionConfig.THRUST_MAX * (1 + rng.normal(0, SIG["thrust_frac"])),
            "isp": PropulsionConfig.ISP * (1 + rng.normal(0, SIG["isp_frac"])),
            "mass_wet": PropulsionConfig.MASS_WET + rng.normal(0, SIG["prop_kg"])}



def run_one(moon, p):
    # build fresh vehicles with the dispersed parameters. no attitude, so the
    # lander flies the fast seven-element model.
    relay = Spacecraft("Relay", mass=250.0)
    lander = Spacecraft("Lander", mass=p["mass_wet"])
    for c in (relay, lander):
        c.set_orbit(moon.mu, OrbitConfig.SMA, OrbitConfig.ECC,
                    OrbitConfig.INC, OrbitConfig.RAAN,
                    OrbitConfig.AOP, OrbitConfig.TA)
    lander.set_propulsion(PropulsionConfig.MASS_DRY, p["thrust"], p["isp"],
                          PropulsionConfig.THROTTLE_MIN)

    _, _, h_lander, guidance, _ = run_mission(
        moon, relay, lander,
        n_orbits_before_sep=OrbitConfig.N_ORBITS_BEFORE_SEP,
        dt_descent=DT_DESCENT,
        sep_dp=p["dp"], sep_dv=p["dv"], accel_bias=p["accel_bias"],
        verbose=False)

    landed = (guidance.phase == DescentGuidance.LANDED)
    r, v = h_lander.r[-1], h_lander.v[-1]
    lat, lon, _ = moon.geodetic(r, h_lander.t[-1])
    up = r / np.linalg.norm(r)
    v_vert = abs(float(v @ up)) * 1e3
    v_horz = float(np.linalg.norm(v - (v @ up) * up)) * 1e3
    dist = great_circle(lat, lon, TARGET_LAT, TARGET_LON, moon.radius)
    prop = p["mass_wet"] - h_lander.mass[-1]

    soft = landed and v_vert < V_LIMIT and v_horz < V_LIMIT
    if not landed:
        reason = "no_land"
    elif v_vert >= V_LIMIT or v_horz >= V_LIMIT:
        reason = "hard"
    else:
        reason = "ok"

    return {"landed": landed, "soft": soft, "lat": np.degrees(lat),
            "lon": np.degrees(lon), "v_vert": v_vert, "v_horz": v_horz,
            "prop": prop, "dist": dist, "reason": reason}


# ---- campaign ---------------------------------------------------------

def main():
    moon = Moon()

    # nominal check first: confirms the coarser step still lands softly.
    nom = run_one(moon, nominal_params())
    print(f"nominal check (dt={DT_DESCENT}): "
          f"lat {nom['lat']:.3f}  lon {nom['lon']:.3f}  "
          f"v_vert {nom['v_vert']:.2f}  v_horz {nom['v_horz']:.2f}  "
          f"dist {nom['dist']:.1f} km")
    if nom["v_vert"] > 1.5:
        print("  !! nominal v_vert high at this dt — drop DT_DESCENT to 0.05")

    rng = np.random.default_rng(SEED)
    rows = []
    print(f"\nrunning {N_RUNS} dispersed cases...")
    for i in range(N_RUNS):
        rows.append(run_one(moon, sample_params(rng)))
        if (i + 1) % 10 == 0:
            rate = np.mean([r["soft"] for r in rows])
            print(f"  {i+1:4d}/{N_RUNS}   soft-landing rate so far {rate:.1%}")

    # save raw results
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # aggregate
    soft = np.array([r["soft"] for r in rows])
    landed = np.array([r["landed"] for r in rows])
    dist = np.array([r["dist"] for r in rows])
    v_vert = np.array([r["v_vert"] for r in rows])
    v_horz = np.array([r["v_horz"] for r in rows])
    prop = np.array([r["prop"] for r in rows])

    soft_dist = dist[soft]     # distances of the soft landings only

    print("\n" + "=" * 60)
    print(" MONTE CARLO SUMMARY")
    print("=" * 60)
    print(f"\n   runs                 {N_RUNS}")
    print(f"   reached surface      {landed.mean():.1%}")
    print(f"   soft landing         {soft.mean():.1%}  "
          f"(v_vert & v_horz < {V_LIMIT:.0f} m/s)")
    print(f"\n   touchdown v_vert     {v_vert.mean():.2f} +/- {v_vert.std():.2f} m/s")
    print(f"   touchdown v_horz     {v_horz.mean():.2f} +/- {v_horz.std():.2f} m/s")
    print(f"   propellant used      {prop.mean():.1f} +/- {prop.std():.1f} kg")

    if len(soft_dist):
        print(f"\n   landing distance from target (soft landings):")
        print(f"     CEP (50%)          {np.percentile(soft_dist, 50):.2f} km")
        print(f"     95%                {np.percentile(soft_dist, 95):.2f} km")
        print(f"     max                {soft_dist.max():.2f} km")

    # success probability vs zone radius (the option-3 sweep)
    print(f"\n   success probability vs landing-zone radius:")
    print(f"     {'R_zona [km]':>12}{'P(success)':>13}")
    for R in R_ZONA_SWEEP:
        ok = soft & (dist < R)
        print(f"     {R:12.1f}{ok.mean():12.1%}")

    # failure breakdown
    reasons = [r["reason"] for r in rows]
    print(f"\n   outcome breakdown:")
    for k in ("ok", "hard", "no_land"):
        print(f"     {k:10s}         {reasons.count(k):4d}  "
              f"({reasons.count(k)/N_RUNS:.1%})")

    print(f"\n   raw results written to {CSV_OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()