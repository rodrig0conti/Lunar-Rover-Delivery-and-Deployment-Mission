# Reaction control system: the actuators that rotate the vehicle.

# COUPLED PAIRS. Two thrusters on opposite sides firing opposite ways give a
# pure couple, torque with no net force. A single thruster also pushes the
# vehicle, spending delta-v the budget never saw. The uncoupled case is kept
# available so that penalty can be quantified instead of assumed away.

# ON/OFF, NOT THROTTLED. A thruster is a valve. The shortest pulse it can
# deliver is the minimum impulse bit, which sets the finest correction
# possible, and with a deadband this produces the limit cycle.

# MONOPROPELLANT. Cold gas nitrogen is simpler and non-toxic but gives about
# 65 s of Isp against 220 s for hydrazine, and the mass budget decides it.

# SEPARATE PROPELLANT. RCS consumption is tracked apart from the main engine so
# attitude cost stays visible.

# Units: SI throughout (m, kg, N, N m, s). The trajectory code works in km.

import numpy as np


class Thruster:
    # One thruster: where it is and which way it pushes.
    # direction is the force ON THE VEHICLE, not the exhaust. They are opposite,
    # and confusing them flips every torque sign.

    def __init__(self, name, position, direction, thrust,
                 min_impulse_bit=0.02):
        self.name = name
        self.position = np.asarray(position, dtype=float)      # body axes [m]
        d = np.asarray(direction, dtype=float)
        self.direction = d / np.linalg.norm(d)
        self.thrust = float(thrust)                            # [N]
        self.min_impulse_bit = float(min_impulse_bit)          # [s] min on-time

        self.on = False
        self.total_on_time = 0.0                               # [s]

    @property
    def force(self):
        # Force on the vehicle when firing, body axes [N].
        return self.thrust * self.direction

    @property
    def torque(self):
        # Torque about the centre of mass when firing [N m]: tau = r x F.
        # A thrust line through the centre of mass gives no torque at all.
        return np.cross(self.position, self.force)

    def __repr__(self):
        return (f"<{self.name}: {self.thrust:.1f} N at {self.position}, "
                f"tau = {np.round(self.torque, 3)}>")


class ThrusterPair:
    # Two thrusters forming a pure couple about one axis. Validated on
    # construction: a net force that is not essentially zero means the geometry is
    # not a couple, and the caller is told instead of silently being pushed.

    def __init__(self, name, thruster_a, thruster_b, tol=1e-9):
        self.name = name
        self.a = thruster_a
        self.b = thruster_b

        net_force = thruster_a.force + thruster_b.force
        self.residual_force = float(np.linalg.norm(net_force))
        self.is_pure_couple = self.residual_force < tol * max(
            1.0, thruster_a.thrust)

        self.torque = thruster_a.torque + thruster_b.torque

    def fire(self, on):
        self.a.on = bool(on)
        self.b.on = bool(on)


class ThrusterCluster:
    # The complete RCS. Given on/off commands it reports the resulting torque, the
    # net force, and the propellant spent.

    def __init__(self, thrusters, isp=220.0, name="RCS"):
        self.name = name
        self.thrusters = list(thrusters)
        self.isp = float(isp)
        self.g0 = 9.80665
        self.propellant_used = 0.0        # [kg]
        self.total_impulse = 0.0          # [N s]
        self.pairs = {}

    def add_pair(self, name, index_a, index_b):
        # Register a coupled pair by index into the thruster list.
        pair = ThrusterPair(name, self.thrusters[index_a],
                            self.thrusters[index_b])
        self.pairs[name] = pair
        return pair

    def torque(self):
        # Total torque from the thrusters currently on [N m].
        return sum((t.torque for t in self.thrusters if t.on),
                   start=np.zeros(3))

    def force(self):
        # Total force from the thrusters currently on [N]. Zero for a correctly paired
        # command; anything else belongs in the delta-v budget.
        return sum((t.force for t in self.thrusters if t.on),
                   start=np.zeros(3))

    def step(self, dt):
        # Advance by dt with the current on/off pattern. Returns propellant used [kg].
        impulse = 0.0
        for t in self.thrusters:
            if t.on:
                t.total_on_time += dt
                impulse += t.thrust * dt
        mass = impulse / (self.isp * self.g0)
        self.propellant_used += mass
        self.total_impulse += impulse
        return mass

    def all_off(self):
        for t in self.thrusters:
            t.on = False

    def command_torque(self, tau_desired, deadband_torque=1e-6):
        # Fire the pairs that best approximate a desired torque. Bang-bang, so the
        # delivered torque is one of a finite set; it is returned so the caller sees
        # the quantisation rather than assuming it got what it asked for.
        self.all_off()
        tau_desired = np.asarray(tau_desired, dtype=float)

        for pair in self.pairs.values():
            axis = pair.torque
            n = np.linalg.norm(axis)
            if n < 1e-12:
                continue
            projection = float(np.dot(tau_desired, axis)) / n
            if projection > deadband_torque:
                pair.fire(True)

        return self.torque()

    def min_impulse_torque(self, pair_name):
        # Smallest angular impulse a pair can deliver [N m s], which is the floor on
        # pointing accuracy in a limit cycle.
        pair = self.pairs[pair_name]
        return np.linalg.norm(pair.torque) * pair.a.min_impulse_bit

    def report(self):
        lines = [f"   {'thruster':<12}{'position [m]':>26}"
                 f"{'force dir':>22}{'|tau| [N m]':>13}",
                 "   " + "-" * 72]
        for t in self.thrusters:
            p, d = t.position, t.direction
            lines.append(
                f"   {t.name:<12}[{p[0]:+6.2f} {p[1]:+6.2f} {p[2]:+6.2f}]"
                f"   [{d[0]:+5.2f} {d[1]:+5.2f} {d[2]:+5.2f}]"
                f"{np.linalg.norm(t.torque):13.3f}")
        lines.append("")
        lines.append(f"   {'pair':<14}{'torque axis [N m]':>30}"
                     f"{'|tau|':>10}{'residual F [N]':>17}")
        lines.append("   " + "-" * 72)
        for name, p in self.pairs.items():
            tq = p.torque
            flag = "" if p.is_pure_couple else "   NOT A PURE COUPLE"
            lines.append(f"   {name:<14}[{tq[0]:+8.3f} {tq[1]:+8.3f} "
                         f"{tq[2]:+8.3f}]{np.linalg.norm(tq):10.3f}"
                         f"{p.residual_force:17.2e}{flag}")
        return "\n".join(lines)


# A standard layout

def standard_layout(arm=0.90, z=0.35, thrust=8.0, isp=220.0,
                    min_impulse_bit=0.02):
    # Twelve thrusters in four clusters at (+/-arm, +/-arm, z), giving pure couples
    # about all three axes. Each cluster has a +Z, a -Z and a tangential nozzle,
    # formed into six pairs, two per axis.

    # arm is the moment arm and the strongest lever on control authority: torque
    # scales with it linearly.
    corners = [(+1, +1), (-1, +1), (-1, -1), (+1, -1)]
    thrusters = []

    for k, (sx, sy) in enumerate(corners):
        p = [sx * arm, sy * arm, z]
        thrusters.append(Thruster(f"C{k+1}_up", p, [0, 0, +1],
                                  thrust, min_impulse_bit))
        thrusters.append(Thruster(f"C{k+1}_dn", p, [0, 0, -1],
                                  thrust, min_impulse_bit))
        # Tangential nozzle in the XY plane. Alternate corners point the
        # opposite way, otherwise there is no negative-roll authority.
        sense = 1.0 if (k % 2 == 0) else -1.0
        tang = sense * np.array([-sy, sx, 0.0], dtype=float)
        thrusters.append(Thruster(f"C{k+1}_tg", p, tang,
                                  thrust, min_impulse_bit))

    cluster = ThrusterCluster(thrusters, isp=isp)

    # Corner k occupies 3k (up), 3k+1 (down), 3k+2 (tangential).
    # Corners ordered (+,+), (-,+), (-,-), (+,-).
    # Pitch about Y: +Z on the +X side and -Z on the -X side.
    cluster.add_pair("pitch+", 0, 4)      # C1 up (+x+y), C2 down (-x+y)
    cluster.add_pair("pitch-", 1, 3)      # C1 down,      C2 up
    # Yaw about X: push +Z on the +Y side and -Z on the -Y side.
    cluster.add_pair("yaw+", 3, 7)        # C2 up (-x+y), C3 down (-x-y)
    cluster.add_pair("yaw-", 4, 6)        # C2 down,      C3 up
    # Roll about Z: tangential nozzles on opposite corners.
    cluster.add_pair("roll+", 2, 8)       # C1, C3 tangential: counter-clockwise
    cluster.add_pair("roll-", 5, 11)      # C2, C4 tangential: clockwise

    return cluster


if __name__ == "__main__":
    from inertia import build_descent_module
    from config import PropulsionConfig

    print("=" * 78)
    print(" REACTION CONTROL SYSTEM")
    print("=" * 78)
    print()

    rcs = standard_layout()
    print(rcs.report())

    m_prop = PropulsionConfig.MASS_WET - PropulsionConfig.MASS_DRY
    I_full = build_descent_module(m_prop).inertia()
    I_dry = build_descent_module(0.0).inertia()

    print()
    print("=" * 78)
    print(" CONTROL AUTHORITY   (inertia is PROVISIONAL: no CAD yet)")
    print("=" * 78)
    print()
    print(f"   {'pair':<10}{'|tau| [N m]':>13}{'alpha full [deg/s2]':>22}"
          f"{'alpha dry [deg/s2]':>21}")
    print("   " + "-" * 64)
    for name, pair in rcs.pairs.items():
        tq = np.linalg.norm(pair.torque)
        axis = pair.torque / tq
        i_full = float(axis @ (I_full @ axis))
        i_dry = float(axis @ (I_dry @ axis))
        print(f"   {name:<10}{tq:13.3f}{np.rad2deg(tq/i_full):22.4f}"
              f"{np.rad2deg(tq/i_dry):21.4f}")

    print()
    print(f"   minimum impulse bit  {rcs.thrusters[0].min_impulse_bit*1e3:.0f} ms")
    print(f"   min angular impulse  "
          f"{rcs.min_impulse_torque('pitch+'):.4f} N m s")
    dw = rcs.min_impulse_torque("pitch+") / I_full[1, 1]
    print(f"   -> smallest rate step {np.rad2deg(dw)*3600:.2f} deg/h "
          f"({np.rad2deg(dw):.3e} deg/s)")

    print()
    print("   Trajectory demand, from the descent run:")
    print("     thrust axis rotates  54.2 deg over 233.6 s")
    print("     typical rate         0.05 to 0.4 deg/s")
    print("     the RCS above exceeds this by a wide margin, so authority is")
    print("     set by slew manoeuvres and disturbance rejection, not by")
    print("     following the gravity turn.")
    print("=" * 78)