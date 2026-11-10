# Attitude control: a pointing requirement turned into thruster commands.

#     1. guidance gives the inertial direction the thrust axis must take
#     2. a target quaternion puts the body thrust axis there
#     3. the attitude error becomes a desired torque
#     4. the RCS approximates that torque with on/off pairs

# Step 4 is where the limit cycle comes from: the controller asks for a
# continuous torque and the valves can only give a fixed set of values, so the
# vehicle drifts across the deadband, gets kicked back and drifts again.

import numpy as np

from attitude import (quat_multiply, quat_conjugate, quat_normalise,
                      quat_to_dcm, dcm_to_quat, rotate_body_to_inertial)


def attitude_for_pointing(direction_inertial, body_axis=(0.0, 0.0, 1.0),
                          reference=None):
    # Quaternion that points body_axis along direction_inertial.
    # Pointing one axis leaves the roll about it free; reference picks it, by
    # default whatever keeps the other axes closest to the inertial frame.
    d = np.asarray(direction_inertial, dtype=float)
    d = d / np.linalg.norm(d)
    b = np.asarray(body_axis, dtype=float)
    b = b / np.linalg.norm(b)

    # Build an inertial frame whose third column is d
    if reference is None:
        reference = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(reference, d)) > 0.95:
            reference = np.array([0.0, 1.0, 0.0])
    ref = np.asarray(reference, dtype=float)

    z = d
    x = ref - np.dot(ref, z) * z
    nx = np.linalg.norm(x)
    if nx < 1e-9:
        x = np.array([1.0, 0.0, 0.0]) - z[0] * z
        nx = np.linalg.norm(x)
    x = x / nx
    y = np.cross(z, x)
    R_target = np.column_stack([x, y, z])       # body -> inertial

    # If the body axis is not +Z, pre-rotate so that b maps onto +Z first
    if not np.allclose(b, [0.0, 0.0, 1.0]):
        v = np.cross(b, [0.0, 0.0, 1.0])
        s = np.linalg.norm(v)
        c = float(np.dot(b, [0.0, 0.0, 1.0]))
        if s < 1e-12:
            R_align = np.eye(3) if c > 0 else -np.eye(3)
        else:
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
            R_align = np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)
        R_target = R_target @ R_align

    return dcm_to_quat(R_target)


class AttitudeController:
    # Quaternion feedback with rate damping, feeding a bang-bang RCS:
    #     tau = -Kp sign(qe_w) qe_vec - Kd omega
    # The sign term is there because q and -q are the same attitude: without it
    # the controller takes the long way round, rotating 300 deg to achieve 60.

    def __init__(self, rcs, inertia, kp=None, kd=None,
                 deadband_deg=0.5, rate_deadband_deg_s=0.05,
                 body_thrust_axis=(0.0, 0.0, 1.0)):
        self.rcs = rcs
        self.inertia = np.asarray(inertia, dtype=float)
        self.body_thrust_axis = np.asarray(body_thrust_axis, dtype=float)

        # Gains scaled by inertia so the response does not change as the tanks
        # drain. kd = 2 sqrt(kp I) for critical damping: over-damping gives a
        # steady tracking lag against a moving target.
        i_mean = float(np.trace(self.inertia) / 3.0)
        self.kp = 20.0 * i_mean if kp is None else kp
        self.kd = 2.0 * np.sqrt(self.kp * i_mean) if kd is None else kd

        self.deadband = np.deg2rad(deadband_deg)
        self.rate_deadband = np.deg2rad(rate_deadband_deg_s)

        self.target_quat = None
        self.last_error_angle = 0.0
        self.last_torque = np.zeros(3)
        self.firing_time = 0.0

    def set_target_direction(self, direction_inertial):
        # Point the body thrust axis along a given inertial direction.
        self.target_quat = attitude_for_pointing(direction_inertial,
                                                 self.body_thrust_axis)

    def set_target_quat(self, q):
        self.target_quat = quat_normalise(np.asarray(q, dtype=float))

    def error(self, q):
        # Error quaternion and pointing angle. qe = q_target^-1 (x) q is the rotation
        # still to undo, in body axes, which is the frame the torques act in.
        if self.target_quat is None:
            return np.array([1.0, 0.0, 0.0, 0.0]), 0.0
        qe = quat_multiply(quat_conjugate(self.target_quat), q)
        qe = quat_normalise(qe)
        angle = 2.0 * np.arccos(np.clip(abs(qe[0]), -1.0, 1.0))
        return qe, angle

    def update(self, q, omega_body, dt):
        # Compute and issue thruster commands for this step.
        # Returns (torque_actual, force_actual). The force is zero for correctly paired
        # thrusters; a non-zero value is a translational disturbance on the trajectory.
        qe, angle = self.error(q)
        self.last_error_angle = angle

        rate = float(np.linalg.norm(omega_body))
        if angle < self.deadband and rate < self.rate_deadband:
            # inside the deadband, firing would only start a limit cycle
            self.rcs.all_off()
            self.last_torque = np.zeros(3)
            return np.zeros(3), np.zeros(3)

        sign = 1.0 if qe[0] >= 0.0 else -1.0
        tau_desired = -self.kp * sign * qe[1:4] - self.kd * omega_body

        tau_actual = self.rcs.command_torque(tau_desired)
        self.last_torque = tau_actual
        if np.any([t.on for t in self.rcs.thrusters]):
            self.firing_time += dt

        return tau_actual, self.rcs.force()

    def slew_time_estimate(self, angle, axis_inertia, torque):
        # Bang-bang minimum-time slew, accelerate then decelerate: t = 2 sqrt(I a / tau).
        # Checks that a reorientation fits in the time available before a burn.
        if torque <= 0.0:
            return np.inf
        return 2.0 * np.sqrt(axis_inertia * angle / torque)


def lvlh_quaternion(r_inertial, v_inertial, nadir_axis=(0.0, 0.0, 1.0)):
    # Attitude of a vehicle held in the local-vertical / local-horizontal frame,
    # the usual parking attitude. nadir_axis selects which body axis goes on nadir;
    # with the default +Z the body frame comes out as
    #     +Z  nadir
    #     +X  along the velocity vector
    #     +Y  completes the right-handed set (anti-orbit-normal)

    # Kinematic definition, not propagated dynamics: it says where the vehicle
    # would point with a perfect controller and costs no propellant. Used for the
    # vehicles whose attitude control is not modelled, the stack and the relay.
    # The lander after separation integrates its attitude from torques instead.
    r = np.asarray(r_inertial, dtype=float)
    v = np.asarray(v_inertial, dtype=float)

    nadir = -r / np.linalg.norm(r)
    h = np.cross(r, v)
    orbit_normal = h / np.linalg.norm(h)
    along = np.cross(nadir, orbit_normal)          # along the velocity vector
    along = along / np.linalg.norm(along)

    # Columns are where the body axes end up in inertial coordinates.
    R = np.column_stack([along, np.cross(nadir, along), nadir])

    b = np.asarray(nadir_axis, dtype=float)
    b = b / np.linalg.norm(b)
    if not np.allclose(b, [0.0, 0.0, 1.0]):
        vv = np.cross(b, [0.0, 0.0, 1.0])
        s = np.linalg.norm(vv)
        cth = float(np.dot(b, [0.0, 0.0, 1.0]))
        if s < 1e-12:
            R_align = np.eye(3) if cth > 0 else np.diag([1.0, -1.0, -1.0])
        else:
            vx = np.array([[0, -vv[2], vv[1]],
                           [vv[2], 0, -vv[0]],
                           [-vv[1], vv[0], 0]])
            R_align = np.eye(3) + vx + vx @ vx * ((1 - cth) / s**2)
        R = R @ R_align

    return dcm_to_quat(R)