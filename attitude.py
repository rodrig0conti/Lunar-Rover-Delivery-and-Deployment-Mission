# Rotational kinematics and rigid-body dynamics.
# Independent of the translational code: rotation about the centre of mass and
# the motion of that centre of mass only couple through the thrust direction,
# so each can be verified on its own.

# State:
#     y = [qw, qx, qy, qz, wx, wy, wz]
#     q  unit quaternion, body -> inertial, scalar first
#     w  angular velocity relative to inertial, in BODY axes [rad/s]

# Body axes for w because the inertia tensor is constant there, which is what
# gives Euler equations their form. Quaternions rather than Euler angles
# because a descent that pitches over 90 deg runs straight into gimbal lock.

import numpy as np


# Quaternion algebra

def quat_identity():
    # The rotation that does nothing.
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_normalise(q):
    # Scale to unit length. Integration drifts off the unit sphere every step, and
    # a non-unit quaternion scales every rotation it performs, so this runs always.
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else quat_identity()


def quat_multiply(a, b):
    # Hamilton product a (x) b: rotation b followed by rotation a.
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])


def quat_conjugate(q):
    # Inverse rotation, for a unit quaternion.
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_dcm(q):
    # BODY -> INERTIAL rotation matrix: v_inertial = quat_to_dcm(q) @ v_body.
    w, x, y, z = quat_normalise(q)
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def dcm_to_quat(R):
    # Rotation matrix -> quaternion, using the numerically stable branch.
    tr = np.trace(R)
    if tr > 0.0:
        s = 0.5 / np.sqrt(tr + 1.0)
        return np.array([0.25 / s,
                         (R[2, 1] - R[1, 2]) * s,
                         (R[0, 2] - R[2, 0]) * s,
                         (R[1, 0] - R[0, 1]) * s])
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[i + 1] = 0.25 * s
    q[j + 1] = (R[j, i] + R[i, j]) / s
    q[k + 1] = (R[k, i] + R[i, k]) / s
    return quat_normalise(q)


def quat_from_axis_angle(axis, angle):
    # Rotation of angle radians about axis.
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return quat_identity()
    a = a / n
    s = np.sin(0.5 * angle)
    return np.array([np.cos(0.5 * angle), a[0]*s, a[1]*s, a[2]*s])


def quat_derivative(q, w_body):
    # q_dot = 0.5 q (x) [0, w]. The half is there because a quaternion carries half
    # the rotation angle in its trigonometric arguments.
    return 0.5 * quat_multiply(q, np.array([0.0, w_body[0], w_body[1], w_body[2]]))


def rotate_body_to_inertial(q, v_body):
    # Express a body-frame vector in inertial coordinates.
    return quat_to_dcm(q) @ np.asarray(v_body, dtype=float)


def rotate_inertial_to_body(q, v_inertial):
    # Express an inertial vector in body coordinates.
    return quat_to_dcm(q).T @ np.asarray(v_inertial, dtype=float)


# Rigid body

class RigidBody:
    # Rotation about the centre of mass. Euler equations in body axes:
    #     I w_dot = tau - w x (I w)
    # The cross term is non-zero unless w lies along a principal axis, so a freely
    # spinning body does not generally spin about a fixed axis.

    def __init__(self, inertia, name="body"):
        self.name = name
        self.inertia = np.asarray(inertia, dtype=float)
        if self.inertia.ndim == 1:                     # principal axes given
            self.inertia = np.diag(self.inertia)
        self.inertia_inv = np.linalg.inv(self.inertia)

        # commanded torque in body axes [N m], set by a controller
        self.torque = np.zeros(3)

    def initial_state(self, q=None, w_body=None):
        # Assemble a rotational state vector.
        q = quat_identity() if q is None else quat_normalise(np.asarray(q, float))
        w = np.zeros(3) if w_body is None else np.asarray(w_body, float)
        return np.concatenate([q, w])

    def derivative(self, t, y):
        # y = [q (4), w (3)], dydt = [q_dot, I^-1 (tau - w x I w)].
        q = y[0:4]
        w = y[4:7]

        q_dot = quat_derivative(q, w)
        w_dot = self.inertia_inv @ (self.torque - np.cross(w, self.inertia @ w))

        dydt = np.empty(7)
        dydt[0:4] = q_dot
        dydt[4:7] = w_dot
        return dydt

    # Diagnostics: exact invariants with zero torque, so their drift
    # measures integration error.

    def angular_momentum_body(self, y):
        # h in body axes [N m s]. Its magnitude is conserved, not the vector.
        return self.inertia @ y[4:7]

    def angular_momentum_inertial(self, y):
        # h in inertial axes. Conserved as a vector when torque is zero.
        return rotate_body_to_inertial(y[0:4], self.angular_momentum_body(y))

    def kinetic_energy(self, y):
        # Rotational kinetic energy [J]: 0.5 w . I w.
        w = y[4:7]
        return 0.5 * float(w @ (self.inertia @ w))

    def set_torque(self, torque_body):
        # Apply a torque in body axes [N m].
        self.torque = np.asarray(torque_body, dtype=float)

    def clear_torque(self):
        self.torque = np.zeros(3)

    def body_axes_inertial(self, y):
        # The three body axes as inertial unit vectors, for drawing a body triad.
        R = quat_to_dcm(y[0:4])
        return R[:, 0], R[:, 1], R[:, 2]


# Standalone propagation, for verification before coupling to translation

def propagate_attitude(body, y0, t_final, dt=0.01, t0=0.0, log_every=1):
    # Integrate rotational motion alone. Returns a dict of time series.
    # RK4 does not know q must stay on the unit sphere, so it is renormalised
    # after every step.
    from mathutils import rk4_step

    t = t0
    y = np.array(y0, dtype=float)
    ts, ys = [t], [y.copy()]
    step = 0

    while t < t_final - 1e-12:
        h = min(dt, t_final - t)
        y = rk4_step(body.derivative, t, y, h)
        y[0:4] = quat_normalise(y[0:4])
        t += h
        step += 1
        if step % log_every == 0:
            ts.append(t)
            ys.append(y.copy())

    ts = np.asarray(ts)
    ys = np.asarray(ys)
    return {
        "t": ts,
        "states": ys,
        "q": ys[:, 0:4],
        "w": ys[:, 4:7],
        "h_inertial": np.array([body.angular_momentum_inertial(s) for s in ys]),
        "energy": np.array([body.kinetic_energy(s) for s in ys]),
    }