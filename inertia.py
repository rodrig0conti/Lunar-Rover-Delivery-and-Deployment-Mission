# Inertia tensors, built from a component breakdown rather than assumed, so
# that changing the configuration changes the answer.

# The tensor is not constant: the descent module burns 46% of its mass as
# propellant held in off-centreline tanks, so both the inertia and the centre
# of mass move as they drain. Both are computed against propellant remaining.

# SI throughout: metres, kilograms, kg m^2. The trajectory code works in km.

# All geometry here is a Phase A placeholder.

import numpy as np


# Primitive shapes, about their own centre of mass, in body axes

def solid_cylinder(mass, radius, height, axis=2):
    # Solid cylinder. axis is the symmetry axis: 0=x, 1=y, 2=z.
    i_axial = 0.5 * mass * radius**2
    i_trans = mass * (3.0 * radius**2 + height**2) / 12.0
    d = [i_trans, i_trans, i_trans]
    d[axis] = i_axial
    return np.diag(d)


def hollow_cylinder(mass, radius, height, axis=2):
    # Thin-walled cylindrical shell: a structure panel, not a solid billet.
    i_axial = mass * radius**2
    i_trans = mass * (6.0 * radius**2 + height**2) / 12.0
    d = [i_trans, i_trans, i_trans]
    d[axis] = i_axial
    return np.diag(d)


def solid_sphere(mass, radius):
    # Solid sphere: a full propellant tank.
    return np.eye(3) * (0.4 * mass * radius**2)


def spherical_shell(mass, radius):
    # Thin spherical shell: an empty tank.
    return np.eye(3) * (2.0 / 3.0 * mass * radius**2)


def solid_box(mass, a, b, c):
    # Rectangular box with sides a, b, c along x, y, z.
    return np.diag([mass * (b*b + c*c) / 12.0,
                    mass * (a*a + c*c) / 12.0,
                    mass * (a*a + b*b) / 12.0])


def flat_plate(mass, width, height, normal=0):
    # Thin flat plate, e.g. a solar array. normal is its thin direction.
    dims = [0.0, 0.0, 0.0]
    idx = [i for i in range(3) if i != normal]
    dims[idx[0]] = width
    dims[idx[1]] = height
    return solid_box(mass, *dims)


def thin_rod(mass, length, axis=2):
    # Thin rod along axis: a landing leg strut.
    i = mass * length**2 / 12.0
    d = [i, i, i]
    d[axis] = 0.0
    return np.diag(d)


# Composition

def parallel_axis(inertia, mass, offset):
    # Shift an inertia tensor from its own centre of mass by offset:
    #     I_new = I_cm + m (|d|^2 I - d (x) d)
    # The familiar m d^2 is the diagonal entry perpendicular to the offset.
    d = np.asarray(offset, dtype=float)
    return inertia + mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))


class Component:
    # One item in the mass breakdown: mass, position, and its own inertia.

    def __init__(self, name, mass, position, inertia_cm):
        self.name = name
        self.mass = float(mass)
        self.position = np.asarray(position, dtype=float)
        self.inertia_cm = np.asarray(inertia_cm, dtype=float)

    def __repr__(self):
        return f"<{self.name}: {self.mass:.1f} kg at {self.position}>"


class Assembly:
    # A collection of components, with the combined properties.

    def __init__(self, name="vehicle"):
        self.name = name
        self.components = []

    def add(self, component):
        self.components.append(component)
        return self

    @property
    def mass(self):
        return sum(c.mass for c in self.components)

    @property
    def centre_of_mass(self):
        # Mass-weighted mean position [m].
        m = self.mass
        if m <= 0.0:
            return np.zeros(3)
        return sum(c.mass * c.position for c in self.components) / m

    def inertia(self, about=None):
        # Combined inertia tensor [kg m^2], about the centre of mass by default,
        # which is the point the rotational equations are written about.
        origin = self.centre_of_mass if about is None else np.asarray(about, float)
        total = np.zeros((3, 3))
        for c in self.components:
            total += parallel_axis(c.inertia_cm, c.mass, c.position - origin)
        return total

    def principal(self, about=None):
        # Principal moments (eigenvalues) and axes (eigenvectors).
        # Their order matters: spinning about the intermediate one is unstable.
        vals, vecs = np.linalg.eigh(self.inertia(about))
        return vals, vecs

    def report(self, about=None):
        # Formatted breakdown.
        origin = self.centre_of_mass if about is None else np.asarray(about, float)
        lines = [f"   {'component':<22}{'mass [kg]':>11}"
                 f"{'position [m]':>26}",
                 "   " + "-" * 60]
        for c in self.components:
            p = c.position
            lines.append(f"   {c.name:<22}{c.mass:11.2f}"
                         f"    [{p[0]:+6.3f} {p[1]:+6.3f} {p[2]:+6.3f}]")
        lines.append("   " + "-" * 60)
        lines.append(f"   {'TOTAL':<22}{self.mass:11.2f}")
        com = self.centre_of_mass
        lines.append(f"   centre of mass        "
                     f"     [{com[0]:+6.3f} {com[1]:+6.3f} {com[2]:+6.3f}] m")
        I = self.inertia(origin)
        lines.append("")
        lines.append("   inertia tensor about the centre of mass [kg m^2]:")
        for row in I:
            lines.append(f"     [{row[0]:9.2f} {row[1]:9.2f} {row[2]:9.2f}]")
        vals, _ = self.principal(origin)
        lines.append(f"   principal moments      "
                     f"{vals[0]:8.2f} {vals[1]:8.2f} {vals[2]:8.2f}")
        return "\n".join(lines)


# Descent module, Phase A placeholder. Every dimension is invented: no CAD
# exists yet. The method (component breakdown, parallel axis, variation with
# propellant) is trustworthy, the specific numbers are not. Swap in the real
# dimensions when CAD exists and everything downstream updates.

PROVISIONAL = True

class DescentModuleConfig:
    # Placeholder geometry. Body axes: +Z along the thrust axis (engine at -Z),
    # origin at the geometric centre of the bus.
    BUS_RADIUS = 0.90          # m
    BUS_HEIGHT = 0.80          # m
    BUS_MASS = 70.0            # kg, structure and avionics

    TANK_RADIUS = 0.35         # m
    TANK_COUNT = 4
    TANK_RING_RADIUS = 0.62    # m, from the centreline
    TANK_DRY_MASS = 8.0        # kg each

    ENGINE_MASS = 22.0         # kg
    ENGINE_Z = -0.55           # m

    LEG_COUNT = 4
    LEG_MASS = 9.0             # kg each
    LEG_SPAN = 1.75            # m, footpad radius
    LEG_Z = -0.60              # m

    PANEL_MASS = 19.15         # kg total, from the power budget
    PANEL_WIDTH = 1.50         # m
    PANEL_HEIGHT = 0.75        # m
    PANEL_ARM = 1.15           # m from the centreline

    ROVER_MASS = 50.0          # kg
    ROVER_Z = 0.65             # m, stowed on top

    AVIONICS_MASS = 20.0       # kg, distributed in the bus


def build_descent_module(propellant_mass, cfg=DescentModuleConfig):
    # Assemble the descent module with a given propellant load.
    # Propellant is split equally between the tanks and modelled as a shrinking
    # solid sphere: crude, but it captures the falling mass and the falling
    # off-axis contribution.
    a = Assembly("Descent Module")

    a.add(Component("bus structure", cfg.BUS_MASS, [0, 0, 0],
                    hollow_cylinder(cfg.BUS_MASS, cfg.BUS_RADIUS,
                                    cfg.BUS_HEIGHT)))
    a.add(Component("avionics", cfg.AVIONICS_MASS, [0, 0, 0.10],
                    solid_box(cfg.AVIONICS_MASS, 0.4, 0.4, 0.3)))
    a.add(Component("engine", cfg.ENGINE_MASS, [0, 0, cfg.ENGINE_Z],
                    solid_cylinder(cfg.ENGINE_MASS, 0.25, 0.5)))

    # tanks, evenly spaced around the centreline
    m_prop_each = propellant_mass / cfg.TANK_COUNT
    for k in range(cfg.TANK_COUNT):
        ang = 2.0 * np.pi * k / cfg.TANK_COUNT
        pos = [cfg.TANK_RING_RADIUS * np.cos(ang),
               cfg.TANK_RING_RADIUS * np.sin(ang), 0.0]
        a.add(Component(f"tank {k+1} shell", cfg.TANK_DRY_MASS, pos,
                        spherical_shell(cfg.TANK_DRY_MASS, cfg.TANK_RADIUS)))
        if m_prop_each > 1e-6:
            # radius of the liquid, assuming it fills the tank from the bottom
            frac = min(1.0, m_prop_each / (propellant_mass / cfg.TANK_COUNT + 1e-9))
            r_liq = cfg.TANK_RADIUS * frac ** (1.0 / 3.0)
            a.add(Component(f"tank {k+1} propellant", m_prop_each, pos,
                            solid_sphere(m_prop_each, r_liq)))

    # landing legs, modelled as rods from the bus to the footpads
    for k in range(cfg.LEG_COUNT):
        ang = 2.0 * np.pi * k / cfg.LEG_COUNT + np.pi / 4.0
        pos = [0.5 * cfg.LEG_SPAN * np.cos(ang),
               0.5 * cfg.LEG_SPAN * np.sin(ang), cfg.LEG_Z]
        a.add(Component(f"leg {k+1}", cfg.LEG_MASS, pos,
                        thin_rod(cfg.LEG_MASS, cfg.LEG_SPAN, axis=0)))

    # solar arrays, two panels facing the orbit normal (see power_geometry)
    m_panel = cfg.PANEL_MASS / 2.0
    for sign in (+1.0, -1.0):
        a.add(Component(f"array {'+' if sign > 0 else '-'}", m_panel,
                        [sign * cfg.PANEL_ARM, 0.0, 0.0],
                        flat_plate(m_panel, cfg.PANEL_WIDTH,
                                   cfg.PANEL_HEIGHT, normal=0)))

    a.add(Component("rover (stowed)", cfg.ROVER_MASS, [0, 0, cfg.ROVER_Z],
                    solid_box(cfg.ROVER_MASS, 0.9, 0.7, 0.5)))
    return a


if __name__ == "__main__":
    from config import PropulsionConfig

    m_prop_full = PropulsionConfig.MASS_WET - PropulsionConfig.MASS_DRY

    print("=" * 66)
    print(" DESCENT MODULE — MASS PROPERTIES")
    print("=" * 66)
    print()
    full = build_descent_module(m_prop_full)
    print(full.report())

    print()
    print("=" * 66)
    print(" VARIATION THROUGH THE DESCENT")
    print("=" * 66)
    print()
    print(f"   {'prop [kg]':>11}{'mass [kg]':>11}{'CoM z [m]':>11}"
          f"{'Ixx':>9}{'Iyy':>9}{'Izz':>9}")
    print("   " + "-" * 60)
    for frac in (1.0, 0.75, 0.5, 0.25, 0.0):
        m_p = m_prop_full * frac
        a = build_descent_module(m_p)
        I = a.inertia()
        com = a.centre_of_mass
        print(f"   {m_p:11.1f}{a.mass:11.1f}{com[2]:11.4f}"
              f"{I[0,0]:9.2f}{I[1,1]:9.2f}{I[2,2]:9.2f}")

    I_full = build_descent_module(m_prop_full).inertia()
    I_dry = build_descent_module(0.0).inertia()
    print()
    print(f"   Ixx falls by {100*(1-I_dry[0,0]/I_full[0,0]):.1f}% as the tanks drain,")
    print(f"   Izz by {100*(1-I_dry[2,2]/I_full[2,2]):.1f}%. A controller tuned for the")
    print("   full vehicle is over-damped when empty, and one tuned for the")
    print("   empty vehicle is sluggish when full. Both cases must be checked.")
    print("=" * 66)

    if PROVISIONAL:
        print()
        print("!" * 66)
        print(" PROVISIONAL: geometry is assumed, not designed. No CAD exists.")
        print(" Absolute inertia values are not yet defensible. Ratios and")
        print(" trends (34-40% variation with propellant) are structural and")
        print(" will survive a change of dimensions.")
        print("!" * 66)