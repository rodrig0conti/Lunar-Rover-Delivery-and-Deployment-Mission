"""
moon.py
=======
Everything that belongs to the central body: its gravity field, its rotation,
and the transformation between the two reference frames used by the simulator.

Reference frames
----------------
MCI  - Moon-Centred Inertial. Origin at the Moon's centre of mass, non-rotating.
       This is the frame in which the equations of motion are integrated.

MCMF - Moon-Centred Moon-Fixed. Origin at the centre of mass, rotating with the
       body at OMEGA about +Z. Landing sites, ground tracks and terrain live
       here. At t = 0 the two frames are defined to coincide.

The distinction matters: the landing site is fixed in MCMF but moves in MCI,
so descent targeting is fundamentally a problem of hitting a moving point.
"""

import numpy as np

from config import MoonConstants


class Moon:
    """Central body model: gravitational field, rotation and frame handling."""

    def __init__(self, enable_j2: bool = True):
        self.mu = MoonConstants.MU
        self.radius = MoonConstants.RADIUS_MEAN
        self.omega = MoonConstants.OMEGA
        self.j2 = MoonConstants.J2
        self.enable_j2 = enable_j2

        # Angular velocity vector in MCI. The spin axis is +Z by construction.
        self.omega_vec = np.array([0.0, 0.0, self.omega])

    # -----------------------------------------------------------------
    # Gravity
    # -----------------------------------------------------------------

    def gravity(self, r_mci: np.ndarray) -> np.ndarray:
        """
        Gravitational acceleration at position `r_mci` [km], returned in MCI
        [km/s^2].

        Point-mass term plus, optionally, the J2 oblateness term. J2 is the
        largest departure from spherical symmetry and is what makes near-polar
        orbits precess; it is the reason the mission baseline uses 86 deg
        rather than exactly 90 deg.
        """
        r = np.linalg.norm(r_mci)
        if r < 1e-9:
            raise ValueError("Position vector is at the centre of the body.")

        a = -self.mu * r_mci / r**3

        if self.enable_j2:
            a = a + self._j2_acceleration(r_mci, r)

        return a

    def _j2_acceleration(self, r_vec: np.ndarray, r: float) -> np.ndarray:
        """Acceleration due to the J2 zonal harmonic, in MCI [km/s^2]."""
        x, y, z = r_vec
        k = 1.5 * self.j2 * self.mu * self.radius**2 / r**5
        zr2 = 5.0 * (z / r) ** 2

        return np.array([
            k * x * (zr2 - 1.0),
            k * y * (zr2 - 1.0),
            k * z * (zr2 - 3.0),
        ])

    def gravity_potential(self, r_mci: np.ndarray) -> float:
        """
        Gravitational potential energy per unit mass [km^2/s^2].
        Point-mass only, used for the energy-conservation diagnostic.
        """
        return -self.mu / np.linalg.norm(r_mci)

    # -----------------------------------------------------------------
    # Rotation and frame transformations
    # -----------------------------------------------------------------

    def rotation_angle(self, t: float) -> float:
        """Rotation angle of the body-fixed frame relative to MCI at time t."""
        return self.omega * t

    def dcm_mci_to_mcmf(self, t: float) -> np.ndarray:
        """
        Direction cosine matrix taking a vector from MCI to MCMF at time t.
        A simple rotation about +Z by -theta.
        """
        th = self.rotation_angle(t)
        c, s = np.cos(th), np.sin(th)
        return np.array([
            [c, s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0],
        ])

    def dcm_mcmf_to_mci(self, t: float) -> np.ndarray:
        """Inverse of `dcm_mci_to_mcmf`; for a rotation matrix, the transpose."""
        return self.dcm_mci_to_mcmf(t).T

    def to_mcmf(self, r_mci: np.ndarray, t: float) -> np.ndarray:
        """Convert a position from MCI to MCMF."""
        return self.dcm_mci_to_mcmf(t) @ r_mci

    def to_mci(self, r_mcmf: np.ndarray, t: float) -> np.ndarray:
        """Convert a position from MCMF to MCI."""
        return self.dcm_mcmf_to_mci(t) @ r_mcmf

    def velocity_to_mcmf(self, r_mci: np.ndarray, v_mci: np.ndarray,
                         t: float) -> np.ndarray:
        """
        Convert an inertial velocity to the rotating frame.

        v_rel = R (v_inertial - omega x r_inertial)

        The transport term is what makes a ground track drift westward even for
        a perfectly polar orbit.
        """
        v_rel_mci = v_mci - np.cross(self.omega_vec, r_mci)
        return self.dcm_mci_to_mcmf(t) @ v_rel_mci

    # -----------------------------------------------------------------
    # Surface geometry
    # -----------------------------------------------------------------

    def surface_point_mcmf(self, lat: float, lon: float,
                           alt: float = 0.0) -> np.ndarray:
        """Body-fixed position of a point on (or above) the surface [km]."""
        r = self.radius + alt
        return np.array([
            r * np.cos(lat) * np.cos(lon),
            r * np.cos(lat) * np.sin(lon),
            r * np.sin(lat),
        ])

    def surface_point_mci(self, lat: float, lon: float, t: float,
                          alt: float = 0.0) -> np.ndarray:
        """Inertial position, at time t, of a point fixed to the surface."""
        return self.to_mci(self.surface_point_mcmf(lat, lon, alt), t)

    def geodetic(self, r_mci: np.ndarray, t: float):
        """
        Sub-spacecraft latitude, longitude and altitude.

        Returns
        -------
        (lat, lon, alt) : latitude and longitude in rad, altitude in km,
                          relative to a spherical Moon of mean radius.
        """
        r_bf = self.to_mcmf(r_mci, t)
        r = np.linalg.norm(r_bf)
        lat = np.arcsin(r_bf[2] / r)
        lon = np.arctan2(r_bf[1], r_bf[0])
        return lat, lon, r - self.radius

    def altitude(self, r_mci: np.ndarray) -> float:
        """Altitude above the mean spherical surface [km]."""
        return np.linalg.norm(r_mci) - self.radius

    # -----------------------------------------------------------------
    # Orbit helpers
    # -----------------------------------------------------------------

    def circular_velocity(self, r: float) -> float:
        """Circular orbital speed at radius r [km/s]."""
        return np.sqrt(self.mu / r)

    def orbital_period(self, sma: float) -> float:
        """Keplerian period for a given semi-major axis [s]."""
        return 2.0 * np.pi * np.sqrt(sma**3 / self.mu)

    def mesh(self, n_lat: int = 40, n_lon: int = 80):
        """
        Spherical surface mesh in the body-fixed frame, for visualisation.
        Returns (X, Y, Z) arrays of shape (n_lat, n_lon).
        """
        lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
        lon = np.linspace(-np.pi, np.pi, n_lon)
        lon_g, lat_g = np.meshgrid(lon, lat)

        x = self.radius * np.cos(lat_g) * np.cos(lon_g)
        y = self.radius * np.cos(lat_g) * np.sin(lon_g)
        z = self.radius * np.sin(lat_g)
        return x, y, z
