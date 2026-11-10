# Lunar environment: rotation, gravity, solar geometry (J2 and mascons later)

import numpy as np
from config import MoonConstants


class Moon:
    # Central body model. Two frames, coincident at t = 0:
    #   MCI  - inertial, does not rotate. Equations of motion are integrated here.
    #   MCMF - rotates with the body. Landing site, terrain and ground track.

    def __init__(self, mu: float = MoonConstants.MU,
                 omega: float = MoonConstants.OMEGA,
                 radius: float = MoonConstants.RADIUS_MEAN):
        self.mu = mu
        self.omega = omega
        self.radius = radius

    # Gravity

    def gravity(self, r_vec: np.ndarray) -> np.ndarray:
        # Point-mass gravitational acceleration in MCI [km/s^2].
        r = np.linalg.norm(r_vec)
        return -self.mu * r_vec / r**3

    # Rotation

    def rotation_angle(self, t: float) -> float:
        # Angle turned since t = 0 [rad].
        return self.omega * t

    def dcm_mcmf_to_mci(self, t: float) -> np.ndarray:
        # Body-fixed -> inertial at time t.
        theta = self.rotation_angle(t)
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ])

    def dcm_mci_to_mcmf(self, t: float) -> np.ndarray:
        # Inertial -> body-fixed.
        return self.dcm_mcmf_to_mci(t).T

    def to_mci(self, r_mcmf: np.ndarray, t: float) -> np.ndarray:
        # Position from body-fixed to inertial.
        return self.dcm_mcmf_to_mci(t) @ r_mcmf

    def to_mcmf(self, r_mci: np.ndarray, t: float) -> np.ndarray:
        # Position from inertial to body-fixed.
        return self.dcm_mci_to_mcmf(t) @ r_mci

    # Surface geometry

    def surface_point_mcmf(self, lat: float, lon: float,
                           alt: float = 0.0) -> np.ndarray:
        # Body-fixed position of a surface point. Angles in rad, alt in km.
        r = self.radius + alt
        return np.array([
            r * np.cos(lat) * np.cos(lon),
            r * np.cos(lat) * np.sin(lon),
            r * np.sin(lat),
        ])

    def surface_point_mci(self, lat: float, lon: float, t: float,
                          alt: float = 0.0) -> np.ndarray:
        # Same point in inertial space at time t.
        return self.to_mci(self.surface_point_mcmf(lat, lon, alt), t)

    def geodetic(self, r_mci: np.ndarray, t: float):
        # Sub-spacecraft latitude, longitude [rad] and altitude [km].
        r_bf = self.to_mcmf(r_mci, t)
        r = np.linalg.norm(r_bf)
        lat = np.arcsin(r_bf[2] / r)
        lon = np.arctan2(r_bf[1], r_bf[0])
        return lat, lon, r - self.radius


    # Solar geometry

    def sun_direction_mci(self, t: float) -> np.ndarray:
        # Unit vector Moon -> Sun in MCI. The ecliptic is tilted from the lunar equator
        # by OBLIQUITY, which is what drives the seasonal illumination at the poles.
        from config import SolarConstants as S

        lam = S.RA0 + S.OMEGA * t          # ecliptic longitude
        eps = S.OBLIQUITY
        return np.array([
            np.cos(lam),
            np.sin(lam) * np.cos(eps),
            np.sin(lam) * np.sin(eps),
        ])

    def solar_elevation(self, lat: float, lon: float, t: float) -> float:
        # Sun elevation over the local horizon [rad], negative below it.
        # Flat horizon: real terrain at Shackleton needs the LOLA elevation map.
        n_bf = self.surface_point_mcmf(lat, lon)
        n_bf = n_bf / np.linalg.norm(n_bf)            # local vertical, MCMF
        s_bf = self.to_mcmf(self.sun_direction_mci(t), t)
        return np.arcsin(np.clip(np.dot(n_bf, s_bf), -1.0, 1.0))

    def is_sunlit(self, r_mci: np.ndarray, t: float) -> bool:
        # Sunlight test for a point in orbit. Cylindrical umbra, so the penumbra
        # (a couple of seconds per pass) is ignored.
        s = self.sun_direction_mci(t)
        along = np.dot(r_mci, s)
        if along >= 0.0:
            return True                                # sunward hemisphere
        perp = np.linalg.norm(r_mci - along * s)       # distance from the axis
        return perp > self.radius

    def solar_incidence(self, normal_mci: np.ndarray, t: float) -> float:
        # cos of the angle between a surface normal and the Sun, clipped at 0.
        # Times flux and panel area, this is the power in.
        n = normal_mci / np.linalg.norm(normal_mci)
        return max(0.0, float(np.dot(n, self.sun_direction_mci(t))))