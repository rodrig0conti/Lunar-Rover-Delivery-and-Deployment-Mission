"""
visualization.py
================
Everything that draws. Two families of output:

    plot_*      static diagnostic figures (trajectory, ground track, altitude,
                integration-error check)
    animate_*   the 3D animation of the rotating Moon and the orbiting vehicle

A note on the animation
-----------------------
The Moon's sidereal rotation period is 27.3 days while a 100 km orbit takes
about 2 hours. Over a realistic run the body turns by only a few degrees, so
the rotation is real but essentially invisible. The `spin_exaggeration`
argument multiplies the *displayed* spin rate for clarity. It affects nothing
but the picture: the dynamics and the ground track are always computed with
the true rate. Leave it at 1.0 for any figure that goes into the report.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: F401

from config import SiteConfig

# Consistent look across all figures
_MOON_SURFACE = "#8c8c8c"
_MOON_GRID = "#5a5a5a"
_TRAJ = "#e05252"
_SITE = "#f5c542"
_ACCENT = "#4a90d9"


# ===========================================================================
# Static plots
# ===========================================================================


def plot_trajectory_3d(history, moon, ax=None, show_site: bool = True):
    """Inertial-frame trajectory around the Moon."""
    created = ax is None
    if created:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")

    x, y, z = moon.mesh(30, 60)
    ax.plot_surface(x, y, z, color=_MOON_SURFACE, alpha=0.55,
                    linewidth=0, antialiased=True, zorder=1)

    r = history.r
    ax.plot(r[:, 0], r[:, 1], r[:, 2], color=_TRAJ, lw=1.4,
            label="Trajectory (MCI)")
    ax.scatter(*r[0], color="white", edgecolor="k", s=45, label="Start")
    ax.scatter(*r[-1], color=_ACCENT, edgecolor="k", s=45, label="End")

    if show_site:
        p0 = moon.surface_point_mci(SiteConfig.LATITUDE,
                                    SiteConfig.LONGITUDE, history.t[0])
        ax.scatter(*p0, color=_SITE, edgecolor="k", s=60,
                   label="Landing site (t0)")

    _equal_aspect_3d(ax, history.radius.max() * 1.05)
    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")
    ax.set_title("Trajectory in the Moon-Centred Inertial frame")
    ax.legend(loc="upper right", fontsize=8)

    return ax.figure


def plot_groundtrack(history):
    """Sub-spacecraft track in the body-fixed frame."""
    fig, ax = plt.subplots(figsize=(11, 5.5))

    lon = np.rad2deg(history.lon)
    lat = np.rad2deg(history.lat)

    # Break the line where longitude wraps, so no spurious horizontal segments
    jumps = np.where(np.abs(np.diff(lon)) > 180.0)[0]
    segments = np.split(np.arange(len(lon)), jumps + 1)

    for k, seg in enumerate(segments):
        if len(seg) < 2:
            continue
        ax.plot(lon[seg], lat[seg], color=_TRAJ, lw=1.2,
                label="Ground track" if k == 0 else None)

    ax.scatter(lon[0], lat[0], color="white", edgecolor="k", s=45, zorder=5,
               label="Start")
    ax.scatter(np.rad2deg(SiteConfig.LONGITUDE),
               np.rad2deg(SiteConfig.LATITUDE),
               color=_SITE, edgecolor="k", s=90, marker="*", zorder=6,
               label="Landing site")

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(np.arange(-180, 181, 30))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.grid(alpha=0.3, ls=":")
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title("Ground track (Moon-Centred Moon-Fixed frame)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_diagnostics(history):
    """Altitude, speed, mass and integration-error panels."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    t_h = history.t / 3600.0

    ax = axes[0, 0]
    ax.plot(t_h, history.altitude, color=_TRAJ, lw=1.3)
    ax.set_ylabel("Altitude [km]")
    ax.set_title("Altitude above mean radius")

    ax = axes[0, 1]
    ax.plot(t_h, history.speed, color=_ACCENT, lw=1.3)
    ax.set_ylabel("Speed [km/s]")
    ax.set_title("Inertial speed")

    ax = axes[1, 0]
    ax.plot(t_h, history.mass, color="#5aa469", lw=1.3)
    ax.set_ylabel("Mass [kg]")
    ax.set_title("Vehicle mass")

    ax = axes[1, 1]
    ax.semilogy(t_h, np.maximum(history.energy_error, 1e-18),
                color="#9b59b6", lw=1.3)
    ax.set_ylabel("|ΔE / E₀| [-]")
    ax.set_title("Relative energy drift (integration quality)")

    for ax in axes.ravel():
        ax.set_xlabel("Time [h]")
        ax.grid(alpha=0.3, ls=":")

    fig.suptitle("Run diagnostics", fontsize=13)
    fig.tight_layout()
    return fig


# ===========================================================================
# Animation
# ===========================================================================


def animate_orbit(history, moon, n_frames: int = 240,
                  trail_length: int = 1600,
                  spin_exaggeration: float = 1.0,
                  elev: float = 22.0, azim0: float = -60.0,
                  rotate_camera: bool = False,
                  filename: str = None, fps: int = 25):
    """
    Animate the rotating Moon with the vehicle moving along its trajectory.

    Parameters
    ----------
    history           : finalised History object
    moon              : Moon instance
    n_frames          : number of animation frames (history is subsampled)
    trail_length      : samples of trajectory drawn behind the vehicle
    spin_exaggeration : display-only multiplier on the Moon's spin rate
    rotate_camera     : slowly orbit the viewpoint as well
    filename          : if given, save as GIF (Pillow writer)

    Returns
    -------
    (fig, anim). Keep a reference to `anim` alive or the animation stops.
    """
    idx = np.linspace(0, len(history.t) - 1, n_frames).astype(int)

    fig = plt.figure(figsize=(8.5, 7.5))
    # computed_zorder=False turns off matplotlib's automatic depth sorting.
    # With it on, the Moon mesh is treated as a single object and hides every
    # line artist regardless of actual depth, so the orbit disappears. With it
    # off, artists are drawn in explicit zorder: the body first, the vehicle on
    # top. The trade is that the far half of the orbit is drawn over the Moon
    # instead of behind it, which is the convention most orbit viewers use.
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    fig.patch.set_facecolor("#0d0d14")
    ax.set_facecolor("#0d0d14")

    # --- Static sphere. A sphere is rotationally symmetric, so it need only
    # --- be drawn once; the visible rotation comes from the grid lines.
    xs, ys, zs = moon.mesh(28, 56)
    ax.plot_surface(xs, ys, zs, color=_MOON_SURFACE, alpha=1.0,
                    linewidth=0, antialiased=True, shade=True, zorder=0)

    # --- Body-fixed reference lines, rebuilt each frame after rotation
    grid_lines_bf = _body_fixed_grid(moon)
    grid_artists = [ax.plot([], [], [], color=_MOON_GRID, lw=0.8,
                            alpha=0.55, zorder=1)[0] for _ in grid_lines_bf]

    site_bf = moon.surface_point_mcmf(SiteConfig.LATITUDE,
                                      SiteConfig.LONGITUDE)
    site_artist, = ax.plot([], [], [], marker="*", color=_SITE,
                           markersize=15, linestyle="none", zorder=3,
                           label="Landing site")

    trail_artist, = ax.plot([], [], [], color=_TRAJ, lw=1.8, alpha=0.95,
                            zorder=4, label="Trajectory")
    craft_artist, = ax.plot([], [], [], marker="o", color="white",
                            markersize=8, linestyle="none", zorder=5,
                            markeredgecolor=_ACCENT, markeredgewidth=1.2,
                            label="Spacecraft")

    hud = fig.text(0.03, 0.93, "", color="white",
                   fontsize=9, family="monospace", va="top")

    lim = history.radius.max() * 1.02
    _equal_aspect_3d(ax, lim)
    ax.set_axis_off()
    # The 3D bounding box leaves generous margins by default; shrink them so
    # the body fills the frame. Legend and title are therefore anchored to the
    # figure rather than the axes, or they would fall outside the canvas.
    ax.set_position([-0.08, -0.10, 1.16, 1.10])
    fig.legend(loc="upper right", bbox_to_anchor=(0.985, 0.94), fontsize=8,
               facecolor="#1a1a24", edgecolor="none", labelcolor="white")
    fig.text(0.5, 0.965,
             "Lunar orbit — Step 1: central body and orbital dynamics",
             color="white", ha="center", fontsize=12)

    def update(frame):
        i = idx[frame]
        t = history.t[i]
        theta = moon.omega * t * spin_exaggeration

        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

        for artist, line_bf in zip(grid_artists, grid_lines_bf):
            pts = line_bf @ rot.T
            artist.set_data(pts[:, 0], pts[:, 1])
            artist.set_3d_properties(pts[:, 2])

        site = rot @ site_bf
        site_artist.set_data([site[0]], [site[1]])
        site_artist.set_3d_properties([site[2]])

        j0 = max(0, i - trail_length)
        trail = history.r[j0:i + 1]
        trail_artist.set_data(trail[:, 0], trail[:, 1])
        trail_artist.set_3d_properties(trail[:, 2])

        p = history.r[i]
        craft_artist.set_data([p[0]], [p[1]])
        craft_artist.set_3d_properties([p[2]])

        hud.set_text(
            f"t        {t / 3600.0:8.3f} h\n"
            f"altitude {history.altitude[i]:8.2f} km\n"
            f"speed    {history.speed[i]:8.4f} km/s\n"
            f"mass     {history.mass[i]:8.2f} kg\n"
            f"lat/lon  {np.rad2deg(history.lat[i]):6.1f} / "
            f"{np.rad2deg(history.lon[i]):6.1f} deg"
        )

        if rotate_camera:
            ax.view_init(elev=elev, azim=azim0 + 360.0 * frame / n_frames)

        return grid_artists + [site_artist, trail_artist, craft_artist, hud]

    ax.view_init(elev=elev, azim=azim0)

    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                   interval=1000 / fps, blit=False)

    if filename:
        anim.save(filename, writer=animation.PillowWriter(fps=fps),
                  savefig_kwargs={"facecolor": fig.get_facecolor()})
        print(f"Animation written to {filename}")

    return fig, anim


# ===========================================================================
# Helpers
# ===========================================================================


def _body_fixed_grid(moon, n_meridians: int = 12, n_parallels: int = 5,
                     n_pts: int = 120):
    """Meridians and parallels as body-fixed point arrays, shape (n_pts, 3)."""
    lines = []
    r = moon.radius * 1.002  # lift slightly so lines are not hidden by the mesh

    for lon in np.linspace(-np.pi, np.pi, n_meridians, endpoint=False):
        lat = np.linspace(-np.pi / 2, np.pi / 2, n_pts)
        lines.append(np.column_stack([
            r * np.cos(lat) * np.cos(lon),
            r * np.cos(lat) * np.sin(lon),
            r * np.sin(lat),
        ]))

    for lat in np.linspace(-60.0, 60.0, n_parallels):
        la = np.deg2rad(lat)
        lon = np.linspace(-np.pi, np.pi, n_pts)
        lines.append(np.column_stack([
            r * np.cos(la) * np.cos(lon),
            r * np.cos(la) * np.sin(lon),
            np.full(n_pts, r * np.sin(la)),
        ]))

    return lines


def _equal_aspect_3d(ax, lim: float) -> None:
    """Force an equal aspect ratio so the Moon looks spherical."""
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass
