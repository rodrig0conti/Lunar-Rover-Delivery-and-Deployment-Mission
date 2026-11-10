# Everything that animates or plots... mostly AI I guess, don't 
# want to focus too much on it, but want it to look nice

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.image as mpimg
import time
from abc import ABC, abstractmethod


def _sphere_mesh(radius, n_lat=60, n_lon=120):
    # Sphere mesh. Latitude runs north -> south so row 0 of the mesh matches
    # row 0 of an equirectangular image (north pole).
    lat = np.linspace(np.pi/2, -np.pi/2, n_lat)
    lon = np.linspace(-np.pi, np.pi, n_lon)
    lon_g, lat_g = np.meshgrid(lon, lat)
    return (radius * np.cos(lat_g) * np.cos(lon_g),
            radius * np.cos(lat_g) * np.sin(lon_g),
            radius * np.sin(lat_g))


def _load_texture(path, n_lat=60, n_lon=120):
    # Load an equirectangular map, downsampled to the mesh resolution.
    # Returns None if the file is missing so the animation still runs.
    try:
        img = mpimg.imread(path)
    except (FileNotFoundError, OSError):
        print(f"[visualization] texture '{path}' not found, using plain grey")
        return None

    if img.dtype == np.uint8:
        img = img.astype(float) / 255.0
    if img.ndim == 2:                      # greyscale
        img = np.dstack([img] * 3)
    img = img[:, :, :3]                    # drop alpha if present

    h, w = img.shape[:2]
    rows = np.linspace(0, h - 1, n_lat).astype(int)
    cols = np.linspace(0, w - 1, n_lon).astype(int)
    return img[np.ix_(rows, cols)]


def _grid_lines(radius, n_meridians=12, n_parallels=5, n_pts=100):
    # Meridians and parallels as body-fixed point arrays.
    r = radius * 1.002          # lift slightly so they sit above the surface
    lines = []
    for lon in np.linspace(-np.pi, np.pi, n_meridians, endpoint=False):
        lat = np.linspace(-np.pi/2, np.pi/2, n_pts)
        lines.append(np.column_stack([r*np.cos(lat)*np.cos(lon),
                                      r*np.cos(lat)*np.sin(lon),
                                      r*np.sin(lat)]))
    for lat_deg in np.linspace(-60, 60, n_parallels):
        la = np.deg2rad(lat_deg)
        lon = np.linspace(-np.pi, np.pi, n_pts)
        lines.append(np.column_stack([r*np.cos(la)*np.cos(lon),
                                      r*np.cos(la)*np.sin(lon),
                                      np.full(n_pts, r*np.sin(la))]))
    return lines
def animate_moon(moon, t_end, site_latlon=None, n_frames=180,
                 spin_speedup=1.0, texture="moon_texture.jpg",
                 n_lat=60, n_lon=120, show_grid=True, filename=None):
    # Rotating Moon. spin_speedup multiplies only the displayed rotation.
    # The body is rotated by rolling the texture in longitude: for an
    # equirectangular map and a spin about the polar axis that is the same
    # operation, and far cheaper than rebuilding 7200 polygons.
    times = np.linspace(0.0, t_end, n_frames)
    tex = _load_texture(texture, n_lat, n_lon)
    X, Y, Z = _sphere_mesh(moon.radius, n_lat, n_lon)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    # Dark background. Three separate things have to be set: the figure
    # patch, the axes patch (this is the white square), and the 3D panes.
    fig.patch.set_facecolor("#05050a")
    ax.set_facecolor("#05050a")
    ax.patch.set_alpha(0.0)

    lines_bf = _grid_lines(moon.radius) if show_grid else []
    grid_artists = [ax.plot([], [], [], color="#c8c8c8", lw=0.5,
                            alpha=0.25, zorder=2)[0] for _ in lines_bf]

    site_bf = None
    site_artist = None
    if site_latlon is not None:
        site_bf = moon.surface_point_mcmf(*np.deg2rad(site_latlon))
        site_artist, = ax.plot([], [], [], marker="*", color="#f5c542",
                               markersize=16, linestyle="none", zorder=3)

    lim = moon.radius * 1.02
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_position([-0.10, -0.10, 1.20, 1.20])
    hud = fig.text(0.04, 0.95, "", color="white", family="monospace",
                   fontsize=9, va="top")

    surf = {"artist": None}      # dict so the closure can reassign it

    def update(k):
        t = times[k]
        theta = moon.rotation_angle(t * spin_speedup)   # display only

        # --- body: roll the texture instead of rotating the mesh
        if surf["artist"] is not None:
            surf["artist"].remove()
        if tex is None:
            surf["artist"] = ax.plot_surface(
                X, Y, Z, color="#8c8c8c", rstride=2, cstride=2,
                linewidth=0, antialiased=False, shade=True, zorder=0)
        else:
            shift = int(round(theta / (2*np.pi) * n_lon))
            surf["artist"] = ax.plot_surface(
                X, Y, Z, facecolors=np.roll(tex, shift, axis=1),
                rstride=1, cstride=1, linewidth=0,
                antialiased=False, shade=False, zorder=0)

        # --- grid and site: rotated with the real DCM
        rot = moon.dcm_mcmf_to_mci(t * spin_speedup)
        for art, line in zip(grid_artists, lines_bf):
            p = line @ rot.T
            art.set_data(p[:, 0], p[:, 1])
            art.set_3d_properties(p[:, 2])

        if site_artist is not None:
            s = rot @ site_bf
            site_artist.set_data([s[0]], [s[1]])
            site_artist.set_3d_properties([s[2]])

        hud.set_text(f"t        = {t/86400:8.2f} days\n"
                     f"rotation = {np.rad2deg(moon.rotation_angle(t)) % 360:8.1f} deg")
        return []

    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                   interval=50, blit=False)
    if filename:
        anim.save(filename, writer=animation.PillowWriter(fps=20),
                  savefig_kwargs={"facecolor": fig.get_facecolor()})
    return fig, anim


# Interactive 3D scene (PyVista / VTK). matplotlib has no real 3D engine:
# no texture interpolation and frame cost grows with mesh density.


def _texture_coordinates(mesh, radius):
    # Equirectangular UV coordinates: u = longitude, v = latitude, both in [0, 1].
    p = mesh.points
    u = 0.5 + np.arctan2(-p[:, 0], p[:, 1]) / (2.0 * np.pi)
    v = 0.5 + np.arcsin(np.clip(p[:, 2] / radius, -1.0, 1.0)) / np.pi
    return np.c_[u, v]

def _triad(radius, tip_scale=1.45, from_surface=True,
           shaft=0.006, tip=0.020):
    # Three arrows along +X, +Y, +Z.
    #     tip_scale     where the tip sits, as a multiple of the radius
    #     from_surface  start the arrow at the surface, not the centre. Depth
    #                   testing hides the inside anyway; this is only legibility
    import pyvista as pv

    out = {}
    for name, d in (("x", (1, 0, 0)), ("y", (0, 1, 0)), ("z", (0, 0, 1))):
        d = np.array(d, dtype=float)
        start = d * radius if from_surface else np.zeros(3)
        length = radius * tip_scale - np.linalg.norm(start)
        out[name] = pv.Arrow(start=start, direction=d, scale=length,
                             shaft_radius=shaft, tip_radius=tip,
                             tip_length=0.14)
    return out
def scene_moon(moon, site_latlon=None, texture="moon_texture.jpg",
               theta_res=180, phi_res=90,
               show_mcmf_frame=True, show_mci_frame=True,
               frame_labels=True):
    # Interactive VTK scene of the Moon. Returns (pl, body_actors).
    # body_actors holds everything rigidly attached to the body; anything
    # inertial (MCI triad, orbit traces, Sun arrow) must stay out of it.
    import pyvista as pv

    sphere = pv.Sphere(radius=moon.radius,
                       theta_resolution=theta_res,
                       phi_resolution=phi_res,
                       start_theta=270.001, end_theta=270)
    sphere.active_texture_coordinates = _texture_coordinates(sphere, moon.radius)

    pl = pv.Plotter(window_size=(1000, 800))
    pl.background_color = "#05050a"

    body_actors = []

    # ambient/diffuse/specular make the surface respond to the Sun light added
    # later; without them the terminator is invisible.
    try:
        tex = pv.read_texture(texture)
        body_actors.append(pl.add_mesh(sphere, texture=tex,
                                       smooth_shading=True,
                                       ambient=0.02, diffuse=1.0,
                                       specular=0.0))
    except (FileNotFoundError, OSError):
        print(f"[visualization] texture '{texture}' not found, using plain grey")
        body_actors.append(pl.add_mesh(sphere, color="#8c8c8c",
                                       smooth_shading=True,
                                       ambient=0.02, diffuse=1.0,
                                       specular=0.0))

    # --- landing site marker: body-fixed
    if site_latlon is not None:
        lat, lon = np.deg2rad(site_latlon)
        pt = moon.surface_point_mcmf(lat, lon)
        body_actors.append(
            pl.add_mesh(pv.Sphere(radius=moon.radius * 0.001, center=pt),
                        color="#f5c542"))

    # --- MCMF triad: rotates with the body
    if show_mcmf_frame:
        colours = {"x": "#ff5555", "y": "#55ff88", "z": "#5599ff"}
        for axis, mesh in _triad(moon.radius, tip_scale=1.45).items():
            body_actors.append(pl.add_mesh(mesh, color=colours[axis]))
        if frame_labels:
            L = moon.radius * 1.42
            pl.add_point_labels(
                np.array([[L, 0, 0], [0, L, 0], [0, 0, L]]),
                ["X_MCMF", "Y_MCMF", "Z_MCMF"],
                text_color="#ffdddd", font_size=11, shape=None,
                always_visible=True, name="mcmf_labels")

    # --- MCI triad: inertial, NOT in body_actors
    if show_mci_frame:
        for mesh in _triad(moon.radius, tip_scale=1.85,
                           shaft=0.003, tip=0.012).values():
            pl.add_mesh(mesh, color="#dddddd", opacity=0.5)
        if frame_labels:
            L = moon.radius * 1.92
            pl.add_point_labels(
                np.array([[L, 0, 0], [0, L, 0], [0, 0, L]]),
                ["X_MCI", "Y_MCI", "Z_MCI"],
                text_color="#bbbbbb", font_size=11, shape=None,
                always_visible=True, name="mci_labels")

    pl.camera.position = (moon.radius * 5, -moon.radius * 2, moon.radius * 1.8)
    pl.camera.focal_point = (0.0, 0.0, 0.0)
    return pl, body_actors

def show_moon_rotating(moon, t_start=0.0, t_end=None,
                       site_latlon=None, texture="moon_texture.jpg",
                       time_scale=1.0, fps=30,
                       show_mcmf_frame=True, show_mci_frame=True,
                       filename=None):
    # Interactive window advancing mission time at time_scale mission seconds per
    # wall second. Every object is placed by asking where it is at mission time t;
    # only the rate at which t advances changes, so bodies never desynchronise.
    # Keys: +/- time scale by 2x, space pause, r reset.
    if t_end is None:
        from config import MoonConstants
        t_end = MoonConstants.SIDEREAL_PERIOD

    pl, body_actors = scene_moon(moon, site_latlon, texture,
                                 show_mcmf_frame=show_mcmf_frame,
                                 show_mci_frame=show_mci_frame)

    clock = {"t": t_start, "scale": float(time_scale), "paused": False}

    def set_body_attitude(t):
        # Absolute attitude of the body group at mission time t. Recomputed from t
        # each frame rather than accumulated, so pausing and rewinding work.
        theta_deg = np.rad2deg(moon.rotation_angle(t))
        for actor in body_actors:
            actor.orientation = (0.0, 0.0, theta_deg)

    hud = pl.add_text("", position=(0.02, 0.86), color="white", font_size=9)

    def refresh_hud():
        hud.input = (
            f"t          {clock['t'] / 86400.0:10.4f} days\n"
            f"rotation   {np.rad2deg(moon.rotation_angle(clock['t'])) % 360:10.2f} deg\n"
            f"time scale {clock['scale']:10.0f} x"
            + ("   [PAUSED]" if clock["paused"] else "")
        )

    pl.add_key_event("plus",  lambda: clock.update(scale=clock["scale"] * 2.0))
    pl.add_key_event("minus", lambda: clock.update(scale=max(1.0, clock["scale"] / 2.0)))
    pl.add_key_event("space", lambda: clock.update(paused=not clock["paused"]))
    pl.add_key_event("r",     lambda: clock.update(t=t_start))

    set_body_attitude(clock["t"])
    refresh_hud()
    pl.show(auto_close=False, interactive_update=True)

    if filename:
        pl.open_movie(filename, framerate=fps)

    while clock["t"] < t_end:
        if not clock["paused"]:
            clock["t"] += clock["scale"] / fps      # mission seconds per frame
        set_body_attitude(clock["t"])
        refresh_hud()
        pl.update()
        if filename:
            pl.write_frame()

    if filename:
        pl.close()
    else:
        pl.show()

    return pl

def _prepare_mesh(path, display_size):
    # Load an OBJ, centre it on its centroid and scale it to display_size km across.
    # Display only: a 3 m spacecraft beside a 1737 km Moon is below one pixel.
    import pyvista as pv

    mesh = pv.read(path)
    mesh.points = mesh.points - np.array(mesh.center)

    bounds = np.array(mesh.bounds).reshape(3, 2)
    extent = np.max(bounds[:, 1] - bounds[:, 0])
    if extent > 0:
        mesh.points = mesh.points * (display_size / extent)
    return mesh

def add_spacecraft(pl, craft, display_size, show_trace=True,
                   trace_period=None, n_trace=400):
    # Add a vehicle to an existing scene. Returns (actors, centre), in MCI.
    import pyvista as pv

    mesh, rgb = (read_obj_with_vertex_colours(craft.mesh_path)
                 if craft.mesh_path else (None, None))

    if mesh is not None:
        mesh.points = mesh.points - np.array(mesh.center)
        b = np.array(mesh.bounds).reshape(3, 2)
        extent = np.max(b[:, 1] - b[:, 0])
        if extent > 0:
            mesh.points = mesh.points * (display_size / extent)

        if rgb is not None:
            mesh["rgb"] = rgb
            actors = [pl.add_mesh(mesh, scalars="rgb", rgb=True,
                                  smooth_shading=True)]
        else:
            actors = [pl.add_mesh(mesh, color=craft.colour,
                                  smooth_shading=True)]
    else:
        actors = [pl.add_mesh(pv.Sphere(radius=display_size * 0.5),
                              color=craft.colour)]

    centre = np.zeros(3)
    place_group(actors, centre, craft.position)

    if show_trace and trace_period:
        ts = np.linspace(0.0, trace_period, n_trace)
        pts = np.array([craft.state_at(t)[0:3] for t in ts])
        # opacity 1.0: translucent actors skip the depth buffer in VTK, which
        # would put the far half of the orbit on top of the Moon.
        pl.add_mesh(pv.lines_from_points(pts), color=craft.colour,
                    line_width=2)

    return actors, centre
def show_mission(moon, crafts, t_start=0.0, t_end=None,
                 site_latlon=None, texture="moon_texture.jpg",
                 display_size=70.0, time_scale=1.0, fps=30,
                 show_mcmf_frame=True, show_mci_frame=True,
                 show_sun=True):
    # Animate the Moon, the Sun and every vehicle on one shared mission clock.
    # Only the rate at which t advances is scaled, so the relative geometry stays
    # right at any playback speed. Keys: +/- time scale, space pause, r reset.
    from config import MoonConstants, OrbitConfig
    from mathutils import orbital_period

    if t_end is None:
        t_end = MoonConstants.SIDEREAL_PERIOD
    period = orbital_period(moon.mu, OrbitConfig.SMA)

    pl, body_actors = scene_moon(moon, site_latlon, texture,
                                 show_mcmf_frame=show_mcmf_frame,
                                 show_mci_frame=show_mci_frame)

    groups = [(c,) + add_spacecraft(pl, c, display_size, trace_period=period)
              for c in crafts]

    light, sun_arrow = (add_sun(pl, moon, t_start) if show_sun
                        else (None, None))

    clock = {"t": t_start, "scale": float(time_scale), "paused": False}
    hud = pl.add_text("", position=(0.02, 0.86), color="white", font_size=9)

    def update_scene(t):
        theta = np.rad2deg(moon.rotation_angle(t))
        for a in body_actors:
            a.orientation = (0.0, 0.0, theta)
        for craft, actors, centre in groups:
            place_group(actors, centre, craft.state_at(t)[0:3])
        if show_sun:
            update_sun(moon, t, light, sun_arrow)

    def refresh_hud():
        t = clock["t"]
        lines = [f"t          {t/3600.0:10.3f} h",
                 f"time scale {clock['scale']:10.0f} x"
                 + ("   [PAUSED]" if clock["paused"] else "")]

        for craft, _, _ in groups:
            s = craft.state_at(t)
            alt = np.linalg.norm(s[0:3]) - moon.radius
            sunlit = "sun" if moon.is_sunlit(s[0:3], t) else "ECLIPSE"
            lines.append(f"{craft.name:<14} alt {alt:7.1f} km  "
                         f"v {np.linalg.norm(s[3:6]):6.4f} km/s  {sunlit}")

        if site_latlon is not None:
            elev = np.rad2deg(moon.solar_elevation(
                np.deg2rad(site_latlon[0]), np.deg2rad(site_latlon[1]), t))
            lines.append(f"sun elev @site {elev:7.2f} deg")

        hud.input = "\n".join(lines)

    pl.add_key_event("plus",  lambda: clock.update(scale=clock["scale"] * 2.0))
    pl.add_key_event("minus", lambda: clock.update(scale=max(1.0, clock["scale"] / 2.0)))
    pl.add_key_event("space", lambda: clock.update(paused=not clock["paused"]))
    pl.add_key_event("r",     lambda: clock.update(t=t_start))

    update_scene(clock["t"])
    refresh_hud()

    pl.show(auto_close=False, interactive_update=True)
    wall_prev = time.perf_counter()

    while clock["t"] < t_end:
        now = time.perf_counter()
        dt_wall = now - wall_prev
        wall_prev = now

        if not clock["paused"]:
            clock["t"] += clock["scale"] * dt_wall

        update_scene(clock["t"])
        refresh_hud()

        try:
            pl.update()
        except RuntimeError:
            break
        if not pl.render_window or pl.render_window.GetNeverRendered():
            break

    pl.close()
    return pl

def read_obj_with_vertex_colours(path):
    # Parse an OBJ directly, keeping per-vertex colours and discarding vt/vn.
    # pv.read() is avoided because VTK splits vertices that carry different
    # texture coordinates (so colours no longer map), and because vt without a
    # texture image pushes VTK onto a shader path that fails on some drivers.
    # Returns (mesh, rgb), rgb an (N,3) uint8 array or None.
    import pyvista as pv

    pts, cols, faces = [], [], []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            if p[0] == "v":
                pts.append([float(p[1]), float(p[2]), float(p[3])])
                cols.append([float(p[4]), float(p[5]), float(p[6])]
                            if len(p) >= 7 else None)
            elif p[0] == "f":
                # 'f v', 'f v/vt', 'f v//vn', 'f v/vt/vn': keep v only
                idx = [int(tok.split("/")[0]) for tok in p[1:]]
                n = len(pts)
                faces.append([(i - 1) if i > 0 else (n + i) for i in idx])

    if not pts or not faces:
        return None, None

    flat = []
    for fc in faces:
        flat.extend([len(fc)] + list(fc))
    mesh = pv.PolyData(np.array(pts, dtype=float), faces=np.array(flat))

    rgb = None
    if cols and all(c is not None for c in cols):
        c = np.array(cols, dtype=float)
        if c.max() <= 1.5:               # some exporters write 0-1, others 0-255
            c = c * 255.0
        rgb = c.astype(np.uint8)

    return mesh, rgb

def load_obj_group(pl, path, display_size):
    # Import an OBJ with its materials. pv.read() ignores the .mtl, so this uses
    # pl.import_obj(), which creates one actor per material that must be moved
    # together. Scaling goes through the actor transform since pv.wrap() returns
    # a copy. Returns (actors, centroid); the centroid is needed by place_group.
    before = set(pl.renderer.actors.keys())
    pl.import_obj(path)
    actors = [a for k, a in pl.renderer.actors.items() if k not in before]

    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for a in actors:
        b = np.array(a.GetMapper().GetInput().GetBounds()).reshape(3, 2)
        lo = np.minimum(lo, b[:, 0])
        hi = np.maximum(hi, b[:, 1])

    centre = (lo + hi) / 2.0
    extent = float(np.max(hi - lo))
    scale = display_size / extent if extent > 0 else 1.0

    for a in actors:
        a.SetOrigin(*centre)          # scale about the model's own centroid
        a.SetScale(scale, scale, scale)

    return actors, centre


def place_group(actors, centre, target):
    # Move a material group so its centroid sits at target (MCI, km).
    # The actor transform maps the centroid to centre + position, so subtract it.
    p = np.asarray(target, dtype=float) - np.asarray(centre, dtype=float)
    for a in actors:
        a.SetPosition(*p)


def add_sun(pl, moon, t, show_arrow=True, show_lighting=True,
            intensity=1.4, fill=0.06):
    # Add solar illumination and, optionally, an arrow marking the Sun.
    # Returns (light, arrow_actor) so the loop can update both. The terminator
    # shows at a glance where the surface is lit and where an orbit enters eclipse.
    import pyvista as pv

    d = moon.sun_direction_mci(t)

    light = None
    if show_lighting:
        pl.remove_all_lights()
        light = pv.Light(position=tuple(d * moon.radius * 100),
                         focal_point=(0.0, 0.0, 0.0),
                         color="white", light_type="scene light")
        light.positional = False
        light.intensity = intensity
        pl.add_light(light)
        # A faint headlight keeps the night side from going pure black, so the
        # geometry stays readable. Cosmetic only.
        pl.add_light(pv.Light(light_type="headlight", intensity=fill))

    arrow = None
    if show_arrow:
        L = moon.radius * 2.4
        mesh = pv.Arrow(start=tuple(d * moon.radius * 1.15), direction=tuple(d),
                        scale=L - moon.radius * 1.15,
                        shaft_radius=0.004, tip_radius=0.016, tip_length=0.12)
        arrow = pl.add_mesh(mesh, color="#ffd24a")

    return light, arrow


def update_sun(moon, t, light, arrow_actor):
    # Point the light and the arrow at the Sun direction for time t.
    d = moon.sun_direction_mci(t)
    if light is not None:
        light.position = tuple(d * moon.radius * 100)
        light.focal_point = (0.0, 0.0, 0.0)
    if arrow_actor is not None:
        # The arrow was built along the t0 direction; rotate it to the new one.
        arrow_actor.orientation = _direction_to_euler(d)


def _direction_to_euler(d):
    # Euler angles (deg) rotating +X onto the unit vector d.
    d = np.asarray(d, dtype=float)
    d = d / np.linalg.norm(d)
    yaw = np.degrees(np.arctan2(d[1], d[0]))
    pitch = np.degrees(np.arcsin(np.clip(d[2], -1.0, 1.0)))
    return (0.0, -pitch, yaw)

def show_descent(moon, history, guidance,
                 site_latlon=(-89.0, 0.0),
                 texture="moon_texture.jpg",
                 time_scale=60.0, fps=30,
                 theta_res=120, phi_res=60,
                 window=(1000, 800),
                 follow=False):
    # Play back an integrated descent trajectory by interpolating a History.
    # The physics is finished, so playback speed cannot alter it.
    # follow keeps the camera at a distance proportional to altitude, which is
    # the only way to see both a 100 km coast and a 5 m touchdown in one run.
    import pyvista as pv
    import time as _time

    W, H = window
    t_arr = history.t
    r_arr = history.r

    sphere = pv.Sphere(radius=moon.radius,
                       theta_resolution=theta_res, phi_resolution=phi_res,
                       start_theta=270.001, end_theta=270)
    sphere.active_texture_coordinates = _texture_coordinates(sphere, moon.radius)

    pl = pv.Plotter(window_size=(W, H))
    pl.background_color = "#05050a"

    body_actors = []
    try:
        tex = pv.read_texture(texture)
        body_actors.append(pl.add_mesh(sphere, texture=tex,
                                       smooth_shading=True, ambient=0.05,
                                       diffuse=1.0, specular=0.0))
    except (FileNotFoundError, OSError):
        body_actors.append(pl.add_mesh(sphere, color="#8c8c8c",
                                       smooth_shading=True, ambient=0.05,
                                       diffuse=1.0, specular=0.0))

    if site_latlon is not None:
        pt = moon.surface_point_mcmf(*np.deg2rad(site_latlon))
        body_actors.append(
            pl.add_mesh(pv.Sphere(radius=moon.radius * 0.001, center=pt),
                        color="#f5c542"))

    pl.add_mesh(pv.lines_from_points(r_arr), color="#ff6644", line_width=2)
    craft = pl.add_mesh(pv.Sphere(radius=moon.radius * 0.001), color="white")

    light, _ = add_sun(pl, moon, t_arr[0], show_arrow=False)

    # Camera along the orbit normal: the whole arc is then seen face-on
    # instead of edge-on from an arbitrary viewpoint.
    n = np.cross(r_arr[0], r_arr[len(r_arr) // 4])
    n = n / np.linalg.norm(n)
    pl.camera.position = tuple(n * moon.radius * 5.5)
    pl.camera.focal_point = (0.0, 0.0, 0.0)
    pl.camera.up = tuple(r_arr[0] / np.linalg.norm(r_arr[0]))

    # PyVista reads a tuple position as PIXELS, not normalised coordinates.
    hud = pl.add_text("", position=(14, H - 130), color="white", font_size=10)

    def phase_at(t):
        phase = guidance.events[0]["phase"]
        for e in guidance.events:
            if t >= e["t"]:
                phase = e["phase"]
        return phase

    def position_at(t):
        return np.array([np.interp(t, t_arr, r_arr[:, i]) for i in range(3)])

    clock = {"t": t_arr[0], "scale": float(time_scale), "paused": False}

    def update(t):
        p = position_at(t)
        craft.SetPosition(*p)
        theta = np.rad2deg(moon.rotation_angle(t))
        for a in body_actors:
            a.orientation = (0.0, 0.0, theta)
        update_sun(moon, t, light, None)

        if follow:
            alt = np.interp(t, t_arr, history.altitude)
            d = max(alt * 6.0, 3.0)
            pl.camera.focal_point = tuple(p)
            pl.camera.position = tuple(p + n * d)
            pl.camera.up = tuple(p / np.linalg.norm(p))

        hud.input = (
            f"t       {t/60:9.2f} min\n"
            f"phase   {phase_at(t):>9}\n"
            f"alt     {np.interp(t, t_arr, history.altitude):9.3f} km\n"
            f"speed   {np.interp(t, t_arr, history.speed)*1e3:9.1f} m/s\n"
            f"mass    {np.interp(t, t_arr, history.mass):9.1f} kg\n"
            f"scale   {clock['scale']:9.0f} x"
            + ("  [PAUSED]" if clock["paused"] else ""))

    pl.add_key_event("plus", lambda: clock.update(scale=clock["scale"] * 2.0))
    pl.add_key_event("minus", lambda: clock.update(scale=max(1.0, clock["scale"] / 2.0)))
    pl.add_key_event("space", lambda: clock.update(paused=not clock["paused"]))
    pl.add_key_event("r", lambda: clock.update(t=t_arr[0]))

    update(clock["t"])
    pl.show(auto_close=False, interactive_update=True)
    wall_prev = _time.perf_counter()

    while clock["t"] < t_arr[-1]:
        now = _time.perf_counter()
        dt_wall = now - wall_prev
        wall_prev = now
        if not clock["paused"]:
            clock["t"] = min(clock["t"] + clock["scale"] * dt_wall, t_arr[-1])
        update(clock["t"])
        try:
            pl.update()
        except RuntimeError:
            break
        if not pl.render_window or pl.render_window.GetNeverRendered():
            break

    pl.show()
    return pl

def show_full_mission(moon, h_stack, h_relay, h_lander, guidance, t_sep,
                      site_latlon=(-89.0, 0.0),
                      texture="moon_texture.jpg",
                      relay_mesh="satellite.obj",
                      lander_mesh="satellite.obj",
                      display_size=70.0, offset_factor=0.8,
                      time_scale=180.0, fps=30,
                      theta_res=120, phi_res=60,
                      window=(1000, 800),
                      show_frames=True, show_thrust_axis=True,
                      show_body_frames=True, frame_scale=2.0):
    # Play back the full mission: attached orbit, separation, divergence.
    # Three trajectories are drawn statically and the vehicles move along them.

    # MCI is drawn in white and never moves. MCMF is drawn in colour and joins
    # body_actors, so it turns with the Moon. The angle between the two X axes is
    # the rotation angle at that instant.

    # The orange arrow is the COMMANDED thrust direction. Once attitude is
    # coupled the achieved body axis lags it, and the gap is the control error.
    import time as _time
    import pyvista as pv
    from attitude import quat_to_dcm, quat_normalise
    from control import lvlh_quaternion

    W, H = window

    # --- Moon -------------------------------------------------------
    sphere = pv.Sphere(radius=moon.radius, theta_resolution=theta_res,
                       phi_resolution=phi_res,
                       start_theta=270.001, end_theta=270)
    sphere.active_texture_coordinates = _texture_coordinates(sphere, moon.radius)

    pl = pv.Plotter(window_size=(W, H))
    pl.background_color = "#05050a"

    body_actors = []
    try:
        tex = pv.read_texture(texture)
        body_actors.append(pl.add_mesh(sphere, texture=tex, smooth_shading=True,
                                       ambient=0.06, diffuse=1.0, specular=0.0))
    except (FileNotFoundError, OSError):
        print(f"[visualization] texture '{texture}' not found, using plain grey")
        body_actors.append(pl.add_mesh(sphere, color="#8c8c8c",
                                       smooth_shading=True, ambient=0.06,
                                       diffuse=1.0, specular=0.0))

    if site_latlon is not None:
        pt = moon.surface_point_mcmf(*np.deg2rad(site_latlon))
        body_actors.append(
            pl.add_mesh(pv.Sphere(radius=moon.radius * 0.001, center=pt),
                        color="#f5c542"))

    # --- reference frames -------------------------------------------
    if show_frames:
        # MCI: inertial, must NOT join body_actors
        for mesh in _triad(moon.radius, tip_scale=1.85,
                           shaft=0.003, tip=0.012).values():
            pl.add_mesh(mesh, color="#dddddd", opacity=0.5)
        pl.add_point_labels(
            np.array([[moon.radius*1.92, 0, 0], [0, moon.radius*1.92, 0],
                      [0, 0, moon.radius*1.92]]),
            ["X_MCI", "Y_MCI", "Z_MCI"], text_color="#bbbbbb",
            font_size=10, shape=None, always_visible=True, name="mci_lbl")

        # MCMF: rotates with the body, so it joins body_actors
        colours = {"x": "#ff5555", "y": "#55ff88", "z": "#5599ff"}
        for axis, mesh in _triad(moon.radius, tip_scale=1.45).items():
            body_actors.append(pl.add_mesh(mesh, color=colours[axis]))

    # --- trajectories -----------------------------------------------
    if h_stack is not None:
        pl.add_mesh(pv.lines_from_points(h_stack.r),
                    color="#8899aa", line_width=2)
    pl.add_mesh(pv.lines_from_points(h_relay.r), color="#66ccff", line_width=2)
    pl.add_mesh(pv.lines_from_points(h_lander.r), color="#ff6644", line_width=2)

    # --- vehicles ---------------------------------------------------
    def _make_vehicle(path, size, fallback):
        mesh, rgb = read_obj_with_vertex_colours(path) if path else (None, None)
        if mesh is None:
            return [pl.add_mesh(pv.Sphere(radius=size * 0.4), color=fallback)]
        mesh.points = mesh.points - np.array(mesh.center)
        b = np.array(mesh.bounds).reshape(3, 2)
        ext = np.max(b[:, 1] - b[:, 0])
        if ext > 0:
            mesh.points = mesh.points * (size / ext)
        if rgb is not None and len(rgb) == mesh.n_points:
            mesh["rgb"] = rgb
            return [pl.add_mesh(mesh, scalars="rgb", rgb=True,
                                smooth_shading=True)]
        return [pl.add_mesh(mesh, color=fallback, smooth_shading=True)]

    relay_actors = _make_vehicle(relay_mesh, display_size, "#66ccff")
    lander_actors = _make_vehicle(lander_mesh, display_size * 0.85, "#ff8866")

    def _make_body_frame(size, opacity=1.0):
        # Three arrows along the body axes, drawn at the origin.
        cols = ("#ff5555", "#55ff88", "#5599ff")
        dirs = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        return [pl.add_mesh(
            pv.Arrow(start=(0, 0, 0), direction=d, scale=size,
                     shaft_radius=0.03, tip_radius=0.10, tip_length=0.28),
            color=c, opacity=opacity) for d, c in zip(dirs, cols)]

    lander_frame = _make_body_frame(display_size * frame_scale) if show_body_frames else None
    relay_frame = _make_body_frame(display_size * frame_scale * 0.8, 0.55) if show_body_frames else None

    def _pose(actors, q, position):
        # Place a group of actors with a full pose. The mesh gets the SAME matrix as
        # its triad so the model and its axes can never drift apart.
        M = np.eye(4)
        M[0:3, 0:3] = quat_to_dcm(q)
        M[0:3, 3] = np.asarray(position, dtype=float)
        for a in actors:
            a.user_matrix = M

    thrust_arrow = None
    if show_thrust_axis:
        thrust_arrow = pl.add_mesh(
            pv.Arrow(start=(0, 0, 0), direction=(1, 0, 0),
                     scale=display_size * 2.5,
                     shaft_radius=0.02, tip_radius=0.07, tip_length=0.25),
            color="#ffaa33")

    # --- lighting and camera ----------------------------------------
    light, _ = add_sun(pl, moon, 0.0)

    n = np.cross(h_relay.r[0], h_relay.r[len(h_relay.r) // 4])
    n = n / np.linalg.norm(n)
    pl.camera.position = tuple(n * moon.radius * 5.5)
    pl.camera.focal_point = (0.0, 0.0, 0.0)
    pl.camera.up = tuple(h_relay.r[0] / np.linalg.norm(h_relay.r[0]))

    # PyVista reads a tuple position as PIXELS, not normalised coordinates.
    hud = pl.add_text("", position=(14, H - 150), color="white", font_size=10)

    # --- playback ---------------------------------------------------
    def interp_pos(hist, t):
        return np.array([np.interp(t, hist.t, hist.r[:, i]) for i in range(3)])

    def interp_vel(hist, t):
        return np.array([np.interp(t, hist.t, hist.v[:, i]) for i in range(3)])

    def phase_at(t):
        ph = guidance.events[0]["phase"]
        for e in guidance.events:
            if t >= e["t"]:
                ph = e["phase"]
        return ph

    t_end = h_lander.t[-1]
    clock = {"t": 0.0, "scale": float(time_scale), "paused": False,
             "follow": False}

    def interp_quat(t):
        # Attitude at time t. Nearest sample: quaternions do not interpolate
        # linearly and slerp is more machinery than a display needs.
        k = int(np.argmin(np.abs(h_lander.t - t)))
        return quat_normalise(h_lander.q[k])


    def update(t):
        attached = (h_stack is not None) and (t < t_sep)
 
        if attached:
            p = interp_pos(h_stack, t)
            v = interp_vel(h_stack, t)
            along = v / np.linalg.norm(v)
            d = display_size * offset_factor

            q_relay = lvlh_quaternion(p, v)
            _pose(relay_actors, q_relay, p + along * d * 0.5)
            if relay_frame: _pose(relay_frame, q_relay, p + along * d * 0.5)

            # the lander is already pointing retrograde on the stack,
            # ready for the deorbit burn, no jump at separation
            from control import attitude_for_pointing
            # full orientation: z_body = retrograde, x_body = radial out
            z_b = -along
            r_hat = p / np.linalg.norm(p)
            y_b = np.cross(z_b, r_hat)
            y_b = y_b / np.linalg.norm(y_b)
            x_b = np.cross(y_b, z_b)
            dcm = np.column_stack([x_b, y_b, z_b])
            from attitude import dcm_to_quat
            q_lander = dcm_to_quat(np.column_stack([x_b, y_b, z_b]))
            _pose(lander_actors, q_lander, p - along * d * 0.5)
            if lander_frame: _pose(lander_frame, q_lander, p - along * d * 0.5)

            p_lander, v_lander = p, v

        else:
            p_lander = interp_pos(h_lander, t)
            v_lander = interp_vel(h_lander, t)
            p_relay = interp_pos(h_relay, t)
            v_relay = interp_vel(h_relay, t)

            q_relay = lvlh_quaternion(p_relay, v_relay)
            _pose(relay_actors, q_relay, p_relay)
            if relay_frame: _pose(relay_frame, q_relay, p_relay)

            q_lander = (interp_quat(t) if getattr(h_lander, "q", None) is not None
                        else lvlh_quaternion(p_lander, v_lander))
            _pose(lander_actors, q_lander, p_lander)
            if lander_frame: _pose(lander_frame, q_lander, p_lander)
        
        if thrust_arrow is not None:
            axis = -v_lander / np.linalg.norm(v_lander)   # commanded: retrograde
            thrust_arrow.SetPosition(*p_lander)
            thrust_arrow.orientation = _direction_to_euler(axis)
            thrust_arrow.SetVisibility(int(not attached))

        theta = np.rad2deg(moon.rotation_angle(t))
        for a in body_actors:
            a.orientation = (0.0, 0.0, theta)
        update_sun(moon, t, light, None)

        if clock["follow"]:
            # Translate the camera with the target instead of re-aiming it,
            # so mouse zoom and rotation survive between frames.
            fp = np.array(pl.camera.focal_point, dtype=float)
            delta = p_lander - fp
            pl.camera.focal_point = tuple(p_lander)
            pl.camera.position = tuple(np.array(pl.camera.position,
                                                dtype=float) + delta)

        if attached:
            hud.input = (
                f"t       {t/60:9.2f} min\n"
                f"STACK attached\n"
                f"alt     {np.interp(t, h_stack.t, h_stack.altitude):9.2f} km\n"
                f"mass    {np.interp(t, h_stack.t, h_stack.mass):9.1f} kg\n"
                f"rot     {theta % 360:9.2f} deg\n"
                f"scale   {clock['scale']:9.0f} x"
                + ("  [PAUSED]" if clock["paused"] else "")
                + ("  [FOLLOW]" if clock["follow"] else ""))
        else:
            hud.input = (
                f"t       {t/60:9.2f} min   (sep at {t_sep/60:.1f})\n"
                f"relay   alt {np.interp(t, h_relay.t, h_relay.altitude):9.2f} km\n"
                f"lander  alt {np.interp(t, h_lander.t, h_lander.altitude):9.2f} km\n"
                f"phase   {phase_at(t):>9}\n"
                f"mass    {np.interp(t, h_lander.t, h_lander.mass):9.1f} kg\n"
                f"rot     {theta % 360:9.2f} deg\n"
                f"scale   {clock['scale']:9.0f} x"
                + ("  [PAUSED]" if clock["paused"] else "")
                + ("  [FOLLOW]" if clock["follow"] else ""))

    pl.add_key_event("plus", lambda: clock.update(scale=clock["scale"] * 2.0))
    pl.add_key_event("minus", lambda: clock.update(scale=max(1.0, clock["scale"] / 2.0)))
    pl.add_key_event("space", lambda: clock.update(paused=not clock["paused"]))
    pl.add_key_event("r", lambda: clock.update(t=0.0))

    def _toggle_follow():
        clock["follow"] = not clock["follow"]
        if clock["follow"]:
            # Jump to a sensible distance once, then leave the camera alone
            # so the mouse keeps working.
            p = interp_pos(h_lander, clock["t"])
            pl.camera.focal_point = tuple(p)
            pl.camera.position = tuple(p + n * display_size * 12.0)
            pl.camera.up = tuple(p / np.linalg.norm(p))
        else:
            pl.camera.position = tuple(n * moon.radius * 5.5)
            pl.camera.focal_point = (0.0, 0.0, 0.0)
            pl.camera.up = tuple(h_relay.r[0] / np.linalg.norm(h_relay.r[0]))

    pl.add_key_event("h", _toggle_follow)

    update(0.0)
    pl.show(auto_close=False, interactive_update=True)
    wall_prev = _time.perf_counter()

    while clock["t"] < t_end:
        now = _time.perf_counter()
        dtw = now - wall_prev
        wall_prev = now
        if not clock["paused"]:
            clock["t"] = min(clock["t"] + clock["scale"] * dtw, t_end)
        update(clock["t"])
        try:
            pl.update()
        except RuntimeError:
            break
        if not pl.render_window or pl.render_window.GetNeverRendered():
            break

    pl.show()
    return pl

def add_body_frame(pl, position, axes, size, labels=("X_b", "Y_b", "Z_b")):
    # Body-fixed triad at position. axes is the (x, y, z) tuple from
    # RigidBody.body_axes_inertial, the columns of the body-to-inertial matrix.
    import pyvista as pv
    colours = ("#ff5555", "#55ff88", "#5599ff")
    actors = []
    for axis, colour in zip(axes, colours):
        actors.append(pl.add_mesh(
            pv.Arrow(start=position, direction=axis, scale=size,
                     shaft_radius=0.02, tip_radius=0.07, tip_length=0.25),
            color=colour))
    return actors


class v(ABC):
    # Lightweight animation controller for a PyVista scene. Subclasses provide
    # the state and drawing hooks; stepping, pause, reset and render are here.

    def __init__(self, plotter=None, time_scale=1.0, t=0.0):
        self.plotter = plotter
        self.time_scale = float(time_scale)
        self.t = float(t)
        self.paused = False
        self._last_wall = time.perf_counter()

    @abstractmethod
    def _scene_state(self, t):
        # State object for the scene at mission time t.
        raise NotImplementedError

    @abstractmethod
    def _render_frame(self, state):
        # Apply the state to the plotter or actors.
        raise NotImplementedError

    def update(self, t=None):
        # Recompute the scene state at a given time and draw it.
        if t is not None:
            self.t = float(t)
        state = self._scene_state(self.t)
        self._render_frame(state)
        if self.plotter is not None:
            try:
                self.plotter.update()
            except RuntimeError:
                pass
        return state

    def tick(self, dt_wall=None):
        # Advance the clock by wall-clock time and redraw.
        if dt_wall is None:
            now = time.perf_counter()
            dt_wall = now - self._last_wall
            self._last_wall = now
        if not self.paused:
            self.t += self.time_scale * dt_wall
        self.update(self.t)
        return self.t

    def set_time_scale(self, scale):
        # Playback speed in simulation seconds per wall second.
        self.time_scale = max(0.0, float(scale))
        return self.time_scale

    def toggle_pause(self):
        # Pause or resume playback.
        self.paused = not self.paused
        return self.paused

    def reset(self, t=0.0):
        # Reset to a mission time and redraw.
        self.t = float(t)
        self._last_wall = time.perf_counter()
        return self.update(self.t)

    def close(self):
        # Close the render window if one exists.
        if self.plotter is not None:
            try:
                self.plotter.close()
            except Exception:
                pass
        return None