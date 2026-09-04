"""
Lightweight Smoothed Particle Hydrodynamics (SPH)-style solver for
dam-break flood simulation — comparison engine against the grid-based
SWE2D solver.

Pure Python + Numba implementation (no external compiled dependencies,
unlike PySPH which requires a C++ build toolchain). Implements a
simplified Weakly Compressible SPH (WCSPH) scheme: particles carry mass,
position, and velocity; density and pressure are computed via kernel
summation; forces are integrated explicitly (leapfrog).

Particles are seeded at the reservoir/breach location and tracked as they
flow across DEM-derived terrain. Output is resampled onto the same
regular lat/lon grid as the SWE2D engine, and written to NetCDF in the
same schema so both engines are directly comparable via
services/engine_comparator.py.
"""

import os
import numpy as np
import rasterio
import xarray as xr
from numba import njit, prange
from datetime import datetime

G = 9.81
RHO0 = 1000.0        # reference water density (kg/m3)
H_SMOOTH = 3.0        # SPH smoothing radius (m)
STIFFNESS = 50.0      # Tait EOS stiffness constant
GAMMA = 7.0            # Tait EOS exponent
VISCOSITY = 0.05       # artificial viscosity coefficient
PARTICLE_SPACING = 5.0  # initial particle spacing (m)
DT = 0.02              # integration timestep (s)


def load_terrain_grid(dem_path: str):
    """Loads DEM as a lookup grid for particle-terrain interaction (bed elevation)."""
    with rasterio.open(dem_path) as src:
        z = src.read(1).astype(np.float64)
        transform = src.transform
        bounds = src.bounds
        nrows, ncols = z.shape
        dx = abs(transform.a)
        dy = abs(transform.e)
        lon = np.linspace(bounds.left, bounds.right, ncols)
        lat = np.linspace(bounds.top, bounds.bottom, nrows)

    z = np.where(np.isnan(z) | (z < -1000), 9999.0, z)
    return z, dx, dy, lon, lat


def _seed_reservoir_particles(reservoir_volume: float, particle_spacing: float = PARTICLE_SPACING):
    """
    Seeds a 3D block of particles representing the reservoir water mass,
    positioned just upstream of the breach (local coordinate origin = breach).
    """
    nominal_depth = 10.0
    footprint_area = reservoir_volume / nominal_depth
    side_length = max(20.0, np.sqrt(footprint_area))

    n_per_side = max(5, int(side_length / particle_spacing))
    xs = np.linspace(-side_length, -5.0, n_per_side)  # upstream (negative x = behind breach)
    ys = np.linspace(-side_length / 2, side_length / 2, n_per_side)
    zs = np.linspace(1.0, nominal_depth, max(3, int(nominal_depth / particle_spacing)))

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    x = xx.flatten()
    y = yy.flatten()
    z = zz.flatten()

    mass_per_particle = RHO0 * (particle_spacing ** 3)
    n_particles = len(x)

    vx = np.zeros(n_particles)
    vy = np.zeros(n_particles)
    vz = np.zeros(n_particles)

    return x, y, z, vx, vy, vz, np.full(n_particles, mass_per_particle)


@njit(cache=True, fastmath=True, parallel=True)
def _compute_density_pressure(x, y, z, mass, h):
    """Kernel-weighted density summation (cubic spline kernel) + Tait EOS pressure."""
    n = len(x)
    rho = np.zeros(n)
    norm = 315.0 / (64.0 * np.pi * h ** 9)

    for i in prange(n):
        density = 0.0
        for j in range(n):
            dx_ = x[i] - x[j]
            dy_ = y[i] - y[j]
            dz_ = z[i] - z[j]
            r2 = dx_ * dx_ + dy_ * dy_ + dz_ * dz_
            h2 = h * h
            if r2 < h2:
                density += mass[j] * norm * (h2 - r2) ** 3
        rho[i] = max(density, RHO0 * 0.5)  # floor to avoid negative/zero density

    pressure = STIFFNESS * ((rho / RHO0) ** GAMMA - 1.0)
    return rho, pressure


@njit(cache=True, fastmath=True, parallel=True)
def _compute_forces(x, y, z, vx, vy, vz, mass, rho, pressure, h, terrain_interp_z):
    """
    Computes SPH pressure gradient force + viscosity + gravity + simple
    terrain collision (bounce-back when particle z falls below local bed).
    """
    n = len(x)
    ax = np.zeros(n)
    ay = np.zeros(n)
    az = np.full(n, -G)  # gravity

    spiky_grad_norm = -45.0 / (np.pi * h ** 6)

    for i in prange(n):
        fx, fy, fz = 0.0, 0.0, 0.0
        for j in range(n):
            if i == j:
                continue
            dx_ = x[i] - x[j]
            dy_ = y[i] - y[j]
            dz_ = z[i] - z[j]
            r2 = dx_ * dx_ + dy_ * dy_ + dz_ * dz_
            r = np.sqrt(r2) + 1e-6

            if r < h:
                # Pressure force (spiky kernel gradient)
                grad_coeff = spiky_grad_norm * (h - r) ** 2
                pij = (pressure[i] + pressure[j]) / (2.0 * rho[j] + 1e-6)
                fx -= mass[j] * pij * grad_coeff * (dx_ / r)
                fy -= mass[j] * pij * grad_coeff * (dy_ / r)
                fz -= mass[j] * pij * grad_coeff * (dz_ / r)

                # Artificial viscosity (velocity smoothing between neighbors)
                dvx = vx[i] - vx[j]
                dvy = vy[i] - vy[j]
                dvz = vz[i] - vz[j]
                fx -= VISCOSITY * mass[j] * dvx / (rho[j] + 1e-6) * (h - r)
                fy -= VISCOSITY * mass[j] * dvy / (rho[j] + 1e-6) * (h - r)
                fz -= VISCOSITY * mass[j] * dvz / (rho[j] + 1e-6) * (h - r)

        ax[i] += fx / (rho[i] + 1e-6)
        ay[i] += fy / (rho[i] + 1e-6)
        az[i] += fz / (rho[i] + 1e-6)

        # Terrain collision: bounce particle if below local bed elevation
        bed_z = terrain_interp_z[i]
        if z[i] < bed_z:
            az[i] += (bed_z - z[i]) * 50.0  # spring-back force
            az[i] -= vz[i] * 5.0            # damping on impact

    return ax, ay, az


def _interpolate_bed_elevation(x, y, terrain_z, dx, dy, breach_lon, breach_lat, lon_arr, lat_arr):
    """Maps each particle's local x/y offset (meters from breach) to a bed elevation from DEM."""
    n = len(x)
    bed_z = np.zeros(n)
    nrows, ncols = terrain_z.shape

    for i in range(n):
        p_lon = breach_lon + (x[i] / (111320.0 * np.cos(np.radians(breach_lat))))
        p_lat = breach_lat + (y[i] / 110540.0)

        col = int(np.clip(np.searchsorted(lon_arr, p_lon), 0, ncols - 1))
        row = int(np.clip(np.searchsorted(-lat_arr, -p_lat), 0, nrows - 1))

        bed_z[i] = terrain_z[row, col] - terrain_z.min()  # relative elevation

    return bed_z


def _run_particle_simulation(x, y, z, vx, vy, vz, mass, sim_duration_s, terrain_z, dx, dy,
                               breach_lon, breach_lat, lon_arr, lat_arr):
    """Explicit leapfrog time integration loop with periodic snapshotting."""
    n_steps = max(1, int(sim_duration_s / DT))
    snapshot_interval = max(1, n_steps // 10)

    bed_z = _interpolate_bed_elevation(x, y, terrain_z, dx, dy, breach_lon, breach_lat, lon_arr, lat_arr)

    snapshots = []

    for step in range(n_steps):
        rho, pressure = _compute_density_pressure(x, y, z, mass, H_SMOOTH)
        ax, ay, az = _compute_forces(x, y, z, vx, vy, vz, mass, rho, pressure, H_SMOOTH, bed_z)

        vx += ax * DT
        vy += ay * DT
        vz += az * DT

        x += vx * DT
        y += vy * DT
        z += vz * DT

        if step % snapshot_interval == 0:
            snapshots.append({
                "time_s": step * DT,
                "x": x.copy(), "y": y.copy(), "z": z.copy(),
                "vx": vx.copy(), "vy": vy.copy(),
            })

    snapshots.append({
        "time_s": n_steps * DT,
        "x": x.copy(), "y": y.copy(), "z": z.copy(),
        "vx": vx.copy(), "vy": vy.copy(),
    })

    return snapshots


def _particles_to_grid(snapshots, lon, lat, dx, dy, breach_lon, breach_lat, particle_spacing):
    """Bins particle positions onto the regular lat/lon grid to produce depth/velocity fields."""
    nrows, ncols = len(lat), len(lon)
    n_snapshots = len(snapshots)

    depth_stack = np.zeros((n_snapshots, nrows, ncols))
    vx_stack = np.zeros((n_snapshots, nrows, ncols))
    vy_stack = np.zeros((n_snapshots, nrows, ncols))
    time_stack = []

    cell_area = dx * dy
    particle_volume = particle_spacing ** 3

    for t_idx, snap in enumerate(snapshots):
        time_stack.append(snap["time_s"])

        particle_lon = breach_lon + (snap["x"] / (111320.0 * np.cos(np.radians(breach_lat))))
        particle_lat = breach_lat + (snap["y"] / 110540.0)

        col_idx = np.clip(np.searchsorted(lon, particle_lon), 0, ncols - 1)
        row_idx = np.clip(np.searchsorted(-lat, -particle_lat), 0, nrows - 1)

        counts = np.zeros((nrows, ncols))
        vx_sum = np.zeros((nrows, ncols))
        vy_sum = np.zeros((nrows, ncols))

        for k in range(len(snap["x"])):
            r, c = row_idx[k], col_idx[k]
            counts[r, c] += 1
            vx_sum[r, c] += snap["vx"][k]
            vy_sum[r, c] += snap["vy"][k]

        depth_stack[t_idx] = (counts * particle_volume) / cell_area
        with np.errstate(divide="ignore", invalid="ignore"):
            vx_stack[t_idx] = np.where(counts > 0, vx_sum / np.maximum(counts, 1), 0.0)
            vy_stack[t_idx] = np.where(counts > 0, vy_sum / np.maximum(counts, 1), 0.0)

    return depth_stack, vx_stack, vy_stack, time_stack


def run_sph_engine(params: dict, hydrograph: dict, output_dir: str = "data/simulation_output") -> str:
    """
    Main entrypoint: runs the lightweight SPH-style dam-break simulation and
    writes results to NetCDF in the same schema as run_swe2d_engine, for
    direct comparison via engine_comparator.py.

    Args:
        params: dict with 'latitude', 'longitude', 'dam_name'
        hydrograph: output of breach_model.generate_breach_hydrograph()
        output_dir: directory to write NetCDF output

    Returns:
        Path to output NetCDF file
    """
    os.makedirs(output_dir, exist_ok=True)

    dem_path = "data/dem/processed/catchment_dem.tif"
    if not os.path.exists(dem_path):
        dem_path = "data/dem/raw_dem.tif"
        if not os.path.exists(dem_path):
            raise FileNotFoundError(
                "No DEM found. Run dem_processor.prepare_dem() first, or place a DEM at data/dem/raw_dem.tif"
            )

    terrain_z, dx, dy, lon, lat = load_terrain_grid(dem_path)

    reservoir_volume = hydrograph["reservoir_volume_m3"]
    x, y, z, vx, vy, vz, mass = _seed_reservoir_particles(reservoir_volume)

    # Cap particle count for CPU feasibility (prototype-scale)
    max_particles = 3000
    if len(x) > max_particles:
        idx = np.random.choice(len(x), max_particles, replace=False)
        x, y, z = x[idx], y[idx], z[idx]
        vx, vy, vz = vx[idx], vy[idx], vz[idx]
        mass = mass[idx]

    sim_duration_s = min(hydrograph["time_hours"][-1] * 3600.0, 300.0)  # cap at 5 min sim time

    snapshots = _run_particle_simulation(
        x, y, z, vx, vy, vz, mass, sim_duration_s, terrain_z, dx, dy,
        params["longitude"], params["latitude"], lon, lat,
    )

    depth_stack, vx_stack, vy_stack, time_stack = _particles_to_grid(
        snapshots, lon, lat, dx, dy, params["longitude"], params["latitude"], PARTICLE_SPACING
    )

    ds = xr.Dataset(
        {
            "depth": (["time", "lat", "lon"], depth_stack),
            "velocity_x": (["time", "lat", "lon"], vx_stack),
            "velocity_y": (["time", "lat", "lon"], vy_stack),
        },
        coords={"time": time_stack, "lat": lat, "lon": lon},
        attrs={
            "engine": "custom_sph",
            "dam_name": params.get("dam_name", "unknown"),
            "n_particles": len(x),
            "created": datetime.utcnow().isoformat(),
        },
    )

    out_path = os.path.join(output_dir, f"sph_{params.get('dam_name', 'sim').replace(' ', '_')}.nc")
    ds.to_netcdf(out_path)

    return out_path
