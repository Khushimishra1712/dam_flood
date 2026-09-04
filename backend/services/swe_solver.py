"""
Custom 2D Shallow Water Equations (SWE) solver for dam-break flood
inundation modelling. Numba-accelerated finite-volume explicit scheme
(Lax-Friedrichs flux) — runs on CPU, no GPU/external binary required.

Governing equations (conservative form):
    dh/dt   + d(hu)/dx + d(hv)/dy = 0
    d(hu)/dt + d(hu^2 + 0.5*g*h^2)/dx + d(huv)/dy = -g*h*dz/dx - g*n^2*u*|V|/h^(1/3)
    d(hv)/dt + d(huv)/dx + d(hv^2 + 0.5*g*h^2)/dy = -g*h*dz/dy - g*n^2*v*|V|/h^(1/3)

Where h=depth, u,v=velocity components, z=bed elevation, n=Manning's roughness.

Output: NetCDF with depth, velocity_x, velocity_y on a lat/lon grid — matches
the format expected by services/converter.py.
"""

import os
import numpy as np
import rasterio
from numba import njit, prange
import xarray as xr
from datetime import datetime, timedelta

G = 9.81           # gravitational acceleration (m/s2)
MANNING_N = 0.035   # default Manning's roughness coefficient (mixed terrain)
CFL = 0.4           # Courant number for stability
MIN_DEPTH = 1e-4    # wet/dry threshold (m)


def load_terrain(dem_path: str):
    """Loads bed elevation raster and returns array + spatial metadata."""
    with rasterio.open(dem_path) as src:
        z = src.read(1).astype(np.float64)
        transform = src.transform
        bounds = src.bounds
        nrows, ncols = z.shape

        # Cell size in meters (approx, assumes roughly square projected cells)
        dx = abs(transform.a)
        dy = abs(transform.e)

        lon = np.linspace(bounds.left, bounds.right, ncols)
        lat = np.linspace(bounds.top, bounds.bottom, nrows)

    # Replace nodata/invalid elevations with a high value (acts as a wall)
    z = np.where(np.isnan(z) | (z < -1000), 9999.0, z)

    return z, dx, dy, lon, lat


@njit(cache=True, fastmath=True)
def _compute_timestep(h, u, v, dx, dy):
    """Computes stable timestep from CFL condition given current wave speeds."""
    max_speed = 1e-6
    nrows, ncols = h.shape
    for i in range(nrows):
        for j in range(ncols):
            if h[i, j] > MIN_DEPTH:
                c = np.sqrt(G * h[i, j])
                speed_x = abs(u[i, j]) + c
                speed_y = abs(v[i, j]) + c
                if speed_x > max_speed:
                    max_speed = speed_x
                if speed_y > max_speed:
                    max_speed = speed_y
    dt = CFL * min(dx, dy) / max_speed
    return dt


@njit(cache=True, fastmath=True, parallel=True)
def _step_lax_friedrichs(h, hu, hv, z, dx, dy, dt, manning_n):
    """
    Single explicit timestep update using Lax-Friedrichs scheme for
    numerical stability with wet/dry fronts (standard for shallow flood
    routing on complex terrain).
    """
    nrows, ncols = h.shape
    h_new = h.copy()
    hu_new = hu.copy()
    hv_new = hv.copy()

    for i in prange(1, nrows - 1):
        for j in range(1, ncols - 1):
            if h[i, j] < MIN_DEPTH and h[i - 1, j] < MIN_DEPTH and h[i + 1, j] < MIN_DEPTH \
               and h[i, j - 1] < MIN_DEPTH and h[i, j + 1] < MIN_DEPTH:
                continue  # fully dry neighborhood, skip

            # Neighbor states (with dry-cell safety)
            hE, huE, hvE = h[i, j + 1], hu[i, j + 1], hv[i, j + 1]
            hW, huW, hvW = h[i, j - 1], hu[i, j - 1], hv[i, j - 1]
            hN, huN, hvN = h[i - 1, j], hu[i - 1, j], hv[i - 1, j]
            hS, huS, hvS = h[i + 1, j], hu[i + 1, j], hv[i + 1, j]

            uE = huE / hE if hE > MIN_DEPTH else 0.0
            uW = huW / hW if hW > MIN_DEPTH else 0.0
            vN = hvN / hN if hN > MIN_DEPTH else 0.0
            vS = hvS / hS if hS > MIN_DEPTH else 0.0

            # Mass flux (continuity)
            flux_h_x = (huE - huW) / (2 * dx)
            flux_h_y = (hvN - hvS) / (2 * dy)

            # Momentum flux x-direction
            momE_x = huE * uE + 0.5 * G * hE ** 2
            momW_x = huW * uW + 0.5 * G * hW ** 2
            flux_hu_x = (momE_x - momW_x) / (2 * dx)

            # Momentum flux y-direction
            momN_y = hvN * vN + 0.5 * G * hN ** 2
            momS_y = hvS * vS + 0.5 * G * hS ** 2
            flux_hv_y = (momN_y - momS_y) / (2 * dy)

            # Bed slope source term
            dzdx = (z[i, j + 1] - z[i, j - 1]) / (2 * dx)
            dzdy = (z[i - 1, j] - z[i + 1, j]) / (2 * dy)

            u_c = hu[i, j] / h[i, j] if h[i, j] > MIN_DEPTH else 0.0
            v_c = hv[i, j] / h[i, j] if h[i, j] > MIN_DEPTH else 0.0
            vel_mag = np.sqrt(u_c ** 2 + v_c ** 2)

            # Manning friction source term
            if h[i, j] > MIN_DEPTH:
                friction_x = G * manning_n ** 2 * u_c * vel_mag / (h[i, j] ** (4.0 / 3.0))
                friction_y = G * manning_n ** 2 * v_c * vel_mag / (h[i, j] ** (4.0 / 3.0))
            else:
                friction_x = 0.0
                friction_y = 0.0

            source_x = -G * h[i, j] * dzdx - friction_x
            source_y = -G * h[i, j] * dzdy - friction_y

            # Lax-Friedrichs diffusive averaging (stabilizes shocks/wet-dry fronts)
            h_avg = 0.25 * (hE + hW + hN + hS)
            hu_avg = 0.25 * (huE + huW + huN + huS) if False else 0.25 * (huE + huW + huN + huS)
            hv_avg = 0.25 * (hvE + hvW + hvN + hvS)

            h_new[i, j] = h_avg - dt * (flux_h_x + flux_h_y)
            hu_new[i, j] = hu_avg - dt * (flux_hu_x) + dt * source_x
            hv_new[i, j] = hv_avg - dt * (flux_hv_y) + dt * source_y

            if h_new[i, j] < MIN_DEPTH:
                h_new[i, j] = 0.0
                hu_new[i, j] = 0.0
                hv_new[i, j] = 0.0

    return h_new, hu_new, hv_new


def _inject_breach_inflow(h, hu, breach_row, breach_col, breach_width_cells, discharge, dx, dy, dt):
    """
    Injects breach outflow discharge into the domain at the dam location as
    a source term — distributed across breach_width_cells to represent the
    physical breach opening width.
    """
    half_width = max(1, breach_width_cells // 2)
    r0 = max(0, breach_row - 1)
    r1 = min(h.shape[0], breach_row + 2)
    c0 = max(0, breach_col - half_width)
    c1 = min(h.shape[1], breach_col + half_width)

    n_cells = max(1, (r1 - r0) * (c1 - c0))
    cell_area = dx * dy
    depth_increment = (discharge * dt) / (n_cells * cell_area)

    h[r0:r1, c0:c1] += depth_increment
    # Impart downstream momentum (positive x-direction assumed toward valley;
    # in production this should follow the DEM-derived flow direction)
    hu[r0:r1, c0:c1] = h[r0:r1, c0:c1] * 2.0

    return h, hu


def run_swe2d_engine(params: dict, hydrograph: dict, output_dir: str = "data/simulation_output") -> str:
    """
    Main entrypoint: runs the 2D SWE simulation using the breach hydrograph
    as inflow forcing, on the catchment DEM, and writes results to NetCDF.

    Args:
        params: dict with 'latitude', 'longitude', 'dam_name' etc (from API request)
        hydrograph: output of breach_model.generate_breach_hydrograph()
        output_dir: directory to write NetCDF output

    Returns:
        Path to output NetCDF file
    """
    os.makedirs(output_dir, exist_ok=True)

    dem_path = "data/dem/processed/catchment_dem.tif"
    if not os.path.exists(dem_path):
        # Fallback: use raw uploaded DEM if catchment processing hasn't been run
        dem_path = "data/dem/raw_dem.tif"
        if not os.path.exists(dem_path):
            raise FileNotFoundError(
                "No DEM found. Run dem_processor.prepare_dem() first, or place a DEM at data/dem/raw_dem.tif"
            )

    z, dx, dy, lon, lat = load_terrain(dem_path)
    nrows, ncols = z.shape

    # Initialize state: dry domain
    h = np.zeros((nrows, ncols), dtype=np.float64)
    hu = np.zeros((nrows, ncols), dtype=np.float64)
    hv = np.zeros((nrows, ncols), dtype=np.float64)

    # Locate breach cell (nearest grid cell to dam coordinates)
    breach_row = int(np.argmin(np.abs(lat - params["latitude"])))
    breach_col = int(np.argmin(np.abs(lon - params["longitude"])))
    breach_width_cells = max(1, int(hydrograph["breach_width_m"] / dx))

    time_arr = np.array(hydrograph["time_hours"]) * 3600.0  # convert to seconds
    discharge_arr = np.array(hydrograph["discharge_m3s"])

    sim_time = 0.0
    end_time = time_arr[-1]
    output_snapshots = []
    snapshot_interval = end_time / 10.0  # save 10 timesteps for output NetCDF
    next_snapshot = 0.0

    max_iterations = 20000  # safety cap to prevent runaway loops
    iteration = 0

    while sim_time < end_time and iteration < max_iterations:
        dt = _compute_timestep(h, hu / np.maximum(h, MIN_DEPTH), hv / np.maximum(h, MIN_DEPTH), dx, dy)
        dt = min(dt, end_time - sim_time, 30.0)  # cap max timestep at 30s for stability

        current_discharge = float(np.interp(sim_time, time_arr, discharge_arr))
        h, hu = _inject_breach_inflow(h, hu, breach_row, breach_col, breach_width_cells, current_discharge, dx, dy, dt)

        h, hu, hv = _step_lax_friedrichs(h, hu, hv, z, dx, dy, dt, MANNING_N)

        sim_time += dt
        iteration += 1

        if sim_time >= next_snapshot:
            u_out = np.where(h > MIN_DEPTH, hu / np.maximum(h, MIN_DEPTH), 0.0)
            v_out = np.where(h > MIN_DEPTH, hv / np.maximum(h, MIN_DEPTH), 0.0)
            output_snapshots.append({
                "time_s": sim_time,
                "depth": h.copy(),
                "velocity_x": u_out.copy(),
                "velocity_y": v_out.copy(),
            })
            next_snapshot += snapshot_interval

    # Ensure final state is always captured
    if not output_snapshots or output_snapshots[-1]["time_s"] < sim_time:
        u_out = np.where(h > MIN_DEPTH, hu / np.maximum(h, MIN_DEPTH), 0.0)
        v_out = np.where(h > MIN_DEPTH, hv / np.maximum(h, MIN_DEPTH), 0.0)
        output_snapshots.append({
            "time_s": sim_time,
            "depth": h.copy(),
            "velocity_x": u_out.copy(),
            "velocity_y": v_out.copy(),
        })

    depth_stack = np.stack([s["depth"] for s in output_snapshots], axis=0)
    vx_stack = np.stack([s["velocity_x"] for s in output_snapshots], axis=0)
    vy_stack = np.stack([s["velocity_y"] for s in output_snapshots], axis=0)
    time_stack = [s["time_s"] for s in output_snapshots]

    ds = xr.Dataset(
        {
            "depth": (["time", "lat", "lon"], depth_stack),
            "velocity_x": (["time", "lat", "lon"], vx_stack),
            "velocity_y": (["time", "lat", "lon"], vy_stack),
        },
        coords={
            "time": time_stack,
            "lat": lat,
            "lon": lon,
        },
        attrs={
            "engine": "custom_swe2d",
            "dam_name": params.get("dam_name", "unknown"),
            "peak_discharge_m3s": hydrograph["peak_discharge_m3s"],
            "created": datetime.utcnow().isoformat(),
        },
    )

    out_path = os.path.join(output_dir, f"swe2d_{params.get('dam_name', 'sim').replace(' ', '_')}.nc")
    ds.to_netcdf(out_path)

    return out_path
