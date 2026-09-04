"""
DEM preprocessing service for dam-break inundation modelling.
Handles: sink filling, flow direction, flow accumulation, catchment
delineation, and clipping — producing a hydrologically-corrected DEM
ready for the SWE/SPH hydro engines.
"""

import os
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pysheds.grid import Grid


def load_dem(dem_path: str):
    """Loads a DEM raster and returns a pysheds Grid + raw elevation array."""
    if not os.path.exists(dem_path):
        raise FileNotFoundError(f"DEM file not found: {dem_path}")

    grid = Grid.from_raster(dem_path)
    dem = grid.read_raster(dem_path)
    return grid, dem


def fill_sinks_and_flats(grid: Grid, dem):
    """
    Hydrologically corrects the DEM:
    1. Fills pits (single-cell depressions)
    2. Fills depressions (larger sinks)
    3. Resolves flats (ambiguous flow direction areas)
    """
    pit_filled = grid.fill_pits(dem)
    flooded = grid.fill_depressions(pit_filled)
    inflated = grid.resolve_flats(flooded)
    return inflated


def compute_flow_direction(grid: Grid, conditioned_dem, routing: str = "d8"):
    """
    Computes flow direction grid.
    routing: 'd8' (single flow direction, standard) or 'dinf' (multiple flow direction)
    """
    if routing == "d8":
        fdir = grid.flowdir(conditioned_dem, routing="d8")
    elif routing == "dinf":
        fdir = grid.flowdir(conditioned_dem, routing="dinf")
    else:
        raise ValueError("routing must be 'd8' or 'dinf'")
    return fdir


def compute_flow_accumulation(grid: Grid, fdir, routing: str = "d8"):
    """Computes flow accumulation grid from flow direction — identifies channels."""
    acc = grid.accumulation(fdir, routing=routing)
    return acc


def delineate_catchment(grid: Grid, fdir, pour_point_lon: float, pour_point_lat: float, routing: str = "d8"):
    """
    Delineates the upstream catchment/basin boundary given a pour point
    (typically the dam location). Snaps pour point to nearest high-accumulation
    cell first for stability.
    """
    catchment = grid.catchment(
        x=pour_point_lon,
        y=pour_point_lat,
        fdir=fdir,
        routing=routing,
        xytype="coordinate",
    )
    return catchment


def snap_pour_point_to_stream(grid: Grid, acc, lon: float, lat: float, snap_threshold_percentile: float = 95):
    """
    Snaps a user-provided dam coordinate to the nearest high-flow-accumulation
    cell, since exact clicked coordinates rarely land precisely on a stream cell.
    """
    threshold = np.nanpercentile(acc, snap_threshold_percentile)
    snapped_x, snapped_y = grid.snap_to_mask(acc > threshold, (lon, lat), xytype="coordinate")
    return snapped_x, snapped_y


def clip_dem_to_catchment(grid: Grid, conditioned_dem, catchment):
    """Clips the conditioned DEM to just the delineated catchment extent."""
    grid.clip_to(catchment)
    clipped_dem = grid.view(conditioned_dem)
    return grid, clipped_dem


def export_raster(grid: Grid, data, out_path: str, dtype: str = "float32"):
    """Writes a pysheds grid array back out to a GeoTIFF."""
    grid.to_raster(data, out_path, dtype=dtype)
    return out_path


def prepare_dem(dem_path: str, dam_lon: float, dam_lat: float, output_dir: str = "data/dem/processed"):
    """
    Full DEM preprocessing pipeline entrypoint.

    Args:
        dem_path: path to raw input DEM (.tif)
        dam_lon, dam_lat: dam location, used as the pour point for catchment delineation
        output_dir: directory to save processed outputs

    Returns:
        dict with paths to conditioned DEM, flow accumulation raster, and catchment-clipped DEM
    """
    os.makedirs(output_dir, exist_ok=True)

    grid, dem = load_dem(dem_path)

    # Step 1: Hydrological conditioning
    conditioned = fill_sinks_and_flats(grid, dem)

    # Step 2: Flow direction & accumulation
    fdir = compute_flow_direction(grid, conditioned, routing="d8")
    acc = compute_flow_accumulation(grid, fdir, routing="d8")

    # Step 3: Snap dam coordinates to nearest stream cell for stable catchment delineation
    snapped_lon, snapped_lat = snap_pour_point_to_stream(grid, acc, dam_lon, dam_lat)

    # Step 4: Delineate upstream catchment from the dam location
    catchment = delineate_catchment(grid, fdir, snapped_lon, snapped_lat, routing="d8")

    # Step 5: Clip conditioned DEM to catchment extent
    clipped_grid, clipped_dem = clip_dem_to_catchment(grid, conditioned, catchment)

    # Step 6: Export outputs
    conditioned_path = export_raster(grid, conditioned, os.path.join(output_dir, "conditioned_dem.tif"))
    accumulation_path = export_raster(grid, acc, os.path.join(output_dir, "flow_accumulation.tif"))
    clipped_path = export_raster(clipped_grid, clipped_dem, os.path.join(output_dir, "catchment_dem.tif"))

    return {
        "conditioned_dem": conditioned_path,
        "flow_accumulation": accumulation_path,
        "catchment_dem": clipped_path,
        "snapped_pour_point": {"lon": float(snapped_lon), "lat": float(snapped_lat)},
    }
