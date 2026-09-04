"""
Google Earth Engine (GEE) validation service.

Fetches Sentinel-1 SAR imagery (pre-event and post-event) for the study
area, derives an observed flood extent mask via VV/VH backscatter
thresholding, and compares it against the simulated inundation extent
using the same agreement metrics as engine_comparator.py.

Requires:
  - A Google Earth Engine account with API access enabled
  - A GEE service account JSON key (path set via GEE_SERVICE_ACCOUNT_KEY
    environment variable) OR interactive `earthengine authenticate` for
    local development

Setup (one-time, local dev):
    pip install earthengine-api
    earthengine authenticate
"""

import os
import numpy as np
import xarray as xr

try:
    import ee
    EE_AVAILABLE = True
except ImportError:
    EE_AVAILABLE = False

from services.engine_comparator import compute_extent_metrics


def initialize_gee():
    """
    Initializes the Earth Engine API session. Uses a service account key
    if GEE_SERVICE_ACCOUNT_KEY env var is set (production/server use),
    otherwise falls back to locally cached user credentials (dev use,
    requires prior `earthengine authenticate` run once in terminal).
    """
    if not EE_AVAILABLE:
        raise ImportError(
            "earthengine-api is not installed. Install via 'pip install earthengine-api'."
        )

    service_account_key = os.getenv("GEE_SERVICE_ACCOUNT_KEY")

    if service_account_key and os.path.exists(service_account_key):
        credentials = ee.ServiceAccountCredentials(None, service_account_key)
        ee.Initialize(credentials)
    else:
        # Falls back to cached credentials from `earthengine authenticate`
        ee.Initialize()


def fetch_sentinel1_sar_extent(
    aoi_bounds: tuple,
    event_date: str,
    pre_event_days: int = 30,
    post_event_days: int = 10,
    orbit_pass: str = "DESCENDING",
) -> dict:
    """
    Fetches Sentinel-1 SAR image pair (pre/post event) over the area of
    interest and derives a flood extent mask via VV backscatter change
    detection (standard SAR flood mapping technique: water surfaces show
    markedly lower backscatter than land).

    Args:
        aoi_bounds: (min_lon, min_lat, max_lon, max_lat) bounding box
        event_date: ISO date string 'YYYY-MM-DD' of the flood event
        pre_event_days: days before event_date to search for baseline image
        post_event_days: days after event_date to search for flood image
        orbit_pass: 'ASCENDING' or 'DESCENDING' — keep consistent for both
            images to avoid orbit-geometry-induced backscatter differences

    Returns:
        dict with GEE image references and metadata; actual pixel export
        handled separately in export_flood_mask_to_array()
    """
    if not EE_AVAILABLE:
        raise ImportError("earthengine-api is not installed.")

    min_lon, min_lat, max_lon, max_lat = aoi_bounds
    aoi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

    from datetime import datetime, timedelta
    event_dt = datetime.fromisoformat(event_date)
    pre_start = (event_dt - timedelta(days=pre_event_days)).strftime("%Y-%m-%d")
    pre_end = event_dt.strftime("%Y-%m-%d")
    post_start = event_dt.strftime("%Y-%m-%d")
    post_end = (event_dt + timedelta(days=post_event_days)).strftime("%Y-%m-%d")

    s1_collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select("VV")
    )

    pre_image = s1_collection.filterDate(pre_start, pre_end).sort("system:time_start", False).first()
    post_image = s1_collection.filterDate(post_start, post_end).sort("system:time_start", True).first()

    return {
        "pre_image": pre_image,
        "post_image": post_image,
        "aoi": aoi,
        "pre_date_range": (pre_start, pre_end),
        "post_date_range": (post_start, post_end),
    }


def derive_flood_extent_mask(sar_data: dict, vv_threshold_db: float = -17.0, change_threshold_db: float = 3.0):
    """
    Derives a binary flood mask using two complementary SAR techniques:
    1. Absolute threshold: post-event VV backscatter below vv_threshold_db
       (water surfaces are smooth = low backscatter)
    2. Change detection: post-event backscatter drops by more than
       change_threshold_db relative to pre-event baseline (isolates NEW
       water, filtering out permanent water bodies)

    Final flood mask = intersection of both conditions (more conservative,
    reduces false positives from shadow/other low-backscatter surfaces).

    Args:
        sar_data: output of fetch_sentinel1_sar_extent()
        vv_threshold_db: absolute VV backscatter threshold (dB) for water
        change_threshold_db: minimum backscatter drop (dB) to flag as new flooding

    Returns:
        ee.Image binary flood mask (1 = flooded, 0 = not flooded)
    """
    if not EE_AVAILABLE:
        raise ImportError("earthengine-api is not installed.")

    pre_image = sar_data["pre_image"]
    post_image = sar_data["post_image"]

    # Speckle filtering (standard SAR preprocessing - reduces salt-and-pepper noise)
    pre_smooth = pre_image.focal_median(radius=50, units="meters")
    post_smooth = post_image.focal_median(radius=50, units="meters")

    absolute_water_mask = post_smooth.lt(vv_threshold_db)
    change_mask = pre_smooth.subtract(post_smooth).gt(change_threshold_db)

    flood_mask = absolute_water_mask.And(change_mask).rename("flood_extent")
    return flood_mask


def export_flood_mask_to_array(flood_mask, aoi_bounds: tuple, target_shape: tuple, scale: int = 10) -> np.ndarray:
    """
    Exports the GEE flood mask image to a local numpy array, resampled to
    match the simulation grid shape for direct comparison.

    Args:
        flood_mask: ee.Image binary flood mask
        aoi_bounds: (min_lon, min_lat, max_lon, max_lat)
        target_shape: (nrows, ncols) to resample to — must match simulation grid
        scale: native resolution to sample at before resizing (meters, Sentinel-1 native = 10m)

    Returns:
        numpy boolean array of shape target_shape
    """
    if not EE_AVAILABLE:
        raise ImportError("earthengine-api is not installed.")

    min_lon, min_lat, max_lon, max_lat = aoi_bounds
    region = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

    # getDownloadURL / sampleRectangle approach for small AOIs (prototype scale)
    sample = flood_mask.sampleRectangle(region=region, defaultValue=0)
    band_data = sample.get("flood_extent").getInfo()

    raw_array = np.array(band_data, dtype=np.float64)

    # Resample to match simulation grid shape using simple nearest-neighbor
    if raw_array.shape != target_shape:
        from scipy.ndimage import zoom
        zoom_factors = (target_shape[0] / raw_array.shape[0], target_shape[1] / raw_array.shape[1])
        raw_array = zoom(raw_array, zoom_factors, order=0)

    return raw_array.astype(bool)


def validate_against_sar(
    simulation_nc_path: str,
    dam_lat: float,
    dam_lon: float,
    event_date: str,
    buffer_km: float = 15.0,
) -> dict:
    """
    Main entrypoint: validates simulated flood extent against Sentinel-1
    SAR-observed extent for a real historical event.

    Args:
        simulation_nc_path: path to simulation NetCDF (SWE2D or SPH output)
        dam_lat, dam_lon: dam/event location
        event_date: ISO date string 'YYYY-MM-DD' of the actual flood event
        buffer_km: radius around dam location to define AOI for SAR fetch

    Returns:
        dict with extent agreement metrics (IoU, CSI, F1) comparing
        simulated vs SAR-observed flood extent
    """
    initialize_gee()

    ds = xr.open_dataset(simulation_nc_path)
    depth = ds["depth"].values
    if depth.ndim == 3:
        depth = depth[-1]  # final timestep = max extent

    lat_arr = ds["lat"].values
    lon_arr = ds["lon"].values
    ds.close()

    sim_mask = depth > 0.05  # same wet threshold convention as engine_comparator.py

    # Derive AOI bounds from dam location + buffer
    lat_deg_buffer = buffer_km / 110.54
    lon_deg_buffer = buffer_km / (111.32 * np.cos(np.radians(dam_lat)))
    aoi_bounds = (
        dam_lon - lon_deg_buffer, dam_lat - lat_deg_buffer,
        dam_lon + lon_deg_buffer, dam_lat + lat_deg_buffer,
    )

    sar_data = fetch_sentinel1_sar_extent(aoi_bounds, event_date)
    flood_mask_ee = derive_flood_extent_mask(sar_data)
    sar_mask = export_flood_mask_to_array(flood_mask_ee, aoi_bounds, target_shape=sim_mask.shape)

    metrics = compute_extent_metrics(sim_mask, sar_mask)

    return {
        "validation_metrics": metrics,
        "event_date": event_date,
        "aoi_bounds": aoi_bounds,
        "sar_pre_date_range": sar_data["pre_date_range"],
        "sar_post_date_range": sar_data["post_date_range"],
        "simulation_source": simulation_nc_path,
    }
