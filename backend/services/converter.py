"""
Converts hydrodynamic simulation NetCDF output (depth, velocity_x, velocity_y)
into hazard classification vector layers (KML/GeoJSON) using the standard
Hazard Rating formula: HR = Depth * (Velocity + 0.5)

Hazard Classes:
  HR < 0.5         -> Low
  0.5 <= HR < 1.0  -> Medium
  1.0 <= HR < 2.0  -> High
  HR >= 2.0        -> Extreme
"""

import os
import numpy as np
import xarray as xr
from rasterio import features
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import geopandas as gpd
from shapely.geometry import shape
import simplekml

HAZARD_BINS = [0, 0.5, 1.0, 2.0, np.inf]
HAZARD_LABELS = ["Low", "Medium", "High", "Extreme"]
HAZARD_COLORS = {
    "Low": "3f00ff00",       # AABBGGRR (KML) - translucent green
    "Medium": "3f00ffff",    # yellow
    "High": "3f0080ff",      # orange
    "Extreme": "3f0000ff",   # red
}


def _compute_hazard_raster(nc_path: str):
    """Reads NetCDF, computes HR grid and returns array + geotransform + crs."""
    ds = xr.open_dataset(nc_path)

    required = ["depth", "velocity_x", "velocity_y"]
    for var in required:
        if var not in ds.variables:
            raise ValueError(f"Missing required variable '{var}' in NetCDF: {nc_path}")

    depth = ds["depth"].values.astype("float64")
    vx = ds["velocity_x"].values.astype("float64")
    vy = ds["velocity_y"].values.astype("float64")

    # Handle time dimension - use final timestep (max inundation extent)
    if depth.ndim == 3:
        depth = depth[-1]
        vx = vx[-1]
        vy = vy[-1]

    velocity_mag = np.sqrt(vx**2 + vy**2)
    hazard = depth * (velocity_mag + 0.5)

    # Mask dry cells
    hazard = np.where(depth > 0.01, hazard, np.nan)

    # Extract spatial bounds
    lat = ds["lat"].values if "lat" in ds.variables else ds["y"].values
    lon = ds["lon"].values if "lon" in ds.variables else ds["x"].values

    minx, maxx = float(np.min(lon)), float(np.max(lon))
    miny, maxy = float(np.min(lat)), float(np.max(lat))

    height, width = hazard.shape
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    crs = CRS.from_epsg(4326)

    ds.close()
    return hazard, transform, crs


def _classify_hazard(hazard_array: np.ndarray) -> np.ndarray:
    """Bins continuous HR values into class indices (0-3), 255 for nodata."""
    classified = np.full(hazard_array.shape, 255, dtype="uint8")
    valid_mask = ~np.isnan(hazard_array)

    class_idx = np.digitize(hazard_array[valid_mask], HAZARD_BINS[1:-1])
    classified[valid_mask] = class_idx.astype("uint8")

    return classified


def _polygonize_classified_raster(classified: np.ndarray, transform, crs) -> gpd.GeoDataFrame:
    """Vectorizes a classified raster into polygons per hazard class."""
    mask = classified != 255

    shapes_gen = features.shapes(classified, mask=mask, transform=transform)

    geometries = []
    labels = []
    for geom, value in shapes_gen:
        idx = int(value)
        if 0 <= idx < len(HAZARD_LABELS):
            geometries.append(shape(geom))
            labels.append(HAZARD_LABELS[idx])

    if not geometries:
        raise ValueError("No inundated hazard zones found (all cells dry or nodata).")

    gdf = gpd.GeoDataFrame({"hazard_class": labels, "geometry": geometries}, crs=crs)

    # Dissolve adjacent polygons of the same class for cleaner output
    gdf = gdf.dissolve(by="hazard_class", as_index=False)
    return gdf


def _export_geojson(gdf: gpd.GeoDataFrame, out_path: str):
    gdf.to_file(out_path, driver="GeoJSON")


def _export_kml(gdf: gpd.GeoDataFrame, out_path: str):
    kml = simplekml.Kml()
    for _, row in gdf.iterrows():
        label = row["hazard_class"]
        color = HAZARD_COLORS.get(label, "3f7f7f7f")
        geom = row["geometry"]

        polygons = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)

        for poly in polygons:
            pol = kml.newpolygon(name=f"Hazard: {label}")
            exterior_coords = [(x, y) for x, y in poly.exterior.coords]
            pol.outerboundaryis = exterior_coords

            for interior in poly.interiors:
                pol.innerboundaryis = [(x, y) for x, y in interior.coords]

            pol.style.polystyle.color = color
            pol.style.polystyle.outline = 1
            pol.style.linestyle.color = "ff000000"
            pol.style.linestyle.width = 1

    kml.save(out_path)


def netcdf_to_hazard_vector(nc_path: str, out_format: str = "kml") -> dict:
    """
    Main entrypoint: converts simulation NetCDF to hazard vector layer(s).

    Args:
        nc_path: path to input NetCDF file with depth, velocity_x, velocity_y
        out_format: "kml", "geojson", or "both"

    Returns:
        dict with output file path(s)
    """
    if not os.path.exists(nc_path):
        raise FileNotFoundError(f"NetCDF file not found: {nc_path}")

    if out_format not in ("kml", "geojson", "both"):
        raise ValueError("out_format must be 'kml', 'geojson', or 'both'")

    hazard_array, transform, crs = _compute_hazard_raster(nc_path)
    classified = _classify_hazard(hazard_array)
    gdf = _polygonize_classified_raster(classified, transform, crs)

    base = os.path.splitext(nc_path)[0]
    outputs = {}

    if out_format in ("geojson", "both"):
        geojson_path = f"{base}_hazard.geojson"
        _export_geojson(gdf, geojson_path)
        outputs["geojson"] = geojson_path

    if out_format in ("kml", "both"):
        kml_path = f"{base}_hazard.kml"
        _export_kml(gdf, kml_path)
        outputs["kml"] = kml_path

    return outputs
