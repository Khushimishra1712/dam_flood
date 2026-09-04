"""
Impact/loss assessment service.

Overlays hazard classification polygons (from converter.py) with population
raster data (e.g., WorldPop) to estimate affected population by hazard
severity class. Also supports overlay with land-use/land-cover (LULC) data
to estimate affected infrastructure/agricultural land categories.

Designed to work even without population data present (skips gracefully)
so the pipeline never hard-fails if optional datasets aren't available yet.
"""

import os
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import reproject, Resampling, calculate_default_transform
import geopandas as gpd


def _reproject_raster_to_match(src_path: str, ref_transform, ref_crs, ref_shape):
    """Reprojects/resamples a raster (e.g., population) onto the reference grid (hazard raster's grid)."""
    with rasterio.open(src_path) as src:
        dst_array = np.zeros(ref_shape, dtype=np.float64)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear,
        )
    return dst_array


def estimate_population_impact(
    hazard_gdf: gpd.GeoDataFrame,
    population_raster_path: str,
    ref_transform,
    ref_crs,
    ref_shape,
) -> dict:
    """
    Estimates population affected within each hazard class polygon by
    summing population raster values inside each polygon's footprint.

    Args:
        hazard_gdf: GeoDataFrame with 'hazard_class' and 'geometry' columns
        population_raster_path: path to population count raster (e.g., WorldPop)
        ref_transform, ref_crs, ref_shape: hazard raster's spatial reference,
            used to align population data if reprojection is needed

    Returns:
        dict mapping hazard_class -> estimated affected population
    """
    if not os.path.exists(population_raster_path):
        return {"warning": "Population raster not found - impact estimate skipped.", "by_class": {}}

    results = {}
    total_affected = 0.0

    with rasterio.open(population_raster_path) as pop_src:
        for _, row in hazard_gdf.iterrows():
            geom = [row["geometry"].__geo_interface__]
            label = row["hazard_class"]

            try:
                out_image, _ = rio_mask(pop_src, geom, crop=True, nodata=0)
                class_population = float(np.nansum(out_image))
            except ValueError:
                # Polygon doesn't overlap population raster extent
                class_population = 0.0

            results[label] = results.get(label, 0.0) + class_population
            total_affected += class_population

    return {
        "by_class": results,
        "total_affected_population": total_affected,
    }


def estimate_landuse_impact(
    hazard_gdf: gpd.GeoDataFrame,
    lulc_raster_path: str,
    lulc_class_map: dict | None = None,
) -> dict:
    """
    Estimates affected area (in km2) per land-use/land-cover class within
    the inundated hazard extent.

    Args:
        hazard_gdf: GeoDataFrame with hazard polygons
        lulc_raster_path: path to LULC classification raster (e.g., Bhuvan LULC)
        lulc_class_map: optional dict mapping raster class codes -> readable
            labels (e.g., {1: "Agriculture", 2: "Built-up", 3: "Forest"})

    Returns:
        dict mapping LULC class label -> affected area in km2
    """
    if not os.path.exists(lulc_raster_path):
        return {"warning": "LULC raster not found - land-use impact estimate skipped.", "by_class": {}}

    if lulc_class_map is None:
        lulc_class_map = {
            1: "Agriculture",
            2: "Built-up / Settlement",
            3: "Forest",
            4: "Water Body",
            5: "Barren / Wasteland",
            6: "Grassland",
        }

    class_areas_km2 = {}

    with rasterio.open(lulc_raster_path) as lulc_src:
        pixel_area_km2 = abs(lulc_src.transform.a * lulc_src.transform.e) / 1_000_000.0

        # Merge all hazard polygons into a single inundation extent for LULC overlay
        combined_geom = [hazard_gdf.geometry.unary_union.__geo_interface__]

        try:
            out_image, _ = rio_mask(lulc_src, combined_geom, crop=True, nodata=0)
            class_codes, counts = np.unique(out_image[out_image != 0], return_counts=True)

            for code, count in zip(class_codes, counts):
                label = lulc_class_map.get(int(code), f"Class_{int(code)}")
                class_areas_km2[label] = float(count) * pixel_area_km2

        except ValueError:
            pass

    return {
        "by_class": class_areas_km2,
        "total_affected_area_km2": sum(class_areas_km2.values()),
    }


def run_impact_assessment(
    hazard_geojson_path: str,
    population_raster_path: str = "data/validation/population.tif",
    lulc_raster_path: str = "data/validation/lulc.tif",
) -> dict:
    """
    Main entrypoint: runs full impact assessment on a hazard GeoJSON output
    from converter.netcdf_to_hazard_vector(). Gracefully skips any
    sub-assessment whose input dataset is missing.

    Args:
        hazard_geojson_path: path to hazard GeoJSON (from converter.py)
        population_raster_path: path to population raster (optional dataset)
        lulc_raster_path: path to LULC raster (optional dataset)

    Returns:
        dict with population and land-use impact summaries
    """
    if not os.path.exists(hazard_geojson_path):
        raise FileNotFoundError(f"Hazard GeoJSON not found: {hazard_geojson_path}")

    hazard_gdf = gpd.read_file(hazard_geojson_path)

    if hazard_gdf.empty:
        raise ValueError("Hazard GeoDataFrame is empty - no inundation zones to assess.")

    ref_bounds = hazard_gdf.total_bounds
    ref_crs = hazard_gdf.crs

    population_impact = estimate_population_impact(
        hazard_gdf, population_raster_path, None, ref_crs, None
    )

    landuse_impact = estimate_landuse_impact(hazard_gdf, lulc_raster_path)

    return {
        "population_impact": population_impact,
        "landuse_impact": landuse_impact,
        "hazard_classes_present": hazard_gdf["hazard_class"].unique().tolist(),
        "total_inundated_polygons": len(hazard_gdf),
    }
