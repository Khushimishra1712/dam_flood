"""
Engine comparison service.

Compares inundation extent and hazard outputs from the two hydro engines
(SWE2D and SPH) run on the same dam-break scenario. Computes standard
spatial agreement metrics used in flood model validation:

  - IoU (Intersection over Union) of inundated extent
  - Critical Success Index (CSI) / Threat Score
  - Extent area difference (%)
  - Mean depth difference in overlapping wet cells

These same metrics are reused in gee_validator.py to compare simulation
output against real satellite-observed flood extent (Sentinel-1 SAR).
"""

import numpy as np
import xarray as xr


def _load_depth_grid(nc_path: str, timestep: int = -1) -> np.ndarray:
    """Loads the depth array from a simulation NetCDF at a given timestep (default: final)."""
    ds = xr.open_dataset(nc_path)
    depth = ds["depth"].values

    if depth.ndim == 3:
        depth = depth[timestep]

    ds.close()
    return depth


def _binarize_extent(depth_grid: np.ndarray, wet_threshold: float = 0.05) -> np.ndarray:
    """Converts a continuous depth grid into a binary wet/dry mask."""
    return depth_grid > wet_threshold


def compute_extent_metrics(mask_a: np.ndarray, mask_b: np.ndarray) -> dict:
    """
    Computes standard binary agreement metrics between two inundation
    extent masks (must be the same shape/grid).

    Args:
        mask_a: binary wet/dry mask (e.g., SWE2D extent)
        mask_b: binary wet/dry mask (e.g., SPH extent, or SAR-observed extent)

    Returns:
        dict of IoU, CSI, precision, recall, F1, and area statistics
    """
    if mask_a.shape != mask_b.shape:
        raise ValueError(
            f"Grid shape mismatch: mask_a {mask_a.shape} vs mask_b {mask_b.shape}. "
            "Both engines must run on the same DEM/grid for valid comparison."
        )

    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    area_a = mask_a.sum()
    area_b = mask_b.sum()

    iou = float(intersection / union) if union > 0 else 0.0

    # Critical Success Index (aka Threat Score) - standard flood validation metric
    # CSI = hits / (hits + misses + false_alarms) = intersection / union (same as IoU for binary masks)
    csi = iou

    # Precision/Recall treating mask_b as "reference/observed" and mask_a as "predicted"
    hits = intersection
    false_alarms = np.logical_and(mask_a, np.logical_not(mask_b)).sum()
    misses = np.logical_and(np.logical_not(mask_a), mask_b).sum()

    precision = float(hits / (hits + false_alarms)) if (hits + false_alarms) > 0 else 0.0
    recall = float(hits / (hits + misses)) if (hits + misses) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    area_diff_pct = float(abs(area_a - area_b) / max(area_a, area_b, 1) * 100.0)

    return {
        "iou": iou,
        "critical_success_index": csi,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "area_a_cells": int(area_a),
        "area_b_cells": int(area_b),
        "area_difference_pct": area_diff_pct,
    }


def compute_depth_difference(depth_a: np.ndarray, depth_b: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray) -> dict:
    """
    Computes depth statistics in cells where both engines agree water is
    present (overlapping wet extent) — RMSE and mean absolute difference.
    """
    overlap_mask = np.logical_and(mask_a, mask_b)

    if overlap_mask.sum() == 0:
        return {"warning": "No overlapping wet cells between engines - depth comparison skipped.", "rmse": None, "mae": None}

    diff = depth_a[overlap_mask] - depth_b[overlap_mask]
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))

    return {
        "rmse_m": rmse,
        "mae_m": mae,
        "overlapping_wet_cells": int(overlap_mask.sum()),
    }


def compare_engines(swe2d_nc_path: str, sph_nc_path: str, wet_threshold: float = 0.05) -> dict:
    """
    Main entrypoint: full comparison between SWE2D and SPH simulation outputs
    for the same dam-break scenario.

    Args:
        swe2d_nc_path: path to SWE2D NetCDF output
        sph_nc_path: path to SPH NetCDF output
        wet_threshold: minimum depth (m) to classify a cell as "wet"

    Returns:
        dict with extent agreement metrics and depth difference statistics
    """
    depth_swe = _load_depth_grid(swe2d_nc_path)
    depth_sph = _load_depth_grid(sph_nc_path)

    if depth_swe.shape != depth_sph.shape:
        raise ValueError(
            f"Cannot compare: SWE2D grid shape {depth_swe.shape} != SPH grid shape {depth_sph.shape}. "
            "Both engines must be run on the same DEM/catchment extent."
        )

    mask_swe = _binarize_extent(depth_swe, wet_threshold)
    mask_sph = _binarize_extent(depth_sph, wet_threshold)

    extent_metrics = compute_extent_metrics(mask_swe, mask_sph)
    depth_metrics = compute_depth_difference(depth_swe, depth_sph, mask_swe, mask_sph)

    return {
        "extent_agreement": extent_metrics,
        "depth_agreement": depth_metrics,
        "swe2d_source": swe2d_nc_path,
        "sph_source": sph_nc_path,
    }
