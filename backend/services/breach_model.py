"""
Dam breach hydrograph generator.

Implements the Froehlich (2008) empirical breach parameter equations to
convert dam geometry (crest elevation, breach width, formation time,
reservoir volume) into a time-series outflow hydrograph (m3/s vs time).

This hydrograph becomes the upstream inflow boundary condition for the
SWE2D and SPH hydro engines.

Reference: Froehlich, D.C. (2008). "Embankment Dam Breach Parameters and
Their Uncertainties." Journal of Hydraulic Engineering, ASCE.
"""

import numpy as np


def estimate_breach_parameters(
    crest_elevation: float,
    breach_width: float,
    breach_formation_time: float,
    reservoir_volume: float | None = None,
    dam_height: float | None = None,
):
    """
    Estimates/validates breach geometry parameters.

    If reservoir_volume is not supplied, it is roughly estimated from
    dam_height using a simple empirical relation (very approximate — should
    be replaced with actual reservoir capacity data when available).

    Args:
        crest_elevation: dam crest elevation (m, MSL)
        breach_width: average breach width (m) — user input
        breach_formation_time: time for breach to fully form (hours)
        reservoir_volume: reservoir storage volume at breach (m3), optional
        dam_height: height of dam from breach invert to crest (m), optional

    Returns:
        dict of validated/estimated breach parameters
    """
    if dam_height is None:
        # Fallback assumption: breach height ~ 80% of crest elevation above
        # an assumed streambed offset if no explicit dam height is given.
        dam_height = crest_elevation * 0.8

    if reservoir_volume is None:
        # Rough empirical fallback (Froehlich regression uses volume directly;
        # this is a placeholder until real reservoir survey data is available)
        reservoir_volume = 1000 * (dam_height ** 2.5)

    return {
        "dam_height": dam_height,
        "breach_width": breach_width,
        "breach_formation_time_hr": breach_formation_time,
        "reservoir_volume": reservoir_volume,
    }


def froehlich_peak_outflow(dam_height: float, reservoir_volume: float) -> float:
    """
    Froehlich (2008) peak breach outflow regression equation:
        Qp = 0.607 * V^0.295 * Hb^1.24

    Args:
        dam_height: breach height (m)
        reservoir_volume: reservoir volume at time of breach (m3)

    Returns:
        Peak outflow discharge Qp (m3/s)
    """
    Qp = 0.607 * (reservoir_volume ** 0.295) * (dam_height ** 1.24)
    return Qp


def generate_breach_hydrograph(
    crest_elevation: float,
    breach_width: float,
    breach_formation_time: float,
    reservoir_volume: float | None = None,
    dam_height: float | None = None,
    n_timesteps: int = 100,
    total_duration_hours: float | None = None,
) -> dict:
    """
    Generates a full time-series breach outflow hydrograph.

    Uses a standard triangular/exponential-decay hydrograph shape:
    - Rising limb: 0 to Qp over breach_formation_time (matches physical
      breach widening process)
    - Falling limb: exponential decay as reservoir drains, over a duration
      roughly 3-5x the formation time

    Args:
        crest_elevation: dam crest elevation (m, MSL)
        breach_width: breach width (m)
        breach_formation_time: hours for breach to form
        reservoir_volume: reservoir volume (m3), optional
        dam_height: dam height (m), optional
        n_timesteps: number of points in the output hydrograph
        total_duration_hours: total simulation duration; defaults to
            5x breach_formation_time if not provided

    Returns:
        dict with 'time_hours', 'discharge_m3s' arrays and breach metadata
    """
    params = estimate_breach_parameters(
        crest_elevation=crest_elevation,
        breach_width=breach_width,
        breach_formation_time=breach_formation_time,
        reservoir_volume=reservoir_volume,
        dam_height=dam_height,
    )

    Qp = froehlich_peak_outflow(params["dam_height"], params["reservoir_volume"])

    if total_duration_hours is None:
        total_duration_hours = breach_formation_time * 5.0

    time = np.linspace(0, total_duration_hours, n_timesteps)
    discharge = np.zeros(n_timesteps)

    tf = breach_formation_time  # time to peak (breach fully formed)

    # Rising limb: linear ramp-up to Qp as breach widens
    rising_mask = time <= tf
    discharge[rising_mask] = Qp * (time[rising_mask] / tf)

    # Falling limb: exponential decay as reservoir empties
    falling_mask = time > tf
    decay_constant = 2.0 / (total_duration_hours - tf)  # tuned so discharge decays to ~13% of Qp by end
    discharge[falling_mask] = Qp * np.exp(-decay_constant * (time[falling_mask] - tf))

    return {
        "time_hours": time.tolist(),
        "discharge_m3s": discharge.tolist(),
        "peak_discharge_m3s": float(Qp),
        "time_to_peak_hours": float(tf),
        "breach_width_m": breach_width,
        "dam_height_m": params["dam_height"],
        "reservoir_volume_m3": params["reservoir_volume"],
    }
