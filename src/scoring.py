"""
Rule-based ski scoring engine with penalty decomposition.

Each day gets a score from 0 to 100 built from four factors.
The penalty decomposition identifies which factor is dragging the score down most,
and produces a human-readable reason string (e.g. "No recent snowfall + strong winds").

This explainability layer is what makes the output useful rather than just numeric.
"""

import numpy as np
import pandas as pd

from src.config import WEIGHT_SNOW, WEIGHT_TEMP, WEIGHT_WIND, WEIGHT_SUN, LABEL_EPIC, LABEL_GOOD, LABEL_MEH


def norm(x, lo: float, hi: float):
    """Linear scale to 0–1 with clip to [0, 1]."""
    return ((x - lo) / (hi - lo)).clip(0, 1)


def label_score(s: float) -> str:
    if s >= LABEL_EPIC: return "Epic"
    if s >= LABEL_GOOD: return "Good"
    if s >= LABEL_MEH:  return "Meh"
    return "Bad"


def compute_factors_and_penalties(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rule-based ski score + per-factor penalty decomposition.

    Expected input columns:
        new_snow_cm      — daily snowfall averaged across elevations
        avg_temp_c       — average temperature (°C)
        wind_max_kmh     — maximum wind speed (km/h)
        sky_clear_ratio  — fraction of sky that is clear (0–1)

    Returns a DataFrame with:
        {snow,temp,wind,sun}_factor   — normalized factor values (0–1)
        ski_score                     — weighted score (0–100)
        ski_label                     — Epic / Good / Meh / Bad
        penalty_{snow,temp,wind,sun}  — weighted penalty per factor
        main_penalty_reason           — primary (+ optional secondary) reason string
    """
    snow_factor = norm(df["new_snow_cm"], 0, 10)
    temp_factor = 1 - norm(df["avg_temp_c"], -5, 5)    # ideal = cold end of range
    wind_factor = 1 - norm(df["wind_max_kmh"], 0, 50)  # ideal = calm
    sun_factor  = norm(df["sky_clear_ratio"], 0.3, 1.0)

    penalty_snow = WEIGHT_SNOW * (1 - snow_factor)
    penalty_temp = WEIGHT_TEMP * (1 - temp_factor)
    penalty_wind = WEIGHT_WIND * (1 - wind_factor)
    penalty_sun  = WEIGHT_SUN  * (1 - sun_factor)

    penalties = pd.DataFrame(
        {"snow": penalty_snow, "temp": penalty_temp, "wind": penalty_wind, "sun": penalty_sun}
    )

    # Primary reason: dominant penalty factor with value-aware text
    max_cols = penalties.idxmax(axis=1)

    cloud_ratio = 1 - df["sky_clear_ratio"]
    sun_text  = np.where(cloud_ratio >= 0.90, "Low visibility",
                np.where(cloud_ratio >= 0.75, "Overcast skies", "Cloudy conditions"))
    snow_text = np.where(df["new_snow_cm"] < 0.1, "No recent snowfall",
                np.where(df["new_snow_cm"] < 2,   "Very low snowfall", "Low snowfall"))
    wind_text = np.where(df["wind_max_kmh"] > 70, "Dangerous wind",
                np.where(df["wind_max_kmh"] > 40, "Too windy", "Gusty conditions"))
    temp_text = np.where(df["avg_temp_c"] > 5,  "Spring thaw – too warm",
                np.where(df["avg_temp_c"] > 3,  "Warm snow conditions",
                np.where(df["avg_temp_c"] > 1,  "Mild temperatures",
                                                "Above-ideal temperatures")))

    primary_lookup = pd.DataFrame(
        {"snow": snow_text, "temp": temp_text, "wind": wind_text, "sun": sun_text},
        index=df.index,
    )
    primary_reason = pd.Series(
        [primary_lookup.at[i, col] for i, col in zip(df.index, max_cols)],
        index=df.index,
    )

    # Secondary reason: second-ranked factor, included only when material
    penalty_ranks = penalties.rank(axis=1, ascending=False, method="first").astype(int)
    second_cols   = (penalty_ranks == 2).idxmax(axis=1)

    primary_vals = penalties.max(axis=1)
    second_vals  = pd.Series(
        [penalties.at[i, col] for i, col in zip(penalties.index, second_cols)],
        index=penalties.index,
    )

    sec_wind_text = np.where(df["wind_max_kmh"] > 50, "dangerous gusts",
                   np.where(df["wind_max_kmh"] > 40, "strong winds",
                   np.where(df["wind_max_kmh"] > 35, "gusty winds", "moderate gusts")))
    sec_temp_text = np.where(df["avg_temp_c"] > 3, "warm conditions",
                   np.where(df["avg_temp_c"] > 1,  "mild temperatures", "marginal cold"))
    sec_snow_text = np.where(df["new_snow_cm"] < 0.1, "no fresh snow", "limited fresh snow")

    sec_lookup = pd.DataFrame(
        {
            "snow": sec_snow_text,
            "temp": sec_temp_text,
            "wind": sec_wind_text,
            "sun":  pd.Series(["low visibility"] * len(df), index=df.index),
        },
        index=df.index,
    )
    secondary_label = pd.Series(
        [sec_lookup.at[i, col] for i, col in zip(df.index, second_cols)],
        index=df.index,
    )

    # Include secondary only when ≥25% of primary penalty AND >5% absolute score impact
    include_secondary = (second_vals >= 0.25 * primary_vals) & (second_vals >= 0.05)

    main_penalty_reason = pd.Series(
        [f"{p} + {s}" if inc else p
         for p, s, inc in zip(primary_reason, secondary_label, include_secondary)],
        index=df.index,
    )

    ski_score = (
        WEIGHT_SNOW * snow_factor +
        WEIGHT_TEMP * temp_factor +
        WEIGHT_WIND * wind_factor +
        WEIGHT_SUN  * sun_factor
    ) * 100

    return pd.DataFrame({
        "snow_factor":         snow_factor,
        "temp_factor":         temp_factor,
        "wind_factor":         wind_factor,
        "sun_factor":          sun_factor,
        "ski_score":           ski_score.round(0),
        "ski_label":           ski_score.round(0).apply(label_score),
        "penalty_snow":        penalty_snow.round(3),
        "penalty_temp":        penalty_temp.round(3),
        "penalty_wind":        penalty_wind.round(3),
        "penalty_sun":         penalty_sun.round(3),
        "main_penalty_reason": main_penalty_reason,
    })
