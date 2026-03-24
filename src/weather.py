"""
Open-Meteo data fetchers.

Two APIs are used:
- Archive API: historical hourly data (2021–present), aggregated to daily.
- Forecast API: next 7 days at daily resolution.

No API key required. Rate limits are generous for daily cron usage.
"""

from datetime import date

import pandas as pd
import requests

ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def download_hourly_archive(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch hourly historical weather from Open-Meteo archive API."""
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start_date,
        "end_date":   end_date,
        "hourly":     "temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m,snowfall,cloudcover",
        "timezone":   "auto",
    }
    r = requests.get(ARCHIVE_URL, params=params, timeout=30)
    r.raise_for_status()
    h = r.json()["hourly"]

    return pd.DataFrame({
        "date":        pd.to_datetime(h["time"]),
        "temperature": h["temperature_2m"],
        "precip_mm":   h["precipitation"],
        "wind_kmh":    h["wind_speed_10m"],
        "gust_kmh":    h["wind_gusts_10m"],
        "new_snow_cm": h["snowfall"],
        "cloud_pct":   h["cloudcover"],
    })


def hourly_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly weather records to daily statistics."""
    return (
        df.set_index("date")
        .resample("D")
        .agg({
            "temperature": "mean",
            "precip_mm":   "sum",
            "wind_kmh":    "max",
            "gust_kmh":    "max",
            "new_snow_cm": "sum",
            "cloud_pct":   "mean",
        })
        .reset_index()
    )


def download_forecast_daily(lat: float, lon: float, start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch 7-day daily forecast from Open-Meteo forecast API."""
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "timezone":   "auto",
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date":   end_date.strftime("%Y-%m-%d"),
        "daily": (
            "temperature_2m_max,temperature_2m_min,"
            "snowfall_sum,windspeed_10m_max,cloudcover_mean"
        ),
    }
    r = requests.get(FORECAST_URL, params=params, timeout=30)
    r.raise_for_status()
    d = r.json()["daily"]

    return pd.DataFrame({
        "date":         pd.to_datetime(d["time"]),
        "t_min":        d["temperature_2m_min"],
        "t_max":        d["temperature_2m_max"],
        "snow_sum_cm":  d["snowfall_sum"],
        "wind_max_kmh": d["windspeed_10m_max"],
        "cloud_pct":    d["cloudcover_mean"],
    })
