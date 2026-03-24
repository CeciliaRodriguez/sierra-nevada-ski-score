"""
Main pipeline entry point.

Execution order:
  1. Download historical hourly weather (Open-Meteo archive) for both elevations
  2. Aggregate to daily, merge base + summit stations
  3. Compute rule-based scores on historical data → ski_score_historical.csv
  4. Train RandomForest on historical rule scores
  5. Download 7-day forecast (Open-Meteo forecast API)
  6. Apply rule-based scoring + RF prediction to forecast → ski_data.csv

Run with:
    python -m src.pipeline
"""

from datetime import date, timedelta

import pandas as pd

from src import config
from src.model import MODEL_FEATURES, predict, train
from src.scoring import compute_factors_and_penalties, label_score
from src.weather import download_forecast_daily, download_hourly_archive, hourly_to_daily


def run() -> None:
    today        = date.today()
    hist_end     = today.strftime("%Y-%m-%d")
    forecast_end = today + timedelta(days=6)

    print(f"Historical window : {config.HIST_START} → {hist_end}")
    print(f"Forecast window   : {today} → {forecast_end}")

    # ------------------------------------------------------------------ #
    # 1. Historical weather → daily aggregates
    # ------------------------------------------------------------------ #
    print("\nDownloading historical weather...")
    base_h = download_hourly_archive(*config.LOCATIONS["Pradollano"], config.HIST_START, hist_end)
    top_h  = download_hourly_archive(*config.LOCATIONS["Veleta"],     config.HIST_START, hist_end)

    base_daily = hourly_to_daily(base_h).add_prefix("base_").rename(columns={"base_date": "date"})
    top_daily  = hourly_to_daily(top_h).add_prefix("top_").rename(columns={"top_date": "date"})

    df_hist = pd.merge(base_daily, top_daily, on="date", how="inner").sort_values("date")

    df_hist["avg_temp_c"]      = df_hist[["base_temperature", "top_temperature"]].mean(axis=1)
    df_hist["wind_max_kmh"]    = df_hist[["base_wind_kmh", "top_wind_kmh"]].max(axis=1)
    df_hist["new_snow_cm"]     = df_hist[["base_new_snow_cm", "top_new_snow_cm"]].mean(axis=1)
    df_hist["sky_clear_ratio"] = 1 - df_hist[["base_cloud_pct", "top_cloud_pct"]].mean(axis=1) / 100.0

    # ------------------------------------------------------------------ #
    # 2. Rule-based scores on historical data
    # ------------------------------------------------------------------ #
    hist_scores = compute_factors_and_penalties(
        df_hist[["new_snow_cm", "avg_temp_c", "wind_max_kmh", "sky_clear_ratio"]]
    )
    df_hist = pd.concat([df_hist, hist_scores.add_prefix("hist_")], axis=1)

    config.DATA_DIR.mkdir(exist_ok=True)
    df_hist.to_csv(config.DATA_DIR / "ski_score_historical.csv", index=False)
    print(f"Historical scores saved ({len(df_hist)} days)")
    print(df_hist["hist_ski_label"].value_counts().to_string())

    # ------------------------------------------------------------------ #
    # 3. Train model on historical rule scores
    # ------------------------------------------------------------------ #
    print("\nTraining RandomForest...")
    rf, r2 = train(df_hist)
    print(f"R² on test split: {r2}")

    # ------------------------------------------------------------------ #
    # 4. Forecast weather
    # ------------------------------------------------------------------ #
    print("\nDownloading 7-day forecast...")
    base_f = download_forecast_daily(*config.LOCATIONS["Pradollano"], today, forecast_end).add_prefix("base_")
    top_f  = download_forecast_daily(*config.LOCATIONS["Veleta"],     today, forecast_end).add_prefix("top_")

    f = (
        pd.merge(base_f, top_f, left_on="base_date", right_on="top_date", how="inner")
        .rename(columns={"base_date": "date"})
        .sort_values("date")
        .reset_index(drop=True)
    )

    f["avg_temp_c"]      = (f["base_t_min"] + f["base_t_max"] + f["top_t_min"] + f["top_t_max"]) / 4
    f["new_snow_cm"]     = (f["base_snow_sum_cm"] + f["top_snow_sum_cm"]) / 2
    f["wind_max_kmh"]    = f[["base_wind_max_kmh", "top_wind_max_kmh"]].max(axis=1)
    f["sky_clear_ratio"] = 1 - (f["base_cloud_pct"] + f["top_cloud_pct"]) / 2 / 100.0

    # ------------------------------------------------------------------ #
    # 5. Rule-based scores on forecast
    # ------------------------------------------------------------------ #
    rule_input  = f[["new_snow_cm", "avg_temp_c", "wind_max_kmh", "sky_clear_ratio"]].copy()
    rule_scores = compute_factors_and_penalties(rule_input)

    f["ski_score_rule"]           = rule_scores["ski_score"]
    f["ski_label_rule"]           = rule_scores["ski_label"]
    f["rule_snow_factor"]         = rule_scores["snow_factor"]
    f["rule_temp_factor"]         = rule_scores["temp_factor"]
    f["rule_wind_factor"]         = rule_scores["wind_factor"]
    f["rule_sun_factor"]          = rule_scores["sun_factor"]
    f["rule_penalty_snow"]        = rule_scores["penalty_snow"]
    f["rule_penalty_temp"]        = rule_scores["penalty_temp"]
    f["rule_penalty_wind"]        = rule_scores["penalty_wind"]
    f["rule_penalty_sun"]         = rule_scores["penalty_sun"]
    f["main_penalty_reason_rule"] = rule_scores["main_penalty_reason"]
    f["score_type_rule"]          = "7_day_forecast_rule_based"

    # ------------------------------------------------------------------ #
    # 6. ML scores on forecast
    # ------------------------------------------------------------------ #
    f["ski_score_ml"]           = predict(rf, f)
    f["ski_label_ml"]           = f["ski_score_ml"].apply(label_score)
    f["score_type_ml"]          = "7_day_forecast_ml_trained_on_history"

    ml_scores = compute_factors_and_penalties(rule_input)
    f["ml_snow_factor"]         = ml_scores["snow_factor"]
    f["ml_temp_factor"]         = ml_scores["temp_factor"]
    f["ml_wind_factor"]         = ml_scores["wind_factor"]
    f["ml_sun_factor"]          = ml_scores["sun_factor"]
    f["ml_penalty_snow"]        = ml_scores["penalty_snow"]
    f["ml_penalty_temp"]        = ml_scores["penalty_temp"]
    f["ml_penalty_wind"]        = ml_scores["penalty_wind"]
    f["ml_penalty_sun"]         = ml_scores["penalty_sun"]
    f["main_penalty_reason_ml"] = ml_scores["main_penalty_reason"]

    # ------------------------------------------------------------------ #
    # 7. Save forecast output
    # ------------------------------------------------------------------ #
    f.to_csv(config.DATA_DIR / "ski_data.csv", index=False)
    print("\nForecast saved to data/ski_data.csv")
    print(
        f[["date", "ski_score_rule", "ski_label_rule", "ski_score_ml",
           "ski_label_ml", "main_penalty_reason_rule"]].to_string(index=False)
    )


if __name__ == "__main__":
    run()
