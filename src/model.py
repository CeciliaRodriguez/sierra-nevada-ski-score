"""
RandomForest model for ski score forecasting.

The model is trained on rule-based scores computed from 3+ years of historical
weather data. This serves two purposes:
  1. Smooth noise in the rule-based signal (weather data has outliers).
  2. Learn non-linear interactions between features that the weighted formula misses.

Design decision — retrain on every pipeline run:
  The model trains fresh on the full historical dataset each time the pipeline runs.
  This ensures the latest data is always included and avoids managing model artifacts
  (serialized .pkl files, versioning, drift detection) for what is a simple daily job.
  Retraining takes ~2 seconds; the tradeoff is clearly worth it at this scale.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from src.config import RF_N_ESTIMATORS, RF_RANDOM_STATE, RF_TEST_SIZE

MODEL_FEATURES = ["new_snow_cm", "avg_temp_c", "wind_max_kmh", "sky_clear_ratio"]


def train(df_hist: pd.DataFrame) -> tuple[RandomForestRegressor, float]:
    """
    Fit a RandomForest on historical rule-based ski scores.

    Returns the fitted model and R² on the held-out test split.
    """
    X = df_hist[MODEL_FEATURES]
    y = df_hist["hist_ski_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=RF_TEST_SIZE, random_state=RF_RANDOM_STATE
    )

    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    return rf, round(rf.score(X_test, y_test), 4)


def predict(rf: RandomForestRegressor, df_forecast: pd.DataFrame) -> pd.Series:
    """Apply fitted model to forecast feature matrix."""
    return pd.Series(
        rf.predict(df_forecast[MODEL_FEATURES]).round(0),
        index=df_forecast.index,
    )
