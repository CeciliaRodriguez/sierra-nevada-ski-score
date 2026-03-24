from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"

# Two elevation bands for Sierra Nevada, Spain
# Pradollano (~2100m) = base station weather
# Veleta (~3400m)     = summit / upper-mountain conditions
LOCATIONS = {
    "Pradollano": (37.0953, -3.3976),
    "Veleta":     (37.0649, -3.3754),
}

HIST_START = "2021-12-01"

# Scoring weights — must sum to 1.0
# Snow dominates because powder quality is the primary driver of a good ski day.
# Temperature is second: snow texture degrades above 0°C.
# Wind affects safety and visibility more than enjoyment.
# Sun is the weakest signal; a great powder day can be overcast.
WEIGHT_SNOW = 0.50
WEIGHT_TEMP = 0.25
WEIGHT_WIND = 0.15
WEIGHT_SUN  = 0.10

# Score thresholds for human-readable labels
LABEL_EPIC = 70
LABEL_GOOD = 50
LABEL_MEH  = 30

# RandomForest hyperparameters
RF_N_ESTIMATORS  = 180
RF_RANDOM_STATE  = 42
RF_TEST_SIZE     = 0.3
