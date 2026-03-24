# Sierra Nevada Ski Score

Daily ski condition scoring for Sierra Nevada, Spain.

A Python pipeline that pulls weather data from two elevation bands, computes a
rule-based score with penalty decomposition, trains a RandomForest on 3+ years
of historical data, and applies the model to a 7-day forecast.
Refreshed automatically each morning via GitHub Actions.

---

## Problem

Publicly available forecasts show raw weather numbers — snowfall, temperature, wind —
but don't translate them into a skier-relevant quality signal.
A "heavy snowfall" day with 80 km/h winds scores very differently from one with calm conditions.
And when the score is low, you want to know *why* — not just see a number.

---

## Approach

### 1. Data collection
Hourly weather pulled from [Open-Meteo](https://open-meteo.com/) (free, no API key required)
for two stations:

| Station    | Elevation | Role |
|------------|-----------|------|
| Pradollano | ~2100m    | Base — temperature and snowfall reference |
| Veleta     | ~3400m    | Summit — upper-mountain wind and snow conditions |

Historical window: `2021-12-01` → today (archive API).
Forecast: today + 6 days (forecast API, daily resolution).

### 2. Feature engineering
Four daily features derived from the merged two-station data:

| Feature | Derivation |
|---------|------------|
| `new_snow_cm` | Mean snowfall across both stations |
| `avg_temp_c` | Mean temperature across both stations |
| `wind_max_kmh` | Max wind speed across both stations |
| `sky_clear_ratio` | 1 − mean cloud cover fraction |

### 3. Rule-based score
Weighted combination of four normalized factors (0–1 each):

| Factor | Weight | Normalization range | Rationale |
|--------|--------|---------------------|-----------|
| Snow   | 50%    | 0–10 cm             | Primary driver of powder quality |
| Temp   | 25%    | −5°C to +5°C (inverted) | Ideal = cold end; snow degrades above 0°C |
| Wind   | 15%    | 0–50 km/h (inverted) | Safety and visibility |
| Sun    | 10%    | 0.3–1.0 clear ratio  | Enjoyment; below 0.3 already very cloudy |

Final score = weighted sum × 100, rounded to integer.
Labels: **Epic** (≥70) · **Good** (≥50) · **Meh** (≥30) · **Bad** (<30)

### 4. Penalty decomposition
Each day also includes a `main_penalty_reason` string identifying the one or two factors
dragging the score down most — e.g. *"No recent snowfall + strong winds"*.
This is computed from weighted penalty rankings, with value-aware text for each threshold.
The explainability layer is what makes the output useful for actual decisions, not just display.

### 5. RandomForest model
A `RandomForestRegressor` is trained on historical rule-based scores and applied to the
7-day forecast. This smooths noise in the rule-based signal and captures non-linear
interactions between features.

**Why retrain on every run?**
The model fits fresh each day on the full historical dataset (~1500 rows).
This keeps the latest observations in scope without managing model artifacts or
a separate retraining schedule. Fitting takes ~2 seconds; the tradeoff is worth it.

---

## Output

### `data/ski_data.csv` — 7-day forecast (refreshed daily by CI)

| Column | Description |
|--------|-------------|
| `date` | Forecast date |
| `ski_score_rule` | Rule-based score (0–100) |
| `ski_label_rule` | Epic / Good / Meh / Bad |
| `main_penalty_reason_rule` | Human-readable top penalty reason |
| `ski_score_ml` | RandomForest prediction |
| `ski_label_ml` | Label from ML score |
| `new_snow_cm` | Avg snowfall (cm) |
| `avg_temp_c` | Avg temperature (°C) |
| `wind_max_kmh` | Max wind speed (km/h) |
| `sky_clear_ratio` | Clear sky fraction (0–1) |
| `rule_{snow,temp,wind,sun}_factor` | Per-factor normalized values |
| `rule_penalty_{snow,temp,wind,sun}` | Per-factor weighted penalties |

### `data/ski_score_historical.csv` — full history from 2021-12-01

Same schema as above with `hist_` prefix on score/label/factor columns.
Used as the training set for the RF model on each run.

---

## Key design decisions

**Dual scoring (rule + ML):** The rule-based score is fully interpretable and auditable.
The ML layer adds noise reduction and non-linear calibration. Both scores are surfaced
in the output so the consumer can choose which to display or compare them.

**No model artifact committed:** The model is retrained from scratch each run.
For a daily pipeline at this scale, a saved `.pkl` adds operational complexity
(versioning, drift, retraining triggers) with no meaningful benefit.

**Two elevation bands:** Using only the base station would miss upper-mountain wind
and snow conditions that directly affect ridable terrain. Averaging or taking the max
across both stations captures this without adding a third data source.

**Explainability via penalty ranking:** The `main_penalty_reason` field is derived from
weighted penalty magnitudes, not a post-hoc lookup table. When two factors are close,
both are included. This makes the output trustworthy — it reflects the actual score math.

---

## How to run

```bash
git clone https://github.com/ceciliarodriguez/sierra-nevada-ski-score
cd sierra-nevada-ski-score
pip install -r requirements.txt
python -m src.pipeline
```

Output CSVs written to `data/`.

---

## Automation

A GitHub Actions workflow runs daily at 06:15 UTC (08:15 Madrid time),
executes the full pipeline, and commits updated CSVs if the data changed.

Trigger manually from the [Actions tab](../../actions/workflows/update_scores.yml)
or via `gh workflow run update_scores.yml`.

---

## Project structure

```
sierra-nevada-ski-score/
├── .github/workflows/update_scores.yml   # daily cron + manual dispatch
├── data/
│   ├── ski_data.csv                      # 7-day forecast output (auto-updated)
│   └── ski_score_historical.csv          # full history from 2021-12-01
├── src/
│   ├── config.py                         # locations, weights, RF hyperparameters
│   ├── weather.py                        # Open-Meteo archive + forecast fetchers
│   ├── scoring.py                        # rule-based score + penalty decomposition
│   ├── model.py                          # RandomForest train / predict
│   └── pipeline.py                       # orchestration — runs the full pipeline
├── README.md
└── requirements.txt
```

---

## Tech stack

Python · pandas · scikit-learn · Open-Meteo API · GitHub Actions
