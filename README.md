# Data_Science_Project_23106359
# Short-Term Solar Irradiance Forecasting


NOTE:> !!! PLEASE GO THROUGH D_S_Project_23106359_19.ipynb FILE, D_S_Project_23106359.ipynb MAY NOT DISPLAY CODE AND OUTPUT GRAPHS ON THIS PAGE works on notebook though. THANK YOU!!!

NOTE: (T+H), H can be replaced with (1,2...12) for multi-horizons forecasting. In this project as I have predicted 1 hour ahead, its 't+2', my data are at 30-minute intervals, a 1-hour-ahead forecast is two steps ahead (2 × 30 min = 60 min).

One-Hour-Ahead Solar Irradiance Forecasting Using
Machine Learning and Deep Learning Models: A
Comparative Study


## Abstract

Accurate solar forecasting is critical for the integration of photovoltaic energy into the power grid. This project develops and compares three forecasting models—XGBoost, GRU, and SARIMAX—to predict GHI one hour ahead using engineered features from historical irradiance, meteorological data, and temporal information. The models were rigorously trained and tested using chronological, expanding-window cross-validation to prevent data leakage.

## Key Results Summary

XGBoost was the best-performing model, achieving the lowest error (MAE: 38.45 W/m², RMSE: 57.83 W/m²) and the highest directional accuracy (90.3%). It also trained the fastest (~30 seconds). The GRU model was a close second in accuracy but was significantly slower to train. The SARIMAX model served as a decent but less accurate statistical benchmark. The results demonstrate that machine learning models, particularly XGBoost, are highly effective for short-term solar forecasting.

**Conclusion:** XGBoost demonstrated superior performance across all accuracy metrics while being significantly faster to train, making it the recommended model for operational use.

##  Methodology

 1. Data & Feature Engineering
- Target: Global Horizontal Irradiance (GHI) at time `t+2` hour.
- (T+H), H can be replaced with (1,2...12) for multi-horizons forecasting. In this project as I have predicted 1 hour ahead, its 't+2', my data are at 30-minute intervals, a 1-hour-ahead forecast is two steps ahead (2 × 30 min = 60 min).
- Features Engineered:
  - Temporal: Hour-of-day (sine/cosine encoding), solar zenith angle.
  - Historical GHI: Lagged values (lag1, lag2, lag4, lag8, lag12).
  - Rolling Statistics: Rolling means and standard deviations (windows: 2, 4, 8, 12).
  - Meteorological: Air temperature, relative humidity, pressure, wind speed (lagged).
- Train/Test Split: Chronological split with a held-out test set to preserve temporal order.

2. Models Implemented

#XGBoost (eXtreme Gradient Boosting)
- A powerful gradient-boosted tree model for tabular data.
- Optimization: Hyperparameters tuned using Bayesian Optimization (Optuna, 100 trials).
- Key Tuned Parameters: `learning_rate=0.045`, `max_depth=7`, regularized with `lambda=10`.
- Target Formulation: Predicted the residual (change from `t` to `t+H`), which significantly improved directional accuracy.

#GRU (Gated Recurrent Unit)
- A recurrent neural network designed for sequence modeling.
- Architecture: Two-layer GRU (96 → 48 units) with dropout (`~0.12`) for regularization.
- Input: A sequence of the last 48 time steps (24 hours of data).
- Target Formulation: Predicted the absolute GHI value at `t+H`.

# SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables)
- A classical statistical benchmark model for time series.
- Order: `(2, 0, 1)` with a seasonal component `(1, 0, 1, 48)` (48-step daily seasonality).
- Target Formulation: Predicted the residual to stabilize the series.
- Exogenous Vars: Top 10 features selected by correlation with the target.

#3. Validation Strategy
- Expanding Window Cross-Validation:** Used for robust hyperparameter tuning and model evaluation, ensuring no data leakage from the future.
- Day-Only Evaluation:** All models were trained and evaluated only on daylight hours to avoid learning the trivial pattern of zero irradiance at night.

# How to Use / Reproduce

# Installation
1. Clone this repository.
2. Install the required dependencies:
```bash
pip install -r requirements.txt

Environment versions:
  pandas        2.2.2
  numpy         2.0.2
  statsmodels   0.14.5
  scikit-learn  1.6.1
  xgboost       3.0.4
  optuna        4.5.0
  tensorflow    2.19.0
  pvlib         0.13.0
