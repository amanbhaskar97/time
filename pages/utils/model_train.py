import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")


# ==============================
# DATA
# ==============================

def get_data(ticker: str) -> pd.DataFrame:
    """Download OHLCV data and return the Close series."""
    stock_data = yf.download(ticker, start="2022-01-01", auto_adjust=True, progress=False)
    return stock_data[["Close"]]


# ==============================
# STATIONARITY
# ==============================

def stationary_check(close_price: pd.Series) -> float:
    result = adfuller(close_price.dropna())
    return round(result[1], 4)


def get_differencing_order(close_price: pd.Series) -> int:
    """Return the minimum differencing order d that makes the series stationary."""
    series = close_price.copy().dropna()
    d = 0
    for _ in range(3):  # cap at 3 differences
        if stationary_check(series) <= 0.05:
            break
        series = series.diff().dropna()
        d += 1
    return d


# ==============================
# SMOOTHING
# ==============================

def get_rolling_mean(close_price: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Smooth the Close column with a rolling mean to reduce noise."""
    smoothed = close_price.rolling(window=window, min_periods=1).mean().dropna()
    return smoothed


# ==============================
# SCALING  (MinMax keeps values > 0, friendlier for ARIMA)
# ==============================

def scaling(close_price: pd.DataFrame):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(np.array(close_price).reshape(-1, 1))
    return scaled, scaler


def inverse_scaling(scaler, scaled_data):
    return scaler.inverse_transform(np.array(scaled_data).reshape(-1, 1))


# ==============================
# AUTO ORDER SELECTION  (AIC grid search over small p/q grid)
# ==============================

def _best_arima_order(data: np.ndarray, d: int):
    """
    Grid-search over p in {5,10,15,20} and q in {0,5,10} and return the
    (p, d, q) tuple with the lowest AIC.  Falls back to (20, d, 10) on error.
    """
    best_aic = np.inf
    best_order = (20, d, 10)

    p_candidates = [5, 10, 20]
    q_candidates = [0, 5, 10]

    for p in p_candidates:
        for q in q_candidates:
            try:
                m = ARIMA(data, order=(p, d, q))
                res = m.fit()
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order = (p, d, q)
            except Exception:
                continue

    return best_order


# ==============================
# MODEL FIT & FORECAST
# ==============================

def fit_model(data, differencing_order: int, steps: int = 30):
    """
    Fit an ARIMA model with auto-selected order and return forecast values.
    Also returns a (lower, upper) 95 % confidence interval tuple.
    """
    flat = np.array(data).flatten()
    order = _best_arima_order(flat, differencing_order)

    model     = ARIMA(flat, order=order)
    model_fit = model.fit()

    forecast_obj = model_fit.get_forecast(steps=steps)
    predictions  = forecast_obj.predicted_mean
    conf_int     = forecast_obj.conf_int(alpha=0.05)   # 95 % CI

    return predictions, conf_int, order


def evaluate_model(original_price, differencing_order: int) -> float:
    """Walk-forward RMSE on the last 30 scaled observations."""
    train, test = original_price[:-30], original_price[-30:]
    preds, _, _ = fit_model(train, differencing_order, steps=30)
    rmse = float(np.sqrt(mean_squared_error(test, preds)))
    return round(rmse, 4)


# ==============================
# FORECAST DATAFRAME
# ==============================

def get_forecast(original_price, differencing_order: int) -> tuple[pd.DataFrame, pd.DataFrame, tuple]:
    """
    Returns:
        forecast_df  – DataFrame with columns ['Close'] indexed by future dates
        conf_df      – DataFrame with columns ['Lower', 'Upper'] for the CI
        order        – the chosen ARIMA (p,d,q) order
    """
    steps = 30
    predictions, conf_int, order = fit_model(original_price, differencing_order, steps=steps)

    # Build daily date index starting tomorrow
    start_date    = datetime.now() + timedelta(days=1)
    forecast_index = pd.date_range(start=start_date, periods=steps, freq="B")   # business days

    # Pad or trim if freq="B" gives a different length
    forecast_index = forecast_index[:len(predictions)]

    forecast_df = pd.DataFrame({"Close": predictions}, index=forecast_index)

    conf_df = pd.DataFrame(
        {"Lower": conf_int[:, 0], "Upper": conf_int[:, 1]},
        index=forecast_index,
    )

    return forecast_df, conf_df, order