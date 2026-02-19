import streamlit as st
import pandas as pd
import numpy as np
from pages.utils.model_train import (
    get_data,
    get_rolling_mean,
    get_differencing_order,
    scaling,
    evaluate_model,
    get_forecast,
    inverse_scaling,
)
from pages.utils.plotly_figure import plotly_table, Moving_average_forecast

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Forecast",
    page_icon="📈",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace !important;
    background-color: #0d1117;
    color: #e6edf3;
}
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px 20px;
}
[data-testid="metric-container"] label { color: #8b949e !important; }
[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #58a6ff !important;
    font-size: 1.6rem !important;
}
[data-testid="metric-container"] [data-testid="metric-delta"] { font-size: 0.85rem; }
h2, h3 { color: #58a6ff !important; letter-spacing: 0.03em; }
hr { border-color: #30363d; }
input { background: #161b22 !important; color: #e6edf3 !important; }
.stAlert { border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("## 📈 Stock Price Forecast")
st.markdown("<p style='color:#8b949e;margin-top:-12px'>ARIMA-based 30-day forward projection</p>",
            unsafe_allow_html=True)
st.markdown("---")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    ticker        = st.text_input("Ticker Symbol", "AAPL").upper().strip()
    smooth_window = st.slider("Smoothing Window (days)", 3, 21, 7)
    st.markdown("---")
    st.markdown("<p style='color:#8b949e;font-size:0.75rem'>Data sourced from Yahoo Finance.<br>"
                "Forecast is for informational purposes only.</p>", unsafe_allow_html=True)

# ── Data loading ───────────────────────────────────────────────────────────────
with st.spinner(f"Fetching data for **{ticker}** …"):
    try:
        raw_data      = get_data(ticker)
        rolling_price = get_rolling_mean(raw_data, window=smooth_window)
        scaled_data, scaler = scaling(rolling_price)
        d_order       = get_differencing_order(rolling_price.squeeze())
    except Exception as e:
        st.error(f"Could not load data for **{ticker}**: {e}")
        st.stop()

# ── Summary metrics row 1 ──────────────────────────────────────────────────────
latest_close = float(raw_data["Close"].iloc[-1])
prev_close   = float(raw_data["Close"].iloc[-2])
pct_change   = (latest_close - prev_close) / prev_close * 100
high_52w     = float(raw_data["Close"].tail(252).max())
low_52w      = float(raw_data["Close"].tail(252).min())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest Close",   f"${latest_close:,.2f}", f"{pct_change:+.2f}%")
col2.metric("52-Week High",   f"${high_52w:,.2f}")
col3.metric("52-Week Low",    f"${low_52w:,.2f}")
col4.metric("Differencing d", str(d_order),
            help="ARIMA integration order d chosen automatically via ADF test")

st.markdown("---")
st.subheader(f"30-Day Forecast — {ticker}")

# ── Model training ─────────────────────────────────────────────────────────────
with st.spinner("Training ARIMA model …"):
    try:
        rmse = evaluate_model(scaled_data, d_order)
        forecast_df, conf_df, arima_order = get_forecast(scaled_data, d_order)

        # Inverse-scale forecast and CI back to dollars
        forecast_df["Close"] = inverse_scaling(scaler, forecast_df["Close"]).flatten()
        conf_df["Lower"]     = inverse_scaling(scaler, conf_df["Lower"]).flatten()
        conf_df["Upper"]     = inverse_scaling(scaler, conf_df["Upper"]).flatten()

    except Exception as e:
        st.error(f"Model training failed: {e}")
        st.stop()

# ── Dollar RMSE ────────────────────────────────────────────────────────────────
price_range  = float(rolling_price["Close"].max() - rolling_price["Close"].min())
rmse_dollars = rmse * price_range

# ── Summary metrics row 2 ──────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Model RMSE",
    f"${rmse_dollars:,.2f}",
    f"scaled: {rmse:.4f}",
    help="Average prediction error in dollars. Scaled RMSE uses the 0-1 MinMax range.",
)
c2.metric("ARIMA Order", f"({arima_order[0]}, {arima_order[1]}, {arima_order[2]})")
c3.metric("Forecast End Price", f"${forecast_df['Close'].iloc[-1]:,.2f}")
forecast_return = (forecast_df["Close"].iloc[-1] - latest_close) / latest_close * 100
c4.metric(
    "Projected 30d Return",
    f"{forecast_return:+.2f}%",
    delta_color="normal" if forecast_return >= 0 else "inverse",
)

st.markdown("---")

# ── Forecast table ─────────────────────────────────────────────────────────────
with st.expander("📋 View Forecast Data Table", expanded=False):
    display_df = forecast_df.copy().round(3)
    display_df.index = display_df.index.strftime("%Y-%m-%d")
    display_df["Lower CI"] = conf_df["Lower"].round(3).values
    display_df["Upper CI"] = conf_df["Upper"].round(3).values
    fig_table = plotly_table(display_df)
    fig_table.update_layout(height=320)
    st.plotly_chart(fig_table, use_container_width=True)

# ── Forecast chart ─────────────────────────────────────────────────────────────
# KEY FIX: rolling_price is in scaled (0-1) space — inverse-transform it back
# to real dollar prices before concatenating with the already-unscaled forecast.
hist_prices = inverse_scaling(scaler, scaled_data).flatten()
hist = pd.DataFrame(
    {"Close": hist_prices},
    index=pd.to_datetime(rolling_price.index).tz_localize(None),
)

# Align forecast + CI index timezone
fcast = forecast_df[["Close"]].copy()
fcast.index = pd.to_datetime(fcast.index).tz_localize(None)

conf_aligned = conf_df.copy()
conf_aligned.index = pd.to_datetime(conf_aligned.index).tz_localize(None)

# Full history + 30-day forecast — no truncation, range slider lets user zoom
combined = pd.concat([hist, fcast])

st.plotly_chart(
    Moving_average_forecast(combined, conf_aligned),
    use_container_width=True,
)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='color:#8b949e;font-size:0.75rem;text-align:center'>"
    "⚠️ This forecast is generated by a statistical model and is not financial advice."
    "</p>",
    unsafe_allow_html=True,
)