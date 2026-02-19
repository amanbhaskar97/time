import streamlit as st
import pandas as pd
import yfinance as yf
import datetime

from pages.utils.plotly_figure import (
    plotly_table,
    candlestick_chart,
    close_chart,
    rsi_chart,
    macd_chart,
    moving_average_chart
)

# -------------------- PAGE CONFIG --------------------

st.set_page_config(
    page_title="Stock Analysis",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Stock Analysis")

# -------------------- INPUT SECTION --------------------

col1, col2, col3 = st.columns(3)

today = datetime.date.today()

with col1:
    ticker_symbol = st.text_input("Stock Ticker", "TSLA").upper()

with col2:
    start_date = st.date_input(
        "Choose Start Date",
        datetime.date(today.year - 1, today.month, today.day)
    )

with col3:
    end_date = st.date_input(
        "Choose End Date",
        today
    )

# -------------------- LOAD DATA (CACHED SAFE VERSION) --------------------

@st.cache_data
def load_stock_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)

    ticker = yf.Ticker(symbol)
    try:
        info = ticker.get_info()
    except:
        info = {}

    return data, info

data, info = load_stock_data(ticker_symbol, start_date, end_date)

if data.empty:
    st.error("Invalid ticker symbol or no historical data found.")
    st.stop()

st.subheader(ticker_symbol)

# -------------------- COMPANY INFO --------------------

if info:
    st.write(info.get('longBusinessSummary', 'No summary available.'))
    st.write("**Sector:**", info.get('sector', 'N/A'))
    st.write("**Full Time Employees:**", info.get('fullTimeEmployees', 'N/A'))
    st.write("**Website:**", info.get('website', 'N/A'))
else:
    st.warning("Company information could not be loaded.")

# -------------------- FUNDAMENTALS TABLES --------------------

if info:
    col1, col2 = st.columns(2)

    with col1:
        df1 = pd.DataFrame(index=['Market Cap', 'Beta', 'EPS', 'PE Ratio'])
        df1[''] = [
            info.get("marketCap"),
            info.get("beta"),
            info.get("trailingEps"),
            info.get("trailingPE")
        ]
        st.plotly_chart(plotly_table(df1), use_container_width=True)

    with col2:
        df2 = pd.DataFrame(index=[
            'Quick Ratio',
            'Revenue per Share',
            'Profit Margins',
            'Debt To Equity',
            'Return on Equity'
        ])

        df2[''] = [
            info.get("quickRatio"),
            info.get("revenuePerShare"),
            info.get("profitMargins"),
            info.get("debtToEquity"),
            info.get("returnOnEquity")
        ]

        st.plotly_chart(plotly_table(df2), use_container_width=True)

# -------------------- METRICS --------------------

col1, col2, col3 = st.columns(3)

if len(data) > 1:
    last_close = float(data['Close'].iloc[-1])
    prev_close = float(data['Close'].iloc[-2])
    daily_change = last_close - prev_close
else:
    last_close = float(data['Close'].iloc[-1])
    daily_change = 0.0

col1.metric(
    "Daily Close",
    round(last_close, 2),
    round(daily_change, 2)
)

# -------------------- LAST 10 DAYS TABLE --------------------

last_10_df = data.tail(10).sort_index(ascending=False).round(3)

st.write("### Historical Data (Last 10 Days)")
st.plotly_chart(plotly_table(last_10_df), use_container_width=True)

# -------------------- PERIOD BUTTONS --------------------

col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

num_period = ''

with col1:
    if st.button('5D'):
        num_period = '5d'
with col2:
    if st.button('1M'):
        num_period = '1mo'
with col3:
    if st.button('6M'):
        num_period = '6mo'
with col4:
    if st.button('YTD'):
        num_period = 'ytd'
with col5:
    if st.button('1Y'):
        num_period = '1y'
with col6:
    if st.button('5Y'):
        num_period = '5y'
with col7:
    if st.button('MAX'):
        num_period = 'max'

# -------------------- CHART OPTIONS --------------------

col1, col2, col3 = st.columns([1, 1, 4])

with col1:
    chart_type = st.selectbox("Chart Type", ('Candle', 'Line'))

with col2:
    if chart_type == 'Candle':
        indicators = st.selectbox("Indicator", ('RSI', 'MACD'))
    else:
        indicators = st.selectbox("Indicator", ('RSI', 'Moving Average', 'MACD'))

# -------------------- LOAD FULL HISTORY --------------------

ticker_obj = yf.Ticker(ticker_symbol)
full_data = ticker_obj.history(period='max')

if full_data.empty:
    st.warning("No chart data available.")
    st.stop()

selected_period = '1y' if num_period == '' else num_period

# -------------------- CHART RENDER --------------------

if chart_type == 'Candle':

    st.plotly_chart(
        candlestick_chart(full_data, selected_period),
        use_container_width=True
    )

    if indicators == 'RSI':
        st.plotly_chart(
            rsi_chart(full_data, selected_period),
            use_container_width=True
        )

    if indicators == 'MACD':
        st.plotly_chart(
            macd_chart(full_data, selected_period),
            use_container_width=True
        )

if chart_type == 'Line':

    if indicators == 'Moving Average':
        st.plotly_chart(
            moving_average_chart(full_data, selected_period),
            use_container_width=True
        )

    else:
        st.plotly_chart(
            close_chart(full_data, selected_period),
            use_container_width=True
        )

        if indicators == 'RSI':
            st.plotly_chart(
                rsi_chart(full_data, selected_period),
                use_container_width=True
            )

        if indicators == 'MACD':
            st.plotly_chart(
                macd_chart(full_data, selected_period),
                use_container_width=True
            )
