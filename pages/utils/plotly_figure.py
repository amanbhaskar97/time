import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from dateutil.relativedelta import relativedelta
import datetime
import pandas_ta as pta

# ==============================
# DESIGN TOKENS
# ==============================

PALETTE = {
    "bg_dark":     "#0d1117",
    "bg_card":     "#161b22",
    "bg_card2":    "#1c2330",
    "border":      "#30363d",
    "accent_blue": "#58a6ff",
    "accent_cyan": "#39d0e0",
    "accent_green":"#3fb950",
    "accent_red":  "#f85149",
    "accent_gold": "#e3b341",
    "accent_purple":"#bc8cff",
    "text_primary":"#e6edf3",
    "text_muted":  "#8b949e",
}

LEGEND_STYLE = dict(
    bgcolor=PALETTE["bg_card2"],
    bordercolor=PALETTE["border"],
    borderwidth=1,
    font=dict(size=11, color=PALETTE["text_primary"]),
)

COMMON_LAYOUT = dict(
    plot_bgcolor=PALETTE["bg_dark"],
    paper_bgcolor=PALETTE["bg_card"],
    font=dict(family="'JetBrains Mono', 'Courier New', monospace", color=PALETTE["text_primary"], size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(
        gridcolor=PALETTE["border"],
        linecolor=PALETTE["border"],
        tickfont=dict(color=PALETTE["text_muted"]),
        zerolinecolor=PALETTE["border"],
    ),
    yaxis=dict(
        gridcolor=PALETTE["border"],
        linecolor=PALETTE["border"],
        tickfont=dict(color=PALETTE["text_muted"]),
        zerolinecolor=PALETTE["border"],
    ),
)


# ==============================
# TABLE
# ==============================

def plotly_table(dataframe: pd.DataFrame):
    dataframe = dataframe.copy()

    header_color  = PALETTE["bg_card2"]
    row_odd_color = PALETTE["bg_dark"]
    row_even_color= PALETTE["bg_card"]

    row_colors = [
        row_odd_color if i % 2 == 0 else row_even_color
        for i in range(len(dataframe))
    ]

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["<b></b>"] + [f"<b>{str(col)[:12]}</b>" for col in dataframe.columns],
            line_color=PALETTE["accent_blue"],
            fill_color=header_color,
            align="center",
            font=dict(color=PALETTE["accent_blue"], size=13, family="'JetBrains Mono', monospace"),
            height=38,
        ),
        cells=dict(
            values=[
                [f"<b>{str(i)}</b>" for i in dataframe.index]
            ] + [dataframe[col] for col in dataframe.columns],
            fill_color=[row_colors] * (len(dataframe.columns) + 1),
            align="left",
            line_color=PALETTE["border"],
            font=dict(color=PALETTE["text_primary"], size=12, family="'JetBrains Mono', monospace"),
            height=30,
        ),
    )])

    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=PALETTE["bg_card"],
    )
    return fig


# ==============================
# DATA FILTER
# ==============================

def filter_data(dataframe: pd.DataFrame, num_period: str):
    df = dataframe.copy().sort_index()
    last_date = df.index[-1]

    period_map = {
        "5d":  lambda: last_date - relativedelta(days=5),
        "1mo": lambda: last_date - relativedelta(months=1),
        "6mo": lambda: last_date - relativedelta(months=6),
        "1y":  lambda: last_date - relativedelta(years=1),
        "5y":  lambda: last_date - relativedelta(years=5),
        "ytd": lambda: datetime.datetime(last_date.year, 1, 1),
    }

    if num_period in period_map:
        start_date = period_map[num_period]()
        df = df[df.index >= start_date]

    return df.reset_index()


# ==============================
# CLOSE CHART
# ==============================

def close_chart(dataframe: pd.DataFrame, num_period: str = None):
    df = dataframe.copy()
    df = filter_data(df, num_period) if num_period else df.reset_index()

    fig = go.Figure()

    traces = [
        ("High",  PALETTE["accent_blue"],   dict(width=1.5)),
        ("Low",   PALETTE["accent_red"],    dict(width=1.5)),
        ("Open",  PALETTE["text_muted"],    dict(width=1.5, dash="dot")),
        ("Close", PALETTE["accent_cyan"],   dict(width=2.5)),
    ]

    for col, color, line_style in traces:
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df[col],
            mode="lines", name=col,
            line=dict(color=color, **line_style),
            hovertemplate=f"<b>{col}</b>: %{{y:.2f}}<extra></extra>",
        ))

    # Shaded area under Close
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Close"],
        fill="tozeroy",
        fillcolor="rgba(57,208,224,0.05)",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig.update_xaxes(rangeslider_visible=True, rangeslider=dict(bgcolor=PALETTE["bg_card2"], thickness=0.06))
    fig.update_layout(height=520, **COMMON_LAYOUT)
    fig.update_layout(legend=dict(**LEGEND_STYLE, orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1))
    return fig


# ==============================
# CANDLESTICK
# ==============================

def candlestick_chart(dataframe: pd.DataFrame, num_period: str):
    df = filter_data(dataframe, num_period)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["Date"],
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing=dict(line=dict(color=PALETTE["accent_green"]), fillcolor=PALETTE["accent_green"]),
        decreasing=dict(line=dict(color=PALETTE["accent_red"]),   fillcolor=PALETTE["accent_red"]),
    ))

    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(showlegend=False, height=520, **COMMON_LAYOUT)
    return fig


# ==============================
# RSI
# ==============================

def rsi_chart(dataframe: pd.DataFrame, num_period: str):
    df = dataframe.copy()
    df["RSI"] = pta.rsi(df["Close"])
    df = filter_data(df, num_period)

    fig = go.Figure()

    # Overbought / oversold bands
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(248,81,73,0.08)",  line_width=0)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(63,185,80,0.08)", line_width=0)

    fig.add_trace(go.Scatter(
        x=df["Date"], y=[70]*len(df),
        name="Overbought", line=dict(width=1, color=PALETTE["accent_red"], dash="dash"),
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=[30]*len(df),
        name="Oversold", line=dict(width=1, color=PALETTE["accent_green"], dash="dash"),
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["RSI"],
        name="RSI", line=dict(width=2, color=PALETTE["accent_gold"]),
        hovertemplate="<b>RSI</b>: %{y:.1f}<extra></extra>",
    ))

    fig.update_layout(yaxis_range=[0, 100], height=220, **COMMON_LAYOUT)
    fig.update_layout(legend=dict(**LEGEND_STYLE, orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


# ==============================
# MOVING AVERAGE
# ==============================

def moving_average_chart(dataframe: pd.DataFrame, num_period: str):
    df = dataframe.copy()
    df["SMA_20"]  = pta.sma(df["Close"], length=20)
    df["SMA_50"]  = pta.sma(df["Close"], length=50)
    df["EMA_12"]  = pta.ema(df["Close"], length=12)
    df = filter_data(df, num_period)

    fig = go.Figure()

    base_traces = [
        ("Close", PALETTE["accent_cyan"],   dict(width=2.5)),
        ("High",  PALETTE["accent_blue"],   dict(width=1, dash="dot")),
        ("Low",   PALETTE["accent_red"],    dict(width=1, dash="dot")),
    ]
    ma_traces = [
        ("SMA_20",  PALETTE["accent_gold"],   dict(width=1.5)),
        ("SMA_50",  PALETTE["accent_purple"], dict(width=1.5)),
        ("EMA_12",  PALETTE["accent_green"],  dict(width=1.5, dash="dash")),
    ]

    for col, color, line_style in base_traces + ma_traces:
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df[col],
            mode="lines", name=col,
            line=dict(color=color, **line_style),
            hovertemplate=f"<b>{col}</b>: %{{y:.2f}}<extra></extra>",
        ))

    fig.update_xaxes(rangeslider_visible=True, rangeslider=dict(bgcolor=PALETTE["bg_card2"], thickness=0.06))
    fig.update_layout(height=520, **COMMON_LAYOUT)
    fig.update_layout(legend=dict(**LEGEND_STYLE, orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1))
    return fig


# ==============================
# MACD
# ==============================

def macd_chart(dataframe: pd.DataFrame, num_period: str):
    df = dataframe.copy()
    macd_data        = pta.macd(df["Close"])
    df["MACD"]       = macd_data.iloc[:, 0]
    df["MACD_Signal"]= macd_data.iloc[:, 1]
    df["MACD_Hist"]  = macd_data.iloc[:, 2]
    df = filter_data(df, num_period)

    colors = [PALETTE["accent_green"] if v >= 0 else PALETTE["accent_red"] for v in df["MACD_Hist"]]

    fig = go.Figure()
    fig.add_bar(
        x=df["Date"], y=df["MACD_Hist"],
        marker_color=colors,
        name="Histogram",
        opacity=0.7,
    )
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["MACD"],
        name="MACD", line=dict(width=2, color=PALETTE["accent_blue"]),
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["MACD_Signal"],
        name="Signal", line=dict(width=2, color=PALETTE["accent_gold"], dash="dash"),
    ))

    fig.update_layout(height=220, **COMMON_LAYOUT)
    fig.update_layout(legend=dict(**LEGEND_STYLE, orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


# ==============================
# FORECAST CHART
# ==============================

def Moving_average_forecast(forecast: pd.DataFrame, conf_df: pd.DataFrame = None):
    """
    forecast  – DataFrame with 'Close' column (full history + 30-day forecast).
                The last 30 rows are treated as the forecasted period.
    conf_df   – Optional DataFrame with 'Lower' and 'Upper' columns (ARIMA 95% CI),
                indexed to match the forecast rows. Falls back to ±2% if not provided.
    """
    fig = go.Figure()

    historical = forecast.iloc[:-30]
    future     = forecast.iloc[-31:]   # 1-row overlap for visual continuity

    # ── Historical line ────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=historical.index, y=historical["Close"],
        mode="lines",
        name="Historical Close",
        line=dict(width=2, color=PALETTE["accent_cyan"]),
        hovertemplate="<b>Historical</b>: %{y:.2f}<extra></extra>",
    ))

    # Subtle area fill under historical
    fig.add_trace(go.Scatter(
        x=historical.index, y=historical["Close"],
        fill="tozeroy",
        fillcolor="rgba(57,208,224,0.04)",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    # ── Confidence band (real ARIMA CI preferred, else ±2%) ───────────────────
    if conf_df is not None and not conf_df.empty:
        # Align conf_df to the future index (drop the overlap row if needed)
        ci = conf_df.reindex(future.index, method="nearest")
        upper = ci["Upper"]
        lower = ci["Lower"]
    else:
        upper = future["Close"] * 1.02
        lower = future["Close"] * 0.98

    fig.add_trace(go.Scatter(
        x=list(future.index) + list(future.index[::-1]),
        y=list(upper) + list(lower[::-1]),
        fill="toself",
        fillcolor="rgba(227,179,65,0.15)",
        line=dict(width=0),
        name="95% CI Band",
        hoverinfo="skip",
    ))

    # ── Upper / lower CI boundary lines ───────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=future.index, y=upper,
        mode="lines", name="Upper CI",
        line=dict(width=1, color=PALETTE["accent_gold"], dash="dot"),
        hovertemplate="<b>Upper CI</b>: %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=future.index, y=lower,
        mode="lines", name="Lower CI",
        line=dict(width=1, color=PALETTE["accent_gold"], dash="dot"),
        hovertemplate="<b>Lower CI</b>: %{y:.2f}<extra></extra>",
    ))

    # ── Forecast centre line ───────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=future.index, y=future["Close"],
        mode="lines",
        name="30-Day Forecast",
        line=dict(width=2.5, color=PALETTE["accent_gold"]),
        hovertemplate="<b>Forecast</b>: %{y:.2f}<extra></extra>",
    ))

    # Vertical divider at forecast start
    # add_vline with annotations is broken on date axes in many Plotly versions;
    # use add_shape + add_annotation instead.
    vline_x = str(future.index[0].date())
    fig.add_shape(
        type="line",
        x0=vline_x, x1=vline_x,
        y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color=PALETTE["accent_gold"], width=1, dash="dash"),
    )
    fig.add_annotation(
        x=vline_x,
        y=1,
        xref="x", yref="paper",
        text="Forecast →",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(color=PALETTE["accent_gold"], size=11),
    )

    fig.update_xaxes(rangeslider_visible=True, rangeslider=dict(bgcolor=PALETTE["bg_card2"], thickness=0.06))
    fig.update_layout(
        height=540,
        **COMMON_LAYOUT,
        title=dict(
            text="<b>30-Day Price Forecast</b>",
            font=dict(size=16, color=PALETTE["accent_blue"]),
            x=0.01,
        ),
    )
    fig.update_layout(legend=dict(**LEGEND_STYLE, orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig