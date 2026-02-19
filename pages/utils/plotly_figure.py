import plotly.graph_objects as go
import pandas as pd
from dateutil.relativedelta import relativedelta
import datetime
import pandas_ta as pta


# ==============================
# TABLE
# ==============================

def plotly_table(dataframe: pd.DataFrame):

    dataframe = dataframe.copy()

    header_color = '#0078ff'
    row_even_color = '#f8fafd'
    row_odd_color = '#e1efff'

    # Alternate row colors
    row_colors = [
        row_odd_color if i % 2 == 0 else row_even_color
        for i in range(len(dataframe))
    ]

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["<b></b>"] + [f"<b>{str(col)[:10]}</b>" for col in dataframe.columns],
            line_color=header_color,
            fill_color=header_color,
            align='center',
            font=dict(color='white', size=15),
            height=35
        ),
        cells=dict(
            values=[
                [f"<b>{str(i)}</b>" for i in dataframe.index]
            ] + [dataframe[col] for col in dataframe.columns],
            fill_color=[row_colors] * (len(dataframe.columns) + 1),
            align='left',
            line_color='white',
            font=dict(color='black', size=14)
        )
    )])

    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    return fig


# ==============================
# DATA FILTER
# ==============================

def filter_data(dataframe: pd.DataFrame, num_period: str):

    df = dataframe.copy()
    df = df.sort_index()

    last_date = df.index[-1]

    if num_period == '1mo':
        start_date = last_date - relativedelta(months=1)

    elif num_period == '5d':
        start_date = last_date - relativedelta(days=5)

    elif num_period == '6mo':
        start_date = last_date - relativedelta(months=6)

    elif num_period == '1y':
        start_date = last_date - relativedelta(years=1)

    elif num_period == '5y':
        start_date = last_date - relativedelta(years=5)

    elif num_period == 'ytd':
        start_date = datetime.datetime(last_date.year, 1, 1)

    else:
        return df.reset_index()

    df = df[df.index >= start_date]
    return df.reset_index()


# ==============================
# CLOSE CHART
# ==============================

def close_chart(dataframe: pd.DataFrame, num_period: str = None):

    df = dataframe.copy()

    if num_period:
        df = filter_data(df, num_period)
    else:
        df = df.reset_index()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Open'],
        mode='lines', name='Open',
        line=dict(width=2, color='#5ab7ff')
    ))

    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Close'],
        mode='lines', name='Close',
        line=dict(width=2, color='black')
    ))

    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['High'],
        mode='lines', name='High',
        line=dict(width=2, color='#0078ff')
    ))

    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Low'],
        mode='lines', name='Low',
        line=dict(width=2, color='red')
    ))

    fig.update_xaxes(rangeslider_visible=True)

    fig.update_layout(
        height=500,
        margin=dict(l=0, r=20, t=20, b=0),
        plot_bgcolor='white',
        paper_bgcolor='#e1efff',
        legend=dict(yanchor="top", xanchor="right")
    )

    return fig


# ==============================
# CANDLESTICK
# ==============================

def candlestick_chart(dataframe: pd.DataFrame, num_period: str):

    df = filter_data(dataframe, num_period)

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    ))

    fig.update_layout(
        showlegend=False,
        height=500,
        margin=dict(l=0, r=20, t=20, b=0),
        plot_bgcolor='white'
    )

    return fig


# ==============================
# RSI
# ==============================

def rsi_chart(dataframe: pd.DataFrame, num_period: str):

    df = dataframe.copy()
    df['RSI'] = pta.rsi(df['Close'])

    df = filter_data(df, num_period)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['RSI'],
        name='RSI',
        line=dict(width=2, color='orange')
    ))

    fig.add_trace(go.Scatter(
        x=df['Date'], y=[70] * len(df),
        name='Overbought',
        line=dict(width=2, color='red', dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=df['Date'], y=[30] * len(df),
        name='Oversold',
        fill='tonexty',
        line=dict(width=2, color='#79da84', dash='dash')
    ))

    fig.update_layout(
        yaxis_range=[0, 100],
        height=200,
        plot_bgcolor='white',
        paper_bgcolor='#e1efff',
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


# ==============================
# MOVING AVERAGE
# ==============================

def moving_average_chart(dataframe: pd.DataFrame, num_period: str):

    df = dataframe.copy()
    df['SMA_50'] = pta.sma(df['Close'], length=50)

    df = filter_data(df, num_period)

    fig = go.Figure()

    for col, color in [
        ('Open', '#5ab7ff'),
        ('Close', 'black'),
        ('High', '#0078ff'),
        ('Low', 'red'),
        ('SMA_50', 'purple')
    ]:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df[col],
            mode='lines',
            name=col,
            line=dict(width=2, color=color)
        ))

    fig.update_xaxes(rangeslider_visible=True)

    fig.update_layout(
        height=500,
        margin=dict(l=0, r=20, t=20, b=0),
        plot_bgcolor='white',
        paper_bgcolor='#e1efff',
        legend=dict(yanchor="top", xanchor="right")
    )

    return fig


# ==============================
# MACD
# ==============================

def macd_chart(dataframe: pd.DataFrame, num_period: str):

    df = dataframe.copy()

    macd_data = pta.macd(df['Close'])

    df['MACD'] = macd_data.iloc[:, 0]
    df['MACD_Signal'] = macd_data.iloc[:, 1]
    df['MACD_Hist'] = macd_data.iloc[:, 2]

    df = filter_data(df, num_period)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['MACD'],
        name='MACD',
        line=dict(width=2, color='orange')
    ))

    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['MACD_Signal'],
        name='Signal',
        line=dict(width=2, color='red', dash='dash')
    ))

    colors = ['red' if val < 0 else 'green' for val in df['MACD_Hist']]

    fig.add_bar(
        x=df['Date'],
        y=df['MACD_Hist'],
        marker_color=colors,
        name='Histogram'
    )

    fig.update_layout(
        height=200,
        plot_bgcolor='white',
        paper_bgcolor='#e1efff',
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig
def Moving_average_forecast(forecast):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=forecast.index[:-30],
            y=forecast['Close'].iloc[:-30],
            mode='lines',
            name='Close Price',
            line=dict(width=2, color='black')
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast.index[-31:],
            y=forecast['Close'].iloc[-31:],
            mode='lines',
            name='Future Close Price',
            line=dict(width=2, color='red')
        )
    )

    fig.update_xaxes(rangeslider_visible=True)

    fig.update_layout(
        height=500,
        margin=dict(l=0, r=20, t=20, b=0),
        plot_bgcolor='white',
        paper_bgcolor='#e1efff',
        legend=dict(
            yanchor="top",
            xanchor="right"
        )
    )

    return fig
