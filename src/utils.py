import tempfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import requests
import json
import os
from pandas import DateOffset

CONFIG_PATH = "data/forecast_config.json"


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_moex_stock_list():
    """
    Fetches a list of MOEX stocks with current prices and percentage changes.

    :return: pd.DataFrame: A DataFrame containing columns like SECID, SHORTNAME, PRICE, CHANGE_PCT, and CHANGE.
    """
    url = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json"
    params = {
        "iss.meta": "off",
        "securities.columns": "SECID,SHORTNAME,SECNAME,SECTORID",
        "marketdata.columns": "SECID,LAST,LASTTOPREVPRICE"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Extract securities data
        securities_columns = data['securities']['columns']
        securities_data = data['securities']['data']
        securities_df = pd.DataFrame(securities_data, columns=securities_columns)

        # Extract marketdata
        marketdata_columns = data['marketdata']['columns']
        marketdata_data = data['marketdata']['data']
        marketdata_df = pd.DataFrame(marketdata_data, columns=marketdata_columns)

        # Merge the dataframes
        df = pd.merge(securities_df, marketdata_df, on='SECID', how='left')

        # Clean and format the data
        df = df.rename(columns={
            'SECTORID': 'SECTOR',
            'LAST': 'PRICE',
            'LASTTOPREVPRICE': 'CHANGE_PCT'
        })

        # Filter out invalid entries
        df = df[df['SECID'].notna()]
        df = df[~df['SECID'].str.contains('TOTAL', case=False, na=False)]

        # Calculate absolute change
        df['CHANGE'] = df['CHANGE_PCT'].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A"
        )

        return df

    except Exception as e:
        st.error(f"Error loading stock data: {str(e)}")
        return pd.DataFrame()


def calculate_volatility(prices):
    """
    Calculates annualized volatility of a time series of prices based on log returns.

    :param prices: pd.Series of price values (e.g., closing prices)
    :return: Annualized volatility (float), calculated using log returns
    """
    if len(prices) < 2:
        return 0
    returns = np.log(prices / prices.shift(1))
    return returns.std() * np.sqrt(252)


def classify_volatility(volatility):
    """
    Classifies the volatility into categories: Low, Medium, High, Very High.

    :param volatility: float value of annualized volatility
    :return: str: 'Low', 'Medium', 'High', or 'Very High' based on thresholds
    """
    if volatility < 0.2:
        return "Low"
    elif 0.2 <= volatility < 0.4:
        return "Medium"
    elif 0.4 <= volatility < 0.6:
        return "High"
    else:
        return "Very High"


def color_volatility(val):
    """
    Returns a CSS style string to color a cell based on volatility classification.

    :param val: str volatility level ('Low', 'Medium', 'High', 'Very High')
    :return: str: CSS style string for coloring cell background and font
    """
    color = {
        'Low': '#4CAF50',
        'Medium': '#FFC107',
        'High': '#FF9800',
        'Very High': '#F44336'
    }.get(val, 'black')
    return f'background-color: {color}; color: white; font-weight: bold;'


def read_forecast_config():
    """
    Reads forecast configuration from the JSON file if it exists.

    :return: dict with forecast configuration loaded from CONFIG_PATH
    """
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def write_forecast_config(config):
    """
    Saves the given configuration dictionary to a JSON file.

    :param config: dict to be saved as forecast configuration to CONFIG_PATH
    :return: None
    """
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


def candlestickchart(stock_data, ticker):
    """
    Creates and renders a Plotly candlestick chart in Streamlit with OHLC data.
    Also generates a trimmed 1-month version for export to Telegram.

    :param stock_data: pd.DataFrame with OHLC data and datetime index
    :param ticker: str name of the stock
    :return: go.Figure (Plotly chart) for export or None if columns missing
    """
    if all(col in stock_data.columns for col in ["OPEN", "HIGH", "LOW", "CLOSE"]):
        candle_fig = go.Figure(data=[
            go.Candlestick(
                x=stock_data.index.to_list(),
                open=stock_data['OPEN'],
                high=stock_data['HIGH'],
                low=stock_data['LOW'],
                close=stock_data['CLOSE'],
                name=f"{ticker} Candlestick",
                increasing_line_color='#2CA02C',
                decreasing_line_color='#FF0000',
                increasing_fillcolor='rgba(44, 160, 44, 0.7)',
                decreasing_fillcolor='rgba(255, 0, 0, 0.7)'
            )
        ])

        start_date = stock_data.index[0] - DateOffset(days=3)
        end_date = stock_data.index[-1] + DateOffset(days=3)

        candle_fig.update_layout(
            title=f"{ticker} Candlestick Chart",
            xaxis_title="Date",
            yaxis_title="Price (RUB)",
            xaxis_rangeslider_visible=True,
            template='plotly_dark',
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(step="all")
                    ]),
                    bgcolor="rgba(50, 50, 50, 0.8)",
                    activecolor="rgba(80, 80, 80, 1)",
                    font=dict(color="white"),
                ),
                rangeslider=dict(visible=True),
                type="date",
                range=[
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                ],
            ),
            yaxis=dict(fixedrange=False),
            dragmode="pan",
            hovermode="x unified",
            margin=dict(l=40, r=40, t=60, b=40),
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)'
        )

        st.plotly_chart(candle_fig, use_container_width=True)

        last_date = stock_data.index[-1]
        start_date_1m = last_date - pd.DateOffset(months=1)
        data_1m = stock_data.loc[start_date_1m:last_date]

        fig_1m = go.Figure(data=[go.Candlestick(
            x=data_1m.index.to_list(),
            open=data_1m['OPEN'],
            high=data_1m['HIGH'],
            low=data_1m['LOW'],
            close=data_1m['CLOSE'],
            name=f"{ticker} Candlestick",
            increasing_line_color='#2CA02C',
            decreasing_line_color='#FF0000',
            increasing_fillcolor='rgba(44, 160, 44, 0.7)',
            decreasing_fillcolor='rgba(255, 0, 0, 0.7)'
        )])

        fig_1m.update_layout(
            title=f"{ticker} — Candlestick (Last 1 Month)",
            template="plotly_dark",
            xaxis=dict(
                title="Date",
                type="date",
                tickformat="%b %d, %Y",
                hoverformat="%b %d, %Y",
                range=[
                    (data_1m.index[0] - pd.DateOffset(days=2)).strftime('%Y-%m-%d'),
                    (data_1m.index[-1] + pd.DateOffset(days=2)).strftime('%Y-%m-%d'),
                ]
            ),
            yaxis_title="Price",
            height=400,
            plot_bgcolor='rgb(17,17,17)',
            paper_bgcolor='rgb(17,17,17)',
            font=dict(color='white')
        )

        export_fig = fig_1m
        export_fig.update_layout(
            plot_bgcolor='rgb(17,17,17)',
            paper_bgcolor='rgb(17,17,17)',
            font=dict(color='white')
        )

        return export_fig
    else:
        st.warning("Candlestick chart requires OPEN, HIGH, LOW, and CLOSE columns.")
        return None


def save_plotly_figure(fig, filename_prefix="plot"):
    """
    Saves a Plotly figure as a PNG image to a temporary file.

    :param fig: Plotly figure to be saved
    :param filename_prefix: str prefix for temporary PNG filename
    :return: str path to saved image file
    """
    path = os.path.join(tempfile.gettempdir(), f"{filename_prefix}.png")
    fig.write_image(path, format="png", scale=2)
    return path


def create_price_chart(data, title, price_col='CLOSE', color='#1f77b4'):
    """
    Builds a line chart of price data with optional SMA overlays.

    :param data: pd.DataFrame with time-series price data
    :param title: str title of the chart
    :param price_col: str column name to be used for plotting price
    :param color: str line color for the price trace
    :return: go.Figure: Plotly line chart with optional moving averages
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data.index.to_list(),
        y=data[price_col],
        name='Price',
        line=dict(width=2, color=color)
    ))

    if 'SMA_50' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index.to_list(),
            y=data['SMA_50'],
            name='SMA 50',
            line=dict(width=1.5, dash='dash', color='#ff7f0e')
        ))

    if 'SMA_200' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index.to_list(),
            y=data['SMA_200'],
            name='SMA 200',
            line=dict(width=1.5, dash='dot', color='#2ca02c')
        ))

    start_date = data.index[0] - DateOffset(days=3)
    end_date = data.index[-1] + DateOffset(days=3)

    fig.update_layout(
        template='plotly_dark',
        title=title,
        title_font=dict(size=20, color='white'),
        xaxis=dict(
            title='Date',
            type='date',
            tickformat='%b %Y',
            hoverformat='%b %d, %Y',
            title_font=dict(color='white'),
            tickfont=dict(color='white'),
            range=[
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            ]
        ),
        yaxis=dict(
            title='Price',
            title_font=dict(color='white'),
            tickfont=dict(color='white')
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color='white')
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        height=450
    )

    return fig


def add_moving_averages(data, price_col='CLOSE'):
    """
    Adds 50-day and 200-day simple moving averages to the data if sufficient length.

    :param data: pd.DataFrame containing the price column
    :param price_col: str name of the price column (default 'CLOSE')
    :return: pd.DataFrame with optional SMA_50 and SMA_200 columns added
    """
    if len(data) >= 50:
        data['SMA_50'] = data[price_col].rolling(50).mean()
    if len(data) >= 200:
        data['SMA_200'] = data[price_col].rolling(200).mean()
    return data
