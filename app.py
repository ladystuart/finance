"""
Stock Market Analysis Dashboard - Streamlit Application

This file implements a comprehensive stock market analysis dashboard with multiple tabs for different analytical functions:

Key Components:
1. Stock Analysis (Tab 1)
   - MOEX stock data visualization with interactive charts
   - Technical indicators (SMA/EMA)
   - Regression analysis for price prediction
   - LSTM neural network forecasting
   - News sentiment analysis

2. Precious Metals Analysis (Tab 2)
   - Central Bank of Russia metal price data
   - Gold, silver, platinum, and palladium tracking
   - Similar analytical tools as stock tab

3. Backtrader Strategy Tester (Tab 3)
   - Three trading strategy implementations:
     * SMA Crossover
     * RSI-based
     * MACD-based
   - Backtesting engine with performance metrics
   - Interactive visualization of results

4. Portfolio Management (Tab 4)
   - Portfolio composition tracking
   - Performance metrics
   - Asset allocation visualization

5. Theory Reference (Tab 5)
   - Educational content about:
     * Technical analysis
     * Indicator explanations
     * Trading strategies
     * Market concepts

Technical Architecture:
- Frontend: Streamlit framework
- Data Sources:
  * MOEX ISS API for stocks
  * CBR API for metals
  * News APIs for sentiment analysis
- Analytics:
  * scikit-learn for regression
  * TensorFlow/Keras for LSTM
  * Backtrader for strategy testing
- Visualization:
  * Plotly for interactive charts
  * Matplotlib for static plots

Key Features:
- Caching for performance optimization
- Session state management
- Telegram integration for report sharing
- Responsive layout design
- Comprehensive error handling

Configuration:
- settings.json for API keys and parameters
- portfolio.json for user portfolio definition
- user_state.json for user states (bot)

The application provides both real-time market data analysis and educational resources for traders and investors.
"""

import streamlit as st
import pandas as pd
import requests
from io import StringIO
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from src.analysis_regression import show_regression_analysis
from src.analysis_lstm import show_lstm_analysis
from src.analysis_news import show_news_analysis, find_english_name
from bot.telegram_bot import send_report_to_telegram
from src.utils import create_price_chart, add_moving_averages, candlestickchart, calculate_volatility, \
    classify_volatility, get_moex_stock_list
from bs4 import BeautifulSoup
from src.portfolio import show_portfolio_section
import json
import backtrader as bt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None


# ==================== STRATEGY CLASSES ====================
class SmaCrossStrategy(bt.Strategy):
    """
    Simple Moving Average (SMA) Crossover Trading Strategy.

    This strategy generates buy signals when the fast SMA crosses above the slow SMA,
    and sell signals (close position) when the fast SMA crosses below the slow SMA.

    Parameters:
    -----------
    fast : int
        Period for the fast SMA (default is 10).
    slow : int
        Period for the slow SMA (default is 50).
    """
    params = (
        ('fast', 10),
        ('slow', 50),
    )

    def __init__(self):
        """
        Initialize the SMA indicators with specified fast and slow periods.
        """
        self.sma_fast = bt.indicators.SMA(period=self.p.fast)
        self.sma_slow = bt.indicators.SMA(period=self.p.slow)

    def next(self):
        """
        Define the trading logic executed on each new data point.

        - If not in the market and fast SMA is above slow SMA, enter a buy order.
        - If in the market and fast SMA falls below slow SMA, close the position.

        :return: None
        """
        if not self.position:
            if self.sma_fast[0] > self.sma_slow[0]:
                self.buy()
        elif self.sma_fast[0] < self.sma_slow[0]:
            self.close()


class RsiStrategy(bt.Strategy):
    """
    Relative Strength Index (RSI) based Trading Strategy.

    This strategy generates buy signals when the RSI indicates oversold conditions,
    and sell signals (close position) when the RSI indicates overbought conditions.

    Parameters:
    -----------
    rsi_period : int
        The period for calculating the RSI indicator (default is 14).
    rsi_oversold : float
        The RSI value below which the asset is considered oversold and a buy signal is generated (default is 30).
    rsi_overbought : float
        The RSI value above which the asset is considered overbought and a sell (close) signal is generated (default is 70).
    """
    params = (
        ('rsi_period', 14),
        ('rsi_oversold', 30),
        ('rsi_overbought', 70),
    )

    def __init__(self):
        """
        Initialize the RSI indicator with the specified period.
        """
        self.rsi = bt.indicators.RSI(
            self.data.close,
            period=self.p.rsi_period
        )

    def next(self):
        """
        Define the trading logic executed on each new data point.

        - If not currently in position and RSI is below the oversold threshold, buy.
        - If currently in position and RSI is above the overbought threshold, close the position.

        :return: None
        """
        if not self.position:
            if self.rsi[0] < self.p.rsi_oversold:
                self.buy()
        elif self.rsi[0] > self.p.rsi_overbought:
            self.close()


class MacdStrategy(bt.Strategy):
    """
    MACD (Moving Average Convergence Divergence) Crossover Trading Strategy.

    This strategy generates buy signals when the MACD line crosses above the signal line,
    and sell signals (close position) when the MACD line crosses below the signal line.

    Parameters:
    -----------
    fast : int
        Period for the fast EMA (default is 12).
    slow : int
        Period for the slow EMA (default is 26).
    signal : int
        Period for the MACD signal line EMA (default is 9).
    """
    params = (
        ('fast', 12),
        ('slow', 26),
        ('signal', 9),
    )

    def __init__(self):
        """
        Initialize the MACD indicator with specified fast, slow, and signal periods.
        """
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.p.fast,
            period_me2=self.p.slow,
            period_signal=self.p.signal
        )

    def next(self):
        """
        Define the trading logic executed on each new data point.

        - If not currently in position and MACD line is above signal line, enter a buy order.
        - If currently in position and MACD line falls below signal line, close the position.

        :return: None
        """
        if not self.position:
            if self.macd.macd[0] > self.macd.signal[0]:
                self.buy()
        elif self.macd.macd[0] < self.macd.signal[0]:
            self.close()


def load_portfolio_tickers(path="config/portfolio.json"):
    """
    Load a list of stock tickers from a portfolio JSON file, excluding metals.

    :param path: Path to the portfolio JSON file.
    :return: List of ticker symbols (excluding metals like gold, silver, platinum, palladium).
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            portf_data = json.load(file)
            metals = {"gold", "silver", "platinum", "palladium"}
            tickers = [
                item["ticker"]
                for item in portf_data.get("portfolio", [])
                if item["ticker"].lower() not in metals
            ]
            return tickers
    except FileNotFoundError:
        st.warning(f"Portfolio file not found: {path}")
        return []
    except json.JSONDecodeError:
        st.warning(f"Invalid JSON format in portfolio file: {path}")
        return []
    except Exception as exc:
        st.warning(f"Could not load portfolio: {exc}")
        return []


with open("config/settings.json", "r", encoding="utf-8") as f:
    config = json.load(f)

st.set_page_config(layout="wide")

# Centered title using HTML
st.markdown("<h1 style='text-align: center; margin-bottom: 50px;'> Market Analysis Dashboard </h1>",
            unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Stocks", "🪙 Precious Metals", "🤖 Backtrader", "💼 Portfolio", "📚 Theory"])

# ==================== STOCKS TAB ====================
with tab1:
    st.header("Stock Market Analysis")

    portfolio_tickers = load_portfolio_tickers()

    if "stock_ticker" not in st.session_state:
        st.session_state["stock_ticker"] = portfolio_tickers[-1] if portfolio_tickers else "SBER"

    moex_stocks = get_moex_stock_list()

    if not moex_stocks.empty:
        moex_stocks['CHANGE'] = moex_stocks['CHANGE_PCT'].apply(
            lambda x: f"{x:.2f}%" if x else "N/A")

        moex_stocks = moex_stocks.sort_values('PRICE', ascending=False)

        with st.expander("📊 Available Stocks (MOEX TQBR Board)", expanded=True):
            st.info("""
            **Volatility Guide:**
            - 🟢 Low (<20%) - Stable (e.g., blue chips)
            - 🟡 Medium (20-40%) - Moderate risk
            - 🟠 High (40-60%) - High risk
            - 🔴 Very High (>60%) - Speculative
            """)

            default_tickers = ["SBER", "GAZP", "LKOH", "YDEX", "VTBR", "TATN", "ROSN", "MGNT"]

            if portfolio_tickers:
                unique_tickers = []
                seen = set()
                for ticker in portfolio_tickers:
                    if ticker not in seen:
                        seen.add(ticker)
                        unique_tickers.append(ticker)
                default_index = 0
            else:
                unique_tickers = default_tickers
                default_index = 0

            if st.session_state["stock_ticker"] in unique_tickers:
                default_index = unique_tickers.index(st.session_state["stock_ticker"])

            selected = st.selectbox(
                "Quick select popular stocks:",
                options=unique_tickers,
                index=default_index,
                key="popular_select"
            )

            if selected != st.session_state["stock_ticker"]:
                st.session_state["stock_ticker"] = selected

            st.dataframe(
                moex_stocks[['SECID', 'SHORTNAME', 'PRICE', 'CHANGE']],
                column_config={
                    "SECID": "Ticker",
                    "SHORTNAME": "Company",
                    "PRICE": st.column_config.NumberColumn("Price", format="%.2f ₽"),
                    "CHANGE": "Change %"
                },
                hide_index=True,
                use_container_width=True
            )

    ticker = st.text_input(
        "Enter stock ticker (or select above):",
        value=st.session_state["stock_ticker"],
        key="stock_ticker_input"
    )

    if ticker != st.session_state["stock_ticker"]:
        st.session_state["stock_ticker"] = ticker

    @st.cache_data
    def load_stock_data(ticker, start_date=None, end_date=None):
        """
        Load historical stock data from MOEX for a given ticker and date range.

        :param ticker: Stock ticker symbol.
        :param start_date: Start date as string "YYYY-MM-DD". Defaults to 1 year ago if None.
        :param end_date: End date as string "YYYY-MM-DD". Defaults to today if None.
        :return: Pandas DataFrame with stock data including volatility classification.
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        base_url = "https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/TQBR/securities"
        url = f"{base_url}/{ticker}.csv"

        all_data = []
        start = 0
        batch_size = 100

        while True:
            params = {
                "from": start_date,
                "till": end_date,
                "start": start
            }

            try:
                response = requests.get(url, params=params)
                response.raise_for_status()

                df = pd.read_csv(StringIO(response.text), sep=';', skiprows=2)

                if df.empty:
                    break

                # Filter out summary rows that might contain "TOTAL" or other non-date values
                df = df[pd.to_datetime(df['TRADEDATE'], errors='coerce').notna()]

                if df.empty:
                    break

                all_data.append(df)
                start += batch_size

                if len(df) < batch_size:
                    break

            except Exception as e:
                st.error(f"Error loading data: {e}")
                break

        if not all_data:
            st.warning(f"No data available for ticker {ticker}")
            return pd.DataFrame()

        full_data = pd.concat(all_data)

        # Volatility
        if not full_data.empty:
            full_data['TRADEDATE'] = pd.to_datetime(full_data['TRADEDATE'])
            full_data = full_data.sort_values('TRADEDATE')
            volatility = calculate_volatility(full_data['CLOSE'])
            full_data['VOLATILITY'] = classify_volatility(volatility)

        full_data['TRADEDATE'] = pd.to_datetime(full_data['TRADEDATE'])
        full_data.name = ticker
        st.session_state.stock_data = full_data
        return full_data

    stock_data = load_stock_data(st.session_state["stock_ticker"])
    last_close_price = stock_data['CLOSE'].iloc[-1]

    regression_data = load_stock_data(
        st.session_state["stock_ticker"],
        start_date=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    )

    lstm_data = load_stock_data(
        st.session_state["stock_ticker"],
        start_date=(datetime.now() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")
    )

    if not stock_data.empty:
        volatility_type = stock_data['VOLATILITY'].iloc[0]
        volatility_color = {
            'Low': 'green',
            'Medium': 'orange',
            'High': 'red',
            'Very High': 'darkred'
        }.get(volatility_type, 'gray')

        st.markdown(f"""
        <div style="
            border-left: 5px solid {volatility_color};
            padding: 10px 15px;
            background: linear-gradient(to right, rgba(40,40,40,0.85), rgba(20,20,20,0.95));
            border-radius: 8px;
            color: {volatility_color};
            font-family: sans-serif;
            font-size: 16px;
            font-weight: bold;
            width: fit-content;
        ">
            {volatility_type}
        </div>
        """, unsafe_allow_html=True)

        stock_data = stock_data[pd.to_datetime(stock_data['TRADEDATE'], format='%Y-%m-%d', errors='coerce').notnull()]
        stock_data['TRADEDATE'] = pd.to_datetime(stock_data['TRADEDATE'], format='%Y-%m-%d')
        stock_data.set_index('TRADEDATE', inplace=True)
        stock_data.sort_index(inplace=True)
        stock_data = add_moving_averages(stock_data)

        fig = create_price_chart(stock_data, f"{ticker} Stock Price and Moving Averages")

        chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Basic Chart", "Interactive Chart", "Candlestick Chart"])
        with chart_tab1:
            st.line_chart(stock_data['CLOSE'])
        with chart_tab2:
            st.plotly_chart(fig, use_container_width=True)
        with chart_tab3:
            fig_candlestick = candlestickchart(stock_data, ticker)

    stock_data = stock_data.dropna(subset=['CLOSE'])
    regression_data = regression_data.dropna(subset=['CLOSE'])
    lstm_data = lstm_data.dropna(subset=['CLOSE'])

    if not stock_data.empty and not regression_data.empty and not lstm_data.empty:
        col1, col2 = st.columns(2)

        with col1:
            regression_data['TRADEDATE'] = pd.to_datetime(regression_data['TRADEDATE'])
            regression_data.set_index('TRADEDATE', inplace=True)
            with st.expander("Regression Analysis", expanded=True):
                fig_forecast, forecast, r2, forecast_dates = show_regression_analysis(regression_data, "Stock", config)

        with col2:
            lstm_data['TRADEDATE'] = pd.to_datetime(lstm_data['TRADEDATE'])
            lstm_data.set_index('TRADEDATE', inplace=True)
            with st.expander("LSTM Forecast", expanded=True):
                fig_lstm, lstm_forecast, lstm_dates, fig_loss = show_lstm_analysis(lstm_data, "Stock", config)

        with st.expander("News Sentiment Analysis"):
            company_info = moex_stocks[moex_stocks['SECID'] == ticker].iloc[0]
            company_name = company_info['SHORTNAME']

            english_name = find_english_name(company_name)

            news_results = show_news_analysis(
                ticker,
                company_name=company_name,
                english_name=english_name
            )

        if st.button("📤 Send Full Report + Charts to Telegram", key="send_stock"):
            send_report_to_telegram(
                ticker, r2, forecast, lstm_forecast,
                fig_forecast, fig_lstm, fig, f"{ticker} Stock", fig_candlestick, news_results, last_close_price
            )

# ==================== PRECIOUS METALS TAB ====================
with tab2:
    st.header("Precious Metals Analysis")

    metals = {
        "Gold": {
            "code": "1",
            "ticker": "XAU",  # ISO 4217
            "ru_name": "Золото",
            "aliases": ["Gold", "XAUUSD", "GC=F", "GLD"],
            "color": "#FFD700"
        },
        "Silver": {
            "code": "2",
            "ticker": "XAG",
            "ru_name": "Серебро",
            "aliases": ["Silver", "XAGUSD", "SI=F", "SLV"],
            "color": "#C0C0C0"
        },
        "Platinum": {
            "code": "3",
            "ticker": "XPT",
            "ru_name": "Платина",
            "aliases": ["Platinum", "XPTUSD", "PL=F"],
            "color": "#E5E4E2"
        },
        "Palladium": {
            "code": "4",
            "ticker": "XPD",
            "ru_name": "Палладий",
            "aliases": ["Palladium", "XPDUSD", "PA=F"],
            "color": "#B4C424"
        }
    }

    metal_options = list(metals.keys())

    selected_metal = st.selectbox(
        "Select metal:",
        options=metal_options,
        index=0,
        key="metal_select",
        format_func=lambda x: f"{x} ({metals[x]['ticker']})"
    )

    metal_code = metals[selected_metal]["code"]
    st.session_state["metal_ticker"] = metal_code

    @st.cache_data
    def load_metal_data(metal_code, start_date=None, end_date=None):
        """
        Load historical metal price data from the Central Bank of Russia.

        :param metal_code: Metal code identifier as string.
        :param start_date: Start date as string "dd/mm/YYYY". Defaults to 1 year ago if None.
        :param end_date: End date as string "dd/mm/YYYY". Defaults to today if None.
        :return: Pandas DataFrame with metal price data including 'BUY', 'SELL', and average 'CLOSE' prices.
        """
        if end_date is None:
            end_date = datetime.now().strftime("%d/%m/%Y")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%d/%m/%Y")

        url = "https://www.cbr.ru/scripts/xml_metall.asp"

        params = {
            "date_req1": start_date,
            "date_req2": end_date,
            "VAL_NM_RQ": metal_code
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'xml')

            records = []
            for record in soup.find_all('Record'):
                if record['Code'] != metal_code:
                    continue

                date = record['Date']
                buy_tag = record.find('Buy')
                sell_tag = record.find('Sell')

                if buy_tag is None or sell_tag is None:
                    continue

                try:
                    buy = float(buy_tag.text.replace(',', '.'))
                    sell = float(sell_tag.text.replace(',', '.'))
                except ValueError:
                    continue

                records.append({
                    'TRADEDATE': date,
                    'BUY': buy,
                    'SELL': sell,
                    'CLOSE': (buy + sell) / 2
                })

            if not records:
                st.warning("No data found in the response")
                return pd.DataFrame()

            df = pd.DataFrame(records)
            return df

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()


    metal_data = load_metal_data(metal_code)

    last_close_price = metal_data['CLOSE'].iloc[-1]

    regression_metals_data = load_metal_data(
        st.session_state["metal_ticker"],
        start_date=(datetime.now() - timedelta(days=365)).strftime("%d/%m/%Y")
    )

    lstm_metals_data = load_metal_data(
        st.session_state["metal_ticker"],
        start_date=(datetime.now() - timedelta(days=365 * 3)).strftime("%d/%m/%Y")
    )

    if not metal_data.empty:
        try:
            metal_data['TRADEDATE'] = pd.to_datetime(metal_data['TRADEDATE'], format='%d.%m.%Y')
            metal_data.set_index('TRADEDATE', inplace=True)
            metal_data.sort_index(inplace=True)
            metal_data = add_moving_averages(metal_data)

            # Create and display chart
            fig = create_price_chart(
                metal_data,
                f"{selected_metal} Price (RUB per gram) and Moving Averages",
                color='#FFD700'
            )

            # Tabs for chart types
            chart_tab1, chart_tab2 = st.tabs(["Basic Chart", "Interactive Chart"])
            with chart_tab1:
                st.line_chart(metal_data['CLOSE'])
            with chart_tab2:
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.write("Debug data:", metal_data)

        st.info("""
        **Note:**
        - Data source: Central Bank of Russia
        - Prices are in Russian Rubles per gram
        - Data is updated daily by CBR
        - Buy/Sell prices represent CBR rates
        """)

        stock_data = metal_data.dropna(subset=['CLOSE'])
        regression_metals_data = regression_metals_data.dropna(subset=['CLOSE'])
        lstm_metals_data = lstm_metals_data.dropna(subset=['CLOSE'])

    else:
        st.warning(f"No data available for {selected_metal}. Please try another time range.")

    if not metal_data.empty and not regression_metals_data.empty and not lstm_metals_data.empty:

        col1, col2 = st.columns(2)

        with col1:
            regression_metals_data['TRADEDATE'] = pd.to_datetime(regression_metals_data['TRADEDATE'], format='%d.%m.%Y')
            regression_metals_data.set_index('TRADEDATE', inplace=True)
            with st.expander("Regression Analysis", expanded=True):
                fig_forecast, forecast, r2, forecast_dates = show_regression_analysis(regression_metals_data, "Metal",
                                                                                      config)

        with col2:
            with st.expander("LSTM Forecast", expanded=True):
                lstm_metals_data['TRADEDATE'] = pd.to_datetime(lstm_metals_data['TRADEDATE'], format='%d.%m.%Y')
                lstm_metals_data.set_index('TRADEDATE', inplace=True)
                fig_lstm, lstm_forecast, lstm_dates, fig_loss = show_lstm_analysis(lstm_metals_data, "Metal", config)

        with st.expander("News Sentiment Analysis"):
            metal_name = selected_metal.split()[0]

            metal_names_ru = {
                "Gold": "Золото",
                "Silver": "Серебро",
                "Platinum": "Платина",
                "Palladium": "Палладий"
            }

            metal_info = metals[selected_metal]

            news_results = show_news_analysis(
                query=metal_info["ticker"],
                company_name=metal_names_ru.get(metal_name),
                english_name=metal_name
            )

        if st.button("📤 Send Full Report + Charts to Telegram", key="send_metal"):
            send_report_to_telegram(
                selected_metal, r2, forecast, lstm_forecast,
                fig_forecast, fig_lstm, fig, selected_metal, None, news_results, last_close_price
            )

with tab3:
    st.header("🤖 Backtrader Strategy Tester")

    if 'stock_data' in st.session_state and st.session_state.stock_data is not None:
        st.subheader(f"Ticker: {st.session_state.stock_data.name}")
    else:
        st.warning("Please load stock data first in the '📈 Stocks' tab")
        st.stop()

    rsi_period = 14
    rsi_oversold = 30
    rsi_overbought = 70
    fast_period = 10
    slow_period = 50
    signal_period = 9

    strategy_type = st.selectbox(
        "Select Strategy",
        [
            ("SMA Crossover", "Buy when the fast SMA crosses the slow SMA from above"),
            ("RSI Strategy", "Buy when RSI < 30, sell when RSI > 70"),
            ("MACD Strategy", "Buy when the MACD crosses the signal line from below")
        ],
        format_func=lambda x: f"{x[0]} - {x[1]}",
        index=0
    )

    st.subheader("Strategy Parameters")
    col1, col2 = st.columns(2)

    with col1:
        start_cash = st.number_input("Initial Cash", 1000, 1000000, 10000)
        commission = st.number_input("Commission (%)", 0.0, 5.0, 0.1, step=0.05)

    if strategy_type[0] == "SMA Crossover":
        with col2:
            fast_period = st.slider("Fast SMA Period", 5, 50, 10)
            slow_period = st.slider("Slow SMA Period", 20, 200, 50)
    elif strategy_type[0] == "RSI Strategy":
        with col2:
            rsi_period = st.slider("RSI Period", 5, 30, 14)
            rsi_oversold = st.slider("Oversold Level", 5, 40, 30)
            rsi_overbought = st.slider("Overbought Level", 60, 95, 70)
    else:  # MACD
        with col2:
            fast_period = st.slider("MACD Fast", 5, 20, 12)
            slow_period = st.slider("MACD Slow", 20, 50, 26)
            signal_period = st.slider("Signal Period", 5, 20, 9)

    if st.button("🚀 Run Backtest"):
        if 'stock_data' not in st.session_state or st.session_state.stock_data is None:
            st.warning("Please load stock data first in Stocks tab")
        else:
            stock_data = st.session_state.stock_data.copy()

            if not pd.api.types.is_datetime64_any_dtype(stock_data.index):
                try:
                    stock_data.index = pd.to_datetime(stock_data.index)
                except Exception as e:
                    st.error(f"Error converting dates: {e}")
                    st.stop()

            stock_data = stock_data.reset_index()

            required_cols = ['TRADEDATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
            missing_cols = [col for col in required_cols if col not in stock_data.columns]
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                st.stop()

            stock_data = stock_data.dropna(subset=required_cols)

            cerebro = bt.Cerebro()
            cerebro.broker.setcash(start_cash)
            cerebro.broker.setcommission(commission=commission / 100)

            data = bt.feeds.PandasData(
                dataname=stock_data,
                datetime='TRADEDATE',
                open='OPEN',
                high='HIGH',
                low='LOW',
                close='CLOSE',
                volume='VOLUME',
                openinterest=None
            )
            cerebro.adddata(data)

            if strategy_type[0] == "SMA Crossover":
                cerebro.addstrategy(SmaCrossStrategy)
            elif strategy_type[0] == "RSI Strategy":
                cerebro.addstrategy(RsiStrategy)
            else:
                cerebro.addstrategy(MacdStrategy)

            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

            results = cerebro.run()
            strat = results[0]

            st.subheader("📊 Backtest Results")
            cols = st.columns(4)

            sharpe = strat.analyzers.sharpe.get_analysis()['sharperatio']
            cols[0].metric("Sharpe Ratio", f"{sharpe:.2f}",
                           "Good" if sharpe > 1 else "Poor")

            drawdown = strat.analyzers.drawdown.get_analysis()['max']['drawdown']
            cols[1].metric("Max Drawdown", f"{drawdown:.2f}%",
                           "Risky" if drawdown > 20 else "Safe")

            returns = strat.analyzers.returns.get_analysis()['rtot']
            cols[2].metric("Total Return", f"{returns:.2f}%",
                           "Good" if returns > 15 else "Poor")

            end_value = cerebro.broker.getvalue()
            cols[3].metric("Final Value", f"{end_value:,.2f} ₽")

            st.subheader("🔍 Trade Analysis")
            trade_stats = strat.analyzers.trades.get_analysis()

            if trade_stats.total.closed > 0:
                win_rate = trade_stats.won.total / trade_stats.total.closed * 100
                avg_win = trade_stats.won.pnl.average
                avg_loss = trade_stats.lost.pnl.average if trade_stats.lost.total > 0 else 0

                if trade_stats.lost.total > 0:
                    profit_factor = trade_stats.won.pnl.total / abs(trade_stats.lost.pnl.total)
                else:
                    profit_factor = float('inf')

                st.write(f"""
                - Total Trades: {trade_stats.total.closed}
                - Win Rate: {win_rate:.1f}%
                - Avg Win: {avg_win:.2f} ₽
                - Avg Loss: {avg_loss:.2f} ₽
                - Profit Factor: {"∞" if profit_factor == float('inf') else f"{profit_factor:.2f}"}
                """)
            else:
                st.warning("No trades were made during this period")

            st.subheader("📈 Strategy Visualization (Interactive)")

            try:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    vertical_spacing=0.05,
                                    row_heights=[0.7, 0.3])

                fig.add_trace(go.Candlestick(
                    x=stock_data.index,
                    open=stock_data['OPEN'],
                    high=stock_data['HIGH'],
                    low=stock_data['LOW'],
                    close=stock_data['CLOSE'],
                    name='Candles',
                    increasing_line_color='#2ECC71',
                    decreasing_line_color='#E74C3C'
                ), row=1, col=1)

                fig.add_trace(go.Bar(
                    x=stock_data.index,
                    y=stock_data['VOLUME'],
                    name='Volume',
                    marker_color='#3498DB',
                    opacity=0.5
                ), row=2, col=1)

                fig.update_layout(
                    template='plotly_dark',
                    height=800,
                    title=f"{st.session_state.stock_data.name} - Backtest Results",
                    xaxis_rangeslider_visible=False,
                    hovermode='x unified',
                    margin=dict(l=20, r=20, t=40, b=20),
                    plot_bgcolor='rgba(14,17,23,1)',
                    paper_bgcolor='rgba(14,17,23,1)',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
                )

                fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(255,255,255,0.1)')
                fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(255,255,255,0.1)')

                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error creating interactive plot: {e}")

                st.warning("Showing static plot instead")
                try:
                    fig = cerebro.plot(style='candlestick', iplot=False, show=False)[0][0]
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e2:
                    st.error(f"Failed to plot: {e2}")

# ==================== PORTFOLIO TAB ====================
with tab4:
    st.header("💼 Investment Portfolio")

    portfolio_text, total_value = show_portfolio_section(
        stock_data if 'stock_data' in globals() else None,
        metal_data if 'metal_data' in globals() else None
    )

# ==================== THEORY TAB ====================
with tab5:
    st.markdown("""
    ## 📈 Stock Market Analysis
    ### Fundamental Concepts
    Stock market analysis involves evaluating financial instruments to predict future price movements and make investment decisions. 
    Two primary approaches exist:

    - **Technical Analysis:** Studies historical price patterns and market activity
    - **Fundamental Analysis:** Examines company financials and economic factors

    This dashboard focuses on technical analysis with these key components:
    """)

    with st.expander("🔍 Price Charts Explained"):
        st.markdown("""
        ### 📊 Basic Chart
        The simplest visualization showing closing prices over time:
        - Single line representing daily closing values
        - Ideal for quick trend identification
        - Limited interactivity but fast rendering

        **Best for:** Quick glance at overall trend direction

        ### 🖱️ Interactive Chart
        Enhanced visualization with Plotly-powered features:
        - Hover tooltips showing exact values
        - Zoom and pan functionality
        - Moving average overlays (SMA/EMA)
        - Customizable time ranges

        **Best for:** Detailed technical analysis

        ### 🕯️ Candlestick Chart (Japanese Candles)
        Professional traders' preferred format showing:
        - **Body:** Opening vs closing price range
        - **Wicks (Shadows):** Highest/lowest prices during period
        - Colors: 
          - Green/White = Price increased (close > open)
          - Red/Black = Price decreased (close < open)

        **Candlestick Components:**
        - **Upper Shadow:** High price to closing/opening
        - **Real Body:** Opening to closing price
        - **Lower Shadow:** Low price to opening/closing

        **Common Candlestick Patterns:**
        - **Single Candle Patterns:**
          * Doji (open ≈ close): Market indecision
          * Hammer (short body, long lower wick): Potential bullish reversal
          * Hanging Man (similar to hammer but in uptrend): Bearish reversal
          * Shooting Star (short body, long upper wick): Potential bearish reversal

        - **Multi-Candle Patterns:**
          * Engulfing (large body covers previous): 
            - Bullish when green after red
            - Bearish when red after green
          * Harami (small body inside previous large body): Potential reversal
          * Morning Star (3-candle bullish reversal pattern)
          * Evening Star (3-candle bearish reversal pattern)
        """)

    with st.expander("📈 Technical Indicators"):
        st.markdown("""
        ### Moving Averages (Trend Indicators)
        - **SMA (Simple Moving Average):** 
          - Calculation: Sum of closing prices over N periods divided by N
          - Equal weight to all periods
          - Smoother but more lagging indicator
          - Formula: SMA = (P₁ + P₂ + ... + Pₙ) / n

        - **EMA (Exponential Moving Average):**
          - Calculation: More weight to recent prices using multiplier
          - More responsive to price changes but noisier
          - Formula: EMA = [Close - EMA(previous)] × multiplier + EMA(previous)
          - Multiplier = 2 ÷ (N + 1) where N = period count
          - Initial EMA typically uses SMA as seed value

        ### Common Uses:
        - **Trend identification:** 
          - Price above MA = uptrend
          - Price below MA = downtrend
        - **Support/resistance levels**
        - **Crossover strategies:** 
          - Golden Cross: 50-day crosses above 200-day SMA (bullish)
          - Death Cross: 50-day crosses below 200-day SMA (bearish)
        - **Multiple MA confluence:** Using different periods (e.g., 9, 20, 50, 200)
        """)

    with st.expander("🤖 Machine Learning Models"):
        st.markdown("""
        ### Regression Analysis
        - Predicts future prices using linear relationships between variables
        - **Key metrics:**
          - **R² Score (Coefficient of Determination):**
            * Range: 0 (no fit) to 1 (perfect fit)
            * Interpretation: % of variance in dependent variable explained by model
            * Formula: R² = 1 - (SSres / SStot)
            * SSres = sum of squared residuals
            * SStot = total sum of squares

          - **MAE (Mean Absolute Error):**
            * Average absolute difference between predicted and actual values
            * Less sensitive to outliers than RMSE
            * Formula: MAE = (1/n) × Σ|yᵢ - ŷᵢ|
            * Where y = actual, ŷ = predicted

          - **RMSE (Root Mean Squared Error):**
            * Square root of average squared differences
            * More weight to large errors (punishes outliers)
            * Formula: RMSE = √[(1/n) × Σ(yᵢ - ŷᵢ)²]
            * In same units as original data

        ### LSTM Networks (Long Short-Term Memory)
        Specialized recurrent neural networks for time series:
        - **Architecture Features:**
          * Input gate: Controls information updates
          * Forget gate: Determines what to discard
          * Output gate: Controls next hidden state
          * Cell state: Maintains memory across time steps

        - **Advantages for Financial Data:**
          * Handles long-term dependencies (unlike simple RNNs)
          * Learns complex non-linear patterns
          * Adapts to market regime changes
          * Can process sequences of varying lengths

        - **Training Process:**
          * Uses sliding windows of historical data
          * Typical input: Multiple time steps of OHLCV data
          * Output: Next period's price or direction
          * Loss function: Usually Mean Squared Error (MSE)
          * Optimization: Adam or RMSprop commonly used
          * Regularization: Dropout layers prevent overfitting

        ### 📉 Training Loss Graph Interpretation
        The loss curve visualization shows model learning progress.

        - **X-axis (Epoch):**  
          One complete pass through the entire training dataset
          - Typical range: 50-500 epochs for financial models
          - Early stopping often used to prevent overfitting

        - **Y-axis (Loss):**  
          Value of the loss function (usually MSE):
          - Lower values = better predictions
          - Scale depends on data normalization

        **Healthy Patterns:**
        - Steady downward trend (model learning)
        - Final plateau (convergence reached)
        - Small gap between train/validation loss

        **Warning Signs:**
        - ↗️ Upward validation loss = Overfitting
          - Solution: Add dropout, reduce model complexity
        - ↔️ High variance between lines = Data inconsistency
          - Solution: More training data, better normalization
        - 🔄 Oscillations = Learning rate too high
          - Solution: Decrease learning rate

        **Example Interpretation:**
        - Epoch 0-50: Rapid improvement
        - Epoch 50-150: Gradual refinement
        - Epoch 150+: Stable convergence
        """)

    st.markdown("---")

    # Precious Metals Section
    st.markdown("""
    ## 🪙 Precious Metals Analysis
    ### Unique Characteristics
    """)

    with st.expander("🟡 Gold Market Dynamics"):
        st.markdown("""
        - **Safe haven asset:** Tends to rise during market turmoil
        - **Inverse USD relationship:** Stronger dollar typically pressures gold
        - **Industrial use:** ~10% of demand (electronics, dentistry)
        """)

    with st.expander("⚪ Silver Market Dynamics"):
        st.markdown("""
        - **Dual nature:** Both precious metal and industrial commodity
        - **Higher volatility:** Smaller market than gold
        - **Solar panel demand:** Significant industrial use case
        """)

    with st.expander("🟠 Platinum Group Metals"):
        st.markdown("""
        ### Platinum
        - Automotive catalysts (diesel vehicles)
        - Jewelry applications
        - Limited supply (80% from South Africa)

        ### Palladium
        - Gasoline vehicle catalysts
        - Supply deficits common
        - Price often exceeds platinum
        """)

    st.markdown("---")

    # Sentiment Analysis Section
    st.markdown("""
    ## 🗞️ News Sentiment Analysis
    """)

    with st.expander("📰 How News Affects Markets"):
        st.markdown("""
        - **Immediate impact:** Earnings reports, economic data
        - **Cumulative effect:** Sentiment trends over time
        - **Sector-specific:** Commodity news vs. general market

        **Analysis Methods:**
        - Keyword extraction
        - Sentiment scoring (positive/negative)
        - Topic modeling
        """)

    st.header("🤖 Backtrader Strategy Tester")

    with st.expander("📚 Strategy Theory & Parameters Explanation", expanded=False):
        st.markdown("""
            ## Framework Overview
            **Backtrader** is a Python framework for backtesting trading strategies with:
            - Built-in technical indicators
            - Multiple data feed support
            - Realistic broker simulation (commissions, slippage)
            - Comprehensive performance analytics

            ## Strategy Details

            ### 📊 SMA Crossover Strategy
            **Logic:**  
            Uses two Simple Moving Averages (SMA) with different periods:
            - **Buy Signal:** When faster SMA crosses above slower SMA
            - **Sell Signal:** When faster SMA crosses below slower SMA

            **Parameters:**
            - `Fast SMA Period` (10 by default): 
              - Shorter-term trend (more sensitive to price changes)
              - Typical values: 5-20 periods
            - `Slow SMA Period` (50 by default):
              - Longer-term trend (smoothes out noise)
              - Typical values: 20-200 periods

            **Mathematically:**
            ```math
            SMA = (Price₁ + Price₂ + ... + Priceₙ) / n
            ```
            Where n = period length

            ### 📈 RSI Strategy
            **Logic:**  
            Based on Relative Strength Index (momentum oscillator):
            - **Buy Signal:** When RSI < Oversold level (30 by default)
            - **Sell Signal:** When RSI > Overbought level (70 by default)

            **Parameters:**
            - `RSI Period` (14 by default):
              - Number of periods for RSI calculation
              - Shorter periods = more sensitive but noisy
            - `Oversold Level` (30 by default):
              - Threshold for buying (lower = more conservative)
            - `Overbought Level` (70 by default):
              - Threshold for selling (higher = holds positions longer)

            **RSI Formula:**
            ```math
            RSI = 100 - (100 / (1 + RS))
            RS = Avg Gain / Avg Loss (over period)
            ```

            ### 📉 MACD Strategy
            **Logic:**  
            Uses Moving Average Convergence Divergence:
            - **Buy Signal:** When MACD line crosses above Signal line
            - **Sell Signal:** When MACD line crosses below Signal line

            **Parameters:**
            - `MACD Fast` (12 by default):
              - Period for faster EMA (responsive to recent prices)
            - `MACD Slow` (26 by default):
              - Period for slower EMA (identifies broader trend)
            - `Signal Period` (9 by default):
              - EMA of MACD line (creates trigger line)

            **MACD Components:**
            ```math
            MACD Line = EMA(fast) - EMA(slow)
            Signal Line = EMA(MACD Line)
            Histogram = MACD Line - Signal Line
            ```
            """)

    st.header("💼 Investment Portfolio")

    with st.expander("📊 Portfolio Theory & Functionality"):
        st.markdown("""
        ## Portfolio Analysis Overview
        This tab provides comprehensive tools to monitor and analyze your investment portfolio performance across different asset classes.

        ### Key Features:

        **1. Portfolio Composition**
        - Real-time valuation of all holdings (stocks & precious metals)
        - Visual allocation breakdown (pie chart)
        - Asset-type classification (equities vs commodities)

        **2. Performance Metrics**
        - Current value vs total invested capital
        - Absolute profit/loss (in currency and percentage)
        - Daily high/low valuation range
        - Color-coded profit indicators (green/red)

        **3. Position Analysis**
        - Average purchase price per asset
        - Quantity held and unit of measurement
        - Purchase date range (first/last buy)
        - Individual position performance

        **4. Risk Assessment**
        - Portfolio concentration/diversification
        - Volatility estimation (via price ranges)
        - Asset correlation visualization

        **5. Reporting Tools**
        - Telegram integration for report sharing
        - Export-ready visualizations
        - Tabular data presentation

        ## Modern Portfolio Theory (MPT) Principles
        This implementation incorporates core MPT concepts:

        **Diversification Benefits**
        - Measures how effectively your portfolio combines:
          - Stocks (higher growth potential)
          - Precious metals (inflation hedge)
        - Highlights over-concentration risks

        **Risk-Return Profile**
        - Shows whether returns compensate for:
          - Market volatility
          - Asset-specific risks
          - Opportunity costs

        **Performance Attribution**
        - Identifies which assets contribute most to:
          - Overall returns
          - Portfolio volatility
          - Diversification benefits

        ## Practical Applications
        1. **Rebalancing Decisions**  
           - Identify under/over-weighted assets
           - Compare current vs target allocations

        2. **Tax Optimization**  
           - Sort positions by profit/loss
           - Identify tax-loss harvesting candidates

        3. **Risk Management**  
           - Monitor position concentrations
           - Track volatility patterns
           - Set stop-loss thresholds

        4. **Performance Benchmarking**  
           - Compare against market indices
           - Evaluate asset class performance
        """)
