import os
import textwrap
import streamlit as st
import plotly.graph_objects as go
import requests
from src.utils import save_plotly_figure
from bot.telegram_bot import send_telegram_message, send_telegram_photo
import matplotlib.pyplot as plt
import json
from pathlib import Path
from datetime import datetime

PORTFOLIO_PATH = Path("config/portfolio.json")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def save_table_image(df, filename="portfolio_table.png"):
    """
    Save a styled image of a DataFrame as a PNG table with conditional row coloring based on 'Profit' column.

    :param df: DataFrame to be converted into a styled image table
    :param filename: Name of the output image file (PNG format)
    :return: Path to the saved image file
    """
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.5 + 1))
    ax.axis("off")

    row_colors = [
        "#e6ffed" if row["Profit"].startswith("+") or not row["Profit"].startswith("-") else "#ffe6e6"
        for _, row in df.iterrows()
    ]

    def text_len(text):
        """
        Estimate visual text length based on character types for column width normalization.

        :param text: Text content to evaluate
        :return: Estimated visual length as a float
        """
        return sum(1.5 if c.isdigit() else 1.8 if c.isalpha() else 1.2 for c in str(text))

    max_lengths = []
    for col in df.columns:
        lengths = [text_len(val) for val in df[col]] + [text_len(col)]
        max_lengths.append(max(lengths))

    total = sum(max_lengths)
    col_widths = [length / total for length in max_lengths]

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
        rowLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for i in range(len(df) + 1):
        for j in range(len(df.columns)):
            cell = table[i, j]
            cell.set_width(col_widths[j])

            if i == 0:
                cell.set_fontsize(10)
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#f0f0f0')
            else:
                cell.set_facecolor(row_colors[i - 1])

    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()

    return filename


def get_stock_candle_high_low(ticker):
    """
    Retrieve the latest HIGH and LOW candle prices for a given stock ticker from the MOEX API.

    :param ticker: Stock ticker symbol (e.g., 'SBER')
    :return: Tuple of (HIGH price, LOW price) or (None, None) if data is unavailable
    """
    try:
        url = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.json"
        response = requests.get(url)
        data = response.json()
        columns = data["marketdata"]["columns"]
        rows = data["marketdata"]["data"]

        if not rows:
            return None, None

        high_idx = columns.index("HIGH")
        low_idx = columns.index("LOW")

        for row in rows:
            if row[1] == "TQBR":
                return row[high_idx], row[low_idx]

        return rows[0][high_idx], rows[0][low_idx]

    except Exception as e:
        print(f"Error getting HIGH/LOW for {ticker}: {e}")
        return None, None


def get_latest_stock_price(ticker, date_str=None):
    """
    Retrieve the latest stock price (LAST or CLOSE) for a given ticker, optionally for a specific date.

    :param ticker: Stock ticker symbol (e.g., 'GAZP')
    :param date_str: Optional date in 'YYYY-MM-DD' format to fetch historical CLOSE price
    :return: Price as float or None if not found
    """
    try:
        if date_str:
            url = f"https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.json?from={date_str}&till={date_str}"
            data_key = "history"
            price_field = "CLOSE"
        else:
            url = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.json"
            data_key = "marketdata"
            price_field = "LAST"

        response = requests.get(url)
        data = response.json()

        if date_str:
            rows = data[data_key]["data"]
            if not rows:
                return None
            columns = data[data_key]["columns"]
            price_index = columns.index(price_field)
            return rows[0][price_index]
        else:
            for row in data[data_key]['data']:
                if row[1] == "TQBR":
                    return row[12]
            return data[data_key]['data'][0][12] if data[data_key]['data'] else None

    except Exception as e:
        print(f"Error getting price for {ticker}" + (f" on {date_str}" if date_str else "") + f": {e}")
        return None


def show_portfolio_section(stock_data=None, metal_data=None):
    """
    Display the user's current portfolio in the Streamlit interface, including summary statistics,
    profit/loss per asset, and asset allocation pie chart. Also allows sending a Telegram report.

    :param stock_data: Optional DataFrame with recent stock prices (used for price lookup)
    :param metal_data: Optional DataFrame with metal price data (must contain 'CLOSE', 'LOW', 'HIGH' columns)
    :return: Tuple (full_report, total_portfolio_value)
    """
    total_min_value = 0
    total_max_value = 0
    if 'portfolio' not in st.session_state or 'portfolio_details' not in st.session_state:

        saved_portfolio = load_portfolio()

        st.session_state.portfolio = {'stocks': {}, 'metals': {}}
        st.session_state.portfolio_details = {'stocks': {}, 'metals': {}}

        for item in saved_portfolio.get("portfolio", []):
            ticker = item["ticker"]
            quantity = item["quantity"]
            purchase_price = item["purchase_price"]
            purchase_date = item["purchase_date"]

            if ticker.lower() in ["gold", "silver", "platinum", "palladium"]:
                st.session_state.portfolio['metals'][ticker] = quantity
                st.session_state.portfolio_details['metals'][ticker] = {
                    'purchase_price': purchase_price,
                    'purchase_date': purchase_date
                }
            else:
                st.session_state.portfolio['stocks'][ticker] = quantity
                st.session_state.portfolio_details['stocks'][ticker] = {
                    'purchase_price': purchase_price,
                    'purchase_date': purchase_date
                }

    col1, col2 = st.columns([2, 1])

    portfolio_data = load_portfolio()

    aggregated_stocks = {}
    aggregated_metals = {}
    stock_values = {}
    metal_values = {}

    for item in portfolio_data.get("portfolio", []):
        ticker = item["ticker"]
        quantity = item["quantity"]
        purchase_price = item["purchase_price"]

        if ticker.lower() in ["gold", "silver", "platinum", "palladium"]:
            target_dict = aggregated_metals
            asset_type = "Metal"
            units = "g"
        else:
            target_dict = aggregated_stocks
            asset_type = "Stock"
            units = "shares"

        if ticker not in target_dict:
            target_dict[ticker] = {
                "total_quantity": quantity,
                "total_investment": purchase_price * quantity,
                "purchase_dates": [item["purchase_date"]],
                "asset_type": asset_type,
                "units": units
            }
        else:
            target_dict[ticker]["total_quantity"] += quantity
            target_dict[ticker]["total_investment"] += purchase_price * quantity
            target_dict[ticker]["purchase_dates"].append(item["purchase_date"])

    table_data = []
    total_stock_value = 0
    total_stock_profit = 0
    total_stock_investment = 0
    total_metal_value = 0
    total_metal_profit = 0
    total_metal_investment = 0

    for ticker, data in aggregated_stocks.items():
        current_price = get_latest_stock_price(ticker)
        high_price, low_price = get_stock_candle_high_low(ticker)

        if current_price is not None and high_price is not None and low_price is not None:
            avg_price = data["total_investment"] / data["total_quantity"]
            current_value = current_price * data["total_quantity"]
            profit = current_value - data["total_investment"]
            profit_pct = (profit / data["total_investment"]) * 100

            min_value = low_price * data["total_quantity"]
            max_value = high_price * data["total_quantity"]

            total_min_value += min_value
            total_max_value += max_value

            total_stock_value += current_value
            total_stock_investment += data["total_investment"]
            total_stock_profit += profit
            stock_values[ticker] = current_value

            table_data.append({
                "Asset": ticker,
                "Type": data["asset_type"],
                "Quantity": round(data["total_quantity"], 2),
                "Units": data["units"],
                "Avg Buy Price": f"{avg_price:.2f} RUB",
                "First Buy Date": min(data["purchase_dates"]),
                "Last Buy Date": max(data["purchase_dates"]),
                "Current Price": f"{current_price:.2f} RUB",
                "Value": f"{current_value:.2f} RUB",
                "Profit": f"{profit:.2f} RUB",
                "Profit %": f"{profit_pct:.2f}%",
                "Status": "✅ Profit" if profit >= 0 else "❌ Loss",
                "Value_num": current_value,
                "Profit_num": profit,
                "Profit_pct_num": profit_pct,
                "Min Value": min_value,
                "Max Value": max_value
            })

    for metal, data in aggregated_metals.items():
        if metal_data is not None and metal.lower() in ["gold", "silver", "platinum", "palladium"]:
            current_price = metal_data['CLOSE'].iloc[-1]
            avg_price = data["total_investment"] / data["total_quantity"]
            current_value = current_price * data["total_quantity"]
            profit = current_value - data["total_investment"]
            profit_pct = (profit / data["total_investment"]) * 100

            if 'LOW' in metal_data.columns and 'HIGH' in metal_data.columns:
                low_price = metal_data['LOW'].iloc[-1]
                high_price = metal_data['HIGH'].iloc[-1]
            else:
                low_price = current_price
                high_price = current_price

            min_value = low_price * data["total_quantity"]
            max_value = high_price * data["total_quantity"]

            total_min_value += min_value
            total_max_value += max_value

            total_metal_value += current_value
            total_metal_investment += data["total_investment"]
            total_metal_profit += profit
            metal_values[metal] = current_value

            table_data.append({
                "Asset": metal,
                "Type": data["asset_type"],
                "Quantity": round(data["total_quantity"], 2),
                "Units": data["units"],
                "Avg Buy Price": f"{avg_price:.2f} RUB/g",
                "First Buy Date": min(data["purchase_dates"]),
                "Last Buy Date": max(data["purchase_dates"]),
                "Current Price": f"{current_price:.2f} RUB/g",
                "Value": f"{current_value:.2f} RUB",
                "Profit": f"{profit:.2f} RUB",
                "Profit %": f"{profit_pct:.2f}%",
                "Status": "✅ Profit" if profit >= 0 else "❌ Loss",
                "Value_num": current_value,
                "Profit_num": profit,
                "Profit_pct_num": profit_pct,
                "Min Value": min_value,
                "Max Value": max_value
            })

    total_value = total_stock_value + total_metal_value
    total_investment = total_stock_investment + total_metal_investment
    total_profit = total_stock_profit + total_metal_profit
    total_profit_percent = (total_value / total_investment - 1) * 100 if total_investment > 0 else 0

    with col1:
        st.markdown("### 📋 Current Portfolio")

        if table_data:
            import pandas as pd
            df = pd.DataFrame(table_data).sort_values('Value_num', ascending=False)

            table_for_telegram = df[[
                "Asset", "Type", "Quantity", "Avg Buy Price", "Current Price",
                "Value", "Profit", "Profit %"
            ]].copy()

            def color_profit(col):
                """
                Apply conditional styling to highlight profit/loss columns in the DataFrame.

                - 'Profit_num' and 'Profit_pct_num' are colored green if profit is positive, red if negative.
                - 'Status' is colored green for "✅ Profit" and red for "❌ Loss".

                :param col: A pandas Series (column from the DataFrame)
                :return: A list of CSS style strings for each cell in the column
                """
                if col.name in ['Profit_num', 'Profit_pct_num']:
                    return ['color: green' if x >= 0 else 'color: red' for x in col]
                elif col.name == 'Status':
                    return ['color: green' if 'Profit' in str(x) else 'color: red' if 'Loss' in str(x) else '' for x in
                            col]
                else:
                    return [''] * len(col)

            styled_df = df.style.apply(color_profit, subset=['Profit_num', 'Profit_pct_num', 'Status'])

            format_dict = {
                'Quantity': '{:,.2f}',
                'Value_num': '{:,.2f} RUB',
                'Profit_num': '{:+,.2f} RUB',
                'Profit_pct_num': '{:+,.2f}%',
                'Min Value': '{:,.2f} RUB',
                'Max Value': '{:,.2f} RUB'
            }

            st.dataframe(
                styled_df.format(format_dict).hide(axis='index'),
                column_config={
                    "Value_num": st.column_config.NumberColumn("Value", format="%.2f"),
                    "Current Price": st.column_config.TextColumn("Current Price"),
                    "Profit_num": st.column_config.NumberColumn("Profit", format="%+.2f"),
                    "Profit_pct_num": st.column_config.NumberColumn("Profit %", format="%+.2f%%"),
                    "Status": st.column_config.TextColumn("Status"),
                    "Min Value": st.column_config.NumberColumn("Min Value", format="%.2f"),
                    "Max Value": st.column_config.NumberColumn("Max Value", format="%.2f")
                },
                use_container_width=True,
                hide_index=True
            )

        st.markdown("### 📊 Summary")
        st.markdown(f"**Total Value:** {total_value:.2f} RUB")
        st.markdown(f"**Total Invested:** {total_investment:.2f} RUB")
        st.markdown(f"**Total Profit:** {total_profit:.2f} RUB ({total_profit_percent:.2f}%)")
        st.markdown(f"**Minimum If Sold Now:** {total_min_value:.2f} RUB")
        st.markdown(f"**Maximum If Sold Now:** {total_max_value:.2f} RUB")

    with col2:
        st.markdown("### 📊 Portfolio Allocation")
        if total_value > 0:
            labels = [f"{t} (Stock)" for t in stock_values.keys() if stock_values[t] > 0] + \
                     [f"{m} (Metal)" for m in metal_values.keys() if metal_values[m] > 0]
            values = [v for v in stock_values.values() if v > 0] + \
                     [v for v in metal_values.values() if v > 0]

            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=.3,
                textinfo='percent+label',
                insidetextorientation='radial'
            )])
            fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Portfolio is empty. Add assets to see allocation.")

    full_report = None

    if st.button("📤 Send Portfolio Report to Telegram"):
        summary_text = textwrap.dedent(f"""
        📋 *Portfolio Summary*
            💰 Total Value: {total_value:.2f} RUB
            💸 Total Invested: {total_investment:.2f} RUB
            📈 Total Profit: {total_profit:.2f} RUB ({total_profit_percent:.2f}%)
            🔻 Minimum If Sold Now: {total_min_value:.2f} RUB
            🔺 Maximum If Sold Now: {total_max_value:.2f} RUB
        """)

        message = f"*📅 Portfolio Report — {datetime.now().strftime('%Y-%m-%d')}*\n\n" + summary_text
        send_telegram_message(message)

        temp_files = []

        df_short = table_for_telegram.copy()
        file_path = save_table_image(df_short, "portfolio_table.png")
        file_path = PROJECT_ROOT / file_path
        temp_files.append(file_path)

        send_telegram_photo("portfolio_table.png", caption="📊 Portfolio Table")

        if total_value > 0:
            fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
            fig_pie.update_layout(
                title="Portfolio Allocation",
                colorway=[
                    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
                    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
                    '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173',
                    '#3182bd', '#e6550d', '#31a354', '#756bb1', '#636363'
                ],
                plot_bgcolor='white',
                paper_bgcolor='white'
            )

            path_pie = save_plotly_figure(fig_pie, "portfolio_allocation")
            send_telegram_photo(path_pie, "📊 Portfolio Allocation")

            for file_path in temp_files:
                try:
                    if Path(file_path).exists():
                        os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting temporary file {file_path}: {e}")

            st.success("Portfolio report sent to Telegram!")
        else:
            st.warning("Cannot send empty portfolio")

    return full_report, total_value


def load_portfolio():
    """
    Load the saved portfolio from the predefined JSON file.

    :return: Dictionary with portfolio data, format:
             {
                 "portfolio": [
                     {
                         "ticker": str,
                         "quantity": float,
                         "purchase_price": float,
                         "purchase_date": str
                     },
                     ...
                 ]
             }
    """
    if PORTFOLIO_PATH.exists():
        with open(PORTFOLIO_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"portfolio": []}
