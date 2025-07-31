import textwrap
from datetime import datetime
import streamlit as st
from src.utils import save_plotly_figure, read_forecast_config, write_forecast_config
import requests
import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")


def send_telegram_message(text):
    """
    Sends a plain text message to a predefined Telegram chat using the bot API.

    :param text: str — Message text to send (Markdown supported)
    :return: None
    """
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, data=payload)
        response.raise_for_status()
    except Exception as e:
        print(f"Telegram error: {e}")


def send_telegram_photo(image_path, caption=None):
    """
    Sends a single image to Telegram with an optional caption.

    :param image_path: str — Path to the image file
    :param caption: str or None — Optional caption for the image (Markdown supported)
    :return: None
    """
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    with open(image_path, 'rb') as photo:
        files = {"photo": photo}
        data = {
            "chat_id": CHAT_ID,
            "caption": caption or "",
            "parse_mode": "Markdown"
        }
        try:
            response = requests.post(url, files=files, data=data)
            response.raise_for_status()
        except Exception as e:
            print(f"Telegram photo error: {e}")


def send_report_to_telegram(asset_name, r2, forecast, lstm_forecast, fig_forecast, fig_lstm, fig_main, chart_type,
                            fig_candlestick, news_results=None, last_close_price=None):
    """
    Sends a detailed market report with forecasts, charts, and news sentiment to Telegram.

    :param asset_name: str — Name of the asset (e.g., 'SBER', 'GOLD')
    :param r2: float — R² score of the regression model
    :param forecast: list — List of regression forecasted prices
    :param lstm_forecast: list or None — List of LSTM forecasted prices
    :param fig_forecast: plotly.Figure — Forecast chart (regression)
    :param fig_lstm: plotly.Figure — LSTM forecast chart
    :param fig_main: plotly.Figure — Main asset price chart
    :param chart_type: str — Type of the main chart (e.g., 'Line', 'OHLC')
    :param fig_candlestick: plotly.Figure — Candlestick chart of the asset
    :param news_results: dict or None — News analysis results including sentiment and headlines
    :param last_close_price: float or None — Last close price of the asset
    :return: None
    """
    try:
        lstm_text = (
            f"{lstm_forecast[-1]:.2f}"
            if lstm_forecast is not None else
            "- Forecast not available"
        )

        message = textwrap.dedent(f"""
            📅 *Market Report — {datetime.now().strftime('%Y-%m-%d')}*

            📈 *Asset:* {asset_name}
            💰 *Last Close Price:* {last_close_price:.2f} ₽
            🔎 *Linear Regression:*
            - R²: {r2:.2f}
            - Forecast End Price: {forecast[-1]:.2f}

            🤖 *LSTM Forecast:*
            - Forecast End Price: {lstm_text}
        """).strip()

        if news_results:
            sentiment_emoji = "🟢" if news_results["textblob_score"] > 0.1 else \
                "🔴" if news_results["textblob_score"] < -0.1 else "🟡"

            transformers_emoji = "🟢" if news_results["transformers_score"] > 0.1 else \
                "🔴" if news_results["transformers_score"] < -0.1 else "🟡"

            message += textwrap.dedent(f"""

                📰 *News Sentiment Analysis ({news_results['language'].upper()})*
                - TextBlob: {news_results["textblob_score"]:.2f} {sentiment_emoji}
                - {news_results["transformers_model"]}: {news_results["transformers_score"]:.2f} {transformers_emoji}

                *Top News Headlines:*
            """)

            for i, article in enumerate(news_results["articles"][:3], 1):
                message += f"\n{i}. [{article['title']}]({article['url']})"

        send_telegram_message(message)

        paths = []
        captions = []

        if fig_main:
            p = save_plotly_figure(fig_main, "main_chart")
            paths.append(p)
            captions.append(f"📉 {chart_type} Chart")

        if fig_forecast:
            p = save_plotly_figure(fig_forecast, "forecast_chart")
            paths.append(p)
            captions.append("📊 Linear Regression Forecast")

        if fig_lstm:
            p = save_plotly_figure(fig_lstm, "lstm_chart")
            paths.append(p)
            captions.append("🤖 LSTM Price Forecast")

        if fig_candlestick:
            p = save_plotly_figure(fig_candlestick, "candlestick_chart")
            paths.append(p)
            captions.append(f"🕯️ {asset_name} Candlestick Chart")

        send_telegram_media_group(paths, captions)

        st.success("📨 Report and charts successfully sent to Telegram!")
    except Exception as e:
        st.error(f"❌ Error sending to Telegram: {e}")


def send_telegram_media_group(image_paths, captions=None):
    """
    Sends multiple images as a media group to Telegram with optional individual captions.

    :param image_paths: list[str] — List of paths to image files
    :param captions: list[str] or None — Optional captions for each image (same length as image_paths)
    :return: None
    """
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMediaGroup"

    media = []
    for i, img_path in enumerate(image_paths):
        caption = captions[i] if captions and i < len(captions) else ""
        media.append({
            "type": "photo",
            "media": f"attach://photo{i}",
            "caption": caption,
            "parse_mode": "Markdown"
        })

    files = {}
    for i, img_path in enumerate(image_paths):
        files[f"photo{i}"] = open(img_path, 'rb')

    data = {
        "chat_id": CHAT_ID,
        "media": str(media).replace("'", '"')
    }

    try:
        response = requests.post(url, data=data, files=files)
        response.raise_for_status()
    except Exception as e:
        print(f"Telegram media group error: {e}")
    finally:
        for f in files.values():
            f.close()


def process_command(text):
    """
    Processes a user input command to update forecast configuration (e.g., LSTM or regression days).

    :param text: str — Command string in the format: "set <AssetType> <ModelType> <Days>"
    :return: str — Result message indicating success or error details
    """
    parts = text.strip().split()
    if len(parts) != 4 or parts[0].lower() != "set":
        return "❌ Invalid command format. Example: set Stock lstm_days 10"

    _, asset_type, model_type, days = parts
    try:
        days = int(days)
    except ValueError:
        return "❌ Number of days must be an integer."

    config = read_forecast_config()
    if "forecast_defaults" not in config:
        config["forecast_defaults"] = {}

    if asset_type not in config["forecast_defaults"]:
        config["forecast_defaults"][asset_type] = {}

    config["forecast_defaults"][asset_type][model_type] = days
    write_forecast_config(config)

    return f"✅ Set: {asset_type} → {model_type} = {days} days"
