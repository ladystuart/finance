from bot.telegram_bot import BOT_TOKEN
import json
import time
from pathlib import Path
from src.portfolio import get_latest_stock_price
import requests
from datetime import datetime
from bs4 import BeautifulSoup

BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_DIR.mkdir(exist_ok=True)

SETTINGS_PATH = CONFIG_DIR / "settings.json"
STATE_PATH = CONFIG_DIR / "user_state.json"
PORTFOLIO_PATH = CONFIG_DIR / "portfolio.json"

CBR_METAL_CODES = {
    "gold": "1",
    "silver": "2",
    "platinum": "3",
    "palladium": "4"
}


def get_latest_cbr_metal_price(metal):
    """
    Fetches the latest Central Bank of Russia (CBR) price for a given metal.

    :param metal: str name of the metal (e.g., 'gold', 'silver')
    :return: float or None – latest metal price in RUB or None if unavailable
    """
    metal = metal.lower()
    metal_code = CBR_METAL_CODES.get(metal)

    if not metal_code:
        print(f"❌ Unsupported metal: {metal}")
        return None

    try:
        url = "https://www.cbr.ru/scripts/xml_metall.asp"
        today = datetime.now().strftime("%d/%m/%Y")
        params = {
            "date_req1": today,
            "date_req2": today,
            "VAL_NM_RQ": metal_code
        }

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
        }

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "xml")
        records = soup.find_all("Record", {"Code": metal_code})

        latest_date = None
        latest_value = None

        for record in records:
            try:
                date = datetime.strptime(record["Date"], "%d.%m.%Y")
                buy_tag = record.find("Buy")
                if buy_tag is None:
                    continue
                value = float(buy_tag.text.replace(",", "."))
            except (KeyError, ValueError, AttributeError) as e:
                print(f"⚠️ Error parsing record for {metal}: {e}")
                continue
            except Exception as e:
                print(f"⚠️ Unexpected error parsing record for {metal}: {e}")
                continue

            if latest_date is None or date > latest_date:
                latest_date = date
                latest_value = value

        return latest_value

    except requests.RequestException as e:
        print(f"⚠️ Request error getting metal price from CBR for {metal}: {e}")
        return None
    except Exception as e:
        print(f"⚠️ Unexpected error getting metal price for {metal}: {e}")
        return None


def handle_final_portfolio_entry(chat_id, state):
    """
    Finalizes a user's buy action for a stock by recording it in the portfolio.

    :param chat_id: int Telegram chat ID of the user
    :param state: dict user state with stock, quantity, and date
    :return: None
    """
    entry = state[str(chat_id)]
    ticker = entry["ticker"]
    quantity = entry["quantity"]
    purchase_date = entry["purchase_date"]
    action = entry["portfolio_action"]

    if action == "buy":
        price = get_latest_stock_price(ticker, purchase_date)
        if price is None:
            send_message(chat_id, f"⚠️ Could not get price for {ticker} on {purchase_date}")
            return

        portfolio = load_portfolio()["portfolio"]
        portfolio.append({
            "ticker": ticker,
            "quantity": quantity,
            "purchase_price": price,
            "purchase_date": purchase_date
        })
        save_portfolio({"portfolio": portfolio})
        send_message(chat_id, f"✅ Bought {quantity} of {ticker} at {price:.2f} RUB on {purchase_date}")
    else:
        send_message(chat_id, "❌ Only buy is implemented for historical date")

    state.pop(str(chat_id))
    save_state(state)


def load_portfolio():
    """
    Loads the current user's portfolio from file.

    :return: dict with 'portfolio' key, each item contains ticker, qty, price, date
    """
    if PORTFOLIO_PATH.exists():
        with open(PORTFOLIO_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"portfolio": []}


def save_portfolio(data):
    """
    Saves portfolio data to a JSON file.

    :param data: dict with portfolio information
    :return: None
    """
    with open(PORTFOLIO_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_settings():
    """
    Loads forecast configuration settings from file.

    :return: dict with default settings or empty structure if file missing
    """
    try:
        with open(SETTINGS_PATH, "r") as f:
            return json.load(f)
    except:
        return {"forecast_defaults": {"Stock": {}, "Metal": {}}}


def save_settings(data):
    """
    Saves forecast configuration settings to file.

    :param data: dict of settings to save
    :return: None
    """
    with open(SETTINGS_PATH, "w") as f:
        json.dump(data, f, indent=4)


def load_state():
    """
    Loads current user interaction state from file.

    :return: dict representing the state for all users
    """
    try:
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    except:
        return {}


def save_state(state):
    """
    Saves user interaction state to file.

    :param state: dict of chat_id to state mappings
    :return: None
    """
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=4)


def send_message(chat_id, text, reply_markup=None):
    """
    Sends a text message to a specific Telegram chat.

    :param chat_id: int chat ID to send the message to
    :param text: str message text (supports Markdown)
    :param reply_markup: dict optional inline keyboard layout
    :return: None
    """
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown"
    }
    if reply_markup:
        payload["reply_markup"] = json.dumps(reply_markup)
    requests.post(f"{BASE_URL}/sendMessage", data=payload)


def get_forecast_config_text():
    """
    Generates a formatted message summarizing current forecast config.

    :return: str summary of config values for stocks and metals
    """
    config = load_settings()
    forecast_defaults = config.get("forecast_defaults", {})
    if not forecast_defaults:
        return "⚠️ Forecast configuration is empty or file not found."

    lines = ["📊 Current forecast settings:"]
    for asset_type in ["Stock", "Metal"]:
        asset_config = forecast_defaults.get(asset_type, {})
        lstm_days = asset_config.get("lstm_days", "not set")
        regression_days = asset_config.get("regression_days", "not set")
        lines.append(f"*{asset_type}*:")
        lines.append(f"  - regression days: {regression_days}")
        lines.append(f"  - lstm days: {lstm_days}")
    return "\n".join(lines)


def handle_command(chat_id, text):
    """
    Processes text-based Telegram bot commands.

    :param chat_id: int chat ID of the sender
    :param text: str text of the command (e.g., '/start', '/set_lstm_days')
    :return: None
    """
    state = load_state()

    if text == "/set_lstm_days":
        keyboard = {
            "inline_keyboard": [[
                {"text": "Stock", "callback_data": "set_lstm_stock"},
                {"text": "Metal", "callback_data": "set_lstm_metal"}
            ]]
        }
        send_message(chat_id, "Choose what to set `lstm_days` for:", keyboard)
        state[str(chat_id)] = {"action": "awaiting_lstm_choice"}
        save_state(state)

    elif text == "/set_regression_days":
        keyboard = {
            "inline_keyboard": [[
                {"text": "Stock", "callback_data": "set_regression_stock"},
                {"text": "Metal", "callback_data": "set_regression_metal"}
            ]]
        }
        send_message(chat_id, "Choose what to set `regression_days` for:", keyboard)
        state[str(chat_id)] = {"action": "awaiting_regression_choice"}
        save_state(state)

    elif text == "/show_config":
        config_text = get_forecast_config_text()
        send_message(chat_id, config_text)

    elif text == "/edit_portfolio":
        keyboard = {
            "inline_keyboard": [[
                {"text": "Buy", "callback_data": "portfolio_buy"},
                {"text": "Sell", "callback_data": "portfolio_sell"}
            ]]
        }
        send_message(chat_id, "Choose an action:", keyboard)
        state[str(chat_id)] = {"action": "awaiting_portfolio_action"}
        save_state(state)

    elif text == "/show_portfolio":
        portfolio_data = load_portfolio()
        portfolio = portfolio_data.get("portfolio", [])
        if not portfolio:
            send_message(chat_id, "📂 Your portfolio is empty.")
            return
        summary = {}
        for item in portfolio:
            t = item["ticker"]
            q = item["quantity"]
            p = item["purchase_price"]
            if t not in summary:
                summary[t] = {"total_qty": 0, "total_cost": 0}
            summary[t]["total_qty"] += q
            summary[t]["total_cost"] += q * p
        lines = ["📊 *Your Portfolio:*"]
        for ticker, data in summary.items():
            qty = data["total_qty"]
            total_cost = data["total_cost"]
            avg_price = total_cost / qty
            try:
                if ticker.lower() in CBR_METAL_CODES:
                    current_price = get_latest_cbr_metal_price(ticker.lower())
                else:
                    current_price = get_latest_stock_price(ticker)
            except Exception:
                current_price = None

            if current_price is None:
                lines.append(f"- *{ticker}*: {qty} pcs @ avg {avg_price:.2f} ₽ — ❌ _Current price not available_")
            else:
                current_value = qty * current_price
                lines.append(
                    f"- *{ticker}*: {qty} pcs\n"
                    f"    • Avg Buy: {avg_price:.2f} ₽\n"
                    f"    • Buy Total: {total_cost:.2f} ₽\n"
                    f"    • Now: {current_price:.2f} ₽ → *{current_value:.2f} ₽*"
                )
        message = "\n".join(lines)
        send_message(chat_id, message)

    else:
        send_message(chat_id, "Unknown command")


def handle_callback_query(chat_id, data):
    """
    Handles button presses from inline keyboards in the bot.

    :param chat_id: int Telegram user ID
    :param data: str callback data from button click
    :return: None
    """
    state = load_state()
    user_id = str(chat_id)

    if data.startswith("set_lstm_"):
        asset = data.split("_")[-1].capitalize()
        send_message(chat_id, f"Enter new `lstm_days` value for {asset} (14-21):")
        state[user_id] = {"action": "awaiting_lstm_value", "asset": asset}

    elif data.startswith("set_regression_"):
        asset = data.split("_")[-1].capitalize()
        send_message(chat_id, f"Enter new `regression_days` value for {asset} (7-10):")
        state[user_id] = {"action": "awaiting_regression_value", "asset": asset}

    elif data in ["portfolio_buy", "portfolio_sell"]:
        action = "buy" if data == "portfolio_buy" else "sell"
        state[user_id] = {
            "action": "awaiting_ticker_quantity",
            "portfolio_action": action
        }
        send_message(chat_id, "Enter ticker and quantity (e.g., `GAZP 20`):")

    elif data in ["date_today", "date_custom"]:
        entry = state.get(user_id, {})
        if not entry:
            return

        if data == "date_today":
            entry["purchase_date"] = datetime.today().strftime("%Y-%m-%d")
            state[user_id] = entry
            handle_final_portfolio_entry(chat_id, state)
        elif data == "date_custom":
            entry["action"] = "awaiting_custom_date"
            send_message(chat_id, "Enter date in format `YYYY-MM-DD`:")
        save_state(state)

    save_state(state)


def handle_text_input(chat_id, text):
    """
    Handles arbitrary text input depending on the user’s current state.

    :param chat_id: int Telegram user ID
    :param text: str user message not starting with '/'
    :return: None
    """
    state = load_state()
    settings = load_settings()
    user_id = str(chat_id)

    if user_id not in state:
        return

    entry = state[user_id]
    action = entry["action"]

    if action in ["awaiting_lstm_value", "awaiting_regression_value"]:
        if not text.isdigit():
            send_message(chat_id, "Please enter a whole number")
            return
        value = int(text)
        asset = entry["asset"]

    if action == "awaiting_lstm_value":
        if 14 <= value <= 21:
            settings["forecast_defaults"][asset]["lstm_days"] = value
            save_settings(settings)
            send_message(chat_id, f"✅ `lstm_days` for {asset} set to {value}")
            state.pop(user_id)
        else:
            send_message(chat_id, "❌ Please enter a number between *14* and *21* for `lstm_days`.")

    elif action == "awaiting_regression_value":
        if 7 <= value <= 10:
            settings["forecast_defaults"][asset]["regression_days"] = value
            save_settings(settings)
            send_message(chat_id, f"✅ `regression_days` for {asset} set to {value}")
            state.pop(user_id)
        else:
            send_message(chat_id, "❌ Please enter a number between *7* and *10* for `regression_days`.")
    elif entry["action"] == "awaiting_ticker_quantity":
        parts = text.strip().split()
        if len(parts) != 2:
            send_message(chat_id, "❌ Please enter both ticker and quantity, like: `GAZP 10`")
            return
        ticker, qty_str = parts[0].upper(), parts[1]
        try:
            quantity = float(qty_str)
        except ValueError:
            send_message(chat_id, "❌ Quantity must be a number.")
            return

        action = entry["portfolio_action"]

        if action == "buy":
            state[user_id] = {
                "action": "awaiting_date_choice",
                "portfolio_action": action,
                "ticker": ticker,
                "quantity": quantity
            }
            save_state(state)
            keyboard = {
                "inline_keyboard": [[
                    {"text": "📅 Today", "callback_data": "date_today"},
                    {"text": "📆 Select Date", "callback_data": "date_custom"}
                ]]
            }

            send_message(chat_id, "Choose purchase date:", keyboard)

        elif action == "sell":
            portfolio = load_portfolio()["portfolio"]
            matching = sorted([p for p in portfolio if p["ticker"] == ticker], key=lambda x: x["purchase_date"])
            total_available = sum(p["quantity"] for p in matching)
            if not matching:
                send_message(chat_id, f"❌ You don't own any {ticker}")
            elif total_available < quantity:
                send_message(chat_id, f"❌ You only own {total_available} of {ticker}, cannot sell {quantity}")
            else:
                remaining = quantity
                new_portfolio = []
                for record in portfolio:
                    if record["ticker"] != ticker:
                        new_portfolio.append(record)
                    else:
                        if remaining <= 0:
                            new_portfolio.append(record)
                        elif record["quantity"] <= remaining:
                            remaining -= record["quantity"]
                        else:
                            record["quantity"] -= remaining
                            new_portfolio.append(record)
                            remaining = 0
                save_portfolio({"portfolio": new_portfolio})
                send_message(chat_id, f"✅ Sold {quantity} of {ticker}")

            state.pop(user_id)

            save_state(state)

    elif action == "awaiting_custom_date":
        try:
            datetime.strptime(text, "%Y-%m-%d")
        except ValueError:
            send_message(chat_id, "❌ Invalid date format. Use `YYYY-MM-DD`.")
            return

        entry["purchase_date"] = text
        state[user_id] = entry
        save_state(state)
        handle_final_portfolio_entry(chat_id, state)


def listen():
    """
    Main polling loop for handling incoming Telegram updates.

    :return: None – runs indefinitely, processes commands and input
    """
    offset = None
    print("🤖 Bot started. Waiting for commands...")
    while True:
        try:
            url = f"{BASE_URL}/getUpdates"
            if offset:
                url += f"?offset={offset}"
            response = requests.get(url)
            updates = response.json().get("result", [])

            for update in updates:
                offset = update["update_id"] + 1

                if "message" in update:
                    msg = update["message"]
                    chat_id = msg["chat"]["id"]
                    text = msg.get("text", "")
                    if text.startswith("/"):
                        handle_command(chat_id, text)
                    else:
                        handle_text_input(chat_id, text)

                elif "callback_query" in update:
                    cb = update["callback_query"]
                    chat_id = cb["from"]["id"]
                    data = cb["data"]
                    handle_callback_query(chat_id, data)

        except Exception as e:
            print("Error:", e)
        time.sleep(1)


if __name__ == "__main__":
    print("🔄 Starting bot...")
    listen()
