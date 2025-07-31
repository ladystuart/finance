import subprocess
import threading


def run_streamlit():
    subprocess.run(["streamlit", "run", "app.py"])


def run_bot():
    from bot.bot_listener import listen
    listen()


if __name__ == "__main__":
    threading.Thread(target=run_streamlit).start()
    run_bot()
