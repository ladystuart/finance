import streamlit as st
import hashlib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASSWORD_FILE = os.path.join(BASE_DIR, "config", "password.txt")


def check_password():
    """
    Authenticates the user by checking the password against a stored hash.

    The function checks if the user is already authenticated via session state.
    If not, it verifies the entered password against a SHA-256 hash stored in a file.
    On first run, creates a password file with default hash for 'admin'.

    Returns:
        bool: True if authentication is successful, False otherwise.

    Side Effects:
        - Creates password file with default hash if not exists
        - Modifies session state upon successful authentication
        - Triggers app rerun on successful authentication
    """
    if "authenticated" in st.session_state and st.session_state["authenticated"]:
        return True

    if not os.path.exists(PASSWORD_FILE):
        with open(PASSWORD_FILE, "w") as f:
            f.write(hashlib.sha256("admin".encode()).hexdigest())
        st.warning("Password file made. Use password 'admin' for auth.")

    with open(PASSWORD_FILE, "r") as f:
        correct_hash = f.read().strip()

    with st.form("auth_form"):
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Enter")

        if submit_button:
            if hashlib.sha256(password.encode()).hexdigest() == correct_hash:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Incorrect password")
                return False

    return False
