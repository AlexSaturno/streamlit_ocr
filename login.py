# https://docs.streamlit.io/knowledge-base/deploy/authentication-without-sso
import hmac
import streamlit as st


def hide_sidebar():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = None

    st.markdown(
        """
    <style>
        section[data-testid="stSidebar"][aria-expanded="true"]{
            display: none;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


def check_password():
    """Returns `True` if the user had a correct password."""
    hide_sidebar()

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Login", on_click=password_entered)

    def password_entered():
        st.session_state["password_correct"] = True

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", True):
        return True

    # Show inputs for username + password.
    st.title("Login")
    login_form()
    if st.session_state["password_correct"] is False:
        st.error("Usuário ou senha incorretos.")
    return False


if not check_password():
    st.stop()  # Do not continue if check_password is not True.

if st.session_state.get("password_correct", True):
    st.switch_page("pages/01_Processamento_Padrão.py")
