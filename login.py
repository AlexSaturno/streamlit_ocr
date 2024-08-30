# https://docs.streamlit.io/knowledge-base/deploy/authentication-without-sso
import streamlit as st

# Leitura do arquivo css de estilização
with open("./login-styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


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
            st.write("ㅤ")  # espaço em branco para logo
            st.title("FPO — Login")
            username = st.text_input("Username", key="username")
            st.session_state["logged_user"] = username
            st.text_input("Password", type="password", key="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                # Verificação se o usuário está dentro do padrão do banco (7 letras)
                if len(username) == 7:
                    st.session_state["password_correct"] = True
                else:
                    st.session_state["password_correct"] = False

    # Show inputs for username + password.
    login_form()

    # Return True if the username + password is validated.
    if st.session_state["password_correct"] is True:
        return True
    elif st.session_state["password_correct"] is False:
        st.error("Usuário ou senha incorretos.")
    return False


if not check_password():
    st.stop()  # Do not continue if check_password is not True.

if st.session_state.get("password_correct", True):
    st.switch_page("pages/01_Processamento_Padrão.py")
