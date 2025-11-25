import streamlit as st
from datetime import date
import requests


API_URL = "http://127.0.0.1:5000"  # Flask backend URL

# Import dashboards
import user
import admin

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Text Morph – Advanced Summarisation using AI",
    layout="wide"
)

# --- CSS ---
st.markdown("""
<style>
.stApp { background-color: #515D59; }
section[data-testid="stSidebar"] { background-color: #959F96; }
input, select, textarea { background-color: #959F96 !important; color: white !important; border-radius: 6px !important; }
.stButton > button { background-color: #CACCCI !important; color: white !important; border-radius: 8px !important; font-weight: bold !important; }
.stButton > button:hover { background-color: #a0a2a1 !important; color: white !important; }
label, .stMarkdown, .stSelectbox, .stNumberInput, .stTextInput, .stPasswordInput { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "role" not in st.session_state:
    st.session_state.role = "user"
if "page" not in st.session_state:
    st.session_state.page = "login"
if "token" not in st.session_state:
    st.session_state.token = None  # <-- store JWT here


# ---------------- LOGIN PAGE ----------------
def login_page():
    st.title("Text Morph – Advanced Summarisation using AI")
    st.subheader("Login or Create an Account")

    auth_option = st.radio("Choose action:", ["Login", "Create Account", "Forgot Password"], horizontal=True)
    st.session_state.auth_flow = auth_option.lower().replace(" ", "_")

    # ---------------- LOGIN ----------------
    if st.session_state.auth_flow == "login":
        login_type = st.radio("Login as:", ["User", "Admin"], horizontal=True)
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            payload = {
                "email": email,
                "password": password,
                "login_type": login_type.lower()
            }
            try:
                res = requests.post(f"{API_URL}/auth/login", json=payload)
                data = res.json()
                if res.status_code == 200:
                    st.session_state.logged_in = True
                    st.session_state.role = login_type.lower()
                    st.session_state.token = str(data.get("user_id") or data.get("token"))
                    st.session_state.user_id = data.get("user_id")
                    st.session_state.page = "dashboard"
                    st.success(data["message"])
                    st.rerun()
                else:
                    st.error(data["message"])
            except Exception as e:
                st.error(f"Server error: {e}")

    # ---------------- CREATE ACCOUNT ----------------
    elif st.session_state.auth_flow == "create_account":
        st.subheader("Create Account")
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        age = st.number_input("Age", 1, 120)
        dob = st.date_input("Date of Birth", date.today(),min_value=date(1900, 1, 1),max_value=date.today())
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        password = st.text_input("Password", type="password")
        otp_input = st.text_input("Enter OTP (after receiving email)")

        col1, col2 = st.columns(2)

        # --- Send OTP ---
        with col1:
            if st.button("Send OTP"):
                if not all([name, email, age, dob, gender, password]):
                    st.error("Fill all details first.")
                else:
                    payload = {
                        "name": name,
                        "email": email,
                        "age": age,
                        "dob": str(dob),
                        "gender": gender,
                        "password": password
                    }
                    try:
                        res = requests.post(f"{API_URL}/auth/signup", json=payload)
                        data = res.json()
                        if res.status_code == 200:
                            st.success("✅ OTP sent successfully to your email.")
                        else:
                            st.error(data["message"])
                    except Exception as e:
                        st.error(f"Server error: {e}")

        # --- Verify OTP ---
        with col2:
            if st.button("Verify & Create Account"):
                if not otp_input.strip():
                    st.error("Enter OTP to verify.")
                else:
                    payload = {
                        "name": name,
                        "email": email,
                        "age": age,
                        "dob": str(dob),
                        "gender": gender,
                        "password": password,
                        "otp": otp_input
                    }
                    try:
                        res = requests.post(f"{API_URL}/auth/verify_signup", json=payload)
                        data = res.json()
                        if res.status_code == 201:
                            st.success("🎉 Account created! You can now log in.")
                        else:
                            st.error(data["message"])
                    except Exception as e:
                        st.error(f"Server error: {e}")

    # ---------------- FORGOT PASSWORD ----------------
    elif st.session_state.auth_flow == "forgot_password":
        st.subheader("Forgot Password")
        email = st.text_input("Registered Email")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        otp_input = st.text_input("Enter OTP (if sent)")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Send OTP"):
                if not email.strip():
                    st.error("Enter your registered email")
                else:
                    try:
                        res = requests.post(f"{API_URL}/auth/forgot_password", json={"email": email})
                        data = res.json()
                        if res.status_code == 200:
                            st.success("✅ OTP sent successfully to your email.")
                        else:
                            st.error(data["message"])
                    except Exception as e:
                        st.error(f"Server error: {e}")

        with col2:
            if st.button("Verify & Reset Password"):
                if not all([otp_input, new_password, confirm_password]):
                    st.error("Fill all fields")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    try:
                        payload = {"email": email, "new_password": new_password, "otp": otp_input}
                        res = requests.post(f"{API_URL}/auth/verify_forgot_password", json=payload)
                        data = res.json()
                        if res.status_code == 200:
                            st.success("🎉 Password reset successful! You can now log in.")
                        else:
                            st.error(data["message"])
                    except Exception as e:
                        st.error(f"Server error: {e}")


# ---------------- MAIN ----------------
if not st.session_state.logged_in or st.session_state.page == "login":
    login_page()
else:
    if st.session_state.role == "admin":
        admin.admin_dashboard()  # <-- no token argument
    else:
        user.user_dashboard()    # <-- no token argument

