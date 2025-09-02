import streamlit as st
import requests
import textstat
import matplotlib.pyplot as plt
import docx
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import string

# Download nltk resources (first time only)
nltk.download("punkt", quiet=True)

API_URL = "http://127.0.0.1:5000"
st.set_page_config(page_title="Text Summarization using AI", layout="centered")

# ---------- SESSION STATE INIT ----------
for key, default in {
    "reg_otp_sent": False,
    "forgot_otp_sent": False,
    "page": "login",  # navigation state
    "token": None,
    "user": None,
    "confirm_edit": False,
    "confirm_delete": False,
    "confirm_logout": False,
    "show_history": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ------------------ API CALL WRAPPER ------------------
def api_call(method, endpoint, data=None, auth=False):
    headers = {}
    if auth and st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"

    try:
        if method == "POST":
            r = requests.post(API_URL + endpoint, json=data, headers=headers)
        elif method == "PUT":
            r = requests.put(API_URL + endpoint, json=data, headers=headers)
        elif method == "DELETE":
            r = requests.delete(API_URL + endpoint, headers=headers)
        else:
            r = requests.get(API_URL + endpoint, headers=headers)
        return r.json(), r.status_code
    except:
        return {"error": "Backend not reachable"}, 500


# ------------------ LOGIN PAGE ------------------
def login_page():
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Text Summarization using AI</h1>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🔑 Login", "🆕 Create Account"])

    # -------- Login --------
    with tab1:
        st.markdown("### Login to your account")
        email = st.text_input("📧 Email", key="login_email")
        password = st.text_input("🔒 Password", type="password", key="login_password")

        if st.button("Login", use_container_width=True):
            res, code = api_call("POST", "/login", {"email": email, "password": password})
            if code == 200:
                st.success("✅ Login successful!")
                st.session_state.token = res["token"]
                st.session_state.user = res["user"]
                st.session_state.page = "summarizer"
            else:
                st.error(res.get("error", "Login failed"))

        # -------- Forgot Password --------
        with st.expander("Forgot Password?"):
            forgot_email = st.text_input("Enter your registered email", key="forgot_email")

            if st.button("Send OTP", key="forgot_send_otp", use_container_width=True):
                res, code = api_call("POST", "/send-otp", {"email": forgot_email})
                if code == 200:
                    st.info(res.get("message"))
                    st.session_state.forgot_otp_sent = True
                else:
                    st.error(res.get("error", "Error"))

            if st.session_state.forgot_otp_sent:
                otp_forgot = st.text_input("Enter OTP", key="forgot_otp")
                new_password = st.text_input("Enter New Password", type="password", key="forgot_new_password")

                if st.button("Reset Password", use_container_width=True):
                    res, code = api_call("POST", "/reset-password",
                                         {"email": forgot_email, "otp": otp_forgot, "new_password": new_password})
                    if code == 200:
                        st.success(res.get("message"))
                        st.session_state.forgot_otp_sent = False
                    else:
                        st.error(res.get("error", "Error"))

    # -------- Register --------
    with tab2:
        st.markdown("### Create a new account")
        reg_name = st.text_input("👤 Full Name", key="reg_name")
        reg_email = st.text_input("📧 Email", key="reg_email")
        reg_password = st.text_input("🔒 Password", type="password", key="reg_password")
        reg_age = st.slider("🎂 Age", 10, 100, 18, key="reg_age")
        reg_gender = st.radio("⚧ Gender", ["Male", "Female", "Other"], key="reg_gender")

        if st.button("Send OTP", key="reg_send_otp", use_container_width=True):
            res, code = api_call("POST", "/send-otp", {"email": reg_email})
            if code == 200:
                st.info(res.get("message"))
                st.session_state.reg_otp_sent = True
            else:
                st.error(res.get("error", "Error"))

        if st.session_state.reg_otp_sent:
            reg_otp = st.text_input("Enter OTP", key="reg_otp")
            if st.button("Create Account", key="reg_create_account", use_container_width=True):
                res, code = api_call("POST", "/register",
                                     {"name": reg_name, "email": reg_email, "password": reg_password,
                                      "age": reg_age, "gender": reg_gender, "otp": reg_otp})
                if code == 201:
                    st.success("🎉 Account created successfully!")
                    st.session_state.reg_otp_sent = False
                else:
                    st.error(res.get("error", "Error"))


# ------------------ PROFILE PAGE ------------------
def profile_page():
    
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Profile</h1>", unsafe_allow_html=True)

    data, code = api_call("GET", "/profile", auth=True)
    if code != 200:
        st.error("Session expired, please login again.")
        st.session_state.page = "login"
        return
    user = data["user"]

    st.subheader("Overview")
    st.text(f"Name: {user['name']}")
    st.text(f"Email: {user['email']}")
    st.text(f"Age: {user['age']}")
    st.text(f"Gender: {user['gender']}")
    st.markdown("---")

    # --- Edit/Delete/Logout Confirmations ---
    if not st.session_state.confirm_edit:
        if st.button("✏️ Edit Profile"):
            st.session_state.confirm_edit = True
    else:
        st.warning("Are you sure you want to edit your profile?")
        col1, col2 = st.columns(2)
        if col1.button("✅ Yes"):
            st.session_state.page = "edit_profile"
            st.session_state.confirm_edit = False
        if col2.button("❌ Cancel"):
            st.session_state.confirm_edit = False

    if not st.session_state.confirm_delete:
        if st.button("🗑️ Delete Account"):
            st.session_state.confirm_delete = True
    else:
        st.error("This will permanently delete your account. Continue?")
        col1, col2 = st.columns(2)
        if col1.button("✅ Yes, Delete"):
            r, c = api_call("DELETE", "/delete-account", auth=True)
            if c == 200:
                st.success("Account deleted.")
                st.session_state.page = "login"
                st.session_state.token = None
                st.session_state.user = None
            else:
                st.error(r.get("error"))
            st.session_state.confirm_delete = False
        if col2.button("❌ Cancel"):
            st.session_state.confirm_delete = False

    if not st.session_state.confirm_logout:
        if st.button("🚪 Logout"):
            st.session_state.confirm_logout = True
    else:
        st.warning("Logout from this session?")
        col1, col2 = st.columns(2)
        if col1.button("✅ Yes, Logout"):
            st.session_state.page = "login"
            st.session_state.token = None
            st.session_state.user = None
            st.session_state.confirm_logout = False
        if col2.button("❌ Cancel"):
            st.session_state.confirm_logout = False


# ------------------ EDIT PROFILE ------------------
def edit_profile_page():
    
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Edit Profile</h1>", unsafe_allow_html=True)


    data, code = api_call("GET", "/profile", auth=True)
    if code != 200:
        st.error("Session expired.")
        st.session_state.page = "login"
        return
    user = data["user"]

    name = st.text_input("Full Name", value=user["name"])
    age = st.slider("Age", 10, 100, int(user["age"]))
    gender = st.radio("Gender", ["Male", "Female", "Other"],
                      index=["Male", "Female", "Other"].index(user["gender"]))
    password = st.text_input("New Password (optional)", type="password")

    if st.button("💾 Save Changes"):
        r, c = api_call("PUT", "/update-profile",
                        {"name": name, "age": age, "gender": gender, "password": password or None}, auth=True)
        if c == 200:
            st.success("Profile updated!")
            st.session_state.page = "profile"
        else:
            st.error(r.get("error"))

    if st.button("❌ Cancel"):
        st.session_state.page = "profile"


# ------------------ FILE UTILITIES ------------------
def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def summarize_text(text, max_sentences=5):
    sentences = sent_tokenize(text)
    if len(sentences) <= max_sentences:
        return text
    words = word_tokenize(text.lower())
    words = [w for w in words if w not in string.punctuation]
    freq = Counter(words)
    sentence_scores = {}
    for sent in sentences:
        for word in word_tokenize(sent.lower()):
            if word in freq:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + freq[word]
    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:max_sentences]
    return " ".join(top_sentences)


# ------------------ SUMMARIZER PAGE ------------------
def summarizer_page():
    
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Text Summarization & Readability</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])
    text = ""
    if uploaded_file is not None:
        try:
            if uploaded_file.type == "text/plain":
                text = uploaded_file.read().decode("utf-8")
            elif uploaded_file.type == "application/pdf":
                text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                text = extract_text_from_docx(uploaded_file)
        except:
            st.error("⚠️ Unable to read file. Please upload a valid text/pdf/docx file.")
    else:
        text = st.text_area("Or paste text here", height=200)

    if st.button("🔎 Analyze"):
        if not text.strip():
            st.error("Please upload or enter some text.")
            return

        # --- Summarization ---
        summary = summarize_text(text, max_sentences=7)
        st.subheader("📝 Summarized Text")
        st.write(summary)

        # --- Save & Get Readability from Backend ---
        res, code = api_call("POST", "/analyze", {"text": text}, auth=True)
        if code == 200:
            flesch = res["flesch_kincaid"]
            fog = res["gunning_fog"]
            smog = res["smog_index"]

            col1, col2, col3 = st.columns(3)
            col1.metric("Flesch-Kincaid", f"{flesch:.2f}")
            col2.metric("Gunning Fog", f"{fog:.2f}")
            col3.metric("SMOG Index", f"{smog:.2f}")

            # Visualization
            levels = ["Beginner", "Intermediate", "Advanced"]
            values = [flesch, fog * 6, smog * 10]
            colors = ["green", "yellow", "red"]
            fig, ax = plt.subplots()
            ax.bar(levels, values, color=colors)
            ax.set_ylabel("Score")
            ax.set_title("Readability Complexity")
            st.pyplot(fig)

        else:
            st.error(res.get("error", "Error"))


# ------------------ HISTORY PAGE ------------------
def history_page():
    
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Analysis History</h1>", unsafe_allow_html=True)

    res, code = api_call("GET", "/history", auth=True)
    if code != 200:
        st.error("Failed to fetch history.")
        return

    for i, item in enumerate(res):
        st.write(f"### Analysis {i+1} ({item['created_at']})")
        st.write("**Original (preview):** " + item["text"][:200] + "...")
        st.write(f"**Flesch:** {item['flesch_kincaid']:.2f}, **Fog:** {item['gunning_fog']:.2f}, **SMOG:** {item['smog_index']:.2f}")
        st.write(f"**Level:** {item['level']}")
        st.markdown("---")


# ------------------ ROUTER ------------------
def router():
    if st.session_state.page == "login":
        login_page()
    elif st.session_state.page == "summarizer":
        summarizer_page()
    elif st.session_state.page == "profile":
        profile_page()
    elif st.session_state.page == "edit_profile":
        edit_profile_page()
    elif st.session_state.page == "history":
        history_page()


# ------------------ SIDEBAR ------------------
if st.session_state.token:
    with st.sidebar:
        st.title("Text Summarization")
        if st.button("🏠 Summarizer"):
            st.session_state.page = "summarizer"
        if st.button("👤 Profile"):
            st.session_state.page = "profile"
        if st.button("📜 History"):
            st.session_state.page = "history"

router()
