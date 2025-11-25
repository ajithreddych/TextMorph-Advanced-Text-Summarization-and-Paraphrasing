from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import jwt
from database import get_connection
from otp_utils import generate_otp, send_otp_email

app = Flask(__name__)

# ---------------- JWT CONFIG ----------------
SECRET_KEY = "your_super_secret_key"  # Replace with a strong secret
ALGORITHM = "HS256"
JWT_EXP_DELTA_MINUTES = 60  # Token valid for 1 hour

# ---------------- SIGNUP: Send OTP ----------------
@app.route("/auth/signup", methods=["POST"])
def signup():
    data = request.json
    name = data.get("name")
    email = data.get("email")
    age = data.get("age")
    dob = data.get("dob")
    gender = data.get("gender")
    password = data.get("password")

    if not all([name, email, age, dob, gender, password]):
        return jsonify({"status": "error", "message": "Fill all fields"}), 400

    otp_code = generate_otp()
    expires_at = datetime.now() + timedelta(minutes=5)

    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS otp_verification (
                otp_id INT AUTO_INCREMENT PRIMARY KEY,
                email VARCHAR(255),
                otp_code VARCHAR(10),
                purpose VARCHAR(50),
                expires_at DATETIME,
                is_verified TINYINT DEFAULT 0
            )
        """)

        cursor.execute("""
            INSERT INTO otp_verification (email, otp_code, purpose, expires_at)
            VALUES (%s, %s, %s, %s)
        """, (email, otp_code, 'signup', expires_at))
        conn.commit()

        if send_otp_email(email, otp_code):
            msg = "OTP sent to your email successfully."
        else:
            msg = "Failed to send OTP. Try again later."

        cursor.close()
        conn.close()
        return jsonify({"status": "ok", "message": msg}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ---------------- VERIFY SIGNUP (OTP) ----------------
@app.route("/auth/verify_signup", methods=["POST"])
def verify_signup():
    data = request.json
    name = data.get("name")
    email = data.get("email")
    age = data.get("age")
    dob = data.get("dob")
    gender = data.get("gender")
    password = data.get("password")
    otp_input = data.get("otp")

    if not all([otp_input, name, email, age, dob, gender, password]):
        return jsonify({"status": "error", "message": "Missing fields"}), 400

    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT * FROM otp_verification 
            WHERE email=%s AND otp_code=%s AND purpose='signup' AND is_verified=0
        """, (email, otp_input))
        record = cursor.fetchone()

        if not record:
            cursor.close()
            conn.close()
            return jsonify({"status": "error", "message": "Invalid OTP"}), 400

        if datetime.now() > record['expires_at']:
            cursor.close()
            conn.close()
            return jsonify({"status": "error", "message": "OTP expired"}), 400

        cursor.execute("UPDATE otp_verification SET is_verified=1 WHERE otp_id=%s", (record['otp_id'],))

        # Create users table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100),
                email VARCHAR(150) UNIQUE,
                age INT,
                dob DATE,
                gender ENUM('Male','Female','Other'),
                password_hash VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            INSERT INTO users (name, email, age, dob, gender, password_hash)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (name, email, age, dob, gender, password))
        conn.commit()

        cursor.close()
        conn.close()

        return jsonify({"status": "success", "message": "Account created successfully!"}), 201

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ---------------- LOGIN ----------------
@app.route("/auth/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")
    login_type = data.get("login_type", "user")  # frontend should send 'user' or 'admin'

    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        if login_type == "admin":
            cursor.execute("""
                SELECT u.*, a.role FROM users u
                JOIN admins a ON u.user_id = a.user_id
                WHERE u.email=%s AND u.password_hash=%s
            """, (email, password))
        else:
            cursor.execute("SELECT * FROM users WHERE email=%s AND password_hash=%s", (email, password))

        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user:
            # ---------------- GENERATE JWT ----------------
            payload = {
                "user_id": user["user_id"],
                "email": user["email"],
                "role": login_type,
                "exp": datetime.utcnow() + timedelta(minutes=JWT_EXP_DELTA_MINUTES)
            }
            token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
            return jsonify({"status": "success", "message": "Login successful", "role": login_type, "token": token, "user_id": user["user_id"]}), 200
        else:
            print(f"[LOGIN FAILED] email={email} type={login_type}")
            return jsonify({"status": "error", "message": "Invalid credentials"}), 401

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

# ---------------- FORGOT PASSWORD - SEND OTP ----------------
@app.route("/auth/forgot_password", methods=["POST"])
def forgot_password():
    data = request.json
    email = data.get("email")

    if not email:
        return jsonify({"status": "error", "message": "Email is required"}), 400

    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()
        if not user:
            cursor.close()
            conn.close()
            return jsonify({"status": "error", "message": "No account found with that email"}), 404

        otp_code = generate_otp()
        expires_at = datetime.now() + timedelta(minutes=5)

        cursor.execute("""
            INSERT INTO otp_verification (email, otp_code, purpose, expires_at)
            VALUES (%s, %s, %s, %s)
        """, (email, otp_code, 'forgot_password', expires_at))
        conn.commit()

        sent = send_otp_email(email, otp_code)
        cursor.close()
        conn.close()

        if sent:
            return jsonify({"status": "ok", "message": "OTP sent successfully to your email"}), 200
        else:
            return jsonify({"status": "error", "message": "Failed to send OTP"}), 500

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ---------------- FORGOT PASSWORD - VERIFY & RESET ----------------
@app.route("/auth/verify_forgot_password", methods=["POST"])
def verify_forgot_password():
    data = request.json
    email = data.get("email")
    new_password = data.get("new_password")
    otp_input = data.get("otp")

    if not all([email, new_password, otp_input]):
        return jsonify({"status": "error", "message": "Missing fields"}), 400

    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT * FROM otp_verification 
            WHERE email=%s AND otp_code=%s AND purpose='forgot_password' AND is_verified=0
        """, (email, otp_input))
        record = cursor.fetchone()

        if not record:
            cursor.close()
            conn.close()
            return jsonify({"status": "error", "message": "Invalid OTP"}), 400

        if datetime.now() > record["expires_at"]:
            cursor.close()
            conn.close()
            return jsonify({"status": "error", "message": "OTP expired"}), 400

        cursor.execute("UPDATE otp_verification SET is_verified=1 WHERE otp_id=%s", (record["otp_id"],))
        cursor.execute("UPDATE users SET password_hash=%s WHERE email=%s", (new_password, email))
        conn.commit()

        cursor.close()
        conn.close()
        return jsonify({"status": "success", "message": "Password reset successful"}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
