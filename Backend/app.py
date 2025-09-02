from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import wraps
import datetime
import jwt

from db import get_db_connection
from email_utils import send_otp_email, generate_otp
from summarizer import analyze_readability

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────
# JWT config (use env vars in production!)
# ─────────────────────────────────────────
app.config["JWT_SECRET"] = "super-secret-key-change-this"
app.config["JWT_ALGO"] = "HS256"
app.config["JWT_EXPIRE_MIN"] = 60 * 24  # 24 hours


def create_token(user_id, email):
    payload = {
        "sub": user_id,
        "email": email,
        "iat": datetime.datetime.utcnow(),
        "exp": datetime.datetime.utcnow()
        + datetime.timedelta(minutes=app.config["JWT_EXPIRE_MIN"]),
    }
    return jwt.encode(
        payload, app.config["JWT_SECRET"], algorithm=app.config["JWT_ALGO"]
    )


def token_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid token"}), 401
        token = auth.split(" ", 1)[1].strip()
        try:
            payload = jwt.decode(
                token,
                app.config["JWT_SECRET"],
                algorithms=[app.config["JWT_ALGO"]],
            )
            request.user_id = payload["sub"]
            request.user_email = payload["email"]
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
        return fn(*args, **kwargs)

    return wrapper


# ─────────────────────────────────────────
# OTP + Auth
# ─────────────────────────────────────────
@app.route("/send-otp", methods=["POST"])
def send_otp():
    data = request.json or {}
    email = data.get("email")
    if not email:
        return jsonify({"error": "Email is required"}), 400

    otp = generate_otp()
    if send_otp_email(email, otp):
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO otps (email, otp_code) VALUES (%s, %s)", (email, otp))
        conn.commit()
        conn.close()
        return jsonify({"message": "OTP sent successfully"}), 200
    return jsonify({"error": "Failed to send OTP"}), 500


@app.route("/verify-otp", methods=["POST"])
def verify_otp():
    data = request.json or {}
    email = data.get("email")
    otp = data.get("otp")

    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute(
        "SELECT * FROM otps WHERE email=%s ORDER BY created_at DESC LIMIT 1", (email,)
    )
    rec = cur.fetchone()
    conn.close()

    if rec and rec["otp_code"] == otp:
        return jsonify({"message": "OTP verified"}), 200
    return jsonify({"error": "Invalid OTP"}), 400


@app.route("/register", methods=["POST"])
def register():
    data = request.json or {}
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")
    age = data.get("age")
    gender = data.get("gender")
    otp = data.get("otp")

    # Validate OTP
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute(
        "SELECT * FROM otps WHERE email=%s ORDER BY created_at DESC LIMIT 1", (email,)
    )
    rec = cur.fetchone()
    if not rec or rec["otp_code"] != otp:
        conn.close()
        return jsonify({"error": "Invalid OTP"}), 400

    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (name, email, password, age, gender, verified) VALUES (%s,%s,%s,%s,%s,%s)",
            (name, email, password, age, gender, True),
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({"error": "Email already exists or DB error"}), 400

    conn.close()
    return jsonify({"message": "Account created successfully"}), 201


@app.route("/login", methods=["POST"])
def login():
    data = request.json or {}
    email = data.get("email")
    password = data.get("password")

    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute(
        "SELECT id, name, email, age, gender, verified FROM users WHERE email=%s AND password=%s",
        (email, password),
    )
    user = cur.fetchone()
    conn.close()

    if not user:
        return jsonify({"error": "Invalid credentials"}), 401
    if not user["verified"]:
        return jsonify({"error": "Account not verified"}), 403

    token = create_token(user_id=user["id"], email=user["email"])
    return (
        jsonify({"message": "Login successful", "token": token, "user": user}),
        200,
    )


@app.route("/reset-password", methods=["POST"])
def reset_password():
    data = request.json or {}
    email = data.get("email")
    otp = data.get("otp")
    new_password = data.get("new_password")

    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute(
        "SELECT * FROM otps WHERE email=%s ORDER BY created_at DESC LIMIT 1", (email,)
    )
    rec = cur.fetchone()

    if not rec or rec["otp_code"] != otp:
        conn.close()
        return jsonify({"error": "Invalid OTP"}), 400

    cur = conn.cursor()
    cur.execute("UPDATE users SET password=%s WHERE email=%s", (new_password, email))
    conn.commit()
    conn.close()
    return jsonify({"message": "Password updated successfully"}), 200


# ─────────────────────────────────────────
# Profile (JWT-protected)
# ─────────────────────────────────────────
@app.route("/profile", methods=["GET"])
@token_required
def profile_get():
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute(
        "SELECT id, name, email, age, gender, verified FROM users WHERE id=%s",
        (request.user_id,),
    )
    user = cur.fetchone()
    conn.close()
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify({"user": user}), 200


@app.route("/update-profile", methods=["PUT"])
@token_required
def profile_update():
    data = request.json or {}
    name = data.get("name")
    age = data.get("age")
    gender = data.get("gender")
    password = data.get("password")  # optional

    sets, vals = [], []
    if name is not None:
        sets.append("name=%s")
        vals.append(name)
    if age is not None:
        sets.append("age=%s")
        vals.append(age)
    if gender is not None:
        sets.append("gender=%s")
        vals.append(gender)
    if password:
        sets.append("password=%s")
        vals.append(password)

    if not sets:
        return jsonify({"message": "Nothing to update"}), 200

    vals.append(request.user_id)
    sql = f"UPDATE users SET {', '.join(sets)} WHERE id=%s"
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(sql, tuple(vals))
    conn.commit()
    conn.close()
    return jsonify({"message": "Profile updated"}), 200


@app.route("/delete-account", methods=["DELETE"])
@token_required
def profile_delete():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM users WHERE id=%s", (request.user_id,))
    conn.commit()
    conn.close()
    return jsonify({"message": "Account deleted"}), 200


# ─────────────────────────────────────────
# Readability Analysis + History
# ─────────────────────────────────────────
@app.route("/analyze", methods=["POST"])
@token_required
def analyze():
    data = request.json
    text = data.get("text")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    scores = analyze_readability(text)

    # Save to DB
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO readability_history 
           (user_id, text, flesch_kincaid, gunning_fog, smog_index, reading_ease, level) 
           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
        (
            request.user_id,
            text,
            scores["flesch_kincaid"],
            scores["gunning_fog"],
            scores["smog_index"],
            scores["reading_ease"],
            scores["level"],
        ),
    )
    conn.commit()
    conn.close()

    return jsonify(scores)


@app.route("/history", methods=["GET"])
@token_required
def history():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """SELECT id, text, flesch_kincaid, gunning_fog, smog_index, reading_ease, level, created_at 
           FROM readability_history WHERE user_id = %s ORDER BY created_at DESC""",
        (request.user_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    return jsonify(rows)


# ─────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
