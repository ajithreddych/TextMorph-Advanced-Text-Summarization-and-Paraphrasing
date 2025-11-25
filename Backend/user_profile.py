from flask import Blueprint, request, jsonify
import jwt
from database import get_connection
from datetime import datetime
from account import SECRET_KEY, ALGORITHM

profile_bp = Blueprint("profile_bp", __name__)

# ------------------- Helper: Decode JWT -------------------
def decode_token(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


# ------------------- GET PROFILE -------------------
@profile_bp.route("/profile/view", methods=["GET"])
def view_profile():
    token = request.headers.get("Authorization")
    if not token:
        return jsonify({"status": "error", "message": "Token missing"}), 401

    decoded = decode_token(token)
    if not decoded:
        return jsonify({"status": "error", "message": "Invalid or expired token"}), 401

    user_id = decoded["user_id"]
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT user_id, name, email, age, dob, gender, created_at FROM users WHERE user_id=%s", (user_id,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if not user:
            return jsonify({"status": "error", "message": "User not found"}), 404

        return jsonify({"status": "success", "user": user}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ------------------- UPDATE PROFILE -------------------
@profile_bp.route("/profile/update", methods=["PUT"])
def update_profile():
    token = request.headers.get("Authorization")
    data = request.json

    if not token:
        return jsonify({"status": "error", "message": "Token missing"}), 401

    decoded = decode_token(token)
    if not decoded:
        return jsonify({"status": "error", "message": "Invalid or expired token"}), 401

    user_id = decoded["user_id"]
    name = data.get("name")
    age = data.get("age")
    dob = data.get("dob")
    gender = data.get("gender")

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE users 
            SET name=%s, age=%s, dob=%s, gender=%s 
            WHERE user_id=%s
        """, (name, age, dob, gender, user_id))
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({"status": "success", "message": "Profile updated successfully"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ------------------- CHANGE PASSWORD -------------------
@profile_bp.route("/profile/change_password", methods=["PUT"])
def change_password():
    token = request.headers.get("Authorization")
    data = request.json
    old_password = data.get("old_password")
    new_password = data.get("new_password")

    if not token:
        return jsonify({"status": "error", "message": "Token missing"}), 401

    decoded = decode_token(token)
    if not decoded:
        return jsonify({"status": "error", "message": "Invalid or expired token"}), 401

    user_id = decoded["user_id"]

    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT password_hash FROM users WHERE user_id=%s", (user_id,))
        user = cursor.fetchone()

        if not user or user["password_hash"] != old_password:
            cursor.close()
            conn.close()
            return jsonify({"status": "error", "message": "Incorrect old password"}), 400

        cursor.execute("UPDATE users SET password_hash=%s WHERE user_id=%s", (new_password, user_id))
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({"status": "success", "message": "Password changed successfully"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ------------------- DELETE PROFILE -------------------
@profile_bp.route("/profile/delete", methods=["DELETE"])
def delete_profile():
    token = request.headers.get("Authorization")
    if not token:
        return jsonify({"status": "error", "message": "Token missing"}), 401

    decoded = decode_token(token)
    if not decoded:
        return jsonify({"status": "error", "message": "Invalid or expired token"}), 401

    user_id = decoded["user_id"]

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE user_id=%s", (user_id,))
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({"status": "success", "message": "Account deleted successfully"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
