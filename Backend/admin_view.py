# backend/admin_view.py
from flask import Blueprint, request, jsonify
from database import get_connection as get_db_connection
import math
import traceback

admin_bp = Blueprint("admin_view", __name__)

# ---------------------------
# is_admin_token helper (use your existing version; this is a compact one)
# ---------------------------
def is_admin_token(conn, token: str) -> bool:
    if not token:
        return False
    token = token.strip()
    if token.lower().startswith("bearer "):
        token = token.split(None, 1)[1]
    try:
        cur = conn.cursor(dictionary=True)
        if token.isdigit():
            cur.execute("SELECT a.user_id FROM admins a WHERE a.user_id = %s", (int(token),))
            row = cur.fetchone()
            cur.close()
            return bool(row)
        if "@" in token and token.count("@") == 1:
            cur.execute("""
                SELECT u.user_id FROM users u
                JOIN admins a ON u.user_id = a.user_id
                WHERE u.email = %s
            """, (token,))
            row = cur.fetchone()
            cur.close()
            return bool(row)
        if token.count('.') == 2:
            try:
                import jwt
                payload = jwt.decode(token, options={"verify_signature": False, "verify_exp": False})
                if "email" in payload:
                    cur.execute("""
                        SELECT u.user_id FROM users u
                        JOIN admins a ON u.user_id = a.user_id
                        WHERE u.email = %s
                    """, (payload["email"],))
                    row = cur.fetchone()
                    cur.close()
                    return bool(row)
                for key in ("user_id", "uid", "sub"):
                    if key in payload and str(payload[key]).isdigit():
                        cur.execute("SELECT a.user_id FROM admins a WHERE a.user_id = %s", (int(payload[key]),))
                        row = cur.fetchone()
                        if row:
                            cur.close()
                            return True
            except Exception:
                pass
        cur.execute("""
            SELECT u.user_id FROM users u
            JOIN admins a ON u.user_id = a.user_id
            WHERE u.password_hash = %s
        """, (token,))
        row = cur.fetchone()
        cur.close()
        return bool(row)
    except Exception:
        try:
            cur.close()
        except Exception:
            pass
        return False


# ---------------------------
# GET /admin/summaries
# ---------------------------
@admin_bp.route("/summaries", methods=["GET"])
def list_summaries():
    try:
        auth = request.headers.get("Authorization", "")
        conn = get_db_connection()
        if conn is None:
            return jsonify({"error": "Database connection failed"}), 500
        if not is_admin_token(conn, auth):
            conn.close()
            return jsonify({"error": "Unauthorized - admin only"}), 401

        q = request.args.get("q", "").strip()
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 50))
        if page < 1:
            page = 1
        if per_page < 1 or per_page > 500:
            per_page = 50

        cur = conn.cursor(dictionary=True)
        base_sql = " FROM summaries s LEFT JOIN users u ON s.user_id = u.user_id "
        where_clauses = []
        params = []
        if q:
            like_q = f"%{q}%"
            where_clauses.append("(s.input_text LIKE %s OR s.summary_text LIKE %s OR s.model_name LIKE %s OR u.name LIKE %s OR u.email LIKE %s)")
            params.extend([like_q, like_q, like_q, like_q, like_q])
        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        count_sql = "SELECT COUNT(*) as cnt " + base_sql + " " + where_sql
        cur.execute(count_sql, tuple(params))
        total = cur.fetchone().get("cnt", 0)
        offset = (page - 1) * per_page
        select_sql = f"""
            SELECT s.summary_id, s.user_id, u.name as user_name, u.email as user_email,
                   s.input_text, s.summary_text, s.model_name, s.size, s.created_at
            {base_sql}
            {where_sql}
            ORDER BY s.created_at DESC
            LIMIT %s OFFSET %s
        """
        cur.execute(select_sql, tuple(params) + (per_page, offset))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        summaries = []
        for r in rows:
            summaries.append({
                "summary_id": r.get("summary_id"),
                "user_id": r.get("user_id"),
                "user_name": r.get("user_name"),
                "user_email": r.get("user_email"),
                "input_text": r.get("input_text") or "",
                "summary_text": r.get("summary_text") or "",
                "model_name": r.get("model_name") or "",
                "size": r.get("size") or "",
                "created_at": r.get("created_at").isoformat() if r.get("created_at") is not None else None
            })

        return jsonify({
            "summaries": summaries,
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": math.ceil(total / per_page) if per_page else 0
        }), 200

    except Exception as e:
        traceback.print_exc()
        try:
            conn.close()
        except Exception:
            pass
        return jsonify({"error": f"Server error: {str(e)}"}), 500


# ---------------------------
# PUT /admin/summary/<id>
# ---------------------------
@admin_bp.route("/summary/<int:summary_id>", methods=["PUT"])
def update_summary(summary_id):
    try:
        auth = request.headers.get("Authorization", "")
        conn = get_db_connection()
        if conn is None:
            return jsonify({"error": "Database connection failed"}), 500
        if not is_admin_token(conn, auth):
            conn.close()
            return jsonify({"error": "Unauthorized - admin only"}), 401

        data = request.get_json() or {}
        summary_text = data.get("summary_text", None)
        model_name = data.get("model_name", None)
        size = data.get("size", None)
        if summary_text is None and model_name is None and size is None:
            conn.close()
            return jsonify({"error": "No updatable fields provided"}), 400

        fields = []
        params = []
        if summary_text is not None:
            fields.append("summary_text = %s"); params.append(summary_text)
        if model_name is not None:
            fields.append("model_name = %s"); params.append(model_name)
        if size is not None:
            if size not in ("Small", "Medium", "Large"):
                conn.close()
                return jsonify({"error": "Invalid size value. Allowed: Small, Medium, Large."}), 400
            fields.append("size = %s"); params.append(size)
        params.append(summary_id)
        cur = conn.cursor()
        sql = f"UPDATE summaries SET {', '.join(fields)} WHERE summary_id = %s"
        cur.execute(sql, tuple(params))
        conn.commit()
        affected = cur.rowcount
        cur.close()
        conn.close()
        if affected == 0:
            return jsonify({"error": "Summary not found"}), 404
        return jsonify({"message": "Summary updated", "summary_id": summary_id}), 200

    except Exception as e:
        traceback.print_exc()
        try:
            conn.close()
        except Exception:
            pass
        return jsonify({"error": f"Server error: {str(e)}"}), 500


# ---------------------------
# DELETE /admin/summary/<id>
# ---------------------------
@admin_bp.route("/summary/<int:summary_id>", methods=["DELETE"])
def delete_summary(summary_id):
    try:
        auth = request.headers.get("Authorization", "")
        conn = get_db_connection()
        if conn is None:
            return jsonify({"error": "Database connection failed"}), 500
        if not is_admin_token(conn, auth):
            conn.close()
            return jsonify({"error": "Unauthorized - admin only"}), 401

        cur = conn.cursor()
        cur.execute("DELETE FROM summaries WHERE summary_id = %s", (summary_id,))
        conn.commit()
        affected = cur.rowcount
        cur.close()
        conn.close()

        if affected == 0:
            return jsonify({"error": "Summary not found"}), 404
        return jsonify({"message": "Summary deleted", "summary_id": summary_id}), 200

    except Exception as e:
        traceback.print_exc()
        try:
            conn.close()
        except Exception:
            pass
        return jsonify({"error": f"Server error: {str(e)}"}), 500


# ---------------------------
# GET /admin/paraphrases
# ---------------------------
@admin_bp.route("/paraphrases", methods=["GET"])
def list_paraphrases():
    try:
        auth = request.headers.get("Authorization", "")
        conn = get_db_connection()
        if conn is None:
            return jsonify({"error": "Database connection failed"}), 500
        if not is_admin_token(conn, auth):
            conn.close()
            return jsonify({"error": "Unauthorized - admin only"}), 401

        q = request.args.get("q", "").strip()
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 50))
        if page < 1:
            page = 1
        if per_page < 1 or per_page > 500:
            per_page = 50

        cur = conn.cursor(dictionary=True)
        base_sql = " FROM paraphrases p LEFT JOIN users u ON p.user_id = u.user_id "
        where_clauses = []
        params = []
        if q:
            like_q = f"%{q}%"
            where_clauses.append("(p.input_text LIKE %s OR p.paraphrase_text LIKE %s OR p.model_name LIKE %s OR u.name LIKE %s OR u.email LIKE %s)")
            params.extend([like_q, like_q, like_q, like_q, like_q])
        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        count_sql = "SELECT COUNT(*) as cnt " + base_sql + " " + where_sql
        cur.execute(count_sql, tuple(params))
        total = cur.fetchone().get("cnt", 0)
        offset = (page - 1) * per_page
        select_sql = f"""
            SELECT p.paraphrase_id, p.user_id, u.name as user_name, u.email as user_email,
                   p.input_text, p.paraphrase_text, p.model_name, p.size, p.complexity, p.created_at
            {base_sql}
            {where_sql}
            ORDER BY p.created_at DESC
            LIMIT %s OFFSET %s
        """
        cur.execute(select_sql, tuple(params) + (per_page, offset))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        paraphrases = []
        for r in rows:
            paraphrases.append({
                "paraphrase_id": r.get("paraphrase_id"),
                "user_id": r.get("user_id"),
                "user_name": r.get("user_name"),
                "user_email": r.get("user_email"),
                "input_text": r.get("input_text") or "",
                "paraphrase_text": r.get("paraphrase_text") or "",
                "model_name": r.get("model_name") or "",
                "size": r.get("size") or "",
                "complexity": r.get("complexity") or "",
                "created_at": r.get("created_at").isoformat() if r.get("created_at") is not None else None
            })

        return jsonify({
            "paraphrases": paraphrases,
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": math.ceil(total / per_page) if per_page else 0
        }), 200

    except Exception as e:
        traceback.print_exc()
        try:
            conn.close()
        except Exception:
            pass
        return jsonify({"error": f"Server error: {str(e)}"}), 500


# ---------------------------
# PUT /admin/paraphrase/<id>
# ---------------------------
@admin_bp.route("/paraphrase/<int:paraphrase_id>", methods=["PUT"])
def update_paraphrase(paraphrase_id):
    try:
        auth = request.headers.get("Authorization", "")
        conn = get_db_connection()
        if conn is None:
            return jsonify({"error": "Database connection failed"}), 500
        if not is_admin_token(conn, auth):
            conn.close()
            return jsonify({"error": "Unauthorized - admin only"}), 401

        data = request.get_json() or {}
        paraphrase_text = data.get("paraphrase_text", None)
        model_name = data.get("model_name", None)
        size = data.get("size", None)
        complexity = data.get("complexity", None)

        if paraphrase_text is None and model_name is None and size is None and complexity is None:
            conn.close()
            return jsonify({"error": "No updatable fields provided"}), 400

        # validate enums if provided
        if size is not None and size not in ("Small", "Medium", "Large"):
            conn.close()
            return jsonify({"error": "Invalid size value. Allowed: Small, Medium, Large."}), 400
        if complexity is not None and complexity not in ("Simple", "Standard", "Creative"):
            conn.close()
            return jsonify({"error": "Invalid complexity. Allowed: Simple, Standard, Creative."}), 400

        fields = []
        params = []
        if paraphrase_text is not None:
            fields.append("paraphrase_text = %s"); params.append(paraphrase_text)
        if model_name is not None:
            fields.append("model_name = %s"); params.append(model_name)
        if size is not None:
            fields.append("size = %s"); params.append(size)
        if complexity is not None:
            fields.append("complexity = %s"); params.append(complexity)
        params.append(paraphrase_id)

        cur = conn.cursor()
        sql = f"UPDATE paraphrases SET {', '.join(fields)} WHERE paraphrase_id = %s"
        cur.execute(sql, tuple(params))
        conn.commit()
        affected = cur.rowcount
        cur.close()
        conn.close()

        if affected == 0:
            return jsonify({"error": "Paraphrase not found"}), 404
        return jsonify({"message": "Paraphrase updated", "paraphrase_id": paraphrase_id}), 200

    except Exception as e:
        traceback.print_exc()
        try:
            conn.close()
        except Exception:
            pass
        return jsonify({"error": f"Server error: {str(e)}"}), 500


# ---------------------------
# DELETE /admin/paraphrase/<id>
# ---------------------------
@admin_bp.route("/paraphrase/<int:paraphrase_id>", methods=["DELETE"])
def delete_paraphrase(paraphrase_id):
    try:
        auth = request.headers.get("Authorization", "")
        conn = get_db_connection()
        if conn is None:
            return jsonify({"error": "Database connection failed"}), 500
        if not is_admin_token(conn, auth):
            conn.close()
            return jsonify({"error": "Unauthorized - admin only"}), 401

        cur = conn.cursor()
        cur.execute("DELETE FROM paraphrases WHERE paraphrase_id = %s", (paraphrase_id,))
        conn.commit()
        affected = cur.rowcount
        cur.close()
        conn.close()

        if affected == 0:
            return jsonify({"error": "Paraphrase not found"}), 404
        return jsonify({"message": "Paraphrase deleted", "paraphrase_id": paraphrase_id}), 200

    except Exception as e:
        traceback.print_exc()
        try:
            conn.close()
        except Exception:
            pass
        return jsonify({"error": f"Server error: {str(e)}"}), 500


# ---- add this GET /admin/usage endpoint into backend/admin_view.py ----
@admin_bp.route("/usage", methods=["GET"])
def usage_stats():
    """
    Returns aggregated usage and feedback stats for admin UI.
    Response JSON:
    {
      "totals": { "total_users": int, "total_summaries": int, "total_paraphrases": int, "total_feedbacks": int },
      "feedback_counts": { "up": int, "down": int },
      "feedback_rows": [ { feedback_id, user_id, user_name, user_email, task_type, task_id, rating, comment, created_at }, ... ],
      "user_feedback_stats": [ { user_id, user_name, user_email, feedback_count, up_count, down_count, last_feedback }, ... ],
      "breakdown": { ... }
    }
    """
    try:
        auth = request.headers.get("Authorization", "")
        conn = get_db_connection()
        if conn is None:
            return jsonify({"error": "Database connection failed"}), 500
        if not is_admin_token(conn, auth):
            conn.close()
            return jsonify({"error": "Unauthorized - admin only"}), 401

        cur = conn.cursor(dictionary=True)

        # totals
        cur.execute("SELECT COUNT(*) AS cnt FROM users")
        total_users = cur.fetchone().get("cnt", 0)
        cur.execute("SELECT COUNT(*) AS cnt FROM summaries")
        total_summaries = cur.fetchone().get("cnt", 0)
        cur.execute("SELECT COUNT(*) AS cnt FROM paraphrases")
        total_paraphrases = cur.fetchone().get("cnt", 0)
        cur.execute("SELECT COUNT(*) AS cnt FROM feedback")
        total_feedbacks = cur.fetchone().get("cnt", 0)

        # feedback counts (up/down)
        cur.execute("SELECT rating, COUNT(*) as cnt FROM feedback GROUP BY rating")
        fc = cur.fetchall()
        feedback_counts = {"up": 0, "down": 0}
        for row in fc:
            rating = row.get("rating")
            cnt = row.get("cnt", 0)
            if rating == "up":
                feedback_counts["up"] = cnt
            elif rating == "down":
                feedback_counts["down"] = cnt

        # recent feedback rows (join users for name/email)
        cur.execute("""
            SELECT f.feedback_id, f.user_id, u.name as user_name, u.email as user_email,
                   f.task_type, f.task_id, f.rating, f.comment, f.created_at
            FROM feedback f
            LEFT JOIN users u ON f.user_id = u.user_id
            ORDER BY f.created_at DESC
            LIMIT 1000
        """)
        feedback_rows = cur.fetchall()
        # normalize datetimes to iso
        for r in feedback_rows:
            if r.get("created_at") is not None:
                r["created_at"] = r["created_at"].isoformat()

        # user-level feedback stats
        cur.execute("""
            SELECT u.user_id, u.name as user_name, u.email as user_email,
                   COUNT(f.feedback_id) as feedback_count,
                   SUM(CASE WHEN f.rating='up' THEN 1 ELSE 0 END) as up_count,
                   SUM(CASE WHEN f.rating='down' THEN 1 ELSE 0 END) as down_count,
                   MAX(f.created_at) as last_feedback
            FROM users u
            LEFT JOIN feedback f ON u.user_id = f.user_id
            GROUP BY u.user_id, u.name, u.email
            HAVING feedback_count > 0
            ORDER BY feedback_count DESC, last_feedback DESC
            LIMIT 500
        """)
        user_stats_raw = cur.fetchall()
        # normalize datetime
        user_feedback_stats = []
        for u in user_stats_raw:
            last = u.get("last_feedback")
            user_feedback_stats.append({
                "user_id": u.get("user_id"),
                "user_name": u.get("user_name"),
                "user_email": u.get("user_email"),
                "feedback_count": int(u.get("feedback_count") or 0),
                "up_count": int(u.get("up_count") or 0),
                "down_count": int(u.get("down_count") or 0),
                "last_feedback": last.isoformat() if last is not None else None
            })

        # optional breakdown by task_type (summarization / paraphrasing)
        cur.execute("""
            SELECT f.task_type, COUNT(*) as cnt
            FROM feedback f
            GROUP BY f.task_type
        """)
        breakdown_rows = cur.fetchall()
        breakdown = {r.get("task_type"): r.get("cnt", 0) for r in breakdown_rows}

        cur.close()
        conn.close()

        return jsonify({
            "totals": {
                "total_users": int(total_users),
                "total_summaries": int(total_summaries),
                "total_paraphrases": int(total_paraphrases),
                "total_feedbacks": int(total_feedbacks)
            },
            "feedback_counts": feedback_counts,
            "feedback_rows": feedback_rows,
            "user_feedback_stats": user_feedback_stats,
            "breakdown": breakdown
        }), 200

    except Exception as e:
        traceback.print_exc()
        try:
            conn.close()
        except Exception:
            pass
        return jsonify({"error": f"Server error: {str(e)}"}), 500
