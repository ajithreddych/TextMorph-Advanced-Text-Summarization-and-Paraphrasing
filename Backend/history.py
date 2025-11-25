# Backend/history.py
from flask import Blueprint, request, jsonify
from database import get_connection as get_db_connection
import math
import traceback

history_bp = Blueprint("history", __name__)

def _parse_pagination_args():
    try:
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 20))
        if page < 1:
            page = 1
        if per_page < 1 or per_page > 500:
            per_page = 20
        return page, per_page
    except Exception:
        return 1, 20

# -----------------------
# Helper to fetch rows
# -----------------------
def _fetch_rows(query, params):
    conn = get_db_connection()
    if conn is None:
        return None, "Database connection failed"
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(query, params)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows, None
    except Exception as e:
        traceback.print_exc()
        try:
            cur.close()
            conn.close()
        except Exception:
            pass
        return None, str(e)

# -----------------------
# Summaries history
# -----------------------
@history_bp.route("/summaries", methods=["GET"])
def list_summaries():
    """
    Query params:
      - user_id (required)
      - page (optional, default 1)
      - per_page (optional, default 20)
      - q (optional) : search/query substring for input_text or summary_text
    Returns:
      JSON array of summary objects or {"error": "..."} on failure.
    """
    try:
        user_id = request.args.get("user_id", None)
        if not user_id:
            return jsonify({"error": "user_id query parameter is required"}), 400

        page, per_page = _parse_pagination_args()
        offset = (page - 1) * per_page

        q = request.args.get("q", "").strip()

        if q:
            # Basic LIKE search across input_text and summary_text (may be slow for long text)
            query = """
                SELECT summary_id, user_id, input_text, summary_text, model_name, size, content_category, created_at
                FROM summaries
                WHERE user_id = %s AND (input_text LIKE %s OR summary_text LIKE %s)
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """
            like_q = f"%{q}%"
            params = (user_id, like_q, like_q, per_page, offset)
        else:
            query = """
                SELECT summary_id, user_id, input_text, summary_text, model_name, size, content_category, created_at
                FROM summaries
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """
            params = (user_id, per_page, offset)

        rows, err = _fetch_rows(query, params)
        if err:
            return jsonify({"error": f"DB error: {err}"}), 500

        # also return total count for pagination convenience
        count_query = "SELECT COUNT(*) as cnt FROM summaries WHERE user_id = %s"
        count_rows, err2 = _fetch_rows(count_query, (user_id,))
        total = count_rows[0]["cnt"] if count_rows and len(count_rows) > 0 else None

        return jsonify({
            "data": rows,
            "page": page,
            "per_page": per_page,
            "total": total
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# -----------------------
# Paraphrases history
# -----------------------
@history_bp.route("/paraphrases", methods=["GET"])
def list_paraphrases():
    """
    Query params:
      - user_id (required)
      - page, per_page (optional)
      - q (optional) search
    """
    try:
        user_id = request.args.get("user_id", None)
        if not user_id:
            return jsonify({"error": "user_id query parameter is required"}), 400

        page, per_page = _parse_pagination_args()
        offset = (page - 1) * per_page

        q = request.args.get("q", "").strip()

        if q:
            query = """
                SELECT paraphrase_id, user_id, input_text, paraphrase_text, model_name, size, complexity, content_category, created_at
                FROM paraphrases
                WHERE user_id = %s AND (input_text LIKE %s OR paraphrase_text LIKE %s)
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """
            like_q = f"%{q}%"
            params = (user_id, like_q, like_q, per_page, offset)
        else:
            query = """
                SELECT paraphrase_id, user_id, input_text, paraphrase_text, model_name, size, complexity, content_category, created_at
                FROM paraphrases
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """
            params = (user_id, per_page, offset)

        rows, err = _fetch_rows(query, params)
        if err:
            return jsonify({"error": f"DB error: {err}"}), 500

        count_query = "SELECT COUNT(*) as cnt FROM paraphrases WHERE user_id = %s"
        count_rows, err2 = _fetch_rows(count_query, (user_id,))
        total = count_rows[0]["cnt"] if count_rows and len(count_rows) > 0 else None

        return jsonify({
            "data": rows,
            "page": page,
            "per_page": per_page,
            "total": total
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# -----------------------
# Single-item endpoints (optional)
# -----------------------
@history_bp.route("/summary/<int:summary_id>", methods=["GET"])
def get_summary(summary_id):
    try:
        user_id = request.args.get("user_id", None)
        if not user_id:
            return jsonify({"error": "user_id query parameter is required"}), 400

        query = """
            SELECT summary_id, user_id, input_text, summary_text, model_name, size, content_category, created_at
            FROM summaries
            WHERE summary_id = %s AND user_id = %s
            LIMIT 1
        """
        rows, err = _fetch_rows(query, (summary_id, user_id))
        if err:
            return jsonify({"error": f"DB error: {err}"}), 500
        if not rows:
            return jsonify({"error": "Summary not found"}), 404
        return jsonify(rows[0]), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@history_bp.route("/paraphrase/<int:paraphrase_id>", methods=["GET"])
def get_paraphrase(paraphrase_id):
    try:
        user_id = request.args.get("user_id", None)
        if not user_id:
            return jsonify({"error": "user_id query parameter is required"}), 400

        query = """
            SELECT paraphrase_id, user_id, input_text, paraphrase_text, model_name, size, complexity, content_category, created_at
            FROM paraphrases
            WHERE paraphrase_id = %s AND user_id = %s
            LIMIT 1
        """
        rows, err = _fetch_rows(query, (paraphrase_id, user_id))
        if err:
            return jsonify({"error": f"DB error: {err}"}), 500
        if not rows:
            return jsonify({"error": "Paraphrase not found"}), 404
        return jsonify(rows[0]), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
