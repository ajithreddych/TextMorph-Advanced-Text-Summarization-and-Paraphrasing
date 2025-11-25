
from flask import Blueprint, request, jsonify
from database import get_connection as get_db_connection
import traceback
import math
from datetime import datetime
admin_usage_bp = Blueprint("admin_usage", __name__)

try:
    from admin_view import is_admin_token 
except Exception:
    def is_admin_token(conn, token):
        return True


@admin_usage_bp.route("/usage", methods=["GET"])
def admin_usage():
    """
    GET /admin/usage
    Returns JSON with the following structure:
    {
      "totals": { total_users, total_summaries, total_paraphrases, total_feedbacks },
      "feedback_counts": { "up": n, "down": m },
      "feedback_rows": [ {feedback_id,user_id,user_name,user_email,task_type,task_id,rating,comment,created_at}, ... ],
      "user_feedback_stats": [ { user_id, user_name, user_email, feedback_count, up_count, down_count, last_feedback }, ... ],
      "breakdown": {
         "top_models": [ {"model": "t5-small", "count": 123}, ... ],
         "by_task": {"summarization": n, "paraphrasing": m},
         "top_users": [ {"user_id": X, "user_name": "A", "count": 50}, ... ]
      }
    }
    """
    auth = request.headers.get("Authorization", "")
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({"error": "Database connection failed"}), 500

        if not is_admin_token(conn, auth):
            conn.close()
            return jsonify({"error": "Unauthorized - admin only"}), 401

        cur = conn.cursor(dictionary=True)

        # Totals
        cur.execute("SELECT COUNT(*) AS cnt FROM users")
        total_users = cur.fetchone().get("cnt", 0)

        cur.execute("SELECT COUNT(*) AS cnt FROM summaries")
        total_summaries = cur.fetchone().get("cnt", 0)

        cur.execute("SELECT COUNT(*) AS cnt FROM paraphrases")
        total_paraphrases = cur.fetchone().get("cnt", 0)

        cur.execute("SELECT COUNT(*) AS cnt FROM feedback")
        total_feedbacks = cur.fetchone().get("cnt", 0)

        totals = {
            "total_users": total_users,
            "total_summaries": total_summaries,
            "total_paraphrases": total_paraphrases,
            "total_feedbacks": total_feedbacks
        }

        # Feedback counts (thumbs up/down)
        cur.execute("SELECT rating, COUNT(*) AS cnt FROM feedback GROUP BY rating")
        fb_counts_rows = cur.fetchall()
        fb_counts = {"up": 0, "down": 0}
        for r in fb_counts_rows:
            key = r.get("rating")
            cnt = r.get("cnt", 0)
            if key == "up":
                fb_counts["up"] = cnt
            elif key == "down":
                fb_counts["down"] = cnt

        # Recent feedback rows (join with users)
        cur.execute("""
            SELECT f.feedback_id, f.user_id, u.name as user_name, u.email as user_email,
                   f.task_type, f.task_id, f.rating, f.comment, f.created_at
            FROM feedback f
            LEFT JOIN users u ON f.user_id = u.user_id
            ORDER BY f.created_at DESC
            LIMIT 1000
        """)
        feedback_rows = []
        for r in cur.fetchall():
            feedback_rows.append({
                "feedback_id": r.get("feedback_id"),
                "user_id": r.get("user_id"),
                "user_name": r.get("user_name") or "",
                "user_email": r.get("user_email") or "",
                "task_type": r.get("task_type"),
                "task_id": r.get("task_id"),
                "rating": r.get("rating"),
                "comment": r.get("comment") or "",
                "created_at": r.get("created_at").isoformat() if r.get("created_at") is not None else None
            })

        # Per-user feedback aggregates
        cur.execute("""
            SELECT f.user_id, u.name as user_name, u.email as user_email,
                   COUNT(*) as feedback_count,
                   SUM(CASE WHEN f.rating = 'up' THEN 1 ELSE 0 END) as up_count,
                   SUM(CASE WHEN f.rating = 'down' THEN 1 ELSE 0 END) as down_count,
                   MAX(f.created_at) as last_feedback
            FROM feedback f
            LEFT JOIN users u ON f.user_id = u.user_id
            GROUP BY f.user_id
            ORDER BY feedback_count DESC
            LIMIT 200
        """)
        user_feedback_stats = []
        for r in cur.fetchall():
            user_feedback_stats.append({
                "user_id": r.get("user_id"),
                "user_name": r.get("user_name") or "",
                "user_email": r.get("user_email") or "",
                "feedback_count": int(r.get("feedback_count") or 0),
                "up_count": int(r.get("up_count") or 0),
                "down_count": int(r.get("down_count") or 0),
                "last_feedback": r.get("last_feedback").isoformat() if r.get("last_feedback") is not None else None
            })

        # Breakdown: top models used for summaries
        cur.execute("""
            SELECT COALESCE(model_name, 'Unknown') as model, COUNT(*) as cnt
            FROM summaries
            GROUP BY model_name
            ORDER BY cnt DESC
            LIMIT 20
        """)
        top_models = [{"model": r.get("model"), "count": int(r.get("cnt", 0))} for r in cur.fetchall()]

        # by_task counts (summaries vs paraphrases) - duplicates of totals but handy
        by_task = {"summarization": total_summaries, "paraphrasing": total_paraphrases}

        # top users by combined activity (summaries+paraphrases+feedback)
        cur.execute("""
            SELECT u.user_id, u.name as user_name, u.email as user_email,
                COALESCE(s.s_count,0) + COALESCE(p.p_count,0) + COALESCE(f.f_count,0) as total_actions
            FROM users u
            LEFT JOIN (
                SELECT user_id, COUNT(*) as s_count FROM summaries GROUP BY user_id
            ) s ON s.user_id = u.user_id
            LEFT JOIN (
                SELECT user_id, COUNT(*) as p_count FROM paraphrases GROUP BY user_id
            ) p ON p.user_id = u.user_id
            LEFT JOIN (
                SELECT user_id, COUNT(*) as f_count FROM feedback GROUP BY user_id
            ) f ON f.user_id = u.user_id
            ORDER BY total_actions DESC
            LIMIT 20
        """)
        top_users = []
        for r in cur.fetchall():
            top_users.append({
                "user_id": r.get("user_id"),
                "user_name": r.get("user_name") or "",
                "user_email": r.get("user_email") or "",
                "count": int(r.get("total_actions") or 0)
            })

        # Compose breakdown
        breakdown = {
            "top_models": top_models,
            "by_task": by_task,
            "top_users": top_users
        }

        # final result
        result = {
            "totals": totals,
            "feedback_counts": fb_counts,
            "feedback_rows": feedback_rows,
            "user_feedback_stats": user_feedback_stats,
            "breakdown": breakdown
        }

        cur.close()
        conn.close()
        return jsonify(result), 200

    except Exception as e:
        traceback.print_exc()
        try:
            if conn:
                conn.close()
        except Exception:
            pass

        if not isinstance(breakdown, dict):
            breakdown = {}
        # by_task fallback using totals
        breakdown_by_task = breakdown.get("by_task") if isinstance(breakdown.get("by_task"), dict) else {}
        breakdown_by_task.setdefault("summarization", int(total_summaries))
        breakdown_by_task.setdefault("paraphrasing", int(total_paraphrases))
        breakdown["by_task"] = {
            "summarization": int(breakdown_by_task.get("summarization", total_summaries)),
            "paraphrasing": int(breakdown_by_task.get("paraphrasing", total_paraphrases))
        }
        # top_models/top_users fallback to empty lists if not present
        breakdown["top_models"] = breakdown.get("top_models") or top_models
        breakdown["top_users"] = breakdown.get("top_users") or top_users

        return jsonify({"error": f"Server error: {str(e)}"}), 500
