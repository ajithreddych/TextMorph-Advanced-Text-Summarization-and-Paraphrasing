from flask import Blueprint, request, jsonify
from database import get_connection as get_db_connection

feedback_bp = Blueprint("feedback", __name__)

@feedback_bp.route("/add", methods=["POST"])
def add_feedback():
    try:
        data = request.get_json() or {}
        user_id = data.get("user_id")
        task_type_in = (data.get("task_type") or "").strip()
        task_id_in = data.get("task_id")
        rating_in = (data.get("rating") or "").strip()
        comment = data.get("comment", "") or ""

        # --- Normalize task_type to DB ENUM ('summarization' or 'paraphrasing') ---
        # Accept common synonyms from frontend (e.g., "summary","summarize","paraphrase","paraphrasing")
        tt = task_type_in.lower()
        if tt in ["summarization", "summary", "summarize", "summaries"]:
            task_type = "summarization"
        elif tt in ["paraphrasing", "paraphrase", "paraphrases", "paraphrased"]:
            task_type = "paraphrasing"
        else:
            # If nothing recognizable was passed, return helpful error
            return jsonify({"error": f"Invalid task_type '{task_type_in}'. Expected 'summarization' or 'paraphrasing'."}), 400

        # --- Normalize rating to DB ENUM ('up' or 'down') ---
        r = rating_in.lower()
        if r in ["up", "👍", "thumbs_up", "thumbsup", "like", "positive"]:
            rating = "up"
        elif r in ["down", "👎", "thumbs_down", "dislike", "negative"]:
            rating = "down"
        else:
            return jsonify({"error": f"Invalid rating '{rating_in}'. Expected 'up' or 'down'."}), 400

        # --- Validate user_id and task_id presence and types ---
        try:
            user_id = int(user_id)
        except Exception:
            return jsonify({"error": "Invalid or missing user_id (must be integer)."}), 400

        # task_id may be 0 if not yet saved; ensure integer
        try:
            task_id = int(task_id_in)
        except Exception:
            return jsonify({"error": "Invalid or missing task_id (must be integer)."}), 400

        # All required fields OK — insert into DB
        conn = get_db_connection()
        if conn is None:
            return jsonify({"error": "Database connection failed"}), 500
        cur = conn.cursor()

        query = """
            INSERT INTO feedback (user_id, task_type, task_id, rating, comment)
            VALUES (%s, %s, %s, %s, %s)
        """
        cur.execute(query, (user_id, task_type, task_id, rating, comment))
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"message": "Feedback added successfully!"}), 201

    except Exception as e:
        import traceback
        traceback.print_exc()
        # Provide clearer error when enum mismatch occurs
        err_str = str(e)
        if "Data truncated for column 'task_type'" in err_str or "1265" in err_str:
            return jsonify({"error": "Database enum mismatch for 'task_type'. Allowed values: 'summarization','paraphrasing'."}), 500
        return jsonify({"error": f"Server error: {e}"}), 500



@feedback_bp.route("/summary/save", methods=["POST"])
def save_summary():
    try:
        data = request.get_json()
        user_id = data.get("user_id")
        input_text = data.get("input_text", "")
        summary_text = data.get("summary_text", "")
        model_name = data.get("model_name", "")
        size = data.get("size", "Medium").capitalize()

        # ✅ Map frontend summary length to database ENUM
        if size.lower() == "short":
            size = "Small"
        elif size.lower() == "medium":
            size = "Medium"
        elif size.lower() == "long":
            size = "Large"

        if not all([user_id, summary_text]):
            return jsonify({"error": "Missing required data"}), 400

        conn = get_db_connection()
        if conn is None:
            return jsonify({"error": "Database connection failed"}), 500
        cur = conn.cursor()

        query = """
            INSERT INTO summaries (user_id, input_text, summary_text, model_name, size)
            VALUES (%s, %s, %s, %s, %s)
        """
        cur.execute(query, (user_id, input_text, summary_text, model_name, size))
        conn.commit()
        summary_id = cur.lastrowid
        cur.close()
        conn.close()
        return jsonify({"message": "Summary saved successfully", "summary_id": summary_id}), 201

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server error: {e}"}), 500




# --------------------- New Endpoint: Save Paraphrase (REPLACEMENT) ---------------------
@feedback_bp.route("/paraphrase/save", methods=["POST"])
def save_paraphrase():
    """
    Expected JSON:
    {
        "user_id": int,
        "input_text": str,
        "paraphrase_text": str,
        "model_name": str,
        "size": str,        # frontend may pass "Creative"/"Standard"/"Basic" or "Small"/"Medium"/"Large"
        "complexity": str   # frontend may pass "creative"/"standard"/"basic"
    }
    This function will map those values to DB ENUMs before inserting.
    """
    try:
        data = request.get_json() or {}

        user_id = data.get("user_id")
        input_text = data.get("input_text", "")
        paraphrase_text = data.get("paraphrase_text", "")
        model_name = data.get("model_name", "")
        size_in = (data.get("size") or "").strip()
        complexity_in = (data.get("complexity") or "").strip()

        # Basic input validation
        if not user_id or not paraphrase_text:
            return jsonify({"error": "Missing required data (user_id and paraphrase_text are required)"}), 400

        # Normalize incoming values (allow a few synonyms)
        # Map arbitrary frontend values to DB-enum for size: Small / Medium / Large
        size_map = {
            "creative": "Large",
            "creative_mode": "Large",
            "large": "Large",
            "standard": "Medium",
            "medium": "Medium",
            "basic": "Small",
            "basic_mode": "Small",
            "small": "Small",
            # also if user already passed Small/Medium/Large (case-insensitive)
            "small": "Small",
            "medium": "Medium",
            "large": "Large"
        }
        # Map incoming complexity to DB complexity enum: Simple / Standard / Creative
        complexity_map = {
            "creative": "Creative",
            "creative_mode": "Creative",
            "standard": "Standard",
            "basic": "Simple",
            "simple": "Simple",
            "simple_mode": "Simple"
        }

        # Lower-case lookup keys
        mapped_size = None
        if size_in:
            mapped_size = size_map.get(size_in.strip().lower())
        if not mapped_size and complexity_in:
            # if frontend didn't supply an explicit 'size', try mapping from complexity
            mapped_size = size_map.get(complexity_in.strip().lower())

        # Fallback to Medium if we couldn't map
        if not mapped_size:
            mapped_size = "Medium"

        mapped_complexity = complexity_map.get(complexity_in.strip().lower()) if complexity_in else None
        if not mapped_complexity:
            # if not provided, infer from size_in as a last resort
            # e.g., Large -> Creative, Medium -> Standard, Small -> Simple
            if mapped_size == "Large":
                mapped_complexity = "Creative"
            elif mapped_size == "Small":
                mapped_complexity = "Simple"
            else:
                mapped_complexity = "Standard"

        # Ensure values conform to DB ENUMs precisely
        size = mapped_size  # "Small"|"Medium"|"Large"
        complexity = mapped_complexity  # "Simple"|"Standard"|"Creative"

        conn = get_db_connection()
        if conn is None:
            return jsonify({"error": "Database connection failed"}), 500

        cur = conn.cursor()

        query = """
            INSERT INTO paraphrases (user_id, input_text, paraphrase_text, model_name, size, complexity)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        cur.execute(query, (user_id, input_text, paraphrase_text, model_name, size, complexity))
        conn.commit()
        paraphrase_id = cur.lastrowid
        cur.close()
        conn.close()

        return jsonify({"message": "Paraphrase saved successfully", "paraphrase_id": paraphrase_id}), 201

    except Exception as e:
        import traceback
        traceback.print_exc()
        # If DB error indicates enum truncation or similar, return helpful error
        err_str = str(e)
        if "Data truncated for column 'size'" in err_str or "1265" in err_str:
            return jsonify({"error": "Database enum mismatch for 'size' — frontend sent an unsupported value. Expected Small/Medium/Large."}), 500
        return jsonify({"error": f"Server error: {e}"}), 500
