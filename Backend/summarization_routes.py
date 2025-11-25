from flask import Blueprint, request, jsonify
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

summarization_bp = Blueprint("summarization", __name__)

# --- MODEL PATHS ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_BASE = os.path.join(BASE_DIR, "training_pipeline", "datasets", "processed", "model_outputs")

MODEL_PATHS = {
    "t5-small": os.path.join(MODEL_BASE, "summarization_t5-small_20251104_165531"),
    "flan-t5-small": os.path.join(MODEL_BASE, "summarization_google_flan-t5-small_20251104_175708"),
    "bart-base": os.path.join(MODEL_BASE, "summarization_facebook_bart-base_20251104_190405"),
}

# --------------------- CHUNKING ---------------------
def chunk_text(text, max_words=600):
    """Split long documents into smaller overlapping chunks."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    step = max_words - 150  # overlap 25%
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), step)]

# --------------------- DYNAMIC LENGTH CONTROL ---------------------
def get_length_ratio(length_type):
    """Define output compression ratio for summary length."""
    ratios = {
        "short": 0.15,   # 15% of input
        "medium": 0.35,  # 35% of input
        "long": 0.55     # 55% of input
    }
    return ratios.get(length_type, 0.35)


# --------------------- MAIN ENDPOINT ---------------------
@summarization_bp.route("/summarize", methods=["POST"])
def summarize_text():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        model_name = data.get("model_name", "t5-small")
        length = data.get("length", "medium")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        model_path = MODEL_PATHS.get(model_name)
        if not model_path or not os.path.exists(model_path):
            return jsonify({"error": f"Model {model_name} not found"}), 404

        # Load model + tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.to("cpu")

        # Split text for long inputs
        chunks = chunk_text(text, max_words=700)
        summaries = []
        ratio = get_length_ratio(length)

        for chunk in chunks:
            if not chunk.strip():
                continue  # skip empty text parts

            inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=1024)
            input_length = len(inputs["input_ids"][0])

            # Compute adaptive summary length safely
            target_length = max(80, min(int(input_length * ratio * 1.8), 800))
            min_len = max(30, int(target_length * 0.5))

            summary_ids = model.generate(
                **inputs,
                max_length=target_length,
                min_length=min_len,
                num_beams=5,
                repetition_penalty=1.4,
                length_penalty=1.2,
                temperature=1.0,
                no_repeat_ngram_size=3,
                early_stopping=True
            )

            decoded = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
            if decoded:
                summaries.append(decoded)

        # If we got no valid summaries
        if not summaries:
            return jsonify({"error": "No valid summary generated."}), 500

        # Merge chunk summaries (refinement step)
        if len(summaries) > 1:
            combined_summary = " ".join(summaries).strip()

            # prevent too short or empty refinement inputs
            if not combined_summary or len(combined_summary.split()) < 20:
                final_summary = combined_summary
            else:
                refine_inputs = tokenizer(
                    combined_summary,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=1024
                )

                # dynamically choose refine max_length
                refine_input_len = len(refine_inputs["input_ids"][0])
                refine_max_len = min(900, int(refine_input_len * 1.2))
                refine_min_len = max(60, int(refine_max_len * 0.5))

                refine_ids = model.generate(
                    **refine_inputs,
                    max_length=refine_max_len,
                    min_length=refine_min_len,
                    num_beams=6,
                    length_penalty=1.1,
                    temperature=0.9,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )

                final_summary = tokenizer.decode(refine_ids[0], skip_special_tokens=True).strip()
        else:
            final_summary = summaries[0]

        return jsonify({
            "summary": final_summary,
            "summary_length": len(final_summary.split()),
            "input_length": len(text.split()),
            "compression_ratio": f"{(len(final_summary.split()) / max(1, len(text.split()))) * 100:.1f}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
