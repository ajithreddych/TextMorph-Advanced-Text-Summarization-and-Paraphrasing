# paraphrase_routes.py
from flask import Blueprint, request, jsonify
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

paraphrase_bp = Blueprint("paraphrase", __name__)

# --- MODEL PATHS (relative to project root) ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_BASE = os.path.join(BASE_DIR, "training_pipeline", "datasets", "processed", "model_outputs")

MODEL_PATHS = {
    "t5-small": os.path.join(MODEL_BASE, "paraphrase_t5-small_20251105_062028"),
    "flan-t5-small": os.path.join(MODEL_BASE, "paraphrase_google_flan-t5-small_20251105_065026"),
    "bart-base": os.path.join(MODEL_BASE, "paraphrase_facebook_bart-base_20251105_071833"),
}

# Simple in-memory cache to avoid reloading models on every request
_LOADED_MODELS = {}

# --------------------- CHUNKING (same idea as summarization) ---------------------
def chunk_text(text, max_words=600):
    words = text.split()
    if len(words) <= max_words:
        return [text]
    step = max_words - 150  # overlap ~25%
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), step)]

# Map complexity to generation kwargs
def complexity_to_generation_kwargs(complexity: str):
    # complexity expected: creative, standard, basic
    complexity = (complexity or "standard").lower()
    if complexity == "creative":
        return {
            "temperature": 1.2,
            "top_k": 50,
            "top_p": 0.95,
            "num_beams": 4,
            "no_repeat_ngram_size": 3,
            "length_penalty": 1.0,
        }
    if complexity == "basic":
        return {
            "temperature": 0.6,
            "top_k": 0,
            "top_p": 0.9,
            "num_beams": 6,
            "no_repeat_ngram_size": 3,
            "length_penalty": 1.2,
        }
    # standard
    return {
        "temperature": 0.9,
        "top_k": 10,
        "top_p": 0.92,
        "num_beams": 5,
        "no_repeat_ngram_size": 3,
        "length_penalty": 1.1,
    }

def load_model_and_tokenizer(model_path):
    """Load and cache tokenizer+model for given model_path"""
    if model_path in _LOADED_MODELS:
        return _LOADED_MODELS[model_path]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to("cpu")
    _LOADED_MODELS[model_path] = (tokenizer, model)
    return tokenizer, model

# --------------------- MAIN ENDPOINT ---------------------
@paraphrase_bp.route("/paraphrase", methods=["POST"])
def paraphrase_text():
    """
    Expected JSON:
    {
      "text": "<input text>",
      "model_name": "t5-small" | "flan-t5-small" | "bart-base",
      "complexity": "creative" | "standard" | "basic"
    }
    Returns:
    {
      "paraphrase": "...",
      "input_length": int,
      "output_length": int
    }
    """
    try:
        data = request.get_json()
        text = (data.get("text") or "").strip()
        model_name = data.get("model_name", "t5-small")
        complexity = data.get("complexity", "standard")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        model_path = MODEL_PATHS.get(model_name)
        if not model_path or not os.path.exists(model_path):
            return jsonify({"error": f"Model {model_name} not found"}), 404

        tokenizer, model = load_model_and_tokenizer(model_path)

        # split into chunks for long input
        chunks = chunk_text(text, max_words=700)
        paraphrases = []

        gen_kwargs = complexity_to_generation_kwargs(complexity)

        for chunk in chunks:
            if not chunk.strip():
                continue
            inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=1024)
            input_len = len(inputs["input_ids"][0])

            # choose safe target lengths relative to input
            target_max = min(1024, max(64, int(input_len * 1.05)))
            target_min = max(20, int(target_max * 0.3))

            # prepare generate args (keep conservative defaults, override via gen_kwargs)
            generate_args = {
                "max_length": target_max,
                "min_length": target_min,
                "early_stopping": True,
                **{k: v for k, v in gen_kwargs.items() if k in ["temperature", "top_k", "top_p", "num_beams", "no_repeat_ngram_size", "length_penalty"]}
            }

            # Some tokenizers/models use `decoder_start_token_id` or require `attention_mask` - pass inputs dict
            outputs = model.generate(**inputs, **generate_args)
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            if decoded:
                paraphrases.append(decoded)

        if not paraphrases:
            return jsonify({"error": "No paraphrase generated."}), 500

        # If multiple paraphrase chunks, combine them (basic join). Optionally refine later.
        final_paraphrase = " ".join(paraphrases).strip()

        return jsonify({
            "paraphrase": final_paraphrase,
            "input_length": len(text.split()),
            "output_length": len(final_paraphrase.split())
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
