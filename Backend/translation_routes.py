from flask import Blueprint, request, jsonify
from transformers import pipeline
from deep_translator import GoogleTranslator

translate_bp = Blueprint("translate", __name__)

# Define which languages we handle via transformers (for high-quality European translations)
TRANSFORMER_LANGS = {
    "hi": "Helsinki-NLP/opus-mt-en-hi",
    "fr": "Helsinki-NLP/opus-mt-en-fr",
    "es": "Helsinki-NLP/opus-mt-en-es",
    "de": "Helsinki-NLP/opus-mt-en-de",
    "ja": "Helsinki-NLP/opus-mt-en-ja",
}


def chunk_text(text, max_chars=500):
    """Split large text into manageable chunks."""
    sentences = text.split(". ")
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) < max_chars:
            current += sent + ". "
        else:
            chunks.append(current.strip())
            current = sent + ". "
    if current:
        chunks.append(current.strip())
    return chunks


@translate_bp.route("/api/translate", methods=["POST"])
def translate_text():
    try:
        data = request.get_json()
        text = data.get("text", "")
        target_lang = data.get("target_lang", "hi").lower()

        if not text.strip():
            return jsonify({"error": "No text provided"}), 400

        # ✅ Handle Indian/regional languages via GoogleTranslator
        if target_lang in ["te", "ta", "kn", "ml", "ja"]:
            try:
                translated_text = GoogleTranslator(source="en", target=target_lang).translate(text)
                return jsonify({"translated_text": translated_text})
            except Exception as e:
                return jsonify({"error": f"Google translation failed: {str(e)}"}), 500

        # ✅ Handle supported Transformer languages
        model_name = TRANSFORMER_LANGS.get(target_lang, "Helsinki-NLP/opus-mt-en-hi")
        translator = pipeline("translation", model=model_name)

        # Chunk for long input
        chunks = chunk_text(text)
        translated_chunks = []
        for chunk in chunks:
            translated = translator(chunk, max_length=1024, truncation=True)
            translated_chunks.append(translated[0]["translation_text"])

        translated_text = " ".join(translated_chunks).strip()
        return jsonify({"translated_text": translated_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
