# backend/dataset.py
from flask import Blueprint, request, jsonify
import pandas as pd
import io
import requests
import math
import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import textstat
import traceback
import time
import os

dataset_bp = Blueprint("dataset", __name__)

# Local host where the same Flask app listens (adjust if different)
LOCAL_API_ROOT = "http://127.0.0.1:5000"

# helper: approx perplexity via word entropy (same method used in frontend)
def approx_perplexity(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    freq = {w: words.count(w) / len(words) for w in set(words)}
    # avoid math domain errors
    entropy = -sum((p * math.log(p, 2)) for p in freq.values() if p > 0)
    perplexity = math.pow(2, entropy)
    return float(perplexity)

# helper: BLEU with smoothing
_smooth = SmoothingFunction().method4
def compute_bleu(reference: str, candidate: str) -> float:
    try:
        ref_tokens = reference.split()
        cand_tokens = candidate.split()
        if not ref_tokens or not cand_tokens:
            return 0.0
        # sentence_bleu expects list of reference lists
        val = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=_smooth)
        return float(val)
    except Exception:
        return 0.0

# compute rouge scores (returns dict: rouge1_f, rouge2_f, rougel_f)
_rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
def compute_rouge(reference: str, candidate: str):
    try:
        r = _rouge.score(reference or "", candidate or "")
        return {
            "rouge1_f": float(r["rouge1"].fmeasure or 0.0),
            "rouge2_f": float(r["rouge2"].fmeasure or 0.0),
            "rougel_f": float(r["rougeL"].fmeasure or 0.0),
            "rouge1_p": float(r["rouge1"].precision or 0.0),
            "rouge1_r": float(r["rouge1"].recall or 0.0),
            "rouge2_p": float(r["rouge2"].precision or 0.0),
            "rouge2_r": float(r["rouge2"].recall or 0.0),
            "rougel_p": float(r["rougeL"].precision or 0.0),
            "rougel_r": float(r["rougeL"].recall or 0.0),
        }
    except Exception:
        return {
            "rouge1_f": 0.0, "rouge2_f": 0.0, "rougel_f": 0.0,
            "rouge1_p": 0.0, "rouge1_r": 0.0,
            "rouge2_p": 0.0, "rouge2_r": 0.0,
            "rougel_p": 0.0, "rougel_r": 0.0
        }

@dataset_bp.route("/evaluate", methods=["POST"])
def evaluate_dataset():
    """
    Accepts multipart-form request:
      - file: uploaded CSV (preferred)
    or
      - file_path: local server path to CSV (optional, for internal usage)
    Required form fields:
      - task: "summarization" or "paraphrasing"
      - model_name: e.g. "t5-small"
      - input_col: name of CSV column containing source text
      - ref_col: name of CSV column containing reference text
      - sample_size: optional int (0 or missing = all)
      - complexity: optional (for paraphrase) e.g. creative/standard/basic
    Returns JSON:
    {
      "aggregate": {...},
      "per_sample": [ { id, input, reference, generated, bleu, rouge1, rouge2, rougel, perplexity, readability_delta }, ... ]
    }
    """
    try:
        # 1) read params
        task = (request.form.get("task") or request.form.get("task_type") or "").strip().lower()
        model_name = (request.form.get("model_name") or "").strip()
        input_col = (request.form.get("input_col") or "").strip()
        ref_col = (request.form.get("ref_col") or "").strip()
        complexity = (request.form.get("complexity") or "").strip().lower()
        sample_size = request.form.get("sample_size")
        try:
            sample_size = int(sample_size) if sample_size is not None else 0
        except Exception:
            sample_size = 0

        # fallback if passed in JSON body (rare)
        if not task and request.json:
            task = (request.json.get("task") or "").lower()

        if task not in ("summarization", "paraphrasing"):
            return jsonify({"error": "Invalid or missing 'task' - must be 'summarization' or 'paraphrasing'"}), 400
        if not model_name:
            return jsonify({"error": "Missing 'model_name'"}), 400
        if not input_col or not ref_col:
            return jsonify({"error": "Missing 'input_col' or 'ref_col'"}), 400

        # 2) obtain CSV file either from multipart upload or server-side path
        file_obj = None
        if "file" in request.files:
            file_storage = request.files["file"]
            # read bytes and create a BytesIO for pandas
            file_bytes = file_storage.read()
            file_obj = io.BytesIO(file_bytes)
            file_obj.seek(0)
        else:
            # support a server-side path (developer note / internal usage)
            file_path = request.form.get("file_path") or request.args.get("file_path")
            if file_path:
                if not os.path.exists(file_path):
                    return jsonify({"error": f"file_path not found on server: {file_path}"}), 400
                file_obj = open(file_path, "rb")
            else:
                return jsonify({"error": "No file uploaded and no file_path provided"}), 400

        # 3) read CSV into pandas
        try:
            df = pd.read_csv(file_obj)
        except Exception as e:
            # try with engine python
            try:
                file_obj.seek(0)
                df = pd.read_csv(file_obj, engine="python")
            except Exception as e2:
                return jsonify({"error": f"Failed to parse CSV file: {e} / {e2}"}), 400

        # check columns
        if input_col not in df.columns or ref_col not in df.columns:
            return jsonify({"error": f"CSV does not contain specified columns. Available columns: {df.columns.tolist()}"}), 400

        # limit rows
        total_rows = len(df)
        if sample_size and sample_size > 0:
            df_sample = df.head(sample_size)
        else:
            df_sample = df

        # prepare loop targets
        per_sample = []
        # accumulators for aggregates
        accum = {
            "bleu": [],
            "rouge1": [],
            "rouge2": [],
            "rougel": [],
            "perplexity": [],
            "readability_delta": []
        }

        # Decide which internal endpoint to call
        if task == "summarization":
            model_endpoint = f"{LOCAL_API_ROOT}/api/summarize"
        else:
            model_endpoint = f"{LOCAL_API_ROOT}/api/paraphrase"

        # small optimization: session for requests
        sess = requests.Session()
        # set a short timeout per call to avoid hanging forever
        REQUEST_TIMEOUT = 30  # seconds; adjust if your models run longer

        # iterate rows
        for idx, row in df_sample.iterrows():
            try:
                src_text = str(row[input_col]) if not pd.isna(row[input_col]) else ""
                ref_text = str(row[ref_col]) if not pd.isna(row[ref_col]) else ""

                if not src_text or not ref_text:
                    # skip empty rows but include a placeholder
                    per_sample.append({
                        "id": int(idx),
                        "input": src_text,
                        "reference": ref_text,
                        "generated": "",
                        "bleu": 0.0,
                        "rouge1": 0.0,
                        "rouge2": 0.0,
                        "rougel": 0.0,
                        "perplexity": 0.0,
                        "readability_delta": 0.0,
                        "error": "empty input or reference"
                    })
                    continue

                # prepare payload for model endpoint
                if task == "summarization":
                    payload = {"text": src_text, "model_name": model_name, "length": "medium"}
                    # call endpoint
                    resp = sess.post(model_endpoint, json=payload, timeout=REQUEST_TIMEOUT)
                else:
                    payload = {"text": src_text, "model_name": model_name}
                    if complexity:
                        payload["complexity"] = complexity
                    resp = sess.post(model_endpoint, json=payload, timeout=REQUEST_TIMEOUT)

                generated = ""
                if resp.status_code == 200:
                    try:
                        out = resp.json()
                        # summarization endpoint returns 'summary' key in your frontend design
                        if task == "summarization":
                            generated = out.get("summary") or out.get("output") or out.get("result") or ""
                        else:
                            generated = out.get("paraphrase") or out.get("output") or out.get("result") or ""
                        if not isinstance(generated, str):
                            generated = str(generated)
                    except Exception:
                        generated = ""
                else:
                    # don't fail whole job for a single sample
                    generated = ""
                    # optionally capture backend error in per-sample
                    try:
                        errtxt = resp.json().get("error", resp.text)
                    except Exception:
                        errtxt = resp.text

                # compute metrics
                bleu_v = compute_bleu(ref_text, generated) if generated else 0.0
                rouge_v = compute_rouge(ref_text, generated) if generated else {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougel_f": 0.0}
                perp_v = approx_perplexity(generated) if generated else 0.0
                # readability: Flesch reading ease difference (ref - gen)
                try:
                    ref_read = textstat.flesch_reading_ease(ref_text)
                    gen_read = textstat.flesch_reading_ease(generated) if generated else 0.0
                    read_delta = float(ref_read - gen_read)
                except Exception:
                    read_delta = 0.0

                # record
                per_sample.append({
                    "id": int(idx),
                    "input": src_text,
                    "reference": ref_text,
                    "generated": generated,
                    "bleu": bleu_v,
                    "rouge1": rouge_v["rouge1_f"],
                    "rouge2": rouge_v["rouge2_f"],
                    "rougel": rouge_v["rougel_f"],
                    "perplexity": perp_v,
                    "readability_delta": read_delta
                })

                # accumulate for aggregate
                accum["bleu"].append(bleu_v)
                accum["rouge1"].append(rouge_v["rouge1_f"])
                accum["rouge2"].append(rouge_v["rouge2_f"])
                accum["rougel"].append(rouge_v["rougel_f"])
                accum["perplexity"].append(perp_v)
                accum["readability_delta"].append(read_delta)

                # small sleep to be polite if you're hitting heavy models (optional)
                # time.sleep(0.05)

            except Exception as ex_row:
                traceback.print_exc()
                per_sample.append({
                    "id": int(idx),
                    "input": str(row.get(input_col, "")),
                    "reference": str(row.get(ref_col, "")),
                    "generated": "",
                    "bleu": 0.0,
                    "rouge1": 0.0,
                    "rouge2": 0.0,
                    "rougel": 0.0,
                    "perplexity": 0.0,
                    "readability_delta": 0.0,
                    "error": f"row processing error: {str(ex_row)}"
                })
                continue

        # compute aggregates (mean, robust to zero-length lists)
        def safe_mean(lst):
            return float(np.mean(lst)) if lst else 0.0

        aggregate = {
            "BLEU": safe_mean(accum["bleu"]),
            "ROUGE-1": safe_mean(accum["rouge1"]),
            "ROUGE-2": safe_mean(accum["rouge2"]),
            "ROUGE-L": safe_mean(accum["rougel"]),
            "Perplexity": safe_mean(accum["perplexity"]),
            "Readability Delta": safe_mean(accum["readability_delta"]),
            "samples_evaluated": len(per_sample),
            "total_rows": total_rows
        }

        return jsonify({"aggregate": aggregate, "per_sample": per_sample}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Server error: {e}"}), 500
