#!/usr/bin/env python3
"""
evaluate_models.py

Usage examples:

# 1) Evaluate a single model (model_id or explicit path). Input CSV should contain columns
#    "text" (or "article") and "summary" (or "highlights").
python evaluate_models.py \
  --model_path training_pipeline/datasets/processed/model_outputs/summarization_t5-small_20251104_165531 \
  --input_csv training_pipeline/datasets/processed/eval_sample.csv \
  --out_dir training_pipeline/evaluation/results_single

# 2) Evaluate all models registered in training_pipeline/versioning/model_registry.json
python evaluate_models.py --all --out_dir training_pipeline/evaluation/results_all

# 3) Quick run: evaluate top 200 rows and save outputs
python evaluate_models.py --model_path <path> --max_samples 200 --out_dir training_pipeline/evaluation/quick
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import sacrebleu
import textstat
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# -------------------------
# CLI
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="Path to local model directory (overrides registry)")
parser.add_argument("--all", action="store_true", help="Evaluate all models listed in versioning/model_registry.json")
parser.add_argument("--registry", type=str, default="training_pipeline/versioning/model_registry.json",
                    help="Path to model registry JSON")
parser.add_argument("--input_csv", type=str, default=None,
                    help="CSV path with columns 'text'/'article' and 'summary'/'highlights'. If omitted and --all, script will still evaluate but will sample from processed/train.csv")
parser.add_argument("--max_samples", type=int, default=500, help="Limit evaluation samples for speed")
parser.add_argument("--batch_size", type=int, default=8, help="Generation batch size")
parser.add_argument("--max_gen_length", type=int, default=128, help="Max tokens to generate")
parser.add_argument("--min_gen_length", type=int, default=8, help="Min tokens to generate")
parser.add_argument("--out_dir", type=str, default="training_pipeline/evaluation/results")
parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps"], help="device for inference (cpu recommended)")
args = parser.parse_args()

OUT_DIR = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Utilities: dataset loading
# -------------------------
def load_eval_dataframe(csv_path: str = None, max_samples: int = None) -> pd.DataFrame:
    # Priority: explicit csv_path -> processed/test.csv -> processed/combined_cleaned.csv
    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        fallback_paths = [
            Path("training_pipeline/datasets/processed/test.csv"),
            Path("training_pipeline/datasets/processed/val.csv"),
            Path("training_pipeline/datasets/processed/combined_cleaned.csv"),
            Path("training_pipeline/datasets/processed/train.csv")
        ]
        found = None
        for p in fallback_paths:
            if p.exists():
                found = p
                break
        if not found:
            raise FileNotFoundError("No input CSV provided and no processed dataset found in training_pipeline/datasets/processed/")
        df = pd.read_csv(found)
    # normalize column names
    if "article" in df.columns and "text" not in df.columns:
        df["text"] = df["article"]
    if "highlights" in df.columns and "summary" not in df.columns:
        df["summary"] = df["highlights"]
    if "text" not in df.columns or "summary" not in df.columns:
        raise ValueError("Input CSV must contain 'text' (or 'article') and 'summary' (or 'highlights') columns.")
    df = df[["text", "summary"]].dropna().reset_index(drop=True)
    if max_samples:
        df = df.head(max_samples)
    return df

# -------------------------
# Metrics helpers
# -------------------------
def compute_bleu(cands: List[str], refs: List[str]) -> Dict:
    # sacrebleu expects list of references or list of list; we use single-reference corpus BLEU
    bleu = sacrebleu.corpus_bleu(cands, [refs])
    return {"bleu_score": bleu.score, "bleu_sys": bleu.__dict__.get("sys_score", None)}

def compute_readability_stats(texts: List[str]) -> Dict[str, float]:
    # returns average Flesch Reading Ease and FK Grade
    fre_scores = []
    fk_grades = []
    for t in texts:
        try:
            fre_scores.append(textstat.flesch_reading_ease(t))
            fk_grades.append(textstat.flesch_kincaid_grade(t))
        except Exception:
            fre_scores.append(float("nan"))
            fk_grades.append(float("nan"))
    return {"flesch_reading_ease": float(np.nanmean(fre_scores)), "flesch_kincaid_grade": float(np.nanmean(fk_grades))}

def compute_perplexity_from_loss(model, tokenizer, texts: List[str], device="cpu", max_len=512) -> float:
    # Compute token-level negative log likelihood on references (approximate perplexity).
    # For seq2seq models we compute loss by feeding input & labels — tokenized with padding/truncation.
    model.eval()
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for t in texts:
            enc = tokenizer(t, truncation=True, padding="longest", return_tensors="pt", max_length=max_len)
            # For encoder-decoder models we can't compute perplexity purely on input text.
            # We'll estimate by using the tokenizer to convert the text to ids and computing loss
            # by asking the model to reconstruct the text (naive LM-style). This is a rough estimate.
            # Many seq2seq models don't support full causal LM perplexity; this is an approximation.
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            # Use the same text as labels (shifted by model internally)
            labels = input_ids.clone().to(device)
            try:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = float(outputs.loss.cpu().item())
                if math.isfinite(loss):
                    loss_sum += loss
                    count += 1
            except Exception:
                # fallback: skip this example
                continue
    if count == 0:
        return float("nan")
    avg_loss = loss_sum / count
    try:
        perplexity = float(math.exp(avg_loss))
    except OverflowError:
        perplexity = float("inf")
    return perplexity

# -------------------------
# Single model evaluation
# -------------------------
def evaluate_one_model(model_path: str, df: pd.DataFrame, device="cpu", batch_size=8, max_length=128, min_length=8) -> Dict:
    print(f"[EVAL] Loading model: {model_path} (device={device})")
    model_path = str(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to(device)

    # generation pipeline (batching ourselves)
    generated_texts = []
    inputs = df["text"].tolist()
    for i in tqdm(range(0, len(inputs), batch_size), desc="Generating"):
        batch = inputs[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        gen = tokenizer.batch_decode(out, skip_special_tokens=True)
        generated_texts.extend(gen)

    references = df["summary"].astype(str).tolist()
    # compute BLEU (corpus)
    bleu_res = compute_bleu(generated_texts, references)

    # compute perplexity on references (approximate)
    # Note: this uses the same model to compute loss of reference texts; it's an approximation
    print("[EVAL] Computing perplexity (may be slow)...")
    ppl = compute_perplexity_from_loss(model, tokenizer, references[: args.max_samples], device=device, max_len=512)

    # readability: compare reference vs generated
    print("[EVAL] Computing readability stats...")
    ref_read = compute_readability_stats(references)
    gen_read = compute_readability_stats(generated_texts)
    read_delta = {
        "flesch_reading_ease_ref": ref_read["flesch_reading_ease"],
        "flesch_reading_ease_gen": gen_read["flesch_reading_ease"],
        "flesch_reading_ease_delta": gen_read["flesch_reading_ease"] - ref_read["flesch_reading_ease"],
        "flesch_kincaid_grade_ref": ref_read["flesch_kincaid_grade"],
        "flesch_kincaid_grade_gen": gen_read["flesch_kincaid_grade"],
        "flesch_kincaid_grade_delta": gen_read["flesch_kincaid_grade"] - ref_read["flesch_kincaid_grade"],
    }

    # per-example DataFrame with generated texts and simple sentence-level BLEU (sacrebleu supports sentence-level)
    per_example = []
    for inp, ref, gen in zip(df["text"].tolist(), references, generated_texts):
        try:
            sent_bleu = sacrebleu.sentence_bleu(gen, [ref]).score
        except Exception:
            sent_bleu = float("nan")
        per_example.append({"text": inp, "reference": ref, "generated": gen, "sentence_bleu": sent_bleu})

    result = {
        "model_path": model_path,
        "num_examples": len(df),
        "bleu_corpus": bleu_res,
        "perplexity": ppl,
        "readability": read_delta,
        "per_example": pd.DataFrame(per_example),
    }
    # clean up
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return result

# -------------------------
# Runner: evaluate either single model or registry
# -------------------------
def load_registry(path: str) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("models", [])
    except Exception:
        return []

def save_summary(results: List[Dict], out_dir: Path):
    # results: list of dict with top-level metrics
    summary_list = []
    for r in results:
        summary_list.append({
            "model_path": r["model_path"],
            "num_examples": r["num_examples"],
            "bleu": r["bleu_corpus"]["bleu_score"],
            "perplexity": r["perplexity"],
            "flesch_reading_ease_delta": r["readability"]["flesch_reading_ease_delta"],
            "fk_grade_delta": r["readability"]["flesch_kincaid_grade_delta"]
        })
    summary_df = pd.DataFrame(summary_list)
    summary_df.to_csv(out_dir / "evaluation_summary.csv", index=False)
    (out_dir / "evaluation_summary.json").write_text(json.dumps(summary_list, indent=2), encoding="utf-8")
    print("[SAVE] Summary saved to", out_dir / "evaluation_summary.csv")

def save_per_example(results: List[Dict], out_dir: Path):
    # Save per-model per-example CSV + a combined HTML for side-by-side comparison
    html_blocks = []
    for r in results:
        df = r["per_example"]
        model_name = Path(r["model_path"]).name
        csv_path = out_dir / f"{model_name}_examples.csv"
        df.to_csv(csv_path, index=False)
        html_blocks.append(f"<h2>{model_name}</h2>")
        html_blocks.append(df.to_html(escape=False, index=False))
    html_content = "<html><body>" + "\n".join(html_blocks) + "</body></html>"
    (out_dir / "side_by_side.html").write_text(html_content, encoding="utf-8")
    print("[SAVE] Per-example CSVs and side_by_side.html saved to", out_dir)

def main():
    # load input data
    df = load_eval_dataframe(args.input_csv, max_samples=args.max_samples)
    print(f"[MAIN] Using {len(df)} eval examples.")

    models_to_eval = []
    if args.all:
        registry = load_registry(args.registry)
        if not registry:
            raise FileNotFoundError(f"No registry found at {args.registry}")
        # registry entries should have 'path' and 'model_id'
        for entry in registry:
            models_to_eval.append(entry["path"])
    else:
        if not args.model_path:
            raise ValueError("Please provide --model_path or use --all to evaluate registry models.")
        models_to_eval.append(args.model_path)

    results = []
    for mpath in models_to_eval:
        try:
            res = evaluate_one_model(mpath, df, device=args.device, batch_size=args.batch_size,
                                     max_length=args.max_gen_length, min_length=args.min_gen_length)
            results.append(res)
        except Exception as e:
            print(f"[ERROR] Failed to evaluate {mpath}: {e}")

    # Save outputs
    save_summary(results, OUT_DIR)
    save_per_example(results, OUT_DIR)
    print("[DONE] All results saved to", OUT_DIR)

if __name__ == "__main__":
    main()
