#!/usr/bin/env python3
"""
train_multi_task.py — unified runner to train multiple summarization & paraphrase models
Saves each model into a separate folder automatically and records a model_registry.json.

Usage:
  # Train all configured models (fast defaults)
  python train_multi_task.py --train_all

  # Train single model/task
  python train_multi_task.py --task summarization --model t5-small --max_train_samples 500 --epochs 1

Notes:
 - Script forces CPU (disables MPS) to avoid macOS MPS allocation errors.
 - Adjust --max_train_samples and --epochs for speed vs quality.
"""

# ---------------------------
# Force CPU / disable MPS early (must come before torch import)
# ---------------------------
import os
os.environ["PYTORCH_DISABLE_MPS"] = "1"

# ---------------------------
# Standard imports
# ---------------------------
import argparse
import datetime
import json
import random
import re
from pathlib import Path
from typing import Dict, List

import torch

# Defensive monkey-patch after torch import (ensure MPS reports False)
try:
    if hasattr(torch, "backends") and hasattr(torch.backends, "mps"):
        torch.backends.mps.is_available = lambda *a, **k: False
        torch.backends.mps.is_built = lambda *a, **k: False
except Exception:
    pass

import evaluate
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# ---------------------------
# Project paths
# ---------------------------
BASE = Path(__file__).resolve().parents[1]  # training_pipeline/
PROCESSED_DIR = BASE / "datasets" / "processed"
MODEL_OUTPUT_ROOT = PROCESSED_DIR / "model_outputs"
MODEL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
MODEL_REGISTRY_PATH = BASE / "versioning" / "model_registry.json"
MODEL_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Quick cleaning helper
# ---------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9.,!?';:\-\s]", "", text)
    return text.strip()

# ---------------------------
# Presets
# ---------------------------
LENGTH_PRESETS = {
    "short": {"min_length": 8, "max_length": 32},
    "medium": {"min_length": 32, "max_length": 96},
    "long": {"min_length": 96, "max_length": 256},
}

COMPLEXITY_PRESETS = {
    "simple": {"num_beams": 6, "temperature": 1.0, "top_k": None, "top_p": None},
    "standard": {"num_beams": 4, "temperature": 1.0, "top_k": None, "top_p": None},
    "creative": {"num_beams": 1, "temperature": 1.2, "top_k": 50, "top_p": 0.95},
}

MODEL_MAP = {
    "flan-t5-small": "google/flan-t5-small",
    "flan-t5-base": "google/flan-t5-base",
    "t5-small": "t5-small",
    "bart-base": "facebook/bart-base",
    "pegasus-xsum": "google/pegasus-xsum",
    "pegasus": "google/pegasus-xsum",
}

# Default sets (will be trained when --train_all)
DEFAULT_SUMMARIZERS = ["t5-small", "google/flan-t5-small", "facebook/bart-base", "google/pegasus-xsum"]
DEFAULT_PARAPHRASERS = ["t5-small", "google/flan-t5-small", "facebook/bart-base"]

# ---------------------------
# CLI
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--train_all", action="store_true", help="Train the default set of models for both tasks (fast defaults).")
parser.add_argument("--task", choices=["summarization", "paraphrase"], help="Train only this task (ignored with --train_all).")
parser.add_argument("--model", type=str, help="Single model id or shorthand to train (ignored with --train_all).")
parser.add_argument("--length", choices=list(LENGTH_PRESETS.keys()), default="short")
parser.add_argument("--complexity", choices=list(COMPLEXITY_PRESETS.keys()), default="standard")
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--max_train_samples", type=int, default=500, help="Limit training samples for fast runs")
parser.add_argument("--max_input_length", type=int, default=512)
parser.add_argument("--max_target_length", type=int, default=128)
parser.add_argument("--auto_paraphrase", action="store_true", help="Synthesize paraphrases if missing")
parser.add_argument("--save_steps", type=int, default=100)
parser.add_argument("--save_total_limit", type=int, default=3)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# Resolve model id if shorthand
def resolve_model_id(name: str) -> str:
    return MODEL_MAP.get(name, name)

# ---------------------------
# Device (force CPU)
# ---------------------------
device = "cpu"
print(f"[INFO] Device forced to: {device}")

random.seed(args.seed)
torch.manual_seed(args.seed)

# ---------------------------
# Load dataset once
# ---------------------------
def load_local_datasets() -> DatasetDict:
    train_path = PROCESSED_DIR / "train.csv"
    val_path = PROCESSED_DIR / "val.csv"
    test_path = PROCESSED_DIR / "test.csv"
    combined_path = PROCESSED_DIR / "combined_cleaned.csv"

    if train_path.exists() and val_path.exists():
        ds = load_dataset("csv", data_files={"train": str(train_path), "validation": str(val_path), "test": str(test_path) if test_path.exists() else str(val_path)})
    elif combined_path.exists():
        ds_all = load_dataset("csv", data_files=str(combined_path))["train"]
        ds_train = ds_all.train_test_split(test_size=0.2, seed=args.seed)
        tmp = ds_train['test'].train_test_split(test_size=0.5, seed=args.seed)
        ds = DatasetDict({"train": ds_train["train"], "validation": tmp["train"], "test": tmp["test"]})
    else:
        raise FileNotFoundError(f"No dataset files found in {PROCESSED_DIR}. Place train/val/test.csv or combined_cleaned.csv there.")
    return ds

print("[INFO] Loading datasets from:", PROCESSED_DIR)
dataset = load_local_datasets()
print("[INFO] Splits:", {k: len(dataset[k]) for k in dataset.keys()})

# Clean / prepare dataset columns (text, summary, optional paraphrase)
def prepare_dataset_for_task(ds: DatasetDict, task: str) -> DatasetDict:
    def map_fn(ex):
        return {
            "text": clean_text(ex.get("text", ex.get("article", ""))),
            "summary": clean_text(ex.get("summary", ex.get("highlights", ""))),
            "paraphrase": clean_text(ex.get("paraphrase", "")) if "paraphrase" in ex else ""
        }
    ds = ds.map(lambda x: map_fn(x), remove_columns=ds["train"].column_names, batched=False)
    if task == "paraphrase" and args.auto_paraphrase:
        # naive synthetic paraphrase: small swap
        def ensure_para(ex):
            if ex.get("paraphrase"):
                return ex
            tokens = ex["text"].split()
            if len(tokens) > 6:
                i, j = 1, min(3, len(tokens)-1)
                tokens[i], tokens[j] = tokens[j], tokens[i]
            ex["paraphrase"] = " ".join(tokens)
            return ex
        ds = ds.map(ensure_para, batched=False)
    # limit samples if requested (fast experiments)
    if args.max_train_samples:
        n = min(args.max_train_samples, len(ds["train"]))
        ds["train"] = ds["train"].select(range(n))
    return ds

# ---------------------------
# Core train function
# ---------------------------
def train_single_model(task: str, model_name: str):
    model_id = resolve_model_id(model_name)
    tag = model_id.replace("/", "_").replace(":", "_")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = MODEL_OUTPUT_ROOT / f"{task}_{tag}_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n[TRAIN] Task={task} Model={model_id} -> outdir={outdir}")

    # prepare task dataset
    ds_task = prepare_dataset_for_task(dataset, task)

    # tokenizer + model
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN", None)
    print("[INFO] Loading tokenizer & model:", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, use_auth_token=hf_token)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, use_auth_token=hf_token)
    model.to(device)

    # preprocessing helpers
    def preprocess_summarization(batch):
        inputs = tokenizer(batch["text"], max_length=args.max_input_length, truncation=True, padding="max_length")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(batch["summary"], max_length=args.max_target_length, truncation=True, padding="max_length")
        inputs["labels"] = labels["input_ids"]
        return inputs

    def preprocess_paraphrase(batch):
        inputs = tokenizer(batch["text"], max_length=args.max_input_length, truncation=True, padding="max_length")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(batch["paraphrase"], max_length=args.max_target_length, truncation=True, padding="max_length")
        inputs["labels"] = labels["input_ids"]
        return inputs

    if task == "summarization":
        tokenized = ds_task.map(preprocess_summarization, batched=True, remove_columns=ds_task["train"].column_names)
    else:
        tokenized = ds_task.map(preprocess_paraphrase, batched=True, remove_columns=ds_task["train"].column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    rouge = evaluate.load("rouge")

    def postprocess_text(preds, labels):
        preds = [p.strip() for p in preds]
        labels = [l.strip() for l in labels]
        return preds, labels

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels_proc = [[(l if l != -100 else tokenizer.pad_token_id) for l in lab] for lab in labels]
        decoded_labels = tokenizer.batch_decode(labels_proc, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        res = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        res = {k: round(v * 100, 4) for k, v in res.items()}
        return res

    # generate kwargs
    gen_kwargs: Dict = {}
    if task == "summarization":
        preset = LENGTH_PRESETS[args.length]
        gen_kwargs.update({
            "min_length": preset["min_length"],
            "max_length": preset["max_length"],
            "num_beams": 4,
            "early_stopping": True,
            "no_repeat_ngram_size": 3
        })
    else:
        comp = COMPLEXITY_PRESETS[args.complexity]
        gen_kwargs.update({
            "num_beams": comp["num_beams"],
            "temperature": comp["temperature"],
            "top_k": comp["top_k"],
            "top_p": comp["top_p"],
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
            "max_length": args.max_target_length,
            "min_length": 8
        })

    # training args (cpu-safe)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(outdir),
        evaluation_strategy="steps",
        eval_steps=args.save_steps,
        save_steps=args.save_steps,
        logging_steps=50,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        fp16=False,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        greater_is_better=True,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer._gen_kwargs = gen_kwargs

    # Train
    print(f"[TRAIN] Starting: {task} | {model_id} | epochs={args.epochs} | samples={len(tokenized['train'])}")
    trainer.train()
    trainer.save_model(str(outdir))
    tokenizer.save_pretrained(str(outdir))
    print(f"[TRAIN] Saved model to {outdir}")

    # Register model in JSON registry
    entry = {
        "task": task,
        "model_id": model_id,
        "path": str(outdir),
        "timestamp": datetime.datetime.now().isoformat(),
        "train_samples": len(tokenized["train"]),
        "notes": f"length={args.length} complexity={args.complexity}"
    }
    _append_registry_entry(entry)

# ---------------------------
# Model registry helper
# ---------------------------
def _load_registry() -> Dict:
    if MODEL_REGISTRY_PATH.exists():
        try:
            return json.loads(MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {"models": []}
    return {"models": []}

def _append_registry_entry(entry: Dict):
    reg = _load_registry()
    reg["models"].append(entry)
    MODEL_REGISTRY_PATH.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    print(f"[REGISTRY] Added entry for {entry['model_id']} -> {entry['path']}")

# ---------------------------
# Main runner
# ---------------------------
def main():
    # Decide runs
    runs = []
    if args.train_all:
        for m in DEFAULT_SUMMARIZERS:
            runs.append(("summarization", m))
        for m in DEFAULT_PARAPHRASERS:
            runs.append(("paraphrase", m))
    else:
        if not args.task or not args.model:
            raise ValueError("Specify --task and --model, or use --train_all")
        runs.append((args.task, args.model))

    # Run sequentially (safe on CPU)
    for task, model_name in runs:
        train_single_model(task, model_name)

    print("\n[ALL DONE] Models saved under:", MODEL_OUTPUT_ROOT)
    print("[ALL DONE] Registry:", MODEL_REGISTRY_PATH)

if __name__ == "__main__":
    main()
