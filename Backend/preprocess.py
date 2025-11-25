import os
import pandas as pd
from transformers import AutoTokenizer
import nltk
import re

nltk.download("punkt")

# ---------------- CONFIG ----------------
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "../data")
PREPROCESSED_FOLDER = os.path.join(DATA_FOLDER, "processed")
os.makedirs(PREPROCESSED_FOLDER, exist_ok=True)

MODEL_TOKENIZERS = {
    "flan-t5": "google/flan-t5-small",
    "bart": "facebook/bart-large-cnn",
    "pegasus": "google/pegasus-xsum",
    "t5": "t5-small",
}

MAX_ROWS = 500  # reduce rows for faster runs

# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

# ---------------- LOAD DATASETS ----------------
def load_datasets():
    datasets = []
    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(DATA_FOLDER, file))
            print(f"Loaded {file} with {len(df)} rows")
            datasets.append(df)
    if datasets:
        combined = pd.concat(datasets, ignore_index=True)
        print(f"Combined dataset length: {len(combined)}")
        combined = combined.sample(n=min(len(combined), MAX_ROWS), random_state=42).reset_index(drop=True)
        print(f"Reduced dataset length: {len(combined)}")
        print(f"Columns in dataset: {list(combined.columns)}")
        return combined
    else:
        raise FileNotFoundError("No CSV files found in data folder")

# ---------------- PREPROCESS ----------------
def preprocess_data(df, text_column="article", summary_column="highlights"):
    df[text_column] = df[text_column].apply(clean_text)
    df[summary_column] = df[summary_column].apply(clean_text)
    df = df.dropna(subset=[text_column, summary_column])
    return df[[text_column, summary_column]]

# ---------------- TOKENIZE & SAVE ----------------
def tokenize_and_save(df):
    for model_name, model_path in MODEL_TOKENIZERS.items():
        print(f"Tokenizing for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        tokenized_data = []
        for idx, row in df.iterrows():
            inputs = tokenizer(row["article"], truncation=True, padding="max_length", max_length=512)
            labels = tokenizer(row["highlights"], truncation=True, padding="max_length", max_length=128)
            tokenized_data.append({
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "labels": labels.input_ids
            })

        save_path = os.path.join(PREPROCESSED_FOLDER, f"{model_name}_dataset.csv")
        tokenized_df = pd.DataFrame(tokenized_data)
        tokenized_df.to_csv(save_path, index=False)
        print(f"Saved preprocessed dataset for {model_name} at {save_path}")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    df = load_datasets()
    df = preprocess_data(df, text_column="article", summary_column="highlights")
    tokenize_and_save(df)
    print("✅ Preprocessing complete for all models!")
