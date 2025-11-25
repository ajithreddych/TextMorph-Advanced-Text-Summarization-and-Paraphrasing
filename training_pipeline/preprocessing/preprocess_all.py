import os
import pandas as pd
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Base path (project-root/training_pipeline)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
OUTPUT_DIR = os.path.join(DATASET_DIR, "processed")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# List all datasets
dataset_files = [
    os.path.join(DATASET_DIR, "data1.csv"),
    os.path.join(DATASET_DIR, "data2.csv"),
]

def clean_text(text: str) -> str:
    """Basic text cleaning"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)  # remove extra spaces/newlines
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^A-Za-z0-9.,!?';:\-\s]", "", text)  # keep readable chars
    return text.strip()

def preprocess_dataset(file_path: str) -> pd.DataFrame:
    print(f"🔹 Loading {file_path} ...")
    try:
        df = pd.read_csv(file_path)
        # Expect CNN/DM style columns
        if {"article", "highlights"}.issubset(df.columns):
            df = df[["article", "highlights"]].rename(
                columns={"article": "text", "highlights": "summary"}
            )
        else:
            print(f"⚠️ Skipping {file_path} (no 'article' or 'highlights' columns)")
            return pd.DataFrame()

        tqdm.pandas(desc=f"Cleaning {os.path.basename(file_path)}")
        df["text"] = df["text"].progress_apply(clean_text)
        df["summary"] = df["summary"].progress_apply(clean_text)
        df.dropna(subset=["text", "summary"], inplace=True)
        df = df[(df["text"].str.len() > 30) & (df["summary"].str.len() > 10)]
        df.reset_index(drop=True, inplace=True)
        print(f"✅ Cleaned {len(df)} rows from {os.path.basename(file_path)}")
        return df

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return pd.DataFrame()

def save_splits(df: pd.DataFrame, output_dir: str):
    """Save train/val/test splits"""
    print("\n📊 Splitting into train/val/test...")
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"✅ Saved splits in {output_dir}")
    print(f"   Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

def preprocess_and_merge(dataset_files, output_path):
    all_dfs = []
    for f in dataset_files:
        if os.path.exists(f):
            df = preprocess_dataset(f)
            if not df.empty:
                all_dfs.append(df)
        else:
            print(f"⚠️ File not found: {f}")

    if not all_dfs:
        print("❌ No valid datasets found to process.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.drop_duplicates(subset=["text"], inplace=True)
    combined.to_csv(output_path, index=False)
    print(f"\n🎉 Combined dataset saved to: {output_path}")
    print(f"Total cleaned samples: {len(combined)}")

    save_splits(combined, OUTPUT_DIR)

if __name__ == "__main__":
    output_path = os.path.join(OUTPUT_DIR, "combined_cleaned.csv")
    preprocess_and_merge(dataset_files, output_path)
