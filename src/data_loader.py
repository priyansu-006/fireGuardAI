"""Dataset loading and ingestion module for the Forest Fire Classification Pipeline.

Handles:
- Loading CSV files safely using Pandas.
- Automatic dataset downloading from reputable source if missing locally.
- Robust parsing and cleaning of the multi-region Algerian Forest Fires dataset.
- Clear error handling for missing files, invalid paths, and unexpected formats.
"""

import os
import urllib.request
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

from src.config import (
    RAW_DATASET_PATH,
    DATASET_SOURCE_URL,
    TARGET_COLUMN,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)


def download_dataset(
    url: str = DATASET_SOURCE_URL,
    destination: Path = RAW_DATASET_PATH,
) -> Path:
    """Download the dataset from a remote URL to the destination path.

    Args:
        url: Remote URL of the dataset.
        destination: Local Path where the file should be saved.

    Returns:
        Path to the downloaded file.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[DataLoader] Downloading dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, destination)
        print(f"[DataLoader] Successfully downloaded to {destination} ({os.path.getsize(destination)} bytes).")
        return destination
    except Exception as e:
        raise RuntimeError(f"Failed to download dataset from {url}: {e}") from e


def load_raw_dataset(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the Algerian Forest Fires dataset from CSV, parsing region blocks cleanly.

    Args:
        file_path: Optional path to the CSV file. Defaults to config.RAW_DATASET_PATH.

    Returns:
        Cleaned pandas DataFrame with consistent columns and types.
    """
    path = Path(file_path) if file_path else RAW_DATASET_PATH

    if not path.exists():
        print(f"[DataLoader] Local dataset file not found at {path}. Attempting download...")
        download_dataset(DATASET_SOURCE_URL, path)

    if not path.is_file():
        raise FileNotFoundError(f"[DataLoader] Invalid dataset path: {path} is not a valid file.")

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        raise IOError(f"[DataLoader] Error reading file {path}: {e}") from e

    records = []
    current_region = "Bejaia"
    header = None

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
        # Region delimiters in original UCI dataset
        if "Bejaia" in line_clean and "Dataset" in line_clean:
            current_region = "Bejaia"
            continue
        if ("Sidi" in line_clean or "Abbes" in line_clean) and "Dataset" in line_clean:
            current_region = "Sidi-Bel Abbes"
            continue

        parts = [p.strip() for p in line_clean.split(",")]
        # Detect header row
        if parts[0].lower() == "day":
            if header is None:
                header = [p.strip() for p in parts] + ["Region"]
            continue

        # Valid data rows have 14 standard columns
        if len(parts) >= 14:
            clean_row = [p.strip() for p in parts[:14]] + [current_region]
            records.append(clean_row)

    if not records or header is None:
        raise ValueError(f"[DataLoader] No valid data rows found in {path}. Check file format.")

    df = pd.DataFrame(records, columns=header)

    # Clean whitespace in column names
    df.columns = df.columns.str.strip()

    # Clean target column whitespace
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip().str.lower()
        # Normalize minor typos in classes (e.g. 'not fire   ' or 'not fire')
        df[TARGET_COLUMN] = df[TARGET_COLUMN].replace({
            "not fire": "not fire",
            "not fire ": "not fire",
            "not  fire": "not fire",
            "fire": "fire",
            "fire ": "fire",
        })

    # Cast numerical columns to numeric types
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def display_dataset_summary(df: pd.DataFrame) -> None:
    """Print an informative, structured summary of the loaded dataset."""
    print("=" * 70)
    print("DATASET INGESTION SUMMARY")
    print("=" * 70)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nColumn Data Types:")
    for col, dtype in df.dtypes.items():
        print(f"  - {col:15s}: {str(dtype):10s} (non-null: {df[col].notna().sum()}/{len(df)})")

    print("\nFirst 3 Sample Records:")
    print(df.head(3).to_string(index=False))

    if TARGET_COLUMN in df.columns:
        print(f"\nTarget Class Distribution ('{TARGET_COLUMN}'):")
        counts = df[TARGET_COLUMN].value_counts()
        for label, count in counts.items():
            pct = (count / len(df)) * 100
            print(f"  - {label:12s}: {count:4d} records ({pct:.2f}%)")
    print("=" * 70)


if __name__ == "__main__":
    df = load_raw_dataset()
    display_dataset_summary(df)
