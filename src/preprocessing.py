"""Preprocessing and Feature Engineering module for Forest Fire Classification.

Builds leakage-free Scikit-learn preprocessing pipelines using ColumnTransformer,
handles feature scaling, categorical encoding, median imputation, and stratified train-test splitting.
"""

from pathlib import Path
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

from src.config import (
    TARGET_COLUMN,
    TARGET_MAPPING,
    INVERSE_TARGET_MAPPING,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    EXCLUDED_COLUMNS,
    TEST_SIZE,
    RANDOM_STATE,
    STRATIFY,
    SCALER_TYPE,
    PROCESSED_TRAIN_PATH,
    PROCESSED_TEST_PATH,
)


def separate_features_and_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    excluded_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate input features X and encoded target series y.

    Excludes identifier/leakage/temporal columns (e.g. day, month, year).
    Ensures missing target labels are handled cleanly without imputation.
    """
    if target_col not in df.columns:
        raise KeyError(f"[Preprocessing] Target column '{target_col}' not found in dataset columns: {list(df.columns)}")

    # Clean target
    clean_df = df.copy()
    clean_df = clean_df.dropna(subset=[target_col])

    # Encode target to binary (1: fire, 0: not fire)
    y_raw = clean_df[target_col].astype(str).str.strip().str.lower()
    y = y_raw.map(TARGET_MAPPING)

    if y.isna().any():
        unknown_labels = clean_df.loc[y.isna(), target_col].unique()
        raise ValueError(
            f"[Preprocessing] Found unmapped target values: {unknown_labels}. "
            f"Expected keys: {list(TARGET_MAPPING.keys())}"
        )
    y = y.astype(int)

    # Exclude columns
    to_drop = [target_col] + (excluded_cols or EXCLUDED_COLUMNS)
    X = clean_df.drop(columns=[col for col in to_drop if col in clean_df.columns])

    # Filter to known numerical and categorical features
    expected_cols = [c for c in (NUMERICAL_FEATURES + CATEGORICAL_FEATURES) if c in X.columns]
    X = X[expected_cols]

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    stratify: bool = STRATIFY,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target into reproducible train and test sets.

    Uses stratification by default for classification to maintain class proportions.
    """
    stratify_target = y if stratify else None

    # Validate dataset size
    min_class_count = y.value_counts().min()
    if min_class_count < 2:
        raise ValueError(f"[Preprocessing] Smallest class has only {min_class_count} samples. Cannot stratify.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    print("\n" + "=" * 70)
    print("DATA SPLITTING SUMMARY (Stratified Train-Test Split)")
    print("=" * 70)
    print(f"Total Samples : {len(X)}")
    print(f"Training Set  : {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Test Set      : {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    print(f"Train Class Dist : {dict(y_train.value_counts())}")
    print(f"Test Class Dist  : {dict(y_test.value_counts())}")
    print("=" * 70)

    return X_train, X_test, y_train, y_test


def build_preprocessor(
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    scaler_type: str = SCALER_TYPE,
) -> ColumnTransformer:
    """Create a leak-free Scikit-learn ColumnTransformer for preprocessing.

    - Numerical pipeline:
        1. SimpleImputer(strategy='median')
        2. StandardScaler() or MinMaxScaler()
    - Categorical pipeline:
        1. SimpleImputer(strategy='most_frequent')
        2. OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    """
    num_cols = numerical_features or NUMERICAL_FEATURES
    cat_cols = categorical_features or CATEGORICAL_FEATURES

    # Choose scaler
    if scaler_type.lower() == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def save_processed_splits(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    train_path: Path = PROCESSED_TRAIN_PATH,
    test_path: Path = PROCESSED_TEST_PATH,
) -> Tuple[Path, Path]:
    """Persist train and test splits to CSV for full auditability and reproducibility."""
    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train.values

    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test.values

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[Preprocessing] Saved processed train split to {train_path}")
    print(f"[Preprocessing] Saved processed test split to {test_path}")

    return train_path, test_path


if __name__ == "__main__":
    from src.data_loader import load_raw_dataset

    raw_df = load_raw_dataset()
    X, y = separate_features_and_target(raw_df)
    X_tr, X_te, y_tr, y_te = split_data(X, y)
    save_processed_splits(X_tr, X_te, y_tr, y_te)

    preprocessor = build_preprocessor()
    preprocessor.fit(X_tr)
    X_tr_trans = preprocessor.transform(X_tr)
    print(f"[Preprocessing] Transformed feature matrix shape: {X_tr_trans.shape}")
