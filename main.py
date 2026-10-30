"""Main entry point for the Forest Fire Machine Learning Classification Pipeline.

Executes the complete end-to-end workflow:
1. Dataset Sourcing & Ingestion (Data Loader)
2. Exploratory Data Analysis & Outlier Profiling (Data Inspection)
3. Leak-free Preprocessing & Stratified Train-Test Splitting (Preprocessing)
4. Pipeline Creation, Multi-Model Training & Cross-Validation (Training)
5. Comprehensive Model Evaluation & Artifact Persistence (Evaluation)
6. Sample Inference Demonstration (Prediction)
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    RAW_DATASET_PATH,
    PROCESSED_TRAIN_PATH,
    PROCESSED_TEST_PATH,
    BEST_MODEL_PATH,
    MODEL_COMPARISON_CSV,
    PLOTS_DIR,
    PRIMARY_METRIC,
)
from src.data_loader import load_raw_dataset, display_dataset_summary
from src.data_inspection import run_full_eda
from src.preprocessing import separate_features_and_target, split_data, save_processed_splits
from src.train import train_and_compare_models
from src.predict import ForestFirePredictor, demonstrate_sample_predictions


def run_pipeline(skip_eda: bool = False, dataset_path: str = None):
    """Run the complete machine learning classification pipeline."""
    print("\n" + "#" * 80)
    print(" " * 15 + "FOREST FIRE CLASSIFICATION PIPELINE: END-TO-END EXECUTION")
    print("#" * 80)

    # Step 1: Data Ingestion
    print("\n>>> STEP 1: DATASET INGESTION")
    df_raw = load_raw_dataset(file_path=Path(dataset_path) if dataset_path else RAW_DATASET_PATH)
    display_dataset_summary(df_raw)

    # Step 2: Data Inspection & EDA
    if not skip_eda:
        print("\n>>> STEP 2: DATA INSPECTION & EXPLORATORY DATA ANALYSIS")
        eda_plots = run_full_eda(df_raw)
    else:
        print("\n>>> STEP 2: EDA Skipped by user flag.")

    # Step 3: Feature-Target Separation & Leak-Free Train-Test Split
    print("\n>>> STEP 3: PREPROCESSING & STRATIFIED DATA SPLIT")
    X, y = separate_features_and_target(df_raw)
    X_train, X_test, y_train, y_test = split_data(X, y)
    save_processed_splits(X_train, X_test, y_train, y_test)

    # Step 4 & 5: Model Training, CV Benchmarking, Evaluation & Persistence
    print("\n>>> STEP 4 & 5: MODEL TRAINING, CROSS-VALIDATION & EVALUATION")
    best_pipeline, best_model_name, comparison_df = train_and_compare_models(
        X_train,
        y_train,
        X_test,
        y_test,
        primary_metric=PRIMARY_METRIC,
    )

    # Step 6: Prediction & Verification
    print("\n>>> STEP 6: MODEL INFERENCE & VERIFICATION")
    demonstrate_sample_predictions()

    print("\n" + "#" * 80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"Winning Pipeline Persisted At : {BEST_MODEL_PATH}")
    print(f"Metrics Table Saved At        : {MODEL_COMPARISON_CSV}")
    print(f"Visualization Plots Saved At  : {PLOTS_DIR}")
    print("#" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Forest Fire Machine Learning Classification Pipeline")
    parser.add_argument("--dataset", type=str, default=None, help="Custom CSV dataset path")
    parser.add_argument("--skip-eda", action="store_true", help="Skip generating EDA visualizations")
    args = parser.parse_args()

    run_pipeline(skip_eda=args.skip_eda, dataset_path=args.dataset)
