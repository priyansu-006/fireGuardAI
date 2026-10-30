"""Configuration module for the Forest Fire Classification Pipeline.

Contains all paths, hyperparameters, column mappings, and settings in one
centralized location to ensure reproducibility and maintainability.
"""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
METRICS_DIR = OUTPUTS_DIR / "metrics"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Ensure runtime directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, PLOTS_DIR, METRICS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset file paths
RAW_DATASET_PATH = RAW_DATA_DIR / "algerian_forest_fires.csv"
DATASET_SOURCE_URL = (
    "https://raw.githubusercontent.com/krishnaik06/Complete-Machine-Learning-2023/main/"
    "Algerian_forest_fires_dataset_UPDATE.csv"
)
PROCESSED_TRAIN_PATH = PROCESSED_DATA_DIR / "train.csv"
PROCESSED_TEST_PATH = PROCESSED_DATA_DIR / "test.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_fire_classification_pipeline.joblib"

# Model Comparison & Reporting paths
MODEL_COMPARISON_CSV = METRICS_DIR / "model_comparison.csv"
CLASSIFICATION_REPORT_TXT = REPORTS_DIR / "classification_report.txt"

# Dataset Column Definitions
TARGET_COLUMN = "Classes"
POSITIVE_CLASS_NAME = "fire"
NEGATIVE_CLASS_NAME = "not fire"

# Target binary encoding
TARGET_MAPPING = {
    "fire": 1,
    "not fire": 0,
}
INVERSE_TARGET_MAPPING = {1: "fire", 0: "not fire"}

# Features to exclude from modeling (temporal identifiers that do not generalize)
EXCLUDED_COLUMNS = ["day", "month", "year"]

# Input features for classification
NUMERICAL_FEATURES = [
    "Temperature",  # Temperature in Celsius (noon)
    "RH",           # Relative Humidity in %
    "Ws",           # Wind Speed in km/h
    "Rain",         # Total day rainfall in mm
    "FFMC",         # Fine Fuel Moisture Code
    "DMC",          # Duff Moisture Code
    "DC",           # Drought Code
    "ISI",          # Initial Spread Index
    "BUI",          # Buildup Index
    "FWI",          # Fire Weather Index
]

CATEGORICAL_FEATURES = [
    "Region",       # Geographic region (Bejaia or Sidi-Bel Abbes)
]

ALL_FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Train / Test Splitting and Cross Validation
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
STRATIFY = True

# Feature Scaling options: 'standard' or 'minmax'
SCALER_TYPE = "standard"

# Primary optimization metric for model selection
# Options: 'f1', 'roc_auc', 'recall', 'precision', 'accuracy'
# In forest fire detection, False Negatives (missing a real fire) are dangerous,
# making F1-score and ROC-AUC ideal balanced metrics.
PRIMARY_METRIC = "f1"
