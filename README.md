# Forest Fire Machine Learning Classification Pipeline

An end-to-end, leak-free Machine Learning classification pipeline built with **Scikit-learn**, **Pandas**, **NumPy**, **Seaborn**, and **XGBoost** to predict wildfire occurrence based on meteorological indicators and the Canadian Fire Weather Index (FWI) system.

---

## 1. Project Overview

Forest fires cause severe environmental, ecological, and economic devastation. Early and accurate prediction of wildfire risk enables proactive disaster mitigation, optimal resource deployment, and preservation of biodiversity.

This project implements a complete, production-grade machine learning classification pipeline that:
- Ingests and cleans multi-region wildfire weather data.
- Conducts comprehensive Exploratory Data Analysis (EDA) and Interquartile Range (IQR) outlier profiling.
- Enforces strict data-leakage prevention by encapsulating median imputation, one-hot encoding, and standard scaling inside Scikit-learn `Pipeline` and `ColumnTransformer` constructs fitted exclusively on training splits.
- Trains, benchmarks, and cross-validates 4 candidate classifiers:
  1. **Logistic Regression**
  2. **Decision Tree Classifier**
  3. **Random Forest Classifier**
  4. **XGBoost Classifier**
- Evaluates models on held-out test data across Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- Persists the winning pipeline for real-time inference and batch prediction.

---

## 2. Project Architecture

```
forestFireML/
├── data/
│   ├── raw/
│   │   ├── algerian_forest_fires.csv      # Primary benchmark dataset
│   │   └── india_forest_fires_by_state.csv# Secondary reference incident statistics
│   └── processed/
│       ├── train.csv                      # Stratified training split (80%)
│       └── test.csv                       # Held-out testing split (20%)
├── src/
│   ├── __init__.py
│   ├── config.py                          # Centralized configuration, hyperparams, & paths
│   ├── data_loader.py                     # Dataset loading, remote fetch, & cleaning
│   ├── data_inspection.py                 # EDA, descriptive stats, IQR outliers, & plots
│   ├── preprocessing.py                   # Feature-target separation, split, & ColumnTransformer
│   ├── train.py                           # Stratified 5-Fold CV training, comparison, & saving
│   ├── evaluate.py                        # Metrics, confusion matrices, & ROC curves
│   └── predict.py                         # Production inference class with schema validation
├── models/
│   └── best_fire_classification_pipeline.joblib # Persisted winning Scikit-learn Pipeline
├── outputs/
│   ├── plots/                             # Generated high-resolution figures (300 DPI)
│   │   ├── target_class_distribution.png
│   │   ├── missing_values_summary.png
│   │   ├── numerical_feature_distributions.png
│   │   ├── correlation_heatmap.png
│   │   ├── outlier_boxplots.png
│   │   ├── confusion_matrix_logistic_regression.png
│   │   ├── confusion_matrix_decision_tree.png
│   │   ├── confusion_matrix_random_forest.png
│   │   ├── confusion_matrix_xgboost.png
│   │   ├── roc_curves_comparison.png
│   │   └── model_comparison_chart.png
│   ├── metrics/
│   │   └── model_comparison.csv           # Model benchmark metrics
│   └── reports/
│       └── classification_report.txt      # Text classification reports and rationale
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py              # Automated tests for preprocessing & leak-prevention
│   └── test_prediction.py                 # Automated tests for predictor & input validation
├── requirements.txt                       # Project dependencies
├── README.md                              # Comprehensive documentation
└── main.py                                # Master orchestrator script
```

---

## 3. Dataset & Feature Descriptions

### Input Features
| Feature | Type | Description |
|---|---|---|
| **Temperature** | Numeric | Noon temperature in degrees Celsius (22°C to 42°C) |
| **RH** | Numeric | Relative Humidity percentage (21% to 90%) |
| **Ws** | Numeric | Wind Speed in km/h (6 to 29 km/h) |
| **Rain** | Numeric | Daily rainfall in mm (0 to 16.8 mm) |
| **FFMC** | Numeric | Fine Fuel Moisture Code (moisture content of surface litter) |
| **DMC** | Numeric | Duff Moisture Code (moisture of decomposing organic layer) |
| **DC** | Numeric | Drought Code (deep organic layers and seasonal drought) |
| **ISI** | Numeric | Initial Spread Index (rate of fire spread combining FFMC & wind) |
| **BUI** | Numeric | Buildup Index (total fuel available for combustion combining DMC & DC) |
| **FWI** | Numeric | Fire Weather Index (general fire danger rating) |
| **Region** | Categorical | Geographic region (`Bejaia` or `Sidi-Bel Abbes`) |

### Target Variable
- **Classes**: Binary label indicating wildfire occurrence.
  - `1` / `fire` : Active wildfire observed (Positive Class).
  - `0` / `not fire` : No fire observed (Negative Class).

---

## 4. Installation & Environment Setup

### Prerequisites
- Python 3.10+ (Tested on Python 3.11.9)
- `pip` package manager

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 5. How to Run

### Run the Local Web Dashboard (Browser UI)
Launch the interactive web application to test predictions in real-time and view model benchmarks and EDA plots in your browser:
```bash
python app.py
```
Then open your browser and navigate to:
👉 **`http://localhost:8000`** (or **`http://127.0.0.1:8000`**)
- **Live Inference Tab**: Adjust sliders or click scenario presets to see instant prediction badges, probabilities, and risk factors.
- **Model Benchmarks Tab**: View side-by-side performance comparison of all 4 models.
- **Visualizations Tab**: Interactive gallery with a lightbox viewer for all 11 generated high-res plots.
- **Interactive API Docs (Swagger UI)**: Available at **`http://localhost:8000/docs`**.

### Run the Complete Pipeline End-to-End (CLI)
Executes ingestion, EDA, preprocessing, model training, cross-validation, evaluation, and inference demonstration:
```bash
python main.py
```

### Run Pipeline with Custom Dataset or Options
```bash
# Provide a custom CSV path:
python main.py --dataset path/to/dataset.csv

# Skip regenerating EDA figures for faster execution:
python main.py --skip-eda
```

### Run Automated Unit Tests
```bash
pytest -v
```

### Run Standalone Predictions
```bash
python -m src.predict
```

---

## 6. Model Evaluation Results

The pipeline was executed with a reproducible Stratified 80/20 train-test split (`random_state=42`). 5-Fold Stratified Cross-Validation was performed on the training split (194 samples), followed by unbiased evaluation on the held-out test split (49 samples: 28 Fire, 21 Not Fire).

### Benchmark Comparison Table

| Classifier | CV Mean F1 (Train) | CV Std F1 | CV Mean ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| **Random Forest** *(Winner)* | **0.9767** | ±0.0255 | **0.9961** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Decision Tree** | 0.9767 | ±0.0151 | 0.9789 | 0.9796 | 0.9655 | 1.0000 | 0.9825 | 0.9762 |
| **XGBoost** | 0.9814 | ±0.0177 | 0.9919 | 0.9388 | 0.9032 | 1.0000 | 0.9492 | 0.9966 |
| **Logistic Regression** | 0.9537 | ±0.0323 | 0.9928 | 0.9388 | 0.9310 | 0.9643 | 0.9474 | 0.9932 |

### Key Findings & Winning Model Justification
1. **Zero False Negatives**: In life-critical disaster detection, a **False Negative** (failing to sound an alarm for an active wildfire) carries disastrous consequences. All tree-based models (Random Forest, Decision Tree, XGBoost) achieved **100% Recall** on the held-out test set.
2. **Top Generalization**: **Random Forest Classifier** achieved the highest test performance (**1.0000 F1-score**, **1.0000 Accuracy**, **1.0000 ROC-AUC**), perfectly segregating both fire and non-fire conditions without overfitting.
3. **Robustness**: The ensemble bagging technique in Random Forest effectively mitigates individual decision tree variance and handles non-linear relationships across meteorological codes (FFMC, ISI, FWI).

---

---

## 7. Indian States Wildfire Intelligence Dataset Integration

The system integrates official historical annual wildfire incident data across **all 35 Indian States and Union Territories** (Seasons `2010–2011`, `2009–10`, `2008–09`):

- **Data Source**: Ministry of Environment & Forests / Forest Survey of India (FSI) records stored at [`data/raw/india_forest_fires_by_state.csv`](file:///c:/Users/priya/OneDrive/Desktop/forestFireML/data/raw/india_forest_fires_by_state.csv).
- **Analytical Metrics**:
  - **National Total 3-Year Incidents**: 43,846 recorded wildfires.
  - **Peak Wildfire State**: **Mizoram** (4,675 incidents in single season; 3-Year Avg: 3,266.7).
  - **Central Sal Belt Hotspots**: **Madhya Pradesh** (2,894 peak; 2,243.7 avg) and **Chhattisgarh** (2,849 peak; 2,252.7 avg).
  - **Vulnerability Index ($0 - 100$)**: State-calibrated vulnerability normalized against national peak.
  - **4-Tier Risk Categorization**:
    * **Tier 4 (Extreme Fire-Prone)**: Avg $\ge 2,000$ incidents/yr (Mizoram, Chhattisgarh, Madhya Pradesh).
    * **Tier 3 (High Fire-Prone)**: Avg $500 - 2,000$ incidents/yr (Assam, Andhra Pradesh, Manipur, Maharashtra, Meghalaya, Nagaland, Odisha, Tripura, Jharkhand, Arunachal Pradesh, Uttarakhand).
    * **Tier 2 (Moderate Fire-Prone)**: Avg $100 - 500$ incidents/yr (Karnataka, UP, Bihar, Gujarat, Tamil Nadu, West Bengal, Himachal Pradesh, Rajasthan, Kerala).
    * **Tier 1 (Low / Monitored Zone)**: Avg $< 100$ incidents/yr (J&K, Punjab, Haryana, Andaman & Nicobar, Goa, Sikkim, Delhi, UTs).
- **Inference & UI Synergy**:
  - **Realistic Burn Extent Calibration**: Burned hectares ($ha$) predicted by the physical FWI system are calibrated by each state's empirical vulnerability score.
  - **Reserve & GPS Pinpointing**: All 35 states/UTs map directly to their primary forest reserve, sector, and latitude/longitude coordinates.
  - **Interactive Dashboard Tab**: Accessible at `http://localhost:8000/#india` with live search, tier filter pills, 35-state interactive ranking table, instant "⚡ Simulate" actions, and publication-grade ranking charts.
- **REST Endpoints**:
  - `GET /api/india/states`: Full 35-state ranked JSON data.
  - `GET /api/india/summary`: National aggregate KPI counts.
  - `GET /api/india/state/{state_name}`: Detailed single-state historical records and coordinates.

---

## 8. College Faculty & Viva Presentation Guide

When presenting this project to evaluators or faculty members, explain the architecture using this 5-stage progression:

1. **Problem Formulation**: Wildfire risk is framed as a supervised binary classification problem ($y \in \{0, 1\}$).
2. **Data Sourcing & Cleaning**: Multi-region tables with headers embedded in data rows are normalized, strip-cleaned, and typed correctly. Outliers in weather indices are retained because extreme drought/deluge are genuine physical markers of wildfire danger.
3. **Leak-Free Engineering**: Preprocessing parameters (median values, scaling means, and standard deviations) are strictly computed **inside the training folds** of the Scikit-learn Pipeline. No test information ever leaks into the training pipeline.
4. **Ensemble Modeling & Cross-Validation**: 5-Fold Stratified Cross-Validation guarantees that class proportions are preserved in each fold and verifies model stability before test evaluation.
5. **Inference Readiness**: The saved artifact (`models/best_fire_classification_pipeline.joblib`) packages the exact preprocessing transformers together with the trained estimator, ensuring that new sensor inputs can be evaluated in real time without retraining.
