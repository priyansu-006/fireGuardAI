"""Model training, cross-validation, and selection module for Forest Fire Classification.

Builds end-to-end Scikit-learn pipelines, trains multiple candidate classifiers:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. XGBoost Classifier (with graceful fallback if unavailable)

Evaluates via Stratified 5-Fold Cross Validation and held-out test data,
selects the top-performing pipeline by F1-Score, and persists the pipeline with joblib.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from src.config import (
    BEST_MODEL_PATH,
    MODEL_COMPARISON_CSV,
    RANDOM_STATE,
    CV_FOLDS,
    PRIMARY_METRIC,
    POSITIVE_CLASS_NAME,
    NEGATIVE_CLASS_NAME,
)
from src.preprocessing import build_preprocessor
from src.evaluate import (
    compute_classification_metrics,
    plot_single_confusion_matrix,
    plot_comparative_roc_curves,
    plot_model_comparison_bar_chart,
    save_comprehensive_evaluation_report,
)


def get_candidate_models(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """Instantiate candidate classification algorithms with reproducible seeds."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=random_state,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=4,
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=4,
            random_state=random_state,
        ),
    }

    # XGBoost with graceful fallback
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            eval_metric="logloss",
            random_state=random_state,
        )
    except ImportError:
        print("[Training] XGBoost is not installed. Proceeding smoothly with other classifiers.")
    except Exception as e:
        print(f"[Training] XGBoost initialization warning ({e}). Proceeding without XGBoost.")

    return models


def build_pipeline_for_model(classifier: Any) -> Pipeline:
    """Combine the feature preprocessor and classifier into a single leak-free Pipeline."""
    preprocessor = build_preprocessor()
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline


def train_and_compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    primary_metric: str = PRIMARY_METRIC,
) -> Tuple[Pipeline, str, pd.DataFrame]:
    """Train candidate pipelines, perform cross-validation, evaluate on test set, and save the best model."""
    candidate_models = get_candidate_models()
    trained_pipelines: Dict[str, Pipeline] = {}
    comparison_records = []
    detailed_reports = {}

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print("\n" + "=" * 75)
    print("MODEL TRAINING & CROSS-VALIDATION BENCHMARK")
    print("=" * 75)

    for name, clf in candidate_models.items():
        print(f"\n[Training] Training pipeline for: {name}...")
        pipeline = build_pipeline_for_model(clf)

        # 1. Stratified Cross-Validation on Training Data (prevents data leakage)
        cv_f1_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
        cv_roc_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
        mean_cv_f1 = float(np.mean(cv_f1_scores))
        std_cv_f1 = float(np.std(cv_f1_scores))
        mean_cv_roc = float(np.mean(cv_roc_scores))

        # 2. Fit full pipeline on entire training partition
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        # 3. Predict on held-out test partition
        y_test_pred = pipeline.predict(X_test)
        y_test_proba = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None

        # 4. Compute Test Metrics
        test_metrics = compute_classification_metrics(y_test, y_test_pred, y_test_proba)

        # 5. Confusion Matrix Plot
        plot_single_confusion_matrix(y_test, y_test_pred, name)

        # 6. Detailed Classification Report String
        rep = classification_report(
            y_test,
            y_test_pred,
            target_names=[NEGATIVE_CLASS_NAME.title(), POSITIVE_CLASS_NAME.title()],
            digits=4,
        )
        detailed_reports[name] = rep

        comparison_records.append({
            "Model": name,
            "CV Mean F1 (Train)": round(mean_cv_f1, 4),
            "CV Std F1": round(std_cv_f1, 4),
            "CV Mean ROC-AUC": round(mean_cv_roc, 4),
            "Test Accuracy": test_metrics["accuracy"],
            "Test Precision": test_metrics["precision"],
            "Test Recall": test_metrics["recall"],
            "Test F1-Score": test_metrics["f1"],
            "Test ROC-AUC": test_metrics["roc_auc"],
        })

        print(
            f"  -> CV F1: {mean_cv_f1:.4f} (+/- {std_cv_f1:.4f}) | "
            f"Test Acc: {test_metrics['accuracy']:.4f} | "
            f"Test Recall: {test_metrics['recall']:.4f} | "
            f"Test F1: {test_metrics['f1']:.4f} | "
            f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}"
        )

    # Compile Comparison DataFrame
    comparison_df = pd.DataFrame(comparison_records)

    # Sort models by primary evaluation metric (e.g. Test F1-Score)
    metric_col_map = {
        "f1": "Test F1-Score",
        "roc_auc": "Test ROC-AUC",
        "accuracy": "Test Accuracy",
        "recall": "Test Recall",
    }
    sort_column = metric_col_map.get(primary_metric.lower(), "Test F1-Score")
    comparison_df = comparison_df.sort_values(by=sort_column, ascending=False).reset_index(drop=True)

    print("\n" + "=" * 75)
    print("FINAL BENCHMARK COMPARISON TABLE")
    print("=" * 75)
    print(comparison_df.to_string(index=False))

    # Identify Winning Model
    best_row = comparison_df.iloc[0]
    best_model_name = best_row["Model"]
    best_pipeline = trained_pipelines[best_model_name]

    print("\n" + "*" * 75)
    print(f"WINNING MODEL: {best_model_name}")
    print(f"Criterion: Highest {sort_column} ({best_row[sort_column]:.4f})")
    print("*" * 75)

    # Save Model Comparison CSV
    MODEL_COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(MODEL_COMPARISON_CSV, index=False)
    print(f"[Training] Saved model comparison metrics to {MODEL_COMPARISON_CSV}")

    # Generate Evaluation Visualizations
    plot_comparative_roc_curves(trained_pipelines, X_test, y_test)
    plot_model_comparison_bar_chart(comparison_df)

    # Save Full Evaluation Report
    save_comprehensive_evaluation_report(
        comparison_df,
        detailed_reports,
        best_model_name,
        primary_metric,
    )

    # Save Winning Pipeline (including Preprocessing Transformers + Model)
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    print(f"[Training] Saved winning pipeline (preprocessor + model) to: {BEST_MODEL_PATH}")

    return best_pipeline, best_model_name, comparison_df


if __name__ == "__main__":
    from src.data_loader import load_raw_dataset
    from src.preprocessing import separate_features_and_target, split_data

    raw_df = load_raw_dataset()
    X, y = separate_features_and_target(raw_df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    best_pipeline, best_name, df_comp = train_and_compare_models(X_train, y_train, X_test, y_test)
