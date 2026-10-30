"""Model evaluation and visualization module for Forest Fire Classification.

Calculates:
- Accuracy, Precision, Recall, F1-Score, ROC-AUC.
- Detailed Scikit-learn Classification Reports.
- Confusion Matrix Heatmaps.
- Multi-Model Comparative ROC Curves.
- Multi-Metric Comparison Bar Charts.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

from src.config import (
    PLOTS_DIR,
    METRICS_DIR,
    REPORTS_DIR,
    POSITIVE_CLASS_NAME,
    NEGATIVE_CLASS_NAME,
    MODEL_COMPARISON_CSV,
    CLASSIFICATION_REPORT_TXT,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute all standard classification performance metrics.

    Explicitly treats label 1 ('fire') as the positive class.
    """
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
    }

    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            # If 2D probability array passed, extract column 1 (positive class)
            if y_proba.ndim == 2:
                prob_positive = y_proba[:, 1]
            else:
                prob_positive = y_proba
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, prob_positive)), 4)
        except Exception:
            metrics["roc_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan

    return metrics


def plot_single_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: Path = PLOTS_DIR,
) -> Path:
    """Plot and save a styled confusion matrix heatmap for a single model."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    class_labels = [NEGATIVE_CLASS_NAME.title(), POSITIVE_CLASS_NAME.title()]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar=False,
        ax=ax,
        annot_kws={"size": 13, "weight": "bold"},
    )
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Ground Truth Label", fontweight="bold")
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
    out_path = output_dir / f"confusion_matrix_{safe_name}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_comparative_roc_curves(
    model_pipelines: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path = PLOTS_DIR,
) -> Path:
    """Plot comparative ROC curves for all evaluated models on a single figure."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]

    for i, (name, pipeline) in enumerate(model_pipelines.items()):
        color = colors[i % len(colors)]
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_val = roc_auc_score(y_test, y_proba)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", lw=2, color=color)
        elif hasattr(pipeline, "decision_function"):
            scores = pipeline.decision_function(X_test)
            fpr, tpr, _ = roc_curve(y_test, scores)
            auc_val = roc_auc_score(y_test, scores)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", lw=2, color=color)

    # Diagonal baseline (random guess)
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.500)", alpha=0.7)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()

    out_path = output_dir / "roc_curves_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_model_comparison_bar_chart(
    comparison_df: pd.DataFrame,
    output_dir: Path = PLOTS_DIR,
) -> Path:
    """Generate a multi-metric grouped bar chart comparing all candidate classifiers."""
    metrics_to_plot = ["Test Accuracy", "Test Precision", "Test Recall", "Test F1-Score", "Test ROC-AUC"]
    available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]

    df_melted = comparison_df.melt(
        id_vars=["Model"],
        value_vars=available_metrics,
        var_name="Metric",
        value_name="Score",
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(
        data=df_melted,
        x="Metric",
        y="Score",
        hue="Model",
        palette="tab10",
        edgecolor="black",
        ax=ax,
    )

    ax.set_ylim(0.0, 1.15)
    ax.set_title("Model Performance Comparison on Held-Out Test Set", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Evaluation Metric", fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold")
    ax.legend(title="Classifier", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    # Annotate score text on top of bars
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=7.5,
                rotation=45,
                xytext=(0, 2),
                textcoords="offset points",
            )

    plt.tight_layout()
    out_path = output_dir / "model_comparison_chart.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def save_comprehensive_evaluation_report(
    comparison_df: pd.DataFrame,
    detailed_reports: Dict[str, str],
    best_model_name: str,
    selection_metric: str,
    report_path: Path = CLASSIFICATION_REPORT_TXT,
) -> Path:
    """Save an aggregated evaluation report containing tables, reports, and rationale."""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("FOREST FIRE CLASSIFICATION PIPELINE: MODEL EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. MODEL BENCHMARK COMPARISON TABLE (Held-Out Test Set)\n")
        f.write("-" * 80 + "\n")
        f.write(comparison_df.to_string(index=False) + "\n\n")

        f.write("2. WINNING MODEL SELECTION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Best Model Selected : {best_model_name}\n")
        f.write(f"Selection Criterion : Highest {selection_metric.upper()} on held-out test data\n")
        f.write(
            "Engineering Justification:\n"
            "In wildfire detection systems, False Negatives (predicting 'not fire' when a real wildfire\n"
            "is raging) lead to catastrophic ecological and human loss. The F1-score provides a balanced\n"
            "harmonic mean of Precision and Recall, avoiding accuracy illusions caused by slight class imbalance.\n\n"
        )

        f.write("3. DETAILED CLASSIFICATION REPORTS PER MODEL\n")
        f.write("-" * 80 + "\n")
        for model_name, rep_str in detailed_reports.items():
            f.write(f"\nModel: {model_name}\n")
            f.write("." * 40 + "\n")
            f.write(rep_str + "\n")

        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")

    print(f"[Evaluation] Full evaluation report saved to {report_path}")
    return report_path
