"""Data inspection and Exploratory Data Analysis (EDA) module.

Performs:
- Dataset dimension and data type profiling.
- Missing-value calculation and reporting.
- Duplicate detection and removal reporting.
- Descriptive statistics generation.
- Outlier detection via the Interquartile Range (IQR) method.
- Generation and saving of publication-quality visualizations into outputs/plots/.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    PLOTS_DIR,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    POSITIVE_CLASS_NAME,
)

# Apply a clean, modern aesthetic for plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.titlesize": 14,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


def inspect_dimensions_and_types(df: pd.DataFrame) -> pd.DataFrame:
    """Return a detailed summary DataFrame of column data types and non-null counts."""
    summary = pd.DataFrame({
        "Data_Type": df.dtypes.astype(str),
        "Non_Null_Count": df.notna().sum(),
        "Null_Count": df.isna().sum(),
        "Null_Percentage": (df.isna().sum() / len(df) * 100).round(2),
        "Unique_Values": df.nunique(),
    })
    return summary


def detect_and_handle_duplicates(df: pd.DataFrame, remove: bool = False) -> Tuple[pd.DataFrame, int]:
    """Detect and optionally remove duplicate rows.

    Args:
        df: Input DataFrame.
        remove: Whether to drop duplicates.

    Returns:
        Tuple of (DataFrame, count of duplicate rows).
    """
    dup_mask = df.duplicated()
    dup_count = int(dup_mask.sum())
    print(f"\n[Inspection] Duplicate Rows Detected: {dup_count}")

    if dup_count > 0 and remove:
        df_cleaned = df.drop_duplicates().reset_index(drop=True)
        print(f"[Inspection] Removed {dup_count} duplicate rows. New shape: {df_cleaned.shape}")
        return df_cleaned, dup_count

    return df, dup_count


def generate_descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute comprehensive descriptive statistics for numerical columns."""
    cols = [col for col in NUMERICAL_FEATURES if col in df.columns]
    desc = df[cols].describe().T
    desc["IQR"] = desc["75%"] - desc["25%"]
    return desc.round(3)


def detect_outliers_iqr(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Detect numerical outliers using the 1.5 * IQR method.

    Returns a dictionary mapping column names to outlier count, lower bound, and upper bound.
    """
    outlier_summary = {}
    cols = [col for col in NUMERICAL_FEATURES if col in df.columns]

    print("\n" + "=" * 70)
    print("OUTLIER DETECTION (1.5 * IQR METHOD)")
    print("=" * 70)
    print(f"{'Feature':<15} | {'Q1':<8} | {'Q3':<8} | {'IQR':<8} | {'Lower':<8} | {'Upper':<8} | {'Outliers':<8}")
    print("-" * 75)

    for col in cols:
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = series[(series < lower_bound) | (series > upper_bound)]
        outlier_count = len(outliers)

        outlier_summary[col] = {
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outlier_count": outlier_count,
            "outlier_pct": round(outlier_count / len(series) * 100, 2),
        }

        print(
            f"{col:<15} | {q1:<8.2f} | {q3:<8.2f} | {iqr:<8.2f} | "
            f"{lower_bound:<8.2f} | {upper_bound:<8.2f} | {outlier_count:<8}"
        )

    print("=" * 70)
    print(
        "Domain Note: In wildfire science, extreme values of Rain, DMC, DC, or FWI represent genuine "
        "extreme drought or heavy deluge events. Deleting them risks losing the most critical wildfire signals. "
        "These are handled via robust scaling or tree-based ensemble partitioning rather than indiscriminate deletion."
    )
    return outlier_summary


def plot_target_distribution(df: pd.DataFrame, output_dir: Path = PLOTS_DIR) -> Path:
    """Plot and save target class distribution."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    counts = df[TARGET_COLUMN].value_counts()
    colors = ["#e74c3c" if c == POSITIVE_CLASS_NAME else "#2ecc71" for c in counts.index]

    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="black", alpha=0.85)
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 2,
            f"{int(yval)} ({yval/len(df)*100:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_title("Forest Fire Target Class Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Class Label", fontweight="bold")
    ax.set_ylabel("Record Count", fontweight="bold")
    ax.set_ylim(0, max(counts.values) * 1.15)
    plt.tight_layout()

    out_path = output_dir / "target_class_distribution.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_missing_values(df: pd.DataFrame, output_dir: Path = PLOTS_DIR) -> Path:
    """Plot missing-value percentages across all columns."""
    fig, ax = plt.subplots(figsize=(8, 4))
    null_pct = (df.isna().sum() / len(df) * 100)
    
    # If no nulls exist, display a completeness chart
    if null_pct.max() == 0:
        ax.text(
            0.5, 0.5,
            "100% Complete Dataset\nZero Missing Values Found Across All Columns",
            ha="center", va="center", fontsize=12, fontweight="bold", color="#27ae60"
        )
        ax.set_axis_off()
        ax.set_title("Data Completeness Summary", fontsize=13, fontweight="bold", pad=12)
    else:
        null_pct.plot(kind="bar", ax=ax, color="#e67e22", edgecolor="black", alpha=0.85)
        ax.set_title("Missing Value Percentage per Column", fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel("Missing %", fontweight="bold")
        ax.set_ylim(0, 100)

    plt.tight_layout()
    out_path = output_dir / "missing_values_summary.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_numerical_distributions(df: pd.DataFrame, output_dir: Path = PLOTS_DIR) -> Path:
    """Plot histograms with KDE overlays for all numerical features."""
    cols = [col for col in NUMERICAL_FEATURES if col in df.columns]
    n_cols = 3
    n_rows = int(np.ceil(len(cols) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.2 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        sns.histplot(df[col], kde=True, ax=ax, color="#2980b9", edgecolor="black", alpha=0.6)
        ax.set_title(f"Distribution: {col}", fontweight="bold", fontsize=10)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Algerian Forest Fire Feature Distributions", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    out_path = output_dir / "numerical_feature_distributions.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path = PLOTS_DIR) -> Path:
    """Plot correlation heatmap of numerical features and encoded target."""
    cols = [col for col in NUMERICAL_FEATURES if col in df.columns]
    corr_df = df[cols].copy()

    # Include encoded target if present
    if TARGET_COLUMN in df.columns:
        corr_df["Target_Fire"] = (df[TARGET_COLUMN] == POSITIVE_CLASS_NAME).astype(int)

    corr = corr_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap: Fire Weather Indices & Target", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    out_path = output_dir / "correlation_heatmap.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_outlier_boxplots(df: pd.DataFrame, output_dir: Path = PLOTS_DIR) -> Path:
    """Generate box plots for all numerical features to visualize outliers."""
    cols = [col for col in NUMERICAL_FEATURES if col in df.columns]
    n_cols = 5
    n_rows = int(np.ceil(len(cols) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        sns.boxplot(y=df[col], ax=ax, color="#f39c12", width=0.4, fliersize=4)
        ax.set_title(col, fontweight="bold", fontsize=10)
        ax.set_ylabel("")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Outlier Boxplots for Numerical Features (IQR Bounds)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    out_path = output_dir / "outlier_boxplots.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def run_full_eda(df: pd.DataFrame) -> Dict[str, Path]:
    """Execute the full data inspection and EDA suite, generating reports and plots."""
    print("\n" + "=" * 70)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    # 1. Dimensions and Types
    summary = inspect_dimensions_and_types(df)
    print("\n[EDA] Feature Inspection:")
    print(summary.to_string())

    # 2. Duplicate Detection
    df_clean, dup_count = detect_and_handle_duplicates(df, remove=False)

    # 3. Descriptive Stats
    desc = generate_descriptive_statistics(df_clean)
    print("\n[EDA] Descriptive Statistics:")
    print(desc.to_string())

    # 4. Outlier Analysis
    outliers = detect_outliers_iqr(df_clean)

    # 5. Visualizations
    plots = {
        "target_dist": plot_target_distribution(df_clean),
        "missing_summary": plot_missing_values(df_clean),
        "num_distributions": plot_numerical_distributions(df_clean),
        "correlation": plot_correlation_heatmap(df_clean),
        "outliers": plot_outlier_boxplots(df_clean),
    }

    print("\n[EDA] Generated and saved EDA plots:")
    for name, path in plots.items():
        print(f"  - {name:20s}: {path}")

    return plots


if __name__ == "__main__":
    from src.data_loader import load_raw_dataset
    df = load_raw_dataset()
    run_full_eda(df)
