"""Automated tests for preprocessing, leak prevention, and feature engineering."""

import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    RANDOM_STATE,
)
from src.preprocessing import (
    separate_features_and_target,
    split_data,
    build_preprocessor,
)


@pytest.fixture
def sample_raw_dataframe():
    """Create a synthetic DataFrame mimicking Algerian Forest Fire schema."""
    data = {
        "day": ["01", "02", "03", "04", "05"],
        "month": ["06", "06", "06", "06", "06"],
        "year": ["2012", "2012", "2012", "2012", "2012"],
        "Temperature": [29.0, 32.0, 26.0, 35.0, 30.0],
        "RH": [57.0, 60.0, 82.0, 40.0, 65.0],
        "Ws": [18.0, 14.0, 22.0, 13.0, 16.0],
        "Rain": [0.0, 0.2, 12.0, 0.0, 1.5],
        "FFMC": [65.7, 85.0, 47.1, 91.0, 75.0],
        "DMC": [3.4, 9.0, 2.5, 20.0, 8.0],
        "DC": [7.6, 25.0, 7.1, 60.0, 20.0],
        "ISI": [1.3, 5.0, 0.3, 10.0, 3.0],
        "BUI": [3.4, 9.5, 2.7, 21.0, 8.5],
        "FWI": [0.5, 6.0, 0.1, 14.0, 4.0],
        "Classes": ["not fire", "fire", "not fire", "fire", "not fire"],
        "Region": ["Bejaia", "Bejaia", "Sidi-Bel Abbes", "Bejaia", "Sidi-Bel Abbes"],
    }
    return pd.DataFrame(data)


def test_separate_features_and_target(sample_raw_dataframe):
    """Test that X and y are separated correctly and target is mapped to {0, 1}."""
    X, y = separate_features_and_target(sample_raw_dataframe)

    assert TARGET_COLUMN not in X.columns
    assert "day" not in X.columns
    assert "month" not in X.columns
    assert "year" not in X.columns

    for num_col in NUMERICAL_FEATURES:
        assert num_col in X.columns
    for cat_col in CATEGORICAL_FEATURES:
        assert cat_col in X.columns

    assert set(y.unique()).issubset({0, 1})
    assert len(y) == len(sample_raw_dataframe)


def test_preprocessor_handles_missing_values_and_scales():
    """Verify that numerical imputer and scaler handle missing values seamlessly."""
    preprocessor = build_preprocessor()

    # Training data with no NaNs
    train_df = pd.DataFrame({
        "Temperature": [30.0, 32.0, 34.0],
        "RH": [50.0, 55.0, 60.0],
        "Ws": [15.0, 14.0, 16.0],
        "Rain": [0.0, 0.0, 0.0],
        "FFMC": [80.0, 85.0, 88.0],
        "DMC": [10.0, 12.0, 14.0],
        "DC": [30.0, 35.0, 40.0],
        "ISI": [5.0, 6.0, 7.0],
        "BUI": [10.0, 12.0, 14.0],
        "FWI": [5.0, 6.5, 8.0],
        "Region": ["Bejaia", "Sidi-Bel Abbes", "Bejaia"],
    })

    # Fit preprocessor on train
    preprocessor.fit(train_df)

    # Test data containing missing values (NaN) and unseen category
    test_df = pd.DataFrame({
        "Temperature": [np.nan, 33.0],
        "RH": [52.0, np.nan],
        "Ws": [15.0, 15.0],
        "Rain": [0.0, np.nan],
        "FFMC": [82.0, 84.0],
        "DMC": [11.0, 13.0],
        "DC": [32.0, 38.0],
        "ISI": [5.5, 6.5],
        "BUI": [11.0, 13.0],
        "FWI": [5.5, 7.0],
        "Region": ["UnknownRegion", "Bejaia"],  # Unknown category test
    })

    transformed = preprocessor.transform(test_df)
    assert not np.isnan(transformed).any(), "Transformed output must not contain NaNs"
    assert transformed.shape[0] == 2
    assert transformed.shape[1] >= 11  # 10 numeric features + one-hot columns


def test_no_data_leakage():
    """Verify that scaler parameters are computed strictly from training split."""
    preprocessor = build_preprocessor()

    train_data = pd.DataFrame({
        "Temperature": [10.0, 20.0, 30.0],
        "RH": [40.0, 50.0, 60.0],
        "Ws": [10.0, 12.0, 14.0],
        "Rain": [0.0, 0.0, 0.0],
        "FFMC": [70.0, 80.0, 90.0],
        "DMC": [5.0, 10.0, 15.0],
        "DC": [20.0, 30.0, 40.0],
        "ISI": [2.0, 4.0, 6.0],
        "BUI": [5.0, 10.0, 15.0],
        "FWI": [2.0, 4.0, 6.0],
        "Region": ["Bejaia", "Bejaia", "Sidi-Bel Abbes"],
    })

    test_data = pd.DataFrame({
        "Temperature": [100.0],  # Outlier in test set
        "RH": [90.0],
        "Ws": [30.0],
        "Rain": [10.0],
        "FFMC": [95.0],
        "DMC": [50.0],
        "DC": [100.0],
        "ISI": [20.0],
        "BUI": [50.0],
        "FWI": [25.0],
        "Region": ["Bejaia"],
    })

    preprocessor.fit(train_data)
    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]

    # Mean of training Temperature is 20.0
    assert np.isclose(scaler.mean_[0], 20.0), "Scaler mean must reflect training distribution only"
