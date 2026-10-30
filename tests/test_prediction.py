"""Automated tests for prediction module, input validation, and inference accuracy."""

import pytest
import numpy as np
import pandas as pd

from src.config import BEST_MODEL_PATH
from src.predict import ForestFirePredictor


@pytest.fixture(scope="module")
def predictor():
    """Ensure trained model exists and return a ForestFirePredictor instance."""
    assert BEST_MODEL_PATH.exists(), f"Model file must exist at {BEST_MODEL_PATH} before running inference tests"
    return ForestFirePredictor(BEST_MODEL_PATH)


def test_predict_single_sample(predictor):
    """Test inference on a single sample dictionary."""
    sample = {
        "Temperature": 35.0,
        "RH": 45.0,
        "Ws": 16.0,
        "Rain": 0.0,
        "FFMC": 90.0,
        "DMC": 20.0,
        "DC": 50.0,
        "ISI": 8.0,
        "BUI": 22.0,
        "FWI": 12.0,
        "Region": "Bejaia",
    }
    result = predictor.predict(sample)
    assert len(result) == 1
    pred = result[0]

    assert pred["prediction_code"] in [0, 1]
    assert pred["predicted_class"] in ["fire", "not fire"]
    assert 0.0 <= pred["confidence"] <= 1.0
    assert 0.0 <= pred["probability_fire"] <= 1.0
    assert 0.0 <= pred["probability_not_fire"] <= 1.0
    assert np.isclose(pred["probability_fire"] + pred["probability_not_fire"], 1.0)


def test_predict_batch_samples(predictor):
    """Test inference on multiple records simultaneously."""
    batch = [
        {
            "Temperature": 38.0,
            "RH": 30.0,
            "Ws": 18.0,
            "Rain": 0.0,
            "FFMC": 93.0,
            "DMC": 30.0,
            "DC": 80.0,
            "ISI": 12.0,
            "BUI": 32.0,
            "FWI": 20.0,
            "Region": "Bejaia",
        },
        {
            "Temperature": 22.0,
            "RH": 90.0,
            "Ws": 12.0,
            "Rain": 15.0,
            "FFMC": 35.0,
            "DMC": 1.5,
            "DC": 5.0,
            "ISI": 0.2,
            "BUI": 2.0,
            "FWI": 0.0,
            "Region": "Sidi-Bel Abbes",
        },
    ]
    results = predictor.predict(batch)
    assert len(results) == 2
    assert results[0]["predicted_class"] == "fire"
    assert results[1]["predicted_class"] == "not fire"


def test_missing_features_raises_error(predictor):
    """Test that input missing essential features throws a clear, descriptive ValueError."""
    incomplete_sample = {
        "Temperature": 30.0,
        "RH": 60.0,
        # Ws, Rain, FFMC, etc. missing
        "Region": "Bejaia",
    }
    with pytest.raises(ValueError, match="Input is missing required feature"):
        predictor.predict(incomplete_sample)


def test_area_and_location_prediction(predictor):
    """Verify that predictions contain geographic area and burned hectares estimation."""
    fire_sample = {
        "Temperature": 38.0,
        "RH": 30.0,
        "Ws": 18.0,
        "Rain": 0.0,
        "FFMC": 92.0,
        "DMC": 26.0,
        "DC": 70.0,
        "ISI": 11.0,
        "BUI": 28.0,
        "FWI": 16.0,
        "Region": "Bejaia",
    }
    result = predictor.predict(fire_sample)[0]

    assert "predicted_area_name" in result
    assert "target_forest_sector" in result
    assert "estimated_burned_area_ha" in result
    assert "spatial_coordinates" in result

    # When fire occurs, burned area should be strictly positive
    assert result["estimated_burned_area_ha"] > 0.0
    assert "Gouraya" in result["predicted_area_name"] or "Bejaia" in result["predicted_area_name"]
    assert "lat" in result["spatial_coordinates"] and "lon" in result["spatial_coordinates"]

    safe_sample = {
        "Temperature": 22.0,
        "RH": 88.0,
        "Ws": 12.0,
        "Rain": 10.0,
        "FFMC": 40.0,
        "DMC": 2.0,
        "DC": 5.0,
        "ISI": 0.3,
        "BUI": 2.0,
        "FWI": 0.1,
        "Region": "Bejaia",
    }
    safe_result = predictor.predict(safe_sample)[0]
    assert safe_result["estimated_burned_area_ha"] == 0.0


def test_indian_state_fire_and_vulnerability_profile(predictor):
    """Verify that predictions on Indian states calibrate burned area and enrich output with official incident stats."""
    mizoram_sample = {
        "Temperature": 39.0,
        "RH": 30.0,
        "Ws": 18.0,
        "Rain": 0.0,
        "FFMC": 94.0,
        "DMC": 32.0,
        "DC": 90.0,
        "ISI": 13.0,
        "BUI": 35.0,
        "FWI": 22.0,
        "Region": "Mizoram",
    }
    res = predictor.predict(mizoram_sample)[0]
    assert res["predicted_class"] == "fire"
    assert res["is_indian_state"] is True
    assert res["peak_annual_fires"] == 4675
    assert "Mizoram" in res["predicted_area_name"]
    assert res["vulnerability_score"] == 100.0
    assert "lat" in res["spatial_coordinates"]
    assert res["estimated_burned_area_ha"] > 0.0

    uttarakhand_safe = {
        "Temperature": 20.0,
        "RH": 85.0,
        "Ws": 10.0,
        "Rain": 12.0,
        "FFMC": 38.0,
        "DMC": 3.0,
        "DC": 8.0,
        "ISI": 0.5,
        "BUI": 3.0,
        "FWI": 0.2,
        "Region": "Uttarakhand",
    }
    res_safe = predictor.predict(uttarakhand_safe)[0]
    assert res_safe["predicted_class"] == "not fire"
    assert res_safe["is_indian_state"] is True
    assert res_safe["peak_annual_fires"] == 855
    assert "Uttarakhand" in res_safe["predicted_area_name"]
    assert res_safe["estimated_burned_area_ha"] == 0.0

