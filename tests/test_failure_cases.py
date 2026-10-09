"""Failure-case tests.

The happy path is easy. These tests pin down what the system does when the
world is messy, because that is what decides whether the demo is trustworthy:

* missing required columns in an uploaded CSV     -> clear error, no crash
* sensor faults (negative / over-capacity / NaN /
  flatlined level)                                -> detected + physics fallback
* out-of-range user input                         -> rejected with a warning
* physical outliers in the raw data               -> clipped, not silently kept
* impact metric sanity                            -> the controller really saves water
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from data.generate_data import generate_dataset
from src.decision import (
    ACTION_CONSERVE,
    ACTION_HARVEST,
    ACTION_PREVENT,
    ACTION_USE,
    decide,
    detect_sensor_fault,
    physics_prediction,
    recommend_next_hour,
    simulate_impact,
    validate_level,
    validate_user_input,
)
from src.preprocess import (
    FEATURES,
    DataValidationError,
    build_live_features,
    clean_data,
    load_data,
    preprocess,
    validate_columns,
)

CAPACITY = 5000.0


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def clean_history() -> pd.DataFrame:
    """48 hours of entirely healthy sensor readings (no faults at all)."""
    n = 48
    timestamps = pd.date_range("2024-03-01 00:00", periods=n, freq="h")
    ramp = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "rainfall_mm": np.where(np.arange(n) % 7 == 0, 2.5, 0.0),
            "inflow_litres": np.where(np.arange(n) % 7 == 0, 320.0, 0.0),
            "usage_litres": 40.0 + 10.0 * np.cos(ramp / 4.0),
            "tank_level_litres": 2600.0 + 120.0 * np.sin(ramp / 5.0),
            "tank_capacity_litres": CAPACITY,
            "temperature_c": 18.0 + 3.0 * np.sin(ramp / 8.0),
        }
    )


@pytest.fixture(scope="module")
def small_bundle():
    """A tiny but real end-to-end run used by the recommendation tests."""
    prepared = preprocess(generate_dataset(seed=7, n_days=30))
    model = RandomForestRegressor(n_estimators=25, random_state=0, n_jobs=1)
    model.fit(prepared.features[FEATURES], prepared.features["target_level_next"])
    return {"prepared": prepared, "model": model}


def _valid_current(level: float = 2600.0) -> dict:
    return {
        "tank_level_litres": level,
        "rainfall_mm": 0.0,
        "inflow_litres": 0.0,
        "usage_litres": 45.0,
        "temperature_c": 18.0,
    }


# --------------------------------------------------------------------------
# 1. Missing columns
# --------------------------------------------------------------------------
def test_validate_columns_reports_missing_names():
    df = pd.DataFrame({"timestamp": ["2024-01-01"], "rainfall_mm": [0.0]})
    missing = validate_columns(df)
    assert "tank_level_litres" in missing
    assert "inflow_litres" in missing
    assert "usage_litres" in missing


def test_missing_required_column_raises_clear_error(clean_history):
    df = clean_history.drop(columns=["tank_level_litres"])
    with pytest.raises(DataValidationError) as excinfo:
        load_data(df)
    message = str(excinfo.value)
    assert "tank_level_litres" in message
    assert "Missing required column" in message


def test_preprocess_rejects_csv_without_usage_column(clean_history):
    df = clean_history.drop(columns=["usage_litres"])
    with pytest.raises(DataValidationError):
        preprocess(df)


def test_optional_columns_are_defaulted(clean_history):
    df = clean_history.drop(columns=["temperature_c", "tank_capacity_litres"])
    loaded = load_data(df)  # must not raise
    assert "temperature_c" in loaded.columns
    assert loaded["tank_capacity_litres"].iloc[0] == CAPACITY


# --------------------------------------------------------------------------
# 2. Sensor faults -> physics fallback
# --------------------------------------------------------------------------
def test_healthy_history_reports_no_fault(clean_history):
    check = detect_sensor_fault(clean_history, CAPACITY)
    assert check.fault is False, check.message
    assert check.reasons == []


def test_negative_level_is_detected_as_sensor_fault(clean_history):
    faulty = clean_history.copy()
    faulty.loc[faulty.index[-2], "tank_level_litres"] = -320.0
    check = detect_sensor_fault(faulty, CAPACITY)
    assert check.fault is True
    assert any("negative" in reason for reason in check.reasons)


def test_level_above_capacity_is_detected_as_sensor_fault(clean_history):
    faulty = clean_history.copy()
    faulty.loc[faulty.index[-1], "tank_level_litres"] = CAPACITY + 750.0
    check = detect_sensor_fault(faulty, CAPACITY)
    assert check.fault is True
    assert any("above capacity" in reason for reason in check.reasons)


def test_nan_reading_is_detected_as_sensor_fault(clean_history):
    faulty = clean_history.copy()
    faulty.loc[faulty.index[-3], "tank_level_litres"] = np.nan
    check = detect_sensor_fault(faulty, CAPACITY)
    assert check.fault is True
    assert any("missing" in reason for reason in check.reasons)


def test_flatlined_sensor_is_detected(clean_history):
    faulty = clean_history.copy()
    faulty.loc[faulty.index[-8:], "tank_level_litres"] = 1234.0  # frozen float
    check = detect_sensor_fault(faulty, CAPACITY)
    assert check.fault is True
    assert check.flatline_run >= 6
    assert any("frozen" in reason for reason in check.reasons)


def test_empty_history_is_a_fault():
    check = detect_sensor_fault(pd.DataFrame(), CAPACITY)
    assert check.fault is True


def test_fallback_is_triggered_and_uses_physics_baseline(clean_history, small_bundle):
    faulty = clean_history.copy()
    faulty.loc[faulty.index[-1], "tank_level_litres"] = -100.0  # sensor fault

    current = _valid_current(level=2600.0)
    result = recommend_next_hour(
        small_bundle["model"], faulty, current, CAPACITY
    )

    expected = physics_prediction(
        current_level=2600.0,
        inflow_litres=current["inflow_litres"],
        average_usage_litres=float(
            build_live_features(clean_history, current, CAPACITY)["usage_roll_mean_24"].iloc[0]
        ),
        capacity=CAPACITY,
    )

    assert result.used_fallback is True
    assert result.fallback_reason is not None
    assert result.predicted_level == pytest.approx(expected)
    assert "Sensor fault" in _fallback_banner(result)


def _fallback_banner(result) -> str:
    """Mirror of the banner the Streamlit app shows."""
    return (
        "Sensor fault suspected - using fallback baseline"
        if result.used_fallback
        else "Model prediction"
    )


def test_healthy_history_uses_the_model_not_the_fallback(clean_history, small_bundle):
    result = recommend_next_hour(
        small_bundle["model"], clean_history, _valid_current(), CAPACITY
    )
    assert result.used_fallback is False
    assert 0.0 <= result.predicted_level <= CAPACITY


def test_recommend_raises_on_invalid_current_level(clean_history, small_bundle):
    bad = _valid_current(level=-50.0)
    with pytest.raises(ValueError, match="Invalid input"):
        recommend_next_hour(small_bundle["model"], clean_history, bad, CAPACITY)


# --------------------------------------------------------------------------
# 3. Out-of-range user input
# --------------------------------------------------------------------------
def test_validate_level_bounds():
    assert validate_level(2500.0, CAPACITY)[0] is True
    assert validate_level(-1.0, CAPACITY)[0] is False
    assert validate_level(CAPACITY + 1.0, CAPACITY)[0] is False
    assert validate_level(float("nan"), CAPACITY)[0] is False


def test_validate_user_input_rejects_out_of_range_values():
    assert validate_user_input(2500, 1.0, 100.0, 40.0, CAPACITY) == []

    errors = validate_user_input(CAPACITY + 500, 1.0, 100.0, 40.0, CAPACITY)
    assert any("exceeds tank capacity" in error for error in errors)

    errors = validate_user_input(-10.0, 1.0, 100.0, 40.0, CAPACITY)
    assert any("negative" in error for error in errors)

    errors = validate_user_input(float("nan"), 1.0, 100.0, 40.0, CAPACITY)
    assert any("real number" in error for error in errors)

    errors = validate_user_input(2500, -2.0, -5.0, -3.0, CAPACITY)
    assert len(errors) == 3

    errors = validate_user_input(2500, 1.0, 90_000.0, 40.0, CAPACITY)
    assert any("implausible" in error for error in errors)


# --------------------------------------------------------------------------
# 4. Decision rules
# --------------------------------------------------------------------------
def test_decision_high_level_prefers_overflow_prevention():
    assert decide(fill_pct=95.0).action == ACTION_PREVENT


def test_decision_middle_band_is_harvest():
    assert decide(fill_pct=60.0).action == ACTION_HARVEST


def test_decision_low_band_is_conserve():
    assert decide(fill_pct=12.0).action == ACTION_CONSERVE


def test_decision_uses_rain_forecast_to_trigger_early():
    # 80% full and dry -> not an emergency.
    assert decide(fill_pct=80.0, rain_roll_3=0.0, rain_roll_24=0.0).action != ACTION_PREVENT
    # Same level, but a wet 3 hours already in the tank's path -> act now.
    assert decide(fill_pct=80.0, rain_roll_3=6.0, rain_roll_24=30.0).action == ACTION_PREVENT


def test_decision_draws_water_when_wet_and_filling():
    assert decide(fill_pct=65.0, rain_roll_3=2.0, rain_roll_24=20.0).action == ACTION_USE


def test_decision_bands_are_exclusive_and_ordered():
    actions = {decide(fill).action for fill in np.arange(0.0, 100.0, 1.0)}
    assert actions <= {ACTION_PREVENT, ACTION_HARVEST, ACTION_USE, ACTION_CONSERVE}


def test_physics_prediction_is_clipped_to_the_tank():
    assert physics_prediction(4900.0, 800.0, 10.0, CAPACITY) == CAPACITY
    assert physics_prediction(10.0, 0.0, 900.0, CAPACITY) == 0.0


# --------------------------------------------------------------------------
# 5. Cleaning behaviour
# --------------------------------------------------------------------------
def test_clean_data_clips_physical_outliers_and_removes_duplicates():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=8, freq="h").tolist()
            + [pd.Timestamp("2024-01-01")],  # duplicate timestamp
            "rainfall_mm": [0, 0, 1, 0, -0.5, 0, 0, 2, 5],
            "inflow_litres": [0, 100, 100, 0, 0, 0, 0, 0, 0],
            "usage_litres": [40, 45, 50, 42, 44, 41, 43, 40, 99],
            "tank_level_litres": [2500, -300, 2600, 5300, 2700, 2750, 2800, 5200, 2900],
            "tank_capacity_litres": [CAPACITY] * 9,
            "temperature_c": [12, 12, np.nan, 13, 14, 14, 15, 15, 16],
        }
    )

    cleaned, report = clean_data(df)

    assert report["duplicate_rows_removed"] == 1
    assert cleaned["tank_level_litres"].between(0, CAPACITY).all()
    assert (cleaned["rainfall_mm"] >= 0).all()
    assert cleaned["temperature_c"].isna().sum() == 0
    assert report["outliers"]["level_negative_repaired"] == 1
    assert report["outliers"]["level_over_capacity_repaired"] == 2
    assert report["outliers"]["rainfall_negative_clipped"] == 1
    assert cleaned["timestamp"].is_monotonic_increasing
    # Repair (not clamp): the -300 and 5300 readings must not survive as 0/5000
    # values that would show up as a bogus multi-thousand-litre one-hour jump.
    assert cleaned["tank_level_litres"].diff().abs().max() < CAPACITY / 2


def test_clean_data_survives_a_fully_missing_column():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
            "rainfall_mm": [0.0] * 5,
            "inflow_litres": [0.0] * 5,
            "usage_litres": [10.0] * 5,
            "tank_level_litres": [1000.0, 1050.0, np.nan, 1150.0, 1200.0],
            "tank_capacity_litres": [CAPACITY] * 5,
            "temperature_c": [np.nan] * 5,
        }
    )
    cleaned, report = clean_data(df)
    assert cleaned.isna().sum().sum() == 0
    assert report["missing_before"]["tank_level_litres"] == 1


# --------------------------------------------------------------------------
# 6. Live feature construction
# --------------------------------------------------------------------------
def test_build_live_features_matches_training_schema(clean_history):
    row = build_live_features(clean_history, _valid_current(), CAPACITY)
    assert len(row) == 1
    assert all(feature in row.columns for feature in FEATURES)
    assert row[FEATURES].isna().sum().sum() == 0
    # Rolling windows must line up with the history we fed in.
    expected_rain_3 = clean_history["rainfall_mm"].tail(2).sum() + 0.0
    assert row["rain_roll_3"].iloc[0] == pytest.approx(expected_rain_3)


# --------------------------------------------------------------------------
# 7. Impact metric
# --------------------------------------------------------------------------
def _storm_frame(hours: int = 24, start_level: float = 4800.0) -> pd.DataFrame:
    """A nearly-full tank hit by heavy inflow - a guaranteed spill."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-05-01", periods=hours, freq="h"),
            "current_level": [start_level] * hours,
            "actual_level": [CAPACITY] * hours,
            "rainfall_mm": [6.0] * hours,
            "inflow_litres": [400.0] * hours,
            "usage_litres": [50.0] * hours,
            "usage_roll_mean_24": [50.0] * hours,
            "rain_roll_3": [18.0] * hours,
            "rain_roll_24": [60.0] * hours,
            "fill_pct": [start_level / CAPACITY * 100.0] * hours,
        }
    )


def test_controller_saves_water_during_a_storm():
    frame = _storm_frame()
    impact = simulate_impact(frame, np.full(len(frame), CAPACITY), CAPACITY)

    assert impact["no_controller"]["overflow_litres"] > 0
    assert impact["litres_saved_from_overflow"] > 0
    assert impact["overflow_events_avoided"] >= 1
    assert impact["with_controller"]["overflow_litres"] <= impact["no_controller"]["overflow_litres"]


def test_conserving_avoids_shortage_when_the_tank_is_dry():
    """A dry spell: the tank drains towards the 15% reserve line.

    Without the controller it crosses the reserve and then runs dry; the
    CONSERVE mode cuts non-essential demand by 40%, so it stays above the line
    for longer and never runs out inside the window.
    """
    hours = 20
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-06-01", periods=hours, freq="h"),
            "current_level": [1000.0] * hours,  # start just above the 750 L reserve
            "actual_level": [0.0] * hours,
            "rainfall_mm": [0.0] * hours,
            "inflow_litres": [0.0] * hours,
            "usage_litres": [60.0] * hours,
            "usage_roll_mean_24": [60.0] * hours,
            "rain_roll_3": [0.0] * hours,
            "rain_roll_24": [0.0] * hours,
            "fill_pct": [20.0] * hours,
        }
    )
    impact = simulate_impact(frame, np.full(hours, 400.0), CAPACITY)  # 8% -> CONSERVE

    assert impact["with_controller"]["shortage_hours"] < impact["no_controller"]["shortage_hours"]
    assert impact["unmet_demand_avoided_litres"] > 0
    assert impact["litres_conserved_by_restricting_demand"] > 0
    assert impact["with_controller"]["levels"][-1] > impact["no_controller"]["levels"][-1]


def test_impact_dict_is_complete_and_non_negative(small_bundle, ):
    bundle_impact = simulate_impact(
        small_bundle["prepared"].features.assign(
            current_level=small_bundle["prepared"].features["tank_level_litres"],
            actual_level=small_bundle["prepared"].features["target_level_next"],
            pred=small_bundle["prepared"].features["tank_level_litres"],
        ),
        small_bundle["prepared"].features["tank_level_litres"].to_numpy(),
        small_bundle["prepared"].capacity,
    )
    for key in (
        "litres_saved_from_overflow",
        "overflow_events_avoided",
        "shortage_events_avoided",
        "overflow_reduction_pct",
    ):
        assert key in bundle_impact
        assert np.isfinite(bundle_impact[key])

    assert bundle_impact["no_controller"]["overflow_litres"] >= 0
    assert bundle_impact["with_controller"]["overflow_litres"] >= 0
    assert len(bundle_impact["with_controller"]["actions"]) == len(
        small_bundle["prepared"].features
    )


# --------------------------------------------------------------------------
# 8. End-to-end smoke test on the messy generator output
# --------------------------------------------------------------------------
def test_generated_dataset_is_messy_and_preprocess_cleans_it():
    raw = generate_dataset(seed=3, n_days=20)
    report_missing = int(raw["tank_level_litres"].isna().sum())
    assert report_missing > 0, "the generator should inject missing values"
    assert raw.duplicated(subset=["timestamp"]).sum() > 0, "the generator should inject duplicates"

    prepared = preprocess(raw)
    assert prepared.clean["tank_level_litres"].between(0, prepared.capacity).all()
    assert prepared.clean.isna().sum().sum() == 0
    assert prepared.clean["timestamp"].is_monotonic_increasing
    assert len(prepared.features) > 100
    assert prepared.summary["duplicate_rows_removed"] > 0
