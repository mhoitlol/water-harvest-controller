"""Loading, cleaning, outlier handling and feature engineering.

Everything in this module is intentionally beginner-readable: each step is a
small function with a docstring, and :func:`clean_data` returns a *report*
dictionary so the Streamlit app can show exactly what was fixed.

The output is one tidy DataFrame with:

* clean, physically-valid sensor columns,
* calendar features (hour / day of week / weekend),
* lag features at t-1, t-2, t-3 for level, inflow, usage and rainfall,
* rolling rainfall sums (3h, 6h, 24h) and rolling mean usage (6h, 24h),
* ``fill_pct`` (level as a % of capacity),
* the supervised target ``target_level_next`` = tank level one hour later.

Feature engineering is exposed as a pure function of a DataFrame so that the
exact same transformation can be reused for a *single* live prediction row
inside the Streamlit app (see :func:`build_live_features`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------
DEFAULT_CAPACITY_LITRES = 5000.0

#: Columns a CSV *must* have. Everything else is optional and defaulted.
REQUIRED_COLUMNS: list[str] = [
    "timestamp",
    "rainfall_mm",
    "inflow_litres",
    "usage_litres",
    "tank_level_litres",
]

#: Columns the pipeline can synthesise a sensible default for.
OPTIONAL_COLUMNS: dict[str, float] = {
    "tank_capacity_litres": DEFAULT_CAPACITY_LITRES,
    "temperature_c": float("nan"),  # filled with the column median later
}

TARGET = "target_level_next"

FEATURES: list[str] = [
    # calendar
    "hour",
    "day_of_week",
    "is_weekend",
    # current readings
    "rainfall_mm",
    "inflow_litres",
    "usage_litres",
    "tank_level_litres",
    "temperature_c",
    "fill_pct",
    "net_flow_litres",
    "headroom_litres",
    "delta_lag_1",
    # lags at t-1, t-2, t-3
    "level_lag_1",
    "level_lag_2",
    "level_lag_3",
    "inflow_lag_1",
    "inflow_lag_2",
    "inflow_lag_3",
    "usage_lag_1",
    "usage_lag_2",
    "usage_lag_3",
    "rain_lag_1",
    "rain_lag_2",
    "rain_lag_3",
    # rolling aggregates
    "rain_roll_3",
    "rain_roll_6",
    "rain_roll_24",
    "usage_roll_mean_6",
    "usage_roll_mean_24",
]

#: Multiplicative IQR fence for winsorising meter readings. 3.0 (instead of the
#: textbook 1.5) keeps genuine demand peaks while still removing sensor spikes.
IQR_MULTIPLIER = 3.0


class DataValidationError(ValueError):
    """Raised when a dataset is missing columns the pipeline requires."""


# --------------------------------------------------------------------------
# 1. Validation + loading
# --------------------------------------------------------------------------
def validate_columns(df: pd.DataFrame, required: list[str] | None = None) -> list[str]:
    """Return the list of required columns that are *absent* (empty == valid)."""
    required = REQUIRED_COLUMNS if required is None else required
    return [column for column in required if column not in df.columns]


def check_columns(df: pd.DataFrame, required: list[str] | None = None) -> None:
    """Raise :class:`DataValidationError` with a friendly message if columns are missing."""
    missing = validate_columns(df, required)
    if missing:
        available = ", ".join(map(str, df.columns)) or "(no columns)"
        raise DataValidationError(
            "Cannot process this dataset. Missing required column(s): "
            f"{', '.join(missing)}. "
            f"Required: {', '.join(required or REQUIRED_COLUMNS)}. "
            f"Found instead: {available}."
        )


def ensure_optional_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add default values for optional columns that are missing."""
    out = df.copy()
    for column, default in OPTIONAL_COLUMNS.items():
        if column not in out.columns:
            out[column] = default
    return out


def load_data(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Load a CSV path (or pass a DataFrame through) and validate its columns."""
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        df = pd.read_csv(path)

    df.columns = [str(column).strip() for column in df.columns]
    check_columns(df)
    return ensure_optional_columns(df)


# --------------------------------------------------------------------------
# 2. Outlier helpers
# --------------------------------------------------------------------------
def iqr_bounds(series: pd.Series, multiplier: float = IQR_MULTIPLIER) -> tuple[float, float]:
    """Return (lower, upper) IQR fences for a numeric series."""
    clean = series.dropna()
    if clean.empty:
        return 0.0, 0.0
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - multiplier * iqr), float(q3 + multiplier * iqr)


def robust_upper_bound(series: pd.Series, multiplier: float = IQR_MULTIPLIER) -> float:
    """Upper outlier fence for a *zero-inflated* meter signal (rainfall-driven inflow).

    A plain IQR fence is useless here: most hours have zero inflow, so Q1 = Q3 = 0
    and the fence collapses to 0 - which would erase every genuine drop of rain.
    We therefore build the fence from the **positive** values only, and fall back
    to a high quantile if that subset is constant (e.g. a flatlined meter).
    """
    positive = series[series > 0].dropna()
    if len(positive) < 10:
        return float(series.max()) if series.notna().any() else 0.0

    _, upper = iqr_bounds(positive, multiplier)
    if not np.isfinite(upper) or upper <= positive.min():
        return float(positive.quantile(0.99))
    return float(upper)


def zscore_mask(series: pd.Series, threshold: float = 3.5) -> pd.Series:
    """Boolean mask of values whose |z-score| exceeds `threshold`."""
    clean = series.dropna()
    if len(clean) < 3 or clean.std(ddof=0) == 0:
        return pd.Series(False, index=series.index)
    z = (series - clean.mean()) / clean.std(ddof=0)
    return z.abs() > threshold


# --------------------------------------------------------------------------
# 3. Cleaning
# --------------------------------------------------------------------------
def clean_data(
    df: pd.DataFrame,
    capacity: float | None = None,
    iqr_multiplier: float = IQR_MULTIPLIER,
    zscore_threshold: float = 3.5,
    max_level_jump_litres: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Clean raw telemetry and return ``(clean_df, report)``.

    Order of operations matters:

    1. parse & sort timestamps, drop duplicate timestamps,
    2. coerce everything to numbers,
    3. clip values that violate physical limits (before interpolation, so a
       -400 L reading cannot poison neighbouring interpolated values),        4. flag meter spiking (IQR for inflow, z-score for usage) and treat those
           readings as unknown rather than clamping them to the fence,
        5. repair level glitches: out-of-range readings *and* physically impossible
       hour-to-hour jumps are marked missing rather than clamped, so the series
       stays smooth,
    6. fill missing values (time interpolation for level/temperature/usage,
       zero for rainfall/inflow where a blank really means "no water moved").
    """
    check_columns(df)
    out = ensure_optional_columns(df)

    report: dict = {
        "rows_raw": int(len(out)),
        "duplicate_rows_removed": 0,
        "bad_timestamps_dropped": 0,
        "missing_before": {},
        "missing_filled": {},
        "outliers": {},
        "non_numeric_coerced": 0,
        "capacity": float(capacity) if capacity is not None else None,
    }

    # --- 1. timestamps ------------------------------------------------------
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    bad_ts = int(out["timestamp"].isna().sum())
    if bad_ts:
        out = out.dropna(subset=["timestamp"])
    report["bad_timestamps_dropped"] = bad_ts

    out = out.sort_values("timestamp").reset_index(drop=True)

    duplicates = int(out.duplicated(subset=["timestamp"], keep="first").sum())
    if duplicates:
        out = out.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)
    report["duplicate_rows_removed"] = duplicates

    # --- 2. numeric coercion ------------------------------------------------
    numeric_columns = [
        "rainfall_mm",
        "inflow_litres",
        "usage_litres",
        "tank_level_litres",
        "tank_capacity_litres",
        "temperature_c",
    ]
    numeric_columns = [c for c in numeric_columns if c in out.columns]
    before_na = int(out[numeric_columns].isna().sum().sum())
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    after_na = int(out[numeric_columns].isna().sum().sum())
    report["non_numeric_coerced"] = after_na - before_na

    capacity_value = float(
        capacity
        if capacity is not None
        else (out["tank_capacity_litres"].dropna().median() or DEFAULT_CAPACITY_LITRES)
    )
    if not np.isfinite(capacity_value) or capacity_value <= 0:
        capacity_value = DEFAULT_CAPACITY_LITRES
    out["tank_capacity_litres"] = capacity_value
    report["capacity"] = capacity_value

    report["missing_before"] = {
        column: int(out[column].isna().sum()) for column in numeric_columns
    }

    # --- 3. physical limits (flow meters) ----------------------------------
    rain_bad_low = int((out["rainfall_mm"] < 0).sum())
    inflow_bad_low = int((out["inflow_litres"] < 0).sum())
    usage_bad_low = int((out["usage_litres"] < 0).sum())

    out["rainfall_mm"] = out["rainfall_mm"].clip(lower=0.0)
    out["inflow_litres"] = out["inflow_litres"].clip(lower=0.0)
    out["usage_litres"] = out["usage_litres"].clip(lower=0.0)

    # --- 3b. the level sensor: repair, do not just clamp --------------------
    # Clamping a -400 L reading to 0 L leaves a *wrong* value in the series, and
    # the next hour then looks like a 4,000 L jump. That single glitch poisons
    # both the lag features and the training target. Instead we mark broken
    # readings as missing and let the interpolator reconstruct them.
    level = out["tank_level_litres"]
    level_bad_low = int((level < 0).sum())
    level_bad_high = int((level > capacity_value).sum())
    out_of_range = (level < 0) | (level > capacity_value)

    # An hour can only add (inflow) or remove (usage) so much water: a step
    # larger than `max_level_jump` litres is physically impossible.
    if max_level_jump_litres is None:
        max_level_jump_litres = 0.30 * capacity_value
    cleaned_level = level.where(~out_of_range)  # broken readings -> NaN
    impossible_jump = cleaned_level.diff().abs() > max_level_jump_litres
    level_jumps = int(impossible_jump.sum())

    out["level_was_outlier"] = (out_of_range | impossible_jump).fillna(False)
    out.loc[out["level_was_outlier"], "tank_level_litres"] = np.nan
    level_values_repaired = int(out["level_was_outlier"].sum())

    # --- 4. meter spikes: IQR for inflow, z-score for usage ----------------
    # Flag them AND repair them by interpolation (step 6) rather than clamping
    # to the fence: a spike reading is *unknown*, and clamping would leave a
    # value 20-50x too high in the series, which is worse than a gap.
    inflow_upper = robust_upper_bound(out["inflow_litres"], iqr_multiplier)
    inflow_mask = (out["inflow_litres"] > inflow_upper).fillna(False)
    out["inflow_was_outlier"] = inflow_mask
    out.loc[inflow_mask, "inflow_litres"] = np.nan

    usage_z = zscore_mask(out["usage_litres"], zscore_threshold)
    usage_mask = (usage_z & (out["usage_litres"] > out["usage_litres"].median())).fillna(False)
    out["usage_was_outlier"] = usage_mask
    out.loc[usage_mask, "usage_litres"] = np.nan

    inflow_spike_positions = inflow_mask.to_numpy(dtype=bool)

    report["outliers"] = {
        "level_negative_repaired": level_bad_low,
        "level_over_capacity_repaired": level_bad_high,
        "level_impossible_jump_repaired": level_jumps,
        "level_values_repaired_total": level_values_repaired,
        "rainfall_negative_clipped": rain_bad_low,
        "inflow_negative_clipped": inflow_bad_low,
        "usage_negative_clipped": usage_bad_low,
        "inflow_iqr_spikes_repaired": int(inflow_mask.sum()),
        "inflow_iqr_fence_litres": round(float(inflow_upper), 1),
        "usage_zscore_spikes_repaired": int(usage_mask.sum()),
        "rows_flagged": int(
            (out["level_was_outlier"] | out["inflow_was_outlier"] | out["usage_was_outlier"]).sum()
        ),
    }

    # --- 5. missing values --------------------------------------------------
    indexed = out.set_index("timestamp")

    # Rainfall: a blank almost always means "the gauge saw nothing".
    indexed["rainfall_mm"] = indexed["rainfall_mm"].fillna(0.0)

    # Inflow is zero-inflated, so a blank means 0 L/h - but a *repaired meter
    # spike* means "unknown", so those positions are interpolated instead.
    # copy=True: pandas hands back a read-only view, and we edit this in place.
    inflow_values = indexed["inflow_litres"].to_numpy(dtype=float, copy=True)
    inflow_values[np.isnan(inflow_values)] = 0.0
    inflow_values[inflow_spike_positions] = np.nan
    indexed["inflow_litres"] = pd.Series(inflow_values, index=indexed.index).interpolate(
        method="time", limit_direction="both"
    )

    # Level, temperature and usage are slow-moving / autocorrelated: interpolate.
    for column in ["tank_level_litres", "temperature_c", "usage_litres"]:
        indexed[column] = indexed[column].interpolate(method="time", limit_direction="both")

    # Anything still missing (e.g. temperature with an all-NaN column) -> median.
    for column in numeric_columns:
        if indexed[column].isna().any():
            fallback = indexed[column].median()
            indexed[column] = indexed[column].fillna(0.0 if pd.isna(fallback) else fallback)

    out = indexed.reset_index()

    # Safety net: interpolation must never leave a non-physical level behind.
    out["tank_level_litres"] = out["tank_level_litres"].clip(0.0, capacity_value)

    report["missing_filled"] = {
        column: int(report["missing_before"][column]) for column in numeric_columns
    }
    report["missing_filled"]["tank_level_litres"] += level_values_repaired
    report["total_values_repaired"] = int(sum(report["missing_filled"].values()))
    report["missing_after"] = {
        column: int(out[column].isna().sum()) for column in numeric_columns
    }
    report["rows_clean"] = int(len(out))

    out = out.drop(
        columns=["_is_sensor_outlier", "overflow_litres"], errors="ignore"
    ).reset_index(drop=True)
    return out, report


# --------------------------------------------------------------------------
# 4. Feature engineering
# --------------------------------------------------------------------------
def add_features(df: pd.DataFrame, capacity: float | None = None) -> pd.DataFrame:
    """Add calendar, lag, rolling and target columns.

    The function only ever looks *backwards plus at the current row*, so it is
    safe to run on the training set, the test set, or a 24-row history window
    for a live prediction - the numbers come out identical either way.
    """
    out = df.copy().sort_values("timestamp").reset_index(drop=True)

    if capacity is None:
        capacity = float(out["tank_capacity_litres"].iloc[0]) if "tank_capacity_litres" in out else DEFAULT_CAPACITY_LITRES
    capacity = float(capacity)

    ts = out["timestamp"]
    out["hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)

    out["fill_pct"] = (out["tank_level_litres"] / capacity * 100.0).clip(lower=0.0)
    out["net_flow_litres"] = out["inflow_litres"] - out["usage_litres"]
    # How much room is left before the tank spills, and which way the level was
    # already moving. A tree cannot subtract two columns, so these explicit
    # "derived" features are worth more than the raw ones they come from.
    out["headroom_litres"] = capacity - out["tank_level_litres"]

    lag_spec = {
        "level": "tank_level_litres",
        "inflow": "inflow_litres",
        "usage": "usage_litres",
        "rain": "rainfall_mm",
    }
    for prefix, column in lag_spec.items():
        for lag in (1, 2, 3):
            out[f"{prefix}_lag_{lag}"] = out[column].shift(lag)

    out["delta_lag_1"] = out["tank_level_litres"] - out["level_lag_1"]

    out["rain_roll_3"] = out["rainfall_mm"].rolling(3, min_periods=1).sum()
    out["rain_roll_6"] = out["rainfall_mm"].rolling(6, min_periods=1).sum()
    out["rain_roll_24"] = out["rainfall_mm"].rolling(24, min_periods=1).sum()
    out["usage_roll_mean_6"] = out["usage_litres"].rolling(6, min_periods=1).mean()
    out["usage_roll_mean_24"] = out["usage_litres"].rolling(24, min_periods=1).mean()

    # Supervised target: the level one hour ahead.
    out[TARGET] = out["tank_level_litres"].shift(-1)

    return out


def build_model_frame(df_features: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that cannot be used for supervised learning."""
    needed = FEATURES + [TARGET]
    return df_features.dropna(subset=needed).reset_index(drop=True)


def build_live_features(
    history: pd.DataFrame,
    current: dict,
    capacity: float,
) -> pd.DataFrame:
    """Build the feature row for a *live* prediction.

    We append the user's current readings to the tail of the cleaned history and
    re-run :func:`add_features`, then take the last row. Reusing the identical
    code path as training removes any chance of train/serve skew.

    Parameters
    ----------
    history : cleaned DataFrame (must contain at least the last 24 hours, with
        the columns ``timestamp``, ``rainfall_mm``, ``inflow_litres``,
        ``usage_litres``, ``tank_level_litres``, ``temperature_c``).
    current : mapping with keys ``rainfall_mm``, ``inflow_litres``,
        ``usage_litres``, ``tank_level_litres``, ``temperature_c``.
    capacity : tank capacity in litres.

    Returns
    -------
    A single-row DataFrame with every engineered column (select
    :data:`FEATURES` from it before calling ``model.predict``). The extra
    columns - ``rain_roll_3``, ``rain_roll_24``, ``usage_roll_mean_24`` - are
    exactly what the decision rules and the fallback prediction need.
    """
    missing = validate_columns(history)
    if missing:
        raise DataValidationError(f"History is missing column(s): {', '.join(missing)}")

    hist = history.sort_values("timestamp").tail(48).copy()
    last_ts = pd.to_datetime(hist["timestamp"].iloc[-1])

    row = {
        "timestamp": last_ts + pd.Timedelta(hours=1),
        "rainfall_mm": float(current.get("rainfall_mm", 0.0)),
        "inflow_litres": float(current.get("inflow_litres", 0.0)),
        "usage_litres": float(current.get("usage_litres", 0.0)),
        "tank_level_litres": float(current.get("tank_level_litres", 0.0)),
        "temperature_c": float(current.get("temperature_c", np.nan)),
        "tank_capacity_litres": float(capacity),
    }
    if not np.isfinite(row["temperature_c"]):
        row["temperature_c"] = float(hist["temperature_c"].median())

    combined = pd.concat(
        [hist[[c for c in hist.columns if c in row]], pd.DataFrame([row])],
        ignore_index=True,
    )
    featured = add_features(combined, capacity=capacity)
    return featured.iloc[[-1]].reset_index(drop=True)


# --------------------------------------------------------------------------
# 5. Convenience wrapper used by the app / pipeline
# --------------------------------------------------------------------------
@dataclass
class PreparedData:
    """Container for everything the modelling steps need."""

    raw: pd.DataFrame
    clean: pd.DataFrame
    features: pd.DataFrame  # rows with complete features AND target
    report: dict
    capacity: float
    feature_names: list[str] = field(default_factory=lambda: list(FEATURES))
    target: str = TARGET

    @property
    def summary(self) -> dict:
        """Human-readable one-stop summary of the cleaning + feature step."""
        return {
            **self.report,
            "capacity": self.capacity,
            "rows_model_ready": int(len(self.features)),
            "rows_dropped_for_lags_or_target": int(
                len(self.clean) - len(self.features)
            ),
            "final_feature_count": len(self.feature_names),
            "final_feature_list": list(self.feature_names),
            "target": self.target,
            "date_min": str(self.clean["timestamp"].min()),
            "date_max": str(self.clean["timestamp"].max()),
        }


def preprocess(
    source: str | Path | pd.DataFrame,
    capacity: float | None = None,
) -> PreparedData:
    """Run the full load -> clean -> feature engineering pipeline."""
    raw = load_data(source)
    clean, report = clean_data(raw, capacity=capacity)
    features_all = add_features(clean, capacity=report["capacity"])
    model_frame = build_model_frame(features_all)

    if model_frame.empty:
        raise DataValidationError(
            "After cleaning there are no usable rows left for modelling. "
            "Please upload a longer history (at least a few dozen hourly rows)."
        )

    return PreparedData(
        raw=raw,
        clean=clean,
        features=model_frame,
        report=report,
        capacity=float(report["capacity"]),
    )


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    import json

    prepared = preprocess(Path("data/tank_data.csv"))
    print(json.dumps(prepared.summary, indent=2, default=str))
