"""Generate a realistic *and deliberately messy* synthetic tank dataset.

Why synthetic?
--------------
Real rainwater-tank telemetry with rainfall, inflow, usage and level is hard to
source, so we simulate the physics ourselves. Because we generate the "truth"
first and inject noise afterwards, we always know what the correct answer is -
which makes it easy to check that the cleaning step actually recovers it.

Physical model
--------------
* Rainfall arrives in clustered storms (a storm lasts 1-6 hours).
* Storm intensity follows a gamma distribution and is modulated by season
  (the simulated period runs from a dry winter into a wetter spring/summer).
* Catchment inflow = rainfall_mm * catchment_area_m2 * runoff_coefficient
  (1 mm of rain over 1 m^2 == 1 litre).
* Demand follows a household pattern: low base flow, a morning peak and a
  larger evening peak, plus occasional irrigation days.
* Tank level is a strict mass balance:
      level[t] = clip(level[t-1] + inflow[t] - usage[t], 0, capacity)
  Water that would push the level above capacity overflows (and is lost).

Deliberate messiness (so the preprocessing step has real work to do)
-------------------------------------------------------------------
* ~4% missing values in every measured column.
* Sensor outliers: negative levels, levels above capacity, inflow spikes,
  a few negative rainfall readings.
* Duplicate timestamps (the logger re-sent a handful of rows).
* Temperatures are physically plausible but useless for the level - a nice
  reminder that feature importance is not the same as causality.

Run
---
    python data/generate_data.py
writes ``data/tank_data.csv`` (~6 months of hourly records).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration (fixed seeds => reproducible dataset)
# --------------------------------------------------------------------------
RANDOM_SEED = 42

TANK_CAPACITY_LITRES = 5000.0
CATCHMENT_AREA_M2 = 150.0
RUNOFF_COEFFICIENT = 0.85
# Starting in April means the six months run from the monsoon onset through the
# start of the dry season, so the last 20% of the timeline (the test window)
# contains both storms *and* a genuine dry spell with no rain to harvest.
START_DATE = "2024-04-09 00:00"
N_DAYS = 182  # ~6 months of hourly data == 4368 rows before messiness
INITIAL_LEVEL_FRACTION = 0.55

MISSING_FRACTION = 0.04  # ~4% missing per column
N_LEVEL_OUTLIERS = 14
N_INFLOW_OUTLIERS = 12
N_USAGE_OUTLIERS = 8
N_RAINFALL_OUTLIERS = 5
N_DUPLICATES = 18

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = DATA_DIR / "tank_data.csv"

MEASURED_COLUMNS = [
    "rainfall_mm",
    "inflow_litres",
    "usage_litres",
    "tank_level_litres",
    "temperature_c",
]


# --------------------------------------------------------------------------
# 1. Rainfall
# --------------------------------------------------------------------------
def simulate_rainfall(timestamps: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Return an hourly rainfall series (mm) made of clustered storm events."""
    n = len(timestamps)
    day_of_year = timestamps.dayofyear.to_numpy()

    # Seasonal wetness: a monsoon peak in late June, genuinely dry afterwards.
    wetness = 0.35 + 0.65 * np.sin(2.0 * np.pi * (day_of_year - 80) / 365.0)
    wetness = np.clip(wetness, 0.0, 1.0)

    # Probability that a *new* storm starts in a given hour. The dry-season floor
    # is deliberately tiny so the tank actually runs low - a harvesting demo is
    # only interesting if a dry spell can empty the tank.
    storm_start_prob = 0.002 + 0.055 * wetness

    rainfall = np.zeros(n, dtype=float)
    hour = 0
    while hour < n:
        if rng.random() < storm_start_prob[hour]:
            duration = int(rng.integers(1, 7))  # 1..6 hours
            intensity = 0.3 + rng.gamma(shape=2.0, scale=0.95)  # mm/h
            for k in range(duration):
                if hour + k >= n:
                    break
                # Slight variation inside a storm, plus a diurnal nudge so
                # afternoon convective showers are a little heavier.
                local = intensity * float(rng.uniform(0.6, 1.4))
                local *= 1.0 + 0.15 * np.sin(2.0 * np.pi * (timestamps[hour + k].hour - 6) / 24.0)
                rainfall[hour + k] = max(0.0, local)
            hour += duration
        else:
            hour += 1

    return np.round(rainfall, 2)


# --------------------------------------------------------------------------
# 2. Usage (demand) and temperature
# --------------------------------------------------------------------------
def simulate_usage(timestamps: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Household/irrigation demand in litres per hour."""
    hours = timestamps.hour.to_numpy()
    day_index = (np.arange(len(timestamps)) // 24)

    usage = np.full(len(timestamps), 2.0)  # low base flow (trickle, losses)
    usage += np.where((hours >= 6) & (hours <= 8), 30.0, 0.0)  # morning peak
    usage += np.where((hours >= 18) & (hours <= 21), 40.0, 0.0)  # evening peak

    # Irrigation happens roughly every third day, midday, and only if it is dry.
    irrigation_day = (day_index % 3 == 0)
    usage += np.where(irrigation_day & (hours >= 11) & (hours <= 13), 55.0, 0.0)

    # Random hot-day extra demand.
    usage *= rng.uniform(0.85, 1.25, size=len(timestamps))

    return np.round(np.clip(usage, 0.0, None), 2)


def simulate_temperature(timestamps: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Seasonal + diurnal temperature in degrees Celsius."""
    day_of_year = timestamps.dayofyear.to_numpy()
    hour = timestamps.hour.to_numpy()

    seasonal = 15.0 + 9.0 * np.sin(2.0 * np.pi * (day_of_year - 105) / 365.0)
    diurnal = 5.0 * np.sin(2.0 * np.pi * (hour - 9) / 24.0)
    noise = rng.normal(0.0, 1.2, size=len(timestamps))

    return np.round(seasonal + diurnal + noise, 1)


# --------------------------------------------------------------------------
# 3. The tank itself (mass balance)
# --------------------------------------------------------------------------
def simulate_tank(
    rainfall_mm: np.ndarray,
    usage_litres: np.ndarray,
    rng: np.random.Generator,
    capacity: float = TANK_CAPACITY_LITRES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the water balance.

    Returns
    -------
    inflow_litres : litres entering the tank each hour
    level_litres  : tank level *after* that hour's inflow and usage
    overflow_litres : litres lost over the overflow weir that hour
    """
    n = len(rainfall_mm)
    catchment_factor = CATCHMENT_AREA_M2 * RUNOFF_COEFFICIENT

    inflow = rainfall_mm * catchment_factor
    # Measurement/calibration noise on the inflow meter.
    inflow = np.clip(inflow * rng.normal(1.0, 0.05, size=n), 0.0, None)
    inflow = np.round(inflow, 2)

    level = np.zeros(n)
    overflow = np.zeros(n)
    current = capacity * INITIAL_LEVEL_FRACTION

    for t in range(n):
        raw = current + inflow[t] - usage_litres[t]
        if raw > capacity:
            overflow[t] = raw - capacity
            raw = capacity
        elif raw < 0.0:
            raw = 0.0  # dry tank: demand simply cannot be met
        level[t] = raw
        current = raw

    return inflow, np.round(level, 2), np.round(overflow, 2)


# --------------------------------------------------------------------------
# 4. Deliberate messiness
# --------------------------------------------------------------------------
def inject_missing_values(
    df: pd.DataFrame, rng: np.random.Generator, fraction: float = MISSING_FRACTION
) -> pd.DataFrame:
    """Blank out `fraction` of the values in every measured column."""
    out = df.copy()
    for column in MEASURED_COLUMNS:
        n_missing = int(round(len(out) * fraction))
        idx = rng.choice(len(out), size=n_missing, replace=False)
        out.loc[idx, column] = np.nan
    return out


def inject_outliers(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add physically impossible / spiky sensor readings."""
    out = df.copy()
    out["_is_sensor_outlier"] = False
    capacity = float(out["tank_capacity_litres"].iloc[0])

    def _pick(n: int) -> np.ndarray:
        return rng.choice(len(out), size=n, replace=False)

    # Negative levels (dead sensor / wiring fault).
    idx = _pick(N_LEVEL_OUTLIERS)
    half = len(idx) // 2 if len(idx) > 1 else 0
    out.loc[idx[:half], "tank_level_litres"] = -rng.uniform(20.0, 600.0, size=half)
    # Levels above capacity (stuck float / scaling error).
    out.loc[idx[half:], "tank_level_litres"] = capacity + rng.uniform(50.0, 900.0, size=len(idx) - half)
    out.loc[idx, "_is_sensor_outlier"] = True

    # Inflow spikes (turbine meter fault).
    idx = _pick(N_INFLOW_OUTLIERS)
    out.loc[idx, "inflow_litres"] = rng.uniform(4000.0, 25000.0, size=len(idx))
    out.loc[idx, "_is_sensor_outlier"] = True

    # Usage spikes (burst pipe / stuck valve opening).
    idx = _pick(N_USAGE_OUTLIERS)
    out.loc[idx, "usage_litres"] = rng.uniform(900.0, 4000.0, size=len(idx))
    out.loc[idx, "_is_sensor_outlier"] = True

    # A few negative rainfall readings (tipping-bucket bounce).
    idx = _pick(N_RAINFALL_OUTLIERS)
    out.loc[idx, "rainfall_mm"] = -rng.uniform(0.1, 1.5, size=len(idx))
    out.loc[idx, "_is_sensor_outlier"] = True

    return out


def inject_duplicate_timestamps(
    df: pd.DataFrame, rng: np.random.Generator, n_duplicates: int = N_DUPLICATES
) -> pd.DataFrame:
    """Re-append some rows with identical timestamps (logger retransmission)."""
    idx = rng.choice(len(df), size=n_duplicates, replace=False)
    copies = df.iloc[idx].copy()
    out = pd.concat([df, copies], ignore_index=True)
    return out


# --------------------------------------------------------------------------
# 5. Public API
# --------------------------------------------------------------------------
def generate_dataset(
    seed: int = RANDOM_SEED,
    n_days: int = N_DAYS,
    start_date: str = START_DATE,
) -> pd.DataFrame:
    """Build the full messy dataset and return it (unsorted, with duplicates)."""
    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(start=start_date, periods=n_days * 24, freq="h")

    rainfall = simulate_rainfall(timestamps, rng)
    usage = simulate_usage(timestamps, rng)
    temperature = simulate_temperature(timestamps, rng)
    inflow, level, overflow = simulate_tank(rainfall, usage, rng)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "rainfall_mm": rainfall,
            "inflow_litres": inflow,
            "usage_litres": usage,
            "tank_level_litres": level,
            "tank_capacity_litres": TANK_CAPACITY_LITRES,
            "temperature_c": temperature,
            # Kept in the CSV so the impact analysis can quantify real losses.
            "overflow_litres": overflow,
        }
    )

    df = inject_outliers(df, rng)
    df = inject_missing_values(df, rng)
    df = inject_duplicate_timestamps(df, rng)

    # Shuffle the row order slightly so "just sort it" is a genuine step.
    order = rng.permutation(len(df))
    return df.iloc[order].reset_index(drop=True)


def save_dataset(df: pd.DataFrame, path: Path = DEFAULT_OUTPUT) -> Path:
    """Write the dataset to CSV (creating the directory if needed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic tank dataset.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--days", type=int, default=N_DAYS)
    args = parser.parse_args()

    df = generate_dataset(seed=args.seed, n_days=args.days)
    path = save_dataset(df, args.output)

    total_inflow = df["inflow_litres"].sum(skipna=True)
    total_usage = df["usage_litres"].sum(skipna=True)
    total_rain = df["rainfall_mm"].sum(skipna=True)
    overflow = df["overflow_litres"].sum()

    print(f"Wrote {len(df):,} rows -> {path}")
    print(f"  period        : {df['timestamp'].min()} .. {df['timestamp'].max()}")
    print(f"  total rainfall: {total_rain:,.0f} mm over {CATCHMENT_AREA_M2:.0f} m2 catchment")
    print(f"  total inflow  : {total_inflow:,.0f} L")
    print(f"  total demand  : {total_usage:,.0f} L")
    print(f"  overflow      : {overflow:,.0f} L lost")
    print(f"  missing values: {int(df[MEASURED_COLUMNS].isna().sum().sum())}")
    print(f"  duplicate rows: {int(df.duplicated(subset=['timestamp']).sum())}")
    print(f"  sensor outlier rows: {int(df['_is_sensor_outlier'].sum())}")


if __name__ == "__main__":
    main()
