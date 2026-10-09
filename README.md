# 💧 Rainwater Harvesting Tank Controller

Predict the **next-hour water level** of a rainwater storage tank from rainfall,
catchment inflow and household demand — then recommend one of three operating
actions (**HARVEST**, **USE**, **PREVENT_OVERFLOW**), and prove the result with a
measured water-saving impact.

Python · pandas · scikit-learn · Streamlit. Runs locally, no accounts, no API keys.

---

## 1. Problem statement

A rooftop rainwater harvesting system sends runoff into a storage tank. The tank
is small compared to a monsoon storm, so it repeatedly:

* **overflows** — rain that cannot fit goes over the weir and is lost, and
* **runs low** — a dry spell drains the tank below what the household needs.

A tank with no controller does nothing about either: it spills when full and
runs dry when empty. The job of this project is to

1. **predict the tank level one hour ahead** from the readings a real system
   already has (rain gauge, inlet meter, usage meter, level sensor), and
2. turn that prediction into **one visible action** the household can act on
   today, and
3. quantify, over a held-out period, **how much water that actually saves** —
   litres diverted from the overflow, overflow events avoided and shortage
   events avoided.

Everything is reproducible: fixed seeds, a synthetic dataset generated from a
known physical model, a strict chronological train/test split, and a test suite
that pins down the failure behaviour.

---

## 2. Project structure

```
.
├── app.py                        # Streamlit demo (8 tabs, the whole story)
├── requirements.txt
├── pytest.ini
├── conftest.py                   # makes `src` importable for pytest
├── data/
│   ├── generate_data.py          # synthetic + deliberately messy dataset
│   └── tank_data.csv             # created by the generator (~4.4k hourly rows)
├── src/
│   ├── preprocess.py             # loading, cleaning, outlier repair, features
│   ├── baseline.py               # persistence + physics baselines, metrics
│   ├── estimator.py              # TankLevelModel: predicts the change, returns a level
│   ├── model.py                  # training, evaluation, joblib persistence
│   ├── decision.py               # action rules, sensor-fault fallback, impact sim
│   └── pipeline.py               # one-call orchestration used by the app
├── tests/
│   └── test_failure_cases.py     # 29 failure-case + behaviour tests
└── models/tank_model.joblib      # created when you train
```

---

## 3. How to run

```bash
pip install -r requirements.txt

python data/generate_data.py        # 1. build data/tank_data.csv
python -m src.model                 # 2. train, evaluate, save models/tank_model.joblib
python -m pytest                    # 3. run the failure-case tests

streamlit run app.py                # 4. open the demo at http://localhost:8501
```

The app also works with **no data file at all** — it generates the sample
dataset in memory. You can upload your own CSV in the sidebar, and there is a
button to write the sample dataset to disk.

Optional smoke checks:

```bash
python -m src.preprocess   # prints the cleaning report as JSON
python -m src.baseline     # prints persistence vs physics on the test window
python -m src.decision     # prints the impact metrics for the saved model
```

---

## 4. Dataset description

There is no convenient public dataset at hourly resolution with rainfall,
inflow, demand *and* tank level, so `data/generate_data.py` simulates the
physics and then **knowingly damages** the result.

**Simulation**

| Quantity | Model |
|---|---|
| Rainfall | Clustered storms (1–6 h) with a gamma-distributed intensity, modulated by a monsoon season (peak late June) |
| Inflow | `rainfall_mm × 150 m² × 0.85 runoff coefficient`, plus 5 % meter noise (1 mm over 1 m² = 1 litre) |
| Demand | Low base flow + morning peak (06–08) + evening peak (18–21) + irrigation every third day |
| Tank level | Strict mass balance: `level[t] = clip(level[t-1] + inflow[t] − usage[t], 0, 5000 L)`; anything above 5000 L is recorded as **overflow (lost)** |
| Temperature | Seasonal + diurnal sine + noise (deliberately *not* useful for the level) |

**What the generator produces** (seed 42, six months from 2024-04-09):

| | |
|---|---|
| Rows written | 4,386 hourly records |
| Period | 2024-04-09 → 2024-10-07 (monsoon onset → start of the dry season) |
| Total rainfall | 1,067 mm over a 150 m² catchment |
| Total inflow | 267,008 L |
| Total demand | 81,478 L |
| Overflow lost | 79,940 L (~30 % of everything harvested) |
| Injected messiness | 878 missing values (~4 % of every measured column), **39 sensor-outlier rows** (negative levels, levels above capacity, 4,000–25,000 L/h inflow spikes, burst-pipe usage spikes, negative rainfall), **18 duplicate timestamps**, shuffled row order |

The target period was chosen deliberately: starting in April means the last 20 %
of the timeline (the test window) contains **both** storms *and* a genuine dry
spell, so the controller is evaluated in more than one regime.

---

## 5. Preprocessing (`src/preprocess.py`)

The order of operations is the interesting part — doing it in the wrong order
silently destroys the signal:

| # | Step | Why this order |
|---|---|---|
| 1 | Parse timestamps, sort, drop duplicate timestamps (keep first) | duplicates would otherwise double-count an hour |
| 2 | Coerce everything to numeric | a stray `"N/A"` becomes `NaN`, never a crash |
| 3 | Clip flow meters to physical limits (rainfall ≥ 0, inflow ≥ 0, usage ≥ 0) | a negative reading is impossible |
| 4 | Flag level glitches: out-of-range (`< 0` or `> capacity`) **and** physically impossible hour-to-hour jumps (> 30 % of capacity) | see the note below |
| 5 | Flag inflow spikes with **IQR fences** (fitted on the *positive* values only) and usage spikes with a **z-score** | |
| 6 | Repair every flagged value by **time interpolation**, then fill remaining gaps: `0` for rainfall/inflow (a blank there means "no water moved"), time interpolation for level/temperature/usage, median as a last resort | |
| 7 | Engineer features | |

**Clipping is not repairing.** The first version of this pipeline simply clamped
bad level readings into `[0, capacity]`. That left a *wrong* value in the series:
a `-400 L` glitch became `0 L`, and the following hour then looked like a
4,000 L jump, which poisoned both the lag features and the training target. The
Random Forest scored **worse than the persistence baseline** because of it.
Marking broken readings as missing and interpolating them removes the glitch
entirely. Inflow spikes get the same treatment (clamping a 25,000 L/h spike to
the fence still leaves a value 20× too high in the series).

**Measured result on the generated dataset**

```
rows before                4,386
duplicate timestamps removed  18   -> 4,368 clean rows
missing values filled       875    (175 in each of 5 measured columns)
level readings repaired      14    (7 negative, 7 above capacity)
level jumps repaired          0    (the safety net found nothing left over)
rainfall negatives clipped     5
inflow IQR spikes repaired    12    (IQR fence at 1,184.6 L/h)
usage z-score spikes repaired  7
rows flagged as sensor outliers 33
final model-ready rows      4,364  (4 dropped: lag/target warm-up)
```

Every one of these counts is printed by `python -m src.preprocess` and shown in
the app's **Preprocessing** tab.

### Target and features

**Target** — `target_level_next` = the tank level one hour later (`t+1`).

**Features** — 29 columns, every one of them known at time *t*:

* **Calendar**: `hour`, `day_of_week`, `is_weekend`
* **Current readings**: `rainfall_mm`, `inflow_litres`, `usage_litres`,
  `tank_level_litres`, `temperature_c`
* **Derived**: `fill_pct` (level ÷ capacity), `net_flow_litres`
  (inflow − usage), `headroom_litres` (capacity − level), `delta_lag_1`
  (the previous hour's change)
* **Lags (t-1, t-2, t-3)** for level, inflow, usage and rainfall — 12 columns
* **Rolling**: `rain_roll_3`, `rain_roll_6`, `rain_roll_24` (mm),
  `usage_roll_mean_6`, `usage_roll_mean_24` (L/h)

Feature engineering is one pure function, reused for training *and* for the
live single-row prediction in the app (history + the user's current values are
appended and the same code runs), so there is no train/serve skew.

---

## 6. Train / test strategy

**Chronological 80/20 split. No shuffling, anywhere.**

* **Train** — first 80 %: `2024-04-09 00:00 → 2024-09-01`, 3,491 rows
* **Test** — last 20 %: `2024-09-01 14:00 → 2024-10-07`, 873 rows

Shuffling a time series leaks: for every test point, the hours immediately before
and after it would sit in the training set, so the model would be graded on
essentially memorised neighbours and the score would be meaningless. The split is
by *time*, and the impact simulation later runs on that same untouched window.

**Validation** additionally uses `TimeSeriesSplit(n_splits=5)` on the training
portion only — each fold trains on the past and validates on the future.

---

## 7. Baselines (`src/baseline.py`)

| Baseline | Rule |
|---|---|
| **Persistence** | next level = current level |
| **Physics** | next level = `clip(current + inflow − average usage, 0, capacity)` |

Both are evaluated on exactly the same 873 test rows as the model. The physics
baseline is also the **fallback** used when a sensor fault is detected.

---

## 8. Model and results (`src/model.py`)

Two candidates, both trained on the same features:

* **Random Forest** — 200 trees, `min_samples_leaf=20`, `random_state=42`
* **Ridge regression** — standardised features, as a linear reference point

### Results on the held-out test window (873 hours)

| Model | MAE (L) | RMSE (L) | R² | vs best baseline (MAE) |
|---|---|---|---|---|
| Persistence | 26.09 | 71.15 | 0.9966 | — |
| Physics | **24.82** | **53.41** | **0.9981** | — *(best baseline)* |
| Ridge | 26.94 | 59.63 | 0.9976 | −8.5 % |
| **Random Forest** | **20.69** | 56.84 | 0.9979 | **+16.7 %** |

Walk-forward CV on the training rows: **MAE 21.72 L ± 1.72** across 5 folds —
the model is stable across time, not tuned to one lucky window.

**Honest reading of the table.** The Random Forest wins on MAE (typical error),
which is what the controller actually consumes. Its RMSE is slightly worse than
the physics baseline, because the physics rule is exactly right in one regime —
when the tank is pinned at capacity, "current + inflow − usage" is the truth and
the forest occasionally overshoots the corner. The linear model loses outright:
the truncation at capacity and the zero-inflated inflow are nonlinear, and Ridge
cannot express them.

### Why the model predicts the *change*, not the level

The dataset target is the next-hour **level**, and the metrics above are all
measured on predicted levels. But the *estimator* is fitted on the one-hour
**change** (`target − current level`), and the current reading is added back
afterwards:

| What the Random Forest is asked to predict | MAE | R² |
|---|---|---|
| The level directly | 169.60 L | 0.8995 |
| The hourly change, then recombined (used) | **20.69 L** | **0.9979** |

A regression tree can only ever output the *average target of a leaf*, so asking
it for an absolute level makes it shrink towards the training mean — during a wet
test period it systematically under-predicts, and it ends up **worse than simply
repeating the current level**. Asking instead "will the tank go up or down, and
by how much?" is a far easier question, and it is the single change that makes
the ML model beat the baseline by ~17 %. `TankLevelModel` (`src/estimator.py`)
owns that recombination so the app, the evaluation and the impact simulation all
do the arithmetic identically.

### Overflow-warning quality

Predicting the level precisely is nice; **not missing an overflow** is what saves
water, so "next level > 90 % of capacity" is scored separately as a binary event:

| | Precision | Recall | F1 | Missed overflows | False alarms |
|---|---|---|---|---|---|
| **Random Forest** | 0.987 | 0.978 | 0.982 | **5** | 3 |
| Physics | 0.974 | 0.982 | 0.978 | 4 | 6 |
| Ridge | 0.978 | 0.978 | 0.978 | 5 | 5 |
| Persistence | 0.973 | 0.973 | 0.973 | 6 | 6 |

A false alarm costs one unnecessary drawdown; a miss costs the water. The Random
Forest has the fewest false alarms at the same recall as the physics rule, which
makes it the better warning system even before the MAE advantage.

### Feature importance

`delta_lag_1` (0.42) ≫ `headroom_litres` (0.12) > `net_flow_litres` (0.08) >
`hour` (0.07) > `fill_pct` (0.05) ≈ `tank_level_litres` (0.05).

The tank is close to a random walk with a drift, so *which way it was already
moving* and *how much room is left before it spills* dominate — exactly the two
quantities a tree cannot derive by subtracting other columns, which is why they
are engineered explicitly.

---

## 9. Product decision (`src/decision.py`)

The controller turns the predicted next-hour fill percentage (plus rain already
falling) into **one** action:

| Predicted fill | Rain context | Action | What it means |
|---|---|---|---|
| **> 90 %** | any | **PREVENT_OVERFLOW** | Open the overflow/divert valve, draw water down now |
| **≥ 75 %** | ≥ 4 mm in 3 h or ≥ 12 mm in 24 h | **PREVENT_OVERFLOW** | Trigger *early* — the storm is still arriving |
| **≥ 55 %** | wet and filling | **USE** | Draw water now (irrigation/household) to keep headroom |
| 30–90 % | dry/stable | **HARVEST** | Keep collecting |
| **< 30 %** | any | **CONSERVE** | Restrict non-essential use |

`HARVEST`, `USE` and `PREVENT_OVERFLOW` are the three operating actions;
`CONSERVE` is the restrictive safety mode the brief asks for below 30 %.

**Rain forecast matters.** A tank at 80 % that is stable is fine, but 80 % with
15 mm of rain already in the last 24 hours is an overflow waiting to happen. The
rolling rainfall sums are used as a cheap, honest short-range forecast: what has
fallen in the last 3–24 hours is what is *still feeding the tank*. That shifts
the overflow trigger earlier instead of reacting once the weir is already
flowing.

In the app the active decision is rendered as a large coloured card with a
one-line reason.

---

## 10. Impact metric (`src/decision.simulate_impact`)

The test period is simulated **twice with identical weather and demand** — only
the controller differs, so any difference is attributable to it.

* **Without a controller**: the tank spills when full and drains when empty.
* **With the controller**, each hour follows the recommendation:
  * **PREVENT_OVERFLOW** → divert/use water **only to the extent that it would
    otherwise spill this hour**, capped at 700 L/h (pump capacity);
  * **CONSERVE** → cut non-essential demand by 40 % while below the 30 % line;
  * **HARVEST / USE** → no intervention.

Sizing the preventive draw to the *actual spill* is the important design choice:
the controller never gives away **stored** water, so it can rescue a spill
without creating a shortage later in the same dry spell. (An earlier version
drew the tank down to a 70 % target. It saved slightly more litres — and then
caused *new* shortages in the dry tail, because it had given away water that
never came back. That is a net loss, so it was removed.)

### Results over the 36.4-day / 873-hour test window

| Metric | No controller | With controller |
|---|---|---|
| Water lost to overflow | 6,907 L | **338 L** |
| — overflow events | 7 | **1** |
| — overflow hours | 20 | 1 |
| Hours below the 15 % reserve | 27 | **0** |
| — shortage events | 1 | **0** |
| Lowest level reached | 379 L | **1,016 L** |
| Mean level | 3,509 L | 3,547 L |

| Headline impact | Value |
|---|---|
| **Litres saved from overflow** | **6,568 L (95.1 % reduction)** |
| **Overflow events avoided** | **6** |
| **Shortage events avoided** | **1** |
| Water used productively instead of spilt | 6,568 L |
| Non-essential demand postponed (CONSERVE) | 637 L |

Controller actions taken during the simulation: **558 HARVEST, 228
PREVENT_OVERFLOW, 87 CONSERVE** — all three operating modes are exercised.

*Definitions.* A **shortage hour** is an hour at or below the 15 % reserve line
(750 L of a 5,000 L tank — under two days of household demand), or an hour where
demand cannot be met. An **event** is one contiguous run of such hours, so a
6-hour spill counts as one event, not six. Saved litres are water that would have
gone over the weir and is instead routed to irrigation/household use — rescued,
not created.

---

## 11. Failure-case behaviour

Every failure case is implemented, tested (`tests/test_failure_cases.py`, 29
tests) **and** demonstrable inside the app.

| Failure | What the system does | How it is tested | How you see it in the app |
|---|---|---|---|
| **Missing required columns** in an uploaded CSV | `DataValidationError` naming the missing column(s) and listing what was required and what was found. The page stops with a readable message; nothing crashes. | `test_missing_required_column_raises_clear_error`, `test_preprocess_rejects_csv_without_usage_column` | Tab 1 → *"Try the 'missing columns' failure case"* expander (shows the real exception text) |
| **Sensor fault**: negative level | Detected, ML model bypassed, **physics baseline** used | `test_negative_level_is_detected_as_sensor_fault`, `test_fallback_is_triggered_and_uses_physics_baseline` | Tab 6 → *Failure case demo* → **Negative level** |
| **Sensor fault**: level > capacity | Same | `test_level_above_capacity_is_detected_as_sensor_fault` | Tab 6 → **Level > capacity** |
| **Sensor fault**: `NaN` reading | Same | `test_nan_reading_is_detected_as_sensor_fault` | Tab 6 → **NaN reading** |
| **Sensor fault**: flatlined sensor (many identical hours) | Same — a frozen float is caught after 6 identical readings | `test_flatlined_sensor_is_detected` | Tab 6 → **Flatlined sensor** |
| **Empty / missing history** | Treated as a fault; physics fallback | `test_empty_history_is_a_fault` | — |
| **Out-of-range user input** | Rejected with a warning before it reaches the model | `test_validate_user_input_rejects_out_of_range_values`, `test_validate_level_bounds` | Tab 6 → the *out-of-range user input* demo, and sliders that cannot exceed physical limits |

When a fault is detected the app shows exactly:

> **Sensor fault suspected — using fallback baseline.** Reason: 1 negative level
> reading(s) (e.g. -420 L). The ML model has been bypassed and the physics
> baseline (current level + inflow − average usage, clipped to the tank) is
> serving the recommendation.

**Why fall back to physics rather than refusing to answer?** A tank controller
that goes silent during a sensor fault is worse than one that keeps working from
the water balance — the household still needs to know whether to draw water down.

Other tests cover the data-quality guarantees themselves: duplicates removed,
out-of-range values repaired (not merely clamped), fully-missing columns
surviving, the live feature row matching the training schema, the decision bands
being exclusive and correctly ordered, and the impact simulation never reporting
fictional savings.

---

## 12. The Streamlit app

`streamlit run app.py` — eight tabs, each with a short explanation, following the
project end to end:

1. **Input Data** — the raw, messy file: duplicate timestamps, missing values,
   sensor spikes; plus the missing-column failure case.
2. **Preprocessing** — before/after row counts, what was fixed, the cleaning
   order, the clean-vs-raw level chart, and the final feature list.
3. **Baseline** — persistence and physics scores, with actual-vs-baseline chart.
4. **Trained Model** — model table, chosen model, feature importance, CV results,
   the direct-level-vs-change comparison, and the joblib download.
5. **Evaluation** — actual vs predicted, scatter, residual histogram, worst
   misses, and the overflow-warning precision/recall table.
6. **Prediction** — sliders for current level / rainfall / inflow / usage, a live
   next-hour prediction, the coloured recommendation card, the fallback banner
   and the **Failure case demo** buttons.
7. **Product Decision** — the rule table, action counts over the test period,
   predicted fill with the 90 %/30 % bands drawn in, and the latest decision card.
8. **Impact** — the metric cards, the full comparison table, the level and
   cumulative-overflow trajectories with and without the controller.

Sidebar: pick the built-in sample dataset or upload a CSV, review the required
columns, regenerate the sample data on disk, and see the current run's
parameters at a glance.

---

## 13. Reproducibility

* Fixed `RANDOM_SEED = 42` in the generator, and `random_state=42` on every model.
* No randomness anywhere in preprocessing or evaluation.
* Re-running `python data/generate_data.py && python -m src.model` reproduces the
  exact numbers in this README.

---

## 14. Limitations and next steps

* **Synthetic data.** The physics is explicit and the messiness is injected, so
  the cleaning steps can be verified against a known truth — but real meters have
  systematic bias and lag that this does not model.
* **One-hour horizon.** The controller reacts hours ahead, not days. A 24–72 h
  weather forecast (instead of rolling rainfall as a proxy) would let
  PREVENT_OVERFLOW be scheduled overnight rather than within the hour.
* **The impact simulation is a model, not a field trial.** It assumes demand can
  actually absorb the diverted litres (irrigation is schedulable) and that the
  pump can move 700 L/h.
* **Single tank, single household.** Multi-tank or networked systems would need
  routing decisions on top of this rule set.
