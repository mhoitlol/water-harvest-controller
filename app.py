"""Rainwater Harvesting Tank Controller - Streamlit demo.

    streamlit run app.py

The app walks through the whole project in the order the brief asks for:
Input Data -> Preprocessing -> Baseline -> Trained Model -> Evaluation ->
Prediction -> Product Decision -> Impact.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Rainwater Harvesting Tank Controller",
    page_icon="💧",
    layout="wide",
)

from data.generate_data import generate_dataset  # noqa: E402  (after set_page_config)
from src.decision import (  # noqa: E402
    ACTION_COLOURS,
    ACTION_CONSERVE,
    ACTION_HARVEST,
    ACTION_PREVENT,
    ACTION_USE,
    CONSERVE_USAGE_REDUCTION,
    EARLY_OVERFLOW_PCT,
    HEAVY_RAIN_24H_MM,
    HEAVY_RAIN_3H_MM,
    LOW_PCT,
    MAX_PREVENTIVE_DRAW_LITRES,
    OVERFLOW_PCT,
    decide,
    detect_sensor_fault,
    recommend_next_hour,
)
from src.model import DEFAULT_MODEL_PATH, load_bundle, save_bundle  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402
from src.preprocess import (  # noqa: E402
    FEATURES,
    TARGET,
    DataValidationError,
    clean_data,
    load_data,
    validate_columns,
)

SAMPLE_CSV = Path("data") / "tank_data.csv"
MODEL_PATH = Path(DEFAULT_MODEL_PATH)

ACTION_LABELS = {
    ACTION_HARVEST: "HARVEST - keep collecting",
    ACTION_USE: "USE - draw water now",
    ACTION_PREVENT: "PREVENT_OVERFLOW - divert / draw down",
    ACTION_CONSERVE: "CONSERVE - restrict non-essential use",
}


# ==========================================================================
# Cached helpers
# ==========================================================================
@st.cache_data(show_spinner=False)
def make_sample_dataset(seed: int = 42) -> pd.DataFrame:
    """Generate the synthetic dataset in memory (used when no CSV is present)."""
    return generate_dataset(seed=seed)


@st.cache_data(show_spinner="Loading and cleaning the dataset...")
def load_dataset_frame(source_name: str, uploaded_bytes: bytes | None) -> pd.DataFrame:
    """Read the selected dataset. Raises DataValidationError with a clear message."""
    if uploaded_bytes is not None:
        import io

        df = pd.read_csv(io.BytesIO(uploaded_bytes))
        df.columns = [str(column).strip() for column in df.columns]
        missing = validate_columns(df)
        if missing:
            available = ", ".join(map(str, df.columns)) or "(no columns)"
            raise DataValidationError(
                "This CSV cannot be used. Missing required column(s): "
                f"{', '.join(missing)}. Required columns are: "
                "timestamp, rainfall_mm, inflow_litres, usage_litres, tank_level_litres. "
                f"Columns found: {available}."
            )
        return df

    if SAMPLE_CSV.exists():
        return pd.read_csv(SAMPLE_CSV)
    return make_sample_dataset()


@st.cache_resource(show_spinner="Training the models (this happens once)...")
def train_artefacts(source_name: str, uploaded_bytes: bytes | None) -> dict:
    """Run preprocessing + baselines + training + impact simulation, then save the bundle."""
    frame = load_dataset_frame(source_name, uploaded_bytes)
    artefacts = run_pipeline(frame)
    try:
        save_bundle(artefacts["bundle"], MODEL_PATH)
        artefacts["model_path"] = MODEL_PATH
    except OSError:  # pragma: no cover - read-only filesystem
        artefacts["model_path"] = None
    return artefacts


def chart_frame(frame: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """Small helper: rename + reorder columns for st.line_chart."""
    return frame[list(columns)].rename(columns=columns).set_index("timestamp")


def decision_card(action: str, reason: str, caption: str = "") -> None:
    """Render the big coloured recommendation card."""
    colour = ACTION_COLOURS.get(action, "#7f7f7f")
    label = ACTION_LABELS.get(action, action)
    st.markdown(
        f"""
        <div style="background:{colour};padding:18px 22px;border-radius:12px;
                    color:white;margin-bottom:6px;">
            <div style="font-size:0.8rem;letter-spacing:0.12em;opacity:0.85;">
                RECOMMENDED ACTION
            </div>
            <div style="font-size:1.7rem;font-weight:700;margin:4px 0;">{label}</div>
            <div style="font-size:1.0rem;line-height:1.35;">{reason}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if caption:
        st.caption(caption)


def fallback_banner(used_fallback: bool, reason: str | None) -> None:
    """Show the sensor-fault banner the brief asks for."""
    if used_fallback:
        st.warning(
            "**Sensor fault suspected — using fallback baseline.** "
            f"Reason: {reason}. The ML model has been bypassed and the physics "
            "baseline (current level + inflow − average usage, clipped to the tank) "
            "is serving the recommendation."
        )
    else:
        st.success("Sensors healthy — the trained ML model produced this prediction.")


# ==========================================================================
# Sidebar
# ==========================================================================
st.sidebar.title("💧 Tank Controller")
st.sidebar.caption("Rainwater harvesting: predict the next hour, then act.")

st.sidebar.subheader("1. Data source")
source_choice = st.sidebar.radio(
    "Dataset",
    ["Built-in sample dataset", "Upload my own CSV"],
    help="The sample dataset is generated by data/generate_data.py (~6 months hourly).",
)

uploaded_bytes: bytes | None = None
source_name = "sample"
if source_choice == "Upload my own CSV":
    uploaded = st.sidebar.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        uploaded_bytes = uploaded.getvalue()
        source_name = uploaded.name

with st.sidebar.expander("Required columns"):
    st.markdown(
        """
        - `timestamp`
        - `rainfall_mm`
        - `inflow_litres`
        - `usage_litres`
        - `tank_level_litres`

        Optional: `tank_capacity_litres` (default 5000), `temperature_c`.
        """
    )

if st.sidebar.button("Regenerate the sample dataset on disk", use_container_width=True):
    frame = generate_dataset(seed=42)
    SAMPLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(SAMPLE_CSV, index=False)
    st.sidebar.success(f"Wrote {len(frame):,} rows to {SAMPLE_CSV}")

# --------------------------------------------------------------------------
# Load + train (all failure handling starts here)
# --------------------------------------------------------------------------
try:
    artefacts = train_artefacts(source_name, uploaded_bytes)
except DataValidationError as error:
    st.error(f"**Dataset rejected.** {error}")
    st.info(
        "Nothing crashed — this is the *missing required column* failure case. "
        "Pick the built-in sample dataset in the sidebar, or upload a CSV that "
        "contains the required columns listed there."
    )
    st.stop()
except FileNotFoundError as error:  # pragma: no cover - defensive
    st.error(f"Could not read the dataset: {error}")
    st.stop()

prepared = artefacts["prepared"]
bundle = artefacts["bundle"]
impact = artefacts["impact"]
clean = prepared.clean
features = prepared.features
capacity = prepared.capacity
test_frame = bundle["test_frame"]
summary = prepared.summary

st.sidebar.subheader("2. About this run")
st.sidebar.markdown(
    f"""
    - rows: **{len(prepared.raw):,}**
    - usable rows: **{len(features):,}**
    - capacity: **{capacity:,.0f} L**
    - model: **{bundle["model_name"]}**
    - test MAE: **{bundle["metrics"][bundle["model_name"]]["mae"]:.1f} L**
    - checkpoint: **{"loaded/saved to disk" if artefacts["model_path"] else "in memory"}**
    """
)

# ==========================================================================
# Header
# ==========================================================================
st.title("💧 Rainwater Harvesting Tank Controller")
st.markdown(
    "Predict the **next-hour tank level** from rainfall, catchment inflow and demand — "
    "then recommend **HARVEST**, **USE** or **PREVENT_OVERFLOW**, and quantify the water saved."
)

tabs = st.tabs(
    [
        "1 · Input Data",
        "2 · Preprocessing",
        "3 · Baseline",
        "4 · Trained Model",
        "5 · Evaluation",
        "6 · Prediction",
        "7 · Product Decision",
        "8 · Impact",
    ]
)

# --------------------------------------------------------------------------
# Tab 1 - Input data
# --------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Raw input data")
    st.markdown(
        f"The demo is reading **{source_name}** — "
        f"{len(prepared.raw):,} rows from `{summary['date_min']}` to `{summary['date_max']}`. "
        "The synthetic generator hands us a deliberately messy file: missing values, "
        "out-of-range sensor spikes and duplicate timestamps all have to be dealt with downstream."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows in file", f"{len(prepared.raw):,}")
    col2.metric("Duplicate timestamps", f"{int(prepared.raw.duplicated(subset=['timestamp']).sum()):,}")
    col3.metric("Missing values", f"{int(prepared.raw.isna().sum().sum()):,}")
    col4.metric("Columns", f"{prepared.raw.shape[1]}")

    st.markdown("**First rows as received** (note the duplicate timestamps and gaps):")
    st.dataframe(prepared.raw.head(12), use_container_width=True)

    st.markdown("**Missing values per column**")
    missing_table = (
        prepared.raw.isna()
        .sum()
        .rename("missing")
        .to_frame()
        .assign(pct=lambda d: (d["missing"] / len(prepared.raw) * 100).round(1))
    )
    st.dataframe(missing_table, use_container_width=True)

    # The raw frame keeps the on-disk dtype (a string in pandas 3), so parse the
    # timestamp before charting or joining it against the cleaned frame.
    raw_level = prepared.raw.dropna(subset=["tank_level_litres"]).copy()
    raw_level["timestamp"] = pd.to_datetime(raw_level["timestamp"], errors="coerce")
    raw_level = raw_level.dropna(subset=["timestamp"]).sort_values("timestamp")
    st.markdown(
        f"**Tank level as recorded** — the spikes above the {capacity:,.0f} L capacity line "
        "and the negative dips are injected sensor faults:"
    )
    st.line_chart(raw_level.set_index("timestamp")["tank_level_litres"])

    with st.expander("Try the 'missing columns' failure case"):
        broken = prepared.raw.drop(columns=["usage_litres"])
        try:
            load_data(broken)
        except DataValidationError as error:
            st.error(f"**Caught DataValidationError (no crash):** {error}")
        st.code(
            "df = df.drop(columns=['usage_litres'])\ntry:\n    load_data(df)\n"
            "except DataValidationError as error:\n    st.error(error)",
            language="python",
        )

# --------------------------------------------------------------------------
# Tab 2 - Preprocessing
# --------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Cleaning, outlier handling and feature engineering")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows before → after", f"{summary['rows_raw']:,} → {summary['rows_clean']:,}")
    c2.metric(
        "Duplicates / bad timestamps removed",
        f"{summary['duplicate_rows_removed']:,} / {summary['bad_timestamps_dropped']:,}",
    )
    c3.metric("Missing values filled", f"{sum(summary['missing_filled'].values()):,}")
    c4.metric("Outlier values handled", f"{summary['outliers']['rows_flagged']:,} rows")

    st.markdown(
        """
        **Order of operations** (it matters):
        1. parse + sort timestamps, drop duplicates,
        2. coerce to numeric (a stray `"N/A"` becomes `NaN`, never a crash),
        3. **clip physical limits first** — `0 ≤ level ≤ capacity`, rainfall and flow ≥ 0 — so a
           `-400 L` reading cannot poison the interpolation around it,
        4. winsorise meter spikes: IQR fences for inflow, z-score for usage,
        5. *then* fill gaps — time interpolation for level/temperature/usage, `0` for
           rainfall/inflow (a blank there means "no water moved"), median as a last resort.
        """
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Missing values fixed**")
        st.dataframe(
            pd.DataFrame(
                {
                    "missing_before": summary["missing_before"],
                    "filled": summary["missing_filled"],
                    "still_missing": summary["missing_after"],
                }
            ),
            use_container_width=True,
        )
    with col_b:
        st.markdown("**Outliers handled**")
        st.dataframe(
            pd.DataFrame(
                {"count": pd.Series(summary["outliers"])}
            ).rename_axis("issue"),
            use_container_width=True,
        )

    st.markdown("**Clean tank level vs the raw recording**")
    compare = (
        raw_level[["timestamp", "tank_level_litres"]]
        .rename(columns={"tank_level_litres": "raw"})
        .merge(
            clean[["timestamp", "tank_level_litres"]].rename(
                columns={"tank_level_litres": "cleaned"}
            ),
            on="timestamp",
            how="inner",
        )
        .set_index("timestamp")
    )
    st.line_chart(compare, height=320)

    st.markdown(
        f"**Engineered features ({len(FEATURES)})** — everything the model sees is known at time *t*, "
        f"the target `{TARGET}` is the level at *t+1*:"
    )
    st.code(", ".join(FEATURES), language="text")
    st.caption(
        f"Rows dropped because a lag or the target was unavailable: "
        f"{summary['rows_dropped_for_lags_or_target']:,}. "
        f"Model-ready rows: {summary['rows_model_ready']:,}."
    )

# --------------------------------------------------------------------------
# Tab 3 - Baseline
# --------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Non-ML baselines (the bar the model has to clear)")
    st.markdown(
        """
        - **Persistence** — *next level = current level.* Tank levels are autocorrelated, so this
          is annoyingly hard to beat on MAE.
        - **Physics** — *next level = current + inflow − average usage*, clipped to `[0, capacity]`.
          This is the actual water balance, and it is also the **fallback** used when a sensor
          fault or missing input makes the ML model untrustworthy.
        """
    )

    baseline_table = pd.DataFrame(bundle["baselines"]).T
    baseline_table = baseline_table[["mae", "rmse", "r2", "n"]].sort_values("mae")
    st.dataframe(
        baseline_table.style.format({"mae": "{:.2f} L", "rmse": "{:.2f} L", "r2": "{:.4f}", "n": "{:.0f}"}),
        use_container_width=True,
    )
    st.success(
        f"Best baseline: **{bundle['best_baseline']}** with MAE "
        f"{bundle['baselines'][bundle['best_baseline']]['mae']:.2f} L on the last 20% of the timeline."
    )

    window = test_frame.head(240)
    st.markdown("**Actual vs both baselines (first 240 test hours)**")
    st.line_chart(
        window.set_index("timestamp")[["actual_level", "pred_persistence", "pred_physics"]].rename(
            columns={
                "actual_level": "actual",
                "pred_persistence": "persistence baseline",
                "pred_physics": "physics baseline",
            }
        ),
        height=340,
    )

    st.caption(
        f"Evaluated on the held-out test window "
        f"{bundle['split']['test_period'][0]} → {bundle['split']['test_period'][1]} "
        f"({bundle['split']['test_rows']:,} hours)."
    )

# --------------------------------------------------------------------------
# Tab 4 - Trained model
# --------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Trained model")

    st.markdown(
        f"""
        A **Random Forest regressor** (200 trees, `random_state={bundle['seed']}`, fixed seed) is trained
        on the first **{bundle['split']['train_fraction']:.0%}** of the timeline and compared against
        **Ridge regression** (with standardised features) as an honest linear reference.
        No shuffling — see the *Evaluation* tab for why that matters.
        """
    )

    metric_rows = []
    for name, scores in bundle["metrics"].items():
        metric_rows.append(
            {
                "model": name,
                "MAE (L)": scores["mae"],
                "RMSE (L)": scores["rmse"],
                "R²": scores["r2"],
                "improvement vs best baseline": scores["improvement_pct"] / 100.0,
            }
        )
    st.dataframe(
        pd.DataFrame(metric_rows).set_index("model").style.format(
            {
                "MAE (L)": "{:.2f}",
                "RMSE (L)": "{:.2f}",
                "R²": "{:.4f}",
                "improvement vs best baseline": "{:+.1%}",
            }
        ),
        use_container_width=True,
    )

    winner = bundle["metrics"][bundle["model_name"]]
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Chosen model MAE",
        f"{winner['mae']:.1f} L",
        f"{winner['improvement_pct']:+.1f}% vs {bundle['best_baseline']} baseline",
        delta_color="normal",
    )
    col2.metric("Chosen model R²", f"{winner['r2']:.4f}")
    col3.metric("Training rows", f"{bundle['split']['train_rows']:,}")

    st.markdown("**Feature importance**")
    top_features = bundle["feature_importance"].head(15)
    st.bar_chart(top_features, height=380)
    st.caption(
        "Lag and rolling features dominate — the tank level one hour ago explains most of the level "
        "in one hour's time, and rolling rainfall captures storms that are still feeding the tank."
    )

    if bundle["cv"].get("ran"):
        cv = bundle["cv"]
        st.markdown("**Walk-forward validation on the training portion**")
        st.markdown(
            f"`TimeSeriesSplit(n_splits={cv['splits']})` — each fold trains on the past and validates on "
            f"the future. Mean MAE **{cv['mean_mae']:.2f} L** (± {cv['std_mae']:.2f}); "
            f"per-fold: {', '.join(f'{score:.1f}' for score in cv['fold_mae'])}."
        )

    st.markdown("**Persistence**")
    if artefacts["model_path"]:
        st.markdown(
            f"Saved with `joblib` to `{artefacts['model_path']}` "
            f"(includes the fitted estimator, the feature list, the metrics and the test frame)."
        )
        with open(artefacts["model_path"], "rb") as handle:
            st.download_button(
                "Download tank_model.joblib",
                data=handle.read(),
                file_name="tank_model.joblib",
                mime="application/octet-stream",
            )
        st.code(
            "from src.model import load_bundle\nbundle = load_bundle('models/tank_model.joblib')",
            language="python",
        )
    else:  # pragma: no cover
        st.info("Model kept in memory (the models/ directory is not writable).")

# --------------------------------------------------------------------------
# Tab 5 - Evaluation
# --------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Evaluation on the held-out test period")

    st.info(
        f"**Time-based split, never shuffled.** Train = first {bundle['split']['train_fraction']:.0%} "
        f"({bundle['split']['train_period'][0]} → {bundle['split']['train_period'][1]}, "
        f"{bundle['split']['train_rows']:,} rows). "
        f"Test = last {1 - bundle['split']['train_fraction']:.0%} "
        f"({bundle['split']['test_period'][0]} → {bundle['split']['test_period'][1]}, "
        f"{bundle['split']['test_rows']:,} rows). Random shuffling would leak the neighbouring hours of "
        "every test point into training and inflate the score, so it is not used anywhere in this project."
    )

    st.markdown("**Actual vs predicted next-hour level**")
    st.line_chart(
        test_frame.set_index("timestamp")[["actual_level", "pred"]].rename(
            columns={"actual_level": "actual", "pred": f"predicted ({bundle['model_name']})"}
        ),
        height=340,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Predicted vs actual (scatter)**")
        st.scatter_chart(
            test_frame.sample(min(len(test_frame), 1500), random_state=0),
            x="actual_level",
            y="pred",
            height=340,
        )
        st.caption("Points hugging the diagonal = good predictions.")
    with col_b:
        st.markdown("**Where the errors are (residuals)**")
        residuals = pd.DataFrame({"error": test_frame["error"]})
        counts, edges = np.histogram(residuals["error"], bins=24)
        st.bar_chart(
            pd.DataFrame(
                {"errors": counts},
                index=[f"{edges[i]:.0f}..{edges[i + 1]:.0f}" for i in range(len(counts))],
            ),
            height=340,
        )
        st.caption("error = predicted − actual, in litres.")

    st.markdown("**Error summary**")
    worst = test_frame.reindex(test_frame["error"].abs().sort_values(ascending=False).index).head(8)
    col_c, col_d = st.columns([1, 2])
    with col_c:
        st.dataframe(
            pd.DataFrame(
                {
                    "MAE (L)": [winner["mae"]],
                    "RMSE (L)": [winner["rmse"]],
                    "R²": [winner["r2"]],
                    "mean error (L)": [test_frame["error"].mean()],
                    "p95 |error| (L)": [test_frame["error"].abs().quantile(0.95)],
                }
            ).T.rename(columns={0: "value"}).style.format("{:.2f}"),
            use_container_width=True,
        )
    with col_d:
        st.markdown("*Biggest misses*")
        worst_display = worst[["timestamp", "current_level", "actual_level", "pred", "error"]].copy()
        worst_display[["current_level", "actual_level", "pred", "error"]] = worst_display[
            ["current_level", "actual_level", "pred", "error"]
        ].round(1)
        st.dataframe(worst_display, use_container_width=True, height=300)
    st.caption(
        "The largest errors tend to sit right at the start of a storm, when inflow jumps and the "
        "tank reaches capacity in the same hour — exactly the moment the overflow warning matters most."
    )

    st.markdown("**Overflow warning quality** (`predicted next level > 90% of capacity`)")
    st.markdown(
        "Missing an overflow is the costly error (the water is gone), so this is scored separately "
        "with precision/recall rather than lumped into MAE."
    )
    overflow_table = pd.DataFrame(bundle["overflow"]).T
    overflow_table["false_negatives"] = overflow_table["false_negatives"].astype(int)
    overflow_table["false_positives"] = overflow_table["false_positives"].astype(int)
    st.dataframe(
        overflow_table[["precision", "recall", "f1", "false_negatives", "false_positives", "actual_positive"]]
        .rename(
            columns={
                "false_negatives": "missed overflows",
                "false_positives": "false alarms",
                "actual_positive": "true overflow hours",
            }
        )
        .style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}"}),
        use_container_width=True,
    )

# --------------------------------------------------------------------------
# Tab 6 - Prediction
# --------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Live next-hour prediction")
    st.markdown(
        "Set the current readings and get a prediction + recommendation. Lag and rolling features are "
        "taken from the most recent 48 hours of the loaded history, then the *current* values below are "
        "appended and the identical feature pipeline runs — no train/serve skew."
    )

    history = clean.tail(48).copy()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        level_in = st.slider(
            "Current tank level (L)", 0.0, float(capacity), float(capacity * 0.55), 10.0
        )
    with col2:
        rainfall_in = st.slider("Rainfall now (mm/h)", 0.0, 40.0, 2.0, 0.5)
    with col3:
        inflow_in = st.slider("Catchment inflow (L/h)", 0.0, 3000.0, 250.0, 10.0)
    with col4:
        usage_in = st.slider("Usage (L/h)", 0.0, 600.0, 60.0, 5.0)

    current = {
        "tank_level_litres": level_in,
        "rainfall_mm": rainfall_in,
        "inflow_litres": inflow_in,
        "usage_litres": usage_in,
        "temperature_c": float(history["temperature_c"].median()),
    }

    # ------------------------------------------------------------------
    # Failure-case demo: it mutates the history the prediction sees.
    # ------------------------------------------------------------------
    fault_mode = st.session_state.get("fault_mode")
    demo_history = history.copy()
    if fault_mode == "negative":
        demo_history.loc[demo_history.index[-1], "tank_level_litres"] = -420.0
    elif fault_mode == "over_capacity":
        demo_history.loc[demo_history.index[-1], "tank_level_litres"] = capacity + 800.0
    elif fault_mode == "nan":
        demo_history.loc[demo_history.index[-1], "tank_level_litres"] = np.nan
    elif fault_mode == "flatline":
        demo_history.loc[demo_history.index[-8:], "tank_level_litres"] = 1234.0

    sensor_check = detect_sensor_fault(demo_history, capacity)

    try:
        recommendation = recommend_next_hour(
            bundle["model"], demo_history, current, capacity, sensor_check=sensor_check
        )
        invalid_input = None
    except ValueError as error:
        recommendation = None
        invalid_input = str(error)

    if invalid_input:
        st.error(f"**Input rejected.** {invalid_input}")
    else:
        left, right = st.columns([2, 1])
        with left:
            decision_card(
                recommendation.action,
                recommendation.reason,
                caption="Rule: >90% → PREVENT_OVERFLOW · 30–90% → HARVEST/USE · <30% → CONSERVE, "
                "with an earlier overflow trigger when heavy rain is already falling.",
            )
        with right:
            st.metric(
                "Predicted next-hour level",
                f"{recommendation.predicted_level:,.0f} L",
                f"{recommendation.predicted_fill_pct - recommendation.current_fill_pct:+.1f} pp of capacity",
            )
            st.metric("Predicted fill", f"{recommendation.predicted_fill_pct:.1f}%")
            st.metric("Current fill", f"{recommendation.current_fill_pct:.1f}%")

        fallback_banner(recommendation.used_fallback, recommendation.fallback_reason)

        st.markdown("**Prediction vs baselines for this input**")
        live_compare = pd.DataFrame(
            {
                "value": [
                    recommendation.predicted_level,
                    level_in,
                    float(np.clip(level_in + inflow_in - float(history["usage_litres"].tail(24).mean()), 0, capacity)),
                    0.9 * capacity,
                    0.3 * capacity,
                ]
            },
            index=[
                f"model prediction ({'fallback' if recommendation.used_fallback else bundle['model_name']})",
                "persistence baseline (current level)",
                "physics baseline",
                "overflow line (90%)",
                "reserve line (30%)",
            ],
        )
        st.dataframe(live_compare.style.format("{:,.0f} L"), use_container_width=True)

    with st.expander("🚨 Failure case demo — click a button", expanded=False):
        st.markdown(
            "These buttons deliberately break the input. The app must never crash: it should detect "
            "the fault, say so, and fall back to the physics baseline."
        )
        b1, b2, b3, b4 = st.columns(4)
        if b1.button("Negative level", use_container_width=True):
            st.session_state["fault_mode"] = "negative"
            st.rerun()
        if b2.button("Level > capacity", use_container_width=True):
            st.session_state["fault_mode"] = "over_capacity"
            st.rerun()
        if b3.button("NaN reading", use_container_width=True):
            st.session_state["fault_mode"] = "nan"
            st.rerun()
        if b4.button("Flatlined sensor", use_container_width=True):
            st.session_state["fault_mode"] = "flatline"
            st.rerun()

        if st.button("Reset to healthy sensors"):
            st.session_state.pop("fault_mode", None)
            st.rerun()

        st.markdown("**What the detector sees right now**")
        st.json(
            {
                "fault": sensor_check.fault,
                "reasons": sensor_check.reasons,
                "flatline_run": sensor_check.flatline_run,
                "rows_checked": sensor_check.checked_rows,
                "active_demo": fault_mode or "none (healthy history)",
            }
        )
        st.markdown("**Out-of-range user input** is rejected before it reaches the model:")
        try:
            recommend_next_hour(
                bundle["model"],
                demo_history,
                {**current, "tank_level_litres": capacity + 1000.0},
                capacity,
            )
        except ValueError as error:
            st.error(str(error))

# --------------------------------------------------------------------------
# Tab 7 - Product decision
# --------------------------------------------------------------------------
with tabs[6]:
    st.subheader("The product decision")

    st.markdown(
        f"""
        The controller looks at the **predicted next-hour fill level** and the **rain already falling**
        (rolling rainfall is a cheap, honest forecast — what is in the last 3/24 hours is what is still
        feeding the tank):

        | Predicted fill | Rain context | Action |
        |---|---|---|
        | > {OVERFLOW_PCT:.0f}% | any | **PREVENT_OVERFLOW** — open divert/overflow, draw down now |
        | ≥ {EARLY_OVERFLOW_PCT:.0f}% | ≥ {HEAVY_RAIN_3H_MM:.0f} mm/3 h or ≥ {HEAVY_RAIN_24H_MM:.0f} mm/24 h |
        **PREVENT_OVERFLOW** — trigger early, the storm is still arriving |
        | ≥ {LOW_PCT:.0f}% | wet, filling | **USE** — draw water now (irrigation/household) to keep headroom |
        | {LOW_PCT:.0f}–{OVERFLOW_PCT:.0f}% | dry/stable | **HARVEST** — keep collecting |
        | < {LOW_PCT:.0f}% | any | **CONSERVE** — restrict non-essential use |

        `HARVEST`, `USE` and `PREVENT_OVERFLOW` are the three operating actions; `CONSERVE` is the
        restrictive safety mode the brief asks for below {LOW_PCT:.0f}%.
        """
    )

    decisions = [
        decide(fill, rain3, rain24)
        for fill, rain3, rain24 in zip(
            test_frame["pred_fill_pct"], test_frame["rain_roll_3"], test_frame["rain_roll_24"]
        )
    ]
    action_series = pd.Series([decision.action for decision in decisions], name="action")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("**Actions taken over the test period**")
        st.dataframe(
            action_series.value_counts().rename("hours").to_frame(),
            use_container_width=True,
        )
    with col2:
        st.bar_chart(action_series.value_counts(), height=280)

    timeline = test_frame[["timestamp", "pred_fill_pct", "rain_roll_24"]].copy()
    timeline["overflow line (90%)"] = OVERFLOW_PCT
    timeline["reserve line (30%)"] = LOW_PCT
    st.markdown("**Predicted fill % through the test period, with the decision bands drawn in**")
    st.line_chart(timeline.set_index("timestamp"), height=340)

    st.markdown("**Decisions taken in the 10 most recent test hours**")
    recent = test_frame.tail(10).copy()
    recent["action"] = action_series.tail(10).to_numpy()
    recent["reason"] = [decision.reason for decision in decisions[-10:]]
    recent = recent[["timestamp", "pred_fill_pct", "rain_roll_3", "rain_roll_24", "action", "reason"]]
    recent[["pred_fill_pct", "rain_roll_3", "rain_roll_24"]] = recent[
        ["pred_fill_pct", "rain_roll_3", "rain_roll_24"]
    ].round(2)
    st.dataframe(recent, use_container_width=True, height=320)

    latest = decisions[-1]
    decision_card(
        latest.action,
        latest.reason,
        caption=f"Most recent test hour: {test_frame['timestamp'].iloc[-1]} "
        f"(predicted fill {latest.fill_pct:.1f}%).",
    )

# --------------------------------------------------------------------------
# Tab 8 - Impact
# --------------------------------------------------------------------------
with tabs[7]:
    st.subheader("Measured impact: what the controller actually changed")

    no_controller = impact["no_controller"]
    with_controller = impact["with_controller"]

    st.markdown(
        f"The test period ({impact['test_days']:.1f} days, {impact['test_hours']:,} hours) is simulated "
        "**twice with identical weather and demand**: once with no controller at all (the tank simply "
        "spills when full and runs dry when empty), and once where every hour follows the controller's "
        "recommendation:"
    )
    st.markdown(
        f"""
        * **PREVENT_OVERFLOW** — divert/use water *only* to the extent that it would otherwise spill
          this hour, capped at {MAX_PREVENTIVE_DRAW_LITRES:,.0f} L/h (pump capacity). Sizing the draw
          this way is the important detail: the controller never gives away *stored* water, so it can
          save a spill without creating a shortage later in the same dry spell.
        * **CONSERVE** — cut non-essential demand by {CONSERVE_USAGE_REDUCTION:.0%} while the tank is
          below the {LOW_PCT:.0f}% line, which keeps the level above the 15% reserve.
        * **HARVEST / USE** — no intervention, water simply stays in the tank.
        """
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Litres saved from overflow",
        f"{impact['litres_saved_from_overflow']:,.0f} L",
        f"{impact['overflow_reduction_pct']:.1f}% less spilt",
    )
    m2.metric("Overflow events avoided", f"{impact['overflow_events_avoided']:,}",
              f"{no_controller['overflow_events']} → {with_controller['overflow_events']}")
    m3.metric("Shortage events avoided", f"{impact['shortage_events_avoided']:,}",
              f"{no_controller['shortage_events']} → {with_controller['shortage_events']}")
    m4.metric(
        "Water used productively",
        f"{impact['water_used_productively_litres']:,.0f} L",
        "diverted instead of spilt",
    )

    m5, m6, m7 = st.columns(3)
    m5.metric(
        "Demand postponed (CONSERVE)",
        f"{impact['litres_conserved_by_restricting_demand']:,.0f} L",
        "non-essential use deferred",
    )
    m6.metric(
        "Lowest level reached",
        f"{impact['with_controller']['levels'].min():,.0f} L",
        f"no controller: {impact['no_controller']['levels'].min():,.0f} L",
        delta_color="off",
    )
    m7.metric(
        "Reserve line (15%)",
        f"{impact['no_controller']['shortage_hours']:,} → {impact['with_controller']['shortage_hours']:,} hours below",
        "shortage exposure",
        delta_color="off",
    )

    st.markdown(
        "*Saved litres are water that would have gone over the weir and is instead routed to "
        "irrigation/household use — it is not created out of nothing, it is rescued.*"
    )

    comparison = pd.DataFrame(
        {
            "metric": [
                "Overflow (L)",
                "Overflow hours",
                "Overflow events",
                "Shortage hours (below 15% reserve)",
                "Shortage events",
                "Unmet demand (L)",
                "Demand postponed by CONSERVE (L)",
                "Lowest level (L)",
                "Mean level (L)",
            ],
            "no controller": [
                no_controller["overflow_litres"],
                no_controller["overflow_hours"],
                no_controller["overflow_events"],
                no_controller["shortage_hours"],
                no_controller["shortage_events"],
                no_controller["unmet_demand_litres"],
                no_controller["demand_restricted_litres"],
                no_controller["levels"].min(),
                no_controller["mean_level"],
            ],
            "with controller": [
                with_controller["overflow_litres"],
                with_controller["overflow_hours"],
                with_controller["overflow_events"],
                with_controller["shortage_hours"],
                with_controller["shortage_events"],
                with_controller["unmet_demand_litres"],
                with_controller["demand_restricted_litres"],
                with_controller["levels"].min(),
                with_controller["mean_level"],
            ],
        }
    ).set_index("metric")
    st.dataframe(comparison.round(1), use_container_width=True)

    trajectories = test_frame[["timestamp"]].copy()
    trajectories["level: no controller"] = no_controller["levels"]
    trajectories["level: with controller"] = with_controller["levels"]
    st.markdown("**Tank level: no controller vs with controller**")
    st.line_chart(trajectories.set_index("timestamp"), height=340)

    cumulative = test_frame[["timestamp"]].copy()
    cumulative["cumulative overflow: no controller"] = np.cumsum(no_controller["overflow_series"])
    cumulative["cumulative overflow: with controller"] = np.cumsum(with_controller["overflow_series"])
    st.markdown("**Cumulative water lost to overflow**")
    st.line_chart(cumulative.set_index("timestamp"), height=320)

    st.markdown("**Actions the controller took during the simulation**")
    actions = pd.Series(with_controller["actions"], name="action").value_counts().rename("hours")
    st.bar_chart(actions, height=280)

    st.success(
        f"**Bottom line:** over {impact['test_days']:.1f} days the controller avoided "
        f"{impact['overflow_events_avoided']} overflow events and "
        f"{impact['shortage_events_avoided']} shortage events, keeping "
        f"{impact['litres_saved_from_overflow']:,.0f} litres of harvested rainwater in productive use."
    )

    with st.expander("How this simulation is defined"):
        st.markdown(
            """
            * **Shortage hour** = the tank is at or below the 15% reserve line (750 L of a 5,000 L
              tank: less than two days of household demand) or demand cannot be met that hour.
            * **Event** = one contiguous run of hours (so a 6-hour spill is 1 event, not 6).
            * The controller reacts to the **model's** predicted next-hour level (and the rain already
              recorded), so a better model directly translates into a better impact number.
            * Both passes start from the same level and see the same inflow and demand series.
            """
        )
