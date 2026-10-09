"""Product decision logic, failure handling and impact metrics.

The controller turns a *predicted* next-hour tank level into one visible action:

===========================  ===================================
Action                        Meaning
===========================  ===================================
``PREVENT_OVERFLOW``          open the overflow/divert valve and schedule a
                              draw-down now - the tank is about to spill
``HARVEST``                   keep collecting; there is headroom
``USE``                       plenty of water is coming, so draw some now
                              (irrigation / household) to make room
``CONSERVE``                  the restrictive mode below 30%: cut
                              non-essential demand
===========================  ===================================

``HARVEST`` / ``USE`` / ``PREVENT_OVERFLOW`` are the three operating actions;
``CONSERVE`` is the low-level safety mode that the brief asks for below 30%.

Rainfall *forecast* (rolling rainfall) deliberately shifts the trigger earlier:
a tank that is 80% full and stable is fine, but 80% full with 15 mm of rain in
the last 24 hours and more incoming is an overflow waiting to happen.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from src.preprocess import FEATURES, build_live_features

# --------------------------------------------------------------------------
# Thresholds
# --------------------------------------------------------------------------
OVERFLOW_PCT = 90.0          # > 90% of capacity -> spill risk
LOW_PCT = 30.0               # < 30% of capacity -> conserve
EARLY_OVERFLOW_PCT = 75.0    # high but not yet critical...
HEAVY_RAIN_3H_MM = 4.0       # ...plus rain in the last 3 hours
HEAVY_RAIN_24H_MM = 12.0     # ...or a wet 24 hours
USE_BAND_PCT = 55.0          # >= this and wet weather -> draw water now

#: Physical capabilities of the controller - not model parameters.
MAX_PREVENTIVE_DRAW_LITRES = 700.0   # max extra draw per hour (pump/irrigation limit)
CONSERVE_USAGE_REDUCTION = 0.40      # 40% cut to non-essential demand

#: Product definitions used by the impact simulation.
#: The reserve line is the level below which the tank can no longer ride out a
#: dry spell, so crossing it is a "shortage".
MIN_RESERVE_FRACTION = 0.15
OVERFLOW_EVENT_EPS_LITRES = 1.0

ACTION_PREVENT = "PREVENT_OVERFLOW"
ACTION_HARVEST = "HARVEST"
ACTION_USE = "USE"
ACTION_CONSERVE = "CONSERVE"

ACTION_COLOURS = {
    ACTION_PREVENT: "#d62728",  # red
    ACTION_HARVEST: "#2ca02c",  # green
    ACTION_USE: "#1f77b4",      # blue
    ACTION_CONSERVE: "#ff7f0e",  # orange
}
ACTION_ICONS = {
    ACTION_PREVENT: "!!",
    ACTION_HARVEST: "OK",
    ACTION_USE: ">>",
    ACTION_CONSERVE: "xx",
}

FLATLINE_HOURS = 6


# --------------------------------------------------------------------------
# 1. The decision rule
# --------------------------------------------------------------------------
@dataclass
class Decision:
    """One controller decision plus the human-readable why."""

    action: str
    reason: str
    fill_pct: float
    rain_roll_3: float = 0.0
    rain_roll_24: float = 0.0

    @property
    def colour(self) -> str:
        return ACTION_COLOURS.get(self.action, "#7f7f7f")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["colour"] = self.colour
        return payload


def decide(
    fill_pct: float,
    rain_roll_3: float = 0.0,
    rain_roll_24: float = 0.0,
) -> Decision:
    """Map a predicted fill percentage (+ rain context) to one action.

    Parameters
    ----------
    fill_pct : predicted next-hour level as a percentage of tank capacity.
    rain_roll_3 : rainfall total over the last 3 hours (mm).
    rain_roll_24 : rainfall total over the last 24 hours (mm).
    """
    fill_pct = float(fill_pct)
    rain_roll_3 = float(rain_roll_3 or 0.0)
    rain_roll_24 = float(rain_roll_24 or 0.0)

    heavy_short = rain_roll_3 >= HEAVY_RAIN_3H_MM
    heavy_long = rain_roll_24 >= HEAVY_RAIN_24H_MM

    if fill_pct > OVERFLOW_PCT:
        return Decision(
            ACTION_PREVENT,
            f"Predicted level {fill_pct:.0f}% of capacity - above the {OVERFLOW_PCT:.0f}% spill line. "
            "Open the overflow/divert valve and draw down now.",
            fill_pct,
            rain_roll_3,
            rain_roll_24,
        )

    if fill_pct >= EARLY_OVERFLOW_PCT and (heavy_short or heavy_long):
        trigger = (
            f"{rain_roll_3:.1f} mm in the last 3 h"
            if heavy_short
            else f"{rain_roll_24:.1f} mm in the last 24 h"
        )
        return Decision(
            ACTION_PREVENT,
            f"Only {fill_pct:.0f}% full but {trigger} of rain is already in the tank's path - "
            "pre-emptive draw-down avoids a spill.",
            fill_pct,
            rain_roll_3,
            rain_roll_24,
        )

    if fill_pct < LOW_PCT:
        return Decision(
            ACTION_CONSERVE,
            f"Predicted level {fill_pct:.0f}% of capacity - below the {LOW_PCT:.0f}% reserve line. "
            "Restrict non-essential use (irrigation, car washing).",
            fill_pct,
            rain_roll_3,
            rain_roll_24,
        )

    if fill_pct >= USE_BAND_PCT and (heavy_short or heavy_long):
        return Decision(
            ACTION_USE,
            f"{fill_pct:.0f}% full with {max(rain_roll_3, rain_roll_24):.1f} mm of incoming rain - "
            "draw water now (irrigation/household) to keep headroom for the storm.",
            fill_pct,
            rain_roll_3,
            rain_roll_24,
        )

    return Decision(
        ACTION_HARVEST,
        f"Predicted level {fill_pct:.0f}% of capacity - inside the comfortable "
        f"{LOW_PCT:.0f}-{OVERFLOW_PCT:.0f}% band. Keep collecting.",
        fill_pct,
        rain_roll_3,
        rain_roll_24,
    )


# --------------------------------------------------------------------------
# 2. Input validation (failure case: out-of-range user input)
# --------------------------------------------------------------------------
def validate_level(level: float, capacity: float) -> tuple[bool, str]:
    """Return ``(is_valid, message)`` for a tank level against physical limits."""
    if level is None or not np.isfinite(level):
        return False, "Tank level must be a real number (got NaN/missing)."
    if level < 0:
        return False, f"Tank level cannot be negative (got {level:.1f} L)."
    if level > capacity:
        return False, f"Tank level {level:.1f} L exceeds tank capacity {capacity:.1f} L."
    return True, "Tank level is within physical limits."


def validate_user_input(
    level: float,
    rainfall_mm: float,
    inflow_litres: float,
    usage_litres: float,
    capacity: float,
    max_rainfall_mm: float = 100.0,
    max_inflow_litres: float = 50_000.0,
    max_usage_litres: float = 10_000.0,
) -> list[str]:
    """Validate the sliders/CSV row a user can type in. Returns a list of problems."""
    errors: list[str] = []

    for label, value in [
        ("Tank level", level),
        ("Rainfall", rainfall_mm),
        ("Inflow", inflow_litres),
        ("Usage", usage_litres),
    ]:
        if value is None or not np.isfinite(float(value)):
            errors.append(f"{label} must be a real number (got NaN/missing).")

    if errors:  # nothing else is meaningful if a value is not a number
        return errors

    ok, message = validate_level(float(level), capacity)
    if not ok:
        errors.append(message)

    if rainfall_mm < 0:
        errors.append(f"Rainfall cannot be negative (got {rainfall_mm:.2f} mm).")
    elif rainfall_mm > max_rainfall_mm:
        errors.append(f"Rainfall {rainfall_mm:.1f} mm is implausible for one hour (max {max_rainfall_mm:.0f} mm).")

    if inflow_litres < 0:
        errors.append(f"Inflow cannot be negative (got {inflow_litres:.1f} L).")
    elif inflow_litres > max_inflow_litres:
        errors.append(f"Inflow {inflow_litres:.0f} L/h is implausible (max {max_inflow_litres:,.0f} L/h).")

    if usage_litres < 0:
        errors.append(f"Usage cannot be negative (got {usage_litres:.1f} L).")
    elif usage_litres > max_usage_litres:
        errors.append(f"Usage {usage_litres:.0f} L/h is implausible (max {max_usage_litres:,.0f} L/h).")

    return errors


# --------------------------------------------------------------------------
# 3. Sensor-fault detection (failure case: bad sensor -> fallback)
# --------------------------------------------------------------------------
@dataclass
class SensorCheck:
    """Result of a sensor health check."""

    fault: bool
    reasons: list[str] = field(default_factory=list)
    flatline_run: int = 0
    checked_rows: int = 0

    @property
    def message(self) -> str:
        return "; ".join(self.reasons) if self.reasons else "Sensors look healthy."


def _longest_flatline_run(series: pd.Series) -> int:
    """Length of the longest run of identical consecutive values."""
    values = series.dropna().to_numpy()
    if len(values) == 0:
        return 0
    best = current = 1
    for previous, value in zip(values[:-1], values[1:]):
        current = current + 1 if value == previous else 1
        best = max(best, current)
    return best


def detect_sensor_fault(
    history: pd.DataFrame,
    capacity: float,
    flatline_hours: int = FLATLINE_HOURS,
    level_column: str = "tank_level_litres",
) -> SensorCheck:
    """Look for the four classic tank-sensor faults.

    1. ``NaN`` readings in the recent window (sensor dropped out),
    2. negative levels (dead float / wiring fault),
    3. levels above capacity (scaling error or a stuck float),
    4. a flatlined value repeated for `flatline_hours` in a row (frozen sensor).
    """
    reasons: list[str] = []

    if history is None or len(history) == 0:
        return SensorCheck(True, ["No recent history available to check."], 0, 0)

    if level_column not in history.columns:
        return SensorCheck(
            True, [f"Column '{level_column}' is missing - cannot trust the reading."], 0, int(len(history))
        )

    window = history.tail(max(flatline_hours + 2, 8))
    level = pd.to_numeric(window[level_column], errors="coerce")
    checked = int(len(window))

    nan_count = int(level.isna().sum())
    if nan_count:
        reasons.append(f"{nan_count} of the last {checked} level readings are missing (NaN)")

    negative = int((level < 0).sum())
    if negative:
        reasons.append(f"{negative} negative level reading(s) (e.g. {level.min():.0f} L)")

    over = int((level > capacity).sum())
    if over:
        reasons.append(f"{over} reading(s) above capacity (max {level.max():.0f} L > {capacity:.0f} L)")

    flatline_run = _longest_flatline_run(level)
    if flatline_run >= flatline_hours:
        frozen = level.dropna().iloc[-1]
        reasons.append(
            f"sensor appears frozen: {flatline_run} identical readings in a row ({frozen:.0f} L)"
        )

    if "inflow_litres" in window.columns:
        inflow = pd.to_numeric(window["inflow_litres"], errors="coerce")
        negative_inflow = int((inflow < 0).sum())
        if negative_inflow:
            reasons.append(f"{negative_inflow} negative inflow reading(s)")

    return SensorCheck(bool(reasons), reasons, flatline_run, checked)


# --------------------------------------------------------------------------
# 4. Fallback prediction
# --------------------------------------------------------------------------
def physics_prediction(
    current_level: float,
    inflow_litres: float,
    average_usage_litres: float,
    capacity: float,
) -> float:
    """Sensor-independent fallback: pure water balance, clipped to the tank."""
    predicted = float(current_level) + float(inflow_litres) - float(average_usage_litres)
    return float(np.clip(predicted, 0.0, capacity))


# --------------------------------------------------------------------------
# 5. End-to-end recommendation for one row
# --------------------------------------------------------------------------
@dataclass
class Recommendation:
    """Everything the UI needs to render one live prediction."""

    predicted_level: float
    predicted_fill_pct: float
    current_fill_pct: float
    action: str
    reason: str
    used_fallback: bool
    fallback_reason: str | None
    decision: Decision

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["colour"] = self.decision.colour
        return payload


def recommend_next_hour(
    model,
    history: pd.DataFrame,
    current: dict,
    capacity: float,
    sensor_check: SensorCheck | None = None,
) -> Recommendation:
    """Predict the next-hour level and turn it into an action.

    If :func:`detect_sensor_fault` flags the recent history, the ML model is
    *not* trusted and the physics baseline is used instead - the app surfaces
    this as "Sensor fault suspected - using fallback baseline".
    """
    errors = validate_user_input(
        current.get("tank_level_litres", float("nan")),
        current.get("rainfall_mm", 0.0),
        current.get("inflow_litres", 0.0),
        current.get("usage_litres", 0.0),
        capacity,
    )
    if errors:
        raise ValueError("Invalid input: " + " ".join(errors))

    check = sensor_check or detect_sensor_fault(history, capacity)

    live_row = build_live_features(history, current, capacity)
    rain_roll_3 = float(live_row["rain_roll_3"].iloc[0])
    rain_roll_24 = float(live_row["rain_roll_24"].iloc[0])

    if check.fault:
        used_fallback = True
        fallback_reason = check.message
        predicted = physics_prediction(
            current["tank_level_litres"],
            current.get("inflow_litres", 0.0),
            float(live_row["usage_roll_mean_24"].iloc[0]),
            capacity,
        )
    else:
        used_fallback = False
        fallback_reason = None
        predicted = float(model.predict(live_row[FEATURES])[0])
        predicted = float(np.clip(predicted, 0.0, capacity))

    fill_pct = predicted / capacity * 100.0
    decision = decide(fill_pct, rain_roll_3, rain_roll_24)

    return Recommendation(
        predicted_level=predicted,
        predicted_fill_pct=fill_pct,
        current_fill_pct=float(current["tank_level_litres"]) / capacity * 100.0,
        action=decision.action,
        reason=decision.reason,
        used_fallback=used_fallback,
        fallback_reason=fallback_reason,
        decision=decision,
    )


# --------------------------------------------------------------------------
# 6. Impact metric: simulate the test period with and without the controller
# --------------------------------------------------------------------------
def _count_events(mask: np.ndarray) -> int:
    """Count runs of consecutive True values (an 'event' = one contiguous run)."""
    events = 0
    previous = False
    for value in mask:
        value = bool(value)
        if value and not previous:
            events += 1
        previous = value
    return events


def _simulate(
    frame: pd.DataFrame,
    capacity: float,
    predicted_level: np.ndarray | None = None,
) -> dict:
    """Run one pass of the water balance.

    When `predicted_level` is ``None`` we simulate the *uncontrolled* tank:
    the level simply follows inflow minus demand and spills once it is full.
    When predictions are supplied, the controller inspects them each hour and
    either draws water down pre-emptively or restricts demand.
    """
    inflow = frame["inflow_litres"].to_numpy(dtype=float)
    demand = frame["usage_litres"].to_numpy(dtype=float)
    rain3 = frame["rain_roll_3"].to_numpy(dtype=float)
    rain24 = frame["rain_roll_24"].to_numpy(dtype=float)
    reserve = capacity * MIN_RESERVE_FRACTION

    controlled = predicted_level is not None
    if controlled:
        predicted_level = np.asarray(predicted_level, dtype=float)

    level = float(frame["current_level"].iloc[0])
    overflow_total = 0.0
    productive_extra_used = 0.0
    demand_reduction_total = 0.0
    unmet_total = 0.0
    overflow_hours: list[float] = []
    shortage_flags: list[bool] = []
    levels: list[float] = []
    actions: list[str] = []

    for t in range(len(frame)):
        available = level + inflow[t]
        action = ACTION_HARVEST
        essential = demand[t]

        if controlled:
            fill_pct = predicted_level[t] / capacity * 100.0
            decision = decide(fill_pct, rain3[t], rain24[t])
            action = decision.action
            if action == ACTION_CONSERVE:
                essential = demand[t] * (1.0 - CONSERVE_USAGE_REDUCTION)
                demand_reduction_total += demand[t] - essential

        essential_used = min(essential, available)
        spare = available - essential_used  # water above what is being drawn now

        # PREVENT_OVERFLOW: draw and *use* water only to the extent that it
        # would otherwise spill over the weir this hour, capped by the pump.
        # Sizing the draw this way guarantees the controller never gives away
        # stored water - so it can save spillage without creating a shortage
        # later in the same dry spell.
        extra_used = 0.0
        if action == ACTION_PREVENT:
            spill_forecast = max(0.0, spare - capacity)
            extra_used = min(MAX_PREVENTIVE_DRAW_LITRES, spill_forecast)

        new_level = available - essential_used - extra_used
        overflow = max(0.0, new_level - capacity)
        overflow_total += overflow
        overflow_hours.append(overflow)
        productive_extra_used += extra_used
        unmet = essential - essential_used
        unmet_total += unmet

        level = float(np.clip(new_level - overflow, 0.0, capacity))
        levels.append(level)
        shortage_flags.append(bool(unmet > OVERFLOW_EVENT_EPS_LITRES or level <= reserve))
        actions.append(action)

    overflow_array = np.asarray(overflow_hours)
    shortage_array = np.asarray(shortage_flags)

    return {
        "controlled": controlled,
        "overflow_litres": float(overflow_total),
        "overflow_hours": int((overflow_array > OVERFLOW_EVENT_EPS_LITRES).sum()),
        "overflow_events": _count_events(overflow_array > OVERFLOW_EVENT_EPS_LITRES),
        "shortage_hours": int(shortage_array.sum()),
        "shortage_events": _count_events(shortage_array),
        "unmet_demand_litres": float(unmet_total),
        "productive_extra_used_litres": float(productive_extra_used),
        "demand_restricted_litres": float(demand_reduction_total),
        "final_level": float(levels[-1]) if levels else float("nan"),
        "mean_level": float(np.mean(levels)) if levels else float("nan"),
        "levels": np.asarray(levels, dtype=float),
        "actions": actions,
        "overflow_series": overflow_array,
    }


def simulate_impact(
    test_frame: pd.DataFrame,
    predicted_level: np.ndarray,
    capacity: float,
) -> dict:
    """Compare the test period *without* and *with* the controller.

    Both passes see identical exogenous inputs (the same rainfall-driven inflow
    and the same demand), so every difference in water saved is attributable to
    the controller's actions: diverting water that would have spilt into
    productive use, and restricting non-essential demand when the tank is low.
    """
    no_controller = _simulate(test_frame, capacity, predicted_level=None)
    with_controller = _simulate(test_frame, capacity, predicted_level=predicted_level)

    saved = no_controller["overflow_litres"] - with_controller["overflow_litres"]
    events_avoided = no_controller["overflow_events"] - with_controller["overflow_events"]
    shortages_avoided = no_controller["shortage_events"] - with_controller["shortage_events"]

    baseline_overflow = no_controller["overflow_litres"]
    reduction_pct = (saved / baseline_overflow * 100.0) if baseline_overflow > 0 else 0.0

    hours = max(len(test_frame), 1)
    return {
        "no_controller": no_controller,
        "with_controller": with_controller,
        "litres_saved_from_overflow": float(max(saved, 0.0)),
        "overflow_events_avoided": int(max(events_avoided, 0)),
        "shortage_events_avoided": int(max(shortages_avoided, 0)),
        "overflow_reduction_pct": float(reduction_pct),
        "water_used_productively_litres": float(with_controller["productive_extra_used_litres"]),
        # Litres of demand that the controller asked the household to postpone.
        "litres_conserved_by_restricting_demand": float(
            with_controller["demand_restricted_litres"]
        ),
        # Essential demand that could not be met at all (was > 0, now 0).
        "unmet_demand_avoided_litres": float(
            no_controller["unmet_demand_litres"] - with_controller["unmet_demand_litres"]
        ),
        "test_hours": int(hours),
        "test_days": round(hours / 24.0, 1),
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    from src.model import load_bundle

    bundle = load_bundle()
    frame = bundle["test_frame"]
    impact = simulate_impact(frame, frame["pred"].to_numpy(), bundle["capacity"])
    print("Impact over the test period:")
    for key in (
        "litres_saved_from_overflow",
        "overflow_events_avoided",
        "shortage_events_avoided",
        "overflow_reduction_pct",
    ):
        print(f"  {key:32s} {impact[key]:,.2f}")
