"""ForecastOps core forecasting engine.

Hierarchical demand forecasting with:
- per-series ETS (ExponentialSmoothing, additive trend + additive seasonality, period 12)
- walk-forward backtest (expanding train, 6 folds, 3-month horizon) with
  WAPE / MAPE / bias per series and aggregated per level (component, zone, national)
- hierarchical reconciliation: bottom-up (sum of leaf forecasts) vs top-down
  (national forecast disaggregated by historical share), winner chosen by
  national WAPE on the backtest
- scenario engine: multiplier by zone / component applied before reconciliation
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

SEASONAL_PERIOD = 12
REQUIRED_COLUMNS = {"date", "component", "zone", "units"}
DEFAULT_COMPONENTS = ["Compressor", "Condenser", "Evaporator"]
DEFAULT_ZONES = ["North", "South", "West", "East"]

BU = "bottom-up"
TD = "top-down"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Weighted Absolute Percentage Error = sum|a-f| / sum|a|."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    denom = float(np.abs(actual).sum())
    if denom == 0:
        return float("nan")
    return float(np.abs(actual - forecast).sum() / denom)


def mape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean Absolute Percentage Error (guarding zero actuals)."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    mask = np.abs(actual) > 1e-9
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])))


def bias(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Signed bias = mean(f - a) / mean(a), in percent."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    denom = float(np.mean(actual))
    if abs(denom) < 1e-9:
        return float("nan")
    return float(100.0 * (np.mean(forecast) - np.mean(actual)) / denom)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Load demand CSV with schema date, component, zone, units (monthly)."""
    df = pd.read_csv(path, parse_dates=["date"])
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    df = df.sort_values(["component", "zone", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M")
    return df


def series_history(df: pd.DataFrame, component: str, zone: str) -> pd.Series:
    """Monthly units history for one leaf (component, zone), indexed by Period."""
    sub = df[(df["component"] == component) & (df["zone"] == zone)]
    s = sub.set_index("date")["units"].astype(float).sort_index()
    return s


def leaf_keys(df: pd.DataFrame) -> list[tuple[str, str]]:
    comps = sorted(df["component"].unique())
    zones = sorted(df["zone"].unique())
    return [(c, z) for c in comps for z in zones]


# ---------------------------------------------------------------------------
# ETS forecasting
# ---------------------------------------------------------------------------

def ets_forecast(values: np.ndarray, horizon: int) -> np.ndarray:
    """Fit additive ETS and forecast `horizon` steps ahead.

    Falls back to seasonal-naive then to a flat last-value forecast if the
    model cannot be fit, so the pipeline never hard-fails on short series.
    """
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.full(horizon, max(float(values[-1]) if len(values) else 0.0, 0.0))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                values,
                trend="add",
                seasonal="add",
                seasonal_periods=SEASONAL_PERIOD,
                initialization_method="estimated",
            ).fit(optimized=True)
            fc = np.asarray(model.forecast(horizon), dtype=float)
        if np.all(np.isfinite(fc)):
            return fc
    except Exception:
        pass
    # fallback 1: seasonal naive
    if len(values) >= SEASONAL_PERIOD:
        last_cycle = values[-SEASONAL_PERIOD:]
        fc = np.array([last_cycle[i % SEASONAL_PERIOD] for i in range(horizon)])
        return fc
    # fallback 2: flat
    return np.full(horizon, float(values[-1]))


def forecast_all_leaves(
    df: pd.DataFrame, horizon: int, keys: list[tuple[str, str]] | None = None
) -> dict[tuple[str, str], np.ndarray]:
    """Forecast every leaf series on full history."""
    keys = keys or leaf_keys(df)
    out: dict[tuple[str, str], np.ndarray] = {}
    for key in keys:
        hist = series_history(df, key[0], key[1]).values
        out[key] = ets_forecast(hist, horizon)
    return out


def national_history(df: pd.DataFrame) -> np.ndarray:
    """National monthly totals (sum over all leaves)."""
    return df.groupby("date")["units"].sum().astype(float).sort_index().values


# ---------------------------------------------------------------------------
# Scenario engine
# ---------------------------------------------------------------------------

def apply_scenario(
    leaf_fc: dict[tuple[str, str], np.ndarray],
    zones: list[str] | None = None,
    components: list[str] | None = None,
    multiplier: float = 1.0,
) -> dict[tuple[str, str], np.ndarray]:
    """Apply a demand multiplier to matching leaf forecasts (pre-reconciliation)."""
    zones = set(zones or [])
    components = set(components or [])
    out: dict[tuple[str, str], np.ndarray] = {}
    for key, fc in leaf_fc.items():
        hit = (key[1] in zones) or (key[0] in components)
        out[key] = fc * multiplier if hit else fc.copy()
    return out


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

def historical_shares(df: pd.DataFrame, trailing: int = 12) -> dict[tuple[str, str], float]:
    """Share of each leaf in trailing national demand (for top-down split)."""
    keys = leaf_keys(df)
    totals: dict[tuple[str, str], float] = {}
    grand = 0.0
    for key in keys:
        hist = series_history(df, key[0], key[1]).values
        t = float(hist[-trailing:].sum())
        totals[key] = t
        grand += t
    if grand <= 0:
        n = len(keys)
        return {k: 1.0 / n for k in keys}
    return {k: v / grand for k, v in totals.items()}


def reconcile(
    leaf_fc: dict[tuple[str, str], np.ndarray],
    national_fc: np.ndarray,
    shares: dict[tuple[str, str], float],
    method: str,
) -> tuple[dict[tuple[str, str], np.ndarray], np.ndarray]:
    """Reconcile leaf forecasts with the national forecast.

    bottom-up: national = sum of leaf forecasts (leaves unchanged).
    top-down: national forecast is authoritative; leaves are its share split.
    Returns (leaf_forecast, national_forecast).
    """
    if method == TD:
        total_share = sum(shares.values()) or 1.0
        reconciled = {
            key: national_fc * (shares.get(key, 0.0) / total_share) for key in leaf_fc
        }
        return reconciled, np.asarray(national_fc, dtype=float).copy()
    # bottom-up (default)
    national = np.zeros_like(np.asarray(next(iter(leaf_fc.values())), dtype=float))
    for fc in leaf_fc.values():
        national = national + np.asarray(fc, dtype=float)
    return {k: np.asarray(v, dtype=float).copy() for k, v in leaf_fc.items()}, national


def choose_reconciliation(bu_wape: float, td_wape: float) -> str:
    """Pick the reconciliation method with the lower national backtest WAPE."""
    if np.isnan(td_wape):
        return BU
    if np.isnan(bu_wape):
        return TD
    return TD if td_wape < bu_wape else BU


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    series_metrics: pd.DataFrame          # one row per leaf series
    level_metrics: pd.DataFrame           # rows: National + per component + per zone
    bu_national_wape: float
    td_national_wape: float
    chosen_method: str
    folds: list[dict] = field(default_factory=list)


def walk_forward_backtest(
    df: pd.DataFrame, folds: int = 6, horizon: int = 3
) -> BacktestResult:
    """Expanding-window walk-forward backtest.

    For each fold: train = all months before the test window, test = next
    `horizon` months. Leaf forecasts are evaluated per series and aggregated
    per level; bottom-up vs top-down national forecasts are compared on
    national WAPE across all folds.
    """
    keys = leaf_keys(df)
    histories = {k: series_history(df, k[0], k[1]) for k in keys}
    n = len(histories[keys[0]])
    if n <= folds * horizon:
        raise ValueError("Not enough history for the requested folds/horizon")

    # error accumulators
    per_series: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = {k: [] for k in keys}
    per_comp: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    per_zone: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    bu_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    td_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    fold_log: list[dict] = []

    for fold in range(folds):
        test_start = n - (folds - fold) * horizon
        train_end = test_start
        test_idx = slice(test_start, test_start + horizon)

        leaf_fc: dict[tuple[str, str], np.ndarray] = {}
        leaf_actual: dict[tuple[str, str], np.ndarray] = {}
        for k in keys:
            hist = histories[k].values
            leaf_fc[k] = ets_forecast(hist[:train_end], horizon)
            leaf_actual[k] = hist[test_idx]

        # national actual
        nat_actual = np.zeros(horizon)
        for k in keys:
            nat_actual = nat_actual + leaf_actual[k]

        # bottom-up national forecast
        bu_fc = np.zeros(horizon)
        for k in keys:
            bu_fc = bu_fc + leaf_fc[k]

        # top-down national forecast: ETS on aggregated national history
        nat_hist = np.zeros(train_end)
        for k in keys:
            nat_hist = nat_hist + histories[k].values[:train_end]
        td_fc = ets_forecast(nat_hist, horizon)

        # accumulate
        for k in keys:
            per_series[k].append((leaf_actual[k], leaf_fc[k]))
            comp, zone = k
            per_comp.setdefault(comp, []).append((leaf_actual[k], leaf_fc[k]))
            per_zone.setdefault(zone, []).append((leaf_actual[k], leaf_fc[k]))
        bu_pairs.append((nat_actual, bu_fc))
        td_pairs.append((nat_actual, td_fc))
        fold_log.append({
            "fold": fold + 1,
            "train_end_month": str(histories[keys[0]].index[train_end - 1]),
            "test_months": str(histories[keys[0]].index[test_start]),
            "bu_national_wape": wape(nat_actual, bu_fc),
            "td_national_wape": wape(nat_actual, td_fc),
        })

    # per-series metrics
    rows = []
    for k, pairs in per_series.items():
        a = np.concatenate([p[0] for p in pairs])
        f = np.concatenate([p[1] for p in pairs])
        rows.append({
            "level": "series", "name": f"{k[0]} / {k[1]}",
            "wape": wape(a, f), "mape": mape(a, f), "bias": bias(a, f),
        })
    for name, pairs in per_comp.items():
        a = np.concatenate([p[0] for p in pairs])
        f = np.concatenate([p[1] for p in pairs])
        rows.append({"level": "component", "name": name,
                     "wape": wape(a, f), "mape": mape(a, f), "bias": bias(a, f)})
    for name, pairs in per_zone.items():
        a = np.concatenate([p[0] for p in pairs])
        f = np.concatenate([p[1] for p in pairs])
        rows.append({"level": "zone", "name": name,
                     "wape": wape(a, f), "mape": mape(a, f), "bias": bias(a, f)})
    a = np.concatenate([p[0] for p in bu_pairs])
    f = np.concatenate([p[1] for p in bu_pairs])
    rows.append({"level": "national", "name": "All (bottom-up leaves)",
                 "wape": wape(a, f), "mape": mape(a, f), "bias": bias(a, f)})
    series_metrics = pd.DataFrame(rows)

    bu_a = np.concatenate([p[0] for p in bu_pairs])
    bu_f = np.concatenate([p[1] for p in bu_pairs])
    td_a = np.concatenate([p[0] for p in td_pairs])
    td_f = np.concatenate([p[1] for p in td_pairs])
    bu_wape = wape(bu_a, bu_f)
    td_wape = wape(td_a, td_f)
    chosen = choose_reconciliation(bu_wape, td_wape)

    nat_rows = [
        {"level": "national", "name": "All",
         "wape": wape(bu_a, bu_f), "mape": mape(bu_a, bu_f), "bias": bias(bu_a, bu_f)}
    ]
    # level metrics: national row reflects the chosen method
    if chosen == TD:
        nat_rows[0].update({"wape": wape(td_a, td_f), "mape": mape(td_a, td_f),
                            "bias": bias(td_a, td_f)})
    level_metrics = pd.DataFrame(
        [r for r in rows if r["level"] in ("component", "zone")] + nat_rows
    )

    return BacktestResult(
        series_metrics=series_metrics,
        level_metrics=level_metrics,
        bu_national_wape=bu_wape,
        td_national_wape=td_wape,
        chosen_method=chosen,
        folds=fold_log,
    )


# ---------------------------------------------------------------------------
# Full pipeline (forecast + scenario + reconciliation)
# ---------------------------------------------------------------------------

@dataclass
class ForecastResult:
    horizon: int
    leaf_fc: dict[tuple[str, str], np.ndarray]
    national_fc: np.ndarray
    method: str
    bu_national_fc: np.ndarray
    td_national_fc: np.ndarray
    future_index: list[str]


def run_forecast(
    df: pd.DataFrame,
    horizon: int,
    scenario_zones: list[str] | None = None,
    scenario_components: list[str] | None = None,
    scenario_multiplier: float = 1.0,
    method: str | None = None,
) -> ForecastResult:
    """Forecast all leaves, apply scenario, reconcile with chosen method."""
    keys = leaf_keys(df)
    leaf_raw = forecast_all_leaves(df, horizon, keys)
    leaf_fc = apply_scenario(leaf_raw, scenario_zones, scenario_components,
                             float(scenario_multiplier))
    nat_hist = national_history(df)
    td_national = ets_forecast(nat_hist, horizon)
    shares = historical_shares(df)

    if method is None:
        bt = walk_forward_backtest(df)
        method = bt.chosen_method

    bu_national = np.sum([leaf_fc[k] for k in keys], axis=0)
    if method == TD:
        # scale the top-down national forecast by the overall scenario impact
        base_total = np.sum([leaf_raw[k] for k in keys], axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(base_total > 0, bu_national / base_total, 1.0)
        td_national_adj = td_national * scale
        reconciled, national = reconcile(leaf_fc, td_national_adj, shares, TD)
    else:
        reconciled, national = reconcile(leaf_fc, td_national, shares, BU)

    last_period = df["date"].max()
    future_index = [(last_period + i + 1).strftime("%Y-%m") for i in range(horizon)]
    return ForecastResult(
        horizon=horizon,
        leaf_fc=reconciled,
        national_fc=national,
        method=method,
        bu_national_fc=bu_national,
        td_national_fc=td_national_adj if method == TD else td_national,
        future_index=future_index,
    )
