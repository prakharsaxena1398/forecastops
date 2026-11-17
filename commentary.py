"""ForecastOps commentary engine.

Demo mode: deterministic rule-based commentary (trend, seasonality peaks,
zone mix shifts, scenario impact) — zero network required.

Live mode: sends metrics and series statistics to an OpenAI-compatible chat
completions endpoint for a sharper written commentary. Any failure (missing
key, network error, bad response) falls back gracefully to the deterministic
commentary with a UI notice.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import requests

from forecasting import national_history, series_history, leaf_keys

DEMO_PREFIX = "[DEMO MODE — rule-based commentary]"
LIVE_PREFIX = "[LIVE — LLM commentary]"
FALLBACK_NOTICE = (
    "[LIVE mode unavailable (missing OPENAI_API_KEY or request failed); "
    "falling back to deterministic commentary.]"
)


# ---------------------------------------------------------------------------
# Deterministic (demo) commentary
# ---------------------------------------------------------------------------

def _yoy_trend(df: pd.DataFrame) -> tuple[float, str]:
    """Percent change of trailing-12m national demand vs the prior 12 months."""
    nat = national_history(df)
    if len(nat) < 24:
        return 0.0, "insufficient history for YoY trend"
    last = float(nat[-12:].sum())
    prior = float(nat[-24:-12].sum())
    pct = 100.0 * (last - prior) / prior if prior else 0.0
    direction = "growth" if pct >= 0 else "decline"
    return pct, f"National demand shows {direction} of {pct:+.1f}% YoY (trailing 12m vs prior 12m)."


def _seasonality(df: pd.DataFrame) -> str:
    """Identify peak calendar months from the national seasonal profile."""
    nat = df.groupby("date")["units"].sum().sort_index()
    by_month: dict[int, list[float]] = {}
    for period, value in nat.items():
        by_month.setdefault(period.month, []).append(float(value))
    means = {m: float(np.mean(v)) for m, v in by_month.items()}
    overall = float(np.mean(list(means.values())))
    ranked = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
    peak = [m for m, v in ranked if v >= 1.10 * overall]
    trough = [m for m, v in ranked if v <= 0.92 * overall]
    names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
             7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    peak_s = ", ".join(names[m] for m in sorted(peak)) or "none"
    trough_s = ", ".join(names[m] for m in sorted(trough)) or "none"
    return (
        "Seasonality: clear annual pattern with peak demand in "
        f"{peak_s} and trough in {trough_s} (summer April-June cooling load)."
    )


def _zone_mix(df: pd.DataFrame, threshold_pct: float = 1.5) -> str:
    """Zone share of demand last 12 months vs prior 12 months."""
    piv = df.pivot_table(index="date", columns="zone", values="units", aggfunc="sum").sort_index()
    if len(piv) < 24:
        return "Zone mix: insufficient history."
    last = piv.iloc[-12:].sum()
    prior = piv.iloc[-24:-12].sum()
    last_share = last / last.sum() * 100.0
    prior_share = prior / prior.sum() * 100.0
    shifts = (last_share - prior_share).sort_values(key=np.abs, ascending=False)
    parts = []
    for zone, delta in shifts.items():
        if abs(delta) >= threshold_pct:
            parts.append(f"{zone} {delta:+.1f}pp (now {last_share[zone]:.1f}%)")
    if parts:
        return "Zone mix shifts vs last year: " + "; ".join(parts) + "."
    return "Zone mix is broadly stable vs last year (no shift above 1.5pp)."


def _stepup_note(df: pd.DataFrame) -> str:
    """Detect a structural step-up in South demand (fleet-taxi migration story)."""
    south = series_history(df, "Compressor", "South")
    if len(south) < 36:
        return ""
    base = float(south.iloc[12:29].mean())
    after = float(south.iloc[29:].mean())
    if base > 0 and after >= 1.20 * base:
        pct = 100.0 * (after - base) / base
        return (f"Structural note: South shows a step-up of about {pct:.0f}% from "
                "month 30 onward (consistent with the fleet-taxi migration ramp), "
                "which raises the South share going forward.")
    return ""


def _scenario_text(scenario: dict | None) -> str:
    if not scenario or (scenario.get("multiplier", 1.0) == 1.0):
        return "No scenario applied (baseline forecast)."
    scope = ", ".join(filter(None, [
        "zone(s) " + "/".join(scenario.get("zones", []) or []),
        "component(s) " + "/".join(scenario.get("components", []) or []),
    ])) or "all series"
    mult = scenario.get("multiplier", 1.0)
    return (f"Scenario: multiplier x{mult:g} applied to {scope} before "
            f"reconciliation; expect roughly {(mult - 1) * 100:+.0f}% movement in "
            "the affected slice of the national forecast.")


def deterministic_commentary(
    df: pd.DataFrame,
    forecast_result=None,
    scenario: dict | None = None,
    backtest=None,
) -> str:
    """Fully deterministic, rule-based commentary (demo mode)."""
    lines = [DEMO_PREFIX, ""]
    lines.append("1. " + _yoy_trend(df)[1])
    lines.append("2. " + _seasonality(df))
    lines.append("3. " + _zone_mix(df))
    step = _stepup_note(df)
    if step:
        lines.append("4. " + step)
    idx = 5 if step else 4
    lines.append(f"{idx}. " + _scenario_text(scenario))
    if backtest is not None:
        lines.append(
            f"{idx + 1}. Reconciliation: {backtest.chosen_method} chosen "
            f"(national backtest WAPE: bottom-up {backtest.bu_national_wape * 100:.1f}% "
            f"vs top-down {backtest.td_national_wape * 100:.1f}%)."
        )
    if forecast_result is not None:
        total = float(np.sum(forecast_result.national_fc))
        lines.append(
            f"{idx + 2}. Horizon total: {total:,.0f} units over the next "
            f"{forecast_result.horizon} months (reconciled, {forecast_result.method})."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Live (LLM) commentary
# ---------------------------------------------------------------------------

def _build_prompt(df: pd.DataFrame, forecast_result, backtest, scenario) -> str:
    nat = national_history(df)
    yoy = 100.0 * (nat[-12:].sum() - nat[-24:-12].sum()) / max(nat[-24:-12].sum(), 1.0)
    stats = {
        "national_yoy_pct": round(float(yoy), 1),
        "national_trailing_12m_units": round(float(nat[-12:].sum()), 0),
        "horizon_months": forecast_result.horizon,
        "reconciled_horizon_total": round(float(np.sum(forecast_result.national_fc)), 0),
        "chosen_reconciliation": forecast_result.method,
        "backtest_national_wape_bu_pct": round(backtest.bu_national_wape * 100, 1),
        "backtest_national_wape_td_pct": round(backtest.td_national_wape * 100, 1),
        "scenario": scenario or "baseline",
    }
    zone_share = (
        df[df["date"] >= df["date"].max() - 11].groupby("zone")["units"].sum()
        / df[df["date"] >= df["date"].max() - 11]["units"].sum() * 100
    ).round(1).to_dict()
    comp_share = (
        df[df["date"] >= df["date"].max() - 11].groupby("component")["units"].sum()
        / df[df["date"] >= df["date"].max() - 11]["units"].sum() * 100
    ).round(1).to_dict()
    stats["zone_share_pct"] = zone_share
    stats["component_share_pct"] = comp_share
    horizon_by_month = {
        m: round(float(v), 0) for m, v in zip(forecast_result.future_index,
                                              forecast_result.national_fc)
    }
    stats["forecast_national_by_month"] = horizon_by_month
    return (
        "You are a demand planning copilot. Write a concise executive commentary "
        "(4-6 sentences, plain text, no bullet lists) for the national demand "
        "forecast. Cover: trend drivers, seasonality, zone mix shifts, any "
        "scenario impact, and forecast confidence given the backtest WAPE. "
        "Be specific with the numbers provided.\n\nMETRICS JSON:\n"
        + json.dumps(stats, indent=2)
    )


def live_commentary(
    df: pd.DataFrame,
    forecast_result,
    backtest,
    scenario: dict | None = None,
    timeout: float = 30.0,
) -> tuple[str, bool]:
    """Call an OpenAI-compatible chat completions endpoint.

    Returns (commentary_text, used_llm). On any failure returns the
    deterministic commentary with (used_llm=False).
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return FALLBACK_NOTICE + "\n\n" + deterministic_commentary(
            df, forecast_result, scenario, backtest), False
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a precise demand planning analyst."},
            {"role": "user", "content": _build_prompt(df, forecast_result, backtest, scenario)},
        ],
        "temperature": 0.4,
        "max_tokens": 500,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        resp = requests.post(f"{base_url}/chat/completions", json=payload,
                             headers=headers, timeout=timeout)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        if not content:
            raise ValueError("empty LLM response")
        return LIVE_PREFIX + "\n\n" + content, True
    except Exception:
        return FALLBACK_NOTICE + "\n\n" + deterministic_commentary(
            df, forecast_result, scenario, backtest), False
