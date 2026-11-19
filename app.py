"""ForecastOps — Hierarchical Demand Forecasting Copilot (Gradio UI).

Run: python app.py  (then open http://127.0.0.1:7862)
"""

from __future__ import annotations

import os
import tempfile

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from commentary import deterministic_commentary, live_commentary
from forecasting import (
    DEFAULT_COMPONENTS,
    DEFAULT_ZONES,
    BacktestResult,
    ForecastResult,
    leaf_keys,
    load_data,
    run_forecast,
    walk_forward_backtest,
)

DATA_PATH = os.environ.get(
    "FORECASTOPS_DATA",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_data", "demand_monthly.csv"),
)


# ---------------------------------------------------------------------------
# Load data and precompute the (slow) backtest once at startup
# ---------------------------------------------------------------------------

DF = load_data(DATA_PATH)
ALL_KEYS = leaf_keys(DF)
BACKTEST: BacktestResult = walk_forward_backtest(DF, folds=6, horizon=3)

SERIES_TABLE = BACKTEST.series_metrics.round(4)
LEVEL_TABLE = BACKTEST.level_metrics.round(4)

RECON_TABLE = pd.DataFrame({
    "method": ["bottom-up (sum of leaves)", "top-down (national split by share)"],
    "national_wape_pct": [round(BACKTEST.bu_national_wape * 100, 2),
                          round(BACKTEST.td_national_wape * 100, 2)],
})
RECON_TABLE["chosen"] = ["YES" if BACKTEST.chosen_method == "bottom-up" else "",
                         "YES" if BACKTEST.chosen_method == "top-down" else ""]

COMPARISON_NOTE = (
    f"**Chosen reconciliation: {BACKTEST.chosen_method}**  \n"
    f"Bottom-up national WAPE: **{BACKTEST.bu_national_wape * 100:.2f}%**  |  "
    f"Top-down national WAPE: **{BACKTEST.td_national_wape * 100:.2f}%**  \n"
    "The winner is the method with the lower national WAPE across 6 "
    "walk-forward folds (3-month horizon, expanding train). All forecasts in "
    "the Forecast tab are reconciled with the chosen method."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _selection_history(df: pd.DataFrame, components: list[str], zones: list[str]) -> pd.Series:
    sub = df[df["component"].isin(components) & df["zone"].isin(zones)]
    return sub.groupby("date")["units"].sum().astype(float).sort_index()


def _make_chart(df: pd.DataFrame, res: ForecastResult, components: list[str],
                zones: list[str], title: str) -> str:
    hist = _selection_history(df, components, zones)
    fc_keys = [k for k in ALL_KEYS if k[0] in components and k[1] in zones]
    fc = np.zeros(res.horizon)
    for k in fc_keys:
        fc = fc + res.leaf_fc[k]
    hist_x = [str(p) for p in hist.index]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(hist_x, hist.values, color="#1f77b4", lw=1.8, label="History")
    ax.plot(res.future_index, fc, color="#d62728", lw=2.0, ls="--",
            marker="o", ms=4, label=f"Forecast ({res.method} reconciled)")
    ax.axvline(hist_x[-1], color="gray", lw=0.8, ls=":")
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Units / month")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    path = os.path.join(tempfile.gettempdir(), f"forecastops_{os.getpid()}_chart.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _reconciliation_table(res: ForecastResult, components: list[str], zones: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "month": res.future_index,
        "national_forecast_reconciled": np.round(res.national_fc, 0).astype(int),
        "national_bottom_up": np.round(res.bu_national_fc, 0).astype(int),
        "national_top_down": np.round(res.td_national_fc, 0).astype(int),
    })


def run_app(components, zones, horizon, scen_zone, scen_component, scen_mult, mode):
    components = list(components) or DEFAULT_COMPONENTS
    zones = list(zones) or DEFAULT_ZONES
    horizon = int(horizon)
    scen_zones = [scen_zone] if scen_zone else []
    scen_components = [scen_component] if scen_component else []
    scen_mult = float(scen_mult if scen_mult not in (None, "") else 1.0)
    scenario = {"zones": scen_zones, "components": scen_components,
                "multiplier": scen_mult if (scen_zones or scen_components) else 1.0}

    res = run_forecast(
        DF, horizon,
        scenario_zones=scenario["zones"] if (scen_zones or scen_components) else None,
        scenario_components=scenario["components"] if (scen_zones or scen_components) else None,
        scenario_multiplier=scen_mult,
        method=BACKTEST.chosen_method,
    )

    title = f"{'+'.join(components)} x {'+'.join(zones)} - history and {horizon}-month forecast"
    chart_path = _make_chart(DF, res, components, zones, title)
    recon_df = _reconciliation_table(res, components, zones)

    if mode == "Live (LLM)":
        commentary, used_llm = live_commentary(DF, res, BACKTEST, scenario)
        notice = "" if used_llm else commentary.split("\n\n")[0]
    else:
        commentary = deterministic_commentary(DF, res, scenario, BACKTEST)
        notice = ""

    def _in_series_selection(name: str) -> bool:
        parts = name.split(" / ")
        return len(parts) == 2 and parts[0] in components and parts[1] in zones

    series_tbl = SERIES_TABLE[SERIES_TABLE["name"].map(_in_series_selection)]
    level_tbl = LEVEL_TABLE[(LEVEL_TABLE["level"] != "zone") | (LEVEL_TABLE["name"].isin(zones))]

    return (
        chart_path,
        level_tbl.round(3),
        series_tbl.round(3),
        recon_df,
        COMPARISON_NOTE,
        commentary,
        notice,
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(title="ForecastOps - Hierarchical Demand Forecasting Copilot") as demo:
        gr.Markdown(
            "# ForecastOps - Hierarchical Demand Forecasting Copilot\n"
            "Coherent component x zone -> national forecasts (ETS), reconciled "
            "bottom-up vs top-down by walk-forward backtest error, with WAPE / "
            "MAPE / bias per level and demand commentary. Demo data: synthetic "
            "monthly demand, 48 months (see README)."
        )
        with gr.Row():
            with gr.Column(scale=1):
                comp_sel = gr.CheckboxGroup(choices=DEFAULT_COMPONENTS,
                                            value=DEFAULT_COMPONENTS, label="Components")
                zone_sel = gr.CheckboxGroup(choices=DEFAULT_ZONES,
                                            value=DEFAULT_ZONES, label="Zones")
                horizon = gr.Slider(minimum=3, maximum=12, step=1, value=6,
                                    label="Forecast horizon (months)")
                gr.Markdown("### Scenario what-if")
                scen_zone = gr.Dropdown(choices=[""] + DEFAULT_ZONES, value="",
                                        label="Scenario zone (empty = none)")
                scen_comp = gr.Dropdown(choices=[""] + DEFAULT_COMPONENTS, value="",
                                        label="Scenario component (empty = none)")
                scen_mult = gr.Number(value=1.10, label="Scenario multiplier", precision=3)
                mode = gr.Radio(choices=["Demo (rule-based)", "Live (LLM)"],
                                value="Demo (rule-based)", label="Commentary mode")
                run_btn = gr.Button("Run forecast", variant="primary")
        with gr.Tabs():
            with gr.Tab("Forecast"):
                chart = gr.Image(label="History + forecast", type="filepath")
            with gr.Tab("Backtest metrics"):
                gr.Markdown(
                    "Walk-forward backtest: 6 expanding-window folds, 3-month "
                    "forecast horizon. WAPE = sum|a-f|/sum|a|; bias = signed "
                    "mean error percent."
                )
                level_df = gr.DataFrame(label="Metrics by level", interactive=False)
                series_df = gr.DataFrame(label="Metrics by series", interactive=False)
            with gr.Tab("Reconciliation"):
                recon_note = gr.Markdown(COMPARISON_NOTE)
                recon_df = gr.DataFrame(label="National totals by method", interactive=False)
            with gr.Tab("Commentary"):
                notice = gr.Markdown("")
                commentary_tb = gr.Textbox(label="Demand commentary", lines=16)

        run_btn.click(
            run_app,
            inputs=[comp_sel, zone_sel, horizon, scen_zone, scen_comp, scen_mult, mode],
            outputs=[chart, level_df, series_df, recon_df, recon_note, commentary_tb, notice],
        )
    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=int(os.environ.get("PORT", 7862)),
               quiet=True, prevent_thread_lock=True, show_error=True)
    print("ForecastOps running at http://127.0.0.1:7862")
    try:
        import time
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
