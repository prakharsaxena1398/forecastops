# ForecastOps — Hierarchical Demand Forecasting Copilot

ForecastOps is the domain-grounded anchor of this portfolio: it models the
exact problem the author works on professionally — aftermarket demand for
automotive AC components (compressors, condensers, evaporators) across sales
zones, where April-June summer peaks drive the service-parts business of a
Tier-1 thermal-systems supplier.

ForecastOps solves two chronic demand-planning problems:

1. **Incoherent hierarchy.** Forecasts made per SKU/component and per region
   independently do not add up to the national total. ForecastOps forecasts
   every leaf (component x zone) and reconciles the hierarchy bottom-up vs
   top-down, choosing the winner by real out-of-sample error from a
   walk-forward backtest — not by taste.
2. **"Why did the forecast move?"** The app reports WAPE / MAPE / bias per
   level (national, component, zone, series) and generates demand commentary
   covering trend, seasonality, zone mix shifts, scenario what-ifs, and
   forecast confidence.

## Demo data

`demo_data/demand_monthly.csv` is **synthetic demo data** (deterministic,
seeded). It is fully replaceable: drop in any CSV with the same schema and
point the app at it via `FORECASTOPS_DATA=/path/to/your.csv`.

Schema (monthly):

| column    | values                                        |
|-----------|-----------------------------------------------|
| date      | monthly dates, `2022-04-01` ... `2026-03-01` (48 months) |
| component | Compressor, Condenser, Evaporator             |
| zone      | North, South, West, East                      |
| units     | demand quantity                               |

The synthetic series include trend, a strong summer seasonality (April-June
peak), noise, per-component scale (Compressor largest), and a deliberate
step-up in South demand from month 30 (fleet-taxi migration story) so the
commentary engine has a structural shift to detect.

## Method

- **Per-series forecast**: Holt-Winters style ETS via
  `statsmodels.ExponentialSmoothing` (additive trend, additive seasonality,
  period 12), fit on full history for the operational forecast and on
  expanding windows for the backtest. Fallbacks: seasonal-naive, then flat.
- **Walk-forward backtest**: 6 expanding-window folds, each forecasting the
  next 3 months; errors accumulated across folds and aggregated per series,
  per component, per zone, and nationally.
- **Reconciliation**:
  - *bottom-up*: national = sum of leaf forecasts.
  - *top-down*: ETS on aggregated national history, disaggregated to leaves
    by trailing-12-month historical shares.
  - The method with the lower **national WAPE on the backtest** is used for
    all operational forecasts.
- **Scenario engine**: a multiplier applied to matching zone / component leaf
  forecasts *before* reconciliation, so the what-if flows through whichever
  reconciliation method won.

## Quickstart

```bash
cd forecastops
pip install -r requirements.txt
python app.py
# open http://127.0.0.1:7862
```

Controls: component multi-select, zone multi-select, horizon slider (3-12
months), scenario zone / component + multiplier. Tabs: Forecast (history +
forecast chart), Backtest metrics (tables by level and by series),
Reconciliation (bottom-up vs top-down national totals, chosen method, WAPE
comparison), Commentary (demand commentary panel).

## Demo vs live commentary mode

- **Demo (default)**: fully deterministic, rule-based commentary, prefix
  `[DEMO MODE — rule-based commentary]`. Works with zero network access.
- **Live**: set environment variables and pick "Live (LLM)" in the UI:
  - `OPENAI_API_KEY` (required)
  - `OPENAI_BASE_URL` (default `https://api.openai.com/v1`, any
    OpenAI-compatible endpoint works)
  - `LLM_MODEL` (default `gpt-4o-mini`)
  Metrics and series statistics are sent to `POST {OPENAI_BASE_URL}/chat/completions`.
  On any failure (missing key, network error, empty response) the app falls
  back to the deterministic commentary and shows a UI notice.

## Tests

```bash
python -m unittest discover -s tests -v
```

10 tests covering: CSV schema and 48-month span, ETS forecast shape and
positivity, hand-computed WAPE, bottom-up = sum of leaves, top-down =
disaggregated national, reconciliation chooser validity, scenario multiplier
math, backtest integrity, commentary content.

## Deploy to a Hugging Face Space

1. Create a new Space (SDK = Gradio), upload this folder
   (`app.py`, `forecasting.py`, `commentary.py`, `requirements.txt`,
   `demo_data/demand_monthly.csv`, `README.md`; tests optional).
2. Space hardware: CPU basic is sufficient (backtest takes ~5 s at startup).
3. If you want live LLM commentary, add `OPENAI_API_KEY` (and optionally
   `OPENAI_BASE_URL`, `LLM_MODEL`) under Settings -> Variables and secrets.
4. The Space entry point is `app.py`; it binds to the port HF provides via
   `PORT` if set — set the Space's app port to 7860-7869 as usual, or run
   with default settings (HF Spaces inject their own port for the Gradio SDK).
5. To use your own data: commit your CSV with the same schema and set the
   Space variable `FORECASTOPS_DATA` to its path (or overwrite
   `demo_data/demand_monthly.csv`).

## Honest limitations

- ETS with additive seasonality is deliberately simple; it will not capture
  multiplicative seasonality, holidays, promotions, or regime changes beyond
  what the trend term absorbs. The South step-up is handled by the trend /
  level update, not by an explicit intervention model.
- Top-down disaggregation uses trailing-12-month shares only; it ignores
  recent share drift and covariance between leaves.
- The reconciliation choice is based on national WAPE alone; per-level or
  multi-objective criteria (e.g., component-level accuracy, bias) are not
  used in the chooser.
- MAPE is unreliable for near-zero actuals (guarded but still unstable).
- The backtest uses only 6 folds x 3 months (18 out-of-sample months); error
  estimates on 48 months of history are noisy.
- The scenario engine is a blunt multiplier; no elasticity, price response,
  or cross-zone substitution is modeled.
- Commentary in demo mode is rule-based and literal; it does not reason
  beyond the statistics it computes.
- No caching of intermediate results across UI runs beyond the startup
  backtest; changing the data file requires an app restart.
