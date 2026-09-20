# Retail Intelligence & Demand Forecasting Platform

An end-to-end analytics platform built on a simulated nationwide retail chain
— 8 stores, 4 Australian cities, 40 SKUs across 5 categories, two years of daily
transactions. It walks the full data science stack in one repo: **advanced
SQL → statistics → EDA → feature engineering → ML forecasting → business
insights → interactive dashboard → containerized deployment.**

## The point of it

A lot of portfolio projects stop at "trained a model, got 85% accuracy."
This one is built around a question retailers actually pay analysts to
answer: *where should we stock more inventory, and what's it worth in
dollars?* Every layer feeds that answer — the SQL views compute the KPIs,
the statistics module checks whether a promotion campaign actually moved the
needle, the model forecasts demand, and the final output is a concrete
profit number tied to a real stocking decision.

## Layout

```
retail-intelligence/
├── data/                              # generated CSVs, engineered feature table, EDA exports
├── sql/
│   ├── 01_define_schema.sql                # tables, PK/FK, indexing strategy
│   ├── 02_reporting_views.sql              # window functions, CTEs (SQLite)
│   ├── 03_postgres_stored_procedures.sql   # native partitioning + stored procedures (Postgres)
│   └── 04_requested_business_queries.sql   # the exact requested query set
├── src/
│   ├── synthesize_dataset.py     # synthetic multi-table dataset generator
│   ├── build_database.py         # CSV -> SQLite loader + view builder
│   ├── engineer_features.py      # lag/rolling/calendar/weather features
│   ├── run_statistical_tests.py  # hypothesis testing, A/B test, ANOVA, CIs
│   ├── exploratory_analysis.py   # seasonality, segmentation, outliers, heatmaps
│   └── train_forecast_models.py  # LightGBM + XGBoost forecasting, SHAP, insight gen
├── api/inference_service.py      # FastAPI inference service
├── dashboard/streamlit_app.py    # Streamlit interactive dashboard (calls the live API)
├── Dockerfile.inference / Dockerfile.streamlit / docker-compose.yml
└── requirements.txt
```

## Running it

```bash
pip install -r requirements.txt

python src/synthesize_dataset.py       # ~215K synthetic transactions
python src/build_database.py           # builds retail_analytics.db (SQLite) + views
python src/engineer_features.py        # builds data/feature_matrix.parquet
python src/run_statistical_tests.py    # prints hypothesis-test / ANOVA / CI report
python src/exploratory_analysis.py     # seasonality, segmentation, outlier report
python src/train_forecast_models.py    # trains + saves models, SHAP, business insight

uvicorn api.inference_service:app --reload --port 8000   # forecasting API
streamlit run dashboard/streamlit_app.py                  # dashboard
```

Or with Docker:

```bash
docker compose up --build
# API:       http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

## 1. Advanced SQL

- **Window functions** — rolling 7/30-day revenue, `RANK()` for top products
  *within each city*, `LAG()` for month-over-month growth
- **CTEs** — multi-step cohort analysis (new vs. returning customer revenue),
  stockout detection joining inventory against historical demand
- **Views** — `vw_customer_rfm`, `vw_rolling_30d_revenue`, `vw_monthly_growth`,
  `vw_product_revenue_rank_by_city`, `vw_repeat_purchase_rate`
- **Stored procedures** — SQLite has no PL/pgSQL, so
  `sql/03_postgres_stored_procedures.sql` contains genuine Postgres stored
  procedures (`refresh_daily_store_kpis()`, `get_customer_ltv()`,
  `estimate_inventory_uplift_profit()`) — runnable against any Postgres
  instance (Supabase/Neon/RDS free tier)
- **Indexing** — composite indexes documented with rationale in
  `01_define_schema.sql` (e.g. `(date, store_id)` as a covering index for
  date-range queries)
- **Partitioning** — native `PARTITION BY RANGE (date)` quarterly
  partitioning specified for Postgres, since SQLite doesn't support it
  natively

Run the exact requested query set directly:

```bash
sqlite3 retail_analytics.db < sql/04_requested_business_queries.sql
```

## 2. Statistics

`src/run_statistical_tests.py` answers, with real output:

- **"Did discount campaign A significantly increase sales?"** — Welch's
  t-test, treatment (promo-active days) vs. control (same categories, no
  promo): **+22.9% lift, p < 0.001, statistically significant**, with a 95%
  CI on the mean difference
- **ANOVA** — daily revenue differs significantly across cities (F=643, p≈0)
- **Correlation** — revenue vs. discount, revenue vs. temperature (Pearson r
  + p-values)
- **Confidence intervals** — repeat purchase rate, average transaction value

## 3. EDA

`src/exploratory_analysis.py` produces: a monthly seasonality chart, a
day-of-week × month revenue heatmap, a regional trend breakdown, RFM-based
customer segmentation (Champions / Loyal / Potential / At Risk / Lost), a
missing-value audit, and IQR-based outlier detection — all rendered in the
dashboard's EDA tab.

## 4. Feature Engineering

Built at the store × category × day grain: lag features (1/7/14/28-day),
rolling mean/std (7-day, 28-day), calendar features (day-of-week, holiday
flags, cyclical day-of-year encoding), a 14-day pre-Diwali "festive window"
flag, weather features with a category-specific interaction term (hot-day ×
weather-sensitive category), and customer RFM features.

## 5. ML Models

LightGBM and XGBoost regressors forecasting daily store-category revenue,
benchmarked against a naive "same as last week" baseline:

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| LightGBM | 16,629 | 43,186 | 23.2% |
| XGBoost | 16,198 | 43,091 | 21.3% |
| Naive (lag-7) | 26,792 | 76,954 | 26.1% |

Both models beat the naive baseline by roughly 35–40% on MAE. SHAP confirms
`revenue_roll_mean_28` and `revenue_roll_mean_7` dominate — sensible, since
recent demand level is the strongest predictor of near-term demand, with
festive-window and day-of-week as the next-strongest signals.

Prediction intervals (empirical residual quantiles) are served alongside
point forecasts via the API, not just point estimates — important for
inventory planning, where the uncertainty band matters as much as the mean.
This interval is computed once, globally across all store-category series —
it isn't calibrated per-segment, so it can look wide for low-volume
categories. That's a known limitation, not a bug: a stronger version would
fit segment-specific quantiles instead of a single global one.

## 6. Business Insights

Auto-generated from the model plus festive-window data, e.g.:

> Increasing inventory of 'Electronics' in Melbourne by 15% ahead of the
> festive season is projected to add $576,637 in incremental profit over a
> 30-day window (avg daily demand: 60 units, blended margin assumption:
> 30%).

This is computed programmatically in `train_forecast_models.py` (and
reproducible as a genuine Postgres function via
`estimate_inventory_uplift_profit()`), not hand-typed — swap in real margin
assumptions and it re-derives the number.

## 7. Dashboard

Streamlit app with 6 tabs: KPIs, Forecast (model comparison + SHAP + a live
demo that calls the FastAPI service in real time, plus a trend explorer),
Store Performance, Product Performance, Customer Segments (RFM), and EDA —
all filterable by date range, city, and category.

## 8. Deployment

- **FastAPI** — `/predict` (single forecast with a 90% prediction interval),
  `/insights/latest`, `/model/comparison`, `/health`. Built from
  `Dockerfile.inference`.
- **Docker** — separate images for the API and the dashboard, orchestrated
  via `docker-compose.yml`.
- **Streamlit** — `dashboard/streamlit_app.py` makes real HTTP calls to the
  deployed API for its "Live Forecast" demo.
- Deploy `Dockerfile.inference` (API) and `Dockerfile.streamlit` (dashboard)
  separately to your platform of choice (Render, Railway, Fly.io, Streamlit
  Community Cloud, etc.) — both auto-redeploy on push if you wire up your
  host's CI hook.

## Troubleshooting

### macOS: `OSError: ... Library not loaded: @rpath/libomp.dylib` when running `train_forecast_models.py`

LightGBM depends on OpenMP (`libomp`), which isn't bundled with the pip
wheel on macOS — it has to come from Homebrew. This error means either
libomp isn't installed, or (on Apple Silicon Macs with an older Homebrew
setup) there's an **architecture mismatch** between your Python and your
Homebrew installation.

**Step 1 — check your setup:**
```bash
uname -m                                    # should print arm64 on Apple Silicon
python3 -c "import platform; print(platform.machine())"   # should also print arm64
which brew
file $(which brew)
```

If `brew` lives at `/usr/local/bin/brew` instead of `/opt/homebrew/bin/brew`,
you're running an Intel/Rosetta Homebrew on an Apple Silicon Mac — that's
the root cause.

**Step 2 — install a native ARM Homebrew:**
```bash
arch -arm64 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
eval "$(/opt/homebrew/bin/brew shellenv)"
```

**Step 3 — install libomp via the native brew:**
```bash
/opt/homebrew/bin/brew install libomp
```

**Step 4 — retry**
```bash
python src/train_forecast_models.py
```

If `uname -m` and the Python check both already say `arm64` and `brew` is
already at `/opt/homebrew/bin/brew`, you likely just need:
```bash
brew install libomp
```
with no architecture juggling required — the mismatch scenario above is
specific to machines where Homebrew was originally installed before
switching to (or under emulation on) Apple Silicon.

### Linux (deployed API): `OSError: libgomp.so.1: cannot open shared object file`

Same underlying cause as the macOS issue above (LightGBM needs an OpenMP
runtime), but the Linux fix is different. The `python:3.11-slim` base image
used in `Dockerfile.inference` doesn't include `libgomp` by default, so it
has to be installed via `apt-get` before `pip install -r requirements.txt`
runs. `Dockerfile.inference` already includes this fix:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
```

If you fork this repo and hit this error on your own deploy, make sure that
line is present in your `Dockerfile.inference` before the `pip install`
step.

### Deploying `Dockerfile.inference` but the host runs it as a plain Python app instead

Some hosts (e.g. Render) auto-detect the runtime from repo contents and can
default to "Python 3" if they see `requirements.txt`, ignoring
`Dockerfile.inference` entirely. When creating the service, explicitly set
the language/runtime to **Docker** and point the Dockerfile path at
`Dockerfile.inference` (build context: `.`) before deploying.

## Honest scope notes

This uses synthetic data (documented generation logic in
`synthesize_dataset.py`, seeded for reproducibility) rather than a
scraped/licensed real dataset — it was designed to be realistic enough that
every technique here transfers directly to real retail data, while keeping
the whole pipeline fast enough to run and iterate on end-to-end.

## Author

**Abdullah Al Yubraj**
📧 yuvrajabdullah99@gmail.com
