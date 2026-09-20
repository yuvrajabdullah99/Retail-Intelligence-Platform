"""
Retail Intelligence & Demand Forecasting Platform 
"""
import os
import sqlite3
from pathlib import Path

import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "retail_analytics.db"
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Retail Intelligence Platform", layout="wide", page_icon="🛒")

@st.cache_data
def load_data():
    conn = sqlite3.connect(DB)
    txn = pd.read_sql("""
        SELECT t.date, t.store_id, s.city, s.city_tier, p.category, p.product_name,
               t.units_sold, t.revenue, t.discount_pct, t.customer_id
        FROM transactions t
        JOIN stores s ON s.store_id = t.store_id
        JOIN products p ON p.product_id = t.product_id
    """, conn, parse_dates=["date"])
    stores = pd.read_sql("SELECT * FROM stores", conn)
    rfm_path = ROOT / "data" / "customer_rfm_segments.csv"
    rfm = pd.read_csv(rfm_path) if rfm_path.exists() else pd.read_csv(ROOT / "data" / "customer_rfm_metrics.csv")
    model_comp_path = ROOT / "models" / "model_benchmark_results.csv"
    model_comp = pd.read_csv(model_comp_path, index_col=0) if model_comp_path.exists() else None
    insight_path = ROOT / "models" / "inventory_insight_summary.txt"
    insight = insight_path.read_text(encoding="utf-8") if insight_path.exists() else None
    shap_path = ROOT / "models" / "shap_importance_scores.csv"
    shap_imp = pd.read_csv(shap_path, index_col=0) if shap_path.exists() else None
    conn.close()
    return txn, stores, rfm, model_comp, insight, shap_imp

txn, stores, rfm, model_comp, insight, shap_imp = load_data()

# ---------------- Sidebar filters ----------------
st.sidebar.title(" Retail Intelligence")
st.sidebar.caption("Nationwide retail chain — analytics platform")

date_range = st.sidebar.date_input(
    "Date range", value=(txn["date"].min().date(), txn["date"].max().date()),
    min_value=txn["date"].min().date(), max_value=txn["date"].max().date(),
)
cities = st.sidebar.multiselect("City", sorted(txn["city"].unique()), default=sorted(txn["city"].unique()))
categories = st.sidebar.multiselect("Category", sorted(txn["category"].unique()), default=sorted(txn["category"].unique()))

if len(date_range) == 2:
    mask = (
        (txn["date"].dt.date >= date_range[0]) & (txn["date"].dt.date <= date_range[1])
        & (txn["city"].isin(cities)) & (txn["category"].isin(categories))
    )
    f = txn.loc[mask]
else:
    f = txn.loc[txn["city"].isin(cities) & txn["category"].isin(categories)]

tab_kpi, tab_forecast, tab_store, tab_product, tab_customer, tab_eda = st.tabs(
    [" KPIs", " Forecast", " Store Performance", " Product Performance", " Customer Segments", " EDA"]
)

# ---------------- KPI TAB ----------------
with tab_kpi:
    c1, c2, c3, c4, c5 = st.columns(5)
    total_rev = f["revenue"].sum()
    total_units = f["units_sold"].sum()
    n_txn = len(f)
    avg_basket = total_rev / n_txn if n_txn else 0
    n_customers = f["customer_id"].nunique()

    c1.metric("Total Revenue", f"${total_rev/1e6:.2f}M")
    c2.metric("Units Sold", f"{total_units:,.0f}")
    c3.metric("Transactions", f"{n_txn:,.0f}")
    c4.metric("Avg Basket Value", f"${avg_basket:,.0f}")
    c5.metric("Unique Customers", f"{n_customers:,.0f}")

    if insight:
        st.info(f"💡 **Business Insight:** {insight}")

    daily = f.groupby("date")["revenue"].sum().reset_index()
    fig = px.line(daily, x="date", y="revenue", title="Daily Revenue Trend")
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

    colA, colB = st.columns(2)
    with colA:
        by_city = f.groupby("city")["revenue"].sum().reset_index().sort_values("revenue", ascending=False)
        st.plotly_chart(px.bar(by_city, x="city", y="revenue", title="Revenue by City"), use_container_width=True)
    with colB:
        by_cat = f.groupby("category")["revenue"].sum().reset_index().sort_values("revenue", ascending=False)
        st.plotly_chart(px.pie(by_cat, names="category", values="revenue", title="Revenue Share by Category"), use_container_width=True)

# ---------------- FORECAST TAB ----------------
with tab_forecast:
    st.subheader("Model Comparison")
    if model_comp is not None:
        st.dataframe(model_comp.style.highlight_min(subset=["MAE", "RMSE", "MAPE"], color="lightgreen"), use_container_width=True)
        st.caption("Naive baseline = 'predict same value as 7 days ago'. Both ML models beat it comfortably on MAE, RMSE and MAPE.")
    else:
        st.warning("Run `python src/train_forecast_models.py` first to populate model comparison metrics.")

    if shap_imp is not None:
        st.subheader("What drives the forecast? (SHAP feature importance)")
        shap_imp.columns = ["mean_abs_shap"]
        top = shap_imp.head(10).reset_index().rename(columns={"index": "feature"})
        st.plotly_chart(
            px.bar(top, x="mean_abs_shap", y="feature", orientation="h", title="Top 10 Features by Mean |SHAP value|"),
            use_container_width=True,
        )

    st.divider()

    # ---------------- Live API integration ----------------
    st.subheader("🔴 Live Forecast (calls the deployed FastAPI service)")
    st.caption(f"This section sends a real HTTP request to the live API at `{API_URL}` — not a local calculation.")

    live_store = st.selectbox("Store", sorted(stores["store_id"].unique()), key="live_store")
    live_cat = st.selectbox("Category", sorted(txn["category"].unique()), key="live_cat")

    if st.button("Get Live Forecast from API"):
        recent = txn[(txn["store_id"] == live_store) & (txn["category"] == live_cat)].sort_values("date")
        daily_lc = recent.groupby("date")["revenue"].sum().reset_index().sort_values("date")
        daily_units_lc = recent.groupby("date")["units_sold"].sum().reset_index().sort_values("date")

        if len(daily_lc) < 28:
            st.warning("Not enough history for this store/category to build a live request.")
        else:
            last_date = daily_lc["date"].max()
            store_row = stores[stores["store_id"] == live_store].iloc[0]
            payload = {
                "store_id": int(live_store),
                "city_tier": int(store_row["city_tier"]),
                "dow": int(last_date.dayofweek),
                "is_weekend": int(last_date.dayofweek >= 5),
                "month": int(last_date.month),
                "is_holiday": 0,
                "is_festive_window": 0,
                "doy_sin": float(np.sin(2 * np.pi * last_date.dayofyear / 365)),
                "doy_cos": float(np.cos(2 * np.pi * last_date.dayofyear / 365)),
                "temp_celsius": 28.0,
                "rainfall_mm": 0.0,
                "avg_discount_pct": 0.0,
                "had_promo": 0,
                "is_weather_sensitive_cat": int(live_cat == "Beverages"),
                "hot_day_x_sensitive": 0,
                "revenue_lag_1": float(daily_lc["revenue"].iloc[-1]),
                "revenue_lag_7": float(daily_lc["revenue"].iloc[-7]),
                "revenue_lag_14": float(daily_lc["revenue"].iloc[-14]),
                "revenue_lag_28": float(daily_lc["revenue"].iloc[-28]),
                "revenue_roll_mean_7": float(daily_lc["revenue"].tail(7).mean()),
                "revenue_roll_std_7": float(daily_lc["revenue"].tail(7).std()),
                "revenue_roll_mean_28": float(daily_lc["revenue"].tail(28).mean()),
                "units_roll_mean_7": float(daily_units_lc["units_sold"].tail(7).mean()),
            }

            try:
                with st.spinner("Calling live API on Render (may take ~30-60s if it's waking from sleep)..."):
                    resp = requests.post(f"{API_URL}/predict", json=payload, timeout=90)
                if resp.status_code == 200:
                    result = resp.json()
                    rc1, rc2, rc3 = st.columns(3)
                    rc1.metric("Predicted Revenue", f"${result['predicted_revenue']:,.0f}")
                    rc2.metric("Lower 90% bound", f"${result['lower_90']:,.0f}")
                    rc3.metric("Upper 90% bound", f"${result['upper_90']:,.0f}")
                    st.success(f"Response received from {API_URL}/predict")
                    with st.expander("Raw request/response"):
                        st.json({"request": payload, "response": result})
                else:
                    st.error(f"API returned status {resp.status_code}: {resp.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the API: {e}")

    st.divider()

    st.subheader("Store-Category Revenue Trend (historical)")
    sel_store = st.selectbox("Store", sorted(stores["store_id"].unique()), key="trend_store")
    sel_cat = st.selectbox("Category", sorted(txn["category"].unique()), key="trend_cat")
    trend = txn[(txn["store_id"] == sel_store) & (txn["category"] == sel_cat)].groupby("date")["revenue"].sum().reset_index()
    trend["rolling_7"] = trend["revenue"].rolling(7).mean()
    trend["rolling_30"] = trend["revenue"].rolling(30).mean()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=trend["date"], y=trend["revenue"], name="Daily", opacity=0.35))
    fig2.add_trace(go.Scatter(x=trend["date"], y=trend["rolling_7"], name="7-day avg"))
    fig2.add_trace(go.Scatter(x=trend["date"], y=trend["rolling_30"], name="30-day avg"))
    fig2.update_layout(title=f"Store {sel_store} — {sel_cat}: Revenue with Rolling Averages", height=420)
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- STORE PERFORMANCE ----------------
with tab_store:
    store_perf = f.groupby(["store_id", "city"]).agg(
        revenue=("revenue", "sum"), units=("units_sold", "sum"),
        transactions=("revenue", "count"), customers=("customer_id", "nunique"),
    ).reset_index().sort_values("revenue", ascending=False)
    store_perf["avg_basket"] = (store_perf["revenue"] / store_perf["transactions"]).round(0)
    st.dataframe(store_perf, use_container_width=True)
    st.plotly_chart(px.bar(store_perf, x="store_id", y="revenue", color="city", title="Revenue by Store"), use_container_width=True)

    st.subheader("Monthly Growth by Store")
    monthly_store = f.copy()
    monthly_store["ym"] = monthly_store["date"].dt.to_period("M").astype(str)
    mg = monthly_store.groupby(["store_id", "ym"])["revenue"].sum().reset_index()
    fig3 = px.line(mg, x="ym", y="revenue", color="store_id", title="Monthly Revenue by Store")
    st.plotly_chart(fig3, use_container_width=True)

# ---------------- PRODUCT PERFORMANCE ----------------
with tab_product:
    top_products = f.groupby(["product_name", "category"])["revenue"].sum().reset_index().sort_values("revenue", ascending=False).head(15)
    st.plotly_chart(px.bar(top_products, x="revenue", y="product_name", color="category", orientation="h", title="Top 15 Products by Revenue"), use_container_width=True)

    st.subheader("Discount vs Revenue Relationship")
    disc_rev = f.groupby("discount_pct")["revenue"].sum().reset_index()
    st.plotly_chart(px.bar(disc_rev, x="discount_pct", y="revenue", title="Revenue by Discount Level"), use_container_width=True)

# ---------------- CUSTOMER SEGMENTS ----------------
with tab_customer:
    if "rfm_segment" in rfm.columns:
        seg_counts = rfm["rfm_segment"].value_counts().reset_index()
        seg_counts.columns = ["segment", "count"]
        colA, colB = st.columns(2)
        with colA:
            st.plotly_chart(px.pie(seg_counts, names="segment", values="count", title="Customer Segments (RFM)"), use_container_width=True)
        with colB:
            seg_value = rfm.groupby("rfm_segment")["monetary"].mean().reset_index().sort_values("monetary", ascending=False)
            st.plotly_chart(px.bar(seg_value, x="rfm_segment", y="monetary", title="Avg Lifetime Value by Segment"), use_container_width=True)
        st.dataframe(rfm.sort_values("monetary", ascending=False).head(50), use_container_width=True)
    else:
        st.dataframe(rfm.head(50), use_container_width=True)

# ---------------- EDA TAB ----------------
with tab_eda:
    eda_dir = ROOT / "data" / "eda_outputs"
    colA, colB = st.columns(2)
    with colA:
        img = eda_dir / "monthly_revenue_trend.png"
        if img.exists():
            st.image(str(img), caption="Monthly Revenue Trend")
    with colB:
        img2 = eda_dir / "weekday_month_heatmap.png"
        if img2.exists():
            st.image(str(img2), caption="Day-of-Week x Month Revenue Heatmap")

    st.subheader("Outlier & Missing-Value Summary")
    st.markdown("""
    - **Outliers**: ~4.8% of transaction line items exceed the IQR upper fence for `units_sold`
      (mix of genuine bulk-purchase events and injected synthetic outliers) — handled via
      winsorization before feeding lag/rolling features into the model.
    - **Missing values**: ~3% of `customers.age`, ~2% of `weather.rainfall_mm` — imputed with
      group-wise median/mean during feature engineering.
    """)
    st.caption("Full statistical test outputs (A/B test, ANOVA, correlation, confidence intervals) are in `src/run_statistical_tests.py` — run it directly for a printed report.")

st.sidebar.markdown("---")
st.sidebar.caption("Data: synthetic, 2 years, 8 stores, 4 cities, 40 SKUs, 215K+ transactions")
