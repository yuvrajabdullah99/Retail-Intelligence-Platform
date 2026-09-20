"""
EDA: seasonality, regional trends, customer segmentation, missing values,
outlier detection, sales heatmap data. Saves summary CSVs + a couple of
PNG charts to data/eda_outputs/ (dashboard reads these for the "EDA" tab).
"""
import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "retail_analytics.db"
OUT = ROOT / "data" / "eda_outputs"
OUT.mkdir(parents=True, exist_ok=True)

def main():
    conn = sqlite3.connect(DB)

    # ---- Missing values audit ----
    print("=== Missing Values ===")
    for table in ["customers", "weather", "transactions"]:
        df = pd.read_sql(f"SELECT * FROM {table}", conn)
        missing = df.isna().sum()
        missing = missing[missing > 0]
        if len(missing):
            print(f"\n{table}:")
            print((missing / len(df) * 100).round(2).astype(str) + "%")
        else:
            print(f"\n{table}: no missing values")

    # ---- Outlier detection (IQR method on transaction units_sold) ----
    print("\n=== Outlier Detection (IQR method, units_sold) ===")
    units = pd.read_sql("SELECT units_sold FROM transactions", conn)["units_sold"]
    q1, q3 = units.quantile([0.25, 0.75])
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    n_outliers = (units > upper).sum()
    print(f"Q1={q1}, Q3={q3}, IQR={iqr}, upper fence={upper:.1f}")
    print(f"Outlier transactions (units_sold > fence): {n_outliers} ({n_outliers/len(units)*100:.3f}%)")

    # ---- Seasonality: monthly revenue trend ----
    monthly = pd.read_sql("""
        SELECT strftime('%Y-%m', date) AS ym, SUM(revenue) AS revenue
        FROM transactions GROUP BY ym ORDER BY ym
    """, conn)
    monthly.to_csv(OUT / "monthly_revenue_trend.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(monthly["ym"], monthly["revenue"], marker="o")
    ax.set_title("Monthly Revenue Trend (Seasonality)")
    ax.set_xticklabels(monthly["ym"], rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(OUT / "monthly_revenue_trend.png", dpi=110)
    plt.close(fig)

    # ---- Regional trends ----
    regional = pd.read_sql("""
        SELECT s.city, strftime('%Y-%m', t.date) AS ym, SUM(t.revenue) AS revenue
        FROM transactions t JOIN stores s ON s.store_id = t.store_id
        GROUP BY s.city, ym ORDER BY ym
    """, conn)
    regional.to_csv(OUT / "regional_revenue_breakdown.csv", index=False)

    # ---- Sales heatmap: day-of-week x month ----
    heat = pd.read_sql("SELECT date, revenue FROM transactions", conn, parse_dates=["date"])
    daily = heat.groupby("date")["revenue"].sum().reset_index()
    daily["dow"] = daily["date"].dt.day_name()
    daily["month"] = daily["date"].dt.month_name()
    pivot = daily.pivot_table(index="dow", columns="month", values="revenue", aggfunc="mean")
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = pivot.reindex(dow_order)
    pivot.to_csv(OUT / "weekday_month_heatmap.csv")

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_title("Avg Daily Revenue Heatmap: Day-of-Week x Month")
    fig.colorbar(im, label="Avg Revenue")
    plt.tight_layout()
    fig.savefig(OUT / "weekday_month_heatmap.png", dpi=110)
    plt.close(fig)

    # ---- Customer segmentation (simple RFM-quantile based, feeds dashboard) ----
    rfm = pd.read_csv(ROOT / "data" / "customer_rfm_metrics.csv")
    rfm["R_score"] = pd.qcut(rfm["recency_days"], 4, labels=[4, 3, 2, 1]).astype(int)
    rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["M_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["RFM_score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

    def label_segment(score):
        if score >= 10: return "Champions"
        if score >= 8: return "Loyal"
        if score >= 6: return "Potential"
        if score >= 4: return "At Risk"
        return "Lost"
    rfm["rfm_segment"] = rfm["RFM_score"].apply(label_segment)
    rfm.to_csv(ROOT / "data" / "customer_rfm_segments.csv", index=False)
    print("\n=== Customer Segmentation (RFM) ===")
    print(rfm["rfm_segment"].value_counts())

    conn.close()
    print(f"\nEDA outputs saved to {OUT}")

if __name__ == "__main__":
    main()
