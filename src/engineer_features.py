"""
Feature engineering: builds a store x category x day modeling table
with lag features, rolling stats, calendar/weather features, and
customer RFM features aggregated to the store level.

"""
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "retail_analytics.db"

def main():
    conn = sqlite3.connect(DB)

    # Base grain: store x category x day revenue/units
    df = pd.read_sql("""
        SELECT
            t.date, t.store_id, s.city, s.city_tier, p.category,
            SUM(t.revenue) AS revenue,
            SUM(t.units_sold) AS units,
            AVG(t.discount_pct) AS avg_discount_pct
        FROM transactions t
        JOIN stores s ON s.store_id = t.store_id
        JOIN products p ON p.product_id = t.product_id
        GROUP BY t.date, t.store_id, s.city, s.city_tier, p.category
    """, conn, parse_dates=["date"])

    weather = pd.read_sql("SELECT * FROM weather", conn, parse_dates=["date"])
    holidays = pd.read_sql("SELECT * FROM holidays", conn, parse_dates=["date"])

    df = df.merge(weather, on=["date", "city"], how="left")
    df["rainfall_mm"] = df["rainfall_mm"].fillna(df.groupby("city")["rainfall_mm"].transform("median"))
    df["temp_celsius"] = df["temp_celsius"].fillna(df.groupby("city")["temp_celsius"].transform("mean"))

    # ---- Calendar features ----
    df["dow"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_holiday"] = df["date"].isin(holidays["date"]).astype(int)

    diwali_dates = holidays.loc[holidays["holiday_name"] == "Diwali", "date"]
    festive_dates = set()
    for d in diwali_dates:
        festive_dates.update(pd.date_range(d - pd.Timedelta(days=14), d))
    df["is_festive_window"] = df["date"].isin(festive_dates).astype(int)

    # cyclical encodings for day-of-year seasonality (better than raw int for trees too, but essential for linear/ARIMA-style baselines)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # ---- Lag & rolling features (per store-category series) ----
    df = df.sort_values(["store_id", "category", "date"])
    grp = df.groupby(["store_id", "category"])["revenue"]

    for lag in [1, 7, 14, 28]:
        df[f"revenue_lag_{lag}"] = grp.shift(lag)

    df["revenue_roll_mean_7"] = grp.transform(lambda s: s.shift(1).rolling(7).mean())
    df["revenue_roll_std_7"] = grp.transform(lambda s: s.shift(1).rolling(7).std())
    df["revenue_roll_mean_28"] = grp.transform(lambda s: s.shift(1).rolling(28).mean())
    df["units_roll_mean_7"] = df.groupby(["store_id", "category"])["units"].transform(
        lambda s: s.shift(1).rolling(7).mean()
    )

    # ---- Weather-sensitivity interaction feature ----
    weather_sensitive_cats = {"Beverages"}
    df["is_weather_sensitive_cat"] = df["category"].isin(weather_sensitive_cats).astype(int)
    df["hot_day_x_sensitive"] = ((df["temp_celsius"] > 32).astype(int)) * df["is_weather_sensitive_cat"]

    # ---- Promo feature (had any discount that day) ----
    df["had_promo"] = (df["avg_discount_pct"].fillna(0) > 0).astype(int)

    # Drop rows without enough lag history (first 28 days per series)
    model_df = df.dropna(subset=["revenue_lag_28", "revenue_roll_mean_28"]).copy()

    out_path = ROOT / "data" / "feature_matrix.parquet"
    model_df.to_parquet(out_path, index=False)
    print(f"Model table: {model_df.shape[0]:,} rows x {model_df.shape[1]} cols -> {out_path}")

    # ---- Customer RFM table (separate output, used by dashboard) ----
    rfm = pd.read_sql("""
        SELECT r.*, c.segment, c.city
        FROM vw_customer_rfm r
        JOIN customers c ON c.customer_id = r.customer_id
    """, conn)
    rfm.to_csv(ROOT / "data" / "customer_rfm_metrics.csv", index=False)
    print(f"Customer RFM table: {rfm.shape[0]:,} rows")

    conn.close()

if __name__ == "__main__":
    main()
