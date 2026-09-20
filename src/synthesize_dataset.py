"""
Retail Intelligence & Demand Forecasting Platform
--------------------------------------------------
Synthetic data generator.

"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

RNG = np.random.default_rng(42)
OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)

START = date(2024, 1, 1)
END = date(2025, 12, 31)
ALL_DATES = pd.date_range(START, END, freq="D")

# ----------------------------------------------------------------------
# 1. STORES
# ----------------------------------------------------------------------
cities = {
    "Sydney":    {"tier": 1, "lat": -33.87, "lon": 151.21, "base_footfall": 1.25},
    "Melbourne": {"tier": 1, "lat": -37.81, "lon": 144.96, "base_footfall": 1.30},
    "Brisbane":  {"tier": 1, "lat": -27.47, "lon": 153.03, "base_footfall": 1.15},
    "Adelaide":  {"tier": 2, "lat": -34.93, "lon": 138.60, "base_footfall": 0.85},
}

stores = []
store_id = 1
for city, meta in cities.items():
    n_stores = 2 if meta["tier"] == 1 else 2
    for i in range(n_stores):
        stores.append({
            "store_id": store_id,
            "store_name": f"{city} Store {i+1}",
            "city": city,
            "city_tier": meta["tier"],
            "latitude": meta["lat"] + RNG.uniform(-0.05, 0.05),
            "longitude": meta["lon"] + RNG.uniform(-0.05, 0.05),
            "footfall_index": round(meta["base_footfall"] * RNG.uniform(0.9, 1.1), 3),
            "store_size_sqft": int(RNG.integers(4000, 15000)),
            "opened_date": pd.Timestamp(START) - pd.Timedelta(days=int(RNG.integers(200, 2000))),
        })
        store_id += 1
stores_df = pd.DataFrame(stores)

# ----------------------------------------------------------------------
# 2. PRODUCTS
# ----------------------------------------------------------------------
categories = {
    "Beverages":    {"n": 8,  "price_range": (30, 150),  "weather_sensitive": True},
    "Snacks":       {"n": 8,  "price_range": (20, 120),  "weather_sensitive": False},
    "Dairy":        {"n": 6,  "price_range": (25, 200),  "weather_sensitive": False},
    "Electronics":  {"n": 8,  "price_range": (500, 15000), "weather_sensitive": False},
    "Apparel":      {"n": 10, "price_range": (300, 3000), "weather_sensitive": False},
}

products = []
product_id = 1
for cat, meta in categories.items():
    for i in range(meta["n"]):
        cost = round(RNG.uniform(*meta["price_range"]), 2)
        margin = RNG.uniform(0.2, 0.45)
        products.append({
            "product_id": product_id,
            "product_name": f"{cat} Item {i+1}",
            "category": cat,
            "unit_cost": round(cost * (1 - margin), 2),
            "unit_price": cost,
            "weather_sensitive": meta["weather_sensitive"],
            "launch_date": pd.Timestamp(START) - pd.Timedelta(days=int(RNG.integers(0, 1500))),
        })
        product_id += 1
products_df = pd.DataFrame(products)

# ----------------------------------------------------------------------
# 3. CUSTOMERS
# ----------------------------------------------------------------------
N_CUSTOMERS = 3000
segments = RNG.choice(
    ["Budget", "Regular", "Premium"], size=N_CUSTOMERS, p=[0.4, 0.45, 0.15]
)
signup_dates = pd.Timestamp(START) - pd.to_timedelta(
    RNG.integers(0, 900, size=N_CUSTOMERS), unit="D"
)
customers_df = pd.DataFrame({
    "customer_id": np.arange(1, N_CUSTOMERS + 1),
    "segment": segments,
    "city": RNG.choice(list(cities.keys()), size=N_CUSTOMERS),
    "signup_date": signup_dates,
    "age": RNG.integers(18, 70, size=N_CUSTOMERS),
    "gender": RNG.choice(["M", "F", "Other"], size=N_CUSTOMERS, p=[0.48, 0.48, 0.04]),
})

# ----------------------------------------------------------------------
# 4. HOLIDAYS (India-relevant + a couple of global retail events)
# ----------------------------------------------------------------------
holidays = [
    ("2024-01-26", "Republic Day"), ("2024-03-25", "Holi"),
    ("2024-08-15", "Independence Day"), ("2024-10-02", "Gandhi Jayanti"),
    ("2024-10-31", "Diwali"), ("2024-12-25", "Christmas"),
    ("2025-01-26", "Republic Day"), ("2025-03-14", "Holi"),
    ("2025-08-15", "Independence Day"), ("2025-10-02", "Gandhi Jayanti"),
    ("2025-10-20", "Diwali"), ("2025-12-25", "Christmas"),
]
holidays_df = pd.DataFrame(holidays, columns=["date", "holiday_name"])
holidays_df["date"] = pd.to_datetime(holidays_df["date"])

# Diwali "festive season" window (elevated demand for ~2 weeks before)
festive_windows = []
for d, name in holidays:
    if name == "Diwali":
        d0 = pd.Timestamp(d)
        for offset in range(-14, 1):
            festive_windows.append(d0 + pd.Timedelta(days=offset))
FESTIVE_DATES = set(festive_windows)
HOLIDAY_DATES = set(holidays_df["date"])

# ----------------------------------------------------------------------
# 5. WEATHER (daily, per city) — simple seasonal sinusoid + noise
# ----------------------------------------------------------------------
weather_rows = []
for city in cities:
    for d in ALL_DATES:
        day_of_year = d.dayofyear
        # India: hot Apr-Jun, monsoon Jul-Sep, mild Oct-Feb
        seasonal_temp = 27 + 10 * np.sin((day_of_year - 60) / 365 * 2 * np.pi)
        temp = seasonal_temp + RNG.normal(0, 2.5)
        is_monsoon = 152 <= day_of_year <= 273  # roughly Jun-Sep
        rain_prob = 0.55 if is_monsoon else 0.08
        rained = RNG.random() < rain_prob
        rainfall_mm = round(RNG.exponential(15), 1) if rained else 0.0
        weather_rows.append({
            "date": d, "city": city,
            "temp_celsius": round(temp, 1),
            "rainfall_mm": rainfall_mm,
            "is_rainy": rained,
        })
weather_df = pd.DataFrame(weather_rows)

# ----------------------------------------------------------------------
# 6. PROMOTIONS
# ----------------------------------------------------------------------
promo_rows = []
promo_id = 1
promo_names = ["Campaign A - Flat 20%", "Campaign B - Buy1Get1", "Campaign C - Flat 10%",
               "Diwali Mega Sale", "End of Season Sale", "Weekend Flash Sale"]
for _ in range(60):
    cat = RNG.choice(list(categories.keys()))
    start = pd.Timestamp(START) + pd.Timedelta(days=int(RNG.integers(0, (END - START).days - 14)))
    dur = int(RNG.integers(3, 14))
    promo_rows.append({
        "promo_id": promo_id,
        "promo_name": RNG.choice(promo_names),
        "category": cat,
        "start_date": start,
        "end_date": start + pd.Timedelta(days=dur),
        "discount_pct": int(RNG.choice([10, 15, 20, 25, 30])),
    })
    promo_id += 1
promotions_df = pd.DataFrame(promo_rows)

def active_promo_discount(cat, d):
    """Return max discount pct active for a category on a date (0 if none)."""
    mask = (
        (promotions_df["category"] == cat)
        & (promotions_df["start_date"] <= d)
        & (promotions_df["end_date"] >= d)
    )
    if mask.any():
        return promotions_df.loc[mask, "discount_pct"].max()
    return 0

# Precompute promo discount per (category, date) for speed
cat_dates = pd.MultiIndex.from_product([categories.keys(), ALL_DATES], names=["category", "date"])
promo_lookup = pd.DataFrame(index=cat_dates).reset_index()
promo_lookup["discount_pct"] = 0
for _, p in promotions_df.iterrows():
    mask = (
        (promo_lookup["category"] == p["category"])
        & (promo_lookup["date"] >= p["start_date"])
        & (promo_lookup["date"] <= p["end_date"])
    )
    promo_lookup.loc[mask, "discount_pct"] = np.maximum(
        promo_lookup.loc[mask, "discount_pct"], p["discount_pct"]
    )
promo_lookup = promo_lookup.set_index(["category", "date"])["discount_pct"]

print("Static tables generated. Building daily transactions (this is the heavy part)...")

# ----------------------------------------------------------------------
# 7. SALES TRANSACTIONS (store-product-day aggregated "basket" model)
#    We simulate at the store x category x day level for demand, then
#    explode into product-level rows and customer-level transactions.
# ----------------------------------------------------------------------
weather_lookup = weather_df.set_index(["city", "date"])[["temp_celsius", "rainfall_mm", "is_rainy"]]

txn_rows = []
inv_rows = []
txn_id = 1

store_city_map = stores_df.set_index("store_id")["city"].to_dict()
store_footfall_map = stores_df.set_index("store_id")["footfall_index"].to_dict()

for _, store in stores_df.iterrows():
    sid, city, footfall = store["store_id"], store["city"], store["footfall_index"]
    for cat, meta in categories.items():
        cat_products = products_df[products_df["category"] == cat]
        base_daily_units = RNG.uniform(15, 40) * footfall  # baseline demand for the whole category at this store

        for d in ALL_DATES:
            dow = d.dayofweek
            weekend_lift = 1.25 if dow >= 5 else 1.0
            holiday_lift = 1.6 if d in HOLIDAY_DATES else 1.0
            festive_lift = 1.8 if d in FESTIVE_DATES else 1.0

            w = weather_lookup.loc[(city, d)]
            weather_lift = 1.0
            if meta["weather_sensitive"]:
                if w["temp_celsius"] > 32:
                    weather_lift *= 1.35
                if w["is_rainy"]:
                    weather_lift *= 0.85

            discount = promo_lookup.loc[(cat, d)]
            promo_lift = 1 + (discount / 100) * 1.4 if discount > 0 else 1.0

            # yearly upward trend (store network growth / brand growth)
            days_elapsed = (d - pd.Timestamp(START)).days
            trend_lift = 1 + 0.00025 * days_elapsed

            noise = RNG.normal(1.0, 0.12)
            demand_units = max(
                0,
                base_daily_units * weekend_lift * holiday_lift * festive_lift
                * weather_lift * promo_lift * trend_lift * noise
            )
            demand_units = int(round(demand_units))
            if demand_units == 0:
                continue

            # distribute across products in the category (popularity-weighted)
            weights = RNG.dirichlet(np.ones(len(cat_products)) * 2)
            units_per_product = RNG.multinomial(demand_units, weights)

            for (_, prod), units in zip(cat_products.iterrows(), units_per_product):
                if units == 0:
                    continue
                price = prod["unit_price"] * (1 - discount / 100)
                txn_rows.append({
                    "transaction_id": txn_id,
                    "date": d,
                    "store_id": sid,
                    "product_id": prod["product_id"],
                    "customer_id": int(RNG.integers(1, N_CUSTOMERS + 1)),
                    "units_sold": int(units),
                    "unit_price_effective": round(price, 2),
                    "discount_pct": int(discount),
                    "revenue": round(price * units, 2),
                })
                txn_id += 1

            # weekly inventory snapshot (Sundays) per store-product
            if dow == 6:
                on_hand = max(0, int(demand_units * RNG.uniform(1.5, 3.0) / max(len(cat_products),1)))
                for _, prod in cat_products.iterrows():
                    inv_rows.append({
                        "date": d, "store_id": sid, "product_id": prod["product_id"],
                        "units_on_hand": on_hand + int(RNG.integers(-5, 15)),
                        "reorder_point": int(RNG.integers(10, 30)),
                    })

print(f"Generated {len(txn_rows):,} transaction line items.")

transactions_df = pd.DataFrame(txn_rows)
inventory_df = pd.DataFrame(inv_rows)

# ----------------------------------------------------------------------
# 8. Inject data-quality issues on purpose (for the EDA section)
# ----------------------------------------------------------------------
# Missing values in customer age / weather rainfall
miss_idx = customers_df.sample(frac=0.03, random_state=1).index
customers_df.loc[miss_idx, "age"] = np.nan

miss_idx2 = weather_df.sample(frac=0.02, random_state=2).index
weather_df.loc[miss_idx2, "rainfall_mm"] = np.nan

# Outliers: a handful of freak bulk-purchase transactions
outlier_idx = transactions_df.sample(n=min(40, len(transactions_df)), random_state=3).index
transactions_df.loc[outlier_idx, "units_sold"] = transactions_df.loc[outlier_idx, "units_sold"] * RNG.integers(8, 20, size=len(outlier_idx))
transactions_df.loc[outlier_idx, "revenue"] = (
    transactions_df.loc[outlier_idx, "units_sold"] * transactions_df.loc[outlier_idx, "unit_price_effective"]
).round(2)

# ----------------------------------------------------------------------
# SAVE
# ----------------------------------------------------------------------
stores_df.to_csv(OUT / "store_locations.csv", index=False)
products_df.to_csv(OUT / "product_catalog.csv", index=False)
customers_df.to_csv(OUT / "customer_profiles.csv", index=False)
holidays_df.to_csv(OUT / "holiday_calendar.csv", index=False)
weather_df.to_csv(OUT / "weather_observations.csv", index=False)
promotions_df.to_csv(OUT / "promotion_campaigns.csv", index=False)
transactions_df.to_csv(OUT / "sales_transactions.csv", index=False)
inventory_df.to_csv(OUT / "inventory_snapshots.csv", index=False)

print("\nSaved CSVs to", OUT)
for f in OUT.glob("*.csv"):
    n = sum(1 for _ in open(f)) - 1
    print(f"  {f.name:<20} {n:>10,} rows")
