"""
Statistical analysis for the Retail Intelligence Platform.

Answers the business question:
  "Did discount campaign A significantly increase sales?"

Also demonstrates: correlation analysis, ANOVA across stores/segments,
and confidence intervals on retention/revenue metrics.
"""
import sqlite3
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "retail_analytics.db"

def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def ab_test_campaign(conn):
    section("A/B TEST: Did 'Campaign A - Flat 20%' significantly increase sales?")

    # Treatment: category-days where Campaign A was active
    # Control:   same categories, non-promo days (baseline demand)
    promos = pd.read_sql("SELECT * FROM promotions WHERE promo_name LIKE 'Campaign A%'", conn,
                          parse_dates=["start_date", "end_date"])
    if promos.empty:
        print("No Campaign A instances found in this data draw.")
        return

    daily_cat = pd.read_sql("""
        SELECT t.date, p.category, SUM(t.units_sold) AS units, SUM(t.revenue) AS revenue
        FROM transactions t JOIN products p ON p.product_id = t.product_id
        GROUP BY t.date, p.category
    """, conn, parse_dates=["date"])

    treatment_mask = pd.Series(False, index=daily_cat.index)
    for _, p in promos.iterrows():
        treatment_mask |= (
            (daily_cat["category"] == p["category"])
            & (daily_cat["date"] >= p["start_date"])
            & (daily_cat["date"] <= p["end_date"])
        )

    treated_categories = promos["category"].unique()
    same_cat_mask = daily_cat["category"].isin(treated_categories)

    treatment = daily_cat.loc[treatment_mask, "units"]
    control = daily_cat.loc[same_cat_mask & ~treatment_mask, "units"]

    print(f"Treatment days (promo active): n={len(treatment)}, mean units/day={treatment.mean():.1f}")
    print(f"Control days (no promo, same categories): n={len(control)}, mean units/day={control.mean():.1f}")

    # Welch's t-test (unequal variance assumption -- more robust default)
    t_stat, p_val = stats.ttest_ind(treatment, control, equal_var=False)
    lift_pct = 100 * (treatment.mean() - control.mean()) / control.mean()

    print(f"\nWelch's t-test: t={t_stat:.3f}, p-value={p_val:.5f}")
    print(f"Observed lift: {lift_pct:+.1f}% in daily units sold")
    alpha = 0.05
    if p_val < alpha:
        print(f"=> Statistically significant at alpha={alpha}. Reject H0: campaign A had a real effect on demand.")
    else:
        print(f"=> NOT statistically significant at alpha={alpha}. Cannot reject H0.")

    # 95% CI on the difference in means
    diff = treatment.mean() - control.mean()
    se = np.sqrt(treatment.var(ddof=1) / len(treatment) + control.var(ddof=1) / len(control))
    ci_low, ci_high = diff - 1.96 * se, diff + 1.96 * se
    print(f"95% CI on mean difference (units/day): [{ci_low:.2f}, {ci_high:.2f}]")


def correlation_analysis(conn):
    section("CORRELATION: weather, discount, and revenue")

    df = pd.read_sql("""
        SELECT t.date, s.city, SUM(t.revenue) AS revenue, AVG(t.discount_pct) AS avg_discount,
               AVG(w.temp_celsius) AS temp, AVG(w.rainfall_mm) AS rainfall
        FROM transactions t
        JOIN stores s ON s.store_id = t.store_id
        LEFT JOIN weather w ON w.date = t.date AND w.city = s.city
        GROUP BY t.date, s.city
    """, conn, parse_dates=["date"])
    df = df.dropna()

    corr_cols = ["revenue", "avg_discount", "temp", "rainfall"]
    corr = df[corr_cols].corr(method="pearson")
    print(corr.round(3))

    r, p = stats.pearsonr(df["avg_discount"], df["revenue"])
    print(f"\nrevenue vs avg_discount: r={r:.3f}, p={p:.5f}")
    r2, p2 = stats.pearsonr(df["temp"], df["revenue"])
    print(f"revenue vs temperature:  r={r2:.3f}, p={p2:.5f}")


def anova_store_revenue(conn):
    section("ANOVA: Does average daily revenue differ significantly across cities?")

    df = pd.read_sql("""
        SELECT t.date, t.store_id, s.city, SUM(t.revenue) AS revenue
        FROM transactions t JOIN stores s ON s.store_id = t.store_id
        GROUP BY t.date, t.store_id, s.city
    """, conn)

    groups = [g["revenue"].values for _, g in df.groupby("city")]
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"One-way ANOVA across {df['city'].nunique()} cities: F={f_stat:.3f}, p-value={p_val:.5g}")
    if p_val < 0.05:
        print("=> Statistically significant differences in average daily revenue exist between cities.")
    else:
        print("=> No statistically significant difference detected between cities.")

    print("\nCity-level means:")
    print(df.groupby("city")["revenue"].agg(["mean", "std", "count"]).round(2))


def confidence_intervals(conn):
    section("CONFIDENCE INTERVALS: repeat purchase rate & avg basket value")

    rfm = pd.read_sql("SELECT * FROM vw_customer_rfm", conn)
    repeat = (rfm["frequency"] > 1).astype(int)
    p_hat = repeat.mean()
    n = len(repeat)
    se = np.sqrt(p_hat * (1 - p_hat) / n)
    ci = (p_hat - 1.96 * se, p_hat + 1.96 * se)
    print(f"Repeat purchase rate: {p_hat*100:.2f}% (n={n})")
    print(f"95% CI: [{ci[0]*100:.2f}%, {ci[1]*100:.2f}%]")

    basket = pd.read_sql("SELECT revenue FROM transactions", conn)["revenue"]
    mean_basket = basket.mean()
    se_basket = basket.std(ddof=1) / np.sqrt(len(basket))
    ci_basket = (mean_basket - 1.96 * se_basket, mean_basket + 1.96 * se_basket)
    print(f"\nAvg transaction line value: {mean_basket:.2f}")
    print(f"95% CI: [{ci_basket[0]:.2f}, {ci_basket[1]:.2f}]")


if __name__ == "__main__":
    conn = sqlite3.connect(DB)
    ab_test_campaign(conn)
    correlation_analysis(conn)
    anova_store_revenue(conn)
    confidence_intervals(conn)
    conn.close()
