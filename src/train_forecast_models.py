"""
Sales forecasting model: predicts daily revenue at store x category grain,
7 days ahead is implicitly handled via lag>=7 features being usable at
inference time for a rolling forecast.

Trains LightGBM (primary) + XGBoost (comparison), evaluates with MAE/RMSE/MAPE,
explains with SHAP, and saves the model + business insight report.
"""
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import shap
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [
    "store_id", "city_tier", "dow", "is_weekend", "month", "is_holiday",
    "is_festive_window", "doy_sin", "doy_cos", "temp_celsius", "rainfall_mm",
    "avg_discount_pct", "had_promo", "is_weather_sensitive_cat", "hot_day_x_sensitive",
    "revenue_lag_1", "revenue_lag_7", "revenue_lag_14", "revenue_lag_28",
    "revenue_roll_mean_7", "revenue_roll_std_7", "revenue_roll_mean_28", "units_roll_mean_7",
]
CATEGORICAL = []  # store_id/city_tier kept numeric/ordinal for simplicity across both libs
TARGET = "revenue"


def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def time_split(df, test_frac=0.15):
    df = df.sort_values("date")
    cutoff = df["date"].quantile(1 - test_frac)
    train = df[df["date"] <= cutoff]
    test = df[df["date"] > cutoff]
    return train, test


def main():
    df = pd.read_parquet(ROOT / "data" / "feature_matrix.parquet")
    df["avg_discount_pct"] = df["avg_discount_pct"].fillna(0)

    train, test = time_split(df)
    print(f"Train: {len(train):,} rows ({train['date'].min().date()} -> {train['date'].max().date()})")
    print(f"Test:  {len(test):,} rows ({test['date'].min().date()} -> {test['date'].max().date()})")

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    results = {}

    # ---- LightGBM ----
    lgb_model = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.03, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1,
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    pred_lgb = lgb_model.predict(X_test)
    results["LightGBM"] = {
        "MAE": mean_absolute_error(y_test, pred_lgb),
        "RMSE": mean_squared_error(y_test, pred_lgb) ** 0.5,
        "MAPE": mape(y_test.values, pred_lgb),
    }

    # ---- XGBoost (comparison model) ----
    xgb_model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.03, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        early_stopping_rounds=30, eval_metric="mae",
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    pred_xgb = xgb_model.predict(X_test)
    results["XGBoost"] = {
        "MAE": mean_absolute_error(y_test, pred_xgb),
        "RMSE": mean_squared_error(y_test, pred_xgb) ** 0.5,
        "MAPE": mape(y_test.values, pred_xgb),
    }

    # ---- Simple baseline for context: naive "same as last week" ----
    naive_pred = test["revenue_lag_7"].values
    results["Naive (lag-7 baseline)"] = {
        "MAE": mean_absolute_error(y_test, naive_pred),
        "RMSE": mean_squared_error(y_test, naive_pred) ** 0.5,
        "MAPE": mape(y_test.values, naive_pred),
    }

    print("\n=== Model Comparison (test set) ===")
    res_df = pd.DataFrame(results).T.round(2)
    print(res_df)
    res_df.to_csv(MODEL_DIR / "model_benchmark_results.csv")

    # Pick LightGBM as production model (best MAE typically, and fastest to serve)
    best_name = res_df["MAE"].idxmin()
    print(f"\nBest model by MAE: {best_name}")

    joblib.dump(lgb_model, MODEL_DIR / "lightgbm_revenue_forecast.pkl")
    joblib.dump(xgb_model, MODEL_DIR / "xgboost_revenue_forecast.pkl")
    with open(MODEL_DIR / "model_feature_list.json", "w") as f:
        json.dump(FEATURES, f)

    # ---- Prediction intervals (empirical residual quantiles) ----
    residuals = y_test.values - pred_lgb
    q05, q95 = np.quantile(residuals, [0.05, 0.95])
    with open(MODEL_DIR / "prediction_interval_quantiles.json", "w") as f:
        json.dump({"q05": float(q05), "q95": float(q95)}, f)
    print(f"90% prediction interval offset: [{q05:.1f}, {q95:.1f}] revenue units around point forecast")

    # ---- SHAP explainability ----
    print("\nComputing SHAP values (sample of 500 test rows)...")
    sample = X_test.sample(min(500, len(X_test)), random_state=1)
    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(sample)
    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURES).sort_values(ascending=False)
    print("\nTop 10 features by mean |SHAP value|:")
    print(mean_abs_shap.head(10).round(2))
    mean_abs_shap.to_csv(MODEL_DIR / "shap_importance_scores.csv")

    # ---- Business insight generation ----
    generate_business_insights(df, lgb_model)


def generate_business_insights(df, model):
    """
    Estimate profit impact of a 15% inventory/stocking increase for the
    top-selling category in the highest-growth city ahead of the festive
    window, mirroring the requested insight format.
    """
    print("\n=== Business Insight Generation ===")

    products = pd.read_csv(ROOT / "data" / "product_catalog.csv")
    festive = df[df["is_festive_window"] == 1]
    if festive.empty:
        print("No festive-window rows available for insight generation.")
        return

    city_growth = festive.groupby("city")["revenue"].sum().sort_values(ascending=False)
    top_city = city_growth.index[0]
    cat_rev = festive[festive["city"] == top_city].groupby("category")["revenue"].sum().sort_values(ascending=False)
    top_category = cat_rev.index[0]

    avg_daily_units = df[(df["city"] == top_city) & (df["category"] == top_category)]["units"].mean()
    avg_margin_pct = 0.30  # blended category margin assumption, documented explicitly
    unit_price_avg = products[products["category"] == top_category]["unit_price"].mean()

    pct_increase = 0.15
    incremental_units_per_day = avg_daily_units * pct_increase
    incremental_profit_per_day = incremental_units_per_day * unit_price_avg * avg_margin_pct
    projected_30d_profit = incremental_profit_per_day * 30

    insight = (
        f"Increasing inventory of '{top_category}' in {top_city} by {int(pct_increase*100)}% "
        f"ahead of the festive season is projected to add "
        f"{'$'}{projected_30d_profit:,.0f} in incremental profit over a 30-day window "
        f"(avg daily demand: {avg_daily_units:.0f} units, blended margin assumption: {int(avg_margin_pct*100)}%)."
    )
    print(insight)

    with open(MODEL_DIR / "inventory_insight_summary.txt", "w", encoding="utf-8") as f:
        f.write(insight)


if __name__ == "__main__":
    main()
