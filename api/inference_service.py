"""
FastAPI service exposing the trained revenue forecasting model.

Run locally:
    uvicorn api.inference_service:app --reload --port 8000

"""
import json
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"

app = FastAPI(
    title="Retail Intelligence Forecasting API",
    description="Serves store x category daily revenue forecasts.",
    version="1.0.0",
)

_model = None
_features = None
_residual_q = None


def get_model():
    global _model, _features, _residual_q
    if _model is None:
        _model = joblib.load(MODEL_DIR / "lightgbm_revenue_forecast.pkl")
        _features = json.loads((MODEL_DIR / "model_feature_list.json").read_text())
        _residual_q = json.loads((MODEL_DIR / "prediction_interval_quantiles.json").read_text())
    return _model, _features, _residual_q


class ForecastRequest(BaseModel):
    store_id: int
    city_tier: int = Field(ge=1, le=2)
    dow: int = Field(ge=0, le=6, description="0=Monday ... 6=Sunday")
    is_weekend: int = Field(ge=0, le=1)
    month: int = Field(ge=1, le=12)
    is_holiday: int = Field(ge=0, le=1)
    is_festive_window: int = Field(ge=0, le=1)
    doy_sin: float
    doy_cos: float
    temp_celsius: float
    rainfall_mm: float
    avg_discount_pct: float = 0.0
    had_promo: int = Field(ge=0, le=1)
    is_weather_sensitive_cat: int = Field(ge=0, le=1)
    hot_day_x_sensitive: int = Field(ge=0, le=1)
    revenue_lag_1: float
    revenue_lag_7: float
    revenue_lag_14: float
    revenue_lag_28: float
    revenue_roll_mean_7: float
    revenue_roll_std_7: Optional[float] = 0.0
    revenue_roll_mean_28: float
    units_roll_mean_7: float


class ForecastResponse(BaseModel):
    predicted_revenue: float
    lower_90: float
    upper_90: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=ForecastResponse)
def predict(req: ForecastRequest):
    model, features, residual_q = get_model()
    row = pd.DataFrame([req.dict()])[features]
    try:
        pred = float(model.predict(row)[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return ForecastResponse(
        predicted_revenue=round(pred, 2),
        lower_90=round(pred + residual_q["q05"], 2),
        upper_90=round(pred + residual_q["q95"], 2),
    )


@app.get("/insights/latest")
def latest_insight():
    path = MODEL_DIR / "inventory_insight_summary.txt"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No insight generated yet. Run train_forecast_models.py first.")
    return {"insight": path.read_text(encoding="utf-8")}


@app.get("/model/comparison")
def model_comparison():
    path = MODEL_DIR / "model_benchmark_results.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No model comparison found.")
    df = pd.read_csv(path, index_col=0)
    return df.to_dict(orient="index")
