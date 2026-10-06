# -*- coding: utf-8 -*-
"""HTTP inference API for Smart MedOps trained models.

Run:
    python ml/model_api.py --port 8000

Quick local test:
    python ml/model_api.py --once
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ML_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ML_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
ER_HOURLY_CSV = PROJECT_ROOT / "Dataset" / "processed" / "hospital_er_hourly_series.csv"
ER_CLEAN_CSV = PROJECT_ROOT / "Dataset" / "processed" / "hospital_er_cleaned.csv"
ER_REPORT = ML_DIR / "outputs" / "er_model_report.json"


def _load_er_training_module():
    path = ML_DIR / "04b_train_er_congestion.py"
    spec = importlib.util.spec_from_file_location("medops_er_training", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load training module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def er_training_module():
    return _load_er_training_module()


@lru_cache(maxsize=8)
def load_joblib(name: str) -> Any:
    path = MODELS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)


@lru_cache(maxsize=1)
def load_er_report() -> dict[str, Any]:
    if not ER_REPORT.exists():
        return {}
    return json.loads(ER_REPORT.read_text(encoding="utf-8"))


def model_and_features(artifact_name: str):
    artifact = load_joblib(artifact_name)
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact["model"], artifact.get("feature_columns")
    return artifact, None


def load_hourly_features() -> pd.DataFrame:
    if not ER_HOURLY_CSV.exists():
        raise FileNotFoundError(f"Missing hourly data: {ER_HOURLY_CSV}")
    hourly = pd.read_csv(ER_HOURLY_CSV, low_memory=False)
    return er_training_module().add_time_pressure_features(hourly)


def latest_feature_row(feature_columns: list[str] | None) -> pd.DataFrame:
    features = load_hourly_features()
    if not feature_columns:
        excluded = {"hour_start", "t", "arrivals", "mean_waittime", "target_next_hour_arrivals"}
        feature_columns = [c for c in features.columns if c not in excluded]

    usable = features.dropna(subset=feature_columns).copy()
    if usable.empty:
        raise ValueError("No complete feature row available for ER arrivals prediction.")
    return usable.iloc[[-1]][feature_columns]


def predict_profile_baseline() -> tuple[float, pd.Timestamp]:
    """Same idea as training baseline: mean next-hour arrivals by dow x hour from first 85%."""
    features = load_hourly_features()
    features["target_next_hour_arrivals"] = features["arrivals"].shift(-1)
    base = features.dropna(subset=["target_next_hour_arrivals"]).copy()
    if base.empty:
        raise ValueError("No hourly target rows available for baseline prediction.")

    split = int(len(base) * 0.85)
    train = base.iloc[:split].copy()
    latest = features.iloc[-1]
    mean_target = float(train["target_next_hour_arrivals"].mean())
    profile = (
        train.groupby(["dow", "hour_of_day"], as_index=False)["target_next_hour_arrivals"]
        .mean()
        .rename(columns={"target_next_hour_arrivals": "profile_prediction"})
    )
    matched = profile[
        (profile["dow"] == int(latest["dow"]))
        & (profile["hour_of_day"] == int(latest["hour_of_day"]))
    ]
    pred = float(matched["profile_prediction"].iloc[0]) if not matched.empty else mean_target
    return max(0.0, pred), pd.to_datetime(latest["t"])


def load_band(predicted: float) -> str:
    if predicted >= 2.0:
        return "High"
    if predicted >= 1.25:
        return "Elevated"
    if predicted >= 0.75:
        return "Moderate"
    return "Stable"


class ArrivalsPredictionResponse(BaseModel):
    predictedArrivalsNextHour: float
    loadBand: str
    modelName: str
    sourceHourStartUtc: str
    artifact: str | None = None


def predict_next_hour_arrivals(method: str = "auto") -> ArrivalsPredictionResponse:
    report = load_er_report()
    best = (
        report.get("next_hour_arrivals", {})
        .get("best_by_mae_on_test", "poisson_regressor")
    )
    selected = best if method == "auto" else method

    if selected == "baseline_dow_hour_profile_train":
        pred, source_hour = predict_profile_baseline()
        return ArrivalsPredictionResponse(
            predictedArrivalsNextHour=round(float(pred), 3),
            loadBand=load_band(float(pred)),
            modelName="baseline_dow_hour_profile_train",
            sourceHourStartUtc=source_hour.isoformat(),
            artifact=None,
        )

    artifact_name = {
        "poisson_regressor": "er_next_hour_arrivals_poisson.joblib",
        "hist_gradient_boosting": "er_next_hour_arrivals_hgb.joblib",
    }.get(selected, "er_next_hour_arrivals_poisson.joblib")
    model, feature_columns = model_and_features(artifact_name)
    row = latest_feature_row(feature_columns)
    pred = float(np.asarray(model.predict(row))[0])
    source_hour = pd.to_datetime(load_hourly_features().iloc[-1]["t"])
    pred = max(0.0, pred)
    return ArrivalsPredictionResponse(
        predictedArrivalsNextHour=round(pred, 3),
        loadBand=load_band(pred),
        modelName=selected,
        sourceHourStartUtc=source_hour.isoformat(),
        artifact=str(MODELS_DIR / artifact_name),
    )


app = FastAPI(title="Smart MedOps ML API", version="1.0")


@app.get("/health")
def health():
    return {
        "ok": True,
        "modelsDir": str(MODELS_DIR),
        "hourlyCsvExists": ER_HOURLY_CSV.exists(),
        "erReportExists": ER_REPORT.exists(),
    }


@app.post("/predict/er-arrivals-next-hour", response_model=ArrivalsPredictionResponse)
def predict_arrivals_next_hour(method: str = "auto"):
    try:
        return predict_next_hour_arrivals(method)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"{type(ex).__name__}: {ex}") from ex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Print one prediction and exit.")
    parser.add_argument("--method", default="auto", help="auto, poisson_regressor, hist_gradient_boosting, baseline_dow_hour_profile_train")
    parser.add_argument("--port", type=int, default=int(os.environ.get("MEDOPS_MODEL_API_PORT", "8000")))
    args = parser.parse_args()
    if args.once:
        print(predict_next_hour_arrivals(args.method).model_dump_json(indent=2))
    else:
        import uvicorn

        uvicorn.run(app, host="127.0.0.1", port=args.port, reload=False)


if __name__ == "__main__":
    main()
