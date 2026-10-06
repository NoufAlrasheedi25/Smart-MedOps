# -*- coding: utf-8 -*-
"""
04b — طوارئ:
(1) انحدار وقت الانتظار: قصّ قيم شاذة (حد تدريب p99.5)، هدف log1p، تقييم على المقياس الأصلي.
(2) تنبّؤ وصولات الساعة التالية: HGB + PoissonRegressor + مقارنة مع baselines (lag1، lag168، متوسط الهدف في التدريب، ملف تعريف dow×hour من التدريب).
"""
import json
import subprocess

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from _paths import ER_CLEAN_CSV, ER_HOURLY_CSV, MODELS_DIR, OUTPUT_DIR


def log(msg: str) -> None:
    print(f"[04b_train_er_congestion] {msg}", flush=True)


def detect_nvidia_gpu() -> str:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return "not detected"


def build_tabular_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                cat,
            ),
        ]
    )


def add_time_pressure_features(h: pd.DataFrame) -> pd.DataFrame:
    """ميزات ضغط زمنية مبنية فقط على الماضي حتى لا يحدث تسريب للمستقبل."""
    h = h.copy()
    h["t"] = pd.to_datetime(h["hour_start"])
    h = h.sort_values("t").reset_index(drop=True)

    h["hour_of_day"] = h["t"].dt.hour
    h["dow"] = h["t"].dt.dayofweek
    h["month"] = h["t"].dt.month
    h["weekofyear"] = h["t"].dt.isocalendar().week.astype(int)
    h["dayofyear"] = h["t"].dt.dayofyear
    h["is_weekend"] = h["dow"].isin([4, 5]).astype(int)  # الجمعة/السبت محليًا كافتراض للعرض
    h["is_night"] = h["hour_of_day"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    h["is_business_hours"] = h["hour_of_day"].between(8, 16).astype(int)

    h["hour_sin"] = np.sin(2 * np.pi * h["hour_of_day"] / 24)
    h["hour_cos"] = np.cos(2 * np.pi * h["hour_of_day"] / 24)
    h["dow_sin"] = np.sin(2 * np.pi * h["dow"] / 7)
    h["dow_cos"] = np.cos(2 * np.pi * h["dow"] / 7)
    h["month_sin"] = np.sin(2 * np.pi * h["month"] / 12)
    h["month_cos"] = np.cos(2 * np.pi * h["month"] / 12)
    h["dayofyear_sin"] = np.sin(2 * np.pi * h["dayofyear"] / 365.25)
    h["dayofyear_cos"] = np.cos(2 * np.pi * h["dayofyear"] / 365.25)

    for lag in (1, 2, 3, 6, 12, 24, 48, 168):
        h[f"arrivals_lag_{lag}"] = h["arrivals"].shift(lag)
        h[f"mean_wait_lag_{lag}"] = h["mean_waittime"].shift(lag)

    shifted_arrivals = h["arrivals"].shift(1)
    shifted_wait = h["mean_waittime"].shift(1)
    for window in (3, 6, 12, 24, 48, 168):
        h[f"arrivals_roll_mean_{window}"] = shifted_arrivals.rolling(window, min_periods=max(2, window // 3)).mean()
        h[f"arrivals_roll_std_{window}"] = shifted_arrivals.rolling(window, min_periods=max(2, window // 3)).std()
        h[f"arrivals_roll_max_{window}"] = shifted_arrivals.rolling(window, min_periods=max(2, window // 3)).max()
        h[f"wait_roll_mean_{window}"] = shifted_wait.rolling(window, min_periods=max(2, window // 3)).mean()
        h[f"wait_roll_std_{window}"] = shifted_wait.rolling(window, min_periods=max(2, window // 3)).std()

    h["arrivals_trend_3"] = h["arrivals_lag_1"] - h["arrivals_lag_3"]
    h["arrivals_trend_6"] = h["arrivals_lag_1"] - h["arrivals_lag_6"]
    h["arrivals_trend_12"] = h["arrivals_lag_1"] - h["arrivals_lag_12"]
    h["wait_trend_3"] = h["mean_wait_lag_1"] - h["mean_wait_lag_3"]
    h["wait_trend_6"] = h["mean_wait_lag_1"] - h["mean_wait_lag_6"]

    # Proxy تقريبي للمرضى المنتظرين/ضغط التشغيل من البيانات المتاحة فقط.
    h["recent_pressure_3h"] = h["arrivals_roll_mean_3"] * (1 + h["wait_roll_mean_3"].fillna(0) / 60.0)
    h["recent_pressure_6h"] = h["arrivals_roll_mean_6"] * (1 + h["wait_roll_mean_6"].fillna(0) / 60.0)
    h["recent_pressure_12h"] = h["arrivals_roll_mean_12"] * (1 + h["wait_roll_mean_12"].fillna(0) / 60.0)
    h["queue_pressure_proxy"] = shifted_arrivals.rolling(6, min_periods=2).sum() * (
        1 + shifted_wait.rolling(6, min_periods=2).mean().fillna(0) / 60.0
    )

    # Placeholders when real staffing/capacity feeds are unavailable.
    h["staff_doctors_available"] = 6
    h["staff_nurses_available"] = 18
    h["beds_available"] = 24
    h["seasonal_event_flag"] = 0
    h["weather_pressure_index"] = 0.0
    return h


def train_waittime_regression() -> dict:
    log("[TRAINING_DEVICE] ER wait-time HistGradientBoostingRegressor = CPU (scikit-learn)")
    df = pd.read_csv(ER_CLEAN_CSV, low_memory=False)
    ts = pd.to_datetime(df["admission_ts"], errors="coerce")
    df = df.assign(
        admission_hour=ts.dt.hour,
        admission_dow=ts.dt.dayofweek,
        admission_month=ts.dt.month,
    )
    df = df.dropna(subset=["patient_waittime"])
    y_raw = df["patient_waittime"].astype(float)
    X = df.drop(columns=["admission_ts", "patient_waittime"])

    pre_template = build_tabular_preprocess(X)
    X_tr, X_te, y_tr_raw, y_te_raw = train_test_split(X, y_raw, test_size=0.2, random_state=42)
    clip_hi = float(np.quantile(y_tr_raw, 0.995))
    y_tr = y_tr_raw.clip(upper=clip_hi)
    y_te = y_te_raw.clip(upper=clip_hi)
    y_tr_log = np.log1p(y_tr.astype(float))

    pre = clone(pre_template)
    pipe = Pipeline(
        [
            ("pre", pre),
            ("reg", HistGradientBoostingRegressor(max_depth=10, max_iter=250, learning_rate=0.08, random_state=42)),
        ]
    )
    pipe.fit(X_tr, y_tr_log)
    pred_log = pipe.predict(X_te)
    pred = np.expm1(pred_log)
    pred = np.clip(pred, 0.0, None)

    mae = mean_absolute_error(y_te_raw, pred)
    rmse = float(np.sqrt(mean_squared_error(y_te_raw, pred)))
    r2 = r2_score(y_te_raw, pred)

    path = MODELS_DIR / "er_waittime_hgb.joblib"
    joblib.dump({"pipeline": pipe, "clip_upper_train_p995": clip_hi, "target": "log1p(patient_waittime clipped train p99.5)"}, path)
    return {
        "task": "er_waittime_regression",
        "model": "HistGradientBoostingRegressor on log1p(target); clip train p99.5 applied to train target only",
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "clip_upper_train_p995": clip_hi,
        "metrics_test_on_original_scale": {"mae": float(mae), "rmse": rmse, "r2": float(r2)},
        "note_ar": "إن بقي R² ضعيفاً فالبيانات قد لا تكفي لشرح وقت الانتظار (عوامل تشغيل غير موجودة في CSV).",
        "artifact": str(path),
    }


def train_next_hour_arrivals() -> dict:
    log("[TRAINING_DEVICE] ER next-hour arrivals models = CPU (scikit-learn HGB + PoissonRegressor)")
    h = pd.read_csv(ER_HOURLY_CSV, low_memory=False)
    h = add_time_pressure_features(h)

    h["target_next_hour_arrivals"] = h["arrivals"].shift(-1)
    feat_cols = [c for c in h.columns if c not in ("hour_start", "t", "arrivals", "mean_waittime", "target_next_hour_arrivals")]
    h2 = h.dropna(subset=feat_cols + ["target_next_hour_arrivals"])

    n = len(h2)
    split = int(n * 0.85)
    train = h2.iloc[:split].copy()
    test = h2.iloc[split:].copy()

    X_tr = train[feat_cols]
    y_tr = train["target_next_hour_arrivals"].astype(float)
    X_te = test[feat_cols]
    y_te = test["target_next_hour_arrivals"].astype(float)

    mean_train_target = float(y_tr.mean())
    pred_mean = np.full(len(y_te), mean_train_target, dtype=float)

    prof = train.groupby(["dow", "hour_of_day"], as_index=False)["target_next_hour_arrivals"].mean().rename(
        columns={"target_next_hour_arrivals": "prof_pred"}
    )
    test_prof = test.merge(prof, on=["dow", "hour_of_day"], how="left")
    test_prof["prof_pred"] = test_prof["prof_pred"].fillna(mean_train_target)
    pred_profile = test_prof["prof_pred"].to_numpy()

    pred_lag1 = X_te["arrivals_lag_1"].astype(float).to_numpy()
    pred_lag168 = X_te["arrivals_lag_168"].astype(float).to_numpy()

    mae_mean = mean_absolute_error(y_te, pred_mean)
    mae_prof = mean_absolute_error(y_te, pred_profile)
    mae_naive_lag1 = mean_absolute_error(y_te, pred_lag1)
    mae_naive_lag168 = mean_absolute_error(y_te, pred_lag168)

    log(f"ER arrivals features after rolling/pressure engineering: {len(feat_cols)}")

    reg_hgb = HistGradientBoostingRegressor(
        max_depth=8,
        max_iter=450,
        learning_rate=0.04,
        l2_regularization=0.01,
        min_samples_leaf=25,
        random_state=43,
    )
    reg_hgb.fit(X_tr, y_tr)
    pred_hgb = reg_hgb.predict(X_te)
    mae_hgb = mean_absolute_error(y_te, pred_hgb)
    rmse_hgb = float(np.sqrt(mean_squared_error(y_te, pred_hgb)))
    r2_hgb = r2_score(y_te, pred_hgb)

    pois = Pipeline(
        [
            ("sc", StandardScaler()),
            ("reg", PoissonRegressor(alpha=1e-2, max_iter=800, tol=1e-4)),
        ]
    )
    pois.fit(X_tr, y_tr)
    pred_pois = pois.predict(X_te)
    pred_pois = np.clip(pred_pois, 0.0, None)
    mae_pois = mean_absolute_error(y_te, pred_pois)
    rmse_pois = float(np.sqrt(mean_squared_error(y_te, pred_pois)))
    r2_pois = r2_score(y_te, pred_pois)

    path_hgb = MODELS_DIR / "er_next_hour_arrivals_hgb.joblib"
    path_pois = MODELS_DIR / "er_next_hour_arrivals_poisson.joblib"
    joblib.dump({"model": reg_hgb, "feature_columns": feat_cols, "feature_builder": "add_time_pressure_features_v2"}, path_hgb)
    joblib.dump({"model": pois, "feature_columns": feat_cols, "feature_builder": "add_time_pressure_features_v2"}, path_pois)

    best_name = min(
        [
            ("hist_gradient_boosting", mae_hgb),
            ("poisson_regressor", mae_pois),
            ("baseline_train_mean_target", mae_mean),
            ("baseline_dow_hour_profile_train", mae_prof),
            ("baseline_arrivals_lag1", mae_naive_lag1),
            ("baseline_arrivals_lag168_same_hour_last_week", mae_naive_lag168),
        ],
        key=lambda x: x[1],
    )[0]

    return {
        "task": "er_next_hour_arrivals_time_series",
        "split_note": "first 85% hours train, last 15% test (no shuffle)",
        "n_train_hours": int(len(X_tr)),
        "n_test_hours": int(len(X_te)),
        "lag_hours": [1, 2, 3, 6, 12, 24, 48, 168],
        "feature_engineering": {
            "n_features": len(feat_cols),
            "added": [
                "weekend/month/week/day-of-year seasonality",
                "cyclical calendar encodings",
                "rolling mean/std/max for arrivals and wait time",
                "3/6/12h trend features",
                "recent pressure and queue-pressure proxies",
                "staff/beds/weather/event placeholders for future real feeds",
            ],
        },
        "hist_gradient_boosting": {
            "metrics_test": {"mae": float(mae_hgb), "rmse": rmse_hgb, "r2": float(r2_hgb)},
            "artifact": str(path_hgb),
        },
        "poisson_regressor": {
            "metrics_test": {"mae": float(mae_pois), "rmse": rmse_pois, "r2": float(r2_pois)},
            "artifact": str(path_pois),
            "note": "Pipeline(StandardScaler, PoissonRegressor) — مناسب للعدّ غير السالب.",
        },
        "baselines_test_mae": {
            "train_mean_of_target": float(mae_mean),
            "dow_hour_mean_target_from_train": float(mae_prof),
            "arrivals_lag_1": float(mae_naive_lag1),
            "arrivals_lag_168_same_hour_last_week": float(mae_naive_lag168),
        },
        "best_by_mae_on_test": best_name,
    }


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"[TRAINING_DEVICE] NVIDIA GPU = {detect_nvidia_gpu()}")
    log("[TRAINING_DEVICE] This ER script currently uses scikit-learn regressors, so training is CPU.")

    waittime_block = train_waittime_regression()
    ts_block = train_next_hour_arrivals()

    summary = {
        "inputs": {"er_clean_csv": str(ER_CLEAN_CSV), "er_hourly_csv": str(ER_HOURLY_CSV)},
        "waittime_regression": waittime_block,
        "next_hour_arrivals": ts_block,
    }
    out = OUTPUT_DIR / "er_model_report.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Wrote:", out)


if __name__ == "__main__":
    main()
