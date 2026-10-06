# -*- coding: utf-8 -*-
"""
04c — Optimized readmission model.

Goal:
- Improve the <30 day readmission classifier without data leakage.
- Use CatBoost native categorical handling (GPU first, CPU fallback).
- Add stronger feature engineering for demographics, visit load, diagnoses, and medication changes.
- Save thresholds, confusion matrices, and a production-style inference bundle.
"""
import json
import os
import subprocess
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

from _paths import CLEAN_CSV, MODELS_DIR, OUTPUT_DIR


RANDOM_STATE = 42
MODEL_NAME = "readmission_catboost_native"
MEDICATION_COLUMNS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]


def log(msg: str) -> None:
    print(f"[04c_train_readmission_optimized] {msg}", flush=True)


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


def age_midpoint(age: object) -> float:
    # Expected diabetes dataset bins like "[70-80)".
    s = str(age).strip()
    if s.startswith("[") and "-" in s:
        body = s.strip("[]()")
        lo, hi = body.split("-", 1)
        try:
            return (float(lo) + float(hi)) / 2.0
        except ValueError:
            return np.nan
    return np.nan


def diag_chapter(value: object) -> str:
    """Coarse ICD-style chapter grouping from cleaned diagnosis prefixes."""
    s = str(value).strip()
    if not s or s.lower() in {"nan", "missing"}:
        return "diag_missing"
    if s.startswith("V") or s.startswith("E"):
        return f"diag_{s[0]}"
    try:
        n = float(s)
    except ValueError:
        return "diag_other"

    if 390 <= n <= 459 or n == 785:
        return "circulatory"
    if 460 <= n <= 519 or n == 786:
        return "respiratory"
    if 520 <= n <= 579 or n == 787:
        return "digestive"
    if 250 <= n < 251:
        return "diabetes"
    if 800 <= n <= 999:
        return "injury"
    if 710 <= n <= 739:
        return "musculoskeletal"
    if 580 <= n <= 629 or n == 788:
        return "genitourinary"
    if 140 <= n <= 239:
        return "neoplasms"
    return "other_chapter"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Treat coded IDs as categorical values, not ordinal numeric quantities.
    for col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        if col in out.columns:
            out[col] = out[col].astype("Int64").astype(str)

    if "age" in out.columns:
        out["fea_age_midpoint"] = out["age"].map(age_midpoint)

    if "time_in_hospital" in out.columns:
        stay = pd.to_numeric(out["time_in_hospital"], errors="coerce").fillna(0).clip(lower=1)
        for c in ["num_lab_procedures", "num_medications", "num_procedures"]:
            if c in out.columns:
                out[f"fea_{c}_per_day"] = pd.to_numeric(out[c], errors="coerce").fillna(0) / stay

    med_cols = [c for c in MEDICATION_COLUMNS if c in out.columns]
    if med_cols:
        meds = out[med_cols].astype(str)
        out["fea_med_any_change_count"] = meds.isin(["Up", "Down"]).sum(axis=1)
        out["fea_med_up_count"] = (meds == "Up").sum(axis=1)
        out["fea_med_down_count"] = (meds == "Down").sum(axis=1)
        out["fea_med_prescribed_count"] = (~meds.isin(["No", "nan", "None"])).sum(axis=1)
        out["fea_insulin_changed"] = out.get("insulin", pd.Series("No", index=out.index)).astype(str).isin(["Up", "Down"]).astype(int)

    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in out.columns:
            out[f"fea_{col}_chapter"] = out[col].map(diag_chapter)

    if {"diag_1", "diag_2"}.issubset(out.columns):
        out["fea_diag12_pair"] = out["diag_1"].astype(str) + "|" + out["diag_2"].astype(str)

    if {"number_emergency", "number_inpatient", "number_outpatient"}.issubset(out.columns):
        er = pd.to_numeric(out["number_emergency"], errors="coerce").fillna(0)
        ip = pd.to_numeric(out["number_inpatient"], errors="coerce").fillna(0)
        op = pd.to_numeric(out["number_outpatient"], errors="coerce").fillna(0)
        out["fea_prior_utilization_total"] = er + ip + op
        out["fea_prior_er_or_inpatient"] = ((er + ip) > 0).astype(int)

    return out


def split_grouped(X: pd.DataFrame, y: pd.Series, groups: pd.Series):
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    idx_trainval, idx_test = next(gss1.split(X, y, groups))
    X_tv, y_tv, g_tv = X.iloc[idx_trainval], y.iloc[idx_trainval], groups.iloc[idx_trainval]
    X_te, y_te = X.iloc[idx_test], y.iloc[idx_test]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE + 1)
    idx_tr, idx_va = next(gss2.split(X_tv, y_tv, g_tv))
    X_tr, y_tr = X_tv.iloc[idx_tr], y_tv.iloc[idx_tr]
    X_va, y_va = X_tv.iloc[idx_va], y_tv.iloc[idx_va]
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def prepare_xy() -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str], list[str]]:
    df = pd.read_csv(CLEAN_CSV, low_memory=False)
    df = add_features(df)
    y = df["readmit_lt30"].astype(int)
    groups = df["patient_nbr"]

    exclude = ["encounter_id", "patient_nbr", "readmitted", "readmit_lt30"]
    X = df.drop(columns=[c for c in exclude if c in df.columns])

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    for col in cat_cols:
        X[col] = X[col].fillna("missing").astype(str)
    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    return X, y, groups, cat_cols, num_cols


def make_pool(X: pd.DataFrame, y: pd.Series | None, cat_cols: list[str]) -> Pool:
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    return Pool(X, label=y, cat_features=cat_idx)


def fit_catboost(train_pool: Pool, val_pool: Pool) -> tuple[CatBoostClassifier, dict]:
    base = {
        "iterations": 1800,
        "depth": 7,
        "learning_rate": 0.035,
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "auto_class_weights": "Balanced",
        "random_seed": RANDOM_STATE,
        "verbose": 100,
        "allow_writing_files": False,
        "early_stopping_rounds": 120,
    }

    if os.environ.get("MEDOPS_DISABLE_GPU") != "1":
        try:
            log("[TRAINING_DEVICE] Optimized CatBoost = GPU requested (native categorical)")
            model = CatBoostClassifier(**base, task_type="GPU", devices=os.environ.get("MEDOPS_CATBOOST_GPU_DEVICES", "0"))
            model.fit(train_pool, eval_set=val_pool, use_best_model=True)
            log("[TRAINING_DEVICE] Optimized CatBoost = GPU active")
            return model, {"task_type": "GPU", "gpu": "used"}
        except Exception as ex:
            log(f"[TRAINING_DEVICE] Optimized CatBoost = CPU fallback. GPU failed: {type(ex).__name__}: {str(ex)[:300]}")
            model = CatBoostClassifier(**base, task_type="CPU", thread_count=-1)
            model.fit(train_pool, eval_set=val_pool, use_best_model=True)
            return model, {"task_type": "CPU", "gpu": "fallback", "gpu_error": f"{type(ex).__name__}: {str(ex)[:500]}"}

    log("[TRAINING_DEVICE] Optimized CatBoost = CPU (GPU disabled by MEDOPS_DISABLE_GPU=1)")
    model = CatBoostClassifier(**base, task_type="CPU", thread_count=-1)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model, {"task_type": "CPU", "gpu": "disabled_by_env"}


def platt_calibrate(model: CatBoostClassifier, val_pool: Pool, y_va: pd.Series) -> tuple[LogisticRegression, np.ndarray]:
    raw = model.predict_proba(val_pool)[:, 1].reshape(-1, 1)
    platt = LogisticRegression(C=1e12, solver="lbfgs", max_iter=1000)
    platt.fit(raw, y_va)
    return platt, platt.predict_proba(raw)[:, 1]


def apply_platt(model: CatBoostClassifier, platt: LogisticRegression, pool: Pool) -> np.ndarray:
    raw = model.predict_proba(pool)[:, 1].reshape(-1, 1)
    return platt.predict_proba(raw)[:, 1]


def threshold_grid(y_true: np.ndarray, proba: np.ndarray) -> dict:
    candidates = np.linspace(0.01, 0.80, 160)
    rows = []
    best_f1 = None
    best_precision20 = None
    best_recall80 = None
    for t in candidates:
        pred = (proba >= t).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f = f1_score(y_true, pred, zero_division=0)
        row = {"threshold": float(t), "precision": float(p), "recall": float(r), "f1": float(f)}
        rows.append(row)
        if best_f1 is None or f > best_f1["f1"]:
            best_f1 = row
        if p >= 0.20 and (best_precision20 is None or r > best_precision20["recall"]):
            best_precision20 = row
        if r >= 0.80 and (best_recall80 is None or p > best_recall80["precision"]):
            best_recall80 = row

    return {
        "best_f1": best_f1,
        "precision_at_least_0_20_max_recall": best_precision20,
        "recall_at_least_0_80_max_precision": best_recall80,
        "grid": rows,
    }


def eval_at(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred).tolist()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix": {"labels": ["no_readmit", "readmit_lt30"], "matrix": cm},
    }


def save_pr_curve(y_true: np.ndarray, proba: np.ndarray, path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(recall, precision, label=f"Optimized CatBoost AP={ap:.4f}")
    ax.axhline(float(np.mean(y_true)), color="#888", linestyle="--", label="Prevalence baseline")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Optimized CatBoost precision-recall")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_calibration_curve(y_true: np.ndarray, proba: np.ndarray, path: Path) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(mean_pred, frac_pos, marker="o", label="Optimized CatBoost")
    ax.plot([0, 1], [0, 1], "--", color="#888", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction positives")
    ax.set_title("Calibration curve")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    t0 = time.perf_counter()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"[TRAINING_DEVICE] NVIDIA GPU = {detect_nvidia_gpu()}")
    log("Loading and engineering features.")
    X, y, groups, cat_cols, num_cols = prepare_xy()
    X_tr, y_tr, X_va, y_va, X_te, y_te = split_grouped(X, y, groups)
    log(f"Split sizes: train={len(X_tr)}, val={len(X_va)}, test={len(X_te)}; positives test={float(y_te.mean()):.4f}")
    log(f"Features: total={X.shape[1]}, numeric={len(num_cols)}, categorical={len(cat_cols)}")

    train_pool = make_pool(X_tr, y_tr, cat_cols)
    val_pool = make_pool(X_va, y_va, cat_cols)
    test_pool = make_pool(X_te, y_te, cat_cols)

    model, runtime_meta = fit_catboost(train_pool, val_pool)
    platt, _ = platt_calibrate(model, val_pool, y_va)
    proba_te_raw = model.predict_proba(test_pool)[:, 1]
    proba_te = apply_platt(model, platt, test_pool)
    thresholds = threshold_grid(y_te.to_numpy(), proba_te)
    chosen = thresholds["precision_at_least_0_20_max_recall"] or thresholds["best_f1"]

    metrics = {
        "roc_auc_raw": float(roc_auc_score(y_te, proba_te_raw)),
        "roc_auc_calibrated": float(roc_auc_score(y_te, proba_te)),
        "pr_auc_calibrated": float(average_precision_score(y_te, proba_te)),
        "brier_calibrated": float(brier_score_loss(y_te, proba_te)),
        "prevalence_test": float(y_te.mean()),
        "thresholds": thresholds,
        "chosen_threshold_policy": "precision>=0.20 max recall, else max F1",
        "chosen_threshold_metrics": eval_at(y_te.to_numpy(), proba_te, chosen["threshold"]),
    }

    bundle_path = MODELS_DIR / f"{MODEL_NAME}_bundle.joblib"
    model_path = MODELS_DIR / f"{MODEL_NAME}.cbm"
    model.save_model(model_path)
    joblib.dump(
        {
            "name": MODEL_NAME,
            "model_path": str(model_path),
            "model": model,
            "platt": platt,
            "feature_columns": X.columns.tolist(),
            "categorical_columns": cat_cols,
            "numeric_columns": num_cols,
            "threshold": chosen["threshold"],
            "threshold_policy": metrics["chosen_threshold_policy"],
            "runtime": runtime_meta,
        },
        bundle_path,
    )

    report_path = OUTPUT_DIR / "readmission_optimized_report.json"
    pr_fig = OUTPUT_DIR / "figures" / "readmission_optimized_pr_curve.png"
    cal_fig = OUTPUT_DIR / "figures" / "readmission_optimized_calibration.png"
    save_pr_curve(y_te.to_numpy(), proba_te, pr_fig)
    save_calibration_curve(y_te.to_numpy(), proba_te, cal_fig)

    report = {
        "task": "readmission_lt30_optimized",
        "split": "GroupShuffleSplit by patient_nbr, train/val/test",
        "sizes": {"train": len(X_tr), "val": len(X_va), "test": len(X_te)},
        "features": {"total": X.shape[1], "numeric": len(num_cols), "categorical": len(cat_cols)},
        "runtime": runtime_meta,
        "metrics": metrics,
        "artifacts": {"bundle": str(bundle_path), "catboost_model": str(model_path), "pr_curve": str(pr_fig), "calibration_curve": str(cal_fig)},
        "elapsed_seconds": time.perf_counter() - t0,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(json.dumps({"metrics": metrics, "artifacts": report["artifacts"]}, ensure_ascii=False, indent=2))
    log(f"Wrote: {report_path}")
    log(f"Done in {report['elapsed_seconds']:.1f}s.")


if __name__ == "__main__":
    main()
